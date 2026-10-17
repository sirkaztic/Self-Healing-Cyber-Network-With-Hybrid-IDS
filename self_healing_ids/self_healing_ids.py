# %% [markdown]
# # Self-Healing Hybrid Intrusion Detection System
# ### with Adversarial Robustness and GenAI-Based Attack Explanation  (K083 / K084 / K085)
#
# Pipeline: **Load -> Preprocess -> Train neural IDS -> real FGSM -> adversarial training -> Evaluate
# -> Monitor live traffic -> Explain (LLM or local fallback) -> Log -> Retrain -> Redeploy (repeat)**
#
# Just run all cells. No API key is needed. If you want real LLM explanations, set
# `ANTHROPIC_API_KEY` or `GEMINI_API_KEY` in the config cell.

# %% [markdown]
# ## 0. Config

# %%
import os, time, json, copy, random, urllib.request, urllib.error
from dataclasses import dataclass, field
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ---- settings you can change ----
TRAIN_SAMPLE   = 20000     # rows of KDDTrain+ used (full set is ~125k; raise it if you want)
EPSILON        = 0.25      # FGSM strength (in standard deviations, features are standardised)
CONF_THRESHOLD = 0.85      # below this confidence a sample is "unknown attack"
CYCLES         = 4         # how many times the self-healing loop runs
BATCH_SIZE     = 1500      # live samples per cycle
SEED           = 42

# Optional: real LLM for explanations (leave empty to use the built-in local explainer)
os.environ.setdefault("ANTHROPIC_API_KEY", "")   # paste key here if you have one
os.environ.setdefault("GEMINI_API_KEY", "")      # or a Gemini key here
LLM_MODEL_CLAUDE = "claude-sonnet-4-5"           # change if your account uses another model name
LLM_MODEL_GEMINI = "gemini-2.0-flash"

random.seed(SEED); np.random.seed(SEED)

BASE_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/"
TRAIN_URL, TEST_URL = BASE_URL + "KDDTrain%2B.txt", BASE_URL + "KDDTest%2B.txt"
NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty",
]
CATEGORICAL = ["protocol_type", "service", "flag"]

# %% [markdown]
# ## 1. Dataset  (NSL-KDD: train file for training, test file as genuinely unseen "live" traffic)

# %%
class Dataset:
    """Loads NSL-KDD. KDDTrain+ is used for training; KDDTest+ (which contains attack
    types never seen in training) is used as the live traffic stream."""

    def __init__(self, sample_size: int = TRAIN_SAMPLE, cache_dir: str = "."):
        self.sample_size = sample_size
        self.cache_dir = cache_dir
        self.train_df: Optional[pd.DataFrame] = None
        self.live_df: Optional[pd.DataFrame] = None

    def _fetch(self, url: str, name: str) -> pd.DataFrame:
        path = os.path.join(self.cache_dir, name)
        if not os.path.exists(path):
            print(f"[Dataset] Downloading {name} ...")
            urllib.request.urlretrieve(url, path)
        df = pd.read_csv(path, names=NSL_KDD_COLUMNS)
        df["binary_label"] = (df["label"] != "normal").astype(int)
        return df

    def loadData(self):
        train = self._fetch(TRAIN_URL, "KDDTrain+.txt")
        live = self._fetch(TEST_URL, "KDDTest+.txt")
        self.train_df = train.sample(n=min(self.sample_size, len(train)),
                                     random_state=SEED).reset_index(drop=True)
        self.live_df = live.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
        seen = set(train["label"])
        novel = (~self.live_df["label"].isin(seen)).sum()
        print(f"[Dataset] Train sample: {len(self.train_df)} rows "
              f"({self.train_df['binary_label'].sum()} attack). "
              f"Live stream: {len(self.live_df)} rows, {novel} of them use attack types never seen in training.")
        return self.train_df, self.live_df

    def splitData(self, ratio: float = 0.8):
        n = int(len(self.train_df) * ratio)
        train, test = self.train_df.iloc[:n].reset_index(drop=True), self.train_df.iloc[n:].reset_index(drop=True)
        print(f"[Dataset] Split into {len(train)} train / {len(test)} test records.")
        return train, test

# %% [markdown]
# ## 2. Preprocessor  (one-hot encode + log scale + standardise)

# %%
class Preprocessor:
    def __init__(self):
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.scaler = StandardScaler()
        self.numeric_cols: List[str] = []
        self.feature_names: List[str] = []
        self.fitted = False

    def fit(self, df: pd.DataFrame):
        self.numeric_cols = [c for c in NSL_KDD_COLUMNS if c not in CATEGORICAL + ["label", "difficulty"]]
        self.encoder.fit(df[CATEGORICAL])
        num = np.log1p(df[self.numeric_cols].astype(float).clip(lower=0).values)  # tame heavy tails
        self.scaler.fit(num)
        self.feature_names = self.numeric_cols + list(self.encoder.get_feature_names_out(CATEGORICAL))
        self.fitted = True
        self.attack_mask = np.r_[np.ones(len(self.numeric_cols)), np.zeros(len(self.feature_names) - len(self.numeric_cols))]
        print(f"[Preprocessor] Fitted. {len(self.feature_names)} input features after encoding.")
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        num = np.log1p(df[self.numeric_cols].astype(float).clip(lower=0).values)
        num = self.scaler.transform(num)
        cat = self.encoder.transform(df[CATEGORICAL])
        return np.hstack([num, cat]).astype(np.float64)

    # names kept from the UML diagram
    def normalize(self, df): return self.transform(df)
    def encodeFeatures(self, df): return self.transform(df)

# %% [markdown]
# ## 3. IDSModel  (small neural network in NumPy, so we can compute REAL gradients for FGSM)

# %%
class IDSModel:
    """MLP: input -> 64 -> 32 -> 1 (sigmoid). Binary: 0 = normal, 1 = attack."""

    def __init__(self, hidden=(64, 32), lr=0.003, epochs=25, batch=256, seed=SEED):
        self.hidden, self.lr, self.epochs, self.batch = hidden, lr, epochs, batch
        self.rng = np.random.default_rng(seed)
        self.weights: Optional[List[np.ndarray]] = None   # [W1,b1,W2,b2,W3,b3]
        self.isRobust: bool = False

    # ---- helpers ----
    def _init(self, d):
        sizes = [d, *self.hidden, 1]
        w = []
        for i in range(len(sizes) - 1):
            w.append(self.rng.normal(0, np.sqrt(2 / sizes[i]), (sizes[i], sizes[i + 1])))
            w.append(np.zeros(sizes[i + 1]))
        self.weights = w

    def _forward(self, X):
        W1, b1, W2, b2, W3, b3 = self.weights
        z1 = X @ W1 + b1; a1 = np.maximum(z1, 0)
        z2 = a1 @ W2 + b2; a2 = np.maximum(z2, 0)
        z3 = a2 @ W3 + b3
        p = 1 / (1 + np.exp(-np.clip(z3, -30, 30)))
        return z1, a1, z2, a2, p

    def _backward(self, X, y, cache):
        """Returns weight gradients and the gradient of the loss w.r.t. the INPUT X."""
        W1, b1, W2, b2, W3, b3 = self.weights
        z1, a1, z2, a2, p = cache
        n = len(X)
        dz3 = (p - y.reshape(-1, 1))                 # d(BCE)/dz3 (per sample)
        dW3 = a2.T @ dz3 / n; db3 = dz3.mean(0)
        da2 = dz3 @ W3.T; dz2 = da2 * (z2 > 0)
        dW2 = a1.T @ dz2 / n; db2 = dz2.mean(0)
        da1 = dz2 @ W2.T; dz1 = da1 * (z1 > 0)
        dW1 = X.T @ dz1 / n; db1 = dz1.mean(0)
        dX = dz1 @ W1.T                              # per-sample input gradient
        return [dW1, db1, dW2, db2, dW3, db3], dX

    # ---- public API ----
    def train(self, X: np.ndarray, y: np.ndarray, warm_start: bool = False,
              adv_eps: float = 0.0, mask: Optional[np.ndarray] = None):
        """adv_eps > 0 turns on adversarial training: every mini-batch is attacked with FGSM
        (against the CURRENT weights) and the model learns from clean + attacked copies."""
        tag = f", adversarial training eps={adv_eps}" if adv_eps > 0 else ""
        print(f"[IDSModel] Training on {len(X)} samples ({int(y.sum())} attack{tag})...")
        if self.weights is None or not warm_start:
            self._init(X.shape[1])
        m = [np.zeros_like(w) for w in self.weights]; v = [np.zeros_like(w) for w in self.weights]
        t = 0
        for ep in range(self.epochs):
            idx = self.rng.permutation(len(X))
            for s in range(0, len(X), self.batch):
                b = idx[s:s + self.batch]
                xb, yb = X[b], y[b]
                if adv_eps > 0:
                    g = self.input_gradient(xb, yb)
                    step = adv_eps * np.sign(g)
                    if mask is not None: step = step * mask
                    xb, yb = np.vstack([xb, xb + step]), np.concatenate([yb, yb])
                grads, _ = self._backward(xb, yb, self._forward(xb))
                t += 1
                for i, g in enumerate(grads):          # Adam
                    m[i] = 0.9 * m[i] + 0.1 * g
                    v[i] = 0.999 * v[i] + 0.001 * g * g
                    mh, vh = m[i] / (1 - 0.9 ** t), v[i] / (1 - 0.999 ** t)
                    self.weights[i] -= self.lr * mh / (np.sqrt(vh) + 1e-8)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._forward(X)[-1].ravel()           # P(attack)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)

    def confidence(self, X: np.ndarray) -> np.ndarray:
        p = self.predict_proba(X)
        return np.maximum(p, 1 - p)

    def input_gradient(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """dLoss/dInput for each sample (what FGSM needs)."""
        _, dX = self._backward(X, y, self._forward(X))
        return dX

# %% [markdown]
# ## 4. FGSMGenerator  (real Fast Gradient Sign Method: x_adv = x + eps * sign(dLoss/dx))

# %%
class FGSMGenerator:
    def __init__(self, epsilon: float = EPSILON, mask: Optional[np.ndarray] = None):
        self.epsilon = epsilon
        self.mask = mask          # 1 for features an attacker may change (numeric), 0 for one-hot flags

    def generateAdversarial(self, X: np.ndarray, y: np.ndarray, model: IDSModel) -> np.ndarray:
        print(f"[FGSMGenerator] Crafting {len(X)} adversarial samples with real gradients (eps={self.epsilon})...")
        step = self.epsilon * np.sign(model.input_gradient(X, y))
        if self.mask is not None: step = step * self.mask
        return X + step

# %% [markdown]
# ## 5. AdversarialTrainer

# %%
class AdversarialTrainer:
    def combineData(self, Xc, yc, Xa, ya):
        X, y = np.vstack([Xc, Xa]), np.concatenate([yc, ya])
        print(f"[AdversarialTrainer] Combined {len(Xc)} clean + {len(Xa)} adversarial = {len(X)}")
        return X, y

    def trainRobustModel(self, model: IDSModel, X, y, warm_start=False, adv_eps=0.0, mask=None) -> IDSModel:
        model.train(X, y, warm_start=warm_start, adv_eps=adv_eps, mask=mask)
        model.isRobust = True
        print("[AdversarialTrainer] Produced robust version of IDSModel.")
        return model

# %% [markdown]
# ## 6. Evaluator

# %%
class Evaluator:
    def testModel(self, model, X, y, name="model") -> Dict:
        pred = model.predict(X)
        r = {"name": name, "accuracy": round(accuracy_score(y, pred), 4),
             "precision": round(precision_score(y, pred, zero_division=0), 4),
             "recall": round(recall_score(y, pred, zero_division=0), 4),
             "f1": round(f1_score(y, pred, zero_division=0), 4), "isRobust": model.isRobust}
        print(f"[Evaluator] {r}")
        return r

    def compareResults(self, baseline: Dict, robust: Dict, key="accuracy") -> Dict:
        c = {f"baseline_{key}": baseline[key], f"robust_{key}": robust[key],
             "improvement": round(robust[key] - baseline[key], 4)}
        print(f"[Evaluator] Comparison: {c}")
        return c

# %% [markdown]
# ## 7. Explanation: ExplanationReport + GenAIExplainer (real LLM if a key is set, otherwise a local explainer that uses the actual feature values)

# %%
@dataclass
class ExplanationReport:
    attackType: str = ""
    explanationText: str = ""
    recommendedAction: str = ""
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))

    def toSummary(self) -> str:
        return (f"[{self.timestamp}] Attack='{self.attackType}' | "
                f"Explanation='{self.explanationText}' | Action='{self.recommendedAction}'")


class GenAIExplainer:
    def __init__(self, llmModel: str = "auto",
                 promptTemplate: str = ("You are a SOC analyst assistant. A network intrusion detector flagged a "
                                        "connection it is unsure about.\nPrediction: {label} (confidence {conf}).\n"
                                        "Most unusual features (value, z-score vs normal traffic): {features}.\n"
                                        "In 3 short sentences: say what kind of attack this resembles, why, and what "
                                        "the analyst should do. Do not invent facts beyond the numbers given.")):
        self.promptTemplate = promptTemplate
        self.anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
        self.gemini_key = os.environ.get("GEMINI_API_KEY", "")
        if llmModel == "auto":
            llmModel = ("claude" if self.anthropic_key else "gemini" if self.gemini_key else "local-rule-explainer")
        self.llmModel = llmModel
        self.normal_mean: Optional[np.ndarray] = None
        self.normal_std: Optional[np.ndarray] = None
        self.feature_names: List[str] = []

    def fit_baseline(self, X_normal: np.ndarray, feature_names: List[str]):
        self.normal_mean, self.normal_std = X_normal.mean(0), X_normal.std(0) + 1e-6
        self.feature_names = feature_names

    def _top_features(self, x: np.ndarray, k=4):
        z = (x - self.normal_mean) / self.normal_std
        order = np.argsort(-np.abs(z))[:k]
        return [(self.feature_names[i], float(x[i]), float(z[i])) for i in order]

    def _call_llm(self, prompt: str) -> Optional[str]:
        try:
            if self.llmModel == "claude":
                body = json.dumps({"model": LLM_MODEL_CLAUDE, "max_tokens": 300,
                                   "messages": [{"role": "user", "content": prompt}]}).encode()
                req = urllib.request.Request("https://api.anthropic.com/v1/messages", body, {
                    "x-api-key": self.anthropic_key, "anthropic-version": "2023-06-01",
                    "content-type": "application/json"})
                return json.load(urllib.request.urlopen(req, timeout=30))["content"][0]["text"].strip()
            if self.llmModel == "gemini":
                url = ("https://generativelanguage.googleapis.com/v1beta/models/"
                       f"{LLM_MODEL_GEMINI}:generateContent?key={self.gemini_key}")
                body = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode()
                req = urllib.request.Request(url, body, {"content-type": "application/json"})
                return json.load(urllib.request.urlopen(req, timeout=30))["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception as e:                      # network / key / model errors -> fall back
            print(f"[GenAIExplainer] LLM call failed ({type(e).__name__}); using local explainer.")
        return None

    def _local(self, feats, label, conf) -> str:
        parts = [f"{n}={v:.2f} ({'+' if z > 0 else ''}{z:.1f} sd)" for n, v, z in feats]
        names = " ".join(n for n, _, _ in feats)
        if any(k in names for k in ("serror", "count", "srv_count")):
            kind = "a flood / DoS style pattern (many connections or SYN errors)"
        elif any(k in names for k in ("rerror", "diff_srv", "dst_host_diff")):
            kind = "a probing / scanning pattern (many services or rejected connections)"
        elif any(k in names for k in ("failed_logins", "guest", "hot", "root", "compromised")):
            kind = "a remote access / privilege abuse pattern"
        else:
            kind = "an unfamiliar traffic pattern"
        return (f"Detector says '{label}' with only {conf:.0%} confidence. The most unusual features are "
                f"{', '.join(parts)}. This resembles {kind}.")

    def explainAttack(self, x: np.ndarray, prediction: Dict) -> str:
        feats = self._top_features(x)
        label = "attack" if prediction["prediction"] == 1 else "normal"
        text = None
        if self.llmModel in ("claude", "gemini"):
            fs = "; ".join(f"{n}={v:.2f} (z={z:+.1f})" for n, v, z in feats)
            text = self._call_llm(self.promptTemplate.format(label=label, conf=prediction["confidence"], features=fs))
        if text is None:
            text = self._local(feats, label, prediction["confidence"])
        print(f"[GenAIExplainer] ({self.llmModel}) Generated explanation.")
        return text

    def summarizePatch(self, cycle: int, recs: List[dict]) -> str:
        """GenAI summary of what was broken in this cycle and what the self-healer patched."""
        fixed = [r for r in recs if r["status"] == "PATCHED (was wrong)"]
        firm = [r for r in recs if r["status"] == "HARDENED (was right but unsure)"]
        bad = [r for r in recs if r["status"] == "STILL WRONG"]
        kinds = pd.Series([r["attack_type"] for r in recs]).value_counts().to_dict() if recs else {}
        facts = (f"Cycle {cycle}: {len(recs)} flagged. {len(fixed)} were misclassified and are now correct, "
                 f"{len(firm)} were correct but low-confidence and are now confident, {len(bad)} are still wrong. "
                 f"Attack families seen: {kinds}.")
        if self.llmModel in ("claude", "gemini") and recs:
            t = self._call_llm("You are a SOC analyst assistant. Write a 3-sentence patch note for a security "
                               "team from these facts, no invented details: " + facts)
            if t: return t
        return facts

    def generateReport(self, x: np.ndarray, explanation: str) -> ExplanationReport:
        low = explanation.lower()
        kind = ("DoS / Flood" if "flood" in low or "dos" in low else
                "Probe / Scan" if "prob" in low or "scan" in low else
                "Remote access / Privilege abuse" if "privilege" in low or "login" in low else "Unknown / Zero-Day")
        return ExplanationReport(kind, explanation,
                                 "Isolate the source host, review logs, and approve the sample for retraining.")

# %% [markdown]
# ## 8. IncidentKnowledgeBase + AttackMonitor

# %%
class IncidentKnowledgeBase:
    def __init__(self):
        self.incidents: List[dict] = []
        self.unlabeledSamples: List[dict] = []

    def addIncident(self, sample: dict, report: ExplanationReport):
        self.incidents.append({"sample": sample, "report": report})
        self.unlabeledSamples.append(sample)

    def getUnlabeledSamples(self) -> List[dict]:
        return self.unlabeledSamples

    def clearProcessed(self):
        print(f"[IncidentKnowledgeBase] Cleared {len(self.unlabeledSamples)} processed samples.")
        self.unlabeledSamples = []


class AttackMonitor:
    def __init__(self, anomalyThreshold: float = CONF_THRESHOLD):
        self.liveTrafficBuffer: List[dict] = []
        self.anomalyThreshold = anomalyThreshold
        self.knownAttackSignatures: List[str] = []

    def monitorPredictions(self, predictions: List[dict]):
        self.liveTrafficBuffer.extend(predictions)

    def detectUnknownAttack(self, prediction: dict) -> bool:
        return prediction.get("confidence", 1.0) < self.anomalyThreshold

# %% [markdown]
# ## 9. SelfHealingController  (detect -> explain -> log -> retrain -> deploy, repeated)

# %%
class SelfHealingController:
    def __init__(self, cycleIntervalSec: int = 0):
        self.isActive = False
        self.cycleIntervalSec = cycleIntervalSec
        self.lastCycleTimestamp = ""
        self.history: List[dict] = []
        self.patch_log: List[dict] = []       # one row per flagged sample: what was broken, what fixed it
        self.patch_notes: List[str] = []      # GenAI patch note per cycle

    def logIncident(self, sample, report, kb):
        kb.addIncident(sample, report)

    def triggerRetraining(self, samples, trainer, model, Xtrain, ytrain, fgsm):
        """Retrain on: original training data + analyst-confirmed new samples, with adversarial training on top."""
        Xn = np.array([s["x"] for s in samples]); yn = np.array([s["label"] for s in samples])
        Xc, yc = np.vstack([Xtrain, Xn]), np.concatenate([ytrain, yn])
        print(f"[SelfHealingController] Retraining with {len(samples)} new incident samples...")
        return trainer.trainRobustModel(model, Xc, yc, adv_eps=fgsm.epsilon, mask=fgsm.mask), Xc, yc

    def deployModel(self, model):
        self.lastCycleTimestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[SelfHealingController] Auto-deployed updated model at {self.lastCycleTimestamp}.")

    def runFeedbackLoop(self, model, monitor, explainer, kb, trainer, fgsm, evaluator,
                        Xtrain, ytrain, X_live, y_live, cycle_no=1, max_explain=5):
        self.isActive = True
        print(f"\n[SelfHealingController] === Feedback loop #{cycle_no} ===")
        conf = model.confidence(X_live); pred = model.predict(X_live)
        preds = [{"prediction": int(p), "confidence": float(c)} for p, c in zip(pred, conf)]
        monitor.monitorPredictions(preds)

        acc_before = accuracy_score(y_live, pred)
        flagged = [i for i, p in enumerate(preds) if monitor.detectUnknownAttack(p)]
        print(f"[AttackMonitor] {len(flagged)} of {len(X_live)} live samples flagged as unknown (conf < {monitor.anomalyThreshold}).")

        pending = []
        if flagged:
            for n, i in enumerate(flagged):
                sample = {"x": X_live[i], "label": int(y_live[i])}    # label = analyst-confirmed (from dataset)
                if n < max_explain or explainer.llmModel not in ("claude", "gemini"):   # cap only real LLM calls (cost)
                    report = explainer.generateReport(X_live[i], explainer.explainAttack(X_live[i], preds[i]))
                    if n == 0: print("   example ->", report.toSummary())
                else:
                    report = ExplanationReport("Unknown", "Auto-logged (explanation skipped to save LLM calls).",
                                               "Queue for retraining.")
                self.logIncident(sample, report, kb)
                pending.append((i, report))
            print(f"[IncidentKnowledgeBase] Logged {len(flagged)} incidents. Total: {len(kb.incidents)}")
            model, Xnew, ynew = self.triggerRetraining(kb.getUnlabeledSamples(), trainer, model, Xtrain, ytrain, fgsm)
            self.deployModel(model)
            kb.clearProcessed()
            Xtrain, ytrain = Xnew, ynew                                # healed model keeps the new knowledge
        else:
            print("[SelfHealingController] Nothing flagged. No action needed.")

        # ---- what was broken, and was it patched? (compare the SAME samples before/after healing) ----
        if pending:
            idx = [i for i, _ in pending]
            p2, c2 = model.predict(X_live[idx]), model.confidence(X_live[idx])
            recs = []
            for k, (i, rep_) in enumerate(pending):
                truth, was_ok, now_ok = int(y_live[i]), preds[i]["prediction"] == int(y_live[i]), int(p2[k]) == int(y_live[i])
                status = ("PATCHED (was wrong)" if (not was_ok and now_ok) else
                          "HARDENED (was right but unsure)" if (was_ok and now_ok) else
                          "STILL WRONG" if not now_ok else "REGRESSED")
                recs.append({"cycle": cycle_no, "sample_id": i, "attack_type": rep_.attackType,
                             "truth": "attack" if truth else "normal",
                             "before": "attack" if preds[i]["prediction"] else "normal", "conf_before": round(preds[i]["confidence"], 2),
                             "after": "attack" if p2[k] else "normal", "conf_after": round(float(c2[k]), 2),
                             "status": status, "explanation": rep_.explanationText})
            self.patch_log.extend(recs)
            note = explainer.summarizePatch(cycle_no, recs)
            self.patch_notes.append(note)
            print(f"[GenAI patch note] {note}")

        acc_after = accuracy_score(y_live, model.predict(X_live))
        self.history.append({"cycle": cycle_no, "flagged": len(flagged),
                             "live_acc_before": round(acc_before, 4), "live_acc_after": round(acc_after, 4)})
        print(f"[SelfHealingController] Accuracy on this live batch: {acc_before:.4f} -> {acc_after:.4f}")
        self.isActive = False
        return model, Xtrain, ytrain

# %% [markdown]
# ## 10. Run the whole project

# %%
dataset = Dataset()
dataset.loadData()
train_df, test_df = dataset.splitData(0.8)

prep = Preprocessor().fit(train_df)
X_train, y_train = prep.transform(train_df), train_df["binary_label"].values
X_test,  y_test  = prep.transform(test_df),  test_df["binary_label"].values

# --- baseline neural IDS ---
evaluator = Evaluator()
baseline = IDSModel(); baseline.train(X_train, y_train)
base_clean = evaluator.testModel(baseline, X_test, y_test, "baseline (clean test)")

# --- real FGSM attack against the baseline ---
fgsm = FGSMGenerator(EPSILON, mask=prep.attack_mask)
X_test_adv = fgsm.generateAdversarial(X_test, y_test, baseline)
base_adv = evaluator.testModel(baseline, X_test_adv, y_test, "baseline (FGSM test)")

# --- adversarial training -> robust model ---
trainer = AdversarialTrainer()
# (a) one-shot version from the UML: craft adversarial copies once against the baseline and add them
X_train_adv = fgsm.generateAdversarial(X_train, y_train, baseline)
trainer.combineData(X_train, y_train, X_train_adv, y_train)
# (b) the stronger version we actually deploy: craft fresh FGSM samples every mini-batch while training
Xc, yc = X_train, y_train
robust = trainer.trainRobustModel(IDSModel(seed=SEED + 1), Xc, yc, adv_eps=EPSILON, mask=prep.attack_mask)
rob_clean = evaluator.testModel(robust, X_test, y_test, "robust (clean test)")
# fair test: attack the robust model itself (white-box), and also the transferred attack from the baseline
X_test_adv_rob = FGSMGenerator(EPSILON, prep.attack_mask).generateAdversarial(X_test, y_test, robust)
rob_adv_wb = evaluator.testModel(robust, X_test_adv_rob, y_test, "robust (FGSM against itself)")
rob_adv_tr = evaluator.testModel(robust, X_test_adv, y_test, "robust (FGSM from baseline)")

print("\n--- Adversarial accuracy: baseline vs robust (attack on each model itself) ---")
evaluator.compareResults(base_adv, rob_adv_wb)
print("--- Clean accuracy ---")
evaluator.compareResults(base_clean, rob_clean)

# --- reference: a classic Random Forest on the same data (the 'signature'-style hybrid partner) ---
rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=SEED, n_jobs=-1).fit(X_train, y_train)
print(f"[Reference] RandomForest clean test accuracy: {accuracy_score(y_test, rf.predict(X_test)):.4f}")

# %% [markdown]
# ## 11. Self-healing loop on genuinely unseen live traffic (KDDTest+ contains new attack types)

# %%
explainer = GenAIExplainer()
explainer.fit_baseline(X_train[y_train == 0], prep.feature_names)
monitor, kb = AttackMonitor(CONF_THRESHOLD), IncidentKnowledgeBase()
controller = SelfHealingController()

live_df = dataset.live_df
X_live_all, y_live_all = prep.transform(live_df), live_df["binary_label"].values
live_model, Xtr, ytr = copy.deepcopy(robust), Xc, yc   # copy, so 'robust' stays as the before-healing reference

for c in range(CYCLES):
    s, e = c * BATCH_SIZE, (c + 1) * BATCH_SIZE
    live_model, Xtr, ytr = controller.runFeedbackLoop(
        live_model, monitor, explainer, kb, trainer, fgsm, evaluator,
        Xtr, ytr, X_live_all[s:e], y_live_all[s:e], cycle_no=c + 1)

# held-out live data that the loop never touched
held_s = CYCLES * BATCH_SIZE
print("\n[Result] Accuracy on LIVE data never used in the loop:")
print(f"   before self-healing (robust model): {accuracy_score(y_live_all[held_s:], robust.predict(X_live_all[held_s:])):.4f}")
print(f"   after  self-healing (healed model): {accuracy_score(y_live_all[held_s:], live_model.predict(X_live_all[held_s:])):.4f}")
print("Incident Knowledge Base incidents logged:", len(kb.incidents))
print(pd.DataFrame(controller.history).to_string(index=False))

# %% [markdown]
# ## 12. Healing report: what was broken and what the self-healer patched (with GenAI explanations)

# %%
pd.set_option("display.max_colwidth", 110, "display.width", 250)
patch = pd.DataFrame(controller.patch_log)
print("=== GenAI patch notes (one per cycle) ===")
for n in controller.patch_notes: print(" -", n)
print("\n=== Status of every flagged sample ===")
print(patch["status"].value_counts().to_string())
print("\n=== Samples that were BROKEN (misclassified) and are now PATCHED ===")
fixed = patch[patch["status"] == "PATCHED (was wrong)"]
print(fixed[["cycle", "sample_id", "attack_type", "truth", "before", "conf_before", "after", "conf_after", "explanation"]]
      .head(10).to_string(index=False))
print("\n=== Still broken after healing (needs more data / analyst) ===")
print(patch[patch["status"] == "STILL WRONG"][["cycle", "sample_id", "attack_type", "truth", "before", "after", "explanation"]]
      .head(5).to_string(index=False))
patch.to_csv("healing_report.csv", index=False)
print("\nFull report saved to healing_report.csv (Colab: Files panel on the left).")
try:
    from IPython.display import display; display(patch.head(15))
except Exception: pass

# %% [markdown]
# ## 13. Charts

# %%
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
names = ["Baseline\nclean", "Baseline\nFGSM", "Robust\nclean", "Robust\nFGSM (own)"]
vals = [base_clean["accuracy"], base_adv["accuracy"], rob_clean["accuracy"], rob_adv_wb["accuracy"]]
ax[0].bar(names, vals, color=["#4C78A8", "#E45756", "#4C78A8", "#54A24B"])
ax[0].set_ylim(0, 1.05); ax[0].set_title("Accuracy under FGSM attack")
for i, v in enumerate(vals): ax[0].text(i, v + 0.01, f"{v:.2f}", ha="center")
h = pd.DataFrame(controller.history)
ax[1].plot(h["cycle"], h["live_acc_before"], "o-", label="before healing")
ax[1].plot(h["cycle"], h["live_acc_after"], "s-", label="after healing")
ax[1].set_xlabel("Self-healing cycle"); ax[1].set_ylabel("Accuracy on live batch")
ax[1].set_title("Self-healing on unseen traffic"); ax[1].legend(); ax[1].set_xticks(h["cycle"])
plt.tight_layout(); plt.savefig("results.png", dpi=120); plt.show()
