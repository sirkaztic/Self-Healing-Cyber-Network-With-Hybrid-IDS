
"""Self_Healing_IDS_Colab (1).ipynb
Original file is located at
    https://colab.research.google.com/drive/1neLf6tOW3DEDw4NdsIU8lIKHB5lYyT-G
## 0. Config
"""

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

"""## 1. Dataset  (NSL-KDD: train file for training, test file as genuinely unseen "live" traffic)"""

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

"""## 2. Preprocessor  (one-hot encode + log scale + standardise)"""

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

"""## 3. IDSModel  (small neural network in NumPy, so we can compute REAL gradients for FGSM)"""

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

"""## 4. FGSMGenerator  (real Fast Gradient Sign Method: x_adv = x + eps * sign(dLoss/dx))"""

class FGSMGenerator:
    def __init__(self, epsilon: float = EPSILON, mask: Optional[np.ndarray] = None):
        self.epsilon = epsilon
        self.mask = mask          # 1 for features an attacker may change (numeric), 0 for one-hot flags

    def generateAdversarial(self, X: np.ndarray, y: np.ndarray, model: IDSModel) -> np.ndarray:
        print(f"[FGSMGenerator] Crafting {len(X)} adversarial samples with real gradients (eps={self.epsilon})...")
        step = self.epsilon * np.sign(model.input_gradient(X, y))
        if self.mask is not None: step = step * self.mask
        return X + step

"""## 5. AdversarialTrainer"""

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

"""## 6. Evaluator"""

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

"""## 7. Explanation: ExplanationReport + GenAIExplainer (real LLM if a key is set, otherwise a local explainer that uses the actual feature values)"""

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

"""## 8. IncidentKnowledgeBase + AttackMonitor"""

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

"""## 9. SelfHealingController  (detect -> explain -> log -> retrain -> deploy, repeated)"""

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


# ==================================================================================
# PART B  -  DESKTOP UI
# ==================================================================================
import os, sys, glob, types, threading, queue, io, copy, shutil, traceback, contextlib, time
from collections import deque as _deque
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ----------------------------------------------------------------------------------
# 2. Dataset helpers (custom CSV support)
# ----------------------------------------------------------------------------------
NORMAL_WORDS = {"normal", "benign", "0", "legitimate", "clean", "none", "no", "false", "background"}
LABEL_HINTS = ("label", "class", "attack", "target", "category", "result")
# Columns that leak the answer or are just row identifiers (UNSW-NB15: id, attack_cat, label ...)
LEAK_NAMES = {"id", "rowid", "row_id", "index", "unnamed: 0", "attack_cat", "attack_category", "attack_type",
              "attack", "label", "class", "category", "target", "difficulty"}
ID_NAMES = {"id", "rowid", "row_id", "index", "unnamed: 0", "srcip", "dstip", "sport", "dsport", "stime", "ltime",
            # CIC-IDS2017 identifiers (kept out of the model, used only to place alerts on the network map)
            "flow id", "source ip", "destination ip", "src ip", "dst ip", "src_ip", "dst_ip", "timestamp",
            "source port", "src port", "srcport", "sourceip", "destinationip", "id.orig_h", "id.resp_h"}
SRC_NAMES = {"source ip", "src ip", "srcip", "src_ip", "sourceip", "id.orig_h"}
DST_NAMES = {"destination ip", "dst ip", "dstip", "dst_ip", "destinationip", "id.resp_h"}


def _find_col(df, names):
    for c in df.columns:
        if str(c).lower().strip() in names:
            return c
    return None


def _read_csv(path, **kw):
    """CIC-IDS2017 files are not UTF-8 (the 'Web Attack – ...' dash), so fall back to latin-1."""
    try:
        return pd.read_csv(path, low_memory=False, **kw)
    except UnicodeDecodeError:
        return pd.read_csv(path, low_memory=False, encoding="latin-1", **kw)


def read_table(path, nsl_columns):
    """Reads a CSV/TXT. Auto-detects the headerless NSL-KDD format. Strips header whitespace (CIC-IDS2017)."""
    probe = _read_csv(path, header=None, nrows=2)
    if probe.shape[1] == len(nsl_columns) and str(probe.iloc[0, 1]).lower() in ("tcp", "udp", "icmp"):
        df = _read_csv(path, names=nsl_columns)
        return df.drop(columns=["difficulty"], errors="ignore"), True
    df = _read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df, False


def guess_label_col(cols):
    low = {c.lower().strip(): c for c in cols}
    for h in LABEL_HINTS:
        for k, c in low.items():
            if k == h or k.endswith(h):
                return c
    return cols[-1]


def guess_normals(series):
    vals = [str(v).strip() for v in pd.Series(series).dropna().unique()]
    hit = [v for v in vals if v.lower() in NORMAL_WORDS]
    return ", ".join(hit) if hit else ""


def build_frame(df, label_col, normals, feat_cols=None, cats=None):
    """Turns any table into the layout the base classes expect (features + label + binary_label)."""
    df = df.copy()
    s = df[label_col]
    if pd.api.types.is_numeric_dtype(s) and (s.dropna() % 1 == 0).all():
        lab = s.fillna(-1).astype(int).astype(str)
    else:
        lab = s.astype(str).str.strip()
    if feat_cols is None:
        feat_cols, cats = [], []
        dropped = []
        y_str = s.astype(str).str.strip()
        for c in df.columns:
            if c == label_col:
                continue
            cl = str(c).lower().strip()
            # 1) known label-companions / row ids (e.g. UNSW-NB15 'attack_cat', 'label', 'id')
            if cl in LEAK_NAMES or cl in ID_NAMES:
                dropped.append(c)
                continue
            col = df[c]
            # 2) automatic leak guard: a non-numeric column that determines the label (near) perfectly
            if not pd.api.types.is_numeric_dtype(col) and col.nunique() > 1:
                purity = df.groupby(col.astype(str))[label_col].agg(lambda x: x.astype(str).value_counts(normalize=True).iloc[0])
                w = col.astype(str).value_counts(normalize=True)
                if float((purity * w.reindex(purity.index)).sum()) > 0.995 and y_str.nunique() > 1 and col.nunique() >= y_str.nunique():
                    dropped.append(c)
                    continue
            if pd.api.types.is_numeric_dtype(col) or pd.api.types.is_bool_dtype(col):
                if col.notna().any():
                    feat_cols.append(c)
            else:
                conv = pd.to_numeric(col, errors="coerce")
                if conv.notna().mean() > 0.95:
                    df[c] = conv
                    feat_cols.append(c)
                elif col.nunique() <= 100:
                    cats.append(c)
                    feat_cols.append(c)
        if dropped:
            print(f"[Dataset] Excluded leakage/ID columns: {dropped}")
        if not feat_cols:
            raise ValueError("No usable feature columns found in this file.")
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        raise ValueError(f"File is missing training columns: {missing[:8]}")
    for c in feat_cols:
        if c in cats:
            df[c] = df[c].fillna("NA").astype(str)
        else:
            v = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
            df[c] = v.fillna(v.median() if v.notna().any() else 0).astype(float)
    out = df[feat_cols].copy()
    sc, dc = _find_col(df, SRC_NAMES), _find_col(df, DST_NAMES)
    if sc is not None and dc is not None:          # host info rides along (never a model feature)
        out["_src"], out["_dst"] = df[sc].astype(str).str.strip().values, df[dc].astype(str).str.strip().values
    out["label"] = lab.values
    out["binary_label"] = (~lab.str.lower().isin(normals)).astype(int).values
    return out, feat_cols, cats


class QueueWriter(io.TextIOBase):
    def __init__(self, q):
        self.q = q

    def write(self, s):
        if s:
            self.q.put(("log", s))
        return len(s)

    def flush(self):
        pass


def load_nsl(cols, n, seed):
    """Downloads NSL-KDD (cached in ./data). Train file -> training sample, test file -> unseen live stream."""
    import urllib.request
    base = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/"
    frames = []
    for url, name in ((base + "KDDTrain%2B.txt", "KDDTrain+.txt"), (base + "KDDTest%2B.txt", "KDDTest+.txt")):
        path = os.path.join(DATA_DIR, name)
        if not os.path.exists(path):
            print(f"[Data] Downloading {name} ...")
            urllib.request.urlretrieve(url, path)
        df = pd.read_csv(path, names=cols)
        df["binary_label"] = (df["label"] != "normal").astype(int)
        frames.append(df)
    train, live = frames
    t = train.sample(n=min(n, len(train)), random_state=seed).reset_index(drop=True)
    l = live.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    print(f"[Data] NSL-KDD: {len(t)} training rows ({int(t['binary_label'].sum())} attack), {len(l)} live rows.")
    return t, l


class StopRequested(Exception):
    pass


def mk(cls, **kw):
    """Create cls(...) passing only the keyword arguments its __init__ really accepts."""
    import inspect
    try:
        params = inspect.signature(cls.__init__).parameters
        accepts_any = any(p.kind == p.VAR_KEYWORD for p in params.values())
        ok = {k: v for k, v in kw.items() if accepts_any or k in params}
    except (TypeError, ValueError):
        ok = kw
    if "cache_dir" in kw and "cache_dir" not in ok:
        os.chdir(DATA_DIR)          # base class has no cache_dir -> let it download into ./data
    return cls(**ok)


# ----------------------------------------------------------------------------------
# 3. The application
# ----------------------------------------------------------------------------------
BG, PRIMARY, ACCENT, GOOD, BAD, MUTED = "#f4f6f9", "#12263f", "#2e86de", "#2ca02c", "#d9534f", "#6b7785"


# ==================================================================================
# PART C  -  NETWORK HEALING MODULE
# Heals a damaged network:  Detect -> Contain -> Fail over -> Restore -> Repair -> Verify -> Harden
# Works on a simulated topology (built-in enterprise network or your own edge-list CSV).
# ==================================================================================
from matplotlib.lines import Line2D

USABLE = ("healthy", "degraded", "restored")
STATE_COLOR = {"healthy": "#2ca02c", "degraded": "#f0ad4e", "compromised": "#d9534f",
               "failed": "#444444", "quarantined": "#8e5ea2", "restored": "#2e86de"}
DAMAGE_SCENARIOS = ["Node failures", "Link cuts", "Malware outbreak", "DDoS attack", "Ransomware", "Combined disaster"]
ROLE_ORDER = {"router": 0, "firewall": 1, "switch": 2, "server": 3, "host": 4, "workstation": 4}


class CyberNetwork:
    def __init__(self, core="ISP"):
        self.core = core
        self.nodes, self.links = {}, {}

    @staticmethod
    def key(a, b):
        return tuple(sorted((a, b)))

    def add_node(self, name, role, x=0.0, y=0.0, backup=True):
        self.nodes[name] = dict(role=role, state="healthy", backup=backup, x=x, y=y, data_lost=False)

    def add_link(self, a, b, standby=False):
        self.links[self.key(a, b)] = dict(up=True, active=not standby, failover=False)

    def copy(self):
        return copy.deepcopy(self)

    # ---- graph helpers ----
    def _bfs(self, usable_node, usable_link):
        if self.core not in self.nodes or not usable_node(self.nodes[self.core]):
            return set()
        adj = {n: [] for n in self.nodes}
        for (u, v), l in self.links.items():
            if usable_link(l) and usable_node(self.nodes[u]) and usable_node(self.nodes[v]):
                adj[u].append(v)
                adj[v].append(u)
        seen, dq = {self.core}, _deque([self.core])
        while dq:
            c = dq.popleft()
            for m in adj[c]:
                if m not in seen:
                    seen.add(m)
                    dq.append(m)
        return seen

    def reachable(self):
        """Nodes the core can reach through trusted (healthy/degraded/restored) nodes on live links."""
        return self._bfs(lambda d: d["state"] in USABLE, lambda l: l["up"] and l["active"])

    def mgmt_reach(self):
        """Management plane: reaches everything not dead, even compromised hosts."""
        return self._bfs(lambda d: d["state"] != "failed", lambda l: l["up"] and l["active"])

    def health(self):
        reach = self.reachable()
        real = [n for n, d in self.nodes.items() if n != self.core and d["role"] != "attacker"]
        good = [n for n in real if self.nodes[n]["state"] in ("healthy", "restored") and n in reach]
        return len(good) / max(1, len(real))

    def state_counts(self):
        c = {}
        for n, d in self.nodes.items():
            c[d["state"]] = c.get(d["state"], 0) + 1
        return c


def default_network():
    net = CyberNetwork("ISP")
    tiers = [[("ISP", "cloud")], [("R1", "router"), ("R2", "router")], [("FW1", "firewall"), ("FW2", "firewall")],
             [("SW1", "switch"), ("SW2", "switch"), ("SW3", "switch")],
             [("WEB", "server"), ("DB", "server"), ("MAIL", "server"), ("DNS", "server")],
             [(f"WS{i}", "workstation") for i in range(1, 9)]]
    for t, row in enumerate(tiers):
        k = len(row)
        for i, (name, role) in enumerate(row):
            net.add_node(name, role, 5.0 if k == 1 else 0.5 + 9.0 * i / (k - 1), -1.3 * t,
                         backup=name not in ("WS8", "DNS"))
    for a, b in [("ISP", "R1"), ("ISP", "R2"), ("R1", "FW1"), ("R2", "FW2"), ("FW1", "SW1"), ("FW1", "SW2"),
                 ("FW2", "SW2"), ("FW2", "SW3"), ("SW1", "WEB"), ("SW1", "DB"), ("SW2", "MAIL"), ("SW2", "DNS"),
                 ("SW2", "WS1"), ("SW2", "WS2"), ("SW3", "WS3"), ("SW3", "WS4"), ("SW3", "WS5"),
                 ("SW1", "WS6"), ("SW1", "WS7"), ("SW1", "WS8")]:
        net.add_link(a, b)
    for a, b in [("SW1", "SW3"), ("SW2", "DB"), ("SW1", "MAIL"), ("SW3", "DNS"), ("FW1", "FW2"), ("R1", "R2")]:
        net.add_link(a, b, standby=True)       # dark links the healer can light up
    return net


def network_from_edges(df):
    """Edge list CSV: first two columns = node A, node B. Optional column 'standby' (1/true = dark spare link)."""
    cols = list(df.columns)
    a, b = df[cols[0]].astype(str).str.strip(), df[cols[1]].astype(str).str.strip()
    sb = df["standby"].astype(str).str.lower().isin(["1", "true", "yes"]) if "standby" in df.columns else pd.Series(False, index=df.index)
    deg = pd.concat([a, b]).value_counts()
    names = list(deg.index)
    net = CyberNetwork(core=names[0])
    for i, n in enumerate(names):
        ang = 2 * np.pi * i / max(1, len(names))
        d = int(deg[n])
        role = "cloud" if n == net.core else "router" if d >= 4 else "switch" if d == 3 else "workstation"
        net.add_node(n, role, 5 + 4.2 * np.cos(ang), 4.2 * np.sin(ang))
    for x, y, s in zip(a, b, sb):
        if x != y:
            net.add_link(x, y, standby=bool(s))
    return net


def inflict_damage(net, scenario, size, rng):
    """Damages the network in place and returns a list of what happened."""
    log = []
    cand = [n for n in net.nodes if n != net.core and net.nodes[n]["state"] == "healthy"]
    live_links = [k for k, l in net.links.items() if l["up"] and l["active"]]

    def node_failures():
        for n in rng.sample(cand, min(size, len(cand))):
            net.nodes[n]["state"] = "failed"
            log.append(f"{n} ({net.nodes[n]['role']}) crashed")

    def link_cuts():
        for k in rng.sample(live_links, min(size, len(live_links))):
            net.links[k]["up"] = False
            log.append(f"link {k[0]}–{k[1]} cut")

    def malware():
        zero = rng.choice([n for n in cand if net.nodes[n]["role"] in ("workstation", "server", "host")] or cand)
        infected, frontier = {zero}, {zero}
        for _ in range(max(1, size - 1)):
            nxt = set()
            for (u, v), l in net.links.items():
                for s, t in ((u, v), (v, u)):
                    if s in frontier and t not in infected and t != net.core and rng.random() < 0.6:
                        nxt.add(t)
            infected |= nxt
            frontier = nxt
        for n in infected:
            net.nodes[n]["state"] = "compromised"
        log.append(f"malware started at {zero} and spread to {len(infected)} node(s): {sorted(infected)}")

    def ddos():
        tgt = [n for n in cand if net.nodes[n]["role"] in ("server", "router", "firewall")]
        for n in rng.sample(tgt, min(size, len(tgt))):
            net.nodes[n]["state"] = "degraded"
            log.append(f"{n} saturated by DDoS traffic")

    def ransom():
        for n in rng.sample(cand, min(size, len(cand))):
            net.nodes[n]["state"] = "compromised"
            net.nodes[n]["data_lost"] = True
            log.append(f"{n} encrypted by ransomware")

    {"Node failures": node_failures, "Link cuts": link_cuts, "Malware outbreak": malware,
     "DDoS attack": ddos, "Ransomware": ransom}.get(scenario, lambda: None)()
    if scenario == "Combined disaster":
        for fn in (link_cuts, node_failures, malware):
            cand[:] = [n for n in net.nodes if n != net.core and net.nodes[n]["state"] == "healthy"]
            live_links[:] = [k for k, l in net.links.items() if l["up"] and l["active"]]
            fn()
    return log


class NetworkHealer:
    """Autonomous network repair: detect -> contain -> fail over -> restore -> repair links -> verify -> harden."""
    BASE = dict(cpu=(30, 6), traffic=(120, 25), err=(0.5, 0.3))

    def __init__(self, seed=42):
        self.rng = random.Random(seed)
        self.actions = []

    def _log(self, phase, target, action, result, secs=0):
        self.actions.append(dict(phase=phase, target=target, action=action, result=result, seconds=secs))

    def _telemetry(self, state):
        r = self.rng.gauss
        if state == "degraded":
            return r(92, 4), r(900, 80), r(8, 1.5)
        if state in ("compromised", "quarantined"):
            return r(78, 8), r(520, 60), r(0.6, 0.3)
        if state == "failed":
            return 0.0, 0.0, 0.0
        return r(30, 6), r(120, 25), r(0.5, 0.3)

    # 1. DETECT -------------------------------------------------------------
    def detect(self, net, detected=None):
        """detected=None -> telemetry sees everything (manual sim). A set -> only hosts the IDS flagged are acted on."""
        mg, issues = net.mgmt_reach(), {}
        for n, d in net.nodes.items():
            if n == net.core:
                continue
            kinds = []
            cpu, trf, err = self._telemetry(d["state"])
            z = max((cpu - self.BASE["cpu"][0]) / self.BASE["cpu"][1],
                    (trf - self.BASE["traffic"][0]) / self.BASE["traffic"][1],
                    (err - self.BASE["err"][0]) / self.BASE["err"][1])
            if d["state"] == "failed":
                kinds.append("node failure")
            else:
                if n not in mg:
                    kinds.append("isolated (link fault)")
                if z > 4:
                    kinds.append("DDoS saturation" if err > 3 else "compromise")
            if kinds and detected is not None and n not in detected:
                self._log("1 Detect", n, "IDS alert check", "no IDS alert: fault NOT detected, left untouched", 1)
                continue
            if kinds:
                issues[n] = dict(kinds=kinds, z=z)
                self._log("1 Detect", n, "heartbeat + telemetry anomaly check",
                          ", ".join(kinds) + (f" (anomaly z={z:.1f})" if z > 4 else ""), 2)
        for k, l in net.links.items():
            if not l["up"]:
                self._log("1 Detect", f"{k[0]}–{k[1]}", "link-state probe", "link down", 1)
        if not issues and all(l["up"] for l in net.links.values()):
            self._log("1 Detect", "network", "full sweep", "no faults found", 2)
        return issues

    # 2. CONTAIN ------------------------------------------------------------
    def contain(self, net, issues):
        for n, i in issues.items():
            if "compromise" in i["kinds"]:
                net.nodes[n]["state"] = "quarantined"
                if net.nodes[n]["role"] == "attacker":
                    self._log("2 Contain", n, "block source IP at perimeter firewall", "attacker blocked", 3)
                else:
                    self._log("2 Contain", n, "quarantine: block all ports, revoke credentials", "spread stopped", 3)

    # 3. FAIL OVER ----------------------------------------------------------
    def reroute(self, net, phase="3 Fail over"):
        changed = True
        while changed:
            changed = False
            reach = net.reachable()
            for (u, v), l in net.links.items():
                if l["up"] and not l["active"] and net.nodes[u]["state"] in USABLE and net.nodes[v]["state"] in USABLE \
                        and ((u in reach) != (v in reach)):
                    l["active"], l["failover"] = True, True
                    lost = v if u in reach else u
                    self._log(phase, f"{u}–{v}", f"activate standby link to re-attach {lost}", "traffic rerouted", 5)
                    changed = True
                    break

    # 4. RESTORE ------------------------------------------------------------
    def restore(self, net, issues):
        order = sorted(issues, key=lambda n: ROLE_ORDER.get(net.nodes[n]["role"], 5))
        for n in order:
            d = net.nodes[n]
            if d["role"] == "attacker":
                continue                              # attackers stay blocked
            if d["state"] == "failed":
                d["state"] = "restored"
                self._log("4 Restore", n, "hard reset / fail over to hot spare", "back online", 30)
            elif d["state"] in ("quarantined", "compromised"):
                if d["backup"]:
                    self._log("4 Restore", n, "wipe, restore last clean backup, apply patches", "clean and rejoined", 90)
                else:
                    self._log("4 Restore", n, "no backup: rebuild from golden image, apply patches", "rebuilt and rejoined", 240)
                d["state"], d["data_lost"] = "restored", False
            elif d["state"] == "degraded":
                d["state"] = "healthy"
                self._log("4 Restore", n, "rate-limit, traffic scrubbing, SYN cookies", "load normal", 20)

    # 5. REPAIR LINKS -------------------------------------------------------
    def repair_links(self, net):
        for k, l in net.links.items():
            if not l["up"]:
                l["up"] = True
                self._log("5 Repair", f"{k[0]}–{k[1]}", "reset interface / switch to spare port", "link up", 60)

    # 7. HARDEN -------------------------------------------------------------
    def weak_points(self, net):
        base = len(net.reachable())
        out = []
        for n in net.nodes:
            if n == net.core or net.nodes[n]["role"] == "attacker":
                continue
            t = net.copy()
            t.nodes[n]["state"] = "failed"
            lost = [m for m in net.reachable() if m != n and m not in t.reachable()]
            if lost:
                out.append((len(lost), f"Node {n}: single point of failure, {len(lost)} node(s) lose service {sorted(lost)[:5]}. Add a redundant uplink."))
        for k, l in net.links.items():
            if l["up"] and l["active"]:
                t = net.copy()
                t.links[k]["up"] = False
                lost = [m for m in net.reachable() if m not in t.reachable()]
                if len(lost) >= 2:
                    out.append((len(lost), f"Link {k[0]}–{k[1]}: single point of failure, {len(lost)} node(s) cut off. Add a parallel link."))
        return [s for _, s in sorted(out, reverse=True)][:6]

    # full cycle -------------------------------------------------------------
    def heal(self, net, detected=None):
        self.actions = []
        h0 = net.health()
        issues = self.detect(net, detected)
        self.contain(net, issues)
        self.reroute(net)
        self.restore(net, issues)
        self.repair_links(net)
        self.reroute(net, "5 Repair")
        h1 = net.health()
        reach = net.reachable()
        resid = [n for n, d in net.nodes.items() if n != net.core and not (d["role"] == "attacker" and d["state"] == "quarantined")
                 and (d["state"] not in ("healthy", "restored") or n not in reach)]
        self._log("6 Verify", "network", "end-to-end reachability + health check",
                  f"health {h0:.0%} -> {h1:.0%}" + (f", residual: {resid}" if resid else ", all nodes healthy"), 5)
        weak = self.weak_points(net)
        for w in weak[:3]:
            self._log("7 Harden", "topology", "redundancy analysis", w, 0)
        return dict(before=h0, after=h1, actions=list(self.actions), issues=issues, residual=resid,
                    weak=weak, seconds=sum(a["seconds"] for a in self.actions))


def attack_effect(label):
    """What an attack does to its victim host. None = recon only (no damage)."""
    l = str(label).lower()
    if any(k in l for k in ("portscan", "port scan", "scan", "probe", "recon")):
        return None
    if any(k in l for k in ("dos", "ddos", "slowloris", "slowhttp", "hulk", "goldeneye", "flood")):
        return "degraded"
    return "compromised"


def build_ids_network(ipdf, n_hosts=12, n_att=4):
    """Builds the network map from real IDS results. ipdf columns: _src,_dst,label,binary_label,flag_before,flag_after.
    Returns None when there is no attack traffic to show."""
    att = ipdf[ipdf["binary_label"] == 1]
    if att.empty:
        return None
    attackers = list(att["_src"].value_counts().index[:n_att])
    hosts = list(att[~att["_dst"].isin(attackers)]["_dst"].value_counts().index[:n_hosts])
    for h in ipdf[ipdf["binary_label"] == 0]["_dst"].value_counts().index:      # pad with normal hosts
        if len(hosts) >= n_hosts:
            break
        if h not in hosts and h not in attackers:
            hosts.append(h)
    net = CyberNetwork("INTERNET")
    net.add_node("INTERNET", "cloud", 5.0, -1.3)
    for i, a in enumerate(attackers):
        net.add_node(a, "attacker", 5.0 if len(attackers) == 1 else 0.5 + 9.0 * i / (len(attackers) - 1), 0.0)
        net.add_link(a, "INTERNET")
    for name, role, x, y in [("FW1", "firewall", 3.0, -2.6), ("FW2", "firewall", 7.0, -2.6),
                             ("SW1", "switch", 3.0, -3.9), ("SW2", "switch", 7.0, -3.9)]:
        net.add_node(name, role, x, y)
    net.add_link("INTERNET", "FW1")
    net.add_link("INTERNET", "FW2")
    for sw in ("SW1", "SW2"):
        net.add_link("FW1", sw)
        net.add_link("FW2", sw, standby=True)
    net.add_link("SW1", "SW2", standby=True)
    for i, h in enumerate(hosts):
        net.add_node(h, "host", 0.5 if len(hosts) == 1 else 0.5 + 9.0 * i / (len(hosts) - 1), -5.4 - 0.5 * (i % 2))
        net.add_link(h, "SW1" if i % 2 == 0 else "SW2")
        net.add_link(h, "SW2" if i % 2 == 0 else "SW1", standby=True)
    dmg = net.copy()
    for a in attackers:
        dmg.nodes[a]["state"] = "compromised"
    for dst, lab in att[["_dst", "label"]].drop_duplicates().itertuples(index=False):
        eff = attack_effect(lab)
        if eff and dst in dmg.nodes and dmg.nodes[dst]["role"] != "attacker":
            cur = dmg.nodes[dst]["state"]
            dmg.nodes[dst]["state"] = "compromised" if (eff == "compromised" or cur == "compromised") else "degraded"
    hit = att[att["flag_after"] == 1]
    hit_b = att[att["flag_before"] == 1]
    det, det_b = set(hit["_dst"]) | set(hit["_src"]), set(hit_b["_dst"]) | set(hit_b["_src"])
    affected = {n for n, d in dmg.nodes.items() if d["state"] != "healthy"}
    return dict(base=net, damaged=dmg, detected=det, affected=affected,
                caught_before=len(affected & det_b), caught_after=len(affected & det),
                flows=len(ipdf), attack_flows=len(att), labels=att["label"].value_counts().head(5).to_dict())


# ---------- CIC-IDS2017 auto-loading ----------
def find_cic_files(extra=None):
    """Looks for CIC-IDS2017 day files. Returns (files_with_IPs, files_without_IPs). Only the TrafficLabelling
    version has the Source/Destination IP columns."""
    here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    home = os.path.expanduser("~")
    roots = ([extra] if extra else []) + [here, os.path.join(here, "CIC-IDS2017"), os.getcwd(),
             os.path.join(home, "Downloads"), os.path.join(home, "Documents"), os.path.join(home, "Desktop")]
    seen, hits = set(), []
    for r in roots:
        if not r or not os.path.isdir(r) or os.path.abspath(r) in seen:
            continue
        seen.add(os.path.abspath(r))
        depth0 = os.path.abspath(r).count(os.sep)
        for dp, dn, fn in os.walk(r):
            dn[:] = [d for d in dn if not d.startswith(".") and d.lower() not in ("node_modules", "appdata", "windows")]
            if os.path.abspath(dp).count(os.sep) - depth0 >= 4:
                dn[:] = []
            hits += [os.path.join(dp, f) for f in fn if f.lower().endswith(".csv") and "iscx" in f.lower()]
    with_ip, no_ip = [], []
    for f in sorted(set(hits)):
        try:
            cols = [str(c).strip().lower() for c in _read_csv(f, nrows=0).columns]
            (with_ip if ("source ip" in cols and "destination ip" in cols) else no_ip).append(f)
        except Exception:
            pass
    return with_ip, no_ip


def load_cic(files, cap=150000):
    """Loads and samples the CIC-IDS2017 day files into one frame (cap = total rows kept)."""
    per = max(5000, cap // max(1, len(files)))
    parts = []
    for f in files:
        d, _ = read_table(f, [])
        if len(d) > per:
            d = d.sample(n=per, random_state=SEED)
        print(f"[CIC-IDS2017] {os.path.basename(f)}: {len(d):,} rows kept")
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def network_report(info, rep, base, dmg, healed, explainer_name, loop_notes=None, mode="ids"):
    """Plain-language report of what happened and what the self-healer did (shown in the GenAI explainer box)."""
    victims = [n for n, d in dmg.nodes.items() if d["state"] != "healthy" and d["role"] != "attacker"]
    attackers = [n for n, d in dmg.nodes.items() if d["role"] == "attacker" and d["state"] != "healthy"]
    acts = rep["actions"]
    cnt = lambda key: sum(1 for a in acts if key in a["result"] or key in a["action"])
    kinds = {}
    for n in victims:
        kinds.setdefault(dmg.nodes[n]["state"], []).append(n)
    L = [f"INCIDENT & SELF-HEALING REPORT   (explainer: {explainer_name})", ""]
    L.append("1. WHAT HAPPENED")
    if mode == "ids" and info:
        lab = ", ".join(f"{k} ({v:,})" for k, v in info["labels"].items())
        L.append(f"   {info['flows']:,} live flows were analysed; {info['attack_flows']:,} were attacks: {lab}.")
    if attackers:
        L.append(f"   Attacking sources: {', '.join(attackers)}.")
    for st, ns in kinds.items():
        verb = {"compromised": "were compromised", "degraded": "were degraded (overloaded)", "failed": "went down"}.get(st, st)
        L.append(f"   {len(ns)} host(s) {verb}: {', '.join(ns[:8])}{'...' if len(ns) > 8 else ''}.")
    cut = [k for k, l in dmg.links.items() if not l["up"]]
    if cut:
        L.append(f"   {len(cut)} link(s) were cut: {', '.join(a + '–' + b for a, b in cut[:5])}.")
    L.append(f"   Network health fell from {base.health():.0%} to {dmg.health():.0%}.")
    L += ["", "2. HOW THE IDS RESPONDED"]
    if mode == "ids" and info:
        n = len(info["affected"])
        L.append(f"   Before self-healing the IDS flagged {info['caught_before']}/{n} affected hosts; after the self-healing "
                 f"loop retrained the model it flagged {info['caught_after']}/{n}.")
        if loop_notes:
            L.append("   Model patch note: " + str(loop_notes[0])[:300])
    else:
        L.append("   Faults were found by heartbeat checks and telemetry anomaly scoring.")
    L += ["", "3. WHAT THE SELF-HEALER DID"]
    steps = [("attacker blocked", "blocked {} attacker IP(s) at the perimeter firewall"),
             ("spread stopped", "quarantined {} compromised host(s) so the infection could not spread"),
             ("back online", "reset {} failed device(s) / switched to hot spare"),
             ("clean and rejoined", "wiped and restored {} host(s) from clean backup, then patched them"),
             ("rebuilt and rejoined", "rebuilt {} host(s) from the golden image (no backup existed)"),
             ("load normal", "applied rate-limiting and traffic scrubbing on {} overloaded host(s)"),
             ("traffic rerouted", "activated {} standby link(s) to reroute traffic"),
             ("link up", "repaired {} link(s)")]
    for key, txt in steps:
        c = sum(1 for a in acts if a["result"] == key)
        if c:
            L.append("   - " + txt.format(c))
    L.append(f"   Estimated repair time: {rep['seconds']} s.")
    L += ["", "4. RESULT", f"   Network health {rep['before']:.0%} -> {rep['after']:.0%}."]
    if rep["residual"]:
        L.append(f"   Still damaged: {', '.join(rep['residual'][:8])}. These were not flagged by the IDS, so no action was taken on them.")
    else:
        L.append("   All hosts are healthy again; attackers remain blocked.")
    L += ["", "5. RECOMMENDATIONS"]
    rec = []
    if attackers:
        rec.append("Keep the blocked attacker IPs on the firewall blocklist and review how they got in.")
    if "compromised" in kinds:
        rec.append("Rotate credentials and verify patch level on the restored hosts.")
    if "degraded" in kinds:
        rec.append("Enable upstream DDoS protection and rate limits on the affected services.")
    if rep["residual"]:
        rec.append("Add the missed hosts' flows to the training data and run another self-healing cycle.")
    rec += rep["weak"][:2]
    L += [f"   - {r}" for r in (rec or ["No further action needed."])]
    return "\n".join(L)


def draw_network(ax, net, title):
    ax.clear()
    ax.set_title(title, fontsize=10, color=PRIMARY)
    ax.axis("off")
    if net is None:
        ax.text(0.5, 0.5, "–", ha="center", va="center", transform=ax.transAxes, color=MUTED)
        return
    for (u, v), l in net.links.items():
        x, y = [net.nodes[u]["x"], net.nodes[v]["x"]], [net.nodes[u]["y"], net.nodes[v]["y"]]
        if not l["up"]:
            kw = dict(color="#d9534f", ls="--", lw=1.8)
        elif not l["active"]:
            kw = dict(color="#cfd6de", ls=":", lw=1.0)
        elif l["failover"]:
            kw = dict(color="#2e86de", lw=2.2)
        else:
            kw = dict(color="#9aa5b1", lw=1.2)
        ax.plot(x, y, zorder=1, **kw)
    for n, d in net.nodes.items():
        ax.scatter(d["x"], d["y"], s=150, color=STATE_COLOR[d["state"]], edgecolor="white", zorder=2)
        ax.text(d["x"], d["y"] - 0.33, n, ha="center", va="top", fontsize=6.5)
    ax.margins(0.08)


class App(tk.Tk):
    def __init__(self, core, core_path):
        super().__init__()
        self.core, self.core_path = core, core_path
        _defaults = {
            "SEED": 42,
            "CATEGORICAL": ["protocol_type", "service", "flag"],
            "NSL_KDD_COLUMNS": [
                "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
                "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
                "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
                "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
                "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
                "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
                "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
                "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
                "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty"],
        }
        for k, v in _defaults.items():      # only fills in what the base file does not define
            if not hasattr(core, k):
                setattr(core, k, v)
        self.orig_cols, self.orig_cat = list(core.NSL_KDD_COLUMNS), list(core.CATEGORICAL)
        self.title("Self-Healing Cyber Network Defense")
        self.geometry("1320x800")
        self.minsize(1100, 700)
        self.configure(bg=BG)
        self.q = queue.Queue()
        self.stop_flag = False
        self.worker = None
        self.results = None
        self.raw_train = None
        self.raw_live = None
        self._style()
        self._header()
        self._body()
        self.after(100, self._poll)

    # ---------- styling ----------
    def _style(self):
        st = ttk.Style(self)
        st.theme_use("clam")
        st.configure(".", background=BG, font=("Segoe UI", 10))
        st.configure("TLabelframe", background=BG, bordercolor="#c9d1db")
        st.configure("TLabelframe.Label", background=BG, foreground=PRIMARY, font=("Segoe UI", 10, "bold"))
        st.configure("TNotebook", background=BG)
        st.configure("TNotebook.Tab", padding=(14, 7), font=("Segoe UI", 10, "bold"))
        st.configure("Accent.TButton", background=ACCENT, foreground="white", font=("Segoe UI", 11, "bold"), padding=8)
        st.map("Accent.TButton", background=[("active", "#1b6fc4"), ("disabled", "#9db9d6")])
        st.configure("Danger.TButton", background=BAD, foreground="white", padding=6)
        st.map("Danger.TButton", background=[("active", "#b83e3a"), ("disabled", "#e0b3b1")])
        st.configure("Treeview", rowheight=24)
        st.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"))
        st.configure("Card.TFrame", background="white", relief="solid", borderwidth=1)
        st.configure("Horizontal.TProgressbar", troughcolor="#dde3ea", background=ACCENT)

    def _header(self):
        h = tk.Frame(self, bg=PRIMARY, height=64)
        h.pack(fill="x")
        tk.Label(h, text="🛡  Self-Healing Cyber Network Defense", bg=PRIMARY, fg="white",
                 font=("Segoe UI", 18, "bold")).pack(side="left", padx=18, pady=10)
        tk.Label(h, text="Detect → Explain → Log → Retrain → Redeploy      |      IDS + FGSM robustness add-on",
                 bg=PRIMARY, fg="#9fb6d3", font=("Segoe UI", 10)).pack(side="left", padx=6)
        self.core_lbl = tk.Label(h, text=f"core: {os.path.basename(self.core_path)}", bg=PRIMARY, fg="#6f87a8",
                                 font=("Segoe UI", 9))
        self.core_lbl.pack(side="right", padx=18)

    # ---------- layout ----------
    def _body(self):
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=10, pady=10)
        left = tk.Frame(body, bg=BG, width=350)
        left.pack(side="left", fill="y", padx=(0, 10))
        left.pack_propagate(False)
        self._left(left)
        right = tk.Frame(body, bg=BG)
        right.pack(side="left", fill="both", expand=True)
        self.nb = ttk.Notebook(right)
        self.nb.pack(fill="both", expand=True)
        self._tab_overview()
        self._tab_healing()
        self._tab_report()
        self._tab_scan()
        self._tab_network()
        self._tab_console()

    def _left(self, p):
        # --- dataset ---
        ds = ttk.LabelFrame(p, text=" 1 · Dataset ", padding=8)
        ds.pack(fill="x")
        self.src = tk.StringVar(value="nsl")
        ttk.Radiobutton(ds, text="NSL-KDD (auto-download, includes unseen attacks)", variable=self.src,
                        value="nsl").pack(anchor="w")
        ttk.Radiobutton(ds, text="CIC-IDS2017 (auto-load, linked to Network Healing)", variable=self.src,
                        value="cic").pack(anchor="w", pady=(2, 0))
        self.cic_lbl = ttk.Label(ds, text="   Put the TrafficLabelling CSVs in Downloads or next to this script; "
                                          "otherwise you'll be asked for the folder.", foreground=MUTED, wraplength=320)
        self.cic_lbl.pack(anchor="w")
        ttk.Radiobutton(ds, text="My own dataset (CSV / NSL-KDD txt)", variable=self.src,
                        value="custom").pack(anchor="w", pady=(2, 4))
        self.train_path, self.live_path = tk.StringVar(), tk.StringVar()
        self._file_row(ds, "Training file", self.train_path, self._browse_train)
        self._file_row(ds, "Live traffic file (optional)", self.live_path, self._browse_live, clearable=True)
        r = ttk.Frame(ds)
        r.pack(fill="x", pady=(4, 0))
        ttk.Label(r, text="Label column").grid(row=0, column=0, sticky="w")
        self.label_col = tk.StringVar()
        self.label_cb = ttk.Combobox(r, textvariable=self.label_col, state="readonly", width=22)
        self.label_cb.grid(row=0, column=1, padx=6, pady=2)
        self.label_cb.bind("<<ComboboxSelected>>", lambda e: self._refresh_normals())
        ttk.Label(r, text="'Normal' value(s)").grid(row=1, column=0, sticky="w")
        self.normals = tk.StringVar(value="normal")
        ttk.Entry(r, textvariable=self.normals, width=25).grid(row=1, column=1, padx=6, pady=2)
        self.ds_info = ttk.Label(ds, text="No custom file loaded.", foreground=MUTED, wraplength=320)
        self.ds_info.pack(anchor="w", pady=(4, 0))

        # --- parameters ---
        pr = ttk.LabelFrame(p, text=" 2 · Parameters ", padding=8)
        pr.pack(fill="x", pady=8)
        self.v = {}
        spec = [("Training rows", "train_sample", 20000, 1000, 125000, 1000),
                ("Training epochs", "epochs", 25, 1, 200, 1),
                ("FGSM epsilon", "epsilon", 0.25, 0.01, 2.0, 0.05),
                ("Confidence threshold", "conf", 0.85, 0.5, 0.99, 0.01),
                ("Self-healing cycles", "cycles", 4, 1, 20, 1),
                ("Live batch size", "batch", 1500, 50, 20000, 50)]
        for i, (lab, key, d, lo, hi, inc) in enumerate(spec):
            ttk.Label(pr, text=lab).grid(row=i, column=0, sticky="w", pady=2)
            var = tk.DoubleVar(value=d) if isinstance(d, float) else tk.IntVar(value=d)
            self.v[key] = var
            ttk.Spinbox(pr, textvariable=var, from_=lo, to=hi, increment=inc, width=10).grid(row=i, column=1, padx=8)

        # --- LLM ---
        ll = ttk.LabelFrame(p, text=" 3 · GenAI explanations (optional) ", padding=8)
        ll.pack(fill="x")
        self.k_claude, self.k_gem = tk.StringVar(), tk.StringVar()
        ttk.Label(ll, text="Anthropic key").grid(row=0, column=0, sticky="w")
        ttk.Entry(ll, textvariable=self.k_claude, show="•", width=27).grid(row=0, column=1, padx=6, pady=2)
        ttk.Label(ll, text="Gemini key").grid(row=1, column=0, sticky="w")
        ttk.Entry(ll, textvariable=self.k_gem, show="•", width=27).grid(row=1, column=1, padx=6, pady=2)
        ttk.Label(ll, text="Leave empty to use the built-in local explainer.", foreground=MUTED).grid(
            row=2, column=0, columnspan=2, sticky="w")

        # --- run ---
        self.run_btn = ttk.Button(p, text="▶  Run self-healing pipeline", style="Accent.TButton", command=self._run)
        self.run_btn.pack(fill="x", pady=(12, 4))
        self.stop_btn = ttk.Button(p, text="■  Stop", style="Danger.TButton", command=self._stop, state="disabled")
        self.stop_btn.pack(fill="x")
        self.prog = ttk.Progressbar(p, maximum=7)
        self.prog.pack(fill="x", pady=(10, 2))
        self.status = ttk.Label(p, text="Ready.", foreground=MUTED, wraplength=330)
        self.status.pack(anchor="w")

    def _file_row(self, p, text, var, cmd, clearable=False):
        ttk.Label(p, text=text).pack(anchor="w")
        f = ttk.Frame(p)
        f.pack(fill="x", pady=(0, 2))
        ttk.Entry(f, textvariable=var).pack(side="left", fill="x", expand=True)
        ttk.Button(f, text="Browse…", width=8, command=cmd).pack(side="left", padx=(4, 0))
        if clearable:
            ttk.Button(f, text="✕", width=3, command=lambda: (var.set(""), setattr(self, "raw_live", None))).pack(
                side="left", padx=(2, 0))

    # ---------- results tabs ----------
    def _tab_overview(self):
        t = ttk.Frame(self.nb, padding=10)
        self.nb.add(t, text="  Overview  ")
        row = tk.Frame(t, bg=BG)
        row.pack(fill="x")
        self.kpi = {}
        for key, cap in [("clean", "Robust IDS accuracy (clean)"), ("fgsm", "Accuracy under FGSM  base → robust"),
                         ("live", "Unseen live traffic  before → after healing"), ("inc", "Incidents logged")]:
            c = ttk.Frame(row, style="Card.TFrame", padding=10)
            c.pack(side="left", fill="x", expand=True, padx=4)
            val = tk.Label(c, text="–", bg="white", fg=PRIMARY, font=("Segoe UI", 20, "bold"))
            val.pack(anchor="w")
            tk.Label(c, text=cap, bg="white", fg=MUTED, font=("Segoe UI", 9)).pack(anchor="w")
            self.kpi[key] = val
        self.fig1 = Figure(figsize=(6, 3.3), dpi=100)
        self.cv1 = FigureCanvasTkAgg(self.fig1, t)
        self.cv1.get_tk_widget().pack(fill="both", expand=True, pady=8)
        self.eval_tv = self._tree(t, ("model", "accuracy", "precision", "recall", "f1"), height=7)

    def _tab_healing(self):
        t = ttk.Frame(self.nb, padding=10)
        self.nb.add(t, text="  Self-Healing  ")
        top = tk.Frame(t, bg=BG)
        top.pack(fill="both", expand=True)
        self.fig2 = Figure(figsize=(5, 3.2), dpi=100)
        self.cv2 = FigureCanvasTkAgg(self.fig2, top)
        self.cv2.get_tk_widget().pack(side="left", fill="both", expand=True)
        right = tk.Frame(top, bg=BG)
        right.pack(side="left", fill="both", expand=True, padx=(10, 0))
        ttk.Label(right, text="Cycle history", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.hist_tv = self._tree(right, ("cycle", "flagged", "live_acc_before", "live_acc_after"), height=6)
        ttk.Label(right, text="GenAI patch notes", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(8, 0))
        self.notes = tk.Text(right, height=10, wrap="word", bg="white", relief="solid", bd=1, font=("Segoe UI", 10))
        self.notes.pack(fill="both", expand=True)
        self.held_lbl = ttk.Label(t, text="", font=("Segoe UI", 10, "bold"), foreground=PRIMARY)
        self.held_lbl.pack(anchor="w", pady=(8, 0))

    def _tab_report(self):
        t = ttk.Frame(self.nb, padding=10)
        self.nb.add(t, text="  Healing Report  ")
        bar = ttk.Frame(t)
        bar.pack(fill="x")
        ttk.Label(bar, text="Status filter:").pack(side="left")
        self.flt = tk.StringVar(value="All")
        self.flt_cb = ttk.Combobox(bar, textvariable=self.flt, state="readonly", values=["All"], width=34)
        self.flt_cb.pack(side="left", padx=6)
        self.flt_cb.bind("<<ComboboxSelected>>", lambda e: self._fill_report())
        ttk.Button(bar, text="Export CSV…", command=self._export_report).pack(side="right")
        cols = ("cycle", "sample_id", "attack_type", "truth", "before", "conf_before", "after", "conf_after", "status")
        self.rep_tv = self._tree(t, cols, height=14)
        self.rep_tv.bind("<<TreeviewSelect>>", self._show_expl)
        ttk.Label(t, text="Explanation of selected sample", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(8, 0))
        self.expl = tk.Text(t, height=5, wrap="word", bg="white", relief="solid", bd=1, font=("Segoe UI", 10))
        self.expl.pack(fill="x")

    def _tab_scan(self):
        t = ttk.Frame(self.nb, padding=10)
        self.nb.add(t, text="  Scan New Traffic  ")
        bar = ttk.Frame(t)
        bar.pack(fill="x")
        ttk.Button(bar, text="Choose a dataset & scan with the healed model…", command=self._scan).pack(side="left")
        ttk.Button(bar, text="Export predictions…", command=self._export_scan).pack(side="right")
        self.scan_lbl = ttk.Label(t, text="Run the pipeline first, then scan any new file with the same columns.",
                                  foreground=MUTED, wraplength=900)
        self.scan_lbl.pack(anchor="w", pady=8)
        self.scan_tv = self._tree(t, ("row", "prediction", "attack_probability", "confidence", "flagged_unsure", "truth"), height=18)
        self.scan_df = None

    # ---------- network healing tab ----------
    def _tab_network(self):
        t = ttk.Frame(self.nb, padding=10)
        self.nb.add(t, text="  Network Healing  ")
        self.net_base = self.net_dmg = self.net_heal = self.net_ids = self.net_report = None
        self.net_rng = random.Random()
        self.net_mode = tk.StringVar(value="ids")
        m = ttk.Frame(t)
        m.pack(fill="x")
        ttk.Radiobutton(m, text="From IDS results (dataset with IP addresses, e.g. CIC-IDS2017)", variable=self.net_mode,
                        value="ids", command=self._net_load_ids).pack(side="left")
        ttk.Radiobutton(m, text="Manual simulation", variable=self.net_mode, value="manual",
                        command=self._net_build).pack(side="left", padx=14)
        bar = ttk.Frame(t)
        bar.pack(fill="x", pady=(4, 0))
        ttk.Label(bar, text="Scenario:").pack(side="left")
        self.net_scn = tk.StringVar(value=DAMAGE_SCENARIOS[0])
        ttk.Combobox(bar, textvariable=self.net_scn, state="readonly", values=DAMAGE_SCENARIOS, width=18).pack(side="left", padx=6)
        ttk.Label(bar, text="Size:").pack(side="left")
        self.net_size = tk.IntVar(value=3)
        ttk.Spinbox(bar, from_=1, to=8, textvariable=self.net_size, width=3).pack(side="left", padx=6)
        ttk.Button(bar, text="Build / reset", command=self._net_build).pack(side="left", padx=2)
        ttk.Button(bar, text="Import topology…", command=self._net_import).pack(side="left", padx=2)
        ttk.Button(bar, text="Inflict damage", command=self._net_damage).pack(side="left", padx=2)
        ttk.Button(bar, text="Heal network", style="Accent.TButton", command=self._net_heal).pack(side="left", padx=6)
        ttk.Button(bar, text="Export actions CSV…", command=self._net_export).pack(side="right")
        ttk.Button(bar, text="Export GenAI report…", command=self._net_export_report).pack(side="right", padx=4)
        row = tk.Frame(t, bg=BG)
        row.pack(fill="x", pady=6)
        self.net_kpi = {}
        for key, cap in [("h0", "Health: intact network"), ("h1", "Health: after damage"),
                         ("h2", "Health: after healing"), ("sec", "Estimated repair time")]:
            c = ttk.Frame(row, style="Card.TFrame", padding=8)
            c.pack(side="left", fill="x", expand=True, padx=4)
            v = tk.Label(c, text="–", bg="white", fg=PRIMARY, font=("Segoe UI", 18, "bold"))
            v.pack(anchor="w")
            tk.Label(c, text=cap, bg="white", fg=MUTED, font=("Segoe UI", 9)).pack(anchor="w")
            self.net_kpi[key] = v
        self.fig3 = Figure(figsize=(12, 3.6), dpi=100)
        self.cv3 = FigureCanvasTkAgg(self.fig3, t)
        self.cv3.get_tk_widget().pack(fill="both", expand=True)
        bottom = tk.Frame(t, bg=BG)
        bottom.pack(fill="both", expand=True, pady=(6, 0))
        lf = ttk.LabelFrame(bottom, text=" Healing actions ", padding=4)
        lf.pack(side="left", fill="both", expand=True)
        self.net_tv = self._tree(lf, ("phase", "target", "action", "result", "seconds"), height=6)
        for c, w in (("phase", 70), ("target", 100), ("action", 190), ("result", 150), ("seconds", 50)):
            self.net_tv.column(c, width=w)
        rf = ttk.LabelFrame(bottom, text=" GenAI explainer report ", padding=4)
        rf.pack(side="left", fill="both", expand=True, padx=(8, 0))
        rsb = ttk.Scrollbar(rf)
        rsb.pack(side="right", fill="y")
        self.net_rep = tk.Text(rf, height=10, wrap="word", bg="white", relief="solid", bd=1, font=("Segoe UI", 9),
                               yscrollcommand=rsb.set)
        self.net_rep.pack(fill="both", expand=True)
        rsb.config(command=self.net_rep.yview)
        self.net_note = tk.Text(t, height=2, wrap="word", bg="white", relief="solid", bd=1, font=("Segoe UI", 10))
        self.net_note.pack(fill="x", pady=(6, 0))
        self.net_tab = t
        self._net_load_ids()

    def _net_say(self, text):
        self.net_note.delete("1.0", "end")
        self.net_note.insert("end", text)

    def _net_set_report(self, text):
        self.net_rep.delete("1.0", "end")
        self.net_rep.insert("end", text)
        self.net_text = text

    def _net_clear(self):
        self.net_base = self.net_dmg = self.net_heal = self.net_ids = self.net_report = None
        self.net_text = ""
        self.net_rep.delete("1.0", "end")
        for k in self.net_kpi:
            self.net_kpi[k].config(text="–")
        self.net_tv.delete(*self.net_tv.get_children())

    def _net_draw(self):
        self.fig3.clear()
        axes = self.fig3.subplots(1, 3)
        if self.net_base is None:
            for a in axes:
                a.axis("off")
            axes[1].text(0.5, 0.5, "Nothing to show", ha="center", va="center", fontsize=16, color=MUTED, transform=axes[1].transAxes)
            self.cv3.draw()
            return
        draw_network(axes[0], self.net_base, "1 · Intact network")
        draw_network(axes[1], self.net_dmg, "2 · After damage")
        draw_network(axes[2], self.net_heal, "3 · After self-healing")
        hs = [Line2D([], [], marker="o", ls="", color=c, label=k) for k, c in STATE_COLOR.items()]
        hs += [Line2D([], [], color="#d9534f", ls="--", label="cut link"), Line2D([], [], color="#2e86de", lw=2, label="failover link"),
               Line2D([], [], color="#cfd6de", ls=":", label="standby (dark)")]
        self.fig3.legend(handles=hs, loc="lower center", ncol=9, fontsize=7, frameon=False)
        self.fig3.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.1, wspace=0.05)
        self.cv3.draw()

    def _net_load_ids(self):
        """Shows the network built from the last pipeline run. Datasets without IPs (NSL-KDD) -> 'Nothing to show'."""
        self.net_mode.set("ids")
        self._net_clear()
        r = self.results
        if r is None:
            self._net_say("Nothing to show yet. Run the self-healing pipeline on a dataset that has source/destination IP "
                          "addresses (CIC-IDS2017 TrafficLabelling). NSL-KDD has no IP data, so it is not linked to the network map.")
        elif r.get("ipmap") is None:
            self._net_say("Nothing to show: this dataset has no source/destination IP columns, so its attacks cannot be placed "
                          "on a network (true for NSL-KDD and the NB15 training CSV). Use CIC-IDS2017 TrafficLabelling, "
                          "or switch to 'Manual simulation'.")
        else:
            info = build_ids_network(r["ipmap"])
            if info is None:
                self._net_say("Nothing to show: no attack traffic in the live stream.")
            else:
                self.net_ids, self.net_base, self.net_dmg = info, info["base"], info["damaged"]
                self.net_kpi["h0"].config(text=f"{self.net_base.health():.0%}")
                self.net_kpi["h1"].config(text=f"{self.net_dmg.health():.0%}")
                lab = ", ".join(f"{k} ({v})" for k, v in info["labels"].items())
                self._net_say(f"Built from {info['flows']:,} live flows ({info['attack_flows']:,} attacks: {lab}). "
                              f"{len(info['affected'])} host(s) affected. Press 'Heal network'; hosts the IDS missed stay damaged.")
        self._net_draw()

    def _net_build(self, net=None):
        self.net_mode.set("manual")
        self._net_clear()
        self.net_base = net if isinstance(net, CyberNetwork) else default_network()
        self.net_kpi["h0"].config(text=f"{self.net_base.health():.0%}")
        self._net_say("Network built. Pick a scenario, press 'Inflict damage', then 'Heal network'.")
        self._net_draw()

    def _net_import(self):
        p = filedialog.askopenfilename(title="Topology edge list (CSV: node A, node B[, standby])",
                                       filetypes=[("CSV", "*.csv *.txt"), ("All", "*.*")])
        if not p:
            return
        try:
            net = network_from_edges(pd.read_csv(p))
            self._net_build(net)
            self._net_say(f"Imported {len(net.nodes)} nodes / {len(net.links)} links. Core = {net.core}.")
        except Exception as e:
            messagebox.showerror("Cannot read topology", str(e))

    def _net_damage(self):
        if self.net_mode.get() == "ids" or self.net_base is None:
            self._net_build()
        self.net_dmg = self.net_base.copy()
        what = inflict_damage(self.net_dmg, self.net_scn.get(), int(self.net_size.get()), self.net_rng)
        self.net_heal = self.net_report = None
        self.net_tv.delete(*self.net_tv.get_children())
        self.net_kpi["h1"].config(text=f"{self.net_dmg.health():.0%}")
        self.net_kpi["h2"].config(text="–")
        self.net_kpi["sec"].config(text="–")
        self._net_say("Damage: " + ("; ".join(what) or "none") + ".")
        self._net_draw()

    def _net_heal(self, auto=False):
        if self.net_dmg is None:
            if auto:
                return
            messagebox.showinfo("Network Healing", "Nothing to heal yet." if self.net_mode.get() == "ids" else "Inflict damage first.")
            return
        ids_mode = self.net_mode.get() == "ids" and self.net_ids is not None
        self.net_heal = self.net_dmg.copy()
        rep = NetworkHealer().heal(self.net_heal, self.net_ids["detected"] if ids_mode else None)
        self.net_report = rep
        self.net_tv.delete(*self.net_tv.get_children())
        for a in rep["actions"]:
            self.net_tv.insert("", "end", values=(a["phase"], a["target"], a["action"], a["result"], a["seconds"]))
        self.net_kpi["h2"].config(text=f"{rep['after']:.0%}")
        self.net_kpi["sec"].config(text=f"{rep['seconds']} s")
        msg = f"Healed {rep['before']:.0%} → {rep['after']:.0%}. " + \
              ("All nodes restored." if not rep["residual"] else f"Still damaged: {rep['residual']}.")
        if ids_mode:
            i = self.net_ids
            msg += f"\nIDS caught {i['caught_before']}/{len(i['affected'])} affected hosts before self-healing, {i['caught_after']}/{len(i['affected'])} after."
        elif rep["weak"]:
            msg += "\nHardening advice: " + " | ".join(rep["weak"][:3])
        self._net_say(msg)
        self._net_draw()
        # ---- GenAI explainer report ----
        os.environ["ANTHROPIC_API_KEY"] = self.k_claude.get().strip()
        os.environ["GEMINI_API_KEY"] = self.k_gem.get().strip()
        ex = self.core.GenAIExplainer()
        notes = (self.results or {}).get("notes") if ids_mode else None
        text = network_report(self.net_ids if ids_mode else None, rep, self.net_base, self.net_dmg, self.net_heal,
                              ex.llmModel, notes, "ids" if ids_mode else "manual")
        self._net_set_report(text)
        if ex.llmModel in ("claude", "gemini"):
            self._net_rep_status = "asking " + ex.llmModel
            def _ask(facts=text):
                out = ex._call_llm("You are a SOC analyst assistant. Write a concise incident report (max 170 words, plain "
                                   "prose) for a security team from these facts only, inventing nothing:\n" + facts)
                if out:
                    self.q.put(("netreport", f"AI SUMMARY ({ex.llmModel})\n{out}\n\n" + "-" * 40 + "\n" + facts))
            threading.Thread(target=_ask, daemon=True).start()

    def _net_export_report(self):
        if not getattr(self, "net_text", ""):
            messagebox.showinfo("Network Healing", "No report yet.")
            return
        p = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text", "*.txt")])
        if p:
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(self.net_text)
            messagebox.showinfo("Saved", p)

    def _net_export(self):
        if not self.net_report:
            messagebox.showinfo("Network Healing", "Heal a network first.")
            return
        p = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if p:
            pd.DataFrame(self.net_report["actions"]).to_csv(p, index=False)
            messagebox.showinfo("Saved", p)

    def _tab_console(self):
        t = ttk.Frame(self.nb, padding=10)
        self.nb.add(t, text="  Console  ")
        self.console = tk.Text(t, bg="#0f1722", fg="#b7f0c0", insertbackground="white", font=("Consolas", 10), wrap="word")
        sb = ttk.Scrollbar(t, command=self.console.yview)
        self.console.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.console.pack(fill="both", expand=True)

    def _tree(self, parent, cols, height=8):
        f = ttk.Frame(parent)
        f.pack(fill="both", expand=(height > 8))
        tv = ttk.Treeview(f, columns=cols, show="headings", height=height)
        for c in cols:
            tv.heading(c, text=c.replace("_", " ").title())
            tv.column(c, width=90, anchor="center")
        sb = ttk.Scrollbar(f, command=tv.yview)
        tv.configure(yscrollcommand=sb.set)
        tv.pack(side="left", fill="both", expand=True)
        sb.pack(side="left", fill="y")
        return tv

    @staticmethod
    def _fill_tv(tv, df):
        tv.delete(*tv.get_children())
        for _, r in df.iterrows():
            tv.insert("", "end", values=[("%.4f" % x if isinstance(x, float) else x) for x in r.tolist()])

    # ---------- dataset browsing ----------
    def _browse_train(self):
        ps = filedialog.askopenfilenames(title="Training dataset (pick several CIC-IDS2017 day files to combine)",
                                         filetypes=[("Data", "*.csv *.txt *.data"), ("All", "*.*")])
        if not ps:
            return
        p = ps[0]
        try:
            parts = [read_table(q, self.orig_cols) for q in ps]
            df, nsl = pd.concat([x[0] for x in parts], ignore_index=True), parts[0][1]
        except Exception as e:
            messagebox.showerror("Cannot read file", str(e))
            return
        self.train_path.set(p + (f"  (+{len(ps) - 1} more)" if len(ps) > 1 else ""))
        self.raw_train = df
        self.src.set("custom")
        cols = list(df.columns)
        self.label_cb["values"] = cols
        self.label_col.set("label" if nsl else guess_label_col(cols))
        self._refresh_normals()
        self.ds_info.config(text=f"{len(df):,} rows × {len(cols)} columns" + ("  (NSL-KDD format detected)" if nsl else ""))

    def _browse_live(self):
        p = filedialog.askopenfilename(title="Live traffic dataset", filetypes=[("Data", "*.csv *.txt *.data"), ("All", "*.*")])
        if not p:
            return
        try:
            self.raw_live, _ = read_table(p, self.orig_cols)
        except Exception as e:
            messagebox.showerror("Cannot read file", str(e))
            return
        self.live_path.set(p)

    def _refresh_normals(self):
        if self.raw_train is not None and self.label_col.get() in self.raw_train.columns:
            g = guess_normals(self.raw_train[self.label_col.get()])
            self.normals.set(g or self.normals.get())
            vals = self.raw_train[self.label_col.get()].astype(str).value_counts().head(6)
            self.ds_info.config(text=f"{len(self.raw_train):,} rows × {self.raw_train.shape[1]} cols.  Top labels: " +
                                ", ".join(f"{k}({v})" for k, v in vals.items()))

    # ---------- running ----------
    def _cfg(self):
        c = {k: v.get() for k, v in self.v.items()}
        c.update(source=self.src.get(), label_col=self.label_col.get(),
                 normals=[x.strip().lower() for x in self.normals.get().split(",") if x.strip()],
                 k_claude=self.k_claude.get().strip(), k_gem=self.k_gem.get().strip())
        return c

    def _run(self):
        cfg = self._cfg()
        if cfg["source"] == "cic":
            ok, noip = find_cic_files(getattr(self, "cic_dir", None))
            if not ok:
                d = filedialog.askdirectory(title="Select the folder with the CIC-IDS2017 TrafficLabelling CSV files")
                if d:
                    self.cic_dir = d
                    ok, noip = find_cic_files(d)
            if not ok:
                messagebox.showwarning("CIC-IDS2017", "No CIC-IDS2017 files with IP columns were found." +
                                       ("\nThe files found are the MachineLearningCVE version (no IPs). Use TrafficLabelling." if noip else
                                        "\nDownload the CSVs from https://www.unb.ca/cic/datasets/ids-2017.html and unzip them."))
                return
            cfg["cic_files"] = ok
            self.cic_lbl.config(text=f"   {len(ok)} CIC-IDS2017 file(s) found.")
        if cfg["source"] == "custom":
            if self.raw_train is None:
                messagebox.showwarning("Dataset", "Choose a training file first (or pick NSL-KDD).")
                return
            if not cfg["normals"]:
                messagebox.showwarning("Dataset", "Enter which label value(s) mean 'normal' traffic.")
                return
        self.stop_flag = False
        self.run_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.prog["value"] = 0
        self.console.delete("1.0", "end")
        self.nb.select(self.console.master)
        self.worker = threading.Thread(target=self._worker, args=(cfg, self.raw_train, self.raw_live), daemon=True)
        self.worker.start()

    def _stop(self):
        self.stop_flag = True
        self.status.config(text="Stopping after the current step…")

    def _stage(self, i, text):
        if self.stop_flag:
            raise StopRequested()
        self.q.put(("stage", (i, text)))

    def _worker(self, cfg, raw_train, raw_live):
        w = QueueWriter(self.q)
        try:
            with contextlib.redirect_stdout(w):
                res = self._pipeline(cfg, raw_train, raw_live)
            self.q.put(("done", res))
        except StopRequested:
            self.q.put(("stopped", None))
        except Exception:
            self.q.put(("error", traceback.format_exc()))

    def _pipeline(self, cfg, raw_train, raw_live):
        core = self.core
        SEED = core.SEED
        os.environ["ANTHROPIC_API_KEY"] = cfg["k_claude"]
        os.environ["GEMINI_API_KEY"] = cfg["k_gem"]
        eps, thr = float(cfg["epsilon"]), float(cfg["conf"])
        evals = []

        # ---- 1. data ----
        self._stage(0, "Loading dataset…")
        if cfg["source"] == "cic":                      # CIC-IDS2017: load the day files, then treat as custom data
            cfg = dict(cfg)
            raw_train, raw_live = load_cic(cfg["cic_files"]), None
            cfg.update(source="custom", label_col="Label", normals=["benign"])
            print(f"[CIC-IDS2017] {len(raw_train):,} rows from {len(cfg['cic_files'])} file(s) (hosts kept for the network map).")
        schema = {}
        if cfg["source"] == "nsl":
            core.NSL_KDD_COLUMNS, core.CATEGORICAL = list(self.orig_cols), list(self.orig_cat)
            ds = types.SimpleNamespace()
            ds.train_df, ds.live_df = load_nsl(list(self.orig_cols), int(cfg["train_sample"]), SEED)
            schema = dict(label_col="label", normals=["normal"], cats=list(self.orig_cat),
                          feat_cols=[c for c in self.orig_cols if c not in self.orig_cat + ["label", "difficulty"]] + list(self.orig_cat))
        else:
            tr, feats, cats = build_frame(raw_train, cfg["label_col"], cfg["normals"])
            if not cats:                      # base OneHotEncoder needs >= 1 categorical column
                tr["_const"] = "all"
                feats, cats = feats + ["_const"], ["_const"]
            if raw_live is not None:
                lv, _, _ = build_frame(raw_live, cfg["label_col"], cfg["normals"], [c for c in feats if c != "_const"], [c for c in cats if c != "_const"])
                if "_const" in feats:
                    lv["_const"] = "all"
                train_part = tr
            else:
                shuf = tr.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
                k = int(len(shuf) * 0.7)
                train_part, lv = shuf.iloc[:k], shuf.iloc[k:].reset_index(drop=True)
            core.NSL_KDD_COLUMNS = feats + ["label"]
            core.CATEGORICAL = list(cats)
            ds = types.SimpleNamespace()
            ds.train_df = train_part.sample(n=min(int(cfg["train_sample"]), len(train_part)), random_state=SEED).reset_index(drop=True)
            ds.live_df = lv.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
            nat = ds.train_df["binary_label"].sum()
            if nat < 2 or nat > len(ds.train_df) - 2:
                raise ValueError("Training data needs both normal and attack rows. Check the label column / normal values.")
            print(f"[Dataset] Custom data: {len(ds.train_df)} train rows ({nat} attack), {len(ds.live_df)} live rows.")
            schema = dict(label_col=cfg["label_col"], normals=cfg["normals"],
                          cats=[c for c in cats if c != "_const"], feat_cols=[c for c in feats if c != "_const"])
        n_tr = int(len(ds.train_df) * 0.8)
        train_df, test_df = ds.train_df.iloc[:n_tr].reset_index(drop=True), ds.train_df.iloc[n_tr:].reset_index(drop=True)
        print(f"[Data] Split into {len(train_df)} train / {len(test_df)} test records.")
        live_df = ds.live_df

        # ---- 2. preprocess + baseline ----
        self._stage(1, "Preprocessing & training baseline neural IDS…")
        prep = core.Preprocessor().fit(train_df)
        X_train, y_train = prep.transform(train_df), train_df["binary_label"].values
        X_test, y_test = prep.transform(test_df), test_df["binary_label"].values
        ev = core.Evaluator()
        baseline = mk(core.IDSModel, epochs=int(cfg["epochs"]), seed=SEED)
        baseline.train(X_train, y_train)
        base_clean = ev.testModel(baseline, X_test, y_test, "Baseline (clean)")

        # ---- 3. FGSM ----
        self._stage(2, "Crafting FGSM adversarial samples…")
        fgsm = core.FGSMGenerator(eps, mask=prep.attack_mask)
        X_test_adv = fgsm.generateAdversarial(X_test, y_test, baseline)
        base_adv = ev.testModel(baseline, X_test_adv, y_test, "Baseline (FGSM)")

        # ---- 4. adversarial training ----
        self._stage(3, "Adversarial training → robust model…")
        trainer = core.AdversarialTrainer()
        X_train_adv = fgsm.generateAdversarial(X_train, y_train, baseline)
        trainer.combineData(X_train, y_train, X_train_adv, y_train)
        Xc, yc = X_train, y_train
        robust = trainer.trainRobustModel(mk(core.IDSModel, epochs=int(cfg["epochs"]), seed=SEED + 1), Xc, yc,
                                          adv_eps=eps, mask=prep.attack_mask)
        rob_clean = ev.testModel(robust, X_test, y_test, "Robust (clean)")
        X_adv_rob = core.FGSMGenerator(eps, prep.attack_mask).generateAdversarial(X_test, y_test, robust)
        rob_adv = ev.testModel(robust, X_adv_rob, y_test, "Robust (FGSM own)")
        rob_tr = ev.testModel(robust, X_test_adv, y_test, "Robust (FGSM transfer)")
        ev.compareResults(base_adv, rob_adv)

        # ---- 5. reference RF ----
        self._stage(4, "Training Random Forest reference…")
        rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=SEED, n_jobs=-1).fit(X_train, y_train)
        p = rf.predict(X_test)
        rf_row = {"name": "RandomForest (clean)", "accuracy": round(accuracy_score(y_test, p), 4),
                  "precision": round(precision_score(y_test, p, zero_division=0), 4),
                  "recall": round(recall_score(y_test, p, zero_division=0), 4), "f1": round(f1_score(y_test, p, zero_division=0), 4)}
        print("[Reference]", rf_row)
        evals = [base_clean, base_adv, rob_clean, rob_adv, rob_tr, rf_row]

        # ---- 6. self-healing loop ----
        self._stage(5, "Self-healing loop on live traffic…")
        explainer = core.GenAIExplainer()
        explainer.fit_baseline(X_train[y_train == 0], prep.feature_names)
        monitor, kb = core.AttackMonitor(thr), core.IncidentKnowledgeBase()
        ctrl = core.SelfHealingController()
        X_live, y_live = prep.transform(live_df), live_df["binary_label"].values
        cycles = int(cfg["cycles"])
        bs = max(50, min(int(cfg["batch"]), len(X_live) // (cycles + 1)))
        while cycles * bs > len(X_live) and cycles > 1:
            cycles -= 1
        print(f"[UI] Live stream {len(X_live)} rows -> {cycles} cycles x {bs} rows. Explainer: {explainer.llmModel}")
        live_model, Xtr, ytr = copy.deepcopy(robust), Xc, yc
        for c in range(cycles):
            self._stage(5, f"Self-healing cycle {c + 1}/{cycles}…")
            s, e = c * bs, (c + 1) * bs
            live_model, Xtr, ytr = ctrl.runFeedbackLoop(live_model, monitor, explainer, kb, trainer, fgsm, ev, Xtr, ytr,
                                                        X_live[s:e], y_live[s:e], cycle_no=c + 1)
        held = cycles * bs
        held_b = held_a = None
        if held < len(X_live) - 20:
            held_b = accuracy_score(y_live[held:], robust.predict(X_live[held:]))
            held_a = accuracy_score(y_live[held:], live_model.predict(X_live[held:]))
            print(f"[Result] Held-out live accuracy: {held_b:.4f} -> {held_a:.4f}")
        self._stage(6, "Done.")
        ipmap = None
        if "_src" in live_df.columns and "_dst" in live_df.columns:     # dataset has host IPs -> link to the network map
            ipmap = live_df[["_src", "_dst", "label", "binary_label"]].copy()
            ipmap["flag_before"] = np.asarray(robust.predict(X_live)).astype(int)
            ipmap["flag_after"] = np.asarray(live_model.predict(X_live)).astype(int)
        return dict(evals=evals, history=pd.DataFrame(ctrl.history), patch=pd.DataFrame(ctrl.patch_log), ipmap=ipmap,
                    notes=ctrl.patch_notes, held=(held_b, held_a), incidents=len(kb.incidents),
                    model=live_model, prep=prep, schema=schema, explainer=explainer.llmModel,
                    held_n=len(X_live) - held)

    # ---------- queue polling ----------
    def _poll(self):
        try:
            while True:
                kind, p = self.q.get_nowait()
                if kind == "log":
                    self.console.insert("end", p)
                    self.console.see("end")
                elif kind == "stage":
                    self.prog["value"] = p[0]
                    self.status.config(text=p[1])
                elif kind == "done":
                    self._finish("Finished. Explainer used: " + p["explainer"])
                    self.results = p
                    self._show_results()
                    self._net_load_ids()
                    if self.net_ids is not None:        # everything happens automatically: heal + explain
                        self._net_heal(auto=True)
                        self.nb.select(self.net_tab)
                elif kind == "netreport":
                    self._net_set_report(p)
                elif kind == "stopped":
                    self._finish("Stopped by user.")
                elif kind == "error":
                    self._finish("Error – see Console.")
                    self.console.insert("end", "\n" + p)
                    messagebox.showerror("Pipeline error", p.strip().splitlines()[-1])
        except queue.Empty:
            pass
        self.after(100, self._poll)

    def _finish(self, msg):
        self.run_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status.config(text=msg)
        if msg.startswith("Finished"):
            self.prog["value"] = 7

    # ---------- show results ----------
    def _show_results(self):
        r = self.results
        ev = {e["name"]: e for e in r["evals"]}
        self.kpi["clean"].config(text=f"{ev['Robust (clean)']['accuracy']:.1%}")
        self.kpi["fgsm"].config(text=f"{ev['Baseline (FGSM)']['accuracy']:.0%} → {ev['Robust (FGSM own)']['accuracy']:.0%}")
        hb, ha = r["held"]
        if hb is not None:
            self.kpi["live"].config(text=f"{hb:.1%} → {ha:.1%}")
            self.held_lbl.config(text=f"Held-out live traffic ({r['held_n']:,} rows never used in the loop): "
                                      f"{hb:.2%} before healing → {ha:.2%} after healing")
        else:
            self.kpi["live"].config(text="n/a")
            self.held_lbl.config(text="Not enough live rows left for a held-out check (lower cycles / batch size).")
        self.kpi["inc"].config(text=str(r["incidents"]))

        self._fill_tv(self.eval_tv, pd.DataFrame(r["evals"])[["name", "accuracy", "precision", "recall", "f1"]])
        # chart 1
        self.fig1.clear()
        ax = self.fig1.add_subplot(111)
        names = ["Baseline\nclean", "Baseline\nFGSM", "Robust\nclean", "Robust\nFGSM"]
        vals = [ev["Baseline (clean)"]["accuracy"], ev["Baseline (FGSM)"]["accuracy"],
                ev["Robust (clean)"]["accuracy"], ev["Robust (FGSM own)"]["accuracy"]]
        bars = ax.bar(names, vals, color=["#4C78A8", "#E45756", "#4C78A8", "#54A24B"])
        ax.set_ylim(0, 1.1)
        ax.set_title("IDS accuracy under FGSM attack (add-on module)")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center")
        self.fig1.tight_layout()
        self.cv1.draw()
        # healing tab
        h = r["history"]
        self._fill_tv(self.hist_tv, h)
        self.fig2.clear()
        ax = self.fig2.add_subplot(111)
        if len(h):
            ax.plot(h["cycle"], h["live_acc_before"], "o-", label="before healing", color="#E45756")
            ax.plot(h["cycle"], h["live_acc_after"], "s-", label="after healing", color="#54A24B")
            ax.set_xticks(h["cycle"])
        ax.set_xlabel("Self-healing cycle")
        ax.set_ylabel("Accuracy on live batch")
        ax.set_title("Self-healing on unseen traffic")
        ax.legend()
        self.fig2.tight_layout()
        self.cv2.draw()
        self.notes.delete("1.0", "end")
        for i, n in enumerate(r["notes"], 1):
            self.notes.insert("end", f"Cycle {i}: {n}\n\n")
        # report
        patch = r["patch"]
        self.flt_cb["values"] = ["All"] + (sorted(patch["status"].unique()) if len(patch) else [])
        self.flt.set("All")
        self._fill_report()
        self.nb.select(0)

    def _fill_report(self):
        if self.results is None or not len(self.results["patch"]):
            return
        p = self.results["patch"]
        if self.flt.get() != "All":
            p = p[p["status"] == self.flt.get()]
        self.cur_report = p.reset_index(drop=True)
        cols = ["cycle", "sample_id", "attack_type", "truth", "before", "conf_before", "after", "conf_after", "status"]
        self.rep_tv.delete(*self.rep_tv.get_children())
        for _, row in self.cur_report.head(2000).iterrows():
            self.rep_tv.insert("", "end", values=[row[c] for c in cols])

    def _show_expl(self, _):
        sel = self.rep_tv.selection()
        if not sel:
            return
        i = self.rep_tv.index(sel[0])
        self.expl.delete("1.0", "end")
        self.expl.insert("end", str(self.cur_report.iloc[i]["explanation"]))

    def _export_report(self):
        if self.results is None:
            return
        p = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="healing_report.csv")
        if p:
            self.results["patch"].to_csv(p, index=False)
            messagebox.showinfo("Saved", p)

    # ---------- scan new traffic ----------
    def _scan(self):
        if self.results is None:
            messagebox.showinfo("Scan", "Run the pipeline first so there is a healed model to scan with.")
            return
        p = filedialog.askopenfilename(title="Dataset to scan", filetypes=[("Data", "*.csv *.txt *.data"), ("All", "*.*")])
        if not p:
            return
        try:
            sch, r = self.results["schema"], self.results
            df, _ = read_table(p, self.orig_cols)
            has_label = sch["label_col"] in df.columns
            if not has_label:
                df[sch["label_col"]] = sch["normals"][0]
            fr, _, _ = build_frame(df, sch["label_col"], sch["normals"], sch["feat_cols"] if self.src.get() == "custom" or True else None,
                                   sch["cats"])
            if "_const" in self.core.CATEGORICAL:
                fr["_const"] = "all"
            X = r["prep"].transform(fr)
            prob = r["model"].predict_proba(X)
            conf = np.maximum(prob, 1 - prob)
            out = pd.DataFrame({"row": np.arange(len(X)), "prediction": np.where(prob >= 0.5, "attack", "normal"),
                                "attack_probability": prob.round(4), "confidence": conf.round(4),
                                "flagged_unsure": conf < float(self.v["conf"].get()),
                                "truth": np.where(fr["binary_label"].values == 1, "attack", "normal") if has_label else "n/a"})
            self.scan_df = out
            n_att, n_flag = int((prob >= 0.5).sum()), int(out["flagged_unsure"].sum())
            msg = f"{len(out):,} rows scanned → {n_att:,} predicted attacks ({n_att / len(out):.1%}); {n_flag:,} flagged as unsure."
            if has_label:
                msg += f"  Accuracy vs file labels: {accuracy_score(fr['binary_label'].values, (prob >= 0.5).astype(int)):.2%}"
            self.scan_lbl.config(text=msg, foreground=PRIMARY)
            self._fill_tv(self.scan_tv, out.head(500))
        except Exception as e:
            messagebox.showerror("Scan failed", str(e))

    def _export_scan(self):
        if self.scan_df is None:
            return
        p = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="predictions.csv")
        if p:
            self.scan_df.to_csv(p, index=False)
            messagebox.showinfo("Saved", p)


def main():
    core = sys.modules[__name__]          # the original classes live in this same file
    App(core, os.path.abspath(__file__)).mainloop()


if __name__ == "__main__":
    main()
