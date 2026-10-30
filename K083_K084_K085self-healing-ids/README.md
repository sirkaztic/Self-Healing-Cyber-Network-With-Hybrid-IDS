# Self-Healing IDS

A machine-learning **intrusion detection system that heals itself**. It detects network attacks, flags traffic it is unsure about (including attack types it has never seen), explains why, and automatically retrains to patch the gap, repeating the loop over several cycles. A desktop UI wraps the whole pipeline.

Built around the [NSL-KDD](https://github.com/defcom17/NSL_KDD) dataset, with optional support for CIC-IDS2017, UNSW-NB15 and your own CSVs.

## How it works

```
live traffic -> detect -> low confidence? -> explain -> log incident -> adversarial retrain -> redeploy
                                                                                      ^__________|
```

| Stage | What it does |
|---|---|
| **Dataset / Preprocessor** | Loads NSL-KDD (`KDDTrain+` for training, `KDDTest+` as genuinely unseen live traffic), one-hot encodes, log-scales and standardises. |
| **IDSModel** | A small MLP written in pure NumPy (64 -> 32 -> 1, sigmoid), so real input gradients are available. |
| **FGSMGenerator** | Real Fast Gradient Sign Method: `x_adv = x + eps * sign(dLoss/dx)`. |
| **AdversarialTrainer** | Trains on clean + FGSM-attacked samples to build a robust model. |
| **GenAIExplainer** | Explains flagged samples with Claude or Gemini if a key is set, otherwise a built-in local explainer that uses the actual feature z-scores. |
| **IncidentKnowledgeBase / AttackMonitor** | Logs flagged incidents and watches the stream. |
| **SelfHealingController** | Detect -> explain -> log -> retrain -> deploy, repeated each cycle. Each sample is marked `PATCHED`, `HARDENED`, `STILL WRONG` or `REGRESSED`. |
| **Desktop UI (Tkinter)** | Overview, Self-Healing, Healing Report, Scan New Traffic, Network Healing (attack/damage/heal simulation on a network map) and Console tabs. |

## Quick start

```bash
git clone https://github.com/<your-username>/self-healing-ids.git
cd self-healing-ids
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python self_healing_ids.py
```

On first run the NSL-KDD files are downloaded automatically, so you need an internet connection once. Tkinter ships with most Python installs; on Debian/Ubuntu run `sudo apt install python3-tk` if it is missing.

## Configuration

Defaults sit at the top of `self_healing_ids.py`:

| Setting | Default | Meaning |
|---|---|---|
| `TRAIN_SAMPLE` | 20000 | Rows of KDDTrain+ used for training |
| `EPSILON` | 0.25 | FGSM strength (in standard deviations) |
| `CONF_THRESHOLD` | 0.85 | Below this confidence a sample is treated as an unknown attack |
| `CYCLES` | 4 | Self-healing loop iterations |
| `BATCH_SIZE` | 1500 | Live samples per cycle |

Most of these can also be changed from the UI.

### Optional LLM explanations

Set a key as an environment variable (or paste it into the UI) and explanations will use a real LLM; otherwise the local explainer is used.

```bash
export ANTHROPIC_API_KEY=...   # Claude
# or
export GEMINI_API_KEY=...      # Gemini
```

Never commit keys. `.env` files are git-ignored.

## Datasets

- **NSL-KDD** (default): downloaded automatically.
- **CIC-IDS2017**: place the `TrafficLabelling` day CSVs in a `CIC-IDS2017/` folder next to the script (or `Downloads`/`Documents`). Only that version includes the IP columns needed for the Network Healing map.
- **Custom CSV**: pick a file in the UI; the label column and "normal" values are auto-detected and can be adjusted.

Datasets are large and not stored in this repo.

## Project layout

```
self_healing_ids.py   # core pipeline (Part A) + desktop UI (Part B), single file
requirements.txt
LICENSE
```

## Origin

Started life as a Google Colab notebook (`Self_Healing_IDS_Colab`) and was extended with a desktop interface.

## License

MIT, see [LICENSE](LICENSE).
