# Self-Healing Hybrid IDS with Adversarial Robustness and GenAI Explanation (K083 / K084 / K085)

Pipeline: load NSL-KDD -> preprocess -> neural-network IDS -> real FGSM attack -> adversarial training ->
evaluate -> monitor live traffic -> explain unknown attacks -> log to knowledge base -> retrain -> redeploy (4 cycles).

## Run in Google Colab
1. Open `Self_Healing_IDS_Colab.ipynb` in Colab (File > Upload notebook) and choose Runtime > Run all.
2. Or paste `self_healing_ids.py` into a single Colab cell and run it.

## Run locally
```
pip install -r requirements.txt
python self_healing_ids.py
```
The dataset (KDDTrain+ / KDDTest+) is downloaded automatically from a public GitHub mirror of NSL-KDD.

## Optional: real LLM explanations
Set `ANTHROPIC_API_KEY` or `GEMINI_API_KEY` in the config cell. Without a key a local explainer is used.

## Classes (match the UML class diagram)
Dataset, Preprocessor, IDSModel, FGSMGenerator, AdversarialTrainer, Evaluator, AttackMonitor,
GenAIExplainer, ExplanationReport, IncidentKnowledgeBase, SelfHealingController.
