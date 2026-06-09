# Anomaly & Fraud Detection Platform — Design Spec

**Date:** 2026-06-09
**Status:** Approved
**Author:** built with Claude Code

## Goal

A runnable-on-a-laptop, production-*style* anomaly & fraud detection platform in Python
that genuinely demonstrates MLOps practices: synthetic data generation, three comparable
models behind a common interface, a saved preprocessing pipeline, honest evaluation under
class imbalance, a FastAPI serving layer with measured latency, a local versioned model
registry, gated automated retraining, drift monitoring, a real pytest suite, and a
Streamlit/Plotly dashboard. Containerised with a Dockerfile.

**All data is SYNTHETIC.** No real financial data is used. Models will look optimistically
good because the data is generated; the README states this explicitly so CV claims stay honest.

## Key decisions (locked)

- **Promotion metric:** PR-AUC (Average Precision) — appropriate for heavy class imbalance.
- **Autoencoder framework:** PyTorch (CPU, no GPU required).
- **Numbers:** Real — deps installed, all 3 models trained, latency measured, README reflects truth.
- **Retrain trigger:** Manual command `python -m fraud_platform.retrain` (cron-schedulable).
  NOT event-driven auto-retraining — documented as such.

## Architecture

```
fraud_platform/
├── config.py                # central config (paths, seeds, thresholds)
├── data/generate.py         # synthetic transaction generator
├── pipeline/features.py     # sklearn ColumnTransformer + feature engineering
├── models/
│   ├── base.py              # AnomalyModel ABC (common interface)
│   ├── iforest.py           # Isolation Forest (unsupervised)
│   ├── xgb.py               # XGBoost (supervised)
│   └── autoencoder.py       # PyTorch autoencoder (reconstruction error)
├── evaluation/evaluate.py   # precision/recall/F1/ROC-AUC/PR-AUC + comparison table
├── registry/registry.py     # local versioned registry
├── monitoring/drift.py      # PSI + KS test per feature
├── serving/api.py           # FastAPI /predict + latency logging
├── train.py                 # train all 3, evaluate, register
└── retrain.py               # retrain → evaluate → gated promotion
dashboard/app.py             # Streamlit + Plotly
tests/                       # pytest suite
requirements.txt, Dockerfile, README.md
```

## Common model interface (`models/base.py`)

```python
class AnomalyModel(ABC):
    name: str
    supervised: bool
    def fit(self, X, y=None) -> "AnomalyModel"
    def score(self, X) -> np.ndarray      # higher = more anomalous, per row
    def predict(self, X, threshold) -> np.ndarray  # 0/1
    def save(self, path) -> None
    @classmethod
    def load(cls, path) -> "AnomalyModel"
```

Unsupervised models (IForest, AE) ignore `y`. XGBoost requires it. `score()` returns a
consistent "anomaly score" semantics across all three so evaluation and serving are uniform.

## Preprocessing pipeline

`build_pipeline()` returns a `ColumnTransformer` wrapped in a `Pipeline`:
- numeric features → median impute + StandardScaler
- categorical features → most-frequent impute + OneHotEncoder(handle_unknown="ignore")
- engineered features added before the transformer (amount log, time-of-day, etc.)

The **fitted pipeline is bundled and saved with each model artifact**, guaranteeing identical
transforms at inference.

## Registry layout

```
registry_store/
├── index.json                       # champion pointer per model_type
└── <model_type>/v<N>/
    ├── model.pkl | model.pt
    ├── pipeline.pkl
    └── metadata.json   # version, utc_timestamp, model_type, metrics, data_hash
```

README notes MLflow would be the production choice (experiment tracking, model staging,
artifact stores); home-grown chosen for zero-dependency laptop transparency.

## Evaluation

On a stratified held-out test set: precision, recall, F1, ROC-AUC, PR-AUC for each model.
Threshold chosen per model (default: maximise F1 on a validation split, documented).
Prints a comparison table and saves `evaluation_results.json`. Emphasises recall under imbalance.

## Drift monitoring

`compute_drift(reference_df, current_df)` → per-feature PSI and KS p-value, plus a flag.
PSI bands: <0.1 none, 0.1–0.25 moderate, >0.25 significant (documented). KS flags p < 0.05.

## Serving

FastAPI `/predict` loads the current champion (model + pipeline) once at startup.
Each request: parse → pipeline.transform → score → threshold → return prediction + score +
`inference_ms`. Latency logged. `/health` and `/model-info` endpoints too.

## Testing (pytest)

- data: label fraction within tolerance, shapes, reproducible seed, SYNTHETIC marker
- pipeline: fit/transform shape, handles unseen categories, round-trips through save/load
- model interface: all 3 conform to ABC; score() shape; predict() is 0/1
- registry: register increments version; promotion gate logic; load returns usable model
- drift: identical data → no drift; shifted data → flagged
- API: /health ok; /predict returns valid schema and a number; latency field present

## Trade-offs

- Synthetic data inflates metrics — stated explicitly.
- Home-grown registry lacks MLflow's UI/lineage — documented.
- PyTorch AE adds install weight but is the honest "real NN" choice.
- Manual/cron retrain, not streaming auto-retrain — documented to keep CV accurate.
