# Anomaly & Fraud Detection Platform

An end-to-end MLOps platform for credit-card fraud detection, built on the real
**Kaggle "Credit Card Fraud Detection"** dataset. It demonstrates: an imbalance-aware data
loader, three comparable models behind a common interface (Isolation Forest, XGBoost,
PyTorch autoencoder), a preprocessing pipeline saved with each model, honest evaluation under
extreme class imbalance, a FastAPI serving layer with measured latency, a versioned model
registry, gated automated retraining, drift monitoring, a pytest suite, and a Streamlit/Plotly
dashboard. Runs on a laptop — no GPU, no paid cloud.

## The dataset (real, not synthetic)

[Kaggle: Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
~284,807 transactions by European cardholders over two days in Sept 2013, of which **492
(~0.172%) are fraud** — extreme class imbalance.

```bash
# download creditcard.csv from Kaggle and place it here:
data/creditcard.csv
```

The platform does **not** generate synthetic data — it loads this CSV directly. (The pytest
suite uses a small in-memory dataframe of the *same shape* purely as test scaffolding, clearly
labelled as such in `tests/conftest.py`.)

### Honest note on the features — they are pre-PCA'd
The columns are `Time`, `Amount`, `Class` (target), and `V1`–`V28`. **`V1`–`V28` are already
the output of a PCA transform** the dataset authors applied to anonymise the original
features — those raw features are not published. This has a direct, honest consequence:

> **Feature engineering is intentionally limited.** You cannot build meaningful domain
> features on anonymised PCA axes. So the work here is **scaling, `Amount`/`Time` handling,
> and resampling/imbalance treatment** — not feature creation. Specifically:
> - `Amount`: right-skewed and on a different scale to the V's → add `log1p(Amount)` and apply `RobustScaler`.
> - `Time`: seconds since the first transaction (it's ordering, not time-of-day, so its predictive value is limited) → kept and scaled.
> - `V1`–`V28`: standardised with `StandardScaler` (already PCA'd, so left as-is otherwise).

---

## How the extreme imbalance is handled

| Concern | What the platform does |
|--------|------------------------|
| Splitting | **Stratified** 60/20/20 train/val/test on `Class`, so all folds keep the ~0.17% fraud ratio (a plain random split could leave a fold with almost no fraud). |
| XGBoost (supervised) | `scale_pos_weight = n_neg / n_pos` so the rare fraud class is weighted up during training. No resampling, so no leakage risk and the full majority data is kept. |
| Isolation Forest, Autoencoder | Treated as **unsupervised anomaly detection** — they learn what normal transactions look like and score outliers. The autoencoder trains on legit rows only and scores by reconstruction error; Isolation Forest doesn't use labels at all. |
| Metrics | We lead with **PR-AUC** and **recall**, not accuracy/ROC-AUC, because on a 0.17% positive rate a "predict everything legit" model is 99.83% accurate and useless. |

> SMOTE/undersampling were considered. We chose **`scale_pos_weight`** because it needs no
> extra dependency, doesn't discard the majority data (undersampling) or fabricate minority
> points (SMOTE), and carries no resampling-leakage risk. The trade-off: it nudges the loss
> rather than rebalancing the data, so it's less aggressive than SMOTE — a reasonable default
> to swap if you want to experiment.

---

## Architecture

```
fraud_platform/
├── config.py                  # schema (Time, V1-V28, Amount, Class), paths, thresholds
├── data/loader.py             # load creditcard.csv + stratified split + hash (NO generation)
├── pipeline/features.py       # log-Amount + RobustScaler/StandardScaler ColumnTransformer
├── models/
│   ├── base.py                # AnomalyModel ABC — the common interface
│   ├── iforest.py             # Isolation Forest (unsupervised)
│   ├── xgb.py                 # XGBoost (supervised, scale_pos_weight)
│   ├── autoencoder.py         # PyTorch autoencoder (reconstruction error, legit-only)
│   └── __init__.py            # name -> class factory (swappable models)
├── evaluation/evaluate.py     # precision/recall/F1/ROC-AUC/PR-AUC + comparison table
├── registry/registry.py       # local versioned model registry + champion pointer
├── monitoring/drift.py        # PSI + KS drift detection
├── serving/api.py             # FastAPI /predict with per-request latency logging
├── train.py                   # train all 3, evaluate, register
└── retrain.py                 # retrain -> evaluate -> gated promotion
dashboard/app.py               # Streamlit + Plotly (EDA, comparison, sample-row scoring, drift)
tests/                         # pytest suite
```

The fitted preprocessing **Pipeline is saved inside every model artifact**, so inference
applies the exact transforms seen at training time. The three models are interchangeable
behind the `AnomalyModel` interface — `score()` always returns a value where *higher = more
anomalous*, supervised or not.

---

## Quick start

> On macOS the interpreter is usually `python3` (no bare `python`). Use `python3` below.

```bash
pip install -r requirements.txt           # see "macOS note" below if XGBoost fails to load

# 1. put the Kaggle file at data/creditcard.csv  (the platform will NOT run without it)
python3 -m fraud_platform.train           # load + train all 3 + evaluate + register
python3 -m pytest -q                       # run the test suite (uses in-memory fixtures)

python3 -m uvicorn fraud_platform.serving.api:app   # serve on http://localhost:8000  (try /docs)
python3 -m streamlit run dashboard/app.py           # dashboard (run the API first for live scoring)
```

### Docker
```bash
docker build -t fraud-platform .
docker run -p 8000:8000 -v "$(pwd)/data:/app/data" fraud-platform
```
The dataset is mounted (not baked in). The entrypoint trains once if `data/creditcard.csv` is
present and no registry exists, then serves.

### macOS note (OpenMP)
PyTorch and XGBoost each ship an OpenMP runtime; running both multithreaded can segfault. The
package pins `OMP_NUM_THREADS=1` at import in
[`fraud_platform/__init__.py`](fraud_platform/__init__.py). You may also need `brew install
libomp` for XGBoost. On Linux/Docker, `libgomp1` is installed and you can relax the pin.

---

## Results (measured on the real Kaggle dataset)

Produced by `python3 -m fraud_platform.train` (full 284,807 rows, stratified 60/20/20, seed
42; test set = 56,962 rows containing 98 fraud cases) and saved to
`artifacts/evaluation_results.json`. Reproducible — the seed makes these deterministic.

| model | precision | recall | F1 | ROC-AUC | PR-AUC |
|-------|-----------|--------|------|---------|--------|
| isolation_forest | 0.2632 | 0.3061 | 0.2830 | 0.9551 | 0.1490 |
| **xgboost** | 0.9412 | 0.8163 | 0.8743 | 0.9788 | **0.8770** |
| autoencoder | 0.5105 | 0.7449 | 0.6058 | 0.9577 | 0.5780 |

**How to read them honestly:**
- The baseline PR-AUC for a random model equals the fraud rate, **~0.0017**. XGBoost's 0.877
  is far above that; the autoencoder (0.578) is respectable for an unsupervised model;
  Isolation Forest (0.149) is weak but still ~88× the random baseline.
- **Recall is the headline number** — the fraction of real fraud caught. Supervised XGBoost
  catches **82%** of fraud at 94% precision. The autoencoder catches **74%** but at only 51%
  precision (more false alarms). Isolation Forest catches just **31%**.
- **ROC-AUC is misleadingly high for every model (≥ 0.955)** purely because of the imbalance —
  this is exactly why PR-AUC and recall are the metrics to trust here, not ROC-AUC.
- Supervised clearly beats unsupervised when labels are available, as expected.

---

## Latency (measured)

The API logs `inference_ms` per request and returns it in the response, so this is verifiable.
Measured on the **real trained models**, single-record scoring, 500 requests after warm-up,
laptop (Apple Silicon, CPU only):

| path | mean | p95 | p99 |
|------|------|-----|-----|
| XGBoost `model.score` (incl. preprocessing) | 1.6 ms | 1.7 ms | 1.8 ms |
| Autoencoder `model.score` | 1.5 ms | 1.6 ms | 1.7 ms |
| Isolation Forest `model.score` | 5.6 ms | 5.8 ms | 5.9 ms |
| **End-to-end `/predict`** (XGBoost, incl. HTTP) | **2.8 ms** | 3.0 ms | 3.4 ms |

**Verdict: the "< 100ms per request" claim holds comfortably** — ~2.8ms end-to-end on a
laptop, single request, single thread. The response includes `inference_ms` so you can verify
it yourself. Sustained throughput under concurrency was not benchmarked — don't claim it.

---

## Model registry & versioning

Home-grown, dependency-free, under `registry_store/`:

```
registry_store/
├── index.json                 # {model_type: {"champion": <version>}}
└── <model_type>/v<N>/
    ├── model.pkl | model.pt    # model + its fitted preprocessing pipeline
    └── metadata.json           # version, utc_timestamp, model_type, metrics, data_hash
```

Each `register()` writes a new **immutable** version; the champion pointer says which version
serving loads. `data_hash` ties each model to the exact training data.

**Why not MLflow?** In production MLflow (or similar) is the right choice — a tracking server,
experiment-comparison UI, a backing artifact store (S3/GCS), and model-stage transitions with
access control. This registry is deliberately minimal so the structure is transparent and runs
with zero setup; it covers versioning, metadata, and a champion pointer, but not MLflow's UI
or lineage graph.

---

## Automated retraining (gated promotion)

```bash
python3 -m fraud_platform.retrain                        # retrain all on data/creditcard.csv
python3 -m fraud_platform.retrain --data new_batch.csv   # retrain on a fresh batch
python3 -m fraud_platform.retrain --model xgboost        # one model type
```

**What triggers a retrain:** this is a **manual command**, designed to run **on a schedule
(cron / CI)** or by hand when a new labelled batch arrives or drift is flagged. It is **not**
event-driven / streaming auto-retraining. Example cron (daily 02:00, inside the container):

```cron
0 2 * * *  cd /app && python -m fraud_platform.retrain
```

**Promotion gate:** a candidate is always registered as a new version, but only becomes
champion if its **PR-AUC** on the held-out test set is *strictly greater* than the current
champion's. A worse model can never silently replace a good one.

---

## Drift detection

```python
from fraud_platform.monitoring.drift import compute_drift
result = compute_drift(reference_df, current_df)   # per-feature PSI + KS, overall flag
```

- **PSI (Population Stability Index)** — effect size of shift. Bands: `< 0.10` none ·
  `0.10–0.25` moderate · `≥ 0.25` significant.
- **KS test** — two-sample Kolmogorov–Smirnov; p `< 0.05` rejects "same distribution". KS
  over-fires on large samples, which is why it's paired with PSI's effect-size view.

A feature is flagged if `PSI ≥ 0.25` **or** `KS p < 0.05`; the batch drifts if any feature is
flagged. The dashboard's Drift tab lets you inject a synthetic `Amount` shift and watch it flag.

---

## Testing

```bash
python3 -m pytest -q
```

Covers the loader (missing file, schema validation, stratified split keeps fraud in every
fold), the preprocessing pipeline (shape, determinism, scaling), the model interface (all
three conform to the ABC; save/load round-trips reproduce scores), the registry (versioning +
promotion gate), drift (no-drift vs injected drift), evaluation metrics, and the API
(`/health`, `/model-info`, `/predict` schema + latency field + input validation).

---

## What the build honestly supports (CV accuracy)

| Claim | Status | Notes for your CV |
|-------|--------|-------------------|
| Built on the real Kaggle Credit Card Fraud dataset (~284k tx, 0.17% fraud) | ✅ | It's a well-known public benchmark, not your own data pipeline — say "Kaggle dataset". |
| Imbalance handled: stratified split + `scale_pos_weight` (supervised), unsupervised framing for IF/AE | ✅ | Accurate to say "handled extreme imbalance"; the resampling used is class weighting, not SMOTE. |
| 3 models behind a common, swappable interface | ✅ | Isolation Forest, XGBoost, PyTorch autoencoder. |
| Preprocessing in a sklearn Pipeline, saved with the model, applied identically at inference | ✅ | Pipeline pickled inside each artifact. |
| Reports precision/recall/F1/ROC-AUC/PR-AUC on a held-out test set | ✅ | Measured: XGBoost PR-AUC 0.877, recall 0.82. Lead with PR-AUC/recall, not ROC-AUC. |
| FastAPI `/predict`, measured per-request latency, **< 100ms** | ✅ | ~2.8ms end-to-end on a laptop, single request. Don't claim high-concurrency throughput. |
| Local versioned model registry with metadata | ✅ | Call it "home-grown"; MLflow is the production choice. |
| Retraining that promotes only if it beats the champion on a metric | ✅ | **Manual/scheduled** command with a PR-AUC gate — not streaming auto-retrain. |
| Drift detection (PSI + KS) with documented thresholds | ✅ | |
| pytest suite over loader, pipeline, model interface, registry, drift, API | ✅ | |
| Streamlit + Plotly dashboard: EDA, comparison, live scoring, drift | ✅ | Live scoring samples a real test row and calls the API. |
| Containerised (requirements.txt + Dockerfile) | ✅ | Dataset is mounted at runtime, not shipped (Kaggle licence). |

**Things to NOT claim:**
- ❌ "Engineered rich domain features" — the inputs are anonymised PCA components; engineering is limited to scaling + Amount/Time + resampling. Be explicit about this.
- ❌ "Used SMOTE" — this build uses `scale_pos_weight` (unless you switch it).
- ❌ "MLflow" — the registry is home-grown.
- ❌ "Real-time streaming / event-driven retraining" — it's a manual/scheduled command.
- ❌ "Production-deployed at scale / N req/s" — laptop / single container; only single-request latency measured.

**Honest one-liner:**
> "Built an end-to-end MLOps pipeline for credit-card fraud detection on the Kaggle dataset
> (284k transactions, 0.17% fraud): stratified imbalance handling, 3 swappable models
> (Isolation Forest, XGBoost with class weighting, PyTorch autoencoder) behind a common
> interface, sklearn preprocessing persisted with each model, a versioned registry with a
> PR-AUC-gated retraining command, PSI/KS drift monitoring, a FastAPI scoring service
> (~2ms/request on a laptop), a pytest suite, and a Streamlit dashboard — all containerised."

---

## Trade-offs

- **`scale_pos_weight` over SMOTE/undersampling** — no extra dep, no leakage, keeps all data.
- **Single-threaded OpenMP on macOS** — avoids the torch/XGBoost segfault; relax on Linux.
- **Home-grown registry over MLflow** — zero setup and transparent, at the cost of a UI/lineage.
- **Dataset mounted, not shipped** — respects the Kaggle licence and keeps the image small.
- **`Time` kept but lightly used** — it's ordering, not time-of-day, so it carries limited signal.
