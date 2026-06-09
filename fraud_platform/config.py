"""Central configuration: paths, seeds, schema, thresholds.

Keeping these in one place means the data generator, pipeline, training, serving and
tests all agree on column names and where artifacts live.
"""
from __future__ import annotations

from pathlib import Path

# ---- paths ----
ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT / "artifacts"
REGISTRY_DIR = ROOT / "registry_store"
DATA_DIR = ARTIFACTS_DIR / "data"

for _d in (ARTIFACTS_DIR, REGISTRY_DIR, DATA_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---- reproducibility ----
RANDOM_SEED = 42

# ---- dataset schema ----
# These are the raw columns the generator emits and the pipeline consumes.
TARGET = "is_fraud"

NUMERIC_FEATURES = [
    "amount",
    "account_age_days",
    "num_tx_last_24h",
    "avg_amount_last_30d",
    "distance_from_home_km",
    "hour",
]
CATEGORICAL_FEATURES = [
    "merchant_category",
    "transaction_type",
    "device_type",
    "country",
]
# id columns we keep for traceability but never feed to the model
ID_COLUMNS = ["transaction_id", "timestamp"]

# ---- promotion / drift thresholds ----
PROMOTION_METRIC = "pr_auc"          # average precision; see registry.promote logic
PSI_MODERATE = 0.10                  # 0.10-0.25 = moderate population shift
PSI_SIGNIFICANT = 0.25               # > 0.25 = significant drift
KS_PVALUE = 0.05                     # KS test: p < 0.05 flags a distribution change

# ---- serving ----
LATENCY_BUDGET_MS = 100.0            # the claim we measure against
