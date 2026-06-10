"""Central configuration: paths, seeds, schema, thresholds.

The dataset is the Kaggle "Credit Card Fraud Detection" set (creditcard.csv): ~284,807
European card transactions from Sept 2013, ~0.172% fraud. Columns are:
  Time              seconds elapsed between this transaction and the first in the set
  V1..V28           anonymised PCA components (the original features are not published)
  Amount            transaction amount
  Class             target — 1 = fraud, 0 = legit

Because V1..V28 are already PCA-transformed, feature engineering is intentionally limited to
scaling and Amount/Time handling (see pipeline/features.py).
"""
from __future__ import annotations

from pathlib import Path

# ---- paths ----
ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT / "artifacts"
REGISTRY_DIR = ROOT / "registry_store"
DATA_DIR = ROOT / "data"
CREDITCARD_CSV = DATA_DIR / "creditcard.csv"
CREDITCARD_DB = DATA_DIR / "creditcard.db"   # SQLite store the platform actually loads from
DB_TABLE = "transactions"

for _d in (ARTIFACTS_DIR, REGISTRY_DIR, DATA_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---- reproducibility ----
RANDOM_SEED = 42

# ---- dataset schema ----
TARGET = "Class"
PCA_FEATURES = [f"V{i}" for i in range(1, 29)]      # V1..V28, already PCA'd
# the only raw, non-PCA numeric columns we actually transform
RAW_NUMERIC = ["Time", "Amount"]
NUMERIC_FEATURES = RAW_NUMERIC + PCA_FEATURES        # everything the model sees
# no categorical features in this dataset
CATEGORICAL_FEATURES: list[str] = []

# ---- promotion / drift thresholds ----
PROMOTION_METRIC = "pr_auc"          # average precision; right signal under heavy imbalance
PSI_MODERATE = 0.10                  # 0.10-0.25 = moderate population shift
PSI_SIGNIFICANT = 0.25               # > 0.25 = significant drift
KS_PVALUE = 0.05                     # KS test: p < 0.05 flags a distribution change

# ---- serving ----
LATENCY_BUDGET_MS = 100.0            # the claim we measure against
