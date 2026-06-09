"""Feature engineering + preprocessing pipeline.

V1..V28 are already PCA components, so there's little feature engineering to do — and
inventing features on anonymised PCA axes wouldn't be meaningful. The pipeline therefore
focuses on what the dataset actually leaves raw:

  * Amount — heavily right-skewed and on a totally different scale to the V's, so we add a
    log1p(Amount) feature and scale it. RobustScaler is used on the raw-numeric block
    (Time, Amount, amount_log) because those have outliers; the PCA block gets StandardScaler.
  * Time — seconds since the first transaction; kept and scaled (its predictive value is
    limited — it's ordering, not time-of-day — which the README notes honestly).

The whole thing is one sklearn Pipeline so it pickles and is saved with each model, giving
identical transforms at inference.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler, StandardScaler

from fraud_platform.config import PCA_FEATURES, RAW_NUMERIC

# engineered column(s) added on top of the raw numeric block
ENGINEERED_NUMERIC = ["amount_log"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add log-amount. Pure function of the input row, safe at inference."""
    df = df.copy()
    df["amount_log"] = np.log1p(df["Amount"].clip(lower=0))
    return df


def build_pipeline() -> Pipeline:
    """Return an unfitted preprocessing Pipeline (engineering + scaling)."""
    # skewed, outlier-heavy raw columns -> RobustScaler; PCA axes -> StandardScaler
    raw_cols = RAW_NUMERIC + ENGINEERED_NUMERIC

    raw_tf = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", RobustScaler()),
    ])
    pca_tf = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    column_tf = ColumnTransformer([
        ("raw", raw_tf, raw_cols),
        ("pca", pca_tf, PCA_FEATURES),
    ], remainder="drop")

    return Pipeline([
        ("engineer", FunctionTransformer(engineer_features)),
        ("columns", column_tf),
    ])


def get_feature_names(pipeline: Pipeline) -> list[str]:
    """Best-effort output feature names from a fitted pipeline (for inspection)."""
    try:
        return list(pipeline.named_steps["columns"].get_feature_names_out())
    except Exception:
        return []
