"""Feature engineering + preprocessing pipeline.

The whole thing is a single sklearn Pipeline so it can be pickled and saved alongside the
model. At inference we load this exact fitted object and call .transform(), guaranteeing the
same imputation, scaling and encoding the model saw at train time.

Flow:  raw dataframe  ->  engineer_features (adds derived cols)  ->  ColumnTransformer
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from fraud_platform.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES

# columns the engineering step adds on top of the raw NUMERIC_FEATURES
ENGINEERED_NUMERIC = [
    "amount_log",
    "amount_to_avg_ratio",
    "hour_sin",
    "hour_cos",
    "is_night",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add a few derived features. Pure function of the input row, safe at inference."""
    df = df.copy()
    df["amount_log"] = np.log1p(df["amount"].clip(lower=0))
    # how big is this amount vs the account's recent average (guards divide-by-zero)
    df["amount_to_avg_ratio"] = df["amount"] / (df["avg_amount_last_30d"].abs() + 1.0)
    # hour is cyclical: encode so 23:00 and 00:00 are close
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["is_night"] = ((df["hour"] < 6) | (df["hour"] >= 22)).astype(int)
    return df


def build_pipeline() -> Pipeline:
    """Return an unfitted preprocessing Pipeline (engineering + column transforms)."""
    numeric_cols = NUMERIC_FEATURES + ENGINEERED_NUMERIC

    numeric_tf = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_tf = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    column_tf = ColumnTransformer([
        ("num", numeric_tf, numeric_cols),
        ("cat", categorical_tf, CATEGORICAL_FEATURES),
    ], remainder="drop")

    return Pipeline([
        # engineer_features returns a DataFrame, so ColumnTransformer can still select by name
        ("engineer", FunctionTransformer(engineer_features)),
        ("columns", column_tf),
    ])


def get_feature_names(pipeline: Pipeline) -> list[str]:
    """Best-effort output feature names from a fitted pipeline (for inspection)."""
    try:
        return list(pipeline.named_steps["columns"].get_feature_names_out())
    except Exception:
        return []
