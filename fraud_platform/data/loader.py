"""Load the Kaggle creditcard.csv and split it, handling the extreme imbalance explicitly.

There is NO synthetic data generation here — the platform runs on the real dataset. Download
creditcard.csv from Kaggle ("Credit Card Fraud Detection") and put it in data/.

The fraud rate is ~0.17%, so:
  * every split is STRATIFIED on Class, so train/val/test all keep the same tiny fraud ratio
    (a plain random split could easily put almost no fraud in a fold);
  * the supervised model (XGBoost) handles imbalance with scale_pos_weight (see models/xgb.py);
  * the unsupervised models (Isolation Forest, Autoencoder) don't resample — they learn what
    "normal" looks like and flag outliers.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from fraud_platform.config import CREDITCARD_CSV, NUMERIC_FEATURES, RANDOM_SEED, TARGET


def load_data(path: str | Path = CREDITCARD_CSV) -> pd.DataFrame:
    """Read creditcard.csv. Raises a clear error if the file is not present."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"creditcard.csv not found at {path}.\n"
            "Download the Kaggle 'Credit Card Fraud Detection' dataset and place "
            "creditcard.csv in the data/ folder. This project does NOT generate synthetic data."
        )
    df = pd.read_csv(path)
    missing = [c for c in NUMERIC_FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"creditcard.csv is missing expected columns: {missing}")
    return df


def split(df: pd.DataFrame, seed: int = RANDOM_SEED):
    """Stratified 60/20/20 train/val/test split that preserves the fraud ratio in each fold.

    val is used to pick the decision threshold, test is the held-out report set.
    Returns (X_train, y_train, X_val, y_val, X_test, y_test).
    """
    y = df[TARGET].values
    X = df.drop(columns=[TARGET])
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=seed)  # 0.25*0.8 = 0.2
    return X_train, y_train, X_val, y_val, X_test, y_test


def data_hash(df: pd.DataFrame) -> str:
    """Short stable hash of a dataframe's contents for registry lineage."""
    return hashlib.sha256(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()
    ).hexdigest()[:12]
