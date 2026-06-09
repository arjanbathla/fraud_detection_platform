"""Data loading + splitting helpers shared by training, retraining and tests."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from fraud_platform.config import DATA_DIR, RANDOM_SEED, TARGET
from fraud_platform.data.generate import generate

DEFAULT_DATA = DATA_DIR / "transactions.csv"


def load_or_generate(path: str | Path = DEFAULT_DATA, rows: int = 50_000,
                     fraud_frac: float = 0.01) -> pd.DataFrame:
    """Load the CSV if present, otherwise generate it once and cache to disk."""
    path = Path(path)
    if path.exists():
        return pd.read_csv(path, parse_dates=["timestamp"])
    df = generate(rows, fraud_frac)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def split(df: pd.DataFrame, seed: int = RANDOM_SEED):
    """Stratified 60/20/20 train/val/test split.

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
