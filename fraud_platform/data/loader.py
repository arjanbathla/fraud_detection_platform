"""Load the creditcard transactions from SQLite and split them, handling imbalance explicitly.

There is NO synthetic data generation here — the platform runs on the real dataset. First import
the Kaggle creditcard.csv into SQLite once (python -m fraud_platform.data.load_to_sqlite); this
loader then queries the `transactions` table rather than re-reading the 144MB CSV every run.

The fraud rate is ~0.17%, so:
  * every split is STRATIFIED on Class, so train/val/test all keep the same tiny fraud ratio
    (a plain random split could easily put almost no fraud in a fold);
  * the supervised model (XGBoost) handles imbalance with scale_pos_weight (see models/xgb.py);
  * the unsupervised models (Isolation Forest, Autoencoder) don't resample — they learn what
    "normal" looks like and flag outliers.
"""
from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from fraud_platform.config import CREDITCARD_DB, DB_TABLE, NUMERIC_FEATURES, RANDOM_SEED, TARGET


def load_data(db_path: str | Path = CREDITCARD_DB, table: str = DB_TABLE) -> pd.DataFrame:
    """Query all transactions from the SQLite db. Raises if the db hasn't been built yet."""
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(
            f"SQLite db not found at {db_path}.\n"
            "Build it first with: python -m fraud_platform.data.load_to_sqlite "
            "(needs the Kaggle creditcard.csv in data/). This project does NOT generate "
            "synthetic data."
        )
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
    finally:
        conn.close()
    missing = [c for c in NUMERIC_FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"{table} table is missing expected columns: {missing}")
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
