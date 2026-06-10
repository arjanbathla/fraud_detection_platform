"""Shared pytest fixtures.

These build a small *creditcard.csv-shaped* dataframe in memory (Time, V1..V28, Amount,
Class) so the tests run fast and don't need the real 144MB download. This is test scaffolding
only — the platform itself never generates data; it loads the real Kaggle CSV via
fraud_platform.data.loader. Fraud rows here are shifted on a few V columns so the models have
something to learn during tests.
"""
import sqlite3

import numpy as np
import pandas as pd
import pytest

from fraud_platform.config import DB_TABLE, PCA_FEATURES, TARGET


def make_creditcard_like(n: int = 3000, fraud_frac: float = 0.05, seed: int = 7) -> pd.DataFrame:
    """A tiny synthetic stand-in with the same schema as creditcard.csv (TEST USE ONLY)."""
    rng = np.random.default_rng(seed)
    n_fraud = max(int(round(n * fraud_frac)), 2)
    n_legit = n - n_fraud

    # PCA-like components ~ standard normal
    legit = rng.normal(0, 1, size=(n_legit, len(PCA_FEATURES)))
    fraud = rng.normal(0, 1, size=(n_fraud, len(PCA_FEATURES)))
    # shift a few "V" columns for fraud so there is signal to detect
    fraud[:, [0, 2, 9, 13]] += rng.normal(3.0, 1.0, size=(n_fraud, 4))

    V = np.vstack([legit, fraud])
    df = pd.DataFrame(V, columns=PCA_FEATURES)
    df["Time"] = rng.integers(0, 172_792, size=n)
    df["Amount"] = np.round(rng.lognormal(3.0, 1.2, size=n), 2)
    df[TARGET] = np.r_[np.zeros(n_legit, dtype=int), np.ones(n_fraud, dtype=int)]
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def write_db(df: pd.DataFrame, path, table: str = DB_TABLE) -> None:
    """Dump a creditcard-shaped dataframe into a SQLite file (TEST USE ONLY)."""
    conn = sqlite3.connect(path)
    try:
        df.to_sql(table, conn, if_exists="replace", index=False)
    finally:
        conn.close()


@pytest.fixture(scope="session")
def sample_df():
    return make_creditcard_like(n=3000, fraud_frac=0.05, seed=7)


@pytest.fixture
def sample_db(tmp_path, sample_df):
    """A small SQLite db loaded from sample_df, for loader/EDA tests."""
    p = tmp_path / "creditcard.db"
    write_db(sample_df, p)
    return p


@pytest.fixture(scope="session")
def split_data(sample_df):
    from fraud_platform.data.loader import split
    return split(sample_df, seed=7)


@pytest.fixture(scope="session")
def fitted_xgb(split_data):
    from fraud_platform.evaluation.evaluate import choose_threshold
    from fraud_platform.models import make_model
    X_train, y_train, X_val, y_val, X_test, y_test = split_data
    model = make_model("xgboost", n_estimators=80)
    model.fit(X_train, y_train)
    model.threshold = choose_threshold(y_val, model.score(X_val))
    return model
