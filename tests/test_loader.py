"""Tests for the SQLite-backed loader, the CSV->SQLite import, and the imbalance-aware split."""
import sqlite3

import pytest

from fraud_platform.config import DB_TABLE, NUMERIC_FEATURES, TARGET
from fraud_platform.data.load_to_sqlite import load_csv_to_sqlite
from fraud_platform.data.loader import data_hash, load_data, split
from tests.conftest import make_creditcard_like


def test_load_missing_db_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_data(tmp_path / "nope.db")


def test_load_reads_db_and_checks_schema(sample_db, sample_df):
    df = load_data(sample_db)
    assert len(df) == len(sample_df)
    for col in NUMERIC_FEATURES + [TARGET]:
        assert col in df.columns


def test_load_rejects_bad_schema(tmp_path, sample_df):
    from tests.conftest import write_db
    p = tmp_path / "bad.db"
    write_db(sample_df.drop(columns=["V1"]), p)
    with pytest.raises(ValueError):
        load_data(p)


def test_csv_to_sqlite_roundtrips(tmp_path, sample_df):
    csv = tmp_path / "creditcard.csv"
    db = tmp_path / "creditcard.db"
    sample_df.to_csv(csv, index=False)

    n = load_csv_to_sqlite(csv, db, chunksize=500)
    assert n == len(sample_df)

    # the table exists, has every expected column, and the right row count
    conn = sqlite3.connect(db)
    try:
        rows = conn.execute(f"SELECT COUNT(*) FROM {DB_TABLE}").fetchone()[0]
        cols = {r[1] for r in conn.execute(f"PRAGMA table_info({DB_TABLE})")}
    finally:
        conn.close()
    assert rows == len(sample_df)
    assert set(NUMERIC_FEATURES + [TARGET]).issubset(cols)


def test_csv_to_sqlite_missing_csv_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_csv_to_sqlite(tmp_path / "nope.csv", tmp_path / "out.db")


def test_split_is_stratified(sample_df):
    X_train, y_train, X_val, y_val, X_test, y_test = split(sample_df, seed=7)
    overall = sample_df[TARGET].mean()
    for y in (y_train, y_val, y_test):
        assert y.sum() > 0  # fraud present in every fold
        assert abs(y.mean() - overall) < 0.02


def test_split_sizes_disjoint(sample_df):
    X_train, y_train, X_val, y_val, X_test, y_test = split(sample_df, seed=7)
    assert len(X_train) + len(X_val) + len(X_test) == len(sample_df)


def test_data_hash_stable_and_sensitive():
    a = make_creditcard_like(n=500, seed=1)
    b = make_creditcard_like(n=500, seed=1)
    c = make_creditcard_like(n=500, seed=2)
    assert data_hash(a) == data_hash(b)
    assert data_hash(a) != data_hash(c)
