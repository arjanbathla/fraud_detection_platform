"""Tests for the real-CSV loader and the imbalance-aware split."""
import numpy as np
import pytest

from fraud_platform.config import NUMERIC_FEATURES, TARGET
from fraud_platform.data.loader import data_hash, load_data, split
from tests.conftest import make_creditcard_like


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_data(tmp_path / "nope.csv")


def test_load_reads_csv_and_checks_schema(tmp_path, sample_df):
    p = tmp_path / "creditcard.csv"
    sample_df.to_csv(p, index=False)
    df = load_data(p)
    assert len(df) == len(sample_df)
    for col in NUMERIC_FEATURES + [TARGET]:
        assert col in df.columns


def test_load_rejects_bad_schema(tmp_path, sample_df):
    bad = sample_df.drop(columns=["V1"])
    p = tmp_path / "bad.csv"
    bad.to_csv(p, index=False)
    with pytest.raises(ValueError):
        load_data(p)


def test_split_is_stratified(sample_df):
    X_train, y_train, X_val, y_val, X_test, y_test = split(sample_df, seed=7)
    overall = sample_df[TARGET].mean()
    # each fold should keep roughly the same fraud rate (stratified)
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
