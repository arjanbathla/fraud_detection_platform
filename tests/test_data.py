"""Tests for the synthetic data generator."""
import numpy as np
import pytest

from fraud_platform.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET
from fraud_platform.data.generate import generate


def test_row_count_and_columns():
    df = generate(n_rows=2000, fraud_frac=0.03, seed=1)
    assert len(df) == 2000
    for col in NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]:
        assert col in df.columns


def test_fraud_fraction_close_to_target():
    df = generate(n_rows=10000, fraud_frac=0.02, seed=1)
    rate = df[TARGET].mean()
    # exact count is round(n*frac), so rate should be essentially exact
    assert abs(rate - 0.02) < 0.002


def test_labels_are_binary():
    df = generate(n_rows=1000, fraud_frac=0.1, seed=1)
    assert set(df[TARGET].unique()).issubset({0, 1})
    assert df[TARGET].sum() > 0


def test_reproducible_with_seed():
    a = generate(n_rows=1000, fraud_frac=0.05, seed=42)
    b = generate(n_rows=1000, fraud_frac=0.05, seed=42)
    # compare the model-facing numeric columns
    assert np.allclose(a[NUMERIC_FEATURES].values, b[NUMERIC_FEATURES].values)


def test_different_seeds_differ():
    a = generate(n_rows=1000, fraud_frac=0.05, seed=1)
    b = generate(n_rows=1000, fraud_frac=0.05, seed=2)
    assert not np.allclose(a["amount"].values, b["amount"].values)


def test_invalid_fraud_frac_rejected():
    with pytest.raises(ValueError):
        generate(n_rows=100, fraud_frac=0.9)


def test_no_missing_in_required_columns():
    df = generate(n_rows=1000, fraud_frac=0.05, seed=1)
    assert df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].isna().sum().sum() == 0
