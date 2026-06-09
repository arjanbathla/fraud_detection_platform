"""Tests for drift detection (PSI + KS)."""
import numpy as np
import pandas as pd

from fraud_platform.config import NUMERIC_FEATURES
from fraud_platform.data.generate import generate
from fraud_platform.monitoring.drift import compute_drift, psi, psi_band


def test_psi_zero_for_identical():
    x = np.random.default_rng(0).normal(size=5000)
    assert psi(x, x) < 1e-6


def test_psi_grows_with_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    small = rng.normal(0.2, 1, 5000)
    big = rng.normal(3, 1, 5000)
    assert psi(ref, small) < psi(ref, big)


def test_psi_band_labels():
    assert psi_band(0.05) == "none"
    assert psi_band(0.15) == "moderate"
    assert psi_band(0.40) == "significant"


def test_no_drift_on_same_distribution():
    a = generate(n_rows=4000, fraud_frac=0.02, seed=1)
    b = generate(n_rows=4000, fraud_frac=0.02, seed=2)  # same process, different draw
    result = compute_drift(a, b)
    # same generating process — should not flag significant drift on most features
    assert result["n_features_flagged"] <= 1


def test_drift_detected_on_shifted_feature():
    a = generate(n_rows=4000, fraud_frac=0.02, seed=1)
    b = a.copy()
    b["amount"] = b["amount"] * 5 + 500  # inject a large shift
    result = compute_drift(b_reference := a, current := b)
    assert result["drift_detected"] is True
    assert result["features"]["amount"]["flagged"] is True


def test_result_structure():
    a = generate(n_rows=2000, fraud_frac=0.02, seed=1)
    result = compute_drift(a, a)
    assert set(result.keys()) == {"drift_detected", "n_features_flagged", "features"}
    for col in [c for c in NUMERIC_FEATURES if c in a.columns]:
        f = result["features"][col]
        assert {"psi", "psi_band", "ks_statistic", "ks_pvalue", "flagged"} <= set(f)
