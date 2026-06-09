"""Data drift detection: Population Stability Index (PSI) + KS test per feature.

You compare a CURRENT batch of incoming data against the REFERENCE distribution the model
was trained on. Two complementary signals per numeric feature:

PSI — measures how much a feature's distribution has shifted, by binning the reference into
quantiles and comparing the proportion of current data falling in each bin:
    PSI = sum( (cur% - ref%) * ln(cur% / ref%) )
  Interpretation (industry-standard bands):
    PSI < 0.10           no significant shift
    0.10 <= PSI < 0.25   moderate shift — worth watching
    PSI >= 0.25          significant shift — investigate / consider retraining

KS test — two-sample Kolmogorov–Smirnov test. The null hypothesis is "same distribution".
A p-value < 0.05 means we reject that: the distributions differ at the 5% significance level.
KS is sensitive to large samples (tiny shifts become "significant"), which is exactly why we
pair it with PSI's effect-size view rather than relying on the p-value alone.

A feature is flagged if PSI >= 0.25 OR KS p-value < 0.05. The overall batch is "drifting" if
any feature is flagged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from fraud_platform.config import KS_PVALUE, NUMERIC_FEATURES, PSI_MODERATE, PSI_SIGNIFICANT


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between a reference and current sample (numeric)."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    # bin edges from reference quantiles so bins are populated
    quantiles = np.linspace(0, 100, bins + 1)
    edges = np.unique(np.percentile(reference, quantiles))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)
    # convert to proportions, floor at a small value to avoid div/0 and log(0)
    ref_pct = np.clip(ref_counts / max(ref_counts.sum(), 1), 1e-6, None)
    cur_pct = np.clip(cur_counts / max(cur_counts.sum(), 1), 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def psi_band(value: float) -> str:
    if value < PSI_MODERATE:
        return "none"
    if value < PSI_SIGNIFICANT:
        return "moderate"
    return "significant"


def compute_drift(reference: pd.DataFrame, current: pd.DataFrame,
                  features: list[str] | None = None) -> dict:
    """Per-feature PSI + KS results plus an overall drift flag."""
    features = features or [c for c in NUMERIC_FEATURES if c in reference.columns]
    per_feature = {}
    any_drift = False
    for col in features:
        ref = reference[col].dropna().values
        cur = current[col].dropna().values
        p = psi(ref, cur)
        ks_stat, ks_p = ks_2samp(ref, cur)
        flagged = (p >= PSI_SIGNIFICANT) or (ks_p < KS_PVALUE)
        any_drift = any_drift or flagged
        per_feature[col] = {
            "psi": round(p, 4),
            "psi_band": psi_band(p),
            "ks_statistic": round(float(ks_stat), 4),
            "ks_pvalue": round(float(ks_p), 6),
            "flagged": bool(flagged),
        }
    return {
        "drift_detected": bool(any_drift),
        "n_features_flagged": sum(f["flagged"] for f in per_feature.values()),
        "features": per_feature,
    }
