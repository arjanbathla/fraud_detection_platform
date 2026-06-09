"""Tests for the evaluation/metrics module."""
import numpy as np

from fraud_platform.evaluation.evaluate import (
    METRIC_KEYS,
    choose_threshold,
    comparison_table,
    evaluate_scores,
)


def test_perfect_scores_give_perfect_metrics():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    m = evaluate_scores(y, scores, threshold=0.5)
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["roc_auc"] == 1.0


def test_choose_threshold_separates_classes():
    y = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    t = choose_threshold(y, scores)
    preds = (scores >= t).astype(int)
    assert (preds == y).all()


def test_metrics_keys_present():
    y = np.array([0, 1, 0, 1])
    scores = np.array([0.2, 0.6, 0.4, 0.7])
    m = evaluate_scores(y, scores, 0.5)
    for k in METRIC_KEYS:
        assert k in m


def test_comparison_table_contains_model_names():
    results = {
        "xgboost": {k: 0.9 for k in METRIC_KEYS},
        "isolation_forest": {k: 0.5 for k in METRIC_KEYS},
    }
    table = comparison_table(results)
    assert "xgboost" in table
    assert "isolation_forest" in table
