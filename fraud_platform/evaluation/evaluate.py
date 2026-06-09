"""Evaluation: metrics, threshold selection, comparison table.

Reports precision, recall, F1 (which depend on the decision threshold) plus ROC-AUC and
PR-AUC (which don't). On imbalanced data ROC-AUC looks flattering, so PR-AUC and recall are
the metrics to trust — they're about catching the rare fraud class.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRIC_KEYS = ["precision", "recall", "f1", "roc_auc", "pr_auc"]


def choose_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Pick the score threshold that maximises F1 (documented default).

    A real deployment would set this from a cost/precision target, but max-F1 is a
    reasonable, reproducible default for comparing models on the same footing.
    """
    prec, rec, thr = precision_recall_curve(y_true, scores)
    # precision_recall_curve returns one fewer threshold than prec/rec points
    f1 = np.where((prec[:-1] + rec[:-1]) > 0,
                  2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12), 0.0)
    if len(thr) == 0:
        return 0.5
    return float(thr[int(np.argmax(f1))])


def evaluate_scores(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """Compute the metric dict from raw scores + a threshold."""
    y_true = np.asarray(y_true).astype(int)
    preds = (scores >= threshold).astype(int)
    return {
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "threshold": float(threshold),
        "n_flagged": int(preds.sum()),
        "n_true_fraud": int(y_true.sum()),
    }


def evaluate_model(model, X_test, y_test) -> dict:
    """Score a fitted model on the test set using its own threshold."""
    scores = model.score(X_test)
    return evaluate_scores(np.asarray(y_test), scores, model.threshold)


def comparison_table(results: dict[str, dict]) -> str:
    """Render {model_name: metrics} as a fixed-width table."""
    header = f"{'model':<18}" + "".join(f"{k:>10}" for k in METRIC_KEYS)
    lines = [header, "-" * len(header)]
    for name, m in results.items():
        row = f"{name:<18}" + "".join(f"{m[k]:>10.4f}" for k in METRIC_KEYS)
        lines.append(row)
    return "\n".join(lines)


def save_results(results: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
