"""Automated retraining with a promotion gate.

    python -m fraud_platform.retrain                      # retrain on default data
    python -m fraud_platform.retrain --data new_batch.csv # retrain on a fresh batch
    python -m fraud_platform.retrain --model xgboost      # just one model type

WHAT TRIGGERS A RETRAIN (be accurate about this on a CV):
  This is a MANUAL command. It is designed to be run on a schedule (e.g. a nightly cron job
  or CI workflow) or by hand when a new labelled batch arrives or drift is detected. It is
  NOT event-driven / streaming auto-retraining — nothing watches data and fires this for you.
  Example cron (daily 02:00):  0 2 * * *  cd /app && python -m fraud_platform.retrain

PROMOTION GATE:
  For each model type we train a candidate, evaluate it on the held-out test set, and compare
  its PROMOTION_METRIC (PR-AUC) against the current champion's stored metric. The candidate is
  always registered as a new version (immutable history), but it only becomes champion if it
  is strictly better. This prevents a worse model silently replacing a good one.
"""
from __future__ import annotations

import argparse

from fraud_platform.config import PROMOTION_METRIC
from fraud_platform.data.loader import data_hash, load_data, split
from fraud_platform.evaluation.evaluate import evaluate_model
from fraud_platform.models import MODEL_TYPES
from fraud_platform.registry.registry import ModelRegistry
from fraud_platform.train import train_one


def retrain_model(model_type: str, registry: ModelRegistry,
                  X_train, y_train, X_val, y_val, X_test, y_test,
                  dh: str, metric: str = PROMOTION_METRIC) -> dict:
    """Train a candidate, register it, promote only if it beats the champion on `metric`."""
    champ_meta = registry.champion_metadata(model_type)
    champ_score = champ_meta["metrics"][metric] if champ_meta else None

    candidate = train_one(model_type, X_train, y_train, X_val, y_val)
    cand_metrics = evaluate_model(candidate, X_test, y_test)
    cand_score = cand_metrics[metric]

    promote = champ_score is None or cand_score > champ_score
    version = registry.register(candidate, cand_metrics, data_hash=dh,
                                make_champion=promote)

    result = {
        "model_type": model_type,
        "candidate_version": version,
        "metric": metric,
        "candidate_score": cand_score,
        "champion_score": champ_score,
        "promoted": promote,
    }
    if promote:
        msg = (f"PROMOTED {model_type} v{version}: {metric} {cand_score:.4f}"
               + (f" > champion {champ_score:.4f}" if champ_score is not None
                  else " (first version)"))
    else:
        msg = (f"kept champion for {model_type}: candidate {metric} {cand_score:.4f} "
               f"did NOT beat {champ_score:.4f} (registered v{version} but not promoted)")
    print(msg)
    return result


def run(data_path: str | None = None, model: str | None = None,
        metric: str = PROMOTION_METRIC) -> list[dict]:
    df = load_data(data_path) if data_path else load_data()
    X_train, y_train, X_val, y_val, X_test, y_test = split(df)
    dh = data_hash(df)
    registry = ModelRegistry()

    types = [model] if model else list(MODEL_TYPES)
    results = []
    for mt in types:
        results.append(retrain_model(mt, registry, X_train, y_train, X_val, y_val,
                                      X_test, y_test, dh, metric))
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Retrain + gated promotion")
    ap.add_argument("--data", type=str, default=None, help="path to a new data CSV")
    ap.add_argument("--model", type=str, default=None, help="single model type to retrain")
    ap.add_argument("--metric", type=str, default=PROMOTION_METRIC)
    args = ap.parse_args()
    run(data_path=args.data, model=args.model, metric=args.metric)


if __name__ == "__main__":
    main()
