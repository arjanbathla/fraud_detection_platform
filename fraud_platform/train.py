"""Train all three models, evaluate on a held-out test set, register them.

    python -m fraud_platform.train

Pipeline for each model:
  fit on train  ->  pick threshold on validation (max F1)  ->  evaluate on test
Then every model is written to the registry as a new version. The first version of each
type becomes the champion automatically; later runs add versions without auto-promoting
(promotion is the retrain script's job).
"""
from __future__ import annotations

import argparse

from fraud_platform.config import ARTIFACTS_DIR
from fraud_platform.data.loader import data_hash, load_or_generate, split
from fraud_platform.evaluation.evaluate import (
    choose_threshold,
    comparison_table,
    evaluate_model,
    save_results,
)
from fraud_platform.models import MODEL_TYPES, make_model
from fraud_platform.registry.registry import ModelRegistry


def train_one(model_type: str, X_train, y_train, X_val, y_val):
    """Fit a model, set its decision threshold from the validation split."""
    model = make_model(model_type)
    model.fit(X_train, y_train)
    val_scores = model.score(X_val)
    model.threshold = choose_threshold(y_val, val_scores)
    return model


def run(rows: int = 50_000, fraud_frac: float = 0.01, register: bool = True) -> dict:
    df = load_or_generate(rows=rows, fraud_frac=fraud_frac)
    X_train, y_train, X_val, y_val, X_test, y_test = split(df)
    dh = data_hash(df)

    registry = ModelRegistry()
    results: dict[str, dict] = {}

    for model_type in MODEL_TYPES:
        print(f"\n=== training {model_type} ===")
        model = train_one(model_type, X_train, y_train, X_val, y_val)
        metrics = evaluate_model(model, X_test, y_test)
        results[model_type] = metrics
        print(f"{model_type}: PR-AUC={metrics['pr_auc']:.4f} "
              f"recall={metrics['recall']:.4f} precision={metrics['precision']:.4f}")
        if register:
            v = registry.register(model, metrics, data_hash=dh)
            print(f"registered {model_type} as v{v}")

    print("\n[SYNTHETIC DATA] model comparison on held-out test set:")
    print(comparison_table(results))
    save_results({"data_hash": dh, "models": results},
                 ARTIFACTS_DIR / "evaluation_results.json")
    print(f"\nsaved metrics to {ARTIFACTS_DIR / 'evaluation_results.json'}")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Train + evaluate + register all models")
    ap.add_argument("--rows", type=int, default=50_000)
    ap.add_argument("--fraud-frac", type=float, default=0.01)
    ap.add_argument("--no-register", action="store_true")
    args = ap.parse_args()
    run(rows=args.rows, fraud_frac=args.fraud_frac, register=not args.no_register)


if __name__ == "__main__":
    main()
