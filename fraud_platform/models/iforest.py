"""Isolation Forest — unsupervised anomaly detector.

Isolation Forest isolates points by random splits; anomalies need fewer splits. sklearn's
score_samples returns higher values for inliers, so we negate it to match our convention
(higher = more anomalous).
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from fraud_platform.config import RANDOM_SEED
from fraud_platform.models.base import AnomalyModel


class IForestModel(AnomalyModel):
    name = "isolation_forest"
    supervised = False

    def __init__(self, contamination: float = 0.01, n_estimators: int = 200):
        super().__init__()
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )

    def fit(self, X: pd.DataFrame, y: np.ndarray | None = None) -> "IForestModel":
        Xt = self.pipeline.fit_transform(X)  # y ignored — unsupervised
        self.model.fit(Xt)
        # default threshold from the model's own offset (its internal decision boundary)
        scores = self.score(X)
        self.threshold = float(np.quantile(scores, 0.99))
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        Xt = self._transform(X)
        # score_samples: higher = more normal. negate so higher = more anomalous.
        return -self.model.score_samples(Xt)

    def save(self, path: str) -> None:
        joblib.dump({"model": self.model, "pipeline": self.pipeline,
                     "threshold": self.threshold}, path)

    @classmethod
    def load(cls, path: str) -> "IForestModel":
        blob = joblib.load(path)
        obj = cls()
        obj.model = blob["model"]
        obj.pipeline = blob["pipeline"]
        obj.threshold = blob["threshold"]
        return obj
