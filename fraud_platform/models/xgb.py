"""XGBoost — supervised classifier.

Uses the labels. score() returns the predicted fraud probability, which already matches our
"higher = more anomalous" convention. scale_pos_weight handles the class imbalance.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from fraud_platform.config import RANDOM_SEED
from fraud_platform.models.base import AnomalyModel


class XGBModel(AnomalyModel):
    name = "xgboost"
    supervised = True

    def __init__(self, n_estimators: int = 300, max_depth: int = 5,
                 learning_rate: float = 0.1):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.model: XGBClassifier | None = None
        self.threshold = 0.5

    def fit(self, X: pd.DataFrame, y: np.ndarray | None = None) -> "XGBModel":
        if y is None:
            raise ValueError("XGBModel is supervised and needs labels (y)")
        y = np.asarray(y)
        Xt = self.pipeline.fit_transform(X)
        # weight the rare positive class so recall doesn't collapse
        pos = max(int(y.sum()), 1)
        neg = max(len(y) - pos, 1)
        self.model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="aucpr",
            scale_pos_weight=neg / pos,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        self.model.fit(Xt, y)
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        Xt = self._transform(X)
        return self.model.predict_proba(Xt)[:, 1]

    def save(self, path: str) -> None:
        joblib.dump({"model": self.model, "pipeline": self.pipeline,
                     "threshold": self.threshold}, path)

    @classmethod
    def load(cls, path: str) -> "XGBModel":
        blob = joblib.load(path)
        obj = cls()
        obj.model = blob["model"]
        obj.pipeline = blob["pipeline"]
        obj.threshold = blob["threshold"]
        return obj
