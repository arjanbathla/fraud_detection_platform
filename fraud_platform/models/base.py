"""Common model interface.

All three models (Isolation Forest, XGBoost, Autoencoder) implement AnomalyModel, so the
training, evaluation and serving code treats them identically. The key shared contract is
score(): it always returns a 1-D array where HIGHER = MORE ANOMALOUS, regardless of whether
the underlying model is supervised or unsupervised.

Each model owns a fitted preprocessing Pipeline. fit() fits the pipeline on the raw training
frame, and score()/predict() transform raw frames through that same pipeline first. save()
persists the model and its pipeline together so inference is self-contained.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from fraud_platform.pipeline.features import build_pipeline


class AnomalyModel(ABC):
    name: str = "base"
    supervised: bool = False

    def __init__(self) -> None:
        # each model gets its own preprocessing pipeline, fitted in fit()
        self.pipeline = build_pipeline()
        self.threshold: float = 0.5  # decision threshold on the anomaly score

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: np.ndarray | None = None) -> "AnomalyModel":
        """Fit the preprocessing pipeline and the underlying model on raw data."""

    @abstractmethod
    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Return anomaly scores (higher = more anomalous), one per input row."""

    def predict(self, X: pd.DataFrame, threshold: float | None = None) -> np.ndarray:
        """Return 0/1 labels by thresholding the anomaly score."""
        t = self.threshold if threshold is None else threshold
        return (self.score(X) >= t).astype(int)

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist model + pipeline to a single artifact file."""

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "AnomalyModel":
        """Load a previously saved model."""

    def _transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.transform(X)
