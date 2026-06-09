"""Shared pytest fixtures.

A small synthetic dataset and a fitted XGBoost model are built once per session and reused,
so the tests run fast. Everything is in-memory / tmp — no dependency on a pre-trained
registry existing on disk.
"""
import numpy as np
import pytest

from fraud_platform.data.generate import generate
from fraud_platform.data.loader import split
from fraud_platform.models import make_model


@pytest.fixture(scope="session")
def synth_df():
    # small but enough fraud rows for stratified splits
    return generate(n_rows=3000, fraud_frac=0.05, seed=7)


@pytest.fixture(scope="session")
def split_data(synth_df):
    return split(synth_df, seed=7)


@pytest.fixture(scope="session")
def fitted_xgb(split_data):
    X_train, y_train, X_val, y_val, X_test, y_test = split_data
    model = make_model("xgboost", n_estimators=80)
    model.fit(X_train, y_train)
    from fraud_platform.evaluation.evaluate import choose_threshold
    model.threshold = choose_threshold(y_val, model.score(X_val))
    return model
