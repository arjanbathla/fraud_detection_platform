"""Tests for the preprocessing pipeline."""
import numpy as np

from fraud_platform.config import TARGET
from fraud_platform.pipeline.features import (
    ENGINEERED_NUMERIC,
    build_pipeline,
    engineer_features,
)


def test_engineered_columns_added(synth_df):
    out = engineer_features(synth_df)
    for col in ENGINEERED_NUMERIC:
        assert col in out.columns


def test_pipeline_fit_transform_shape(synth_df):
    X = synth_df.drop(columns=[TARGET])
    pipe = build_pipeline()
    Xt = pipe.fit_transform(X)
    assert Xt.shape[0] == len(X)
    assert Xt.shape[1] > 0
    assert not np.isnan(Xt).any()


def test_transform_is_deterministic(synth_df):
    X = synth_df.drop(columns=[TARGET])
    pipe = build_pipeline().fit(X)
    a = pipe.transform(X)
    b = pipe.transform(X)
    assert np.allclose(a, b)


def test_handles_unseen_category(synth_df):
    X = synth_df.drop(columns=[TARGET])
    pipe = build_pipeline().fit(X)
    novel = X.iloc[[0]].copy()
    novel["country"] = "ZZ"  # never seen at fit time
    # OneHotEncoder(handle_unknown="ignore") must not raise
    Xt = pipe.transform(novel)
    assert Xt.shape[0] == 1


def test_output_width_stable_for_single_row(synth_df):
    X = synth_df.drop(columns=[TARGET])
    pipe = build_pipeline().fit(X)
    full = pipe.transform(X)
    one = pipe.transform(X.iloc[[0]])
    assert one.shape[1] == full.shape[1]
