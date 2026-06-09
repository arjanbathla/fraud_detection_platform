"""Tests for the preprocessing pipeline."""
import numpy as np

from fraud_platform.config import TARGET
from fraud_platform.pipeline.features import (
    ENGINEERED_NUMERIC,
    build_pipeline,
    engineer_features,
)


def test_engineered_columns_added(sample_df):
    out = engineer_features(sample_df)
    for col in ENGINEERED_NUMERIC:
        assert col in out.columns
    # amount_log should be the log1p of Amount
    assert np.allclose(out["amount_log"], np.log1p(sample_df["Amount"].clip(lower=0)))


def test_pipeline_fit_transform_shape(sample_df):
    X = sample_df.drop(columns=[TARGET])
    pipe = build_pipeline()
    Xt = pipe.fit_transform(X)
    assert Xt.shape[0] == len(X)
    # 28 PCA + Time + Amount + amount_log = 31 columns
    assert Xt.shape[1] == 31
    assert not np.isnan(Xt).any()


def test_transform_is_deterministic(sample_df):
    X = sample_df.drop(columns=[TARGET])
    pipe = build_pipeline().fit(X)
    assert np.allclose(pipe.transform(X), pipe.transform(X))


def test_scaling_centres_pca_block(sample_df):
    # StandardScaler on the PCA block should give ~zero mean, ~unit std on training data
    X = sample_df.drop(columns=[TARGET])
    pipe = build_pipeline().fit(X)
    Xt = pipe.transform(X)
    # last 28 columns are the PCA block (raw block of 3 comes first)
    pca_block = Xt[:, 3:]
    assert np.abs(pca_block.mean(axis=0)).max() < 1e-6


def test_output_width_stable_for_single_row(sample_df):
    X = sample_df.drop(columns=[TARGET])
    pipe = build_pipeline().fit(X)
    one = pipe.transform(X.iloc[[0]])
    assert one.shape[1] == pipe.transform(X).shape[1]
