"""Tests for the common model interface — all three models must conform."""
import numpy as np
import pytest

from fraud_platform.config import TARGET
from fraud_platform.models import MODEL_TYPES, make_model


@pytest.fixture(scope="module")
def small_split(sample_df):
    from fraud_platform.data.loader import split
    return split(sample_df, seed=7)


@pytest.mark.parametrize("model_type", list(MODEL_TYPES))
def test_fit_score_predict_contract(model_type, small_split):
    X_train, y_train, X_val, y_val, X_test, y_test = small_split
    model = make_model(model_type)
    model.fit(X_train, y_train)

    scores = model.score(X_test)
    assert scores.shape == (len(X_test),)
    assert np.isfinite(scores).all()

    preds = model.predict(X_test)
    assert set(np.unique(preds)).issubset({0, 1})
    assert preds.shape == (len(X_test),)


@pytest.mark.parametrize("model_type", list(MODEL_TYPES))
def test_higher_threshold_flags_fewer(model_type, small_split):
    X_train, y_train, X_val, y_val, X_test, y_test = small_split
    model = make_model(model_type).fit(X_train, y_train)
    scores = model.score(X_test)
    low = (scores >= np.quantile(scores, 0.5)).sum()
    high = (scores >= np.quantile(scores, 0.9)).sum()
    assert high <= low  # raising the bar can only flag the same or fewer


@pytest.mark.parametrize("model_type", list(MODEL_TYPES))
def test_save_and_load_roundtrip(model_type, small_split, tmp_path):
    from fraud_platform.models import ARTIFACT_EXT, load_model
    X_train, y_train, X_val, y_val, X_test, y_test = small_split
    model = make_model(model_type).fit(X_train, y_train)
    before = model.score(X_test)

    path = tmp_path / f"model.{ARTIFACT_EXT[model_type]}"
    model.save(str(path))
    reloaded = load_model(model_type, str(path))
    after = reloaded.score(X_test)

    assert np.allclose(before, after, atol=1e-5)
    assert reloaded.threshold == model.threshold


def test_xgb_requires_labels(small_split):
    X_train, y_train, *_ = small_split
    model = make_model("xgboost")
    with pytest.raises(ValueError):
        model.fit(X_train, y=None)


def test_unknown_model_type_raises():
    with pytest.raises(KeyError):
        make_model("not_a_model")
