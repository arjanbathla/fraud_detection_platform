"""Tests for the versioned model registry + promotion gate."""
from fraud_platform.registry.registry import ModelRegistry
from fraud_platform.retrain import retrain_model


def test_register_increments_version(fitted_xgb, tmp_path):
    reg = ModelRegistry(root=tmp_path / "reg")
    v1 = reg.register(fitted_xgb, {"pr_auc": 0.7}, data_hash="abc")
    v2 = reg.register(fitted_xgb, {"pr_auc": 0.8}, data_hash="abc")
    assert v1 == 1 and v2 == 2


def test_first_version_becomes_champion(fitted_xgb, tmp_path):
    reg = ModelRegistry(root=tmp_path / "reg")
    reg.register(fitted_xgb, {"pr_auc": 0.7})
    assert reg.champion_version("xgboost") == 1


def test_second_version_does_not_auto_promote(fitted_xgb, tmp_path):
    reg = ModelRegistry(root=tmp_path / "reg")
    reg.register(fitted_xgb, {"pr_auc": 0.7})
    reg.register(fitted_xgb, {"pr_auc": 0.9})  # no make_champion
    assert reg.champion_version("xgboost") == 1


def test_explicit_promote(fitted_xgb, tmp_path):
    reg = ModelRegistry(root=tmp_path / "reg")
    reg.register(fitted_xgb, {"pr_auc": 0.7})
    reg.register(fitted_xgb, {"pr_auc": 0.9})
    reg.promote("xgboost", 2)
    assert reg.champion_version("xgboost") == 2


def test_metadata_persisted(fitted_xgb, tmp_path):
    reg = ModelRegistry(root=tmp_path / "reg")
    reg.register(fitted_xgb, {"pr_auc": 0.71, "recall": 0.6}, data_hash="hash123")
    meta = reg.get_metadata("xgboost", 1)
    assert meta["model_type"] == "xgboost"
    assert meta["metrics"]["pr_auc"] == 0.71
    assert meta["data_hash"] == "hash123"
    assert "utc_timestamp" in meta


def test_load_champion_scores(fitted_xgb, tmp_path, split_data):
    reg = ModelRegistry(root=tmp_path / "reg")
    reg.register(fitted_xgb, {"pr_auc": 0.7})
    loaded = reg.load_champion("xgboost")
    X_test = split_data[4]
    scores = loaded.score(X_test)
    assert len(scores) == len(X_test)


def test_promotion_gate_keeps_better_champion(split_data, tmp_path):
    """A candidate with a worse PR-AUC must NOT replace the champion."""
    X_train, y_train, X_val, y_val, X_test, y_test = split_data
    reg = ModelRegistry(root=tmp_path / "reg")

    # seed a champion with an artificially high stored metric
    from fraud_platform.models import make_model
    from fraud_platform.evaluation.evaluate import choose_threshold
    m = make_model("xgboost", n_estimators=80).fit(X_train, y_train)
    m.threshold = choose_threshold(y_val, m.score(X_val))
    reg.register(m, {"pr_auc": 0.999}, make_champion=True)

    # retrain a candidate; its real PR-AUC will be < 0.999, so it must not promote
    res = retrain_model("xgboost", reg, X_train, y_train, X_val, y_val,
                        X_test, y_test, dh="h", metric="pr_auc")
    assert res["promoted"] is False
    assert reg.champion_version("xgboost") == 1
