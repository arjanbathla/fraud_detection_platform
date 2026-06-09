"""Tests for the FastAPI serving layer.

We register a small model into the default registry, then drive the app with TestClient so
startup loads a real champion. The /predict payload is a full creditcard-style record
(Time, V1..V28, Amount) taken from a fixture row.
"""
import pytest
from fastapi.testclient import TestClient

from fraud_platform.config import NUMERIC_FEATURES
from fraud_platform.registry.registry import ModelRegistry
from tests.conftest import make_creditcard_like


@pytest.fixture(scope="module")
def client(fitted_xgb, tmp_path_factory):
    # isolate the registry to a temp dir so tests never write into the real registry_store/
    import os
    reg_dir = tmp_path_factory.mktemp("registry")
    prev = os.environ.get("FRAUD_REGISTRY_DIR")
    os.environ["FRAUD_REGISTRY_DIR"] = str(reg_dir)
    try:
        ModelRegistry().register(fitted_xgb, {"pr_auc": 0.7}, make_champion=True)
        from fraud_platform.serving import api
        api._load_champion()
        with TestClient(api.app) as c:
            yield c
    finally:
        if prev is None:
            os.environ.pop("FRAUD_REGISTRY_DIR", None)
        else:
            os.environ["FRAUD_REGISTRY_DIR"] = prev


@pytest.fixture(scope="module")
def sample_payload():
    row = make_creditcard_like(n=200, seed=3).iloc[0]
    return {c: float(row[c]) for c in NUMERIC_FEATURES}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["model_loaded"] is True


def test_model_info(client):
    r = client.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    assert body["model_type"] == "xgboost"
    assert body["version"] >= 1


def test_predict_schema_and_latency(client, sample_payload):
    r = client.post("/predict", json=sample_payload)
    assert r.status_code == 200
    body = r.json()
    assert body["is_fraud"] in (0, 1)
    assert 0.0 <= body["anomaly_score"] <= 1.0
    assert body["model_type"] == "xgboost"
    assert body["inference_ms"] > 0
    assert isinstance(body["within_latency_budget"], bool)


def test_predict_missing_field(client, sample_payload):
    bad = dict(sample_payload)
    del bad["Amount"]
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_rejects_non_numeric(client, sample_payload):
    bad = dict(sample_payload)
    bad["V1"] = "not_a_number"
    r = client.post("/predict", json=bad)
    assert r.status_code == 422
