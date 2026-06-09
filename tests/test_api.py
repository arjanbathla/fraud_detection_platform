"""Tests for the FastAPI serving layer.

We register a small model into the default registry, then drive the app with TestClient so
startup loads a real champion. The /predict assertions check the response schema, that a
score comes back, and that the latency field is present and sane.
"""
import pytest
from fastapi.testclient import TestClient

from fraud_platform.registry.registry import ModelRegistry


@pytest.fixture(scope="module")
def client(fitted_xgb):
    # make sure there is an xgboost champion for the API to load
    reg = ModelRegistry()
    if reg.champion_version("xgboost") is None:
        reg.register(fitted_xgb, {"pr_auc": 0.7}, make_champion=True)

    from fraud_platform.serving import api
    api._load_champion()  # reload state for this process
    with TestClient(api.app) as c:
        yield c


SAMPLE = {
    "amount": 920.50, "account_age_days": 45, "num_tx_last_24h": 11,
    "avg_amount_last_30d": 80.0, "distance_from_home_km": 240.0, "hour": 3,
    "merchant_category": "electronics", "transaction_type": "transfer",
    "device_type": "web", "country": "NG",
}


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


def test_predict_schema_and_latency(client):
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert body["is_fraud"] in (0, 1)
    assert 0.0 <= body["anomaly_score"] <= 1.0
    assert body["model_type"] == "xgboost"
    assert body["inference_ms"] > 0
    assert isinstance(body["within_latency_budget"], bool)


def test_predict_rejects_bad_input(client):
    bad = dict(SAMPLE)
    bad["hour"] = 99  # out of the 0-23 range -> validation error
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_missing_field(client):
    bad = dict(SAMPLE)
    del bad["amount"]
    r = client.post("/predict", json=bad)
    assert r.status_code == 422
