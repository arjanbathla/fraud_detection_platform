"""FastAPI serving layer.

Loads the champion model (model + its preprocessing pipeline) once at startup and scores a
single transaction on /predict. Every request measures wall-clock inference time and returns
it as `inference_ms`, and the server also logs it — so the <100ms claim can be verified, not
assumed. Default model type is XGBoost (configurable via the MODEL_TYPE env var).

Run:
    uvicorn fraud_platform.serving.api:app --reload
Endpoints:
    GET  /health      — liveness + whether a model is loaded
    GET  /model-info  — champion version + metrics + threshold
    POST /predict     — score one transaction
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from fraud_platform.config import (
    CATEGORICAL_FEATURES,
    LATENCY_BUDGET_MS,
    NUMERIC_FEATURES,
)
from fraud_platform.registry.registry import ModelRegistry

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("serving")

MODEL_TYPE = os.environ.get("MODEL_TYPE", "xgboost")

# loaded at startup; kept in module state so it's loaded once, not per request
_STATE: dict = {"model": None, "version": None, "meta": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_champion()  # load the champion once when the server boots
    yield


app = FastAPI(title="Fraud Detection API (SYNTHETIC data)", version="1.0",
              lifespan=lifespan)


class Transaction(BaseModel):
    amount: float = Field(..., ge=0)
    account_age_days: float = Field(..., ge=0)
    num_tx_last_24h: float = Field(..., ge=0)
    avg_amount_last_30d: float = Field(..., ge=0)
    distance_from_home_km: float = Field(..., ge=0)
    hour: int = Field(..., ge=0, le=23)
    merchant_category: str
    transaction_type: str
    device_type: str
    country: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "amount": 920.50, "account_age_days": 45, "num_tx_last_24h": 11,
                "avg_amount_last_30d": 80.0, "distance_from_home_km": 240.0, "hour": 3,
                "merchant_category": "electronics", "transaction_type": "transfer",
                "device_type": "web", "country": "NG",
            }
        }
    }


class PredictionResponse(BaseModel):
    is_fraud: int
    anomaly_score: float
    threshold: float
    model_type: str
    model_version: int
    inference_ms: float
    within_latency_budget: bool


def _load_champion() -> None:
    registry = ModelRegistry()
    model = registry.load_champion(MODEL_TYPE)
    if model is None:
        log.warning("no champion registered for '%s' — run training first", MODEL_TYPE)
        return
    _STATE["model"] = model
    _STATE["version"] = registry.champion_version(MODEL_TYPE)
    _STATE["meta"] = registry.champion_metadata(MODEL_TYPE)
    log.info("loaded %s champion v%s", MODEL_TYPE, _STATE["version"])


@app.get("/")
def root() -> dict:
    return {"service": "fraud-detection (SYNTHETIC data)",
            "docs": "/docs",
            "endpoints": ["/health", "/model-info", "/predict"]}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _STATE["model"] is not None,
            "model_type": MODEL_TYPE}


@app.get("/model-info")
def model_info() -> dict:
    if _STATE["model"] is None:
        raise HTTPException(503, "no model loaded — run `python -m fraud_platform.train`")
    return {"model_type": MODEL_TYPE, "version": _STATE["version"],
            "metadata": _STATE["meta"]}


@app.post("/predict", response_model=PredictionResponse)
def predict(tx: Transaction) -> PredictionResponse:
    model = _STATE["model"]
    if model is None:
        raise HTTPException(503, "no model loaded — run `python -m fraud_platform.train`")

    # build a one-row dataframe with the exact raw columns the pipeline expects
    row = {c: getattr(tx, c) for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES}
    df = pd.DataFrame([row])

    start = time.perf_counter()
    score = float(model.score(df)[0])
    pred = int(score >= model.threshold)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    within = elapsed_ms < LATENCY_BUDGET_MS
    log.info("predict: type=%s score=%.4f pred=%d inference_ms=%.2f within_budget=%s",
             MODEL_TYPE, score, pred, elapsed_ms, within)

    return PredictionResponse(
        is_fraud=pred,
        anomaly_score=score,
        threshold=float(model.threshold),
        model_type=MODEL_TYPE,
        model_version=int(_STATE["version"]),
        inference_ms=round(elapsed_ms, 3),
        within_latency_budget=within,
    )
