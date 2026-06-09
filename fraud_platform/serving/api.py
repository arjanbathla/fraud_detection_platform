"""FastAPI serving layer.

Loads the champion model (model + its preprocessing pipeline) once at startup and scores a
single transaction on /predict. Every request measures wall-clock inference time and returns
it as `inference_ms`, and the server also logs it — so the <100ms claim can be verified, not
assumed. Default model type is XGBoost (configurable via the MODEL_TYPE env var).

The request body is a creditcard.csv-style record: Time, V1..V28, Amount (30 numeric fields).
The model expects exactly these columns; the Pydantic model is built from the dataset schema
so it stays in sync with config.py.

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
from pydantic import BaseModel, create_model

from fraud_platform.config import LATENCY_BUDGET_MS, NUMERIC_FEATURES
from fraud_platform.registry.registry import ModelRegistry

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("serving")

MODEL_TYPE = os.environ.get("MODEL_TYPE", "xgboost")

# loaded at startup; kept in module state so it's loaded once, not per request
_STATE: dict = {"model": None, "version": None, "meta": None}

# Build the request schema from the dataset columns (Time, V1..V28, Amount) so it can't drift
# from config.py. All fields are floats; an example of all-zeros gives /docs something valid.
_FIELDS = {name: (float, ...) for name in NUMERIC_FEATURES}
Transaction = create_model("Transaction", **_FIELDS)
Transaction.model_config = {
    "json_schema_extra": {"example": {name: 0.0 for name in NUMERIC_FEATURES}}
}


class PredictionResponse(BaseModel):
    is_fraud: int
    anomaly_score: float
    threshold: float
    model_type: str
    model_version: int
    inference_ms: float
    within_latency_budget: bool


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_champion()  # load the champion once when the server boots
    yield


app = FastAPI(title="Fraud Detection API (Kaggle creditcard.csv)", version="1.0",
              lifespan=lifespan)


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
    return {"service": "fraud-detection (Kaggle creditcard.csv)",
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

    # one-row dataframe with the exact columns the pipeline expects
    row = {c: getattr(tx, c) for c in NUMERIC_FEATURES}
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
