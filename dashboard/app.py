"""Streamlit + Plotly dashboard.

Four panels:
  1. EDA           — class balance + feature distributions (fraud vs legit)
  2. Model compare — metric bars from the saved evaluation results / registry
  3. Live scoring  — POST a transaction to the running FastAPI /predict endpoint
  4. Drift monitor — PSI + KS of a chosen batch vs the training distribution

Run the API first (uvicorn fraud_platform.serving.api:app), then:
    streamlit run dashboard/app.py

Everything here uses SYNTHETIC data.
"""
from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from fraud_platform.config import (
    ARTIFACTS_DIR,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
)
from fraud_platform.data.generate import (
    COUNTRIES,
    DEVICE_TYPES,
    MERCHANT_CATEGORIES,
    TRANSACTION_TYPES,
    generate,
)
from fraud_platform.data.loader import load_or_generate
from fraud_platform.monitoring.drift import compute_drift

st.set_page_config(page_title="Fraud Detection Platform", layout="wide")
API_URL = "http://localhost:8000"


@st.cache_data
def get_data() -> pd.DataFrame:
    return load_or_generate()


st.title("Anomaly & Fraud Detection Platform")
st.caption("⚠️ All data shown is SYNTHETIC — generated for demonstration, not real transactions.")

tab_eda, tab_models, tab_score, tab_drift = st.tabs(
    ["EDA", "Model comparison", "Live scoring", "Drift monitor"])

# ----------------------------------------------------------------- EDA
with tab_eda:
    df = get_data()
    st.subheader("Dataset overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Transactions", f"{len(df):,}")
    c2.metric("Fraud rows", f"{int(df[TARGET].sum()):,}")
    c3.metric("Fraud rate", f"{df[TARGET].mean():.2%}")

    balance = df[TARGET].map({0: "legit", 1: "fraud"}).value_counts().reset_index()
    balance.columns = ["label", "count"]
    st.plotly_chart(px.bar(balance, x="label", y="count", color="label",
                           title="Class balance (note the imbalance)"),
                    use_container_width=True)

    feat = st.selectbox("Feature distribution by class", NUMERIC_FEATURES)
    plot_df = df.copy()
    plot_df["label"] = plot_df[TARGET].map({0: "legit", 1: "fraud"})
    fig = px.histogram(plot_df, x=feat, color="label", barmode="overlay",
                       nbins=60, marginal="box",
                       title=f"{feat}: fraud vs legit (heavy overlap is expected)")
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------- Model comparison
with tab_models:
    st.subheader("Model comparison on held-out test set")
    results_path = ARTIFACTS_DIR / "evaluation_results.json"
    if not results_path.exists():
        st.warning("No evaluation results yet. Run: `python -m fraud_platform.train`")
    else:
        results = json.loads(results_path.read_text())
        models = results["models"]
        rows = []
        for name, m in models.items():
            for metric in ["precision", "recall", "f1", "roc_auc", "pr_auc"]:
                rows.append({"model": name, "metric": metric, "value": m[metric]})
        mdf = pd.DataFrame(rows)
        st.plotly_chart(px.bar(mdf, x="metric", y="value", color="model",
                               barmode="group", title="Metrics by model"),
                        use_container_width=True)
        st.caption("On imbalanced data, trust PR-AUC and recall over ROC-AUC.")
        st.dataframe(pd.DataFrame(models).T.round(4))

# ----------------------------------------------------------- Live scoring
with tab_score:
    st.subheader("Score a transaction against the live API")
    try:
        health = requests.get(f"{API_URL}/health", timeout=2).json()
        st.success(f"API up — model loaded: {health['model_loaded']} ({health['model_type']})")
    except Exception:
        st.error(f"API not reachable at {API_URL}. Start it: "
                 "`uvicorn fraud_platform.serving.api:app`")

    with st.form("score"):
        col1, col2 = st.columns(2)
        amount = col1.number_input("amount", value=920.5, min_value=0.0)
        account_age = col2.number_input("account_age_days", value=45, min_value=0)
        num_tx = col1.number_input("num_tx_last_24h", value=11, min_value=0)
        avg_amt = col2.number_input("avg_amount_last_30d", value=80.0, min_value=0.0)
        dist = col1.number_input("distance_from_home_km", value=240.0, min_value=0.0)
        hour = col2.slider("hour", 0, 23, 3)
        mcat = col1.selectbox("merchant_category", MERCHANT_CATEGORIES, index=2)
        ttype = col2.selectbox("transaction_type", TRANSACTION_TYPES, index=2)
        dev = col1.selectbox("device_type", DEVICE_TYPES, index=2)
        country = col2.selectbox("country", COUNTRIES, index=5)
        submitted = st.form_submit_button("Score transaction")

    if submitted:
        payload = {
            "amount": amount, "account_age_days": account_age, "num_tx_last_24h": num_tx,
            "avg_amount_last_30d": avg_amt, "distance_from_home_km": dist, "hour": hour,
            "merchant_category": mcat, "transaction_type": ttype,
            "device_type": dev, "country": country,
        }
        try:
            r = requests.post(f"{API_URL}/predict", json=payload, timeout=5).json()
            verdict = "FRAUD" if r["is_fraud"] == 1 else "legit"
            st.metric("Prediction", verdict)
            c1, c2, c3 = st.columns(3)
            c1.metric("Anomaly score", f"{r['anomaly_score']:.4f}")
            c2.metric("Threshold", f"{r['threshold']:.4f}")
            c3.metric("Inference (ms)", f"{r['inference_ms']:.2f}",
                      delta="under 100ms" if r["within_latency_budget"] else "over 100ms")
            st.json(r)
        except Exception as e:
            st.error(f"scoring failed: {e}")

# ------------------------------------------------------------- Drift monitor
with tab_drift:
    st.subheader("Drift: incoming batch vs training distribution")
    st.caption("PSI bands: <0.10 none · 0.10–0.25 moderate · ≥0.25 significant. "
               "KS flags p < 0.05.")
    reference = get_data()

    shift = st.slider("Inject synthetic drift (multiply 'amount' by)", 1.0, 6.0, 1.0, 0.5)
    current = generate(n_rows=4000, fraud_frac=0.02, seed=999)
    if shift > 1.0:
        current = current.copy()
        current["amount"] = current["amount"] * shift

    result = compute_drift(reference, current)
    if result["drift_detected"]:
        st.error(f"DRIFT DETECTED — {result['n_features_flagged']} feature(s) flagged")
    else:
        st.success("No significant drift detected")

    feats = result["features"]
    drift_df = pd.DataFrame([
        {"feature": k, "psi": v["psi"], "band": v["psi_band"],
         "ks_pvalue": v["ks_pvalue"], "flagged": v["flagged"]}
        for k, v in feats.items()
    ])
    st.plotly_chart(px.bar(drift_df, x="feature", y="psi", color="band",
                           title="PSI per feature",
                           color_discrete_map={"none": "green", "moderate": "orange",
                                               "significant": "red"}),
                    use_container_width=True)
    st.dataframe(drift_df)
