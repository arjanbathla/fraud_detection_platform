"""Streamlit + Plotly dashboard for the Kaggle creditcard.csv fraud platform.

Four panels:
  1. EDA           — class balance + Amount/Time/V distributions (fraud vs legit)
  2. Model compare — metric bars from the saved evaluation results
  3. Live scoring  — sample a REAL transaction from the test split and score it via the API
  4. Drift monitor — PSI + KS of a batch vs the training distribution

Run the API first (uvicorn fraud_platform.serving.api:app), then:
    streamlit run dashboard/app.py

Needs data/creditcard.csv present (Kaggle download).
"""
from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from fraud_platform.config import ARTIFACTS_DIR, CREDITCARD_CSV, NUMERIC_FEATURES, TARGET
from fraud_platform.data.loader import load_data, split
from fraud_platform.monitoring.drift import compute_drift

st.set_page_config(page_title="Fraud Detection Platform", layout="wide")
API_URL = "http://localhost:8000"


@st.cache_data
def get_data():
    try:
        return load_data()
    except FileNotFoundError:
        return None


@st.cache_data
def get_split():
    df = get_data()
    return split(df) if df is not None else None


st.title("Credit Card Fraud Detection Platform")
st.caption("Data: Kaggle 'Credit Card Fraud Detection' (creditcard.csv). Features V1–V28 are "
           "anonymised PCA components; only Time and Amount are raw.")

df = get_data()
if df is None:
    st.error(f"creditcard.csv not found at {CREDITCARD_CSV}. "
             "Download it from Kaggle and place it in the data/ folder.")
    st.stop()

tab_eda, tab_models, tab_score, tab_drift = st.tabs(
    ["EDA", "Model comparison", "Live scoring", "Drift monitor"])

# ----------------------------------------------------------------- EDA
with tab_eda:
    st.subheader("Dataset overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Transactions", f"{len(df):,}")
    c2.metric("Fraud rows", f"{int(df[TARGET].sum()):,}")
    c3.metric("Fraud rate", f"{df[TARGET].mean():.3%}")

    balance = df[TARGET].map({0: "legit", 1: "fraud"}).value_counts().reset_index()
    balance.columns = ["label", "count"]
    st.plotly_chart(px.bar(balance, x="label", y="count", color="label", log_y=True,
                           title="Class balance (log scale — fraud is ~0.17%)"),
                    use_container_width=True)

    plot_df = df.copy()
    plot_df["label"] = plot_df[TARGET].map({0: "legit", 1: "fraud"})
    feat = st.selectbox("Feature distribution by class",
                        ["Amount", "Time"] + [c for c in NUMERIC_FEATURES if c.startswith("V")])
    # cap Amount axis so the long tail doesn't flatten the plot
    fig = px.histogram(plot_df, x=feat, color="label", barmode="overlay", nbins=80,
                       histnorm="probability density",
                       title=f"{feat}: fraud vs legit (density)")
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
    st.subheader("Score a real transaction against the live API")
    try:
        health = requests.get(f"{API_URL}/health", timeout=2).json()
        st.success(f"API up — model loaded: {health['model_loaded']} ({health['model_type']})")
    except Exception:
        st.error(f"API not reachable at {API_URL}. Start it: "
                 "`uvicorn fraud_platform.serving.api:app`")

    sp = get_split()
    X_test, y_test = sp[4], sp[5]
    st.caption("Sample a real held-out transaction (28 PCA features can't be hand-typed, so "
               "we pull a genuine row and send it to /predict).")

    which = st.radio("Sample from", ["any", "known fraud", "known legit"], horizontal=True)
    if st.button("Sample a transaction"):
        pool = X_test
        if which == "known fraud":
            pool = X_test[y_test == 1]
        elif which == "known legit":
            pool = X_test[y_test == 0]
        idx = pool.sample(1).index[0]
        st.session_state["row_idx"] = idx

    if "row_idx" in st.session_state:
        idx = st.session_state["row_idx"]
        row = X_test.loc[idx]
        true_label = int(y_test[X_test.index.get_loc(idx)])
        payload = {c: float(row[c]) for c in NUMERIC_FEATURES}
        st.write(f"True label: **{'FRAUD' if true_label else 'legit'}**  ·  Amount: "
                 f"{row['Amount']:.2f}")
        try:
            r = requests.post(f"{API_URL}/predict", json=payload, timeout=5).json()
            verdict = "FRAUD" if r["is_fraud"] == 1 else "legit"
            correct = (r["is_fraud"] == true_label)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Prediction", verdict, delta="correct" if correct else "wrong")
            c2.metric("Anomaly score", f"{r['anomaly_score']:.4f}")
            c3.metric("Threshold", f"{r['threshold']:.4f}")
            c4.metric("Inference (ms)", f"{r['inference_ms']:.2f}",
                      delta="under 100ms" if r["within_latency_budget"] else "over 100ms")
            st.json(r)
        except Exception as e:
            st.error(f"scoring failed: {e}")

# ------------------------------------------------------------- Drift monitor
with tab_drift:
    st.subheader("Drift: incoming batch vs training distribution")
    st.caption("PSI bands: <0.10 none · 0.10–0.25 moderate · ≥0.25 significant. "
               "KS flags p < 0.05.")

    reference = df.sample(min(20000, len(df)), random_state=1)
    current = df.sample(min(20000, len(df)), random_state=2).copy()
    shift = st.slider("Inject synthetic drift (multiply 'Amount' by)", 1.0, 6.0, 1.0, 0.5)
    if shift > 1.0:
        current["Amount"] = current["Amount"] * shift

    result = compute_drift(reference, current)
    if result["drift_detected"]:
        st.error(f"DRIFT DETECTED — {result['n_features_flagged']} feature(s) flagged")
    else:
        st.success("No significant drift detected")

    drift_df = pd.DataFrame([
        {"feature": k, "psi": v["psi"], "band": v["psi_band"],
         "ks_pvalue": v["ks_pvalue"], "flagged": v["flagged"]}
        for k, v in result["features"].items()
    ])
    st.plotly_chart(px.bar(drift_df, x="feature", y="psi", color="band",
                           title="PSI per feature",
                           color_discrete_map={"none": "green", "moderate": "orange",
                                               "significant": "red"}),
                    use_container_width=True)
    st.dataframe(drift_df)
