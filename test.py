# test.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import time
import plotly.graph_objects as go
import os
from drift import psi_score, ConceptDriftTracker, anomaly_score, embedding_drift_from_texts
from groq import Groq

MODEL_PATH = "models/rf_iris.joblib"
BASELINE_PATH = "models/baseline_stats.joblib"

st.set_page_config(page_title="AI Drift Radar", layout="wide")

@st.cache_data(show_spinner=False)
def load_artifacts():
    clf = joblib.load(MODEL_PATH)
    baseline = joblib.load(BASELINE_PATH)
    return clf, baseline

clf, baseline = load_artifacts()

# UI header
st.title("AI Drift Radar — Real-Time Model & Data Drift Visualizer")
st.markdown("Tracks Data / Concept / Anomaly / Embedding drift and generates an explanation (LLM or heuristic).")

col1, col2 = st.columns([2, 1])
with col1:
    run_btn = st.button("Start Simulation")
    stop_btn = st.button("Stop")
with col2:
    auto_explain = st.checkbox("Auto Explain when any drift > 70%", value=True)
    explain_now = st.button("Explain Now (manual)")

# session state initialization
if "running" not in st.session_state:
    st.session_state.running = False
if "history" not in st.session_state:
    st.session_state.history = []
if "concept_tracker" not in st.session_state:
    st.session_state.concept_tracker = ConceptDriftTracker(maxlen=50)
if "raw_pool" not in st.session_state:
    st.session_state.raw_pool = baseline["train_samples"].copy()
if "pred_history" not in st.session_state:
    st.session_state.pred_history = []

# start/stop
if run_btn:
    st.session_state.running = True
if stop_btn:
    st.session_state.running = False

# helper to generate a possibly-drifted sample
def get_stream_sample(X_test):
    # pick a random sample from baseline test/pool
    idx = np.random.randint(0, len(X_test))
    sample = X_test.iloc[idx].values.astype(float)
    # with 15% prob, inject drift noise
    if np.random.rand() < 0.15:
        noise = np.random.normal(loc=2.0, scale=1.0, size=sample.shape)
        sample = sample + noise
    return sample

# explanation function using OpenAI if key provided; otherwise a heuristic fallback

def explain_with_llm(metrics):
    GROQ_KEY = os.environ.get("GROQ_API_KEY")

    if GROQ_KEY:
        client = Groq(api_key=GROQ_KEY)

        prompt = f"""You are an ML monitoring assistant. Analyze these drift metrics and give:
1) Brief diagnosis
2) Likely root cause
3) One actionable recommendation

Data Drift: {metrics['data']:.1f}%
Concept Drift: {metrics['concept']:.1f}%
Anomaly Drift: {metrics['anomaly']:.1f}%
Embedding Drift: {metrics['embedding']:.1f}%

Keep it under 120 words, plain language."""

        try:
            completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",  # FAST + CHEAP + accurate
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
            )

            text = completion.choices[0].message["content"].strip()
            return text

        except Exception as e:
            st.error(f"LLM call failed: {e}")
            return heuristic_explanation(metrics)

    else:
        return heuristic_explanation(metrics)

def heuristic_explanation(m):
    # simple rule-based explanation
    lines = []
    if m["data"] > 60:
        lines.append("High data drift: incoming features distribution differs strongly from training.")
    elif m["data"] > 30:
        lines.append("Moderate data drift: some feature distribution shifts seen.")
    else:
        lines.append("No large shifts in raw feature distributions.")
    if m["concept"] > 50:
        lines.append("Concept drift: model accuracy dropped recently — labels/relationship may have changed.")
    elif m["concept"] > 20:
        lines.append("Slight concept drift: small decline in recent accuracy.")
    if m["anomaly"] > 40:
        lines.append("High anomaly score: many unusual records vs baseline.")
    if m["embedding"] > 40:
        lines.append("Embedding drift: semantic/representation changes detected in recent samples.")
    if not lines:
        lines = ["Metrics are stable. No immediate action required."]
    rec = "Recommendation: inspect incoming data, re-run validation tests, consider retraining with recent data if drift persists."
    return "\n".join(lines + [rec])

# layout placeholders
radar_col, metrics_col = st.columns([2,1])
history_col = st.container()
explain_col = st.container()

# load a pool of baseline rows to sample from
X_pool = baseline["train_samples"]

# main loop: run simulation while running
if st.session_state.running:
    # run for a limited number of iterations to avoid infinite blocking in cloud
    for _ in range(200):  # adjust upper bound as needed
        sample = get_stream_sample(X_pool)
        sample_df = pd.DataFrame([sample], columns=X_pool.columns)

        # prediction and concept tracking
        pred = int(clf.predict(sample_df)[0])
        # we don't generally have truth in streaming; for simulation pick the nearest baseline target
        # We'll assume 'true' is the label predicted on the closest training sample to simulate concept drift detection
        # (In a real system you'd have real labels or periodic ground-truth)
        distances = ((baseline["train_samples"].values - sample)**2).sum(axis=1)
        nearest_idx = np.argmin(distances)
        true = int(baseline["train_targets"].iloc[nearest_idx])
        st.session_state.concept_tracker.add(pred, true)

        # data drift: compute PSI comparing a single feature vector to baseline sample pool (we compute one aggregated PSI across features)
        flat_expected = baseline["train_samples"].values.flatten()
        flat_actual = np.concatenate([flat_expected[-200:], sample])  # approximate recent window
        data_psi = psi_score(flat_expected, flat_actual, buckets=10)

        # anomaly: z-score
        anomaly = anomaly_score(sample, baseline["feature_means"], baseline["feature_stds"])

        # embedding drift: take recent texts (stringified form)
        recent_texts = st.session_state.raw_pool.sample(n=20).astype(str).agg(" ".join, axis=1).tolist()
        # add current sample representation
        recent_texts.append(" ".join(sample_df.astype(str).iloc[0].tolist()))
        emb_drift = embedding_drift_from_texts(baseline["embedding_mean"], recent_texts)

        # concept drift:
        concept = st.session_state.concept_tracker.score()

        # aggregate into driftData percentages
        drift_metrics = {
            "data": float(round(data_psi,1)),
            "concept": float(round(concept,1)),
            "anomaly": float(round(anomaly,1)),
            "embedding": float(round(emb_drift,1))
        }

        # append to history
        hist = st.session_state.history
        hist.append({"ts": time.time(), **drift_metrics})
        if len(hist) > 100:
            hist = hist[-100:]
        st.session_state.history = hist

        # UI update
        with radar_col:
            st.subheader("Drift Radar")
            labels = ["Data Drift", "Concept Drift", "Anomaly Drift", "Embedding Drift"]
            values = [drift_metrics[k.lower()] for k in ["data","concept","anomaly","embedding"]]

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=values, theta=labels, fill='toself', name='drift'))
            fig.update_layout(polar=dict(radialaxis=dict(range=[0,100])), showlegend=False, height=420)
            st.plotly_chart(fig, use_container_width=True)

        with metrics_col:
            st.subheader("Current Metrics")
            for name, val in zip(labels, values):
                sev = "Normal"
                color = "green"
                if val >= 70:
                    sev = "Critical"
                    color = "red"
                elif val >= 40:
                    sev = "Warning"
                    color = "orange"
                st.metric(label=f"{name}", value=f"{val:.1f}%", delta=sev)

            st.markdown("---")
            st.write("Last prediction (simulated):", pred, "Simulated true:", true)

        with history_col:
            st.subheader("Drift History (last samples)")
            if len(st.session_state.history) > 1:
                df = pd.DataFrame(st.session_state.history)
                df["time"] = pd.to_datetime(df["ts"], unit='s').dt.strftime("%H:%M:%S")
                st.line_chart(df.set_index("time")[["data","concept","anomaly","embedding"]])

        # explanation triggers
        any_high = max(values) > 70
        explanation_text = None
        if explain_now or (auto_explain and any_high):
            explanation_text = explain_with_llm(drift_metrics)

        with explain_col:
            st.subheader("AI Explanation")
            if explanation_text:
                st.write(explanation_text)
            else:
                st.info("No explanation requested yet. Click Explain Now or enable Auto Explain.")

        # small wait to simulate real-time ingestion
        time.sleep(1.5)
        # check if user pressed stop (re-run will reset buttons)
        if not st.session_state.running:
            break
    # stop after loop ends
    st.session_state.running = False
else:
    st.info("Press Start Simulation to begin generating drifted samples every ~1.5s. Use Stop to stop the run.")
