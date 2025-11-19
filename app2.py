"""
AI Drift Radar - Simplified Version
Real-Time Model & Data Drift Monitoring
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import time
import matplotlib.pyplot as plt
import os
from drift import psi_score, ConceptDriftTracker, anomaly_score, embedding_drift_from_texts
from groq import Groq

# ==================== CONFIGURATION ====================
MODEL_PATH = "models/rf_iris.joblib"
BASELINE_PATH = "models/baseline_stats.joblib"
MAX_HISTORY = 100

# ==================== PAGE SETUP ====================
st.set_page_config(page_title="AI Drift Radar", layout="wide")

st.title("🎯 AI Drift Radar - Real-Time Monitoring")
st.markdown("Monitor Data, Concept, Anomaly, and Embedding drift in real-time")


# ==================== LOAD MODEL ====================
@st.cache_resource
def load_model():
    clf = joblib.load(MODEL_PATH)
    baseline = joblib.load(BASELINE_PATH)
    return clf, baseline


try:
    clf, baseline = load_model()
except Exception as e:
    st.error(f"❌ Error loading models: {e}")
    st.info("Run: python train_model.py")
    st.stop()

# ==================== HELPER FUNCTIONS ====================
def get_sample():
    """Generate a sample with possible drift"""
    X_train = baseline["train_samples"]
    idx = np.random.randint(0, len(X_train))
    sample = X_train.iloc[idx].values.astype(float)

    # 15% chance to inject drift
    if np.random.rand() < 0.15:
        noise = np.random.normal(loc=2.0, scale=1.0, size=sample.shape)
        sample = sample + noise

    return sample


def calculate_drifts(sample):
    """Calculate all drift metrics"""
    sample_df = pd.DataFrame([sample], columns=baseline["train_samples"].columns)

    # Prediction
    pred = int(clf.predict(sample_df)[0])
    distances = ((baseline["train_samples"].values - sample) ** 2).sum(axis=1)
    nearest_idx = np.argmin(distances)
    true = int(baseline["train_targets"].iloc[nearest_idx])
    st.session_state.concept_tracker.add(pred, true)

    # Data drift (PSI)
    flat_expected = baseline["train_samples"].values.flatten()
    flat_actual = np.concatenate([flat_expected[-200:], sample])
    data_drift = psi_score(flat_expected, flat_actual, buckets=10)

    # Concept drift
    concept_drift = st.session_state.concept_tracker.score()

    # Anomaly drift
    anomaly_drift = anomaly_score(sample, baseline["feature_means"], baseline["feature_stds"])

    # Embedding drift
    recent_texts = baseline["train_samples"].sample(n=20).astype(str).agg(" ".join, axis=1).tolist()
    recent_texts.append(" ".join(sample_df.astype(str).iloc[0].tolist()))
    embedding_drift = embedding_drift_from_texts(baseline["embedding_mean"], recent_texts)

    return {
        "data": round(data_drift, 1),
        "concept": round(concept_drift, 1),
        "anomaly": round(anomaly_drift, 1),
        "embedding": round(embedding_drift, 1),
        "timestamp": st.session_state.total_samples,
        "pred": pred,
        "true": true
    }


def generate_explanation(metrics):
    """Generate explanation using Groq or heuristic"""
    GROQ_KEY = os.environ.get("GROQ_API_KEY")

    if GROQ_KEY:
        try:
            client = Groq(api_key=GROQ_KEY)
            prompt = f"""Analyze these drift metrics:
- Data Drift: {metrics['data']}%
- Concept Drift: {metrics['concept']}%
- Anomaly Drift: {metrics['anomaly']}%
- Embedding Drift: {metrics['embedding']}%

Provide: 1) What's happening, 2) Likely cause, 3) Recommendation. Keep under 100 words."""

            completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200
            )
            return completion.choices[0].message.content.strip()
        except:
            pass

    # Heuristic fallback
    max_drift = max(metrics['data'], metrics['concept'], metrics['anomaly'], metrics['embedding'])
    if max_drift > 70:
        return f"🚨 Critical drift detected ({max_drift:.1f}%). High distribution shift in incoming data. Recommendation: Investigate data source changes and consider model retraining."
    elif max_drift > 40:
        return f"⚠️ Moderate drift ({max_drift:.1f}%). Some distribution changes observed. Recommendation: Monitor closely and review recent predictions."
    else:
        return f"✅ Normal operation ({max_drift:.1f}%). All metrics within acceptable range. Continue monitoring."

# ==================== SESSION STATE ====================
if "running" not in st.session_state:
    st.session_state.running = False
if "history" not in st.session_state:
    st.session_state.history = []
if "concept_tracker" not in st.session_state:
    st.session_state.concept_tracker = ConceptDriftTracker(maxlen=50)
if "total_samples" not in st.session_state:
    st.session_state.total_samples = 0
if "explanation" not in st.session_state:
    st.session_state.explanation = ""

# ==================== SIDEBAR CONTROLS ====================
st.sidebar.header("Controls")

col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("▶️ Start", use_container_width=True):
        st.session_state.running = True
with col2:
    if st.button("⏹️ Stop", use_container_width=True):
        st.session_state.running = False

if st.sidebar.button("🔄 Reset", use_container_width=True):
    st.session_state.history = []
    st.session_state.concept_tracker = ConceptDriftTracker(maxlen=50)
    st.session_state.total_samples = 0
    st.session_state.explanation = ""
    st.rerun()

st.sidebar.markdown("---")
threshold = st.sidebar.slider("Alert Threshold (%)", 0, 100, 70)
auto_explain = st.sidebar.checkbox("Auto-explain on high drift", value=True)

if st.sidebar.button("🧠 Explain Now", use_container_width=True):
    if st.session_state.history:
        with st.spinner("Generating explanation..."):
            latest = st.session_state.history[-1]
            st.session_state.explanation = generate_explanation(latest)

st.sidebar.markdown("---")
st.sidebar.metric("Total Samples", st.session_state.total_samples)
if st.session_state.running:
    st.sidebar.success("🟢 RUNNING")
else:
    st.sidebar.error("🔴 STOPPED")





# ==================== MAIN DISPLAY ====================

# Current metrics
if st.session_state.history:
    latest = st.session_state.history[-1]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📊 Data Drift", f"{latest['data']:.1f}%")
    col2.metric("🧠 Concept Drift", f"{latest['concept']:.1f}%")
    col3.metric("⚠️ Anomaly Drift", f"{latest['anomaly']:.1f}%")
    col4.metric("🔮 Embedding Drift", f"{latest['embedding']:.1f}%")

# Single unified plot
if len(st.session_state.history) > 1:
    st.subheader("📈 All Drift Metrics Over Time")

    df = pd.DataFrame(st.session_state.history)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['timestamp'], df['data'], 'o-', label='Data Drift', linewidth=2, markersize=4)
    ax.plot(df['timestamp'], df['concept'], 's-', label='Concept Drift', linewidth=2, markersize=4)
    ax.plot(df['timestamp'], df['anomaly'], '^-', label='Anomaly Drift', linewidth=2, markersize=4)
    ax.plot(df['timestamp'], df['embedding'], 'd-', label='Embedding Drift', linewidth=2, markersize=4)

    # Threshold line
    ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold}%)')

    ax.set_xlabel('Sample Number', fontsize=12)
    ax.set_ylabel('Drift Percentage (%)', fontsize=12)
    ax.set_title('Real-Time Drift Monitoring', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    st.pyplot(fig)
    plt.close()

# Explanation
if st.session_state.explanation:
    st.subheader("🤖 AI Explanation")
    st.info(st.session_state.explanation)

# Data table (optional)
if st.session_state.history:
    with st.expander("📋 View Raw Data"):
        st.dataframe(pd.DataFrame(st.session_state.history).tail(20))

# ==================== SIMULATION LOOP ====================
if st.session_state.running:
    placeholder = st.empty()

    for i in range(100):  # Run 100 iterations
        # Generate and calculate
        sample = get_sample()
        metrics = calculate_drifts(sample)

        # Update state
        st.session_state.total_samples += 1
        st.session_state.history.append(metrics)
        if len(st.session_state.history) > MAX_HISTORY:
            st.session_state.history = st.session_state.history[-MAX_HISTORY:]

        # Auto-explain
        max_drift = max(metrics['data'], metrics['concept'], metrics['anomaly'], metrics['embedding'])
        if auto_explain and max_drift > threshold:
            st.session_state.explanation = generate_explanation(metrics)

        # Show progress
        with placeholder.container():
            st.write(f"Processing sample #{st.session_state.total_samples}...")
            st.write(
                f"Latest: Data={metrics['data']}%, Concept={metrics['concept']}%, Anomaly={metrics['anomaly']}%, Embedding={metrics['embedding']}%")

        time.sleep(1.0)

        if not st.session_state.running:
            break

    st.session_state.running = False
    st.rerun()
else:
    st.info("👆 Click **Start** in the sidebar to begin monitoring")