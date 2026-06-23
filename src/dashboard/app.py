import streamlit as st
import os
import sys

# Add project root to path so src/ imports work
_here = os.path.abspath(__file__)
for _ in range(6):
    _here = os.path.dirname(_here)
    if os.path.exists(os.path.join(_here, 'main.py')):
        if _here not in sys.path:
            sys.path.insert(0, _here)
        break

from src.dashboard.components import (
    render_metric_row,
    render_batch_analysis,
    render_model_performance,
    render_per_tier_results,
    render_tier_comparison,
    render_per_attack_analysis,
)

st.set_page_config(
    page_title="Adversarial Robust IDS",
    page_icon="🛡️",
    layout="wide"
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🛡️ Adversarially Robust Intrusion Detection System")
st.markdown(
    "Three-tier detection pipeline: **Tier 1** Signature Rules → "
    "**Tier 2** Deep Neural Network (DNN) → "
    "**Tier 3** Adversarially Robust DNN (FGSM + PGD trained)"
)
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🔧 Control Panel")
st.sidebar.markdown("**Dataset:** CICIDS 2017 / 2018")
st.sidebar.markdown("**Tier 2:** Best of CNN / DNN / LSTM")
st.sidebar.markdown("**Tier 3:** PyTorch DNN — 40% clean + 30% FGSM + 30% PGD")
st.sidebar.divider()

mode = st.sidebar.selectbox(
    "Select Mode",
    [
        "Model Performance",
        "Per-Tier Results",
        "Tier Comparison",
        "Per-Attack Analysis",
        "Batch Analysis",
    ]
)

st.sidebar.divider()
st.sidebar.caption(
    "**Model Performance** — overall Tier 2 vs Tier 3 evaluation. "
    "**Per-Tier Results** — separate metrics for each tier. "
    "**Tier Comparison** — Tier 1+2 and Tier 1+2+3 comparisons. "
    "**Per-Attack Analysis** — per-attack-type (DoS, Probe, etc.) charts. "
    "**Batch Analysis** — live inference on uploaded CSV."
)

# ── Top metrics ───────────────────────────────────────────────────────────────
render_metric_row()
st.divider()

# ── Main content ──────────────────────────────────────────────────────────────
if mode == "Model Performance":
    render_model_performance()
elif mode == "Per-Tier Results":
    render_per_tier_results()
elif mode == "Tier Comparison":
    render_tier_comparison()
elif mode == "Per-Attack Analysis":
    render_per_attack_analysis()
elif mode == "Batch Analysis":
    render_batch_analysis()
