"""Page 5: Load and Validate Phase 1 Federated Learning Output."""

from pathlib import Path
import streamlit as st
import pandas as pd

from ui.theme import apply_custom_theme
from ui.navigation import render_step_tracker
from ui.sidebar import render_sidebar
from ui.components import render_header, render_metric_card
from src.utils.artifact_manager import ArtifactManager

apply_custom_theme()
render_sidebar()
render_step_tracker(5)

render_header(
    title="Step 5: Load Phase 1 Output",
    subtitle="Select a completed Phase 1 experiment to evaluate cross-client statistical performance fairness."
)

mgr = ArtifactManager()
experiments = mgr.list_phase1_experiments()

if not experiments:
    st.warning("⚠️ No completed Phase 1 experiments found on disk.")
    st.info("Please run Phase 1 (Steps 1 to 4) first to generate a global model checkpoint and client test partitions.")
    if st.button("⬅️ Go to Step 1: Select Dataset", type="primary"):
        st.session_state["current_step"] = 1
        st.switch_page("pages/phase1_fl/1_Select_Dataset.py")
    st.stop()

# Experiment Selector
exp_options = {
    f"{e['experiment_id']} ({e['dataset']} - {e['scenario']}, K={e['num_clients']})": e["path"]
    for e in experiments
}

selected_label = st.selectbox(
    "Select Completed Phase 1 Experiment",
    options=list(exp_options.keys()),
    index=0
)

selected_path = Path(exp_options[selected_label])

# Validation
validation = mgr.validate_phase1_experiment(selected_path)

if not validation["valid"]:
    st.error(f"❌ Experiment validation failed: {validation['error']}")
    st.stop()

# Store in session state
manifest = validation["manifest"]
st.session_state["phase2_loaded_exp_path"] = str(selected_path)
st.session_state["phase2_loaded_manifest"] = manifest
st.session_state["phase2_loaded_validation"] = validation
st.session_state["phase2_is_loaded"] = True

# Metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    render_metric_card("Dataset & Scenario", f"{manifest['dataset']}", f"Scenario: {manifest['scenario']}")
with col2:
    render_metric_card("Global Accuracy", f"{manifest['final_accuracy']:.4f}", f"Rounds: {manifest['num_rounds']}")
with col3:
    render_metric_card("Decentralized Clients", str(validation["num_clients"]), "Private test sets ready")
with col4:
    render_metric_card("Checkpoint Status", "VERIFIED", "global_model.pt loaded")

st.markdown("<br/>", unsafe_allow_html=True)

# Training logs preview
st.markdown("### 📈 Training History of Loaded Model")
st.dataframe(validation["logs_df"], use_container_width=True)

st.divider()

# Proceed button
col_act1, col_act2 = st.columns([1, 2])
with col_act1:
    if st.button("➡️ Proceed to Step 6: Bootstrap Analysis", type="primary", use_container_width=True):
        st.session_state["current_step"] = 6
        st.switch_page("pages/phase2_fed4fed/6_Bootstrap.py")
