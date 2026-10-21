"""Page 0: FED4FED Home Overview and System Dashboard."""

import streamlit as st
import pandas as pd

from ui.theme import apply_custom_theme
from ui.navigation import render_step_tracker
from ui.sidebar import render_sidebar
from ui.components import render_header, render_metric_card, render_verdict_banner
from src.data.registry import DATASET_METADATA
from src.utils.artifact_manager import ArtifactManager

apply_custom_theme()
render_sidebar()
render_step_tracker(0)

render_header(
    title="FED4FED: Federated Learning & Statistical Fairness Evaluation",
    subtitle="A rigorous, privacy-preserving statistical testing platform for cross-client model fairness."
)

mgr = ArtifactManager()
p1_runs = mgr.list_phase1_experiments()
p2_evals = mgr.list_phase2_evaluations()

# Top metric summary cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    render_metric_card("Registered Datasets", "4", "MNIST, F-MNIST, CIFAR-10, SST")
with col2:
    render_metric_card("Phase 1 Runs", str(len(p1_runs)), "Completed FedAvg training runs")
with col3:
    render_metric_card("Phase 2 Evals", str(len(p2_evals)), "Statistical fairness tests")
with col4:
    latest_verdict = p2_evals[0]["pvalues"]["overall_verdict"] if p2_evals else "No evaluations yet"
    render_metric_card("Latest Verdict", latest_verdict.split()[0] if p2_evals else "N/A", "Most recent test status")

st.markdown("<br/>", unsafe_allow_html=True)

# Latest Fairness Result if exists
if p2_evals:
    latest = p2_evals[0]
    is_unfair = latest["pvalues"].get("reject_h0", False)
    render_verdict_banner(
        verdict=f"Latest Evaluation ({latest['experiment_id']}): {latest['pvalues']['overall_verdict']}",
        text=latest["pvalues"].get("plain_english", ""),
        is_unfair=is_unfair
    )

# Workflow Diagram
st.markdown("### 🔄 End-to-End FED4FED Workflow")
st.markdown(
    """
    ```text
    ┌────────────────────────────────────────────────────────────────────────────────────────┐
    │                               PHASE 1: FEDERATED LEARNING                              │
    │  [Dataset] ──> [Partition Clients] ──> [Local Training] ──> [FedAvg] ──> [Export Model]│
    └───────────────────────────────────────────┬────────────────────────────────────────────┘
                                                │ Artifact Handoff (Model + Private Test Sets)
    ┌───────────────────────────────────────────▼────────────────────────────────────────────┐
    │                       PHASE 2: FED4FED STATISTICAL EVALUATION                          │
    │  [Load Output] ──> [Bootstrap Resampling] ──> [Alg 1 (T1) & Alg 2 (F)] ──> [Alg 3 CI] │
    │                                     │                                                  │
    │                                     └───> [Final Research Report PDF]                  │
    └────────────────────────────────────────────────────────────────────────────────────────┘
    ```
    """
)

# Start Experiment Action Button
st.markdown("<br/>", unsafe_allow_html=True)
action_col1, action_col2 = st.columns([1, 2])
with action_col1:
    if st.button("🚀 Start New Experiment (Phase 1)", type="primary", use_container_width=True):
        st.session_state["current_step"] = 1
        st.switch_page("pages/phase1_fl/1_Select_Dataset.py")
with action_col2:
    if p1_runs:
        if st.button("📂 Evaluate Completed Experiment (Phase 2)", use_container_width=True):
            st.session_state["current_step"] = 5
            st.switch_page("pages/phase2_fed4fed/5_Load_Phase1_Output.py")

st.divider()

# Four Dataset Cards
st.markdown("### 📚 Supported Benchmark Datasets")
d_cols = st.columns(4)

for idx, (name, meta) in enumerate(DATASET_METADATA.items()):
    with d_cols[idx]:
        st.markdown(f"#### **{name}**")
        st.caption(f"**Modality:** {meta['type'].title()}")
        st.markdown(
            f"- **Classes:** `{meta['num_classes']}`<br/>"
            f"- **Train / Test:** `{meta['train_samples']} / {meta['test_samples']}`<br/>"
            f"- **Model:** `{meta['model_architecture'].split()[0]}`<br/>"
            f"- **Fair:** {meta['fair_partition']}<br/>"
            f"- **Unfair:** {meta['unfair_partition']}",
            unsafe_allow_html=True
        )

st.divider()

# Recent Experiment History
st.markdown("### 📜 Recent Phase 1 Experiments")
if p1_runs:
    table_rows = []
    for exp in p1_runs[:8]:
        table_rows.append({
            "Experiment ID": exp.get("experiment_id"),
            "Dataset": exp.get("dataset"),
            "Scenario": exp.get("scenario"),
            "Clients": exp.get("num_clients"),
            "Rounds": exp.get("num_rounds"),
            "Final Accuracy": f"{exp.get('final_accuracy', 0.0):.4f}",
            "Timestamp": exp.get("timestamp", "").replace("T", " ")[:19],
        })
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True)
else:
    st.info("No saved experiments yet. Click **Start New Experiment** to begin Phase 1.")
