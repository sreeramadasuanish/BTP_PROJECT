"""Collapsible sidebar component for system status and active configuration."""

import torch
import streamlit as st
from src.utils.artifact_manager import ArtifactManager


def render_sidebar() -> None:
    """Render sidebar with system metrics, active experiment, and hardware info."""
    with st.sidebar:
        st.markdown("### ⚖️ **FED4FED Platform**")
        st.caption("Privacy-Preserving FL Statistical Fairness Suite")
        st.divider()

        # Active Experiment Status
        st.markdown("#### **Active Pipeline State**")
        dataset = st.session_state.get("dataset_name", "MNIST")
        scenario = st.session_state.get("scenario", "Fair")
        n_clients = st.session_state.get("num_clients", 100)

        scenario_badge = "badge-success" if scenario.lower() == "fair" else "badge-danger"
        st.markdown(
            f"**Dataset:** `{dataset}`<br/>"
            f"**Scenario:** <span class='fed-badge {scenario_badge}'>{scenario}</span><br/>"
            f"**Clients:** `{n_clients}`",
            unsafe_allow_html=True
        )

        st.divider()

        # Hardware & Compute Status
        st.markdown("#### **System Compute**")
        device_name = "CUDA GPU" if torch.cuda.is_available() else "CPU Execution"
        device_badge = "badge-primary" if torch.cuda.is_available() else "badge-warning"
        st.markdown(
            f"**Device:** <span class='fed-badge {device_badge}'>{device_name}</span><br/>"
            f"**PyTorch:** `{torch.__version__}`",
            unsafe_allow_html=True
        )

        st.divider()

        # Completed Artifacts Count
        mgr: ArtifactManager = st.session_state.get("artifact_manager", ArtifactManager())
        p1_exps = mgr.list_phase1_experiments()
        p2_evals = mgr.list_phase2_evaluations()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Phase 1 Runs", len(p1_exps))
        with col2:
            st.metric("Phase 2 Evals", len(p2_evals))

        st.divider()

        # Reset Session State
        if st.button("🔄 Reset Session State", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
