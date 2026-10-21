"""Page 3: Federated Learning Simulation and FedAvg Training."""

import time
import streamlit as st
import pandas as pd

from ui.theme import apply_custom_theme
from ui.navigation import render_step_tracker
from ui.sidebar import render_sidebar
from ui.guards import render_guard_barrier
from ui.components import render_header, render_metric_card
from ui.charts_phase1 import plot_training_curves
from src.phase1_fl.simulation import FLSimulation

apply_custom_theme()
render_sidebar()
render_step_tracker(3)

if not render_guard_barrier(3, "FL Training"):
    st.stop()

render_header(
    title="Step 3: Federated Learning Training",
    subtitle="Train decentralized models locally and aggregate parameters centrally using weighted FedAvg."
)

col_cfg, col_prog = st.columns([1, 2])

with col_cfg:
    st.markdown("### ⚙️ Training Hyperparameters")

    num_rounds = st.slider("Communication Rounds (R)", min_value=1, max_value=30, value=5)
    local_epochs = st.slider("Local Client Epochs (E)", min_value=1, max_value=5, value=1)
    batch_size = st.select_slider("Batch Size (B)", options=[16, 32, 64, 128], value=32)
    lr = st.select_slider("Learning Rate (η)", options=[0.001, 0.005, 0.01, 0.05, 0.1], value=0.01)

    opt_default = "Adam" if st.session_state["dataset_name"] == "SST" else "SGD"
    optimizer_type = st.selectbox("Local Optimizer", options=["SGD", "Adam"], index=0 if opt_default == "SGD" else 1)

    client_fraction = st.slider("Client Participation Fraction (C)", min_value=0.1, max_value=1.0, value=1.0, step=0.1)

    start_training = st.button("🚀 Start Federated Training", type="primary", use_container_width=True)

with col_prog:
    st.markdown("### 📊 Live Training Progress")
    progress_bar = st.progress(0.0)
    status_text = st.empty()
    metric_cols = st.columns(3)
    with metric_cols[0]:
        m_round = st.empty()
    with metric_cols[1]:
        m_loss = st.empty()
    with metric_cols[2]:
        m_acc = st.empty()

    chart_placeholder = st.empty()

# Execute Training
if start_training:
    dataset_name = st.session_state["dataset_name"]
    train_ds = st.session_state["train_dataset"]
    test_ds = st.session_state["test_dataset"]
    train_parts = st.session_state["train_partitions"]
    test_parts = st.session_state["test_partitions"]
    meta = st.session_state["dataset_meta"]
    seed = st.session_state["seed"]
    device = st.session_state["device"]

    sim = FLSimulation(
        dataset_name=dataset_name,
        train_dataset=train_ds,
        test_dataset=test_ds,
        train_partitions=train_parts,
        test_partitions=test_parts,
        model_config=meta,
        device=device,
        seed=seed
    )

    history_logs = []

    def on_round_completed(log_entry: dict):
        history_logs.append(log_entry)
        curr_round = log_entry["round"]
        prog = float(curr_round) / float(num_rounds)
        progress_bar.progress(prog)
        status_text.info(f"Communication Round {curr_round}/{num_rounds} completed in {log_entry['round_time_sec']:.2f}s")

        with metric_cols[0]:
            render_metric_card("Current Round", f"{curr_round}/{num_rounds}")
        with metric_cols[1]:
            render_metric_card("Train Loss", f"{log_entry['train_loss']:.4f}")
        with metric_cols[2]:
            render_metric_card("Val Accuracy", f"{log_entry['val_accuracy']:.4f}")

        df_curr = pd.DataFrame(history_logs)
        chart_placeholder.plotly_chart(plot_training_curves(df_curr), use_container_width=True)

    with st.spinner("Executing federated training rounds across clients..."):
        logs_df = sim.run(
            num_rounds=num_rounds,
            local_epochs=local_epochs,
            batch_size=batch_size,
            lr=lr,
            optimizer_type=optimizer_type,
            client_fraction=client_fraction,
            round_callback=on_round_completed
        )

        st.session_state["round_logs"] = logs_df
        st.session_state["trained_global_model"] = sim.server.global_model
        st.session_state["training_completed"] = True
        st.session_state["simulation_instance"] = sim

    st.success("🎉 Federated training finished successfully!")

elif st.session_state.get("training_completed", False):
    logs_df = st.session_state["round_logs"]
    progress_bar.progress(1.0)
    status_text.success("Completed training run loaded from session.")
    with metric_cols[0]:
        render_metric_card("Total Rounds", f"{len(logs_df)}")
    with metric_cols[1]:
        render_metric_card("Final Train Loss", f"{logs_df['train_loss'].iloc[-1]:.4f}")
    with metric_cols[2]:
        render_metric_card("Final Val Accuracy", f"{logs_df['val_accuracy'].iloc[-1]:.4f}")

    chart_placeholder.plotly_chart(plot_training_curves(logs_df), use_container_width=True)

st.divider()

# Training Logs Table & Next Action
if st.session_state.get("training_completed", False):
    st.markdown("### 📜 Round Logs")
    st.dataframe(st.session_state["round_logs"], use_container_width=True)

    col_next1, col_next2 = st.columns([1, 2])
    with col_next1:
        if st.button("➡️ Proceed to Step 4: Export Model", type="primary", use_container_width=True):
            st.session_state["current_step"] = 4
            st.switch_page("pages/phase1_fl/4_Export_Model.py")
