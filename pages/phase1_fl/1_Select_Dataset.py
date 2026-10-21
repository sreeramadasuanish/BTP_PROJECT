"""Page 1: Select Dataset and Scenario Configuration."""

import streamlit as st
import torch

from ui.theme import apply_custom_theme
from ui.navigation import render_step_tracker
from ui.sidebar import render_sidebar
from ui.components import render_header, render_metric_card
from src.data.registry import DATASET_METADATA, load_dataset
from src.models.registry import get_model, count_parameters
from src.utils.seed import set_seed

apply_custom_theme()
render_sidebar()
render_step_tracker(1)

render_header(
    title="Step 1: Select Dataset and Scenario",
    subtitle="Configure the federated benchmark dataset, fairness scenario, and decentralized client scale."
)

col_ctrl, col_info = st.columns([1, 1])

with col_ctrl:
    st.markdown("### 🎛️ Experiment Parameters")

    # Dataset Selection
    dataset_name = st.selectbox(
        "Benchmark Dataset",
        options=list(DATASET_METADATA.keys()),
        index=0,
        help="Select the machine learning dataset for federated learning."
    )

    # Scenario Selection
    scenario = st.radio(
        "Distribution Scenario",
        options=["Fair", "Unfair"],
        index=0,
        horizontal=True,
        help="Fair: Uniform random allocation (IID). Unfair: Non-IID class/label skew across clients."
    )

    # Number of clients
    default_n = 50 if dataset_name == "SST" else 100
    num_clients = st.slider(
        "Number of Decentralized Clients (K)",
        min_value=5,
        max_value=200,
        value=default_n,
        step=5,
        help="Total number of federated clients participating in the network."
    )

    # Random Seed
    seed = st.number_input("Random Seed", value=42, min_value=1, max_value=99999)

    # Compute Device
    available_devices = ["cpu"]
    if torch.cuda.is_available():
        available_devices.append("cuda")
    device = st.selectbox("Compute Device", options=available_devices, index=0)

    # Dataset-Specific Scenario Controls
    partition_params = {}
    if scenario == "Unfair":
        st.markdown("#### Scenario-Specific Partition Parameters")
        if dataset_name == "CIFAR-10":
            alpha = st.select_slider(
                "Dirichlet Concentration (α)",
                options=[0.001, 0.01, 0.1, 10.0],
                value=0.1,
                help="Smaller α produces extreme class imbalance; larger α approaches uniform distribution."
            )
            partition_params["alpha"] = alpha
        elif dataset_name == "MNIST":
            shards = st.slider("Shards per Client", min_value=1, max_value=5, value=2)
            partition_params["shards_per_client"] = shards
        elif dataset_name == "Fashion-MNIST":
            skew = st.slider("Primary Clothing Categories Share", min_value=0.5, max_value=0.95, value=0.85, step=0.05)
            partition_params["skew_ratio"] = skew
        elif dataset_name == "SST":
            skew = st.slider("Dominant Sentiment Ratio", min_value=0.6, max_value=0.95, value=0.90, step=0.05)
            partition_params["skew_ratio"] = skew

meta = DATASET_METADATA[dataset_name]

with col_info:
    st.markdown("### 📋 Dataset & Architecture Overview")

    m_col1, m_col2 = st.columns(2)
    with m_col1:
        render_metric_card("Input Format", meta["input_shape"], f"Type: {meta['type'].title()}")
    with m_col2:
        render_metric_card("Class Count", str(meta["num_classes"]), f"Classes: {', '.join(meta['classes'][:4])}...")

    # Instantiating model to show parameter count
    temp_model = get_model(dataset_name)
    total_params, trainable_params = count_parameters(temp_model)

    m_col3, m_col4 = st.columns(2)
    with m_col3:
        render_metric_card("Train / Test Samples", f"{meta['train_samples']} / {meta['test_samples']}", "Full ground-truth count")
    with m_col4:
        render_metric_card("Model Parameters", f"{total_params:,}", meta["model_architecture"].split()[0])

    st.markdown("#### ⚙️ Partition Policy")
    policy_desc = meta["fair_partition"] if scenario == "Fair" else meta["unfair_partition"]
    st.info(f"**Selected Strategy:** {policy_desc}")

st.divider()

# Action button to download/load dataset
btn_col1, btn_col2 = st.columns([1, 2])
with btn_col1:
    if st.button("📥 Load & Verify Dataset", type="primary", use_container_width=True):
        with st.spinner(f"Loading and preprocessing {dataset_name} dataset..."):
            set_seed(seed)
            train_ds, test_ds, d_meta = load_dataset(dataset_name)

            # Store in session state
            st.session_state["dataset_name"] = dataset_name
            st.session_state["scenario"] = scenario
            st.session_state["num_clients"] = num_clients
            st.session_state["seed"] = seed
            st.session_state["device"] = device
            st.session_state["partition_params"] = partition_params
            st.session_state["train_dataset"] = train_ds
            st.session_state["test_dataset"] = test_ds
            st.session_state["dataset_meta"] = d_meta
            st.session_state["dataset_loaded"] = True

            # Invalidate downstream steps on configuration change
            st.session_state["partitions_created"] = False
            st.session_state["training_completed"] = False
            st.session_state["phase1_exported"] = False

            st.success(f"✅ Successfully loaded {dataset_name}: {len(train_ds)} train, {len(test_ds)} test samples.")

with btn_col2:
    if st.session_state.get("dataset_loaded", False):
        if st.button("➡️ Proceed to Step 2: Partition Clients", use_container_width=True):
            st.session_state["current_step"] = 2
            st.switch_page("pages/phase1_fl/2_Partition_Clients.py")
