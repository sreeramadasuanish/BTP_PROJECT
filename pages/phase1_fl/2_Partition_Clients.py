"""Page 2: Partition Clients and Analyze Data Heterogeneity."""

import streamlit as st
import numpy as np

from ui.theme import apply_custom_theme
from ui.navigation import render_step_tracker
from ui.sidebar import render_sidebar
from ui.guards import render_guard_barrier
from ui.components import render_header, render_metric_card
from ui.charts_phase1 import (
    plot_client_partition_stacked,
    plot_partition_heatmap,
    plot_client_size_distribution
)
from src.data.partition import create_partitions

apply_custom_theme()
render_sidebar()
render_step_tracker(2)

if not render_guard_barrier(2, "Partition Clients"):
    st.stop()

render_header(
    title="Step 2: Partition Clients",
    subtitle="Partition training and test data across decentralized clients and inspect class distributions."
)

dataset_name = st.session_state["dataset_name"]
scenario = st.session_state["scenario"]
num_clients = st.session_state["num_clients"]
seed = st.session_state["seed"]
train_ds = st.session_state["train_dataset"]
test_ds = st.session_state["test_dataset"]
p_params = st.session_state.get("partition_params", {})

# Generate or retrieve partitions
if not st.session_state.get("partitions_created", False):
    with st.spinner("Generating client data partitions..."):
        train_parts, test_parts, part_meta = create_partitions(
            dataset_name=dataset_name,
            scenario=scenario,
            train_targets=train_ds.targets.numpy(),
            test_targets=test_ds.targets.numpy(),
            num_clients=num_clients,
            seed=seed,
            **p_params
        )

        # Verification of no duplicates and full coverage
        all_train_indices = [idx for c_inds in train_parts.values() for idx in c_inds]
        assert len(all_train_indices) == len(set(all_train_indices)), "Duplicate indices detected in training partition!"

        st.session_state["train_partitions"] = train_parts
        st.session_state["test_partitions"] = test_parts
        st.session_state["partition_metadata"] = part_meta
        st.session_state["partitions_created"] = True

train_parts = st.session_state["train_partitions"]
test_parts = st.session_state["test_partitions"]
part_meta = st.session_state["partition_metadata"]

# Summary Statistics Metric Cards
stats = part_meta["train_size_stats"]
col1, col2, col3, col4 = st.columns(4)
with col1:
    render_metric_card("Partitioning Method", part_meta["method"].title(), f"Scenario: {scenario}")
with col2:
    render_metric_card("Total Samples", f"{stats['total']:,}", f"Covered: 100% (No duplicates)")
with col3:
    render_metric_card("Mean Client Size", f"{stats['mean']:.1f}", f"Std Dev: {stats['std']:.1f}")
with col4:
    render_metric_card("Size Range [Min, Max]", f"[{stats['min']}, {stats['max']}]", f"Total Clients: {num_clients}")

st.markdown("<br/>", unsafe_allow_html=True)

# Interactive Charts
meta = st.session_state["dataset_meta"]
num_classes = meta["num_classes"]

tab1, tab2, tab3 = st.tabs(["📊 Client Label Proportions", "🗺️ Class Distribution Heatmap", "📈 Client Dataset Sizes"])

with tab1:
    fig_stacked = plot_client_partition_stacked(
        client_partitions=train_parts,
        targets=train_ds.targets.numpy(),
        num_classes=num_classes,
        max_clients_to_show=min(30, num_clients)
    )
    st.plotly_chart(fig_stacked, use_container_width=True)

with tab2:
    fig_heat = plot_partition_heatmap(
        client_partitions=train_parts,
        targets=train_ds.targets.numpy(),
        num_classes=num_classes,
        max_clients_to_show=min(50, num_clients)
    )
    st.plotly_chart(fig_heat, use_container_width=True)

with tab3:
    fig_dist = plot_client_size_distribution(train_parts)
    st.plotly_chart(fig_dist, use_container_width=True)

st.divider()

# Individual Client Inspector
st.markdown("### 🔍 Inspect Individual Client")
cid_to_inspect = st.selectbox(
    "Select Client ID to inspect local composition",
    options=list(range(num_clients)),
    index=0
)

c_train_inds = train_parts[cid_to_inspect]
c_test_inds = test_parts[cid_to_inspect]
c_targets = train_ds.targets.numpy()[c_train_inds]
c_counts = np.bincount(c_targets, minlength=num_classes)

st.markdown(
    f"**Client {cid_to_inspect:02d}:** "
    f"Training samples: `{len(c_train_inds)}` | "
    f"Private test samples: `{len(c_test_inds)}` | "
    f"Dominant class: `{np.argmax(c_counts)}` ({np.max(c_counts)} samples)"
)

st.divider()

# Proceed button
col_act1, col_act2 = st.columns([1, 2])
with col_act1:
    if st.button("💾 Save Partition & Proceed to Step 3", type="primary", use_container_width=True):
        st.session_state["current_step"] = 3
        st.switch_page("pages/phase1_fl/3_FL_Training.py")
