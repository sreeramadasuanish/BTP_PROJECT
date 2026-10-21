"""Page 6: Decentralized Client Bootstrap Resampling."""

from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd
import torch

from ui.theme import apply_custom_theme
from ui.navigation import render_step_tracker
from ui.sidebar import render_sidebar
from ui.guards import render_guard_barrier
from ui.components import render_header, render_metric_card
from ui.charts_phase2 import plot_bootstrap_violins, plot_bootstrap_histogram, plot_normality_qq
from src.models.registry import get_model
from src.phase2_fed4fed.bootstrap import run_clients_bootstrap
from src.phase2_fed4fed.normality_check import evaluate_normality

apply_custom_theme()
render_sidebar()
render_step_tracker(6)

if not render_guard_barrier(6, "Bootstrap"):
    st.stop()

render_header(
    title="Step 6: Client Bootstrap Analysis",
    subtitle="Construct empirical performance distributions on private client test partitions via non-parametric bootstrapping."
)

col_ctrl, col_view = st.columns([1, 2])

with col_ctrl:
    st.markdown("### 🎲 Bootstrap Controls")
    num_resamples = st.slider(
        "Bootstrap Replications (B)",
        min_value=50,
        max_value=1000,
        value=st.session_state.get("bootstrap_resamples", 200),
        step=50,
        help="Number of resamples per client to establish empirical distribution."
    )
    metric_name = st.selectbox(
        "Evaluation Metric",
        options=["accuracy", "error_rate"],
        index=0,
        help="Model performance metric evaluated during bootstrapping."
    )
    b_seed = st.number_input("Bootstrap Random Seed", value=42, min_value=1, max_value=99999)

    run_btn = st.button("▶️ Compute Bootstrap Distributions", type="primary", use_container_width=True)

exp_path = Path(st.session_state["phase2_loaded_exp_path"])
validation = st.session_state["phase2_loaded_validation"]
manifest = validation["manifest"]
dataset_name = manifest["dataset"]
model_cfg = validation["model_config"]

# Run Bootstrap computation
if run_btn or not st.session_state.get("bootstrap_completed", False):
    with st.spinner("Evaluating global model across clients and generating bootstrap resamples..."):
        # Load model
        model = get_model(dataset_name, model_cfg)
        model_weights = torch.load(exp_path / "global_model.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(model_weights)
        model.eval()

        # Load client test partitions
        test_dir = exp_path / "client_test_sets"
        test_files = sorted(test_dir.glob("client_*_test.pt"), key=lambda f: int(f.stem.split("_")[1]))

        client_test_pairs = {}
        with torch.no_grad():
            for tf in test_files:
                c_data = torch.load(tf, weights_only=False)
                cid = c_data["client_id"]
                c_x = c_data["data"]
                c_y = c_data["targets"]
                if len(c_y) == 0:
                    continue

                logits = model(c_x)
                preds = torch.argmax(logits, dim=1).numpy()
                client_test_pairs[cid] = (c_y.numpy(), preds)

        boot_res = run_clients_bootstrap(
            client_test_data=client_test_pairs,
            num_resamples=num_resamples,
            metric_name=metric_name,
            seed=b_seed
        )

        st.session_state["bootstrap_results"] = boot_res
        st.session_state["bootstrap_resamples"] = num_resamples
        st.session_state["bootstrap_metric"] = metric_name
        st.session_state["bootstrap_completed"] = True
        st.session_state["fairness_completed"] = False  # Reset downstream fairness

boot_res = st.session_state["bootstrap_results"]

with col_view:
    st.markdown("### 📊 Distribution Summary")
    b_means = boot_res["bootstrap_means"]
    b_vars = boot_res["bootstrap_variances"]
    k_clients = boot_res["num_clients"]

    c1, c2, c3 = st.columns(3)
    with c1:
        render_metric_card("Clients Evaluated", str(k_clients), f"Replications: B={boot_res['num_resamples']}")
    with c2:
        render_metric_card("Overall Mean Performance", f"{np.mean(b_means):.4f}", f"Std across clients: {np.std(b_means):.4f}")
    with c3:
        render_metric_card("Mean Client Variance", f"{np.mean(b_vars):.6f}", f"Max var: {np.max(b_vars):.6f}")

st.markdown("<br/>", unsafe_allow_html=True)

# Tabs for charts
tab1, tab2, tab3 = st.tabs(["🎻 Client Violin Distributions", "📊 Performance Histogram", "📈 Normality Diagnostic (Q-Q)"])

with tab1:
    fig_v = plot_bootstrap_violins(
        bootstrap_data=boot_res["bootstrap_data"],
        observed_points=boot_res["observed_points"],
        max_clients_to_show=min(25, boot_res["num_clients"])
    )
    st.plotly_chart(fig_v, use_container_width=True)

with tab2:
    fig_h = plot_bootstrap_histogram(boot_res["bootstrap_means"])
    st.plotly_chart(fig_h, use_container_width=True)

with tab3:
    # Diagnostic for client 0 or selected client
    cid_norm = st.selectbox("Client for Normality Diagnostic", options=list(range(min(20, boot_res["num_clients"]))), index=0)
    norm_res = evaluate_normality(boot_res["bootstrap_data"][cid_norm])
    fig_qq = plot_normality_qq(norm_res["theoretical_quantiles"], norm_res["empirical_quantiles"])
    st.plotly_chart(fig_qq, use_container_width=True)
    st.caption(
        f"Shapiro-Wilk test: stat = {norm_res['statistic']:.4f}, p = {norm_res['p_value']:.4f} "
        f"({'Consistent with Normal' if norm_res['is_normal'] else 'Non-normal deviation detected'})."
    )

st.divider()

# Proceed button
col_act1, col_act2 = st.columns([1, 2])
with col_act1:
    if st.button("➡️ Proceed to Step 7: Fairness Testing", type="primary", use_container_width=True):
        st.session_state["current_step"] = 7
        st.switch_page("pages/phase2_fed4fed/7_Fairness_Test.py")
