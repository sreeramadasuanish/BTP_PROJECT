"""Page 7: Statistical Fairness Testing (Algorithm 1 & Algorithm 2)."""

import streamlit as st
import numpy as np

from ui.theme import apply_custom_theme
from ui.navigation import render_step_tracker
from ui.sidebar import render_sidebar
from ui.guards import render_guard_barrier
from ui.components import render_header, render_metric_card, render_verdict_banner, render_formula_card
from ui.charts_phase2 import plot_pvalue_comparison
from src.phase2_fed4fed.fairness import evaluate_fairness
from src.reports.result_tables import build_fairness_summary_table

apply_custom_theme()
render_sidebar()
render_step_tracker(7)

if not render_guard_barrier(7, "Fairness Test"):
    st.stop()

render_header(
    title="Step 7: Statistical Fairness Testing",
    subtitle="Evaluate cross-client performance homogeneity using Algorithm 1 (Graybill-Deal T1) and Algorithm 2 (Welch-Type F)."
)

boot_res = st.session_state["bootstrap_results"]

# Controls Section
st.markdown("### 🎛️ Test Configuration")
ctrl_cols = st.columns([1, 1, 1, 1])

with ctrl_cols[0]:
    run_alg1 = st.checkbox("Run Algorithm 1 (Graybill-Deal / T1)", value=True)
with ctrl_cols[1]:
    run_alg2 = st.checkbox("Run Algorithm 2 (Welch-Type F)", value=True)
with ctrl_cols[2]:
    sig_level = st.select_slider(
        "Significance Level (α)",
        options=[0.001, 0.01, 0.05, 0.10],
        value=st.session_state.get("significance_level", 0.05)
    )
with ctrl_cols[3]:
    policy = st.selectbox(
        "Joint Decision Policy",
        options=["conservative", "unanimous"],
        index=0,
        help="Conservative: Disparity flagged if either test rejects H0. Unanimous: Both must reject H0."
    )

if not run_alg1 and not run_alg2:
    st.warning("⚠️ Please select at least one algorithm to run.")
    st.stop()

# Run Evaluation
with st.spinner("Computing test statistics and p-values..."):
    fairness_res = evaluate_fairness(
        client_means=boot_res["bootstrap_means"],
        client_variances=boot_res["bootstrap_variances"],
        run_alg1=run_alg1,
        run_alg2=run_alg2,
        significance_level=sig_level,
        confidence_level=st.session_state.get("confidence_level", 0.95),
        decision_policy=policy
    )

    st.session_state["fairness_results"] = fairness_res
    st.session_state["significance_level"] = sig_level
    st.session_state["decision_policy"] = policy
    st.session_state["fairness_completed"] = True

# Verdict Banner
render_verdict_banner(
    verdict=f"OVERALL VERDICT: {fairness_res['overall_verdict']}",
    text=fairness_res["plain_english"],
    is_unfair=fairness_res["reject_h0"]
)

st.markdown("<br/>", unsafe_allow_html=True)

# Test Comparison Table & Plotly Chart
col_table, col_chart = st.columns([1, 1])

with col_table:
    st.markdown("### 📋 Statistical Test Results")
    summary_df = build_fairness_summary_table(fairness_res)
    st.dataframe(summary_df, use_container_width=True)

with col_chart:
    st.markdown("### 📊 p-Value Comparison")
    p_fig = plot_pvalue_comparison(fairness_res, significance_level=sig_level)
    st.plotly_chart(p_fig, use_container_width=True)

st.divider()

# Detailed Algorithm Breakdown
tab_alg1, tab_alg2 = st.tabs(["Algorithm 1 Details", "Algorithm 2 Details"])

with tab_alg1:
    if fairness_res.get("alg1_result"):
        a1 = fairness_res["alg1_result"]
        render_formula_card(
            title="Algorithm 1 (Graybill-Deal / Chi-Square T1)",
            formula_latex=r"T_1 = \sum_{i=1}^K \frac{(\bar{X}_i - \hat{\mu}_{GD})^2}{S_i^2} \sim \chi^2(K - 1)",
            description="Tests the null hypothesis of equal expected performance under common mean Graybill-Deal weighting."
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            render_metric_card("Test Statistic (T1)", f"{a1['test_statistic']:.4f}")
        with c2:
            render_metric_card("Degrees of Freedom", str(a1["degrees_of_freedom"]))
        with c3:
            render_metric_card("p-Value", f"{a1['p_value']:.4e}" if a1['p_value'] < 1e-3 else f"{a1['p_value']:.4f}")

        st.info(f"**Interpretation:** {a1['interpretation']}")
        if a1["warnings"]:
            for w in a1["warnings"]:
                st.warning(f"⚠️ {w}")
    else:
        st.info("Algorithm 1 was not selected.")

with tab_alg2:
    if fairness_res.get("alg2_result"):
        a2 = fairness_res["alg2_result"]
        render_formula_card(
            title="Algorithm 2 (Welch-Type F Test)",
            formula_latex=r"F_W = \frac{\frac{1}{K-1} \sum_{i=1}^K w_i (\bar{X}_i - \bar{X}^*)^2}{1 + \frac{2(K-2)}{K^2-1} \Lambda} \sim F(\nu_1, \nu_2)",
            description="Tests performance equality without assuming equal variances across client distributions."
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            render_metric_card("Test Statistic (F_W)", f"{a2['test_statistic']:.4f}")
        with c2:
            render_metric_card("Numerator / Denom df", f"({a2['df_numerator']:.0f}, {a2['df_denominator']:.1f})")
        with c3:
            render_metric_card("p-Value", f"{a2['p_value']:.4e}" if a2['p_value'] < 1e-3 else f"{a2['p_value']:.4f}")

        st.info(f"**Interpretation:** {a2['interpretation']}")
    else:
        st.info("Algorithm 2 was not selected.")

st.divider()

# Proceed button
col_act1, col_act2 = st.columns([1, 2])
with col_act1:
    if st.button("➡️ Proceed to Step 8: Confidence Interval", type="primary", use_container_width=True):
        st.session_state["current_step"] = 8
        st.switch_page("pages/phase2_fed4fed/8_Confidence_Interval.py")
