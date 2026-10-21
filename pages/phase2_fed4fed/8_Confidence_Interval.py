"""Page 8: Algorithm 3 Performance Confidence Interval under Decision Rules."""

import streamlit as st

from ui.theme import apply_custom_theme
from ui.navigation import render_step_tracker
from ui.sidebar import render_sidebar
from ui.guards import render_guard_barrier
from ui.components import render_header, render_metric_card, render_formula_card
from ui.charts_phase2 import plot_confidence_intervals_forest

apply_custom_theme()
render_sidebar()
render_step_tracker(8)

if not render_guard_barrier(8, "Confidence Interval"):
    st.stop()

render_header(
    title="Step 8: Global Model Confidence Interval (Algorithm 3)",
    subtitle="Compute statistical performance bounds for the global federated model governed by decision rules."
)

fairness_res = st.session_state["fairness_results"]
ci_res = fairness_res.get("confidence_interval", {})
is_permitted = ci_res.get("is_permitted", False)

# Mathematical Formulation
render_formula_card(
    title="Algorithm 3 (Graybill-Deal Performance Confidence Interval)",
    formula_latex=r"\hat{\mu}_{GD} \pm t_{1-\gamma/2, \nu} \cdot \sqrt{\frac{1}{W} \left(1 + 4 \sum_{i=1}^K \frac{w_i (1 - w_i / W)}{df_i W}\right)}",
    description="Constructs the exact pooled performance confidence interval when the model satisfies cross-client fairness homogeneity."
)

if not is_permitted:
    st.error("🚫 **Algorithm 3 Pooled Confidence Interval Unavailable**")
    st.markdown(
        f"""
        <div class="verdict-unfair">
            <b>DECISION RULE VIOLATION:</b><br/>
            {ci_res.get('reason', 'Fairness test rejected equal performance.')}
        </div>
        """,
        unsafe_allow_html=True
    )
    st.info(
        "💡 **Why is this unavailable?** Under Fed4Fed methodology, constructing a pooled common-mean "
        "confidence interval assumes all clients share a common performance parameter (μ_1 = ... = μ_K). "
        "Because Step 7 rejected this null hypothesis with statistical significance, reporting a single pooled interval "
        "would be scientifically invalid. However, individual client intervals are plotted below for diagnostic review."
    )

else:
    st.success("✅ **Fairness Condition Satisfied: Common Mean Confidence Interval Computed**")
    st.markdown(
        f"""
        <div class="verdict-fair">
            <b>ALGORITHM 3 RESULT:</b><br/>
            {ci_res.get('interpretation', '')}
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Confidence Level", f"{ci_res['confidence_level']*100:.1f}%")
    with c2:
        render_metric_card("Pooled Estimate (μ_GD)", f"{ci_res['pooled_estimate']:.4f}")
    with c3:
        render_metric_card("Lower Bound", f"{ci_res['pooled_ci_lower']:.4f}")
    with c4:
        render_metric_card("Upper Bound", f"{ci_res['pooled_ci_upper']:.4f}")

st.markdown("<br/>", unsafe_allow_html=True)

# Forest Plot
st.markdown("### 🌲 Confidence Intervals Forest Plot")
per_client_cis = ci_res.get("per_client_intervals", [])

fig_forest = plot_confidence_intervals_forest(
    per_client_cis=per_client_cis,
    pooled_estimate=ci_res.get("pooled_estimate") if is_permitted else None,
    pooled_ci_lower=ci_res.get("pooled_ci_lower") if is_permitted else None,
    pooled_ci_upper=ci_res.get("pooled_ci_upper") if is_permitted else None,
    max_clients_to_show=25
)
st.plotly_chart(fig_forest, use_container_width=True)

st.divider()

# Proceed button
col_act1, col_act2 = st.columns([1, 2])
with col_act1:
    if st.button("➡️ Proceed to Step 9: Final Report", type="primary", use_container_width=True):
        st.session_state["current_step"] = 9
        st.switch_page("pages/phase2_fed4fed/9_Final_Report.py")
