"""Plotly interactive charts for Phase 2 FED4FED Statistical Evaluation."""

from typing import Any, Dict, List
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_bootstrap_violins(
    bootstrap_data: np.ndarray,
    observed_points: np.ndarray,
    max_clients_to_show: int = 25
) -> go.Figure:
    """Create violin plot showing empirical bootstrap distributions across clients."""
    k = min(len(bootstrap_data), max_clients_to_show)
    fig = go.Figure()

    for cid in range(k):
        dist = bootstrap_data[cid]
        fig.add_trace(go.Violin(
            y=dist,
            name=f"C{cid:02d}",
            box_visible=True,
            meanline_visible=True,
            points=False,
            line_color="#2563EB"
        ))

    # Add observed points as scatter overlay
    fig.add_trace(go.Scatter(
        x=[f"C{cid:02d}" for cid in range(k)],
        y=observed_points[:k],
        mode="markers",
        name="Observed Point",
        marker=dict(color="#DC2626", size=7, symbol="diamond")
    ))

    fig.update_layout(
        title=f"Bootstrap Performance Distributions (First {k} Clients)",
        yaxis=dict(title="Accuracy / Performance Metric", range=[0.0, 1.05]),
        xaxis=dict(title="Client ID"),
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def plot_bootstrap_histogram(bootstrap_means: np.ndarray) -> go.Figure:
    """Distribution histogram of client bootstrap performance means."""
    fig = px.histogram(
        x=bootstrap_means,
        nbins=25,
        marginal="box",
        title="Distribution of Client Bootstrap Performance Means",
        labels={"x": "Client Mean Performance", "count": "Frequency"},
        template="plotly_white",
        color_discrete_sequence=["#0284C7"]
    )
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    return fig


def plot_normality_qq(theoretical_quantiles: List[float], empirical_quantiles: List[float]) -> go.Figure:
    """Quantile-Quantile (Q-Q) plot for normality diagnostic."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=theoretical_quantiles,
        y=empirical_quantiles,
        mode="markers",
        name="Empirical Quantiles",
        marker=dict(color="#2563EB", size=5)
    ))

    # Add 45-degree reference line
    min_val = min(min(theoretical_quantiles), min(empirical_quantiles))
    max_val = max(max(theoretical_quantiles), max(empirical_quantiles))
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode="lines",
        name="Normal Line",
        line=dict(color="#DC2626", dash="dash")
    ))

    fig.update_layout(
        title="Bootstrap Distribution Normal Q-Q Plot",
        xaxis=dict(title="Theoretical Normal Quantiles"),
        yaxis=dict(title="Empirical Quantiles"),
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig


def plot_pvalue_comparison(
    pvals_dict: Dict[str, Any],
    significance_level: float = 0.05
) -> go.Figure:
    """Bar chart comparing p-values of Algorithm 1 and Algorithm 2 against alpha threshold."""
    categories = []
    p_values = []
    colors = []

    if pvals_dict.get("alg1_result"):
        p1 = pvals_dict["alg1_result"]["p_value"]
        categories.append("Algorithm 1 (Graybill-Deal T1)")
        p_values.append(p1)
        colors.append("#DC2626" if p1 < significance_level else "#059669")

    if pvals_dict.get("alg2_result"):
        p2 = pvals_dict["alg2_result"]["p_value"]
        categories.append("Algorithm 2 (Welch-Type F)")
        p_values.append(p2)
        colors.append("#DC2626" if p2 < significance_level else "#059669")

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=categories,
        y=p_values,
        marker_color=colors,
        text=[f"p = {p:.4f}" if p >= 1e-3 else f"p = {p:.2e}" for p in p_values],
        textposition="auto"
    ))

    # Horizontal significance threshold line
    fig.add_shape(
        type="line",
        x0=-0.5,
        x1=len(categories) - 0.5,
        y0=significance_level,
        y1=significance_level,
        line=dict(color="#D97706", width=2, dash="dash"),
    )

    fig.add_annotation(
        x=len(categories) - 0.5,
        y=significance_level,
        text=f"Threshold α = {significance_level}",
        showarrow=False,
        yshift=10,
        font=dict(color="#D97706", size=11)
    )

    fig.update_layout(
        title="Fairness Test p-Values vs. Significance Level α",
        yaxis=dict(title="p-Value", range=[0.0, max(max(p_values) * 1.15, significance_level * 1.5)]),
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig


def plot_confidence_intervals_forest(
    per_client_cis: List[Dict[str, Any]],
    pooled_estimate: float | None = None,
    pooled_ci_lower: float | None = None,
    pooled_ci_upper: float | None = None,
    max_clients_to_show: int = 25
) -> go.Figure:
    """Forest plot displaying client confidence intervals alongside pooled Graybill-Deal CI."""
    sub_cis = per_client_cis[:max_clients_to_show]
    c_labels = [f"Client {c['client_id']:02d}" for c in sub_cis]
    means = [c["mean"] for c in sub_cis]
    err_low = [c["mean"] - c["ci_lower"] for c in sub_cis]
    err_high = [c["ci_upper"] - c["mean"] for c in sub_cis]

    fig = go.Figure()

    # Per-client intervals
    fig.add_trace(go.Scatter(
        x=means,
        y=c_labels,
        mode="markers",
        name="Client Mean",
        error_x=dict(
            type="data",
            symmetric=False,
            array=err_high,
            arrayminus=err_low,
            color="#2563EB",
            thickness=1.5
        ),
        marker=dict(color="#2563EB", size=7)
    ))

    # Pooled interval if available
    if pooled_estimate is not None and pooled_ci_lower is not None and pooled_ci_upper is not None:
        fig.add_vline(
            x=pooled_estimate,
            line=dict(color="#059669", width=2),
            annotation_text=f"Pooled μ_GD = {pooled_estimate:.4f}",
            annotation_position="top left"
        )
        fig.add_vrect(
            x0=pooled_ci_lower,
            x1=pooled_ci_upper,
            fillcolor="#10B981",
            opacity=0.15,
            line_width=0,
            annotation_text=f"Pooled 95% CI [{pooled_ci_lower:.4f}, {pooled_ci_upper:.4f}]",
            annotation_position="bottom right"
        )

    fig.update_layout(
        title=f"Performance Confidence Intervals Forest Plot (First {len(sub_cis)} Clients)",
        xaxis=dict(title="Performance Metric", range=[0.0, 1.05]),
        yaxis=dict(title="", autorange="reversed"),
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig
