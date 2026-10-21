"""Plotly interactive charts for Phase 1 Federated Learning."""

from typing import Dict, List
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_client_partition_stacked(
    client_partitions: Dict[int, List[int]],
    targets: np.ndarray,
    num_classes: int = 10,
    max_clients_to_show: int = 30
) -> go.Figure:
    """Create stacked bar chart showing class distribution per client."""
    cids = sorted(client_partitions.keys())[:max_clients_to_show]
    data_rows = []

    for cid in cids:
        indices = client_partitions[cid]
        if len(indices) == 0:
            continue
        c_targets = targets[indices]
        counts = np.bincount(c_targets, minlength=num_classes)
        props = counts / len(indices)
        for cls_idx in range(num_classes):
            data_rows.append({
                "Client": f"C{cid:02d}",
                "Class": str(cls_idx),
                "Proportion": float(props[cls_idx]),
                "Count": int(counts[cls_idx]),
            })

    df = pd.DataFrame(data_rows)
    fig = px.bar(
        df,
        x="Client",
        y="Proportion",
        color="Class",
        title=f"Client Class Proportions (First {len(cids)} Clients)",
        labels={"Proportion": "Class Share", "Client": "Client ID"},
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    fig.update_layout(
        barmode="stack",
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def plot_partition_heatmap(
    client_partitions: Dict[int, List[int]],
    targets: np.ndarray,
    num_classes: int = 10,
    max_clients_to_show: int = 50
) -> go.Figure:
    """Create heatmap of sample counts per class across clients."""
    cids = sorted(client_partitions.keys())[:max_clients_to_show]
    matrix = np.zeros((len(cids), num_classes))

    for i, cid in enumerate(cids):
        indices = client_partitions[cid]
        if len(indices) > 0:
            c_targets = targets[indices]
            matrix[i] = np.bincount(c_targets, minlength=num_classes)

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=[f"Class {c}" for c in range(num_classes)],
        y=[f"Client {cid:02d}" for cid in cids],
        colorscale="Blues",
        colorbar=dict(title="Samples")
    ))
    fig.update_layout(
        title="Client-Class Distribution Heatmap",
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig


def plot_client_size_distribution(client_partitions: Dict[int, List[int]]) -> go.Figure:
    """Histogram of client dataset sizes."""
    sizes = [len(v) for v in client_partitions.values()]
    fig = px.histogram(
        x=sizes,
        nbins=20,
        title="Client Partition Size Distribution",
        labels={"x": "Sample Count per Client", "y": "Number of Clients"},
        template="plotly_white",
        color_discrete_sequence=["#2563EB"]
    )
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    return fig


def plot_training_curves(round_logs: pd.DataFrame) -> go.Figure:
    """Plot dual-axis training loss and global accuracy curves."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=round_logs["round"],
        y=round_logs["train_loss"],
        mode="lines+markers",
        name="Training Loss",
        line=dict(color="#DC2626", width=2),
        marker=dict(size=6)
    ))

    fig.add_trace(go.Scatter(
        x=round_logs["round"],
        y=round_logs["val_accuracy"],
        mode="lines+markers",
        name="Global Val Accuracy",
        line=dict(color="#2563EB", width=2),
        marker=dict(size=6),
        yaxis="y2"
    ))

    fig.update_layout(
        title="Federated Learning Convergence Curves",
        xaxis=dict(title=dict(text="Communication Round"), tickmode="linear", dtick=1),
        yaxis=dict(
            title=dict(text="Training Loss", font=dict(color="#DC2626")), 
            tickfont=dict(color="#DC2626")
        ),
        yaxis2=dict(
            title=dict(text="Validation Accuracy", font=dict(color="#2563EB")),
            tickfont=dict(color="#2563EB"),
            overlaying="y",
            side="right",
            range=[0.0, 1.0]
        ),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig