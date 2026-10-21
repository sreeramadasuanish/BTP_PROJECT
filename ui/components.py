"""Reusable UI components and widgets for the FED4FED research dashboard."""

from pathlib import Path
from typing import Any, Dict, Optional
import streamlit as st


def render_header(title: str, subtitle: str) -> None:
    """Render top page header banner."""
    st.markdown(
        f"""
        <div class="fed-banner">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_metric_card(label: str, value: Any, subtext: Optional[str] = None) -> None:
    """Render a styled metric card."""
    sub_html = f"<div class='fed-metric-sub'>{subtext}</div>" if subtext else ""
    st.markdown(
        f"""
        <div class="fed-metric-card">
            <div class="fed-metric-label">{label}</div>
            <div class="fed-metric-value">{value}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True
    )


def render_verdict_banner(verdict: str, text: str, is_unfair: bool) -> None:
    """Render high-impact fairness verdict banner."""
    cls = "verdict-unfair" if is_unfair else "verdict-fair"
    icon = "❌" if is_unfair else "✅"
    st.markdown(
        f"""
        <div class="{cls}">
            <div style="font-size: 1.15rem; margin-bottom: 0.35rem;">{icon} <b>{verdict}</b></div>
            <div style="font-size: 0.88rem; font-weight: normal; opacity: 0.95;">{text}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_formula_card(title: str, formula_latex: str, description: str) -> None:
    """Render math formula explanation box."""
    with st.expander(f"📐 Mathematical Formulation: {title}", expanded=True):
        st.latex(formula_latex)
        st.caption(description)


def render_download_button(file_path: Path | str, label: str = "Download Artifact", mime: str = "application/octet-stream") -> None:
    """Render file download button if file exists."""
    p = Path(file_path)
    if not p.exists():
        st.button(f"File unavailable: {p.name}", disabled=True)
        return

    with open(p, "rb") as f:
        bytes_data = f.read()

    st.download_button(
        label=f"⬇️ {label} ({len(bytes_data)/1024:.1f} KB)",
        data=bytes_data,
        file_name=p.name,
        mime=mime,
        use_container_width=True
    )
