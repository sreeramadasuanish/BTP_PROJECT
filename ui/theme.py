"""UI Theme and custom CSS styling for FED4FED research dashboard."""

import streamlit as st


def apply_custom_theme() -> None:
    """Inject polished research-grade CSS into Streamlit."""
    custom_css = """
    <style>
    /* Global styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Top banner card */
    .fed-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        color: #F8FAFC;
        padding: 1.5rem 2rem;
        border-radius: 0.75rem;
        border: 1px solid #334155;
        margin-bottom: 1.5rem;
    }

    .fed-banner h1 {
        color: #F8FAFC;
        font-size: 1.75rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
        letter-spacing: -0.02em;
    }

    .fed-banner p {
        color: #94A3B8;
        font-size: 0.95rem;
        margin-bottom: 0;
        line-height: 1.4;
    }

    /* Cards */
    .fed-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 0.625rem;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }

    .fed-card-header {
        font-size: 1rem;
        font-weight: 600;
        color: #0F172A;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Metric cards */
    .fed-metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 0.5rem;
        padding: 1rem;
        text-align: left;
    }

    .fed-metric-label {
        font-size: 0.8rem;
        font-weight: 500;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.25rem;
    }

    .fed-metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #0F172A;
    }

    .fed-metric-sub {
        font-size: 0.75rem;
        color: #64748B;
        margin-top: 0.25rem;
    }

    /* Verdict Banners */
    .verdict-fair {
        background-color: #ECFDF5;
        border: 1.5px solid #10B981;
        color: #065F46;
        padding: 1rem 1.25rem;
        border-radius: 0.5rem;
        font-weight: 600;
        margin: 1rem 0;
    }

    .verdict-unfair {
        background-color: #FEF2F2;
        border: 1.5px solid #EF4444;
        color: #991B1B;
        padding: 1rem 1.25rem;
        border-radius: 0.5rem;
        font-weight: 600;
        margin: 1rem 0;
    }

    /* Badges */
    .fed-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 9999px;
    }
    .badge-primary { background-color: #EFF6FF; color: #1D4ED8; border: 1px solid #BFDBFE; }
    .badge-success { background-color: #ECFDF5; color: #047857; border: 1px solid #A7F3D0; }
    .badge-warning { background-color: #FFFBEB; color: #B45309; border: 1px solid #FDE68A; }
    .badge-danger  { background-color: #FEF2F2; color: #B91C1C; border: 1px solid #FECACA; }

    /* Step tracker */
    .step-tracker {
        display: flex;
        gap: 0.5rem;
        margin-bottom: 1.5rem;
        overflow-x: auto;
        padding-bottom: 0.5rem;
    }
    .step-item {
        flex: 1;
        min-width: 110px;
        background: #F1F5F9;
        border: 1px solid #CBD5E1;
        border-radius: 0.375rem;
        padding: 0.5rem 0.6rem;
        font-size: 0.75rem;
        text-align: center;
        color: #64748B;
        font-weight: 500;
    }
    .step-item.active {
        background: #1E293B;
        color: #F8FAFC;
        border-color: #0F172A;
        font-weight: 600;
    }
    .step-item.completed {
        background: #EFF6FF;
        color: #1D4ED8;
        border-color: #93C5FD;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)
