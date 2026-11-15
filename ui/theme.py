"""
TacticalVision Design System & Theme
====================================
Clean, high-contrast light-mode CSS styling optimized for modern high-DPI laptop screens.
"""

import streamlit as st


THEME_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
    }

    #MainMenu, footer, header {visibility: hidden;}

    /* Top Application Bar */
    .top-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%);
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 10px 18px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }

    .brand-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0f172a;
        letter-spacing: -0.01em;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .badge-pill {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 5px;
        display: inline-flex;
        align-items: center;
        gap: 5px;
    }

    .badge-onnx {
        background: rgba(2, 132, 199, 0.10);
        color: #0284c7;
        border: 1px solid rgba(2, 132, 199, 0.25);
    }

    .badge-cache {
        background: rgba(16, 185, 129, 0.10);
        color: #059669;
        border: 1px solid rgba(16, 185, 129, 0.25);
    }

    .badge-live {
        background: rgba(245, 158, 11, 0.10);
        color: #d97706;
        border: 1px solid rgba(245, 158, 11, 0.25);
    }

    /* Control Panel Box */
    .control-deck {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }

    /* Coach Stat Cards */
    .coach-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px 14px;
        height: 100%;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
    }

    .coach-title {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        color: #64748b;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }

    .coach-metric {
        font-size: 1.35rem;
        font-weight: 700;
        color: #0f172a;
        font-family: 'JetBrains Mono', monospace;
    }

    .coach-sub {
        font-size: 0.75rem;
        color: #64748b;
        margin-top: 2px;
    }

    /* Diagnostic Badges */
    .diag-pill-pass {
        background: rgba(16, 185, 129, 0.12);
        color: #059669;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.70rem;
        font-weight: 700;
    }

    .diag-pill-warn {
        background: rgba(245, 158, 11, 0.12);
        color: #d97706;
        border: 1px solid rgba(245, 158, 11, 0.3);
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.70rem;
        font-weight: 700;
    }

    .diag-pill-fail {
        background: rgba(239, 68, 68, 0.12);
        color: #dc2626;
        border: 1px solid rgba(239, 68, 68, 0.3);
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.70rem;
        font-weight: 700;
    }

    /* Responsive Tab Navigation */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #f1f5f9;
        padding: 4px 8px;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        margin-bottom: 12px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 36px;
        background: transparent;
        border-radius: 6px;
        color: #64748b;
        font-size: 0.85rem;
        font-weight: 600;
        padding: 0 16px;
        border: 1px solid transparent;
    }

    .stTabs [aria-selected="true"] {
        background: #ffffff !important;
        color: #0284c7 !important;
        border: 1px solid #cbd5e1 !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    }

    /* Segmented Control Stage Inspector Styling */
    [data-testid="stSegmentedControl"] {
        background: #f1f5f9;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 4px;
        margin-bottom: 12px;
        width: 100%;
    }

    [data-testid="stSegmentedControl"] button {
        border-radius: 6px !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        transition: all 0.2s ease;
    }
</style>
"""


def apply_theme() -> None:
    """Injects custom CSS styling into the Streamlit app."""
    st.markdown(THEME_CSS, unsafe_allow_html=True)
