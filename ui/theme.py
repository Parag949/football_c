"""
TacticalVision Design System & Theme
====================================
Glassmorphic, dark-mode CSS styling optimized for modern high-DPI laptop screens.
"""

import streamlit as st


THEME_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
    }

    #MainMenu, footer, header {visibility: hidden;}

    /* Top Application Bar */
    .top-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: linear-gradient(135deg, #111827 0%, #1e293b 100%);
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 10px 18px;
        margin-bottom: 12px;
    }

    .brand-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #ffffff;
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
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }

    .badge-cache {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .badge-live {
        background: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }

    /* Control Panel Box */
    .control-deck {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 12px;
    }

    /* Coach Stat Cards */
    .coach-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 12px 14px;
        height: 100%;
    }

    .coach-title {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        color: #94a3b8;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }

    .coach-metric {
        font-size: 1.35rem;
        font-weight: 700;
        color: #ffffff;
        font-family: 'JetBrains Mono', monospace;
    }

    .coach-sub {
        font-size: 0.75rem;
        color: #64748b;
        margin-top: 2px;
    }

    /* Diagnostic Badges */
    .diag-pill-pass {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.70rem;
        font-weight: 700;
    }

    .diag-pill-warn {
        background: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.3);
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.70rem;
        font-weight: 700;
    }

    .diag-pill-fail {
        background: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.70rem;
        font-weight: 700;
    }

    /* Responsive Tab Navigation */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #111827;
        padding: 4px 8px;
        border-radius: 8px;
        border: 1px solid #1f2937;
        margin-bottom: 12px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 36px;
        background: transparent;
        border-radius: 6px;
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 600;
        padding: 0 16px;
        border: 1px solid transparent;
    }

    .stTabs [aria-selected="true"] {
        background: #1e293b !important;
        color: #38bdf8 !important;
        border: 1px solid #334155 !important;
    }

    /* Segmented Control Stage Inspector Styling */
    [data-testid="stSegmentedControl"] {
        background: #111827;
        border: 1px solid #1f2937;
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
