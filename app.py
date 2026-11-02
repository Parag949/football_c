"""
TacticalVision — Unified Laptop-Optimized Tactical Analytics & Pipeline Debugger
================================================================================
A high-performance, modular tactical football workstation designed for smooth
laptop operation, low CPU/RAM overhead, and intuitive tactical analysis.

Enforces clear workflow precedence:
Step 1: Mandatory Setup Gate (Video Source -> Camera Calibration -> Execution Mode)
Step 2: Full Tactical & Pipeline Diagnostics Suite (Un-Nerfed 5-Stage Workstation)
"""

import os
import sys

# Force thread-safe single-threaded FFmpeg frame decoding in OpenCV
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "threads;1"

import streamlit as st


# Ensure root workspace is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from ui.theme import apply_theme
from ui.state import init_session_state, sync_video_source, execute_pipeline
from ui.components.header import render_header
from ui.components.setup_gate import render_setup_gate
from ui.components.control_deck import render_control_deck
from ui.views.diagnostics_view import render_diagnostics_view


# Page Configuration
st.set_page_config(
    page_title="TacticalVision Pro",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Apply Modular Design System (Theme & CSS)
apply_theme()

# Initialize Centralized State
init_session_state()

# Synchronize Video Stream & Precomputed Cache
telemetry = sync_video_source()

# Top Application Branding & Telemetry Header
render_header(telemetry)

# Precedence Gate: If mandatory options are not confirmed, render Setup Gate
if not st.session_state.setup_confirmed:
    render_setup_gate(telemetry)
else:
    # Full Tactical Suite Unlocked
    hull_algo_key = render_control_deck(telemetry)
    algo_name = "Graham Scan O(N log N)" if hull_algo_key == "graham" else "Jarvis March O(N·H)"

    # Execute / Retrieve Pipeline Data for Target Frame (Lazy & Fast)
    bundle = execute_pipeline(
        st.session_state.current_frame,
        force_live=False,
        hull_algo=hull_algo_key
    )

    # Comprehensive Tactical Pipeline & Diagnostics Suite (Un-Nerfed Edition)
    render_diagnostics_view(bundle, hull_algo_name=algo_name)
