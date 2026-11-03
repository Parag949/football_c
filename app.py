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
import time

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

    # Automated Native-FPS Frame Playback Ticker
    if st.session_state.get('is_playing', False):
        source_fps = max(1.0, float(telemetry.get('fps', 25.0)))
        playback_speed = max(0.1, float(st.session_state.get('playback_speed', 1.0)))
        effective_fps = source_fps * playback_speed
        target_interval = 1.0 / effective_fps

        # Sleep to strictly pace frames at native video FPS
        time.sleep(target_interval)

        total_frames = max(1, int(telemetry.get('total_frames', 180)))
        cur_frame = st.session_state.current_frame

        if cur_frame + 1 < total_frames:
            st.session_state.current_frame = cur_frame + 1
            st.rerun()
        elif st.session_state.get('playback_loop', True):
            st.session_state.current_frame = 0
            st.rerun()
        else:
            st.session_state.is_playing = False
            st.rerun()
