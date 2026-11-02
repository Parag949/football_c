"""
Top Application Branding & Telemetry Header
===========================================
Displays application title, ONNX acceleration state, precomputed cache status,
and real-time stream resolution/FPS pills.
"""

from typing import Dict, Any
import streamlit as st


def render_header(telemetry: Dict[str, Any]) -> None:
    """Renders the top branding and telemetry pill bar."""
    cached_count = telemetry.get("cached_count", 0)
    total_frames = telemetry.get("total_frames", 1)
    vid_w = telemetry.get("width", 1280)
    vid_h = telemetry.get("height", 720)
    fps_val = telemetry.get("fps", 25.0)

    is_live = st.session_state.get('exec_mode') == 'live' or cached_count == 0
    cache_badge_class = "badge-live" if is_live else "badge-cache"
    cache_badge_text = "● Pure Live Mode" if is_live else f"⚡ Cached {cached_count}f"

    import os
    model_name = getattr(st.session_state.get('detector', None), 'model_name', 'best_int8.onnx')
    model_base = os.path.basename(str(model_name))
    num_threads = getattr(st.session_state.get('detector', None), 'num_threads', None)
    thread_suffix = f" [{num_threads}T]" if num_threads else ""
    onnx_badge_text = f"🚀 {model_base}{thread_suffix}"

    st.markdown(f"""
    <div class="top-bar">
        <div class="brand-title">
            <span>⚽ TacticalVision</span>
            <span style="font-size:0.80rem; font-weight:500; color:#94a3b8;">// Laptop-Optimized Tactical Station</span>
        </div>
        <div style="display:flex; align-items:center; gap:8px;">
            <span class="badge-pill badge-onnx">{onnx_badge_text}</span>
            <span class="badge-pill {cache_badge_class}">{cache_badge_text}</span>
            <span class="badge-pill" style="background:#1e293b; color:#cbd5e1; border:1px solid #334155;">
                {vid_w}x{vid_h} • {fps_val:.1f} FPS
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

