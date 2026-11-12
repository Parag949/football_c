"""
Top Application Branding & Telemetry Header
===========================================
Displays application title, ONNX acceleration state, precomputed cache status,
and real-time stream resolution/FPS pills.
"""

from typing import Dict, Any
import streamlit as st


def render_header(telemetry: Dict[str, Any]) -> None:
    """Renders the top branding and match stream summary bar."""
    cached_count = telemetry.get("cached_count", 0)
    vid_w = telemetry.get("width", 1280)
    vid_h = telemetry.get("height", 720)
    fps_val = telemetry.get("fps", 25.0)

    is_live = st.session_state.get('exec_mode') == 'live' or cached_count == 0
    cache_badge_class = "badge-live" if is_live else "badge-cache"
    cache_badge_text = "● Live Mode" if is_live else f"⚡ Ready ({cached_count} frames)"

    st.markdown(f"""
    <div class="top-bar">
        <div class="brand-title">
            <span>⚽ TacticalVision</span>
            <span style="font-size:0.80rem; font-weight:500; color:#94a3b8;">Match Intelligence & Pitch Mapping</span>
        </div>
        <div style="display:flex; align-items:center; gap:8px;">
            <span class="badge-pill {cache_badge_class}">{cache_badge_text}</span>
            <span class="badge-pill" style="background:#1e293b; color:#cbd5e1; border:1px solid #334155;">
                {vid_w}×{vid_h} • {fps_val:.0f} FPS
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

