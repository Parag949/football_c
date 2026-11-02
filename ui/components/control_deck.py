"""
Unified Workspace Control Deck
==============================
Active analysis control panel: Frame navigation steppers, smooth timeline slider,
Convex Hull algorithm chooser, and a quick 'Reconfigure Setup' button.
"""

from typing import Dict, Any, Optional
import numpy as np
import streamlit as st
from ui.state import VIDEO_SOURCES


def render_control_deck(telemetry: Dict[str, Any]) -> str:
    """
    Renders the streamlined timeline and tactical control deck.
    Returns the selected hull algorithm key ('graham' or 'jarvis').
    """
    total_frames = telemetry.get("total_frames", 1)
    fps_val = telemetry.get("fps", 25.0)
    active_src_title = telemetry.get("title", "Match Footage")

    cam_mode = "Manual Mouse" if st.session_state.manual_calib_enabled else st.session_state.transformer.camera_mode.upper()
    cond_H = np.linalg.cond(st.session_state.transformer.H)

    st.markdown('<div class="control-deck">', unsafe_allow_html=True)

    # Row 1: Active Configuration Telemetry & Algorithmic Controls
    r1_col1, r1_col2, r1_col3, r1_col4 = st.columns([1.4, 2.1, 1.2, 0.9])

    with r1_col1:
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:8px; margin-top:4px;">
            <span style="font-weight:700; color:#f8fafc; font-size:0.85rem;">📹 {active_src_title}</span>
            <span class="badge-pill" style="background:#1e293b; color:#38bdf8; border:1px solid #334155; font-size:0.70rem;">
                {cam_mode} (κ={cond_H:.1f})
            </span>
        </div>
        """, unsafe_allow_html=True)

    with r1_col2:
        # Layer toggles to select and deselect Convex Hull, Voronoi, Dijkstra Corridor, and Passing Graph
        t_c1, t_c2, t_c3, t_c4 = st.columns([1.0, 1.1, 1.1, 1.1])
        with t_c1:
            st.session_state.show_hull = st.checkbox(
                "🛡️ Hull",
                value=st.session_state.get('show_hull', True),
                key="ctrl_toggle_hull",
                help="Toggle Defensive Convex Hull compactness shield"
            )
        with t_c2:
            st.session_state.show_voronoi = st.checkbox(
                "🌐 Voronoi",
                value=st.session_state.get('show_voronoi', True),
                key="ctrl_toggle_voronoi",
                help="Toggle Voronoi pitch dominance territorial partitions"
            )
        with t_c3:
            st.session_state.show_dijkstra = st.checkbox(
                "⚡ Dijkstra",
                value=st.session_state.get('show_dijkstra', True),
                key="ctrl_toggle_dijkstra",
                help="Toggle Dijkstra lowest-risk penetrative passing corridor"
            )
        with t_c4:
            st.session_state.show_passing_graph = st.checkbox(
                "🕸️ Graph",
                value=st.session_state.get('show_passing_graph', False),
                key="ctrl_toggle_passing_graph",
                disabled=not st.session_state.show_dijkstra,
                help="Toggle full candidate passing network graph G=(V, E)"
            )

    with r1_col3:
        hull_algo_choice = st.selectbox(
            "Convex Hull Algorithm",
            ["Graham Scan O(N log N)", "Jarvis March O(N·H)"],
            index=0 if "graham" in st.session_state.get("ui_hull_algo_val", "graham") else 1,
            key="ui_hull_algo",
            label_visibility="collapsed",
            disabled=not st.session_state.show_hull
        )
        hull_algo_key = "graham" if "Graham" in hull_algo_choice else "jarvis"
        st.session_state.ui_hull_algo_val = hull_algo_key

    with r1_col4:
        if st.button("⚙️ Setup", key="reconfig_setup_btn", width="stretch", help="Change video source or recalibrate camera"):
            st.session_state.setup_confirmed = False
            st.rerun()

    # Row 2: Instant Frame Stepping Buttons
    b1, b2, b3, b4, b5, b6 = st.columns(6)

    def step_frame(delta: int, to_abs: Optional[int] = None):
        if to_abs is not None:
            new_f = max(0, min(total_frames - 1, to_abs))
        else:
            new_f = max(0, min(total_frames - 1, st.session_state.current_frame + delta))
        st.session_state.current_frame = new_f
        st.session_state.main_slider_val = new_f

    with b1:
        if st.button("⏮ First Frame", width="stretch", key="nav_first"):
            step_frame(0, to_abs=0)
            st.rerun()
    with b2:
        if st.button("⏪ -10 Frames", width="stretch", key="nav_back10"):
            step_frame(-10)
            st.rerun()
    with b3:
        if st.button("◀ -1 Frame", width="stretch", key="nav_back1"):
            step_frame(-1)
            st.rerun()
    with b4:
        if st.button("▶ +1 Frame", width="stretch", key="nav_fwd1"):
            step_frame(1)
            st.rerun()
    with b5:
        if st.button("⏩ +10 Frames", width="stretch", key="nav_fwd10"):
            step_frame(10)
            st.rerun()
    with b6:
        if st.button("⏭ Last Frame", width="stretch", key="nav_last"):
            step_frame(0, to_abs=total_frames - 1)
            st.rerun()

    # Row 3: Smooth Timeline Slider (Native Streamlit on_change sync, no double reruns!)
    cur_sec = st.session_state.current_frame / fps_val
    tot_sec = total_frames / fps_val
    slider_label = f"Timeline: Frame {st.session_state.current_frame + 1} / {total_frames}  ({cur_sec:.1f}s / {tot_sec:.1f}s)"

    def on_slider_change():
        st.session_state.current_frame = st.session_state.main_slider_val

    st.slider(
        slider_label,
        min_value=0,
        max_value=max(1, total_frames - 1),
        value=st.session_state.current_frame,
        step=1,
        key="main_slider_val",
        on_change=on_slider_change
    )

    st.markdown('</div>', unsafe_allow_html=True)

    return hull_algo_key
