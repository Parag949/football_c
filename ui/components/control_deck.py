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
                {cam_mode}
            </span>
        </div>
        """, unsafe_allow_html=True)

    with r1_col2:
        # Layer toggles to select and deselect Convex Hull, Voronoi, Dijkstra Corridor, Passing Graph, and Ruler
        t_c1, t_c2, t_c3, t_c4, t_c5 = st.columns([1.0, 1.1, 1.1, 1.0, 1.0])
        with t_c1:
            st.session_state.show_hull = st.checkbox(
                "🛡️ Block",
                value=st.session_state.get('show_hull', True),
                key="ctrl_toggle_hull",
                help="Show defensive shape outline"
            )
        with t_c2:
            st.session_state.show_voronoi = st.checkbox(
                "🌐 Space",
                value=st.session_state.get('show_voronoi', True),
                key="ctrl_toggle_voronoi",
                help="Show team space control zones"
            )
        with t_c3:
            st.session_state.show_dijkstra = st.checkbox(
                "⚡ Best Pass",
                value=st.session_state.get('show_dijkstra', True),
                key="ctrl_toggle_dijkstra",
                help="Show recommended forward passing lane"
            )
        with t_c4:
            st.session_state.show_passing_graph = st.checkbox(
                "🕸️ Pass Options",
                value=st.session_state.get('show_passing_graph', False),
                key="ctrl_toggle_passing_graph",
                disabled=not st.session_state.show_dijkstra,
                help="Show all open passing options between teammates"
            )
        with t_c5:
            st.session_state.show_ruler = st.checkbox(
                "📏 Ruler",
                value=st.session_state.get('show_ruler', False),
                key="ctrl_toggle_ruler",
                help="Measure player distances and check offside line"
            )

    with r1_col3:
        hull_algo_choice = st.selectbox(
            "Block Shape Style",
            ["Block Outline (Standard)", "Block Outline (Gift-Wrap)"],
            index=0 if "graham" in st.session_state.get("ui_hull_algo_val", "graham") else 1,
            key="ui_hull_algo",
            label_visibility="collapsed",
            disabled=not st.session_state.show_hull
        )
        hull_algo_key = "graham" if "Standard" in hull_algo_choice else "jarvis"
        st.session_state.ui_hull_algo_val = hull_algo_key

    with r1_col4:
        if st.button("⚙️ Setup", key="reconfig_setup_btn", width="stretch", help="Change video source or pitch alignment"):
            st.session_state.setup_confirmed = False
            st.rerun()

    # Interactive Distance & Offside Tool Drawer
    if st.session_state.get('show_ruler', False):
        with st.expander("📏 Distance & Offside Tool", expanded=True):
            r_c1, r_c2, r_c3 = st.columns([1.3, 1.4, 1.3])
            with r_c1:
                mode_opts = ["Team A Teammates", "Inter-Team Separation", "Ball to Player", "Custom Coordinates"]
                mode_map = {"Team A Teammates": "players", "Inter-Team Separation": "inter_team", "Ball to Player": "ball", "Custom Coordinates": "custom"}
                curr_label = [k for k, v in mode_map.items() if v == st.session_state.get('ruler_mode', 'players')]
                chosen_mode = st.selectbox(
                    "Measurement Mode",
                    mode_opts,
                    index=mode_opts.index(curr_label[0]) if curr_label else 0,
                    key="ui_ruler_mode_choice"
                )
                st.session_state.ruler_mode = mode_map[chosen_mode]

            with r_c2:
                if st.session_state.ruler_mode in ["players", "inter_team", "ball"]:
                    rc_a, rc_b = st.columns(2)
                    with rc_a:
                        st.session_state.ruler_p1_idx = st.number_input(
                            "Player A #",
                            min_value=0, max_value=15,
                            value=int(st.session_state.get('ruler_p1_idx', 0)),
                            step=1,
                            key="ui_ruler_p1"
                        )
                    with rc_b:
                        st.session_state.ruler_p2_idx = st.number_input(
                            "Player B #",
                            min_value=0, max_value=15,
                            value=int(st.session_state.get('ruler_p2_idx', 1)),
                            step=1,
                            key="ui_ruler_p2"
                        )
                else:
                    rc_x1, rc_y1 = st.columns(2)
                    with rc_x1:
                        mx1 = st.slider("Point A: X (m)", 0.0, 105.0, float(st.session_state.get('ruler_custom_p1', (35.0, 20.0))[0]), step=0.5, key="ui_r_x1")
                    with rc_y1:
                        my1 = st.slider("Point A: Y (m)", 0.0, 68.0, float(st.session_state.get('ruler_custom_p1', (35.0, 20.0))[1]), step=0.5, key="ui_r_y1")
                    st.session_state.ruler_custom_p1 = (mx1, my1)

                    rc_x2, rc_y2 = st.columns(2)
                    with rc_x2:
                        mx2 = st.slider("Point B: X (m)", 0.0, 105.0, float(st.session_state.get('ruler_custom_p2', (65.0, 45.0))[0]), step=0.5, key="ui_r_x2")
                    with rc_y2:
                        my2 = st.slider("Point B: Y (m)", 0.0, 68.0, float(st.session_state.get('ruler_custom_p2', (65.0, 45.0))[1]), step=0.5, key="ui_r_y2")
                    st.session_state.ruler_custom_p2 = (mx2, my2)

            with r_c3:
                off_auto = st.checkbox("Auto-Track 2nd Deepest Defender", value=st.session_state.get('ruler_auto_offside', True), key="ui_ruler_auto_off")
                st.session_state.ruler_auto_offside = off_auto
                if not off_auto:
                    st.session_state.ruler_manual_offside_x = st.slider(
                        "Manual Offside X (m)",
                        0.0, 105.0,
                        float(st.session_state.get('ruler_manual_offside_x', 70.0)),
                        step=0.5,
                        key="ui_ruler_man_off_x"
                    )

    # Row 2: Unified Timeline Bar (Playback Engine + Steppers + Scrubbing Slider)
    cur_sec = st.session_state.current_frame / fps_val
    tot_sec = total_frames / fps_val
    time_badge = f"Frame {st.session_state.current_frame + 1} / {total_frames}  ({cur_sec:.1f}s / {tot_sec:.1f}s)"

    c_controls, c_slider = st.columns([1.35, 2.65])

    def step_frame(delta: int, to_abs: Optional[int] = None):
        if to_abs is not None:
            new_f = max(0, min(total_frames - 1, to_abs))
        else:
            new_f = max(0, min(total_frames - 1, st.session_state.current_frame + delta))
        st.session_state.current_frame = new_f
        st.session_state.main_slider_val = new_f

    def on_slider_change():
        st.session_state.current_frame = st.session_state.main_slider_val

    def toggle_playback():
        if not st.session_state.get('is_playing', False):
            if st.session_state.current_frame >= total_frames - 1:
                st.session_state.current_frame = 0
                st.session_state.main_slider_val = 0
            st.session_state.is_playing = True
        else:
            st.session_state.is_playing = False

    if "main_slider_val" not in st.session_state or st.session_state.main_slider_val != st.session_state.current_frame:
        st.session_state.main_slider_val = st.session_state.current_frame

    with c_controls:
        rb = telemetry.get('ring_buffer', {})
        rb_count = rb.get('count', 0)
        rb_pct = rb.get('hit_ratio_pct', 0.0)
        rb_str = f" • ⚡ RAM: {rb_count}f ({rb_pct:.0f}%)" if rb_count > 0 else ""
        st.markdown(
            f"<div style='font-size:0.75rem; color:#94a3b8; font-weight:600; margin-bottom:3px; display:flex; justify-content:space-between;'>"
            f"<span>⏱️ {time_badge}</span>"
            f"<span style='color:#38bdf8;'>{fps_val:.1f} FPS{rb_str}</span>"
            f"</div>",
            unsafe_allow_html=True
        )
        c_play, c_steps = st.columns([1.1, 2.8])
        with c_play:
            is_play = st.session_state.get('is_playing', False)
            btn_label = "⏸ Pause" if is_play else "⏵ Play"
            btn_type = "primary" if is_play else "secondary"
            st.button(
                btn_label,
                width="stretch",
                key="nav_play_pause",
                help=f"Play/Pause at Native Video FPS ({fps_val:.1f} FPS)",
                type=btn_type,
                on_click=toggle_playback
            )
        with c_steps:
            sb1, sb2, sb3, sb4, sb5, sb6 = st.columns(6)
            with sb1:
                st.button("⏮", width="stretch", key="nav_first", help="First Frame", on_click=step_frame, args=(0, 0))
            with sb2:
                st.button("⏪", width="stretch", key="nav_back10", help="-10 Frames", on_click=step_frame, args=(-10, None))
            with sb3:
                st.button("◀", width="stretch", key="nav_back1", help="-1 Frame", on_click=step_frame, args=(-1, None))
            with sb4:
                st.button("▶", width="stretch", key="nav_fwd1", help="+1 Frame", on_click=step_frame, args=(1, None))
            with sb5:
                st.button("⏩", width="stretch", key="nav_fwd10", help="+10 Frames", on_click=step_frame, args=(10, None))
            with sb6:
                st.button("⏭", width="stretch", key="nav_last", help="Last Frame", on_click=step_frame, args=(0, total_frames - 1))

    with c_slider:
        cs_lbl, cs_spd, cs_loop = st.columns([2.0, 1.1, 0.9])
        with cs_lbl:
            st.markdown("<div style='font-size:0.75rem; color:#94a3b8; font-weight:600; margin-top:4px;'>Timeline Scrubbing</div>", unsafe_allow_html=True)
        with cs_spd:
            curr_spd = float(st.session_state.get('playback_speed', 1.0))
            spd_idx = 1 if curr_spd == 1.0 else (0 if curr_spd == 0.5 else 2)
            st.session_state.playback_speed = st.selectbox(
                "Speed",
                [0.5, 1.0, 2.0],
                index=spd_idx,
                format_func=lambda x: f"{x}x Speed",
                key="ui_playback_spd",
                label_visibility="collapsed"
            )
        with cs_loop:
            st.session_state.playback_loop = st.checkbox(
                "🔁 Loop",
                value=st.session_state.get('playback_loop', True),
                key="ui_playback_loop"
            )

        st.slider(
            "Timeline Scrub",
            min_value=0,
            max_value=max(1, total_frames - 1),
            step=1,
            key="main_slider_val",
            on_change=on_slider_change,
            label_visibility="collapsed"
        )

    st.markdown('</div>', unsafe_allow_html=True)

    return hull_algo_key
