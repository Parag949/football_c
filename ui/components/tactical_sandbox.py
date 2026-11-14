"""
2D Tactical Sandbox & Interactive Coaching Simulation Studio
============================================================
A dedicated, standalone tactical whiteboard where coaches can freely position
and drag players to tinker with formations, passing combinations, and tactical
setups without interfering with the live video processing engine.

Integrates all core DAA algorithms:
1. Voronoi Spatial Dominance (Fortune's Sweep-Line & Bounded Pitch Partitions)
2. Convex Hull Defensive Cohesion (Graham Scan O(N log N) / Jarvis March)
3. Dijkstra Shortest-Path Passing Network (Binary Min-Heap Risk Optimization)
4. Passing Graph Topology (All Feasible Lanes & Directional Flow)
5. Ghost Runner Simulation (Calibrated VAR Offside Line & Space Exploitation)
6. Interactive Tactical Ruler (Euclidean Distance, Lateral/Longitudinal Offsets, Pass Angle)
"""

import os
import json
import math
from typing import Dict, Any, List, Optional
import streamlit as st
import numpy as np


# Standard tactical formations (105m x 68m pitch coordinates)
TACTICAL_PRESETS = {
    "4-3-3 Attack": {
        "team_a": [
            [12.0, 34.0],  # GK
            [28.0, 12.0],  # LB
            [25.0, 26.0],  # LCB
            [25.0, 42.0],  # RCB
            [28.0, 56.0],  # RB
            [42.0, 34.0],  # DM (Pivot)
            [55.0, 22.0],  # LCM
            [55.0, 46.0],  # RCM
            [76.0, 14.0],  # LW
            [80.0, 34.0],  # ST
            [76.0, 54.0],  # RW
        ],
        "team_b": [
            [94.0, 34.0],  # GK
            [80.0, 16.0],  # LB
            [82.0, 27.0],  # CB1
            [82.0, 41.0],  # CB2
            [80.0, 52.0],  # RB
            [66.0, 20.0],  # LM
            [68.0, 30.0],  # CM1
            [68.0, 38.0],  # CM2
            [66.0, 48.0],  # RM
            [52.0, 31.0],  # ST1
            [52.0, 37.0],  # ST2
        ],
        "ball": [55.0, 22.0],
        "carrier_idx": 6,
        "ghost": [85.0, 18.0],
        "ruler_a": [40.0, 20.0],
        "ruler_b": [70.0, 18.0]
    },
    "4-4-2 Mid-Block": {
        "team_a": [
            [10.0, 34.0], [24.0, 14.0], [22.0, 28.0], [22.0, 40.0], [24.0, 54.0],
            [38.0, 20.0], [38.0, 34.0], [38.0, 48.0], [54.0, 24.0], [58.0, 34.0], [54.0, 44.0]
        ],
        "team_b": [
            [95.0, 34.0], [75.0, 14.0], [76.0, 26.0], [76.0, 42.0], [75.0, 54.0],
            [62.0, 16.0], [63.0, 28.0], [63.0, 40.0], [62.0, 52.0], [48.0, 30.0], [48.0, 38.0]
        ],
        "ball": [38.0, 34.0],
        "carrier_idx": 6,
        "ghost": [70.0, 48.0],
        "ruler_a": [30.0, 20.0],
        "ruler_b": [50.0, 20.0]
    },
    "3-5-2 High Press": {
        "team_a": [
            [14.0, 34.0], [30.0, 20.0], [28.0, 34.0], [30.0, 48.0],
            [48.0, 10.0], [45.0, 26.0], [46.0, 34.0], [45.0, 42.0], [48.0, 58.0],
            [68.0, 28.0], [68.0, 40.0]
        ],
        "team_b": [
            [92.0, 34.0], [78.0, 16.0], [80.0, 28.0], [80.0, 40.0], [78.0, 52.0],
            [64.0, 24.0], [64.0, 34.0], [64.0, 44.0], [52.0, 20.0], [50.0, 34.0], [52.0, 48.0]
        ],
        "ball": [46.0, 34.0],
        "carrier_idx": 6,
        "ghost": [82.0, 22.0],
        "ruler_a": [35.0, 20.0],
        "ruler_b": [60.0, 20.0]
    },
    "Corner Overload": {
        "team_a": [
            [20.0, 34.0], [45.0, 34.0], [50.0, 20.0], [50.0, 48.0],
            [88.0, 28.0], [92.0, 32.0], [95.0, 38.0], [91.0, 44.0], [86.0, 48.0],
            [104.0, 67.0], [98.0, 58.0]
        ],
        "team_b": [
            [102.0, 34.0], [87.0, 30.0], [91.0, 30.0], [94.0, 34.0], [93.0, 38.0],
            [90.0, 42.0], [86.0, 44.0], [80.0, 34.0], [74.0, 22.0], [70.0, 46.0], [60.0, 34.0]
        ],
        "ball": [104.0, 67.0],
        "carrier_idx": 9,
        "ghost": [94.0, 31.0],
        "ruler_a": [90.0, 60.0],
        "ruler_b": [92.0, 36.0]
    }
}


def render_tactical_sandbox(telemetry: Dict[str, Any]) -> None:
    """Renders the interactive 2D Tactical Simulation whiteboard for coaches."""
    st.markdown("### 📋 2D Tactics Board")
    st.caption(
        "Drag and drop players, the ball, the simulated runner, or distance pins to test formations, passing lanes, and offside traps."
    )

    # Helper to apply preset formation data into session state
    def _apply_preset_to_state(preset_name: str) -> None:
        if preset_name not in TACTICAL_PRESETS:
            preset_name = "4-3-3 Attack"
        data = TACTICAL_PRESETS[preset_name]
        st.session_state.sandbox_state["preset"] = preset_name
        for k in ["team_a", "team_b", "ball", "ghost", "ruler_a", "ruler_b"]:
            st.session_state.sandbox_state[k] = json.loads(json.dumps(data[k]))
        st.session_state.sandbox_state["carrier_idx"] = int(data.get("carrier_idx", 0))

    # Initialize sandbox state in session state if not present
    if "sandbox_state" not in st.session_state:
        st.session_state.sandbox_state = {"imported_frame": None}
        _apply_preset_to_state("4-3-3 Attack")

    # Top action bar: Preset selection & Snapshot from active video
    sb_c1, sb_c2, sb_c3, sb_c4 = st.columns([2, 1.2, 1.2, 1])

    with sb_c1:
        preset_names = list(TACTICAL_PRESETS.keys())
        current_preset = st.session_state.sandbox_state.get("preset", preset_names[0])
        idx = preset_names.index(current_preset) if current_preset in preset_names else 0
        selected_preset = st.selectbox(
            "📋 Formation Preset",
            preset_names,
            index=idx,
            key="sandbox_preset_selector",
            help="Load standard tactical structures for rapid what-if scenario design."
        )
        if selected_preset != st.session_state.sandbox_state.get("preset"):
            _apply_preset_to_state(selected_preset)
            st.session_state.sandbox_state["imported_frame"] = None
            st.rerun()

    with sb_c2:
        cur_frame_num = st.session_state.get("current_frame", 0) + 1
        if st.button("📥 Import Video Frame", key="sandbox_import_frame_btn", help="Snapshot actual player positions from the active match video frame.", width="stretch"):
            from ui.state import execute_pipeline
            bundle = execute_pipeline(st.session_state.get("current_frame", 0), force_live=False)
            pos = bundle.get("positions") if bundle else None
            t_a = (getattr(pos, "team_a", None) or (bundle.get("team_a") if bundle else None))
            t_b = (getattr(pos, "team_b", None) or (bundle.get("team_b") if bundle else None))
            ball_pt = (getattr(pos, "ball", None) or (bundle.get("ball") if bundle else None))
            c_idx = (getattr(pos, "carrier_idx", None) or (bundle.get("carrier") if bundle else None))
            if t_a and len(t_a) >= 2:
                def _clamp_p(pt):
                    return [
                        round(max(0.5, min(104.5, float(pt[0]))), 2),
                        round(max(0.5, min(67.5, float(pt[1]))), 2)
                    ]
                st.session_state.sandbox_state["team_a"] = [_clamp_p(p) for p in t_a if len(p) >= 2]
                st.session_state.sandbox_state["team_b"] = [_clamp_p(p) for p in (t_b or []) if len(p) >= 2]
                st.session_state.sandbox_state["ball"] = _clamp_p(ball_pt) if ball_pt and len(ball_pt) >= 2 else [52.5, 34.0]
                st.session_state.sandbox_state["carrier_idx"] = c_idx if (c_idx is not None and 0 <= c_idx < len(t_a)) else 0
                st.session_state.sandbox_state["imported_frame"] = cur_frame_num
                st.toast(f"Loaded Frame #{cur_frame_num} players!", icon="✅")
                st.rerun()
            else:
                st.warning("No tracked players found in current video frame. Precompute cache required.")

    with sb_c3:
        if st.button("🔄 Reset to Default", key="sandbox_reset_btn", help="Reset board to default 4-3-3 formation.", width="stretch"):
            _apply_preset_to_state("4-3-3 Attack")
            st.session_state.sandbox_state["imported_frame"] = None
            st.rerun()

    with sb_c4:
        imported_note = st.session_state.sandbox_state.get("imported_frame")
        if imported_note:
            st.caption(f"Source: **Frame #{imported_note}**")
        else:
            st.caption("Board: **Preset Formation**")

    # Serialize initial state for JavaScript client engine
    state_payload = json.dumps({
        "team_a": st.session_state.sandbox_state["team_a"],
        "team_b": st.session_state.sandbox_state["team_b"],
        "ball": st.session_state.sandbox_state["ball"],
        "carrier_idx": st.session_state.sandbox_state["carrier_idx"],
        "ghost": st.session_state.sandbox_state["ghost"],
        "ruler_a": st.session_state.sandbox_state["ruler_a"],
        "ruler_b": st.session_state.sandbox_state["ruler_b"],
    })

    from ui.components.tactical_sandbox_canvas import build_sandbox_canvas_html
    html_code = build_sandbox_canvas_html(state_payload)

    if hasattr(st, "iframe"):
        st.iframe(html_code, height=920, width="stretch")
    else:
        st.components.v1.html(html_code, height=920, scrolling=False)
