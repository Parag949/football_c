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
                st.session_state.sandbox_state["team_a"] = [[round(p[0], 2), round(p[1], 2)] for p in t_a]
                st.session_state.sandbox_state["team_b"] = [[round(p[0], 2), round(p[1], 2)] for p in (t_b or [])]
                st.session_state.sandbox_state["ball"] = [round(ball_pt[0], 2), round(ball_pt[1], 2)] if ball_pt else [52.5, 34.0]
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

    html_code = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        :root {{
            --bg-base: #0a0f1d;
            --bg-surface: #111827;
            --bg-elevated: #1e293b;
            --border-color: #334155;
            --accent-blue: #38bdf8;
            --accent-green: #10b981;
            --accent-red: #ef4444;
            --accent-amber: #f59e0b;
            --accent-purple: #c084fc;
            --accent-pink: #f472b6;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
        }}
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }}
        body {{
            background-color: var(--bg-base);
            color: var(--text-main);
            overflow-x: hidden;
            padding: 10px;
        }}
        .board-container {{
            display: flex;
            flex-direction: column;
            gap: 10px;
            max-width: 1200px;
            margin: 0 auto;
        }}
        /* HUD STATUS */
        .hud-bar {{
            background: var(--bg-surface);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 8px 16px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 0.82rem;
        }}
        .hud-badge {{
            background: rgba(16, 185, 129, 0.15);
            border: 1px solid var(--accent-green);
            color: var(--accent-green);
            padding: 4px 10px;
            border-radius: 4px;
            font-weight: 600;
            letter-spacing: 0.03em;
        }}
        .hud-metrics {{
            display: flex;
            gap: 18px;
            color: var(--text-muted);
            font-size: 0.78rem;
        }}
        .hud-metrics b {{
            color: var(--text-main);
            font-family: ui-monospace, monospace;
        }}

        /* CANVAS STAGE CARD */
        .stage-card {{
            background: var(--bg-surface);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
        }}
        .stage-header {{
            padding: 8px 14px;
            background: #0f172a;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 0.82rem;
            font-weight: 600;
        }}
        .stage-viewport {{
            position: relative;
            width: 100%;
            height: 580px;
            background: #064e3b;
            cursor: default;
            user-select: none;
            touch-action: none;
        }}
        #sandbox-canvas {{
            display: block;
            width: 100%;
            height: 100%;
        }}

        /* TOOLBAR CHIPS */
        .toolbar-strip {{
            background: var(--bg-surface);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 8px 12px;
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            justify-content: space-between;
            gap: 10px;
        }}
        .chips-group {{
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            align-items: center;
        }}
        .layer-chip {{
            background: var(--bg-elevated);
            border: 1px solid var(--border-color);
            color: var(--text-muted);
            padding: 4px 10px;
            font-size: 0.72rem;
            font-weight: 600;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.15s ease-in-out;
            user-select: none;
        }}
        .layer-chip:hover {{
            background: #334155;
            color: #ffffff;
        }}
        .layer-chip.active {{
            background: rgba(56, 189, 248, 0.20);
            border-color: var(--accent-blue);
            color: var(--accent-blue);
        }}
        .layer-chip.dijk.active {{
            background: rgba(16, 185, 129, 0.20);
            border-color: var(--accent-green);
            color: var(--accent-green);
        }}
        .layer-chip.graph.active {{
            background: rgba(245, 158, 11, 0.20);
            border-color: var(--accent-amber);
            color: #fbbf24;
        }}
        .layer-chip.ghost.active {{
            background: rgba(168, 85, 247, 0.20);
            border-color: var(--accent-purple);
            color: #c084fc;
        }}
        .layer-chip.ruler.active {{
            background: rgba(236, 72, 153, 0.20);
            border-color: #ec4899;
            color: #f472b6;
        }}

        .action-btn {{
            background: #1e293b;
            border: 1px solid #475569;
            color: #cbd5e1;
            padding: 4px 9px;
            font-size: 0.72rem;
            font-weight: 600;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.15s;
        }}
        .action-btn:hover {{
            background: #334155;
            color: #ffffff;
            border-color: var(--accent-blue);
        }}
        .action-btn.danger:hover {{
            border-color: var(--accent-red);
            color: var(--accent-red);
        }}
        .action-btn.success:hover {{
            border-color: var(--accent-green);
            color: var(--accent-green);
        }}

        /* COACH ANALYTICS CARDS */
        .coach-mini-bar {{
            display: grid;
            grid-template-columns: repeat(6, 1fr);
            gap: 8px;
        }}
        .mini-card {{
            background: var(--bg-surface);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: 6px 10px;
            display: flex;
            flex-direction: column;
        }}
        .mini-card-title {{
            font-size: 0.65rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }}
        .mini-card-value {{
            font-size: 0.88rem;
            font-weight: 700;
            color: var(--text-main);
            font-family: ui-monospace, monospace;
        }}
    </style>
</head>
<body>

    <div class="board-container">
        <!-- HUD STATUS -->
        <div class="hud-bar">
            <div class="hud-badge">
                🎯 Interactive Tactical Sandbox (Real-Time 60 FPS DAA Math)
            </div>
            <div class="hud-metrics">
                <span>Selected: <b id="hud-selected">None</b></span>
                <span>Coordinates: <b id="hud-coords">[52.5m, 34.0m]</b></span>
                <span>Pitch: <b>105m x 68m (FIFA Standard)</b></span>
            </div>
        </div>

        <!-- STAGE CANVAS -->
        <div class="stage-card">
            <div class="stage-header">
                <div style="display:flex; align-items:center; gap:10px;">
                    <span>🗺️ 2D Tactical Whiteboard</span>
                    <span style="color:var(--text-muted); font-size:0.72rem; font-weight:normal;">
                        (Drag players, ball, ghost runner, or ruler pins. Click any teammate to pass & assign carrier.)
                    </span>
                </div>
                <div style="display:flex; align-items:center; gap:8px;">
                    <button id="btn-add-a" class="action-btn">➕ Attacker</button>
                    <button id="btn-add-b" class="action-btn">➕ Defender</button>
                    <button id="btn-delete" class="action-btn danger">🗑️ Remove</button>
                    <button id="btn-export" class="action-btn success" title="Export current whiteboard tactic as PNG">📸 Export PNG</button>
                </div>
            </div>
            <div id="stage-box" class="stage-viewport">
                <canvas id="sandbox-canvas"></canvas>
            </div>
        </div>

        <!-- LAYER TOGGLES & TOOLS -->
        <div class="toolbar-strip">
            <div class="chips-group">
                <span style="font-size:0.72rem; color:var(--text-muted); margin-right:4px;">TACTICAL LAYERS:</span>
                <span id="tog-vor" class="layer-chip active">🌐 Voronoi</span>
                <span id="tog-hull" class="layer-chip active">🛡️ Convex Hull</span>
                <span id="tog-dijk" class="layer-chip dijk active">⚡ Dijkstra Route</span>
                <span id="tog-graph" class="layer-chip graph active">🕸️ Passing Graph</span>
                <span id="tog-ghost" class="layer-chip ghost active">👻 Ghost Runner</span>
                <span id="tog-ruler" class="layer-chip ruler active">📏 Distance Ruler</span>
                <span id="tog-labels" class="layer-chip active">👕 Labels</span>
            </div>
            <div style="display:flex; gap:6px; align-items:center;">
                <span style="font-size:0.70rem; color:var(--text-muted);">Hull Algo:</span>
                <button id="btn-algo-graham" class="action-btn" style="border-color:var(--accent-blue); color:var(--accent-blue);">Graham O(N log N)</button>
                <button id="btn-algo-jarvis" class="action-btn">Jarvis O(N·H)</button>
            </div>
        </div>

        <!-- LIVE COACH ANALYTICS METRICS -->
        <div class="coach-mini-bar">
            <div class="mini-card">
                <span class="mini-card-title">Territory Dominance</span>
                <span id="stat-territory" class="mini-card-value" style="color:var(--accent-blue);">50% - 50%</span>
            </div>
            <div class="mini-card">
                <span class="mini-card-title">Defensive Hull Area</span>
                <span id="stat-hull-area" class="mini-card-value" style="color:var(--accent-red);">0.0 m²</span>
            </div>
            <div class="mini-card">
                <span class="mini-card-title">Hull Compactness</span>
                <span id="stat-hull-comp" class="mini-card-value">0.00</span>
            </div>
            <div class="mini-card">
                <span class="mini-card-title">Dijkstra Progression</span>
                <span id="stat-dijkstra" class="mini-card-value" style="color:var(--accent-green);">+0.0 m (0p)</span>
            </div>
            <div class="mini-card">
                <span class="mini-card-title">Ghost Runner Run</span>
                <span id="stat-ghost" class="mini-card-value" style="color:#c084fc;">Onside (+0.0m)</span>
            </div>
            <div class="mini-card">
                <span class="mini-card-title">Tactical Ruler</span>
                <span id="stat-ruler" class="mini-card-value" style="color:#f472b6;">0.0 m</span>
            </div>
        </div>
    </div>

    <script>
        const INITIAL_STATE = {state_payload};

        // Board entities (metric coordinates [0, 105]m x [0, 68]m)
        let teamA = JSON.parse(JSON.stringify(INITIAL_STATE.team_a || []));
        let teamB = JSON.parse(JSON.stringify(INITIAL_STATE.team_b || []));
        let ball = JSON.parse(JSON.stringify(INITIAL_STATE.ball || [52.5, 34.0]));
        let carrierIdx = INITIAL_STATE.carrier_idx !== undefined ? INITIAL_STATE.carrier_idx : 0;
        let ghostPos = JSON.parse(JSON.stringify(INITIAL_STATE.ghost || [85.0, 18.0]));
        let rulerA = JSON.parse(JSON.stringify(INITIAL_STATE.ruler_a || [40.0, 20.0]));
        let rulerB = JSON.parse(JSON.stringify(INITIAL_STATE.ruler_b || [70.0, 18.0]));

        // Feature toggle flags
        let showVoronoi = true;
        let showHull = true;
        let showDijkstra = true;
        let showGraph = true;
        let showGhost = true;
        let showRuler = true;
        let showLabels = true;
        let hullAlgo = 'graham'; // 'graham' or 'jarvis'

        // Interaction state
        let draggingEntity = null; // {{ type: 'a'|'b'|'ball'|'ghost'|'ruler_a'|'ruler_b', index: int }}
        let selectedEntity = null;
        let dragOffset = [0, 0];

        const canvas = document.getElementById('sandbox-canvas');
        const ctx = canvas.getContext('2d');
        const stageBox = document.getElementById('stage-box');

        function resizeCanvas() {{
            const dpr = window.devicePixelRatio || 1;
            const w = stageBox.clientWidth || 1080;
            const h = stageBox.clientHeight || 580;
            canvas.width = Math.round(w * dpr);
            canvas.height = Math.round(h * dpr);
            canvas.style.width = w + "px";
            canvas.style.height = h + "px";
            renderBoard();
        }}
        window.addEventListener('resize', resizeCanvas);
        if (window.ResizeObserver) {{
            new ResizeObserver(() => resizeCanvas()).observe(stageBox);
        }}

        // Coordinate transforms
        function getPitchDimensions() {{
            const dpr = window.devicePixelRatio || 1;
            const W = canvas.width;
            const H = canvas.height;
            const pad = 24 * dpr;
            const pw = W - 2 * pad;
            const ph = H - 2 * pad;
            return {{ W, H, pad, pw, ph, dpr }};
        }}

        function toCanvas(x, y) {{
            const {{ pad, pw, ph }} = getPitchDimensions();
            return [
                pad + (x / 105.0) * pw,
                pad + (y / 68.0) * ph
            ];
        }}

        function toPitch(cx, cy) {{
            const {{ pad, pw, ph }} = getPitchDimensions();
            const px = ((cx - pad) / pw) * 105.0;
            const py = ((cy - pad) / ph) * 68.0;
            return [
                Math.max(0.5, Math.min(104.5, px)),
                Math.max(0.5, Math.min(67.5, py))
            ];
        }}

        // ==========================================
        // DAA PURE MATHEMATICAL ALGORITHMS (CLIENT)
        // ==========================================

        // 1. Cross product for 2D orientation
        function crossProduct(o, a, b) {{
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
        }}

        // Convex Hull: Graham Scan (Monotone Chain O(N log N)) & Jarvis March O(N*H)
        function computeConvexHull(pts, algo) {{
            if (!pts || pts.length < 3) return pts || [];
            
            // Deduplicate points
            const unique = [];
            const seen = new Set();
            for (let p of pts) {{
                const k = p[0].toFixed(2) + "_" + p[1].toFixed(2);
                if (!seen.has(k)) {{
                    seen.add(k);
                    unique.push(p);
                }}
            }}
            if (unique.length < 3) return unique;

            if (algo === 'jarvis') {{
                // Jarvis March / Gift Wrapping O(N * H)
                let n = unique.length;
                let leftmost = 0;
                for (let i = 1; i < n; i++) {{
                    if (unique[i][0] < unique[leftmost][0] || (unique[i][0] === unique[leftmost][0] && unique[i][1] < unique[leftmost][1])) {{
                        leftmost = i;
                    }}
                }}
                let hull = [];
                let p = leftmost;
                let iter = 0;
                do {{
                    hull.push(unique[p]);
                    let q = (p + 1) % n;
                    for (let i = 0; i < n; i++) {{
                        let cp = crossProduct(unique[p], unique[q], unique[i]);
                        if (cp > 1e-7 || (Math.abs(cp) <= 1e-7 && Math.hypot(unique[i][0] - unique[p][0], unique[i][1] - unique[p][1]) > Math.hypot(unique[q][0] - unique[p][0], unique[q][1] - unique[p][1]))) {{
                            q = i;
                        }}
                    }}
                    p = q;
                    iter++;
                }} while (p !== leftmost && iter < n + 2);
                return hull;
            }} else {{
                // Graham Scan via Andrew's Monotone Chain O(N log N)
                let sorted = unique.slice().sort((a, b) => a[0] === b[0] ? a[1] - b[1] : a[0] - b[0]);
                let lower = [];
                for (let i = 0; i < sorted.length; i++) {{
                    while (lower.length >= 2 && crossProduct(lower[lower.length - 2], lower[lower.length - 1], sorted[i]) <= 0) {{
                        lower.pop();
                    }}
                    lower.push(sorted[i]);
                }}
                let upper = [];
                for (let i = sorted.length - 1; i >= 0; i--) {{
                    while (upper.length >= 2 && crossProduct(upper[upper.length - 2], upper[upper.length - 1], sorted[i]) <= 0) {{
                        upper.pop();
                    }}
                    upper.push(sorted[i]);
                }}
                lower.pop();
                upper.pop();
                return lower.concat(upper);
            }}
        }}

        function polygonArea(poly) {{
            if (!poly || poly.length < 3) return 0.0;
            let area = 0.0;
            for (let i = 0; i < poly.length; i++) {{
                let j = (i + 1) % poly.length;
                area += poly[i][0] * poly[j][1];
                area -= poly[j][0] * poly[i][1];
            }}
            return Math.abs(area) / 2.0;
        }}

        function polygonPerimeter(poly) {{
            if (!poly || poly.length < 2) return 0.0;
            let perim = 0.0;
            for (let i = 0; i < poly.length; i++) {{
                let j = (i + 1) % poly.length;
                perim += Math.hypot(poly[j][0] - poly[i][0], poly[j][1] - poly[i][1]);
            }}
            return perim;
        }}

        // 2. Voronoi Territorial Dominance via Sutherland-Hodgman Pitch Boundary Halfplane Clipping
        function clipPolygonByHalfPlane(poly, pi, pj) {{
            let dx = pj[0] - pi[0];
            let dy = pj[1] - pi[1];
            let mx = (pi[0] + pj[0]) / 2.0;
            let my = (pi[1] + pj[1]) / 2.0;

            function dist(p) {{
                return (p[0] - mx) * dx + (p[1] - my) * dy;
            }}

            let out = [];
            if (!poly || poly.length === 0) return out;
            let s = poly[poly.length - 1];
            let ds = dist(s);

            for (let i = 0; i < poly.length; i++) {{
                let e = poly[i];
                let de = dist(e);
                if (de <= 1e-7) {{
                    if (ds > 1e-7) {{
                        let t = -ds / (de - ds);
                        out.push([s[0] + t * (e[0] - s[0]), s[1] + t * (e[1] - s[1])]);
                    }}
                    out.push(e);
                }} else if (ds <= 1e-7) {{
                    let t = -ds / (de - ds);
                    out.push([s[0] + t * (e[0] - s[0]), s[1] + t * (e[1] - s[1])]);
                }}
                s = e;
                ds = de;
            }}
            return out;
        }}

        function computeVoronoiTessellation(teamA, teamB) {{
            const allSites = [];
            teamA.forEach((p, idx) => allSites.push({{ team: 'a', idx: idx, pt: p }}));
            teamB.forEach((p, idx) => allSites.push({{ team: 'b', idx: idx, pt: p }}));

            const pitchBounds = [[0.0, 0.0], [105.0, 0.0], [105.0, 68.0], [0.0, 68.0]];
            const cellsA = [];
            const cellsB = [];
            let areaA = 0.0;
            let areaB = 0.0;

            for (let i = 0; i < allSites.length; i++) {{
                let poly = pitchBounds.slice();
                let pi = allSites[i].pt;
                for (let j = 0; j < allSites.length; j++) {{
                    if (i === j) continue;
                    let pj = allSites[j].pt;
                    if (Math.hypot(pj[0] - pi[0], pj[1] - pi[1]) < 0.01) continue;
                    poly = clipPolygonByHalfPlane(poly, pi, pj);
                    if (poly.length < 3) break;
                }}
                if (poly.length >= 3) {{
                    let a = polygonArea(poly);
                    if (allSites[i].team === 'a') {{
                        cellsA.push(poly);
                        areaA += a;
                    }} else {{
                        cellsB.push(poly);
                        areaB += a;
                    }}
                }}
            }}

            const total = Math.max(0.1, areaA + areaB);
            const aPct = Math.round((areaA / total) * 1000) / 10;
            const bPct = Math.round((100.0 - aPct) * 10) / 10;

            return {{
                cellsA: cellsA,
                cellsB: cellsB,
                areaA: Math.round(areaA),
                areaB: Math.round(areaB),
                aPct: aPct,
                bPct: bPct
            }};
        }}

        // 3. Dijkstra Shortest-Path Passing Network & Passing Graph
        function pointToSegmentDistance(pt, p1, p2) {{
            let dx = p2[0] - p1[0];
            let dy = p2[1] - p1[1];
            let l2 = dx * dx + dy * dy;
            if (l2 < 1e-6) return Math.hypot(pt[0] - p1[0], pt[1] - p1[1]);
            let t = Math.max(0.0, Math.min(1.0, ((pt[0] - p1[0]) * dx + (pt[1] - p1[1]) * dy) / l2));
            let px = p1[0] + t * dx;
            let py = p1[1] + t * dy;
            return Math.hypot(pt[0] - px, pt[1] - py);
        }}

        function computeDijkstra(teamA, teamB, cIdx) {{
            if (!teamA || teamA.length < 2) return {{ path: [], edges: [], prog: 0.0, cost: 0.0 }};
            let n = teamA.length;
            let start = (cIdx !== undefined && cIdx >= 0 && cIdx < n) ? cIdx : 0;

            // Target is most forward attacker (maximum X) distinct from start
            let target = 0;
            let maxX = -1;
            for (let i = 0; i < n; i++) {{
                if (i !== start && teamA[i][0] > maxX) {{
                    maxX = teamA[i][0];
                    target = i;
                }}
            }}
            if (target === start && n > 1) target = (start + 1) % n;

            // Build weighted graph
            let dists = new Array(n).fill(Infinity);
            let prev = new Array(n).fill(-1);
            let visited = new Array(n).fill(false);
            let edges = [];

            dists[start] = 0;

            for (let u = 0; u < n; u++) {{
                for (let v = 0; v < n; v++) {{
                    if (u === v) continue;
                    let d = Math.hypot(teamA[v][0] - teamA[u][0], teamA[v][1] - teamA[u][1]);
                    if (d > 45.0) continue; // Max pass range

                    let pressure = 0.0;
                    for (let b = 0; b < teamB.length; b++) {{
                        let dDef = pointToSegmentDistance(teamB[b], teamA[u], teamA[v]);
                        if (dDef < 7.0) pressure += (7.0 - dDef) / (dDef + 1.0);
                    }}
                    let prog = teamA[v][0] - teamA[u][0];
                    let weight = Math.max(0.1, d * (1.0 + 2.0 * pressure) - 0.3 * prog + 10.0);
                    edges.push({{
                        s: u, t: v,
                        sp: teamA[u], tp: teamA[v],
                        w: Math.round(weight * 10) / 10,
                        dist: Math.round(d * 10) / 10,
                        pressure: pressure
                    }});
                }}
            }}

            // Dijkstra priority scan
            for (let step = 0; step < n; step++) {{
                let u = -1;
                let minDist = Infinity;
                for (let i = 0; i < n; i++) {{
                    if (!visited[i] && dists[i] < minDist) {{
                        minDist = dists[i];
                        u = i;
                    }}
                }}
                if (u === -1 || u === target) break;
                visited[u] = true;

                for (let e of edges) {{
                    if (e.s === u) {{
                        let v = e.t;
                        if (!visited[v] && dists[u] + e.w < dists[v]) {{
                            dists[v] = dists[u] + e.w;
                            prev[v] = u;
                        }}
                    }}
                }}
            }}

            let path = [];
            let curr = target;
            if (dists[target] < Infinity) {{
                while (curr !== -1) {{
                    path.unshift(curr);
                    curr = prev[curr];
                }}
            }}

            let totalProg = 0.0;
            if (path.length >= 2) {{
                totalProg = teamA[path[path.length - 1]][0] - teamA[path[0]][0];
            }}

            return {{
                path: path,
                edges: edges,
                prog: Math.round(Math.max(0, totalProg) * 10) / 10,
                cost: dists[target] < Infinity ? Math.round(dists[target] * 10) / 10 : 0.0
            }};
        }}

        // 4. Ghost Runner VAR Offside Check (FIFA Law 11 Compliant)
        function computeGhostTelemetry(teamA, teamB, cIdx, ghost, ballPt) {{
            let sortedX = (teamB || []).map(p => p[0]).sort((a, b) => b - a);
            let defOffsideX = 52.5;
            if (sortedX.length >= 2) {{
                defOffsideX = sortedX[1];
            }} else if (sortedX.length === 1) {{
                defOffsideX = sortedX[0];
            }}
            // Effective offside line: must be in opponent half (>= 52.5m), and behind ball if ball is ahead
            let effectiveOffsideX = Math.max(52.5, defOffsideX);
            if (ballPt && ballPt[0] > effectiveOffsideX) {{
                effectiveOffsideX = ballPt[0];
            }}

            let isOffside = ghost[0] > (effectiveOffsideX + 0.1);
            let carrier = (cIdx !== undefined && teamA[cIdx]) ? teamA[cIdx] : (teamA[0] || [52.5, 34.0]);
            let prog = ghost[0] - carrier[0];
            let dist = Math.hypot(ghost[0] - carrier[0], ghost[1] - carrier[1]);

            let verdict = "Optimal Corridor";
            if (isOffside) verdict = "Offside Trap";
            else if (prog > 12.0) verdict = "High-Value Run";
            else if (prog > 0.0) verdict = "Progressive Option";
            else verdict = "Decoy / Reset";

            return {{
                offsideX: Math.round(effectiveOffsideX * 10) / 10,
                isOffside: isOffside,
                carrier: carrier,
                prog: Math.round(prog * 10) / 10,
                dist: Math.round(dist * 10) / 10,
                verdict: verdict
            }};
        }}

        // 5. Tactical Ruler Distance Calculation
        function computeRulerTelemetry(pA, pB) {{
            let dx = pB[0] - pA[0];
            let dy = pB[1] - pA[1];
            let dist = Math.hypot(dx, dy);
            let angle = Math.atan2(dy, dx) * 180 / Math.PI;
            return {{
                dist: Math.round(dist * 10) / 10,
                dx: Math.round(dx * 10) / 10,
                dy: Math.round(dy * 10) / 10,
                angle: Math.round(angle)
            }};
        }}

        // ==========================================
        // CANVAS RENDERING ENGINE
        // ==========================================
        function drawTacticalBadge(c, x, y, text, color, bgColor) {{
            c.save();
            c.font = "bold 10px monospace";
            const tw = c.measureText(text).width + 14;
            const th = 18;
            const bx = x - tw / 2;
            const by = y - th;
            c.fillStyle = bgColor || "rgba(15, 23, 42, 0.90)";
            c.strokeStyle = color;
            c.lineWidth = 1.5;
            c.beginPath();
            if (typeof c.roundRect === 'function') {{
                c.roundRect(bx, by, tw, th, 4);
            }} else {{
                c.rect(bx, by, tw, th);
            }}
            c.fill();
            c.stroke();
            c.fillStyle = color;
            c.textAlign = "center";
            c.textBaseline = "middle";
            c.fillText(text, x, by + th / 2);
            c.restore();
        }}

        function renderBoard() {{
            const {{ W, H, pad, pw, ph, dpr }} = getPitchDimensions();
            ctx.clearRect(0, 0, W, H);

            // Turf background
            ctx.fillStyle = '#064e3b';
            ctx.fillRect(0, 0, W, H);

            // Pitch markings
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.7)';
            ctx.lineWidth = 2 * dpr;

            // Pitch boundary
            ctx.strokeRect(pad, pad, pw, ph);

            // Halfway line
            ctx.beginPath();
            ctx.moveTo(pad + pw / 2, pad);
            ctx.lineTo(pad + pw / 2, pad + ph);
            ctx.stroke();

            // Center circle
            const cRad = (9.15 / 105.0) * pw;
            ctx.beginPath();
            ctx.arc(pad + pw / 2, pad + ph / 2, cRad, 0, Math.PI * 2);
            ctx.stroke();

            // Penalty areas
            const pbW = (16.5 / 105.0) * pw;
            const pbH = (40.32 / 68.0) * ph;
            const pbY = pad + (ph - pbH) / 2;
            ctx.strokeRect(pad, pbY, pbW, pbH);
            ctx.strokeRect(pad + pw - pbW, pbY, pbW, pbH);

            // Goal boxes
            const gbW = (5.5 / 105.0) * pw;
            const gbH = (18.32 / 68.0) * ph;
            const gbY = pad + (ph - gbH) / 2;
            ctx.strokeRect(pad, gbY, gbW, gbH);
            ctx.strokeRect(pad + pw - gbW, gbY, gbW, gbH);

            // 1. Voronoi Dominance Overlay (Full Pitch Vector Tessellation)
            if (showVoronoi) {{
                const vor = computeVoronoiTessellation(teamA, teamB);
                document.getElementById('stat-territory').innerText = vor.aPct + "% - " + vor.bPct + "%";

                ctx.save();
                // Clip strictly to pitch boundary
                ctx.beginPath();
                ctx.rect(pad, pad, pw, ph);
                ctx.clip();

                // Team A Voronoi Cells (Blue)
                ctx.fillStyle = 'rgba(56, 189, 248, 0.18)';
                ctx.strokeStyle = 'rgba(56, 189, 248, 0.50)';
                ctx.lineWidth = 1 * dpr;
                vor.cellsA.forEach(poly => {{
                    if (poly.length < 3) return;
                    ctx.beginPath();
                    let p0 = toCanvas(poly[0][0], poly[0][1]);
                    ctx.moveTo(p0[0], p0[1]);
                    for (let k = 1; k < poly.length; k++) {{
                        let pk = toCanvas(poly[k][0], poly[k][1]);
                        ctx.lineTo(pk[0], pk[1]);
                    }}
                    ctx.closePath();
                    ctx.fill();
                    ctx.stroke();
                }});

                // Team B Voronoi Cells (Red)
                ctx.fillStyle = 'rgba(244, 63, 94, 0.18)';
                ctx.strokeStyle = 'rgba(244, 63, 94, 0.50)';
                ctx.lineWidth = 1 * dpr;
                vor.cellsB.forEach(poly => {{
                    if (poly.length < 3) return;
                    ctx.beginPath();
                    let p0 = toCanvas(poly[0][0], poly[0][1]);
                    ctx.moveTo(p0[0], p0[1]);
                    for (let k = 1; k < poly.length; k++) {{
                        let pk = toCanvas(poly[k][0], poly[k][1]);
                        ctx.lineTo(pk[0], pk[1]);
                    }}
                    ctx.closePath();
                    ctx.fill();
                    ctx.stroke();
                }});

                ctx.restore();
            }} else {{
                document.getElementById('stat-territory').innerText = "--";
            }}

            // 2. Convex Hull (Team B Defensive Block & Team A Attacking Shape)
            if (showHull) {{
                // Team B Defensive Block (Red)
                if (teamB.length >= 3) {{
                    const hullB = computeConvexHull(teamB, hullAlgo);
                    if (hullB && hullB.length >= 3) {{
                        const area = polygonArea(hullB);
                        const perim = polygonPerimeter(hullB);
                        const comp = perim > 0 ? (4 * Math.PI * area) / (perim * perim) : 0;

                        document.getElementById('stat-hull-area').innerText = Math.round(area) + " m²";
                        document.getElementById('stat-hull-comp').innerText = comp.toFixed(2);

                        ctx.save();
                        ctx.fillStyle = 'rgba(239, 68, 68, 0.20)';
                        ctx.strokeStyle = 'rgba(239, 68, 68, 0.85)';
                        ctx.lineWidth = 2 * dpr;
                        ctx.setLineDash([4 * dpr, 4 * dpr]);
                        ctx.beginPath();
                        let p0 = toCanvas(hullB[0][0], hullB[0][1]);
                        ctx.moveTo(p0[0], p0[1]);
                        for (let i = 1; i < hullB.length; i++) {{
                            let pk = toCanvas(hullB[i][0], hullB[i][1]);
                            ctx.lineTo(pk[0], pk[1]);
                        }}
                        ctx.closePath();
                        ctx.fill();
                        ctx.stroke();
                        ctx.restore();
                    }}
                }} else {{
                    document.getElementById('stat-hull-area').innerText = "--";
                    document.getElementById('stat-hull-comp').innerText = "--";
                }}

                // Team A Attacking Shape Hull (Cyan Blue)
                if (teamA.length >= 3) {{
                    const hullA = computeConvexHull(teamA, hullAlgo);
                    if (hullA && hullA.length >= 3) {{
                        ctx.save();
                        ctx.fillStyle = 'rgba(56, 189, 248, 0.10)';
                        ctx.strokeStyle = 'rgba(56, 189, 248, 0.60)';
                        ctx.lineWidth = 1.5 * dpr;
                        ctx.setLineDash([3 * dpr, 3 * dpr]);
                        ctx.beginPath();
                        let p0 = toCanvas(hullA[0][0], hullA[0][1]);
                        ctx.moveTo(p0[0], p0[1]);
                        for (let i = 1; i < hullA.length; i++) {{
                            let pk = toCanvas(hullA[i][0], hullA[i][1]);
                            ctx.lineTo(pk[0], pk[1]);
                        }}
                        ctx.closePath();
                        ctx.fill();
                        ctx.stroke();
                        ctx.restore();
                    }}
                }}
            }} else {{
                document.getElementById('stat-hull-area').innerText = "--";
                document.getElementById('stat-hull-comp').innerText = "--";
            }}

            // 3. Passing Graph & Dijkstra Route
            const dijk = computeDijkstra(teamA, teamB, carrierIdx);
            if (showGraph && dijk.edges) {{
                ctx.save();
                ctx.lineWidth = 1.5 * dpr;
                ctx.setLineDash([3 * dpr, 3 * dpr]);

                // Group edges by source player and select the top 3 passing options per player
                const playerTopEdges = {{}};
                dijk.edges.forEach(e => {{
                    if (!playerTopEdges[e.s]) playerTopEdges[e.s] = [];
                    playerTopEdges[e.s].push(e);
                }});

                Object.values(playerTopEdges).forEach(edgeList => {{
                    edgeList.sort((a, b) => a.w - b.w);
                    edgeList.slice(0, 3).forEach(e => {{
                        let p1 = toCanvas(e.sp[0], e.sp[1]);
                        let p2 = toCanvas(e.tp[0], e.tp[1]);
                        ctx.strokeStyle = e.pressure > 0.8 ? 'rgba(239, 68, 68, 0.45)' : 'rgba(251, 191, 36, 0.55)';
                        ctx.beginPath();
                        ctx.moveTo(p1[0], p1[1]);
                        ctx.lineTo(p2[0], p2[1]);
                        ctx.stroke();
                    }});
                }});
                ctx.restore();
            }}

            if (showDijkstra && dijk.path && dijk.path.length >= 2) {{
                document.getElementById('stat-dijkstra').innerText = 
                    "+" + dijk.prog + " m (" + (dijk.path.length - 1) + "p)";
                ctx.save();
                ctx.strokeStyle = '#10b981';
                ctx.lineWidth = 3.5 * dpr;
                ctx.shadowColor = '#10b981';
                ctx.shadowBlur = 8 * dpr;
                ctx.beginPath();
                let firstPt = toCanvas(teamA[dijk.path[0]][0], teamA[dijk.path[0]][1]);
                ctx.moveTo(firstPt[0], firstPt[1]);
                for (let i = 1; i < dijk.path.length; i++) {{
                    let pt = toCanvas(teamA[dijk.path[i]][0], teamA[dijk.path[i]][1]);
                    ctx.lineTo(pt[0], pt[1]);
                }}
                ctx.stroke();

                // Draw intermediate node dots
                for (let i = 0; i < dijk.path.length; i++) {{
                    let pt = toCanvas(teamA[dijk.path[i]][0], teamA[dijk.path[i]][1]);
                    ctx.beginPath();
                    ctx.arc(pt[0], pt[1], 4 * dpr, 0, Math.PI * 2);
                    ctx.fillStyle = '#10b981';
                    ctx.fill();
                }}
                ctx.restore();
            }} else {{
                document.getElementById('stat-dijkstra').innerText = "--";
            }}

            // 4. Ghost Runner Simulation
            if (showGhost) {{
                const gt = computeGhostTelemetry(teamA, teamB, carrierIdx, ghostPos, ball);
                const gColor = gt.isOffside ? '#ef4444' : '#c084fc';
                document.getElementById('stat-ghost').innerText = 
                    (gt.isOffside ? "OFFSIDE" : "Onside") + " (+" + gt.prog + "m)";
                document.getElementById('stat-ghost').style.color = gColor;

                // Calibrated VAR line
                const offLine = toCanvas(gt.offsideX, 0);
                ctx.save();
                ctx.strokeStyle = gt.isOffside ? 'rgba(239, 68, 68, 0.85)' : 'rgba(56, 189, 248, 0.75)';
                ctx.lineWidth = 1.5 * dpr;
                ctx.setLineDash([4 * dpr, 4 * dpr]);
                ctx.beginPath();
                ctx.moveTo(offLine[0], pad);
                ctx.lineTo(offLine[0], pad + ph);
                ctx.stroke();

                ctx.fillStyle = gt.isOffside ? 'rgba(239, 68, 68, 0.9)' : 'rgba(56, 189, 248, 0.9)';
                ctx.font = (8 * dpr) + 'px sans-serif';
                ctx.textAlign = 'right';
                ctx.fillText('VAR Line: ' + gt.offsideX + 'm', offLine[0] - 4, pad + 12 * dpr);
                ctx.restore();

                // Pass corridor from carrier to ghost
                const crpt = toCanvas(gt.carrier[0], gt.carrier[1]);
                const grpt = toCanvas(ghostPos[0], ghostPos[1]);
                ctx.save();
                ctx.strokeStyle = gColor;
                ctx.lineWidth = 2.5 * dpr;
                ctx.setLineDash([4 * dpr, 3 * dpr]);
                ctx.beginPath();
                ctx.moveTo(crpt[0], crpt[1]);
                ctx.lineTo(grpt[0], grpt[1]);
                ctx.stroke();
                ctx.restore();

                // Ghost node
                const pr = 8 * dpr;
                ctx.save();
                ctx.beginPath();
                ctx.arc(grpt[0], grpt[1], pr * 1.5, 0, Math.PI * 2);
                ctx.fillStyle = gt.isOffside ? 'rgba(239, 68, 68, 0.25)' : 'rgba(168, 85, 247, 0.35)';
                ctx.fill();

                ctx.beginPath();
                ctx.arc(grpt[0], grpt[1], pr * 1.1, 0, Math.PI * 2);
                ctx.fillStyle = gColor;
                ctx.fill();
                ctx.strokeStyle = '#ffffff';
                ctx.lineWidth = 2 * dpr;
                ctx.stroke();

                ctx.fillStyle = '#ffffff';
                ctx.font = 'bold ' + (10 * dpr) + 'px sans-serif';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText('👻', grpt[0], grpt[1]);
                ctx.restore();

                drawTacticalBadge(ctx, grpt[0], grpt[1] - 18 * dpr, gt.verdict + " (+" + gt.prog + "m)", gColor);
            }} else {{
                document.getElementById('stat-ghost').innerText = "Disabled";
                document.getElementById('stat-ghost').style.color = "var(--text-muted)";
            }}

            // 5. Tactical Ruler Measurement
            if (showRuler) {{
                const rt = computeRulerTelemetry(rulerA, rulerB);
                document.getElementById('stat-ruler').innerText = rt.dist + " m (ΔX: " + rt.dx + "m)";

                const ra = toCanvas(rulerA[0], rulerA[1]);
                const rb = toCanvas(rulerB[0], rulerB[1]);

                ctx.save();
                ctx.strokeStyle = '#ec4899';
                ctx.lineWidth = 2.5 * dpr;
                ctx.setLineDash([5 * dpr, 3 * dpr]);
                ctx.beginPath();
                ctx.moveTo(ra[0], ra[1]);
                ctx.lineTo(rb[0], rb[1]);
                ctx.stroke();
                ctx.restore();

                // Pin A (Origin)
                ctx.beginPath();
                ctx.arc(ra[0], ra[1], 7 * dpr, 0, Math.PI * 2);
                ctx.fillStyle = '#10b981';
                ctx.fill();
                ctx.strokeStyle = '#ffffff';
                ctx.lineWidth = 1.5 * dpr;
                ctx.stroke();

                // Pin B (Target)
                ctx.beginPath();
                ctx.arc(rb[0], rb[1], 7 * dpr, 0, Math.PI * 2);
                ctx.fillStyle = '#ec4899';
                ctx.fill();
                ctx.strokeStyle = '#ffffff';
                ctx.lineWidth = 1.5 * dpr;
                ctx.stroke();

                // Midpoint distance badge
                const mx = (ra[0] + rb[0]) / 2;
                const my = (ra[1] + rb[1]) / 2;
                drawTacticalBadge(ctx, mx, my - 10 * dpr, "📏 " + rt.dist + "m (" + rt.angle + "°)", "#f472b6");
            }} else {{
                document.getElementById('stat-ruler').innerText = "--";
            }}

            // 6. Player Nodes: Team A (Blue)
            const pr = 8 * dpr;
            teamA.forEach((p, idx) => {{
                let pt = toCanvas(p[0], p[1]);
                let isCarrier = (idx === carrierIdx);
                let isSel = (selectedEntity && selectedEntity.type === 'a' && selectedEntity.index === idx);

                if (isCarrier) {{
                    ctx.beginPath();
                    ctx.arc(pt[0], pt[1], pr * 1.6, 0, Math.PI * 2);
                    ctx.fillStyle = 'rgba(245, 158, 11, 0.40)';
                    ctx.fill();
                }}
                if (isSel) {{
                    ctx.beginPath();
                    ctx.arc(pt[0], pt[1], pr * 1.8, 0, Math.PI * 2);
                    ctx.strokeStyle = '#38bdf8';
                    ctx.lineWidth = 2 * dpr;
                    ctx.stroke();
                }}

                ctx.beginPath();
                ctx.arc(pt[0], pt[1], pr, 0, Math.PI * 2);
                ctx.fillStyle = '#38bdf8';
                ctx.fill();
                ctx.strokeStyle = '#ffffff';
                ctx.lineWidth = 1.5 * dpr;
                ctx.stroke();

                if (showLabels) {{
                    ctx.fillStyle = '#ffffff';
                    ctx.font = 'bold ' + (9 * dpr) + 'px sans-serif';
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';
                    ctx.fillText(String(idx + 1), pt[0], pt[1]);
                }}
                if (isCarrier) {{
                    drawTacticalBadge(ctx, pt[0], pt[1] - 16 * dpr, "⚽ CARRIER", "#f59e0b");
                }}
            }});

            // 7. Player Nodes: Team B (Red)
            teamB.forEach((p, idx) => {{
                let pt = toCanvas(p[0], p[1]);
                let isSel = (selectedEntity && selectedEntity.type === 'b' && selectedEntity.index === idx);

                if (isSel) {{
                    ctx.beginPath();
                    ctx.arc(pt[0], pt[1], pr * 1.8, 0, Math.PI * 2);
                    ctx.strokeStyle = '#f43f5e';
                    ctx.lineWidth = 2 * dpr;
                    ctx.stroke();
                }}

                ctx.beginPath();
                ctx.arc(pt[0], pt[1], pr, 0, Math.PI * 2);
                ctx.fillStyle = '#f43f5e';
                ctx.fill();
                ctx.strokeStyle = '#ffffff';
                ctx.lineWidth = 1.5 * dpr;
                ctx.stroke();

                if (showLabels) {{
                    ctx.fillStyle = '#ffffff';
                    ctx.font = 'bold ' + (9 * dpr) + 'px sans-serif';
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';
                    ctx.fillText(String(idx + 1), pt[0], pt[1]);
                }}
            }});

            // 8. Ball Node
            let bpt = toCanvas(ball[0], ball[1]);
            ctx.beginPath();
            ctx.arc(bpt[0], bpt[1], pr * 0.9, 0, Math.PI * 2);
            ctx.fillStyle = '#facc15';
            ctx.fill();
            ctx.strokeStyle = '#000000';
            ctx.lineWidth = 1.5 * dpr;
            ctx.stroke();
        }}

        // ==========================================
        // MOUSE & TOUCH DRAGGING EVENT HANDLERS
        // ==========================================
        function getEventPos(e) {{
            const rect = canvas.getBoundingClientRect();
            const dpr = window.devicePixelRatio || 1;
            const clientX = e.touches ? e.touches[0].clientX : e.clientX;
            const clientY = e.touches ? e.touches[0].clientY : e.clientY;
            return [
                (clientX - rect.left) * dpr,
                (clientY - rect.top) * dpr
            ];
        }}

        function findEntityAt(cx, cy) {{
            const {{ dpr }} = getPitchDimensions();
            const hitR = 18 * dpr;
            let closest = null;
            let minDist = hitR;

            function check(type, index, pt, rBonus = 0) {{
                let cpt = toCanvas(pt[0], pt[1]);
                let d = Math.hypot(cx - cpt[0], cy - cpt[1]);
                if (d <= hitR + rBonus && d < minDist) {{
                    minDist = d;
                    closest = {{ type: type, index: index, pt: pt }};
                }}
            }}

            // Priority: Team A & B Players (easier to grab)
            for (let i = 0; i < teamA.length; i++) check('a', i, teamA[i], 2 * dpr);
            for (let j = 0; j < teamB.length; j++) check('b', j, teamB[j], 2 * dpr);

            // Ghost Runner
            if (showGhost) check('ghost', 0, ghostPos, 2 * dpr);

            // Ruler Pins
            if (showRuler) {{
                check('ruler_a', 0, rulerA, 1 * dpr);
                check('ruler_b', 0, rulerB, 1 * dpr);
            }}

            // Match Ball
            check('ball', 0, ball, 0);

            return closest;
        }}

        function onPointerDown(e) {{
            const [cx, cy] = getEventPos(e);
            const found = findEntityAt(cx, cy);
            if (found) {{
                draggingEntity = found;
                selectedEntity = found;
                const pCoord = toPitch(cx, cy);
                dragOffset = [found.pt[0] - pCoord[0], found.pt[1] - pCoord[1]];

                // If Team A player clicked, make them carrier & pass ball to them
                if (found.type === 'a') {{
                    carrierIdx = found.index;
                    ball = [found.pt[0], found.pt[1]];
                }}
                canvas.style.cursor = 'grabbing';
                updateHud(found.type, found.index, found.pt);
                renderBoard();
            }} else {{
                selectedEntity = null;
                document.getElementById('hud-selected').innerText = "None";
                renderBoard();
            }}
        }}

        function onPointerMove(e) {{
            const [cx, cy] = getEventPos(e);
            const pCoord = toPitch(cx, cy);
            document.getElementById('hud-coords').innerText = "[" + pCoord[0].toFixed(1) + "m, " + pCoord[1].toFixed(1) + "m]";

            if (draggingEntity) {{
                let newX = Math.max(0.5, Math.min(104.5, pCoord[0] + dragOffset[0]));
                let newY = Math.max(0.5, Math.min(67.5, pCoord[1] + dragOffset[1]));

                if (draggingEntity.type === 'a') {{
                    teamA[draggingEntity.index] = [newX, newY];
                    if (draggingEntity.index === carrierIdx) ball = [newX, newY];
                }} else if (draggingEntity.type === 'b') {{
                    teamB[draggingEntity.index] = [newX, newY];
                }} else if (draggingEntity.type === 'ball') {{
                    ball = [newX, newY];
                }} else if (draggingEntity.type === 'ghost') {{
                    ghostPos = [newX, newY];
                }} else if (draggingEntity.type === 'ruler_a') {{
                    rulerA = [newX, newY];
                }} else if (draggingEntity.type === 'ruler_b') {{
                    rulerB = [newX, newY];
                }}

                updateHud(draggingEntity.type, draggingEntity.index, [newX, newY]);
                renderBoard();
            }} else {{
                const hover = findEntityAt(cx, cy);
                canvas.style.cursor = hover ? 'grab' : 'crosshair';
            }}
        }}

        function onPointerUp() {{
            draggingEntity = null;
            canvas.style.cursor = 'crosshair';
        }}

        function updateHud(type, idx, pt) {{
            let name = "Unknown";
            if (type === 'a') name = "Attacker A#" + (idx + 1) + (idx === carrierIdx ? " (Carrier ⚽)" : "");
            else if (type === 'b') name = "Defender B#" + (idx + 1);
            else if (type === 'ball') name = "Match Ball ⚽";
            else if (type === 'ghost') name = "Ghost Runner 👻";
            else if (type === 'ruler_a') name = "Ruler Pin A 📍";
            else if (type === 'ruler_b') name = "Ruler Pin B 📍";
            document.getElementById('hud-selected').innerText = name;
        }}

        // Mouse events
        canvas.addEventListener('mousedown', onPointerDown);
        window.addEventListener('mousemove', onPointerMove);
        window.addEventListener('mouseup', onPointerUp);

        // Touch events
        canvas.addEventListener('touchstart', (e) => {{ e.preventDefault(); onPointerDown(e); }}, {{ passive: false }});
        window.addEventListener('touchmove', (e) => {{ if (draggingEntity) e.preventDefault(); onPointerMove(e); }}, {{ passive: false }});
        window.addEventListener('touchend', onPointerUp);

        // Layer toggle buttons
        function setupToggle(id, getter, setter) {{
            const btn = document.getElementById(id);
            btn.addEventListener('click', () => {{
                setter(!getter());
                btn.classList.toggle('active', getter());
                renderBoard();
            }});
        }}

        setupToggle('tog-vor', () => showVoronoi, v => showVoronoi = v);
        setupToggle('tog-hull', () => showHull, v => showHull = v);
        setupToggle('tog-dijk', () => showDijkstra, v => showDijkstra = v);
        setupToggle('tog-graph', () => showGraph, v => showGraph = v);
        setupToggle('tog-ghost', () => showGhost, v => showGhost = v);
        setupToggle('tog-ruler', () => showRuler, v => showRuler = v);
        setupToggle('tog-labels', () => showLabels, v => showLabels = v);

        // Hull Algorithm toggles
        const btnGraham = document.getElementById('btn-algo-graham');
        const btnJarvis = document.getElementById('btn-algo-jarvis');
        btnGraham.addEventListener('click', () => {{
            hullAlgo = 'graham';
            btnGraham.style.borderColor = 'var(--accent-blue)';
            btnGraham.style.color = 'var(--accent-blue)';
            btnJarvis.style.borderColor = '#475569';
            btnJarvis.style.color = '#cbd5e1';
            renderBoard();
        }});
        btnJarvis.addEventListener('click', () => {{
            hullAlgo = 'jarvis';
            btnJarvis.style.borderColor = 'var(--accent-red)';
            btnJarvis.style.color = 'var(--accent-red)';
            btnGraham.style.borderColor = '#475569';
            btnGraham.style.color = '#cbd5e1';
            renderBoard();
        }});

        // Action buttons: Add / Remove / Export
        document.getElementById('btn-add-a').addEventListener('click', () => {{
            teamA.push([45.0, 34.0]);
            renderBoard();
        }});
        document.getElementById('btn-add-b').addEventListener('click', () => {{
            teamB.push([65.0, 34.0]);
            renderBoard();
        }});
        document.getElementById('btn-delete').addEventListener('click', () => {{
            if (selectedEntity) {{
                if (selectedEntity.type === 'a' && teamA.length > 2) {{
                    teamA.splice(selectedEntity.index, 1);
                    if (carrierIdx >= teamA.length) carrierIdx = 0;
                }} else if (selectedEntity.type === 'b' && teamB.length > 2) {{
                    teamB.splice(selectedEntity.index, 1);
                }}
                selectedEntity = null;
                document.getElementById('hud-selected').innerText = "None";
                renderBoard();
            }}
        }});
        document.getElementById('btn-export').addEventListener('click', () => {{
            const link = document.createElement('a');
            const ts = new Date().toISOString().slice(0, 19).replace(/[:T]/g, '_');
            link.download = 'tactical_sandbox_' + ts + '.png';
            link.href = canvas.toDataURL('image/png');
            link.click();
        }});

        // Initial launch & sizing
        setTimeout(resizeCanvas, 50);
    </script>
</body>
</html>
"""

    if hasattr(st, "iframe"):
        st.iframe(html_code, height=920, width="stretch")
    else:
        st.components.v1.html(html_code, height=920, scrolling=False)
