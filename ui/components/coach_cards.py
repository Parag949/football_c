"""
Tactical Intelligence Coach Cards
=================================
Displays live coach analytics: Voronoi pitch dominance, defensive block height,
compactness index, and inter-team centroid distance.
"""

from typing import Dict, Any
import numpy as np
import streamlit as st


def render_coach_cards(
    geometry: Any,
    positions: Any,
    coach_data: Dict[str, Any]
) -> None:
    """Renders 5 tactical intelligence stat cards respecting active layer selections."""
    show_hull = st.session_state.get('show_hull', True)
    show_voronoi = st.session_state.get('show_voronoi', True)
    show_dijkstra = st.session_state.get('show_dijkstra', True)
    show_pitch_control = st.session_state.get('show_pitch_control', False)

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        if (show_voronoi or show_pitch_control) and geometry is not None:
            v_metric = f"{getattr(geometry, 'team_a_pct', 50.0):.1f}% vs {getattr(geometry, 'team_b_pct', 50.0):.1f}%"
            v_sub = "Pitch Control Field" if show_pitch_control else "Voronoi Space Control"
            v_col = "#38bdf8"
        else:
            v_metric = "OFF"
            v_sub = "Hidden"
            v_col = "#64748b"

        st.markdown(f"""
        <div class="coach-card">
            <div class="coach-title">Space Control</div>
            <div class="coach-metric" style="color:{v_col};">{v_metric}</div>
            <div class="coach-sub">{v_sub}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        raw_b = coach_data.get('block', {}) if isinstance(coach_data, dict) else {}
        c_block = raw_b if isinstance(raw_b, dict) else {}
        if show_hull:
            b_metric = f"{c_block.get('line_height_m', 35.0)}m"
            b_sub = f"{c_block.get('block_type', 'Mid Block')} • Risk: {c_block.get('tactical_risk', 'Low')}"
            b_col = "#10b981"
        else:
            b_metric = "OFF"
            b_sub = "Hidden"
            b_col = "#64748b"

        st.markdown(f"""
        <div class="coach-card">
            <div class="coach-title">Defensive Line Height</div>
            <div class="coach-metric" style="color:{b_col};">{b_metric}</div>
            <div class="coach-sub">{b_sub}</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        if show_hull and geometry is not None:
            h_metric = f"{getattr(geometry, 'hull_area', 0.0):.1f} m²"
            h_sub = f"Index: {getattr(geometry, 'hull_compactness', 0.0):.2f} ({c_block.get('compactness_verdict', 'Balanced')})"
            h_col = "#f59e0b"
        else:
            h_metric = "OFF"
            h_sub = "Hidden"
            h_col = "#64748b"

        st.markdown(f"""
        <div class="coach-card">
            <div class="coach-title">Defensive Compactness</div>
            <div class="coach-metric" style="color:{h_col};">{h_metric}</div>
            <div class="coach-sub">{h_sub}</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        # Expected Threat (xT) Tactical Intelligence Card
        show_xt = st.session_state.get('show_xt', True)
        ball = getattr(positions, 'ball', None) if positions is not None else None
        xt_val = 0.0
        threat_zone = "Neutral"
        try:
            from algorithms.expected_threat import get_point_xt, XT_GRID_16x12
            if ball is not None and len(ball) >= 2:
                xt_val = get_point_xt(float(ball[0]), float(ball[1]))
            else:
                xt_val = float(np.mean(XT_GRID_16x12[:, -4:]))
            if xt_val >= 0.15:
                threat_zone = "High Threat (Golden Zone)"
            elif xt_val >= 0.06:
                threat_zone = "Creation Zone"
            else:
                threat_zone = "Build-Up Zone"
        except Exception:
            xt_val = 0.05
            threat_zone = "Build-Up Zone"

        if show_xt:
            p_metric = f"{xt_val:.3f} xT"
            p_sub = threat_zone
            p_col = "#f59e0b"
        else:
            p_metric = "OFF"
            p_sub = "Hidden"
            p_col = "#64748b"

        st.markdown(f"""
        <div class="coach-card">
            <div class="coach-title">Expected Threat (xT)</div>
            <div class="coach-metric" style="color:{p_col};">{p_metric}</div>
            <div class="coach-sub">{p_sub}</div>
        </div>
        """, unsafe_allow_html=True)

    with c5:
        t_a = getattr(positions, 'team_a', None) if positions is not None else None
        t_b = getattr(positions, 'team_b', None) if positions is not None else None
        try:
            from algorithms.formation_detector import detect_team_formation
            f_a = detect_team_formation(t_a, "right")['formation'] if (t_a and len(t_a) >= 4) else "4-3-3"
            f_b = detect_team_formation(t_b, "left")['formation'] if (t_b and len(t_b) >= 4) else "4-4-2"
        except Exception:
            f_a, f_b = "4-3-3", "4-4-2"

        st.markdown(f"""
        <div class="coach-card">
            <div class="coach-title">Detected Formations</div>
            <div class="coach-metric" style="color:#a855f7; font-size:1.0rem;">{f_a} vs {f_b}</div>
            <div class="coach-sub">Team A (Blue) vs Team B (Red)</div>
        </div>
        """, unsafe_allow_html=True)
