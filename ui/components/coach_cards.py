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

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        if show_voronoi and geometry is not None:
            v_metric = f"{getattr(geometry, 'team_a_pct', 50.0):.1f}% vs {getattr(geometry, 'team_b_pct', 50.0):.1f}%"
            v_sub = "Team A (Blue) vs Team B (Red)"
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
        raw_p = coach_data.get('passing', {}) if isinstance(coach_data, dict) else {}
        c_pass = raw_p if isinstance(raw_p, dict) else {}
        if show_dijkstra:
            prog_val = c_pass.get('progression_m', 0.0)
            p_metric = f"+{prog_val:.1f}m"
            is_break = c_pass.get('is_line_breaking', False)
            dir_label = "Line-Breaking" if is_break else c_pass.get('directive', 'Circulate')
            p_sub = f"{dir_label} • {c_pass.get('passes_count', 0)} passes"
            p_col = "#00f0ff" if is_break else "#38bdf8"
        else:
            p_metric = "OFF"
            p_sub = "Hidden"
            p_col = "#64748b"

        st.markdown(f"""
        <div class="coach-card">
            <div class="coach-title">Best Pass Gain</div>
            <div class="coach-metric" style="color:{p_col};">{p_metric}</div>
            <div class="coach-sub">{p_sub}</div>
        </div>
        """, unsafe_allow_html=True)

    with c5:
        dist_str = "N/A"
        t_a = getattr(positions, 'team_a', None) if positions is not None else None
        t_b = getattr(positions, 'team_b', None) if positions is not None else None
        if t_a and t_b:
            c_a = (np.mean([p[0] for p in t_a]), np.mean([p[1] for p in t_a]))
            c_b = (np.mean([p[0] for p in t_b]), np.mean([p[1] for p in t_b]))
            dist_str = f"{np.sqrt((c_a[0]-c_b[0])**2 + (c_a[1]-c_b[1])**2):.1f}m"

        st.markdown(f"""
        <div class="coach-card">
            <div class="coach-title">Team Separation</div>
            <div class="coach-metric" style="color:#a855f7;">{dist_str}</div>
            <div class="coach-sub">Avg. Team Distance</div>
        </div>
        """, unsafe_allow_html=True)
