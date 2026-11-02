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
    """Renders 4 tactical intelligence stat cards."""
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="coach-card">
            <div class="coach-title">Voronoi Pitch Control</div>
            <div class="coach-metric" style="color:#38bdf8;">{geometry.team_a_pct:.1f}% vs {geometry.team_b_pct:.1f}%</div>
            <div class="coach-sub">Team A (Blue) vs Team B (Red)</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        c_block = coach_data['block']
        st.markdown(f"""
        <div class="coach-card">
            <div class="coach-title">Defensive Block Height</div>
            <div class="coach-metric" style="color:#10b981;">{c_block['line_height_m']}m</div>
            <div class="coach-sub">{c_block['block_type']} • Risk: {c_block['tactical_risk']}</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="coach-card">
            <div class="coach-title">Defensive Compactness</div>
            <div class="coach-metric" style="color:#f59e0b;">{geometry.hull_area:.1f} m²</div>
            <div class="coach-sub">Index: {geometry.hull_compactness:.2f} ({c_block['compactness_verdict']})</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        dist_str = "N/A"
        if positions.team_a and positions.team_b:
            c_a = (np.mean([p[0] for p in positions.team_a]), np.mean([p[1] for p in positions.team_a]))
            c_b = (np.mean([p[0] for p in positions.team_b]), np.mean([p[1] for p in positions.team_b]))
            dist_str = f"{np.sqrt((c_a[0]-c_b[0])**2 + (c_a[1]-c_b[1])**2):.1f}m"

        st.markdown(f"""
        <div class="coach-card">
            <div class="coach-title">Team Separation</div>
            <div class="coach-metric" style="color:#a855f7;">{dist_str}</div>
            <div class="coach-sub">Inter-Centroid Distance</div>
        </div>
        """, unsafe_allow_html=True)
