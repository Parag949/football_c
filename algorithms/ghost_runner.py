"""
Ghost Runner Tactical Sandbox Simulation Engine (DAA What-If Analysis)
=====================================================================
Enables interactive what-if tactical simulations by introducing a synthetic
attacking or defending player ("Ghost Runner") into the real-world match state.

Mathematical Foundations:
1. Dynamic Graph Expansion: Adds Ghost node v_ghost to directed graph G=(V, E),
   generating candidate edges from ball carrier and teammates weighted by defensive
   interception pressure.
2. Shortest Path Re-Routing: Re-evaluates Dijkstra's Min-Heap algorithm in O((V+E) log V).
3. Metric Voronoi Tessellation: Recalculates Fortune's sweep-line spatial dominance
   to quantify territory claimed and disrupted space.
4. Calibrated Offside Boundary Verification: Assesses legality against opposition
   second-deepest defender with exact Euclidean margin of safety.
"""

import time
import math
from typing import List, Tuple, Dict, Any, Optional

from algorithms.passing_network import compute_dijkstra_passing_path
from algorithms.voronoi_dominance import compute_spatial_dominance


def resolve_calibrated_offside_line(
    defenders: List[Tuple[float, float]],
    attack_direction: str = "right"
) -> float:
    """
    Computes the metric X coordinate of the second-deepest defender.
    - If attacking right (towards X=105m): 2nd highest X defender.
    - If attacking left (towards X=0m): 2nd lowest X defender.
    """
    if not defenders:
        return 52.5

    sorted_defs = sorted(defenders, key=lambda p: p[0])
    if len(sorted_defs) >= 2:
        return sorted_defs[-2][0] if attack_direction == "right" else sorted_defs[1][0]
    return sorted_defs[0][0]


def simulate_ghost_runner(
    attacking_players: List[Tuple[float, float]],
    defending_players: List[Tuple[float, float]],
    ghost_coord: Tuple[float, float],
    carrier_idx: int = 0,
    ball_coord: Optional[Tuple[float, float]] = None,
    attack_direction: str = "right"
) -> Dict[str, Any]:
    """
    Runs an interactive what-if simulation by injecting a Ghost Runner into the tactical state.

    Returns:
    Dict containing:
      - ghost_coord: (X, Y)
      - is_offside: bool
      - offside_x: float
      - offside_margin_m: float (> 0 onside, < 0 offside)
      - passes_to_ghost: bool
      - progression_delta_m: float
      - baseline_dijkstra: Dict
      - simulated_dijkstra: Dict
      - territory_delta_pct: float
      - space_claimed_m2: float
      - tactical_verdict: str
      - runtime_ms: float
    """
    t0 = time.perf_counter()

    raw_gx = float(ghost_coord[0]) if math.isfinite(float(ghost_coord[0])) else 52.5
    raw_gy = float(ghost_coord[1]) if math.isfinite(float(ghost_coord[1])) else 34.0
    gx = float(max(0.0, min(105.0, raw_gx)))
    gy = float(max(0.0, min(68.0, raw_gy)))
    sanitized_ghost = (gx, gy)

    # 1. Calibrated Offside Boundary Verification (FIFA Law 11)
    offside_x = resolve_calibrated_offside_line(defending_players, attack_direction)
    in_attacking_half = (gx > 52.5) if attack_direction == "right" else (gx < 52.5)

    if in_attacking_half:
        if attack_direction == "right":
            offside_margin = offside_x - gx  # Positive = onside, Negative = offside
            is_offside = gx > offside_x
            # Law 11: Runner level with or behind the ball is strictly onside
            if ball_coord is not None and math.isfinite(ball_coord[0]) and gx <= float(ball_coord[0]):
                is_offside = False
                offside_margin = max(0.0, float(ball_coord[0]) - gx)
        else:
            offside_margin = gx - offside_x
            is_offside = gx < offside_x
            # Law 11: Runner level with or behind the ball is strictly onside
            if ball_coord is not None and math.isfinite(ball_coord[0]) and gx >= float(ball_coord[0]):
                is_offside = False
                offside_margin = max(0.0, gx - float(ball_coord[0]))
    else:
        # Own half is always legally onside
        offside_margin = abs(offside_x - gx)
        is_offside = False

    # 2. Baseline Tactical State (Without Ghost)
    baseline_dijk = compute_dijkstra_passing_path(
        attacking_players,
        defending_players,
        carrier_idx=carrier_idx,
        ball_coord=ball_coord,
        attack_direction=attack_direction
    )
    baseline_vor = compute_spatial_dominance(attacking_players, defending_players)

    # 3. Graph Expansion with Ghost Runner (v_ghost appended as index N)
    ghost_idx = len(attacking_players)
    sim_attacking = list(attacking_players) + [sanitized_ghost]

    sim_dijk = compute_dijkstra_passing_path(
        sim_attacking,
        defending_players,
        carrier_idx=carrier_idx,
        ball_coord=ball_coord,
        attack_direction=attack_direction
    )
    sim_vor = compute_spatial_dominance(sim_attacking, defending_players)

    # 4. Comparative Metrics & Deltas
    base_prog = float(baseline_dijk.get("total_progression", 0.0))
    sim_prog = float(sim_dijk.get("total_progression", 0.0))
    prog_delta = round(sim_prog - base_prog, 2)

    base_pct = float(baseline_vor.get("team_a_pct", 50.0))
    sim_pct = float(sim_vor.get("team_a_pct", 50.0))
    pct_delta = round(sim_pct - base_pct, 2)

    passes_to_ghost = ghost_idx in sim_dijk.get("path", [])

    # Find area claimed by ghost runner Voronoi cell
    ghost_cell_area = 0.0
    for cell in sim_vor.get("cells_a", []):
        if cell.get("player_idx") == ghost_idx:
            ghost_cell_area = round(float(cell.get("area_sq_m", 0.0)), 1)
            break

    # 5. Automated Coaching Tactical Verdict Generation
    if is_offside:
        verdict = f"🚨 OFFSIDE TRAP: Ghost runner is {abs(offside_margin):.2f}m beyond the defensive line (X={offside_x:.1f}m)."
    elif passes_to_ghost:
        verdict = f"🔥 HIGH-VALUE PENETRATIVE RUN: Re-routes optimal sequence with +{prog_delta:.1f}m forward progression!"
    elif pct_delta > 3.0:
        verdict = f"⚪ DECOY RUN: Does not receive pass, but stretches defense (+{pct_delta:.1f}% space gained, {ghost_cell_area:.0f} m² claimed)."
    else:
        verdict = f"⏸️ CONGESTED ZONE: Run is well-covered by defenders; minimal space gained ({ghost_cell_area:.0f} m²)."

    runtime_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "ghost_coord": sanitized_ghost,
        "is_offside": is_offside,
        "offside_x": round(offside_x, 2),
        "offside_margin_m": round(offside_margin, 2),
        "passes_to_ghost": passes_to_ghost,
        "ghost_player_idx": ghost_idx,
        "baseline_progression_m": base_prog,
        "simulated_progression_m": sim_prog,
        "progression_delta_m": prog_delta,
        "baseline_team_a_pct": base_pct,
        "simulated_team_a_pct": sim_pct,
        "territory_delta_pct": pct_delta,
        "space_claimed_m2": ghost_cell_area,
        "baseline_dijkstra": baseline_dijk,
        "simulated_dijkstra": sim_dijk,
        "tactical_verdict": verdict,
        "runtime_ms": round(runtime_ms, 3)
    }

