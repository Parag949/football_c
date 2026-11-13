"""
FW-06: Interactive Tactical Distance & Calibrated Offside Ruler
==============================================================
Provides metric Euclidean spatial distance measurement, longitudinal depth,
lateral width offsets, and projective offside line calculation for tactical analysis.
"""

import math
from typing import Tuple, List, Dict, Any, Optional


def compute_tactical_ruler(
    point_a: Tuple[float, float],
    point_b: Tuple[float, float],
    label_a: str = "Point A",
    label_b: str = "Point B",
    offside_x: Optional[float] = None,
    attacking_players: Optional[List[Tuple[float, float]]] = None,
    defending_players: Optional[List[Tuple[float, float]]] = None,
    attack_direction: str = "right"  # "right": goal at X=105m; "left": goal at X=0m
) -> Dict[str, Any]:
    """
    Computes Euclidean distance, coordinate offsets, and calibrated offside line
    between two metric points on the FIFA pitch [0..105m] x [0..68m].

    Parameters:
    - point_a: Starting metric coordinate (X, Y)
    - point_b: Ending metric coordinate (X, Y)
    - label_a: Human-readable identifier for Point A (e.g., 'Player A#4')
    - label_b: Human-readable identifier for Point B (e.g., 'Player A#9')
    - offside_x: Optional explicit metric X position for calibrated offside line
    - attacking_players: Outfield positions for Team A
    - defending_players: Outfield positions for Team B
    - attack_direction: Pitch progression orientation

    Returns:
    Dict containing distance, dx, dy, angle, offside_x, and offside breach telemetry.
    """
    xa = float(point_a[0]) if math.isfinite(float(point_a[0])) else 0.0
    ya = float(point_a[1]) if math.isfinite(float(point_a[1])) else 0.0
    xb = float(point_b[0]) if math.isfinite(float(point_b[0])) else 0.0
    yb = float(point_b[1]) if math.isfinite(float(point_b[1])) else 0.0

    dx = xb - xa
    dy = yb - ya
    euclidean_dist = math.hypot(dx, dy)
    angle_deg = math.degrees(math.atan2(dy, dx))

    # Determine Calibrated Offside Line (X coordinate)
    resolved_offside_x = offside_x
    if resolved_offside_x is None:
        clean_defs = [
            (float(p[0]), float(p[1])) for p in (defending_players or [])
            if p is not None and len(p) >= 2 and math.isfinite(p[0]) and math.isfinite(p[1])
        ]
        if len(clean_defs) >= 2:
            # Sort defenders by X coordinate
            sorted_defs = sorted(clean_defs, key=lambda p: p[0])
            if attack_direction == "right":
                # Attacking toward X=105m: second deepest defender is 2nd highest X
                resolved_offside_x = sorted_defs[-2][0]
            else:
                # Attacking toward X=0m: second deepest defender is 2nd lowest X
                resolved_offside_x = sorted_defs[1][0]
        elif clean_defs:
            resolved_offside_x = clean_defs[0][0]
        else:
            resolved_offside_x = max(xa, xb)

    # Detect Offside Breaches
    offside_players: List[Dict[str, Any]] = []
    if attacking_players and resolved_offside_x is not None:
        for idx, p in enumerate(attacking_players):
            if p is None or len(p) < 2 or not math.isfinite(p[0]) or not math.isfinite(p[1]):
                continue
            px, py = float(p[0]), float(p[1])
            # Midfield line exemption: players cannot be offside in their own half
            in_attacking_half = px > 52.5 if attack_direction == "right" else px < 52.5
            if not in_attacking_half:
                continue

            is_beyond = (px > resolved_offside_x) if attack_direction == "right" else (px < resolved_offside_x)
            if is_beyond:
                margin = abs(px - resolved_offside_x)
                offside_players.append({
                    'player_idx': idx,
                    'point': (round(px, 2), round(py, 2)),
                    'margin_meters': round(margin, 2),
                    'margin_m': round(margin, 2)
                })

    resolved_ox = round(resolved_offside_x, 2) if resolved_offside_x is not None else 52.5
    d_val = round(euclidean_dist, 2)
    dx_val = round(abs(dx), 2)
    dy_val = round(abs(dy), 2)

    return {
        'point_a': (round(xa, 2), round(ya, 2)),
        'point_b': (round(xb, 2), round(yb, 2)),
        'label_a': label_a,
        'label_b': label_b,
        'distance': d_val,
        'distance_m': d_val,
        'dx': dx_val,
        'delta_x_m': dx_val,
        'dy': dy_val,
        'delta_y_m': dy_val,
        'dx_signed': round(dx, 2),
        'dy_signed': round(dy, 2),
        'angle_deg': round(angle_deg, 1),
        'offside_x': resolved_ox,
        'offside_x_m': resolved_ox,
        'offside_players': offside_players,
        'offside_breaches': offside_players,
        'is_offside': len(offside_players) > 0,
        'offside_count': len(offside_players)
    }
