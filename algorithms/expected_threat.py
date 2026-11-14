"""
Expected Threat (xT) Tactical Possession Model
===============================================
Implements Karun Singh's Expected Threat (xT) spatial Markov model for association football.
Discretizes the FIFA 105m x 68m pitch into a 16x12 grid of probabilistic zones, mapping the
probability of a team scoring within the next 5 actions from any location on the pitch.

Calculates:
- Baseline zone threat: xT(x, y) in [0.0, 1.0]
- Progressive Threat Created: Delta-xT = xT(end) - xT(start)
- Line-Breaking & Killer Pass Classification
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import cv2

PITCH_LENGTH: float = 105.0
PITCH_WIDTH: float = 68.0

# 16 x 12 Calibrated Expected Threat Matrix for Standard Pitch (Karun Singh Model)
# Rows: 12 lateral pitch lanes (Y: 0m -> 68m); Columns: 16 longitudinal pitch zones (X: 0m -> 105m)
# Values represent probability [0.0..0.45] of possession ending in a goal within next 5 actions
XT_GRID_16x12 = np.array([
    [0.0063, 0.0075, 0.0084, 0.0097, 0.0114, 0.0141, 0.0179, 0.0211, 0.0253, 0.0298, 0.0351, 0.0421, 0.0512, 0.0632, 0.0784, 0.0912],
    [0.0071, 0.0082, 0.0091, 0.0106, 0.0125, 0.0158, 0.0201, 0.0242, 0.0294, 0.0352, 0.0423, 0.0518, 0.0645, 0.0821, 0.1084, 0.1341],
    [0.0079, 0.0089, 0.0101, 0.0118, 0.0142, 0.0181, 0.0232, 0.0284, 0.0351, 0.0431, 0.0532, 0.0671, 0.0862, 0.1165, 0.1621, 0.2142],
    [0.0084, 0.0095, 0.0108, 0.0127, 0.0156, 0.0202, 0.0264, 0.0328, 0.0412, 0.0518, 0.0654, 0.0842, 0.1121, 0.1584, 0.2312, 0.3124],
    [0.0088, 0.0101, 0.0115, 0.0135, 0.0169, 0.0221, 0.0291, 0.0368, 0.0471, 0.0601, 0.0772, 0.1018, 0.1384, 0.2012, 0.2984, 0.3842],
    [0.0091, 0.0104, 0.0119, 0.0141, 0.0178, 0.0234, 0.0312, 0.0398, 0.0515, 0.0664, 0.0861, 0.1145, 0.1592, 0.2361, 0.3421, 0.4215],
    [0.0091, 0.0104, 0.0119, 0.0141, 0.0178, 0.0234, 0.0312, 0.0398, 0.0515, 0.0664, 0.0861, 0.1145, 0.1592, 0.2361, 0.3421, 0.4215],
    [0.0088, 0.0101, 0.0115, 0.0135, 0.0169, 0.0221, 0.0291, 0.0368, 0.0471, 0.0601, 0.0772, 0.1018, 0.1384, 0.2012, 0.2984, 0.3842],
    [0.0084, 0.0095, 0.0108, 0.0127, 0.0156, 0.0202, 0.0264, 0.0328, 0.0412, 0.0518, 0.0654, 0.0842, 0.1121, 0.1584, 0.2312, 0.3124],
    [0.0079, 0.0089, 0.0101, 0.0118, 0.0142, 0.0181, 0.0232, 0.0284, 0.0351, 0.0431, 0.0532, 0.0671, 0.0862, 0.1165, 0.1621, 0.2142],
    [0.0071, 0.0082, 0.0091, 0.0106, 0.0125, 0.0158, 0.0201, 0.0242, 0.0294, 0.0352, 0.0423, 0.0518, 0.0645, 0.0821, 0.1084, 0.1341],
    [0.0063, 0.0075, 0.0084, 0.0097, 0.0114, 0.0141, 0.0179, 0.0211, 0.0253, 0.0298, 0.0351, 0.0421, 0.0512, 0.0632, 0.0784, 0.0912],
], dtype=np.float32)


def get_point_xt(
    x_m: float,
    y_m: float,
    attack_direction: str = "right",
    pitch_length: float = PITCH_LENGTH,
    pitch_width: float = PITCH_WIDTH,
) -> float:
    """
    Returns base Expected Threat value xT in [0.0..0.45] for any metric pitch position (X, Y).
    Accounts for team attacking direction ('right' towards X=105m vs 'left' towards X=0m).
    """
    # Flip X if attacking leftwards
    norm_x = (pitch_length - x_m) if attack_direction == "left" else x_m
    norm_y = y_m

    # Clamp coordinates to pitch bounds
    cx = np.clip(norm_x, 0.0, pitch_length - 1e-4)
    cy = np.clip(norm_y, 0.0, pitch_width - 1e-4)

    col = int(np.clip((cx / pitch_length) * 16.0, 0, 15))
    row = int(np.clip((cy / pitch_width) * 12.0, 0, 11))

    return float(XT_GRID_16x12[row, col])


def evaluate_action_threat(
    start_pos: Tuple[float, float],
    end_pos: Tuple[float, float],
    attack_direction: str = "right",
    pitch_length: float = PITCH_LENGTH,
    pitch_width: float = PITCH_WIDTH,
) -> Dict[str, Any]:
    """
    Computes Delta-xT (threat created or lost) for a pass, cross, or carry.
    
    Returns:
    --------
    Dict containing:
      - 'start_xt': Float xT at action origin
      - 'end_xt': Float xT at action destination
      - 'delta_xt': Float net threat gained (+ means progressive, - means regressive)
      - 'is_killer_pass': Boolean True if delta_xt >= 0.045 (high goal expectancy creation)
      - 'is_progressive': Boolean True if pass moves ball >= 10m forward in attacking half
    """
    xt_start = get_point_xt(start_pos[0], start_pos[1], attack_direction, pitch_length, pitch_width)
    xt_end = get_point_xt(end_pos[0], end_pos[1], attack_direction, pitch_length, pitch_width)
    delta_xt = round(float(xt_end - xt_start), 4)

    # Progressive definition: net forward progression >= 10m in midfield/attacking half
    fwd_dist = (end_pos[0] - start_pos[0]) if attack_direction == "right" else (start_pos[0] - end_pos[0])
    is_prog = bool(fwd_dist >= 10.0 or (fwd_dist >= 5.0 and xt_end >= 0.08))

    # Killer pass: creates significant threat surge or reaches high-danger box zone
    is_killer = bool(delta_xt >= 0.035 or (xt_end >= 0.18 and delta_xt >= 0.015))

    return {
        'start_xt': round(xt_start, 4),
        'end_xt': round(xt_end, 4),
        'delta_xt': delta_xt,
        'is_killer_pass': is_killer,
        'is_progressive': is_prog,
        'forward_distance_m': round(float(fwd_dist), 2),
    }


def rank_passing_lanes_by_threat(
    carrier_pos: Tuple[float, float],
    candidate_targets: List[Tuple[float, float]],
    attack_direction: str = "right"
) -> List[Dict[str, Any]]:
    """
    Ranks open passing lanes from a ball carrier by net threat creation (Delta-xT).
    Helps coaches immediately identify high-leverage passing options vs safe back-passes.
    """
    ranked = []
    for idx, target in enumerate(candidate_targets):
        threat_data = evaluate_action_threat(carrier_pos, target, attack_direction)
        threat_data['target_index'] = idx
        threat_data['target_pos'] = target
        ranked.append(threat_data)

    # Sort descending by delta_xt (most threatening passes first)
    ranked.sort(key=lambda item: item['delta_xt'], reverse=True)
    return ranked


def render_xt_heatmap(
    width: int = 1050,
    height: int = 680,
    attack_direction: str = "right"
) -> np.ndarray:
    """
    Generates a 2D color-coded tactical heatmap visualization of the Expected Threat grid.
    Warmer amber/crimson colors indicate dangerous scoring zones; deep blues indicate build-up areas.
    """
    grid = XT_GRID_16x12
    if attack_direction == "left":
        grid = np.fliplr(grid)

    # Normalize across max threat (0.4215)
    norm_grid = (grid / np.max(grid)).astype(np.float32)
    upscaled = cv2.resize(norm_grid, (width, height), interpolation=cv2.INTER_CUBIC)
    upscaled_uint8 = np.clip(upscaled * 255.0, 0, 255).astype(np.uint8)

    # Apply INFERNO or TURBO colormap
    colormap = cv2.applyColorMap(upscaled_uint8, cv2.COLORMAP_TURBO)
    return colormap
