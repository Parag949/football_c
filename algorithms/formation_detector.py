"""
Automated Tactical Formation Classifier & Line Height Analyzer
==============================================================
Analyzes 2D pitch metric coordinates to automatically detect team formations
(e.g., 4-3-3, 4-2-3-1, 4-4-2, 3-5-2, 5-3-2, 3-4-3) without manual user tagging.

Computes:
- Line breakdown: Number of defenders, midfielders, and forwards
- Defensive line height (meters from own goal line)
- Tactical compactness (distance between defensive line and highest forward)
- Team horizontal stretching (lateral pitch width in meters)
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np

PITCH_LENGTH: float = 105.0
PITCH_WIDTH: float = 68.0

# Supported standard football formation templates
KNOWN_FORMATIONS = {
    (4, 3, 3): "4-3-3",
    (4, 2, 3, 1): "4-2-3-1",
    (4, 4, 2): "4-4-2",
    (3, 5, 2): "3-5-2",
    (3, 4, 3): "3-4-3",
    (5, 3, 2): "5-3-2",
    (5, 4, 1): "5-4-1",
    (4, 1, 4, 1): "4-1-4-1",
    (4, 5, 1): "4-5-1",
}


def detect_team_formation(
    players: List[Tuple[float, float]],
    attack_direction: str = "right",
    pitch_length: float = PITCH_LENGTH,
    pitch_width: float = PITCH_WIDTH,
) -> Dict[str, Any]:
    """
    Classifies the tactical formation and spatial geometry of a team.
    
    Parameters:
    -----------
    players : List of (X, Y) metric coordinates in meters
    attack_direction : 'right' (defends X=0, attacks X=105) or 'left' (defends X=105, attacks X=0)
    
    Returns:
    --------
    Dict containing:
      - 'formation': String name of detected formation (e.g. "4-3-3", "4-4-2")
      - 'line_counts': Tuple of player counts per line from back to front, e.g. (4, 3, 3)
      - 'defensive_line_m': Distance in meters of defensive line from own goal line
      - 'compactness_m': Length in meters from deepest defender to highest attacker
      - 'team_width_m': Lateral pitch width in meters spanned by the team
      - 'centroid_m': Tuple (cx, cy) center of team mass in meters
      - 'lines': List of dicts describing each line's average X, player count, and indices
    """
    pts = np.asarray(players, dtype=np.float64).reshape(-1, 2)
    n = len(pts)

    if n < 4:
        return {
            'formation': "Unstructured (< 4 players)",
            'line_counts': (n,),
            'defensive_line_m': 0.0,
            'compactness_m': 0.0,
            'team_width_m': 0.0,
            'centroid_m': (float(np.mean(pts[:, 0])) if n > 0 else 52.5, float(np.mean(pts[:, 1])) if n > 0 else 34.0),
            'lines': []
        }

    # Normalize X coordinates so 0.0 is always own goal line, 105.0 is opponent goal line
    norm_x = pts[:, 0] if attack_direction == "right" else (pitch_length - pts[:, 0])
    norm_y = pts[:, 1]

    # Exclude Goalkeeper if clearly stationed near own goal line (< 14m from own goal)
    gk_idx = -1
    min_x_idx = int(np.argmin(norm_x))
    if norm_x[min_x_idx] < 14.0 and n >= 8:
        gk_idx = min_x_idx

    outfield_indices = [i for i in range(n) if i != gk_idx]
    outfield_x = norm_x[outfield_indices]
    outfield_y = norm_y[outfield_indices]
    n_outfield = len(outfield_indices)

    # Sort outfield players from own goal line forward
    sorted_order = np.argsort(outfield_x)
    sorted_indices = [outfield_indices[i] for i in sorted_order]
    sorted_x = outfield_x[sorted_order]

    # Cluster outfield players into longitudinal bands using 1D gaps
    # Typical spacing between lines (defenders vs midfielders vs attackers) is >= 8.0m - 14.0m
    gaps = np.diff(sorted_x)
    # Target 3 or 4 tactical lines (Defense, Midfield, Attack, or Defense, CDM, CAM, Attack)
    split_candidates = np.argsort(gaps)[::-1]
    
    # Choose 2 or 3 largest gaps to form 3 or 4 lines
    num_splits = 2 if n_outfield < 10 else (3 if (len(gaps) >= 3 and gaps[split_candidates[2]] > 7.5) else 2)
    chosen_splits = sorted(split_candidates[:num_splits] + 1)

    # Slice into bands
    band_indices = np.split(np.arange(n_outfield), chosen_splits)
    line_counts = tuple(len(b) for b in band_indices if len(b) > 0)

    # Best-match against standard formations
    matched_formation = KNOWN_FORMATIONS.get(line_counts)
    if matched_formation is None:
        # Match closest canonical formation by L1 distance on lines
        best_name = "-".join(str(c) for c in line_counts)
        min_diff = 999
        for f_tuple, f_name in KNOWN_FORMATIONS.items():
            if len(f_tuple) == len(line_counts):
                diff = sum(abs(a - b) for a, b in zip(f_tuple, line_counts))
                if diff < min_diff:
                    min_diff = diff
                    best_name = f_name
        matched_formation = best_name if min_diff <= 2 else "-".join(str(c) for c in line_counts)

    # Defensive Line Height (mean X of the deepest outfield line)
    deepest_line_x = float(np.mean(sorted_x[band_indices[0]]))
    highest_line_x = float(np.mean(sorted_x[band_indices[-1]]))
    compactness = round(float(highest_line_x - deepest_line_x), 1)

    # Team lateral width
    team_width = round(float(np.max(outfield_y) - np.min(outfield_y)), 1)
    centroid_x = round(float(np.mean(pts[:, 0])), 1)
    centroid_y = round(float(np.mean(pts[:, 1])), 1)

    lines_info = []
    for line_idx, band in enumerate(band_indices):
        band_player_ids = [sorted_indices[i] for i in band]
        avg_line_x = float(np.mean(sorted_x[band]))
        lines_info.append({
            'line_number': line_idx + 1,
            'player_count': len(band),
            'line_name': "Defense" if line_idx == 0 else ("Attack" if line_idx == len(band_indices) - 1 else "Midfield"),
            'height_from_goal_m': round(avg_line_x, 1),
            'player_indices': band_player_ids
        })

    return {
        'formation': matched_formation,
        'line_counts': line_counts,
        'defensive_line_m': round(deepest_line_x, 1),
        'compactness_m': compactness,
        'team_width_m': team_width,
        'centroid_m': (centroid_x, centroid_y),
        'lines': lines_info
    }
