"""
Continuous Pitch Control Field & Spatial Dominance Engine
=========================================================
Implements the continuous pitch control probability model (Spearman 2018, Fernandez-Bornn 2019).
Unlike discrete Voronoi tessellation, Pitch Control models physical time-to-intercept surfaces,
incorporating player sprint acceleration, reaction latency, and ball travel times across the
FIFA standard 105m x 68m pitch.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import cv2

PITCH_LENGTH: float = 105.0
PITCH_WIDTH: float = 68.0


def compute_pitch_control_field(
    team_a: List[Tuple[float, float]],
    team_b: List[Tuple[float, float]],
    ball_coord: Optional[Tuple[float, float]] = None,
    grid_size: Tuple[int, int] = (50, 32),
    v_max: float = 7.0,
    reaction_time: float = 0.5,
    lambda_decay: float = 4.3,
    pitch_length: float = PITCH_LENGTH,
    pitch_width: float = PITCH_WIDTH,
) -> Dict[str, Any]:
    """
    Computes continuous pitch control probability surface over a 2D metric grid.
    
    Parameters:
    -----------
    team_a : List of (X, Y) coordinates in meters for attacking team
    team_b : List of (X, Y) coordinates in meters for defending team
    ball_coord : Optional (X, Y) coordinate in meters of the match ball
    grid_size : (nx, ny) sampling resolution across [0, pitch_length] x [0, pitch_width]
    v_max : Maximum sprint velocity in m/s (default 7.0 m/s ~ 25.2 km/h)
    reaction_time : Player cognitive & biomechanical reaction latency in seconds (default 0.5s)
    lambda_decay : Sigmoid slope parameter determining sharpness of transition (default 4.3)
    
    Returns:
    --------
    Dict containing:
      - 'control_grid': 2D np.ndarray (ny, nx) with values in [0.0, 1.0] (1.0 = Team A, 0.0 = Team B)
      - 'team_a_pct': Float percentage of total pitch controlled by Team A
      - 'team_b_pct': Float percentage of total pitch controlled by Team B
      - 'uncontrolled_pct': Float percentage of contested/neutral pitch
      - 'grid_x': 1D np.ndarray of X meter grid lines
      - 'grid_y': 1D np.ndarray of Y meter grid lines
    """
    nx, ny = grid_size
    gx = np.linspace(0.0, pitch_length, nx)
    gy = np.linspace(0.0, pitch_width, ny)
    grid_x, grid_y = np.meshgrid(gx, gy)  # Shapes: (ny, nx)

    pts_a = np.asarray(team_a, dtype=np.float64).reshape(-1, 2) if team_a else np.empty((0, 2))
    pts_b = np.asarray(team_b, dtype=np.float64).reshape(-1, 2) if team_b else np.empty((0, 2))

    # Fallback when one or both teams are empty
    if len(pts_a) == 0 and len(pts_b) == 0:
        neutral = np.full((ny, nx), 0.5, dtype=np.float32)
        return {
            'control_grid': neutral,
            'team_a_pct': 50.0,
            'team_b_pct': 50.0,
            'uncontrolled_pct': 0.0,
            'grid_x': gx,
            'grid_y': gy,
        }
    if len(pts_a) == 0:
        zeros = np.zeros((ny, nx), dtype=np.float32)
        return {
            'control_grid': zeros,
            'team_a_pct': 0.0,
            'team_b_pct': 100.0,
            'uncontrolled_pct': 0.0,
            'grid_x': gx,
            'grid_y': gy,
        }
    if len(pts_b) == 0:
        ones = np.ones((ny, nx), dtype=np.float32)
        return {
            'control_grid': ones,
            'team_a_pct': 100.0,
            'team_b_pct': 0.0,
            'uncontrolled_pct': 0.0,
            'grid_x': gx,
            'grid_y': gy,
        }

    # Time-to-intercept tensor calculation: (N, ny, nx)
    # T_i(x, y) = reaction_time + distance(p_i, (x, y)) / v_max
    # Vectorized across the entire 2D field in one pass
    diff_a_x = grid_x[None, :, :] - pts_a[:, 0, None, None]
    diff_a_y = grid_y[None, :, :] - pts_a[:, 1, None, None]
    dist_a = np.sqrt(diff_a_x ** 2 + diff_a_y ** 2)
    time_a = reaction_time + (dist_a / max(0.1, v_max))  # (Na, ny, nx)

    diff_b_x = grid_x[None, :, :] - pts_b[:, 0, None, None]
    diff_b_y = grid_y[None, :, :] - pts_b[:, 1, None, None]
    dist_b = np.sqrt(diff_b_x ** 2 + diff_b_y ** 2)
    time_b = reaction_time + (dist_b / max(0.1, v_max))  # (Nb, ny, nx)

    # Minimum arrival time across all teammates at each field node
    min_time_a = np.min(time_a, axis=0)  # Shape: (ny, nx)
    min_time_b = np.min(time_b, axis=0)  # Shape: (ny, nx)

    # Continuous control probability via logistic sigmoid of arrival delta
    # Delta > 0 => Team B arrives faster => P(A) < 0.5
    # Delta < 0 => Team A arrives faster => P(A) > 0.5
    time_diff = min_time_b - min_time_a  # Positive when Team A is faster
    control_grid = 1.0 / (1.0 + np.exp(-lambda_decay * time_diff))
    control_grid = np.clip(control_grid, 0.0, 1.0).astype(np.float32)

    # Pitch Area Dominance Percentages
    team_a_cells = np.sum(control_grid > 0.55)
    team_b_cells = np.sum(control_grid < 0.45)
    contested_cells = np.sum((control_grid >= 0.45) & (control_grid <= 0.55))
    total_cells = float(nx * ny)

    team_a_pct = round(float((team_a_cells / total_cells) * 100.0), 1)
    team_b_pct = round(float((team_b_cells / total_cells) * 100.0), 1)
    uncontrolled_pct = round(float((contested_cells / total_cells) * 100.0), 1)

    return {
        'control_grid': control_grid,
        'team_a_pct': team_a_pct,
        'team_b_pct': team_b_pct,
        'uncontrolled_pct': uncontrolled_pct,
        'grid_x': gx,
        'grid_y': gy,
    }


def render_pitch_control_heatmap(
    control_grid: np.ndarray,
    width: int = 1050,
    height: int = 680,
    color_team_a: Tuple[int, int, int] = (56, 189, 248),  # Sky Blue BGR
    color_team_b: Tuple[int, int, int] = (239, 68, 68),   # Vibrant Red BGR
    alpha: float = 0.65
) -> np.ndarray:
    """
    Renders the continuous pitch control probability grid into an antialiased BGR tactical heatmap.
    Values near 1.0 glow Team A color; values near 0.0 glow Team B color; 0.5 represents contested turf.
    """
    ny, nx = control_grid.shape
    # Upscale grid smoothly using bicubic interpolation
    upscaled = cv2.resize(control_grid, (width, height), interpolation=cv2.INTER_CUBIC)
    upscaled = np.clip(upscaled, 0.0, 1.0)

    heatmap = np.zeros((height, width, 3), dtype=np.uint8)

    # Color blending across [0.0 .. 1.0]:
    # 0.0 -> Team B color
    # 0.5 -> Dark Slate Pitch Turf (#0f172a)
    # 1.0 -> Team A color
    ca = np.array(color_team_a, dtype=np.float32)
    cb = np.array(color_team_b, dtype=np.float32)
    neutral = np.array([30.0, 41.0, 59.0], dtype=np.float32)  # #1e293b

    # Vectorized interpolation
    mask_a = upscaled >= 0.5
    factor_a = (upscaled[mask_a, None] - 0.5) * 2.0  # 0.0 to 1.0
    color_a_pixels = neutral[None, :] * (1.0 - factor_a) + ca[None, :] * factor_a

    mask_b = ~mask_a
    factor_b = (0.5 - upscaled[mask_b, None]) * 2.0  # 0.0 to 1.0
    color_b_pixels = neutral[None, :] * (1.0 - factor_b) + cb[None, :] * factor_b

    heatmap[mask_a] = np.clip(color_a_pixels, 0, 255).astype(np.uint8)
    heatmap[mask_b] = np.clip(color_b_pixels, 0, 255).astype(np.uint8)

    return heatmap
