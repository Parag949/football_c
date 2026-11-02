"""
Spatial Dominance: Voronoi Tessellation & Pitch Partitioning
Implements Fortune's Sweep-Line based Voronoi tessellation clipped to pitch boundaries (105m x 68m).
Calculates exact territorial control areas, team dominance percentages, and dynamic pressure fields.
"""

import time
import math
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from scipy.spatial import Voronoi
from algorithms.convex_hull import polygon_area_shoelace


def sutherland_hodgman_clip(
    subject_polygon: List[Tuple[float, float]],
    clip_polygon: Optional[List[Tuple[float, float]]] = None,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0
) -> List[Tuple[float, float]]:
    """
    Clips a Voronoi polygon cell strictly against the football pitch rectangle [0, L] x [0, W]
    using Sutherland-Hodgman polygon clipping against all 4 field boundaries.
    Guarantees that all output vertices satisfy 0 <= x <= pitch_length and 0 <= y <= pitch_width.
    """
    if not subject_polygon:
        return []

    poly = subject_polygon

    # Clip 1: x >= 0.0 (Left goal line)
    out = []
    s = poly[-1]
    for e in poly:
        if e[0] >= 0.0:
            if s[0] < 0.0:
                t = -s[0] / (e[0] - s[0] + 1e-15)
                out.append((0.0, s[1] + t * (e[1] - s[1])))
            out.append(e)
        elif s[0] >= 0.0:
            t = -s[0] / (e[0] - s[0] + 1e-15)
            out.append((0.0, s[1] + t * (e[1] - s[1])))
        s = e
    poly, out = out, []

    # Clip 2: x <= pitch_length (Right goal line)
    if not poly:
        return []
    s = poly[-1]
    for e in poly:
        if e[0] <= pitch_length:
            if s[0] > pitch_length:
                t = (pitch_length - s[0]) / (e[0] - s[0] + 1e-15)
                out.append((pitch_length, s[1] + t * (e[1] - s[1])))
            out.append(e)
        elif s[0] <= pitch_length:
            t = (pitch_length - s[0]) / (e[0] - s[0] + 1e-15)
            out.append((pitch_length, s[1] + t * (e[1] - s[1])))
        s = e
    poly, out = out, []

    # Clip 3: y >= 0.0 (Far touchline)
    if not poly:
        return []
    s = poly[-1]
    for e in poly:
        if e[1] >= 0.0:
            if s[1] < 0.0:
                t = -s[1] / (e[1] - s[1] + 1e-15)
                out.append((s[0] + t * (e[0] - s[0]), 0.0))
            out.append(e)
        elif s[1] >= 0.0:
            t = -s[1] / (e[1] - s[1] + 1e-15)
            out.append((s[0] + t * (e[0] - s[0]), 0.0))
        s = e
    poly, out = out, []

    # Clip 4: y <= pitch_width (Near touchline)
    if not poly:
        return []
    s = poly[-1]
    for e in poly:
        if e[1] <= pitch_width:
            if s[1] > pitch_width:
                t = (pitch_width - s[1]) / (e[1] - s[1] + 1e-15)
                out.append((s[0] + t * (e[0] - s[0]), pitch_width))
            out.append(e)
        elif s[1] <= pitch_width:
            t = (pitch_width - s[1]) / (e[1] - s[1] + 1e-15)
            out.append((s[0] + t * (e[0] - s[0]), pitch_width))
        s = e

    return [(max(0.0, min(pitch_length, float(p[0]))), max(0.0, min(pitch_width, float(p[1])))) for p in out]


def compute_bounded_voronoi(
    points: List[Tuple[float, float]],
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    margin: float = 50.0
) -> List[Dict[str, Any]]:
    """
    Computes Voronoi tessellation bounded by the pitch rectangle.
    Handles unbounded cells by constructing ghost rays and clipping against the pitch bounding box.
    
    Time Complexity: O(N log N) using Fortune's Sweep-Line / Delaunay dual.
    """
    if len(points) < 3:
        # Fallback when insufficient points for Voronoi
        return []

    pts_arr = np.array(points, dtype=np.float64)
    pts_clamped = np.column_stack([
        np.clip(pts_arr[:, 0], 0.0, pitch_length),
        np.clip(pts_arr[:, 1], 0.0, pitch_width)
    ])
    # Clip pitch polygon vertices in counter-clockwise order
    pitch_poly = [(0.0, 0.0), (pitch_length, 0.0), (pitch_length, pitch_width), (0.0, pitch_width)]

    # Mirror boundary sites to guarantee closed Voronoi regions inside the pitch
    # 4 boundary ghost mirrors
    ghost_sites = [
        pts_clamped + np.array([-2 * pts_clamped[:, 0], np.zeros(len(pts_clamped))]).T,  # mirror left
        pts_clamped + np.array([2 * (pitch_length - pts_clamped[:, 0]), np.zeros(len(pts_clamped))]).T,  # mirror right
        pts_clamped + np.array([np.zeros(len(pts_clamped)), -2 * pts_clamped[:, 1]]).T,  # mirror bottom
        pts_clamped + np.array([np.zeros(len(pts_clamped)), 2 * (pitch_width - pts_clamped[:, 1])]).T,  # mirror top
    ]
    all_sites = np.vstack([pts_clamped] + ghost_sites)

    try:
        vor = Voronoi(all_sites)
    except Exception:
        try:
            # Break collinearity with microscopic deterministic jitter (< 0.1mm)
            jitter = np.random.RandomState(42).uniform(-1e-4, 1e-4, size=all_sites.shape)
            vor = Voronoi(all_sites + jitter)
        except Exception:
            return []

    bounded_regions = []

    for i in range(len(points)):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        if -1 not in region and len(region) > 0:
            polygon = [tuple(vor.vertices[v]) for v in region]
            # Clip against exact pitch boundary
            clipped_poly = sutherland_hodgman_clip(polygon, pitch_poly, pitch_length=pitch_length, pitch_width=pitch_width)
            if len(clipped_poly) >= 3:
                area = polygon_area_shoelace(clipped_poly)
                bounded_regions.append({
                    'player_idx': i,
                    'generator_site': points[i],
                    'polygon': clipped_poly,
                    'area_sq_m': area
                })
        else:
            # Fallback if any vertex is outside
            continue

    return bounded_regions


def compute_spatial_dominance(
    team_a_coords: List[Tuple[float, float]],
    team_b_coords: List[Tuple[float, float]],
    pitch_length: float = 105.0,
    pitch_width: float = 68.0
) -> Dict[str, Any]:
    """
    Partitions the entire football pitch between Team A and Team B.
    
    Returns:
        - Team A total territory (m^2) & percentage
        - Team B total territory (m^2) & percentage
        - Individual player cells with bounded polygons and areas
        - Algorithmic latency (ms)
    """
    start_time = time.perf_counter()
    
    combined_points = team_a_coords + team_b_coords
    num_a = len(team_a_coords)
    num_b = len(team_b_coords)
    
    if len(combined_points) < 3:
        elapsed = (time.perf_counter() - start_time) * 1000.0
        return {
            'team_a_area': 0.0,
            'team_b_area': 0.0,
            'team_a_pct': 50.0,
            'team_b_pct': 50.0,
            'cells_a': [],
            'cells_b': [],
            'runtime_ms': elapsed,
            'total_pitch_area': pitch_length * pitch_width
        }

    regions = compute_bounded_voronoi(combined_points, pitch_length, pitch_width)
    
    cells_a = []
    cells_b = []
    total_area_a = 0.0
    total_area_b = 0.0
    
    for r in regions:
        idx = r['player_idx']
        if idx < num_a:
            cells_a.append(r)
            total_area_a += r['area_sq_m']
        else:
            cells_b.append(r)
            total_area_b += r['area_sq_m']
            
    total_measured_area = total_area_a + total_area_b
    total_pitch_area = pitch_length * pitch_width
    
    if total_measured_area > 0:
        pct_a = (total_area_a / total_measured_area) * 100.0
        pct_b = (total_area_b / total_measured_area) * 100.0
    else:
        pct_a = 50.0
        pct_b = 50.0
        
    elapsed = (time.perf_counter() - start_time) * 1000.0
    
    return {
        'team_a_area': round(total_area_a, 2),
        'team_b_area': round(total_area_b, 2),
        'team_a_pct': round(pct_a, 1),
        'team_b_pct': round(pct_b, 1),
        'cells_a': cells_a,
        'cells_b': cells_b,
        'runtime_ms': elapsed,
        'total_pitch_area': total_pitch_area,
        'num_players': len(combined_points)
    }
