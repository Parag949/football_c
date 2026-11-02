"""
Defensive Cohesion: Convex Hull Algorithms
Implements Graham Scan (O(N log N)) and Jarvis March / Gift-Wrapping (O(N * H))
for real-time tactical football defensive line analysis.
"""

import time
import math
from typing import List, Tuple, Dict, Any
import numpy as np


def cross_product(o: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """
    Computes the 2D cross product of vector OA and OB.
    Returns:
        > 0 if O, A, B form a counter-clockwise turn
        < 0 if O, A, B form a clockwise turn
        = 0 if O, A, B are collinear
    """
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def euclidean_dist_sq(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Squared Euclidean distance between two 2D points."""
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2


def graham_scan(points: List[Tuple[float, float]]) -> Dict[str, Any]:
    """
    Computes Convex Hull using Graham Scan Algorithm.
    Time Complexity: O(N log N)
    Space Complexity: O(N)
    
    Returns:
        dict containing 'hull': List of points, 'runtime_ms': float, 'operations': int
    """
    start_time = time.perf_counter()
    ops = 0
    
    # Remove duplicate points
    pts = list(dict.fromkeys(points))
    n = len(pts)
    
    if n <= 2:
        elapsed = (time.perf_counter() - start_time) * 1000.0
        return {
            'hull': pts,
            'runtime_ms': elapsed,
            'operations': ops,
            'algorithm': 'Graham Scan'
        }
    
    # 1. Find pivot point with lowest y-coordinate (and leftmost x on tie)
    pivot = min(pts, key=lambda p: (p[1], p[0]))
    ops += n
    
    # 2. Sort points by polar angle with pivot
    def polar_comparator(p):
        angle = math.atan2(p[1] - pivot[1], p[0] - pivot[0])
        dist = euclidean_dist_sq(pivot, p)
        return (angle, dist)
    
    sorted_pts = sorted([p for p in pts if p != pivot], key=polar_comparator)
    ops += int(n * math.log2(n)) if n > 1 else 1
    
    # Filter collinear points, keeping only the furthest one (via 2D cross product)
    unique_polar_pts = []
    for i, p in enumerate(sorted_pts):
        while unique_polar_pts and abs(cross_product(pivot, unique_polar_pts[-1], p)) < 1e-9:
            unique_polar_pts.pop()
            ops += 1
        unique_polar_pts.append(p)
        ops += 1
        
    if len(unique_polar_pts) < 2:
        hull = [pivot] + unique_polar_pts
        elapsed = (time.perf_counter() - start_time) * 1000.0
        return {
            'hull': hull,
            'runtime_ms': elapsed,
            'operations': ops,
            'algorithm': 'Graham Scan'
        }
        
    # 3. Build Hull using Stack
    stack = [pivot, unique_polar_pts[0], unique_polar_pts[1]]
    ops += 2
    
    for p in unique_polar_pts[2:]:
        while len(stack) > 1 and cross_product(stack[-2], stack[-1], p) <= 0:
            stack.pop()
            ops += 1
        stack.append(p)
        ops += 1
        
    elapsed = (time.perf_counter() - start_time) * 1000.0
    return {
        'hull': stack,
        'runtime_ms': elapsed,
        'operations': ops,
        'algorithm': 'Graham Scan'
    }


def jarvis_march(points: List[Tuple[float, float]]) -> Dict[str, Any]:
    """
    Computes Convex Hull using Jarvis March (Gift Wrapping) Algorithm.
    Time Complexity: O(N * H) where H is number of vertices on the hull
    Space Complexity: O(H)
    
    Returns:
        dict containing 'hull': List of points, 'runtime_ms': float, 'operations': int
    """
    start_time = time.perf_counter()
    ops = 0
    
    pts = list(dict.fromkeys(points))
    n = len(pts)
    
    if n <= 2:
        elapsed = (time.perf_counter() - start_time) * 1000.0
        return {
            'hull': pts,
            'runtime_ms': elapsed,
            'operations': ops,
            'algorithm': 'Jarvis March'
        }
    
    # 1. Find the leftmost point
    start_point = min(pts, key=lambda p: (p[0], p[1]))
    ops += n
    
    hull = []
    current_point = start_point
    
    while True:
        hull.append(current_point)
        next_point = pts[0]
        
        for candidate in pts[1:]:
            ops += 1
            if candidate == current_point:
                continue
                
            cp = cross_product(current_point, next_point, candidate)
            # If candidate is more counter-clockwise than next_point, or collinear and farther
            if next_point == current_point or cp > 0 or (
                math.isclose(cp, 0.0, abs_tol=1e-7) and
                euclidean_dist_sq(current_point, candidate) > euclidean_dist_sq(current_point, next_point)
            ):
                next_point = candidate
                
        current_point = next_point
        if current_point == start_point or len(hull) >= n:
            break
            
    elapsed = (time.perf_counter() - start_time) * 1000.0
    return {
        'hull': hull,
        'runtime_ms': elapsed,
        'operations': ops,
        'algorithm': 'Jarvis March'
    }


def polygon_area_shoelace(polygon: List[Tuple[float, float]]) -> float:
    """
    Calculates the polygon area using the Shoelace formula (Gauss's Area Formula).
    Area = 0.5 * |sum(x_i * y_{i+1} - x_{i+1} * y_i)|
    """
    n = len(polygon)
    if n < 3:
        return 0.0
        
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += polygon[i][0] * polygon[j][1]
        area -= polygon[j][0] * polygon[i][1]
        
    return abs(area) / 2.0


def polygon_centroid(polygon: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Calculates the center of mass (centroid) of a convex polygon."""
    n = len(polygon)
    if n == 0:
        return (0.0, 0.0)
    if n <= 2:
        return (float(np.mean([p[0] for p in polygon])), float(np.mean([p[1] for p in polygon])))
        
    area = 0.0
    cx = 0.0
    cy = 0.0
    
    for i in range(n):
        j = (i + 1) % n
        factor = (polygon[i][0] * polygon[j][1] - polygon[j][0] * polygon[i][1])
        area += factor
        cx += (polygon[i][0] + polygon[j][0]) * factor
        cy += (polygon[i][1] + polygon[j][1]) * factor
        
    area = area / 2.0
    if abs(area) < 1e-6:
        return (float(np.mean([p[0] for p in polygon])), float(np.mean([p[1] for p in polygon])))
        
    cx = cx / (6.0 * area)
    cy = cy / (6.0 * area)
    return (cx, cy)


def compute_defensive_cohesion(player_coords: List[Tuple[float, float]], algorithm: str = 'graham') -> Dict[str, Any]:
    """
    Evaluates defensive cohesion metrics for a set of tactical coordinates:
    - Convex Hull boundary
    - Defensive footprint area (m^2)
    - Centroid / line anchor
    - Compactness index = Area / perimeter^2
    """
    if algorithm.lower() == 'jarvis':
        result = jarvis_march(player_coords)
    else:
        result = graham_scan(player_coords)
        
    hull = result['hull']
    area = polygon_area_shoelace(hull)
    centroid = polygon_centroid(hull)
    
    # Calculate perimeter
    perimeter = 0.0
    if len(hull) >= 2:
        for i in range(len(hull)):
            j = (i + 1) % len(hull)
            perimeter += math.sqrt(euclidean_dist_sq(hull[i], hull[j]))
            
    # Compactness (isoperimetric quotient normalized)
    compactness = (4 * math.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0
    
    return {
        'algorithm': result['algorithm'],
        'hull': hull,
        'area_sq_m': area,
        'perimeter_m': perimeter,
        'centroid': centroid,
        'compactness': compactness,
        'runtime_ms': result['runtime_ms'],
        'operations': result['operations'],
        'num_players': len(player_coords),
        'hull_vertices': len(hull)
    }
