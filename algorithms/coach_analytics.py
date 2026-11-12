"""
Coach Analytics Engine
Translates core geometric and graph-theoretic algorithms (Convex Hull, Voronoi, Dijkstra, Kinematics)
into high-level tactical football intelligence used by professional coaching staffs:
- Defensive Block Height & Width (Low Block, Mid Block, High Press)
- Territorial Dominance across Pitch Thirds (Final Third & Middle Third control)
- Line-Breaking Passing Penetration detection
- Team Pressing Intensity & Surge Index
"""

from typing import List, Tuple, Dict, Any, Optional
import math


def evaluate_coach_analytics(
    team_a_coords: List[Tuple[float, float]],
    team_b_coords: List[Tuple[float, float]],
    hull_res: Optional[Dict[str, Any]],
    voronoi_res: Optional[Dict[str, Any]],
    dijkstra_res: Optional[Dict[str, Any]],
    kinematics_a: Optional[Dict[str, Any]],
    ball_coord: Optional[Tuple[float, float]] = None
) -> Dict[str, Any]:
    """
    Computes professional tactical coaching metrics from the existing algorithmic outputs.
    """
    # 1. Defensive Block Structure (Derived from Defensive Convex Hull of Team B)
    line_height = 35.0
    block_info = {
        'block_type': 'Mid Block',
        'line_height_m': 35.0,
        'block_width_m': 38.0,
        'block_depth_m': 18.0,
        'compactness_verdict': 'Balanced Cohesion',
        'tactical_risk': 'Low'
    }
    
    if isinstance(hull_res, dict):
        hull_pts = hull_res.get('hull', hull_res.get('polygon', []))
    elif isinstance(hull_res, (list, tuple)):
        hull_pts = hull_res
    else:
        hull_pts = []
    if len(hull_pts) >= 3:
        xs = [p[0] for p in hull_pts]
        ys = [p[1] for p in hull_pts]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        block_width = max(8.0, max_y - min_y)
        block_depth = max(6.0, max_x - min_x)
        # Deepest outfield line for defending team (distance from goal x=105)
        # Team B defends towards x=105; deepest defender is max_x
        line_height = round(105.0 - max_x, 1)
        
        if line_height > 44.0:
            block_type = 'High Press Block'
            tactical_risk = 'High (Vulnerable Behind)'
        elif line_height >= 26.0:
            block_type = 'Mid Block'
            tactical_risk = 'Optimal (Compact)'
        else:
            block_type = 'Low Block (Deep)'
            tactical_risk = 'Low (Box Shield)'
            
        area = float(hull_res.get('area_sq_m', 0.0)) if isinstance(hull_res, dict) else 0.0
        if area < 550.0:
            compactness_verdict = 'Ultra Compact Shield'
        elif area <= 900.0:
            compactness_verdict = 'Organized Cohesion'
        else:
            compactness_verdict = 'Stretched Block (Pockets Open)'
            
        block_info = {
            'block_type': block_type,
            'line_height_m': line_height,
            'block_width_m': round(block_width, 1),
            'block_depth_m': round(block_depth, 1),
            'compactness_verdict': compactness_verdict,
            'tactical_risk': tactical_risk
        }

    # 2. Territorial Control across Pitch Thirds (Derived from Voronoi Partition)
    # Pitch x is [0, 105]. Final Third is x >= 70m, Middle Third is 35m <= x < 70m
    final_third_a = 0
    final_third_b = 0
    mid_third_a = 0
    mid_third_b = 0
    
    if voronoi_res:
        cells_a = voronoi_res.get('cells_a', [])
        cells_b = voronoi_res.get('cells_b', [])
        
        for c in cells_a:
            gx, gy = c.get('generator_site', (0, 0))
            if gx >= 70.0:
                final_third_a += 1
            elif gx >= 35.0:
                mid_third_a += 1
                
        for c in cells_b:
            gx, gy = c.get('generator_site', (0, 0))
            if gx >= 70.0:
                final_third_b += 1
            elif gx >= 35.0:
                mid_third_b += 1

    total_final = final_third_a + final_third_b
    final_third_pct_a = round((final_third_a / total_final) * 100.0) if total_final > 0 else 50
    
    if final_third_pct_a >= 60:
        territory_verdict = 'Attacking Pressure'
    elif final_third_pct_a <= 40:
        territory_verdict = 'Counter-Attacking Shape'
    else:
        territory_verdict = 'Contested Half-Spaces'

    # 3. Passing Penetration & Directive (Derived from Dijkstra Shortest Path)
    is_line_breaking = False
    pass_directive = 'Circulate Possession'
    d_path = dijkstra_res.get('path', []) if dijkstra_res else []
    
    if len(d_path) >= 2 and team_a_coords:
        start_u = d_path[0]
        end_v = d_path[-1]
        if start_u < len(team_a_coords) and end_v < len(team_a_coords):
            x_start = team_a_coords[start_u][0]
            x_end = team_a_coords[end_v][0]
            progression = x_end - x_start
            
            # Check if pass crosses defender line
            def_line_x = min([p[0] for p in hull_pts]) if hull_pts else 70.0
            if x_start < def_line_x and x_end >= def_line_x:
                is_line_breaking = True
                pass_directive = 'Line-Breaking Corridor (Penetrative)'
            elif progression > 12.0:
                pass_directive = 'Vertical Progression Available'
            else:
                pass_directive = 'Recirculate & Retain Structure'

    # 4. Pressing Intensity (Derived from Kinematics)
    accels = kinematics_a.get('accelerations', []) if kinematics_a else []
    active_pressers = sum(1 for a in accels if a >= 1.8)
    peak_accel = kinematics_a.get('peak_accel_val', 0.0) if kinematics_a else 0.0
    
    if active_pressers >= 3:
        pressing_state = 'Coordinated Press Surge'
    elif active_pressers >= 1:
        pressing_state = 'Active Ball Pressure'
    else:
        pressing_state = 'Rest Defense Shape'

    d_path = dijkstra_res.get('path', []) if dijkstra_res else []
    p_prog = float(dijkstra_res.get('total_progression', dijkstra_res.get('progression_m', 0.0)) or 0.0) if dijkstra_res else 0.0
    p_cnt = max(0, len(d_path) - 1) if d_path else 0
    passing_summary = {
        'progression_m': p_prog,
        'is_line_breaking': is_line_breaking,
        'directive': pass_directive,
        'passes_count': p_cnt,
        'path': d_path
    }

    return {
        'block': block_info,
        'final_third_pct_home': final_third_pct_a,
        'final_third_verdict': territory_verdict,
        'is_line_breaking': is_line_breaking,
        'pass_directive': pass_directive,
        'active_pressers': active_pressers,
        'pressing_state': pressing_state,
        'peak_accel': peak_accel,
        'line_height_m': line_height,
        'passing': passing_summary
    }
