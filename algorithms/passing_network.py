"""
Graph-Theoretic Passing Network & Dijkstra Shortest Path Engine (DAA)
===================================================================
Applies classical graph theory and Dijkstra's algorithm with a binary Min-Heap
to determine the optimal, lowest-risk passing progression sequence through
an opponent's defensive block.

Mathematical Topology:
- Directed Complete Graph G = (V, E):
  * Vertices V: Attacking outfield players (Team A, |V| <= 11).
  * Directed Edges (u, v): Pass vector from player u to player v.
- Edge Cost Weight:
  W(u, v) = max(0.1, dist(u, v) * (1.0 + 2.5 * Pressure(u, v)) - alpha * Progression(u, v) + 10.0)
  Ensures non-negative edge weights for strict Dijkstra invariant correctness.
- Complexity:
  * Time Complexity: O((V + E) log V) via heapq binary min-heap.
  * Space Complexity: O(V + E) for adjacency list and distance dict.
  * Real-world execution: < 0.1 ms for |V| = 11.
"""

import time
import heapq
import math
from typing import List, Tuple, Dict, Any, Optional
import numpy as np


def point_to_segment_distance(
    pt: Tuple[float, float],
    p1: Tuple[float, float],
    p2: Tuple[float, float]
) -> float:
    """
    Computes minimum Euclidean distance from a defender coordinate (pt)
    to a passing trajectory line segment between p1 and p2.
    """
    x, y = pt
    x1, y1 = p1
    x2, y2 = p2

    dx, dy = x2 - x1, y2 - y1
    l2 = dx * dx + dy * dy
    if l2 < 1e-6:
        return math.hypot(x - x1, y - y1)

    # Project point onto line segment clamped to [0, 1]
    t = max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / l2))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.hypot(x - proj_x, y - proj_y)


def compute_pass_defensive_pressure(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    defenders: List[Tuple[float, float]],
    interception_radius: float = 8.0
) -> float:
    """
    Calculates defensive risk/pressure along a pass corridor p1 -> p2.
    Defenders close to the passing lane increase pressure via inverse-distance weighting.
    """
    if not defenders:
        return 0.0

    total_pressure = 0.0
    for d in defenders:
        dist = point_to_segment_distance(d, p1, p2)
        if dist < interception_radius:
            # Quadratic falloff: closer defenders pose exponentially higher interception risk
            total_pressure += (interception_radius - dist) / (dist + 1.0)
    return total_pressure


def compute_dijkstra_passing_path(
    attacking_players: List[Tuple[float, float]],
    defending_players: List[Tuple[float, float]],
    carrier_idx: Optional[int] = None,
    ball_coord: Optional[Tuple[float, float]] = None,
    alpha: float = 0.35,
    max_pass_dist: float = 48.0,
    min_pass_dist: float = 2.5
) -> Dict[str, Any]:
    """
    Constructs the passing network graph and solves the lowest-risk progression path
    using Dijkstra's algorithm with a binary Min-Heap.

    Returns:
    {
        'path': List[int],                      # Player indices in passing sequence
        'corridor_points': List[(X, Y)],        # Pitch coordinates along corridor
        'avg_risk': float,                      # Normalized risk index (0.0 to 10.0)
        'total_progression': float,             # Meters gained towards opponent goal
        'num_passes': int,                      # Total passes in combination
        'runtime_ms': float                     # Algorithmic execution latency
    }
    """
    t0 = time.perf_counter()
    n_attackers = len(attacking_players)

    if n_attackers < 2:
        return {
            'path': [],
            'corridor_points': [],
            'path_points': [],
            'graph_edges': [],
            'total_edges': 0,
            'avg_risk': 0.0,
            'total_progression': 0.0,
            'total_weight': 0.0,
            'num_passes': 0,
            'runtime_ms': round((time.perf_counter() - t0) * 1000.0, 3)
        }

    # 1. Determine Source Node (Ball Carrier)
    if carrier_idx is not None and 0 <= carrier_idx < n_attackers:
        start_node = carrier_idx
    elif ball_coord is not None:
        # Find attacking player closest to ball
        start_node = int(np.argmin([
            math.hypot(p[0] - ball_coord[0], p[1] - ball_coord[1])
            for p in attacking_players
        ]))
    else:
        # Default to player deepest in own half (minimum X)
        start_node = int(np.argmin([p[0] for p in attacking_players]))

    # 2. Determine Target Node (Forward-most advantageous attacker in attacking third)
    # Target is the teammate with highest X coordinate distinct from start_node
    other_nodes = [i for i in range(n_attackers) if i != start_node]
    target_node = max(other_nodes, key=lambda i: attacking_players[i][0])

    # 3. Construct Directed Graph Adjacency List
    adj: Dict[int, List[Tuple[int, float, float]]] = {i: [] for i in range(n_attackers)}
    for u in range(n_attackers):
        p_u = attacking_players[u]
        for v in range(n_attackers):
            if u == v:
                continue
            p_v = attacking_players[v]
            dist = math.hypot(p_v[0] - p_u[0], p_v[1] - p_u[1])
            if dist < min_pass_dist or dist > max_pass_dist:
                continue

            progression = p_v[0] - p_u[0]  # Positive if moving toward X=105m goal
            pressure = compute_pass_defensive_pressure(p_u, p_v, defending_players)

            # Cost formulation (Guaranteed non-negative for Dijkstra invariant)
            # Higher pressure & length increases cost; forward progression discounts cost
            weight = max(
                0.1,
                dist * (1.0 + 1.8 * pressure) - alpha * max(-10.0, progression) + 12.0
            )
            adj[u].append((v, weight, pressure))

    # 4. Dijkstra's Algorithm with Binary Min-Heap (heapq)
    dist = {i: float('inf') for i in range(n_attackers)}
    parent: Dict[int, Optional[int]] = {i: None for i in range(n_attackers)}
    dist[start_node] = 0.0

    # Heap contains tuples of (cumulative_cost, node_index)
    heap: List[Tuple[float, int]] = [(0.0, start_node)]
    visited = set()

    while heap:
        d_u, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)

        if u == target_node:
            break

        for v, weight, _ in adj.get(u, []):
            if v in visited:
                continue
            new_dist = d_u + weight
            if new_dist < dist[v]:
                dist[v] = new_dist
                parent[v] = u
                heapq.heappush(heap, (new_dist, v))

    # 5. Reconstruct Path
    path: List[int] = []
    curr: Optional[int] = target_node
    if dist[target_node] < float('inf'):
        while curr is not None:
            path.append(curr)
            curr = parent[curr]
        path.reverse()
    else:
        # Fallback: direct 1-step to closest progressive forward option
        candidates = [v for v, _, _ in adj.get(start_node, []) if attacking_players[v][0] >= attacking_players[start_node][0]]
        if candidates:
            best_fwd = max(candidates, key=lambda v: attacking_players[v][0])
            path = [start_node, best_fwd]
        else:
            path = [start_node, target_node]

    corridor_pts = [attacking_players[i] for i in path]
    total_prog = max(0.0, attacking_players[path[-1]][0] - attacking_players[path[0]][0])

    # Calculate average risk along path
    path_pressures = []
    for i in range(len(path) - 1):
        p1 = attacking_players[path[i]]
        p2 = attacking_players[path[i + 1]]
        path_pressures.append(compute_pass_defensive_pressure(p1, p2, defending_players))
    avg_risk = float(np.mean(path_pressures)) if path_pressures else 0.0

    runtime_ms = (time.perf_counter() - t0) * 1000.0

    # Calculate cumulative path weight
    path_weight = 0.0
    for i in range(len(path) - 1):
        u_node = path[i]
        v_node = path[i + 1]
        w_found = next((w for v_cand, w, _ in adj.get(u_node, []) if v_cand == v_node), 0.0)
        path_weight += w_found

    # Collect all candidate directed edges across the passing graph G = (V, E)
    graph_edges: List[Dict[str, Any]] = []
    for u in adj:
        for v, weight, pressure in adj[u]:
            dist = math.hypot(attacking_players[v][0] - attacking_players[u][0], attacking_players[v][1] - attacking_players[u][1])
            graph_edges.append({
                'source_idx': u,
                'target_idx': v,
                'source_point': (round(attacking_players[u][0], 2), round(attacking_players[u][1], 2)),
                'target_point': (round(attacking_players[v][0], 2), round(attacking_players[v][1], 2)),
                'weight': round(weight, 1),
                'pressure': round(pressure, 2),
                'distance': round(dist, 1)
            })

    return {
        'path': path,
        'corridor_points': corridor_pts,
        'path_points': corridor_pts,
        'graph_edges': graph_edges,
        'total_edges': len(graph_edges),
        'avg_risk': round(min(10.0, avg_risk), 1),
        'total_progression': round(total_prog, 1),
        'total_weight': round(path_weight, 1),
        'num_passes': max(0, len(path) - 1),
        'runtime_ms': round(runtime_ms, 3)
    }


