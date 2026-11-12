"""
Empirical DAA Benchmarking Engine
Provides empirical performance analysis charting input complexity N (number of tracked nodes)
against execution runtime (latency in milliseconds) and operation counts across paradigms:
1. Graham Scan vs Jarvis March (Convex Hull)
2. Fortune's Sweep-Line Voronoi vs Delaunay / Grid Partition
3. Dijkstra (Min-Heap) vs Greedy Passing Search
"""

import time
import random
import math
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from algorithms.convex_hull import graham_scan, jarvis_march
from algorithms.voronoi_dominance import compute_bounded_voronoi
from algorithms.passing_network import compute_dijkstra_passing_path


def generate_random_pitch_nodes(n: int, pitch_length: float = 105.0, pitch_width: float = 68.0) -> List[Tuple[float, float]]:
    """Generates n random coordinates uniformly distributed on the pitch."""
    return [(random.uniform(5.0, pitch_length - 5.0), random.uniform(5.0, pitch_width - 5.0)) for _ in range(n)]


def benchmark_convex_hull(node_counts: List[int] = None, iterations: int = 5) -> pd.DataFrame:
    """
    Benchmarks Graham Scan O(N log N) vs Jarvis March O(N * H)
    across varying input complexities.
    """
    cols = [
        'Nodes (N)', 'Hull Vertices (H)', 'Graham Scan Runtime (ms)',
        'Jarvis March Runtime (ms)', 'Graham Scan Ops', 'Jarvis March Ops',
        'Theoretical O(N log N)', 'Theoretical O(N * H)'
    ]
    if node_counts is None:
        node_counts = [5, 10, 15, 22, 50, 100, 200, 500]
    valid_counts = [max(3, int(n)) for n in node_counts if n is not None and n > 0]
    if not valid_counts:
        return pd.DataFrame(columns=cols)
        
    records = []
    
    for n in valid_counts:
        graham_times = []
        jarvis_times = []
        graham_ops_list = []
        jarvis_ops_list = []
        hull_sizes = []
        
        for _ in range(iterations):
            pts = generate_random_pitch_nodes(n)
            
            # Benchmark Graham Scan
            res_g = graham_scan(pts)
            graham_times.append(res_g['runtime_ms'])
            graham_ops_list.append(res_g['operations'])
            hull_sizes.append(len(res_g['hull']))
            
            # Benchmark Jarvis March
            res_j = jarvis_march(pts)
            jarvis_times.append(res_j['runtime_ms'])
            jarvis_ops_list.append(res_j['operations'])
            
        avg_h = np.mean(hull_sizes)
        records.append({
            'Nodes (N)': n,
            'Hull Vertices (H)': round(avg_h, 1),
            'Graham Scan Runtime (ms)': np.mean(graham_times),
            'Jarvis March Runtime (ms)': np.mean(jarvis_times),
            'Graham Scan Ops': int(np.mean(graham_ops_list)),
            'Jarvis March Ops': int(np.mean(jarvis_ops_list)),
            'Theoretical O(N log N)': round(n * math.log2(n), 1) if n > 1 else 1,
            'Theoretical O(N * H)': round(n * avg_h, 1)
        })
        
    return pd.DataFrame(records)


def benchmark_voronoi_dominance(node_counts: List[int] = None, iterations: int = 5) -> pd.DataFrame:
    """
    Benchmarks Fortune's sweep-line Voronoi tessellation latency across node scale.
    """
    cols = ['Tracked Nodes (N)', 'Voronoi Runtime (ms)', 'FPS Capability', 'Theoretical O(N log N)']
    if node_counts is None:
        node_counts = [6, 11, 22, 44, 80, 150, 300]
    valid_counts = [max(3, int(n)) for n in node_counts if n is not None and n > 0]
    if not valid_counts:
        return pd.DataFrame(columns=cols)
        
    records = []
    for n in valid_counts:
        times = []
        for _ in range(iterations):
            pts = generate_random_pitch_nodes(n)
            t0 = time.perf_counter()
            _ = compute_bounded_voronoi(pts)
            elapsed = (time.perf_counter() - t0) * 1000.0
            times.append(elapsed)
            
        records.append({
            'Tracked Nodes (N)': n,
            'Voronoi Runtime (ms)': np.mean(times),
            'FPS Capability': round(1000.0 / (np.mean(times) + 1e-6), 1),
            'Theoretical O(N log N)': round(n * math.log2(n), 1) if n > 1 else 1
        })
        
    return pd.DataFrame(records)


def benchmark_dijkstra_passing(node_counts: List[int] = None, iterations: int = 5) -> pd.DataFrame:
    """
    Benchmarks Dijkstra (Min-Heap) O((V + E) log V) vs Greedy Search across node scale.
    """
    cols = [
        'Attacking Nodes (|V|)', 'Graph Edges (|E|)', 'Dijkstra (ms)',
        'Greedy Search (ms)', 'Theoretical O((V+E)log V)'
    ]
    if node_counts is None:
        node_counts = [4, 6, 8, 11, 16, 22]
    valid_counts = [max(2, int(n)) for n in node_counts if n is not None and n > 0]
    if not valid_counts:
        return pd.DataFrame(columns=cols)

    records = []
    for n in valid_counts:
        dijkstra_times = []
        greedy_times = []
        for _ in range(iterations):
            attackers = generate_random_pitch_nodes(n)
            defenders = generate_random_pitch_nodes(min(n, 11))

            # Benchmark Dijkstra Min-Heap
            res = compute_dijkstra_passing_path(attackers, defenders)
            dijkstra_times.append(res['runtime_ms'])

            # Benchmark Simple Greedy Forward Search
            t0 = time.perf_counter()
            curr = 0
            greedy_path = [curr]
            while curr < n - 1:
                fwd_opts = [j for j in range(n) if attackers[j][0] > attackers[curr][0]]
                if not fwd_opts:
                    break
                next_node = min(fwd_opts, key=lambda j: math.hypot(attackers[j][0]-attackers[curr][0], attackers[j][1]-attackers[curr][1]))
                greedy_path.append(next_node)
                curr = next_node
            greedy_elapsed = (time.perf_counter() - t0) * 1000.0
            greedy_times.append(greedy_elapsed)

        v = n
        e = v * (v - 1)
        records.append({
            'Attacking Nodes (|V|)': v,
            'Graph Edges (|E|)': e,
            'Dijkstra (ms)': round(np.mean(dijkstra_times), 3),
            'Greedy Search (ms)': round(np.mean(greedy_times), 3),
            'Theoretical O((V+E)log V)': round((v + e) * math.log2(v), 1) if v > 1 else 1
        })

    return pd.DataFrame(records)




