"""
Algorithmic Core Engine for Real-Time Football Analytics:
- Defensive Cohesion (Convex Hull: Graham Scan & Jarvis March)
- Spatial Dominance (Voronoi Tessellation & Pitch Partition)
- Tactical Coach Analytics (Derived Defensive Block & Territory Metrics)
- DAA Benchmarks & Complexity Evaluation
"""

from .convex_hull import graham_scan, jarvis_march, compute_defensive_cohesion, polygon_area_shoelace
from .voronoi_dominance import compute_spatial_dominance, compute_bounded_voronoi
from .daa_benchmarks import benchmark_convex_hull, benchmark_voronoi_dominance
from .coach_analytics import evaluate_coach_analytics

__all__ = [
    'graham_scan',
    'jarvis_march',
    'compute_defensive_cohesion',
    'polygon_area_shoelace',
    'compute_spatial_dominance',
    'compute_bounded_voronoi',
    'benchmark_convex_hull',
    'benchmark_voronoi_dominance',
    'evaluate_coach_analytics'
]
