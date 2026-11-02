"""
Automated validation suite for FW-01: Dijkstra Optimal Passing Network (Graph Theory).
Tests:
1. Defender-to-corridor perpendicular distance calculations.
2. Defensive interception pressure weighting.
3. Dijkstra shortest path solving via Min-Heap (heapq).
4. Full Stage 3 integration & contract compliance.
5. Stage 4 radar & AR overlay corridor rendering.
6. Coach analytics directive generation.
"""

import sys
import os
import time
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from algorithms.passing_network import (
    point_to_segment_distance,
    compute_pass_defensive_pressure,
    compute_dijkstra_passing_path
)
from core.contracts import PitchPositions, TacticalGeometry
from core.pipeline import stage3_geometry, stage4_render_radar, stage4_render_overlay
from visualization.pitch_renderer import TacticalPitchRenderer
from visualization.overlays import BroadcastOverlayEngine
from perception.homography import HomographyTransformer
from algorithms.coach_analytics import evaluate_coach_analytics


def test_geometry_and_pressure():
    print("Test 1: Perpendicular Distance & Defensive Pressure...")
    # Passing line from (10.0, 34.0) to (50.0, 34.0)
    p1 = (10.0, 34.0)
    p2 = (50.0, 34.0)

    # Defender exactly on passing line
    d_on_line = (30.0, 34.0)
    dist = point_to_segment_distance(d_on_line, p1, p2)
    assert abs(dist) < 1e-4, f"Failed: Distance should be 0.0, got {dist}"

    # Defender 5 meters off the line
    d_off_line = (30.0, 39.0)
    dist = point_to_segment_distance(d_off_line, p1, p2)
    assert abs(dist - 5.0) < 1e-4, f"Failed: Distance should be 5.0, got {dist}"

    # Pressure evaluation
    pressure_high = compute_pass_defensive_pressure(p1, p2, [d_on_line])
    pressure_low = compute_pass_defensive_pressure(p1, p2, [d_off_line])
    pressure_none = compute_pass_defensive_pressure(p1, p2, [(30.0, 60.0)]) # > 8m away

    print(f"  Pressure High (on lane): {pressure_high:.2f}")
    print(f"  Pressure Low (5m away): {pressure_low:.2f}")
    print(f"  Pressure None (>8m away): {pressure_none:.2f}")

    assert pressure_high > pressure_low, "Failed: Direct defender must produce higher pressure than distant defender"
    assert pressure_none == 0.0, "Failed: Defender > 8m away should produce 0.0 pressure"
    print("  PASS: Distance and defensive pressure logic mathematically verified.")


def test_dijkstra_solver():
    print("\nTest 2: Dijkstra Algorithm with Min-Heap (heapq)...")
    # Setup tactical scenario:
    # Attacker 0 (Start, deep): (20.0, 34.0)
    # Attacker 1 (Blocked central midfielder): (40.0, 34.0)
    # Attacker 2 (Open wide winger): (35.0, 10.0)
    # Attacker 3 (Target striker in final third): (70.0, 34.0)
    # Defender 0 (Directly blocking path between 0 and 1, and 1 and 3): (38.0, 34.0)
    attackers = [
        (20.0, 34.0), # 0: Ball carrier
        (40.0, 34.0), # 1: Central midfielder (heavily marked)
        (35.0, 10.0), # 2: Wide open winger
        (70.0, 34.0)  # 3: Striker
    ]
    defenders = [
        (38.0, 34.0), # Heavily blocks pass 0 -> 1 and 1 -> 3
        (39.0, 35.0)
    ]

    t0 = time.perf_counter()
    res = compute_dijkstra_passing_path(attackers, defenders, carrier_idx=0)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    print(f"  Dijkstra Path: {res['path']}")
    print(f"  Total Progression: {res['total_progression']}m")
    print(f"  Average Corridor Risk: {res['avg_risk']}")
    print(f"  Number of Passes: {res['num_passes']}")
    print(f"  Execution Latency: {elapsed_ms:.4f} ms")

    assert len(res['path']) >= 2, "Failed: Expected at least 2 nodes in path"
    assert res['path'][0] == 0, "Failed: Path must start at carrier (0)"
    assert res['path'][-1] == 3, "Failed: Path must target forward striker (3)"
    # Dijkstra should route via winger (node 2) to bypass central defender block
    assert 2 in res['path'], "Failed: Dijkstra did not route around central defensive block via winger!"
    assert elapsed_ms < 1.0, f"Failed: Latency too high ({elapsed_ms} ms, expected < 1.0 ms)"
    print("  PASS: Dijkstra successfully circumvented defensive block via Min-Heap!")


def test_full_pipeline_integration():
    print("\nTest 3: Full Pipeline Stage 3 & Stage 4 Integration...")
    # Create synthetic positions contract
    pos = PitchPositions(
        frame_idx=10,
        team_a=[(20.0, 30.0), (35.0, 15.0), (45.0, 50.0), (68.0, 34.0)],
        team_b=[(40.0, 30.0), (42.0, 38.0), (55.0, 32.0)],
        ball=(20.0, 30.0),
        carrier_idx=0,
        detected_count=7
    )

    # 1. Stage 3
    geom = stage3_geometry(pos, hull_algo='graham')
    assert geom.dijkstra_res is not None, "Failed: geom.dijkstra_res is None!"
    assert 'path' in geom.dijkstra_res, "Failed: 'path' not in geom.dijkstra_res!"
    assert 'corridor_points' in geom.dijkstra_res, "Failed: 'corridor_points' not in geom.dijkstra_res!"
    print(f"  Stage 3 Dijkstra Output Verified: {len(geom.dijkstra_res['path'])} nodes in sequence")

    # 2. Stage 4A (Radar Canvas)
    renderer = TacticalPitchRenderer(width=1120, height=730)
    radar = stage4_render_radar(pos, geom, renderer, show_voronoi=True, show_hull=True)
    assert radar.shape == (730, 1120, 3), f"Failed radar canvas shape: {radar.shape}"
    print("  Stage 4A 2D Radar Canvas with Dijkstra Corridor Rendered Successfully.")

    # 3. Stage 4B (Broadcast AR Overlay)
    trans = HomographyTransformer(frame_width=960, frame_height=540, camera_mode='sideline')
    overlay_engine = BroadcastOverlayEngine(transformer=trans)
    raw_frame = np.zeros((540, 960, 3), dtype=np.uint8)
    overlay = stage4_render_overlay(raw_frame, pos, geom, overlay_engine, show_voronoi=True, show_hull=True)
    assert overlay.shape == (540, 960, 3), f"Failed overlay shape: {overlay.shape}"
    print("  Stage 4B Broadcast AR Overlay with Projected Dijkstra Corridor Rendered Successfully.")

    # 4. Coach Analytics
    vor_res = {'team_a_pct': geom.team_a_pct, 'team_b_pct': geom.team_b_pct}
    coach = evaluate_coach_analytics(
        pos.team_a, pos.team_b,
        {'hull': geom.hull_polygon, 'area_sq_m': geom.hull_area},
        vor_res, geom.dijkstra_res, None, pos.ball
    )
    assert 'pass_directive' in coach, "Failed: pass_directive not in coach output!"
    print(f"  Coach Analytics Directive: '{coach['pass_directive']}' (Line-Breaking: {coach.get('is_line_breaking')})")
    print("  PASS: End-to-end pipeline contracts with Dijkstra Passing Network fully operational!")


def main():
    print("=" * 65)
    print("RUNNING FW-01 DIJKSTRA OPTIMAL PASSING NETWORK VALIDATION SUITE")
    print("=" * 65)
    test_geometry_and_pressure()
    test_dijkstra_solver()
    test_full_pipeline_integration()
    print("=" * 65)
    print("ALL FW-01 DIJKSTRA PASSING NETWORK CHECKS PASSED WITH 100% SUCCESS!")
    print("=" * 65)


if __name__ == "__main__":
    main()
