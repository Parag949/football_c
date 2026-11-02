#!/usr/bin/env python3
"""
scratch/test_tactical_ruler.py

Comprehensive Automated Validation Suite for:
1. FW-04: ONNX Runtime Multi-Thread Concurrency Tuning
2. FW-06: Interactive Tactical Distance & VAR Offside Ruler
3. Stage 4 Renderers (2D Pitch Radar & Broadcast AR Overlay integration)
"""

import sys
import os
import math
import numpy as np

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from algorithms.tactical_ruler import compute_tactical_ruler
from perception.detector import FootballDetector
from core.contracts import PitchPositions, TacticalGeometry
from perception.homography import HomographyTransformer
from visualization.pitch_renderer import TacticalPitchRenderer
from visualization.overlays import BroadcastOverlayEngine
from core.pipeline import stage4_render_radar, stage4_render_overlay


def test_tactical_ruler_math():
    print("🧪 [Test 1] Tactical Ruler Mathematical Accuracy...")
    # 3-4-5 Triangle: (10, 20) to (13, 24) -> dx=3, dy=4, dist=5
    pt_a = (10.0, 20.0)
    pt_b = (13.0, 24.0)
    
    attacking = [(10.0, 20.0), (13.0, 24.0), (80.0, 34.0)]
    # Defending: 3 defenders at X = 75.0, 70.0, 85.0.
    # Second deepest defender (attacking left-to-right, towards X=105):
    # Defenders sorted descending by X: [85.0, 75.0, 70.0] -> 2nd deepest is 75.0
    defending = [(75.0, 30.0), (70.0, 40.0), (85.0, 35.0)]

    ruler = compute_tactical_ruler(
        point_a=pt_a,
        point_b=pt_b,
        label_a="Player A1",
        label_b="Player A2",
        offside_x=None,  # Auto-detect
        attacking_players=attacking,
        defending_players=defending,
        attack_direction="right"
    )

    assert abs(ruler['distance_m'] - 5.0) < 1e-4, f"Expected dist 5.0, got {ruler['distance_m']}"
    assert abs(ruler['delta_x_m'] - 3.0) < 1e-4, f"Expected dx 3.0, got {ruler['delta_x_m']}"
    assert abs(ruler['delta_y_m'] - 4.0) < 1e-4, f"Expected dy 4.0, got {ruler['delta_y_m']}"
    assert abs(ruler['offside_x_m'] - 75.0) < 1e-4, f"Expected offside_x 75.0, got {ruler['offside_x_m']}"
    
    # Attacking player at (80.0, 34.0) is beyond 75.0 -> Offside breach of 5.0m
    assert len(ruler['offside_breaches']) == 1, f"Expected 1 offside breach, got {len(ruler['offside_breaches'])}"
    breach = ruler['offside_breaches'][0]
    assert breach['player_idx'] == 2, f"Expected player index 2, got {breach['player_idx']}"
    assert abs(breach['margin_m'] - 5.0) < 1e-4, f"Expected margin 5.0m, got {breach['margin_m']}"

    print("   ✅ Euclidean math, auto-detected offside line, and breach calculation verified!")


def test_tactical_ruler_manual_and_left_attack():
    print("🧪 [Test 2] Manual Offside Line and Left-Attacking Direction...")
    pt_a = (50.0, 34.0)
    pt_b = (60.0, 34.0)
    attacking = [(20.0, 30.0), (35.0, 40.0)]
    # Attacking to the left (towards X=0):
    # Defenders sorted ascending by X: [15.0, 25.0, 40.0] -> 2nd deepest is 25.0
    defending = [(25.0, 30.0), (15.0, 34.0), (40.0, 50.0)]

    ruler_auto = compute_tactical_ruler(
        point_a=pt_a,
        point_b=pt_b,
        offside_x=None,
        attacking_players=attacking,
        defending_players=defending,
        attack_direction="left"
    )
    assert abs(ruler_auto['offside_x_m'] - 25.0) < 1e-4, f"Expected 25.0m, got {ruler_auto['offside_x_m']}"
    # Attacker at 20.0 is beyond 25.0 (leftwards) -> breach!
    assert len(ruler_auto['offside_breaches']) == 1
    assert abs(ruler_auto['offside_breaches'][0]['margin_m'] - 5.0) < 1e-4

    # Test explicit manual line override:
    ruler_manual = compute_tactical_ruler(
        point_a=pt_a,
        point_b=pt_b,
        offside_x=68.5,
        attacking_players=attacking,
        defending_players=defending,
        attack_direction="right"
    )
    assert abs(ruler_manual['offside_x_m'] - 68.5) < 1e-4, f"Expected manual line 68.5m, got {ruler_manual['offside_x_m']}"
    print("   ✅ Manual line override & bidirectional tactical calculation verified!")


def test_onnx_concurrency_telemetry():
    print("🧪 [Test 3] ONNX Concurrency Tuning & Telemetry (FW-04)...")
    detector = FootballDetector(num_threads=3)
    telemetry = detector.get_onnx_telemetry()
    
    assert telemetry['intra_op_threads'] == 3, f"Expected 3 intra_op_threads, got {telemetry['intra_op_threads']}"
    assert telemetry['inter_op_threads'] == 1, f"Expected 1 inter_op_threads, got {telemetry['inter_op_threads']}"
    assert telemetry['execution_mode'] == 'ORT_SEQUENTIAL', f"Expected ORT_SEQUENTIAL, got {telemetry['execution_mode']}"
    assert telemetry['graph_optimization'] == 'ORT_ENABLE_ALL', f"Expected ORT_ENABLE_ALL, got {telemetry['graph_optimization']}"
    assert telemetry['provider'] == 'CPUExecutionProvider', f"Expected CPUExecutionProvider, got {telemetry['provider']}"
    
    detector.set_num_threads(4)
    telemetry2 = detector.get_onnx_telemetry()
    assert telemetry2['intra_op_threads'] == 4, f"Expected 4 intra_op_threads, got {telemetry2['intra_op_threads']}"
    print("   ✅ Detector ONNX session options and concurrency telemetry verified!")


def test_stage4_renderers_with_ruler():
    print("🧪 [Test 4] Stage 4 Rendering Integration with Tactical Ruler...")
    transformer = HomographyTransformer(camera_mode="sideline")
    renderer = TacticalPitchRenderer(width=1000, height=650, padding=35)
    overlay_engine = BroadcastOverlayEngine(transformer)

    team_a = [(20.0, 30.0), (35.0, 45.0), (55.0, 20.0), (82.0, 38.0)]
    team_b = [(30.0, 30.0), (45.0, 40.0), (75.0, 34.0), (85.0, 34.0)]
    ball = (36.0, 44.0)

    pos = PitchPositions(
        frame_idx=0,
        team_a=team_a,
        team_b=team_b,
        ball=ball,
        carrier_idx=1,
        detected_count=len(team_a) + len(team_b)
    )

    geom = TacticalGeometry(
        frame_idx=0,
        hull_polygon=[(30.0, 30.0), (45.0, 40.0), (75.0, 34.0)],
        hull_player_indices=[0, 1, 2],
        hull_area=150.0,
        hull_compactness=0.75,
        hull_vertices=3,
        voronoi_cells_a=[],
        voronoi_cells_b=[],
        team_a_pct=52.0,
        team_b_pct=48.0,
        dijkstra_res=None,
        latency_ms=1.2
    )

    ruler_data = compute_tactical_ruler(
        point_a=team_a[0],
        point_b=team_a[1],
        label_a="Player A1",
        label_b="Player A2",
        offside_x=None,
        attacking_players=team_a,
        defending_players=team_b
    )

    raw_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    # Render 2D Radar
    radar_img = stage4_render_radar(
        positions=pos,
        geometry=geom,
        renderer=renderer,
        show_voronoi=False,
        show_hull=True,
        show_dijkstra=False,
        ruler_data=ruler_data
    )
    assert isinstance(radar_img, np.ndarray), "Radar render must be np.ndarray"
    assert radar_img.shape == (650, 1000, 3), f"Radar shape mismatch: {radar_img.shape}"

    # Render Broadcast AR Overlay
    overlay_img = stage4_render_overlay(
        raw_frame=raw_frame,
        positions=pos,
        geometry=geom,
        overlay_engine=overlay_engine,
        show_voronoi=False,
        show_hull=True,
        show_dijkstra=False,
        ruler_data=ruler_data
    )
    assert isinstance(overlay_img, np.ndarray), "Overlay render must be np.ndarray"
    assert overlay_img.shape == (720, 1280, 3), f"Overlay shape mismatch: {overlay_img.shape}"

    print("   ✅ Stage 4 2D Radar and Broadcast AR Overlay cleanly rendered with tactical ruler data!")


if __name__ == "__main__":
    print("=" * 70)
    print("🎯 RUNNING TACTICAL RULER & ONNX TUNING TEST SUITE (FW-04 & FW-06)")
    print("=" * 70)
    test_tactical_ruler_math()
    test_tactical_ruler_manual_and_left_attack()
    test_onnx_concurrency_telemetry()
    test_stage4_renderers_with_ruler()
    print("=" * 70)
    print("🎉 ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)
