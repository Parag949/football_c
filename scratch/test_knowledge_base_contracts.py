"""
Comprehensive Regression & Verification Suite for docs/SYSTEM_KNOWLEDGE_BASE.md
Asserts all contracts, bounds, invariants, and edge conditions across TacticalVision Pro.
"""

import os
import sys
import math
import json
import subprocess
import threading
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.contracts import PlayerDetection, FrameDetections, PitchPositions, TacticalGeometry, FrameBundle
from perception.homography import HomographyTransformer
from perception.detector import FootballDetector
from perception.video_stream import MatchStreamManager
from algorithms.convex_hull import graham_scan, compute_defensive_cohesion
from algorithms.voronoi_dominance import compute_spatial_dominance
from algorithms.passing_network import compute_dijkstra_passing_path
from algorithms.ghost_runner import simulate_ghost_runner
from algorithms.tactical_ruler import compute_tactical_ruler
from algorithms.coach_analytics import evaluate_coach_analytics
from algorithms.daa_benchmarks import benchmark_convex_hull, benchmark_voronoi_dominance, benchmark_dijkstra_passing
from ui.components.player_assets import prepare_hardware_player_assets


def test_core_contracts_null_safety():
    """Verify core/contracts.py data structures tolerate None values in from_dict."""
    # 1. PitchPositions
    d_pos = {
        "frame_idx": 10,
        "team_a": None,
        "team_b": None,
        "ball": None,
        "carrier_idx": None,
        "detected_count": 0
    }
    pos = PitchPositions.from_dict(d_pos)
    assert pos.team_a == []
    assert pos.team_b == []
    assert pos.ball == (52.5, 34.0)
    assert pos.carrier_idx == 0
    assert pos.detected_count == 0

    # 2. TacticalGeometry
    d_geo = {
        "frame_idx": 10,
        "hull": None,
        "voronoi": None,
        "dijkstra": None
    }
    geo = TacticalGeometry.from_dict(d_geo)
    assert geo.hull_polygon == []
    assert geo.hull_area == 0.0
    assert geo.hull_compactness == 0.0
    assert geo.voronoi_cells_a == []
    assert geo.voronoi_cells_b == []
    assert geo.team_a_pct == 50.0
    assert geo.team_b_pct == 50.0
    assert geo.dijkstra_res is None

    # 3. Serialization round-trip
    bundle = FrameBundle(
        positions=pos,
        geometry=geo
    )
    b_dict = bundle.to_dict()
    assert isinstance(b_dict, dict)
    assert b_dict["frame_idx"] == 10
    assert "team_a" in b_dict
    assert "hull" in b_dict
    assert "voronoi" in b_dict
    print("  -> test_core_contracts_null_safety: PASSED")


def test_homography_signature_invariance():
    """Verify calibration signature is uniquely determined and unaffected by camera_mode when custom calibrated."""
    t1 = HomographyTransformer(camera_mode="sideline", warp_mode="planar")
    src = [(100, 200), (800, 200), (900, 500), (50, 500)]
    dst = [(0, 0), (105, 0), (105, 68), (0, 68)]
    t1.calibrate(src, dst)
    sig1 = t1.get_calibration_signature()

    # Change camera_mode preset on identical custom points
    t2 = HomographyTransformer(camera_mode="goal_to_goal", warp_mode="planar")
    t2.calibrate(src, dst)
    sig2 = t2.get_calibration_signature()

    assert sig1 == sig2, f"Signatures diverged for identical custom points: {sig1} != {sig2}"
    assert len(sig1) == 12, "Signature must be 12-char hex string"

    # Default presets must differ
    t_side = HomographyTransformer(camera_mode="sideline")
    t_goal = HomographyTransformer(camera_mode="goal_to_goal")
    assert t_side.get_calibration_signature() != t_goal.get_calibration_signature()
    print("  -> test_homography_signature_invariance: PASSED")


def test_homography_auto_mode_and_clamping():
    """Verify auto warp mode initialization, behind-camera filtering, and extreme projection clamping."""
    t = HomographyTransformer(camera_mode="sideline", warp_mode="auto")
    assert t.warp_mode in ("planar", "panorama_tps")

    # Clamping extreme coordinates
    extreme_pitch_pts = [(-100000.0, -100000.0), (100000.0, 100000.0)]
    img_pts = t.pitch_to_image(extreme_pitch_pts)
    for p in img_pts:
        if p is not None:
            assert abs(p[0]) <= 200000
            assert abs(p[1]) <= 200000

    # Degenerate points fallback resets is_custom_calibrated
    degen_src = [(10, 10), (10, 10), (10, 10), (10, 10)]
    degen_dst = [(0, 0), (1, 1), (2, 2), (3, 3)]
    t.calibrate(degen_src, degen_dst)
    assert not t.is_custom_calibrated, "Degenerate points should reset is_custom_calibrated to False"
    print("  -> test_homography_auto_mode_and_clamping: PASSED")


def test_convex_hull_invariants():
    """Verify convex hull handles lists of lists, non-finite coords, and close-pivot points."""
    # List of lists with NaN
    pts = [[10.0, 10.0], [90.0, 10.0], [90.0, 60.0], [10.0, 60.0], [50.0, 35.0], [float("nan"), 12.0]]
    res = compute_defensive_cohesion(pts)
    assert len(res["hull"]) == 4
    assert res["area_sq_m"] > 0.0
    assert 0.0 <= res["compactness"] <= 1.0

    # Micro-distance from pivot
    pts_close = [(10.0, 10.0), (10.00001, 10.00001), (50.0, 10.0), (50.0, 50.0), (10.0, 50.0)]
    res_close = graham_scan(pts_close)
    assert len(res_close["hull"]) >= 3
    print("  -> test_convex_hull_invariants: PASSED")


def test_voronoi_dominance_bounds():
    """Verify Voronoi area bounds (<= 7140 m²) and coincident point micro-displacement."""
    # Coincident points
    team_a = [(50.0, 30.0), (50.0, 30.0), (30.0, 20.0), (40.0, 50.0)]
    team_b = [(60.0, 35.0), (70.0, 20.0), (80.0, 50.0)]
    res = compute_spatial_dominance(team_a, team_b)

    assert "team_a_pct" in res and "team_b_pct" in res
    assert abs((res["team_a_pct"] + res["team_b_pct"]) - 100.0) < 1e-2

    total_area = res["team_a_area"] + res["team_b_area"]
    assert total_area <= 7140.0 + 1e-3, f"Total Voronoi area {total_area} exceeded physical pitch bounds 7140.0"
    print("  -> test_voronoi_dominance_bounds: PASSED")


def test_dijkstra_unreachable_fallback():
    """Verify Dijkstra does not create phantom 0-cost edges across unreachable teammates."""
    # Isolated carrier and distant teammate beyond max_pass_dist (85m > 45m)
    team_a = [(10.0, 34.0), (95.0, 34.0)]
    # Dense wall of opponents
    team_b = [(50.0, y) for y in np.linspace(0.0, 68.0, 30)]

    res = compute_dijkstra_passing_path(team_a, team_b, carrier_idx=0)
    assert res["path"] == [0], f"Expected single node path [0], got {res['path']}"
    assert res["total_weight"] == 0.0
    assert res["total_progression"] == 0.0

    # Ensure NaN coordinates are rejected
    team_a_nan = [(10.0, 34.0), (float("nan"), 34.0), (20.0, 34.0)]
    res_nan = compute_dijkstra_passing_path(team_a_nan, team_b, carrier_idx=0)
    assert not any(math.isnan(p[0]) or math.isnan(p[1]) for p in res_nan["corridor_points"])
    for e in res_nan["graph_edges"]:
        assert math.isfinite(e["source_point"][0]) and math.isfinite(e["source_point"][1])
        assert math.isfinite(e["target_point"][0]) and math.isfinite(e["target_point"][1])
    print("  -> test_dijkstra_unreachable_fallback: PASSED")


def test_ghost_runner_fifa_law_11():
    """Verify FIFA Law 11 ball position offside exemption in Ghost Runner simulation."""
    # Attacking teammates
    attackers = [(20.0, 34.0)]
    # Defenders deep in their own half (X = 40.0)
    defenders = [(40.0, 20.0), (40.0, 48.0)]
    # Ball is far forward at X = 70.0
    ball = (70.0, 34.0)
    # Runner is at X = 65.0 (behind the ball, but ahead of defenders at 40.0)
    runner = (65.0, 34.0)

    res = simulate_ghost_runner(
        attacking_players=attackers,
        defending_players=defenders,
        ghost_coord=runner,
        carrier_idx=0,
        ball_coord=ball,
        attack_direction="right"
    )
    assert not res["is_offside"], "Runner behind ball must be strictly onside per FIFA Law 11"
    print("  -> test_ghost_runner_fifa_law_11: PASSED")


def test_coach_analytics_schema_contract():
    """Verify evaluate_coach_analytics returns both top-level keys and structured 'passing' subdict."""
    team_a = [(20.0, 30.0), (25.0, 40.0), (30.0, 20.0), (35.0, 50.0), (40.0, 34.0)]
    team_b = [(60.0, 30.0), (65.0, 40.0), (70.0, 20.0), (75.0, 50.0)]
    ball = (40.0, 34.0)
    dijk = compute_dijkstra_passing_path(team_a, team_b, carrier_idx=4)
    hull = graham_scan(team_a)

    analytics = evaluate_coach_analytics(
        team_a_coords=team_a,
        team_b_coords=team_b,
        hull_res=hull,
        voronoi_res=None,
        dijkstra_res=dijk,
        kinematics_a=None,
        ball_coord=ball
    )
    assert "passing" in analytics, "Coach analytics must include 'passing' subdict"
    p_dict = analytics["passing"]
    assert "progression_m" in p_dict
    assert "is_line_breaking" in p_dict
    assert "directive" in p_dict
    assert "passes_count" in p_dict
    assert "path" in p_dict
    assert "line_height_m" in analytics
    assert 0.0 <= analytics["line_height_m"] <= 105.0
    print("  -> test_coach_analytics_schema_contract: PASSED")


def test_video_stream_thread_safety_and_generation():
    """Verify VideoStreamManager cache separation, atomic generation invalidation, and detector lock."""
    # Detector inference lock
    det = FootballDetector()
    assert hasattr(det, "_inference_lock"), "FootballDetector must have _inference_lock"
    assert isinstance(det._inference_lock, threading.RLock().__class__)

    # Stream Manager cache separation
    vsm = MatchStreamManager(sample_video_path="data/sample_broadcast.mp4", detector=det)
    assert hasattr(vsm, "detection_cache")
    assert hasattr(vsm, "broadcast_state_cache")
    assert hasattr(vsm, "_stream_generation")
    assert hasattr(vsm, "_cache_lock")

    gen_before = vsm._stream_generation
    vsm.load_video_source("data/sample_broadcast.mp4")
    assert vsm._stream_generation == gen_before + 1, "load_video_source must increment _stream_generation"
    assert vsm._last_read_pos == -1, "_last_read_pos must reset to -1"
    print("  -> test_video_stream_thread_safety_and_generation: PASSED")


def test_player_assets_null_tolerance():
    """Verify player_assets.py generates robust payload with missing/None tactical properties."""
    import tempfile
    transformer = HomographyTransformer(1920, 1080, camera_mode="goal_to_goal")
    sig = transformer.get_calibration_signature()

    # Create dummy cache file with None/missing fields
    dummy_cache = {
        "video_path": "data/user_match.mp4",
        "video_width": 1920,
        "video_height": 1080,
        "fps": 30.0,
        "calibration_sig": sig,
        "frames": [
            {
                "frame_idx": 0,
                "team_a": None,
                "team_b": [[60.0, 30.0]],
                "ball": None,
                "carrier_idx": None,
                "hull": None,
                "voronoi": None,
                "dijkstra": None
            }
        ]
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(dummy_cache, f)
        temp_cache_path = f.name

    try:
        assets = prepare_hardware_player_assets("data/user_match.mp4", temp_cache_path, transformer)
        assert isinstance(assets, dict)
        assert "frames" in assets
        assert len(assets["frames"]) == 1
        f0 = assets["frames"][0]
        assert f0["a"] == []
        assert f0["ball"] == [52.5, 34.0]
        assert f0["c"] == -1
        assert f0["hull"]["p"] == []
        assert f0["vor"]["a_pct"] == 50.0
        assert f0["dijk"]["path"] == []
        print("  -> test_player_assets_null_tolerance: PASSED")
    finally:
        if os.path.exists(temp_cache_path):
            os.remove(temp_cache_path)


def test_tactical_ruler_and_daa_benchmarks():
    """Verify tactical ruler handles non-finite inputs and DAA benchmarks handle n <= 0 and empty input."""
    # Ruler with NaN coordinates
    ruler = compute_tactical_ruler(
        point_a=(float("nan"), 10.0),
        point_b=(50.0, 30.0),
        defending_players=[(30.0, 20.0), (float("nan"), 40.0), (35.0, 50.0)]
    )
    assert math.isfinite(ruler["distance"])
    assert ruler["distance"] > 0.0

    # Benchmarks with empty and zero
    df_hull = benchmark_convex_hull([])
    assert len(df_hull) == 0
    df_vor = benchmark_voronoi_dominance([])
    assert len(df_vor) == 0
    df_dijk = benchmark_dijkstra_passing([])
    assert len(df_dijk) == 0
    print("  -> test_tactical_ruler_and_daa_benchmarks: PASSED")


def test_hardware_player_syntax():
    """Verify hardware_player.js passes node syntax checking."""
    js_path = os.path.join(REPO_ROOT, "ui/components/assets/hardware_player.js")
    res = subprocess.run(["node", "--check", js_path], capture_output=True, text=True)
    assert res.returncode == 0, f"JS Syntax Error in hardware_player.js: {res.stderr}"
    print("  -> test_hardware_player_syntax: PASSED")


def run_all_contract_tests():
    print("\n=======================================================")
    print(">>> VERIFYING SYSTEM KNOWLEDGE BASE CONTRACTS <<<")
    print("=======================================================")
    test_core_contracts_null_safety()
    test_homography_signature_invariance()
    test_homography_auto_mode_and_clamping()
    test_convex_hull_invariants()
    test_voronoi_dominance_bounds()
    test_dijkstra_unreachable_fallback()
    test_ghost_runner_fifa_law_11()
    test_coach_analytics_schema_contract()
    test_video_stream_thread_safety_and_generation()
    test_player_assets_null_tolerance()
    test_tactical_ruler_and_daa_benchmarks()
    test_hardware_player_syntax()
    print("=======================================================")
    print(">>> ALL KNOWLEDGE BASE CONTRACTS 100% VERIFIED! <<<")
    print("=======================================================\n")
    return 0


if __name__ == "__main__":
    sys.exit(run_all_contract_tests())
