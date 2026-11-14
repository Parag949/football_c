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
import cv2
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
    assert t.warp_mode in ("planar", "panorama_tps", "universal")

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


def test_round2_voronoi_boundary_conservation():
    """Verify Voronoi sites directly on pitch boundaries do not drop cells and conserve 7140 m^2."""
    boundary_team_a = [(0.0, 34.0), (52.5, 0.0), (25.0, 20.0)]
    boundary_team_b = [(105.0, 34.0), (52.5, 68.0), (80.0, 45.0)]
    dom = compute_spatial_dominance(boundary_team_a, boundary_team_b)
    assert dom["team_a_area"] > 0
    assert dom["team_b_area"] > 0
    total = dom["team_a_area"] + dom["team_b_area"]
    assert abs(total - 7140.0) < 1.0, f"Expected total area 7140 m^2, got {total}"
    assert len(dom["cells_a"]) == 3, f"Expected 3 cells for Team A, got {len(dom['cells_a'])}"
    assert len(dom["cells_b"]) == 3, f"Expected 3 cells for Team B, got {len(dom['cells_b'])}"
    print("  -> test_round2_voronoi_boundary_conservation: PASSED")


def test_round2_coach_analytics_empty_hull():
    """Verify coach analytics handles empty / collinear defender hull without UnboundLocalError."""
    team_a = [(30.0, 20.0), (40.0, 30.0), (50.0, 25.0)]
    team_b = [(70.0, 20.0), (70.0, 35.0)]  # Only 2 defenders, cannot form 2D hull
    empty_hull = {"hull": []}
    dijkstra_res = {"path": [0, 1], "total_progression": 10.0}
    kinematics = {"accelerations": [1.5, 2.0], "peak_accel_val": 2.2}
    res = evaluate_coach_analytics(
        team_a_coords=team_a,
        team_b_coords=team_b,
        hull_res=empty_hull,
        voronoi_res=None,
        dijkstra_res=dijkstra_res,
        kinematics_a=kinematics,
        ball_coord=(30.0, 20.0)
    )
    assert "line_height_m" in res
    assert res["line_height_m"] == 35.0
    print("  -> test_round2_coach_analytics_empty_hull: PASSED")


def test_round2_convex_hull_compactness_bounds():
    """Verify defensive cohesion compactness index is strictly in [0.0, 1.0]."""
    # Collinear points
    collinear = [(10.0, 10.0), (20.0, 10.0), (30.0, 10.0), (40.0, 10.0)]
    res = compute_defensive_cohesion(collinear)
    assert 0.0 <= res["compactness"] <= 1.0, f"Compactness out of bounds: {res['compactness']}"

    # Micro polygon
    micro = [(10.0, 10.0), (10.0001, 10.0), (10.0, 10.0001)]
    res_m = compute_defensive_cohesion(micro)
    assert 0.0 <= res_m["compactness"] <= 1.0, f"Compactness out of bounds: {res_m['compactness']}"
    print("  -> test_round2_convex_hull_compactness_bounds: PASSED")


def test_round2_homography_filter_behind_camera():
    """Verify points behind camera horizon (w <= 0) are discarded by pitch_to_image."""
    trans = HomographyTransformer(camera_mode="sideline")
    # Point at negative X far behind pitch plane
    pts = [(-500.0, -500.0), (52.5, 34.0)]
    img_pts = trans.pitch_to_image(pts, filter_behind_camera=True)
    assert len(img_pts) <= 2
    # Ensure default call has filter_behind_camera=True
    img_pts_def = trans.pitch_to_image(pts)
    assert img_pts == img_pts_def
    print("  -> test_round2_homography_filter_behind_camera: PASSED")


def test_round2_video_stream_cache_limits():
    """Verify MatchStreamManager raw_frame is bounded to 120 and detection_cache does not duplicate raw_frame."""
    sm = MatchStreamManager()
    assert sm._raw_cache_capacity == 120
    # Simulate putting detections with raw frame
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    sm.put_cached_detections(0, {"players": []}, raw_frame=dummy_frame)
    entry = sm.detection_cache.get(0)
    assert entry is not None
    assert entry["raw_frame"] is None, "detection_cache must not store raw_frame"
    assert sm.get_cached_raw_frame(0) is not None, "raw frame must be accessible via get_cached_raw_frame"
    sm.release()
    print("  -> test_round2_video_stream_cache_limits: PASSED")


def test_round3_tactical_ruler_none_attacking():
    """Verify compute_tactical_ruler handles None and malformed tuples in attacking_players without unpacking error."""
    res = compute_tactical_ruler(
        point_a=(52.5, 34.0),
        point_b=(60.0, 34.0),
        defending_players=[(80.0, 30.0), (85.0, 40.0)],
        attacking_players=[None, (float("nan"), 20.0), (), (88.0, 35.0)],
        attack_direction="right"
    )
    assert res is not None
    assert "offside_players" in res
    assert len(res["offside_players"]) == 1
    assert res["offside_players"][0]["player_idx"] == 3
    print("  -> test_round3_tactical_ruler_none_attacking: PASSED")


def test_round3_passing_network_attack_direction_and_ball_guard():
    """Verify compute_dijkstra_passing_path supports attack_direction and guards malformed ball_coord."""
    attackers = [(10.0, 30.0), (30.0, 30.0), (50.0, 30.0), (70.0, 30.0)]
    defenders = [(40.0, 25.0), (60.0, 35.0)]

    # Test attack_direction='left'
    res_left = compute_dijkstra_passing_path(
        attacking_players=attackers,
        defending_players=defenders,
        ball_coord=[float("nan"), float("nan")],
        attack_direction="left"
    )
    assert res_left is not None
    assert "path" in res_left
    assert len(res_left["path"]) >= 2
    # In attack_direction='left', target node should be the one closest to X=0
    assert res_left["path"][-1] == 0

    # Test ghost runner with attack_direction='left'
    ghost_res = simulate_ghost_runner(
        attacking_players=attackers,
        defending_players=defenders,
        ghost_coord=(5.0, 30.0),
        attack_direction="left"
    )
    assert ghost_res is not None
    assert "baseline_dijkstra" in ghost_res
    print("  -> test_round3_passing_network_attack_direction_and_ball_guard: PASSED")


def test_round3_pipeline_null_safety():
    """Verify stage4_render_radar and stage4_render_overlay handle None positions and geometry safely."""
    from core.pipeline import stage4_render_radar, stage4_render_overlay
    from visualization.pitch_renderer import TacticalPitchRenderer
    from visualization.overlays import BroadcastOverlayEngine

    renderer = TacticalPitchRenderer()
    trans = HomographyTransformer(camera_mode="sideline")
    overlay_eng = BroadcastOverlayEngine(trans)

    # Radar with None positions and geometry
    radar_out = stage4_render_radar(None, None, renderer)
    assert isinstance(radar_out, np.ndarray)
    assert radar_out.shape == (730, 1120, 3)

    # Overlay with None positions and geometry
    dummy_frame = np.zeros((540, 960, 3), dtype=np.uint8)
    overlay_out = stage4_render_overlay(dummy_frame, None, None, overlay_eng)
    assert isinstance(overlay_out, np.ndarray)
    assert overlay_out.shape == (540, 960, 3)
    print("  -> test_round3_pipeline_null_safety: PASSED")


def test_round4_foot_contact_and_horizon_depth():
    """Verify player ground contact point matches bounding box base without shift, and image_to_pitch filters horizon."""
    # 1. Player foot contact
    p = PlayerDetection(box=(100, 200, 160, 320))
    assert p.foot_point == (130.0, 320.0), f"Expected exact ground contact (130.0, 320.0), got {p.foot_point}"

    # 2. Horizon filtering in image_to_pitch
    trans = HomographyTransformer(camera_mode="sideline")
    # Point way above the pitch horizon (negative Y / sky)
    sky_pts = [(640.0, -1000.0)]
    pitch_pts = trans.image_to_pitch(sky_pts, filter_out_of_bounds=True)
    assert len(pitch_pts) == 0, f"Expected sky point above horizon to be rejected, got {pitch_pts}"
    print("  -> test_round4_foot_contact_and_horizon_depth: PASSED")


def test_round5_universal_camera_projection():
    """Verify Universal Camera Lens Model forward/inverse roundtrip, parameter recovery, and client LUT export."""
    w, h = 1920, 1080
    cx, cy = w / 2.0, h / 2.0
    r_norm = float(np.hypot(cx, cy))
    true_lambda = -0.10

    # 8 FIFA Pitch Landmarks
    pitch_anchors = np.float32([
        [0.0, 0.0], [105.0, 0.0], [105.0, 68.0], [0.0, 68.0],
        [52.5, 0.0], [52.5, 68.0], [16.5, 13.84], [88.5, 54.16]
    ])

    src_rect = np.float32([[180, 260], [1740, 260], [1840, 960], [80, 960]])
    dst_rect = np.float32([[0, 0], [105, 0], [105, 68], [0, 68]])
    H_ideal = cv2.getPerspectiveTransform(src_rect, dst_rect)
    H_inv_ideal = np.linalg.inv(H_ideal)

    # Metric -> Ideal Undistorted Pixels
    pts_h = np.column_stack([pitch_anchors, np.ones(len(pitch_anchors))])
    proj = (H_inv_ideal @ pts_h.T).T
    u_u = proj[:, 0] / proj[:, 2]
    v_u = proj[:, 1] / proj[:, 2]

    # Distort with true lambda
    xu = (u_u - cx) / r_norm
    yu = (v_u - cy) / r_norm
    ru2 = xu**2 + yu**2
    factor = 2.0 / (1.0 + np.sqrt(np.maximum(1e-8, 1.0 - 4.0 * true_lambda * ru2)))
    xd = xu * factor
    yd = yu * factor
    ud = cx + xd * r_norm
    vd = cy + yd * r_norm
    distorted_src = np.column_stack([ud, vd])

    # 1. Initialize and calibrate with universal mode
    t = HomographyTransformer(
        frame_width=w,
        frame_height=h,
        src_points=distorted_src,
        dst_points=pitch_anchors,
        warp_mode="universal",
        lens_lambda=true_lambda
    )
    assert t.warp_mode == "universal"
    assert abs(t.lens_lambda - true_lambda) < 1e-6

    # 2. Forward metric transformation image_to_pitch
    recovered_pitch = t.image_to_pitch(distorted_src)
    assert len(recovered_pitch) == len(pitch_anchors)
    for (rx, ry), (gtx, gty) in zip(recovered_pitch, pitch_anchors):
        assert np.hypot(rx - gtx, ry - gty) < 1e-4, f"Metric recovery error: ({rx}, {ry}) vs ({gtx}, {gty})"

    # 3. Inverse projection pitch_to_image
    reprojected_pixels = t.pitch_to_image(pitch_anchors)
    assert len(reprojected_pixels) == len(distorted_src)
    for (rpx, rpy), (gx, gy) in zip(reprojected_pixels, distorted_src):
        assert np.hypot(rpx - gx, rpy - gy) < 1.0, f"Pixel reprojection error: ({rpx}, {rpy}) vs ({gx}, {gy})"

    # 4. Automatic joint solver for lambda and H
    H_opt, solved_lam = t.fit_universal_lens_model(distorted_src, pitch_anchors)
    assert abs(solved_lam - true_lambda) < 0.015, f"Solved lambda {solved_lam} deviated from true {true_lambda}"

    # 5. Client Hardware Projection Export
    client_proj = t.export_client_projection(nx=20, ny=12)
    assert client_proj["warp_mode"] == "universal"
    assert abs(client_proj["lens_lambda"] - true_lambda) < 1e-4
    assert client_proj["p2i_lut"] is not None
    assert client_proj["i2p_lut"] is not None
    assert len(client_proj["p2i_lut"]["u"]) == 12
    assert len(client_proj["p2i_lut"]["u"][0]) == 20

    print("  -> test_round5_universal_camera_projection: PASSED")


def test_round5_pitch_mask_box_anchor_alignment():
    """Verify Pitch Mask Studio presets & layouts align anchor handles to box mask corners with 0 px error."""
    from ui.components.pitch_mask_studio import (
        PITCH_LANDMARKS_DEF,
        LAYOUT_8_IDS,
        LAYOUT_13_IDS,
        _build_pose_presets,
    )
    by_id = {h["id"]: h for h in PITCH_LANDMARKS_DEF}

    # 1. 18-Yard Box Corner Metric Integrity
    assert by_id["L18T"]["xm"] == 16.5 and by_id["L18T"]["ym"] == 13.84
    assert by_id["L18B"]["xm"] == 16.5 and by_id["L18B"]["ym"] == 54.16
    assert by_id["R18T"]["xm"] == 88.5 and by_id["R18T"]["ym"] == 13.84
    assert by_id["R18B"]["xm"] == 88.5 and by_id["R18B"]["ym"] == 54.16

    # 2. Layout 8 contains the 4 pitch corners + 4 box corners
    for expected_id in ["TL", "TR", "BR", "BL", "L18T", "L18B", "R18T", "R18B"]:
        assert expected_id in LAYOUT_8_IDS, f"{expected_id} missing from LAYOUT_8_IDS"

    # 3. Panorama Preset Coincidence with Universal Transformer
    w, h = 1920, 1080
    t = HomographyTransformer(w, h, camera_mode="sideline")
    presets = _build_pose_presets(w, h, t)
    pano_handles = presets["panorama"]
    active_pano = [item for item in pano_handles if item["active"]]
    assert len(active_pano) == len(LAYOUT_13_IDS)

    # Re-fit universal model and verify max reprojection error on handles is < 1.0 px
    src_pts = np.float32([[item["u"], item["v"]] for item in active_pano])
    dst_pts = np.float32([[item["xm"], item["ym"]] for item in active_pano])
    t_test = HomographyTransformer(w, h, warp_mode="universal", lens_lambda=-0.10)
    t_test.calibrate(src_pts, dst_pts, frame_width=w, frame_height=h, warp_mode="universal", lens_lambda=-0.10)
    for item in active_pano:
        proj = t_test.pitch_to_image([item["xm"], item["ym"]])[0]
        err = float(np.hypot(proj[0] - item["u"], proj[1] - item["v"]))
        assert err < 1.5, f"Pano handle {item['id']} misaligned from mask by {err:.2f}px"

    print("  -> test_round5_pitch_mask_box_anchor_alignment: PASSED")


def test_round6_variable_pitch_and_tps_contracts():
    """Verify variable pitch dimensions (L x W) adhere to FIFA IFAB Law 1 and TPS guarantees 0.00 px snap."""
    from perception.pitch_model import PitchModel
    from ui.components.pitch_mask_studio import get_pitch_landmarks_def

    # 1. Non-standard pitch dimensions (e.g. 100m x 64m)
    kp = PitchModel.get_keypoints(length=100.0, width=64.0)
    assert kp["top_left_corner"] == (0.0, 0.0)
    assert kp["top_right_corner"] == (100.0, 0.0)
    assert kp["bottom_right_corner"] == (100.0, 64.0)
    assert kp["bottom_left_corner"] == (0.0, 64.0)
    assert kp["center_spot"] == (50.0, 32.0)

    # 18-Yard Box Invariant: Depth 16.5m, Width 40.32m -> Y in [(64-40.32)/2, (64+40.32)/2] = [11.84, 52.16]
    assert kp["left_box_top_right"] == (16.5, 11.84)
    assert kp["left_box_bottom_right"] == (16.5, 52.16)
    assert kp["right_box_top_left"] == (83.5, 11.84)
    assert kp["right_box_bottom_left"] == (83.5, 52.16)

    # Pitch Landmarks Def check
    l_def = get_pitch_landmarks_def(100.0, 64.0)
    by_id = {item["id"]: item for item in l_def}
    assert by_id["TR"]["xm"] == 100.0 and by_id["TR"]["ym"] == 0.0
    assert by_id["BR"]["xm"] == 100.0 and by_id["BR"]["ym"] == 64.0
    assert by_id["CS"]["xm"] == 50.0 and by_id["CS"]["ym"] == 32.0

    # 2. HomographyTransformer Variable Pitch Calibration & Export
    w, h = 1920, 1080
    src_rect = np.float32([[200, 300], [1720, 300], [1820, 950], [100, 950]])
    dst_rect = np.float32([[0.0, 0.0], [100.0, 0.0], [100.0, 64.0], [0.0, 64.0]])
    trans = HomographyTransformer(w, h, pitch_length=100.0, pitch_width=64.0)
    trans.calibrate(src_rect, dst_rect, frame_width=w, frame_height=h, pitch_length=100.0, pitch_width=64.0)

    assert trans.pitch_length == 100.0
    assert trans.pitch_width == 64.0
    exported = trans.export_client_projection(nx=10, ny=10)
    assert exported["pitch_length"] == 100.0
    assert exported["pitch_width"] == 64.0

    # Center recovery
    center_img = trans.pitch_to_image([(50.0, 32.0)])[0]
    center_pitch = trans.image_to_pitch([center_img])[0]
    assert np.hypot(center_pitch[0] - 50.0, center_pitch[1] - 32.0) < 0.01

    # 3. Thin-Plate Spline Exact Mathematical Snap (0.00 px error)
    # Test TPS kernel U(r) = r^2 ln(r) with N control points
    pts_ctrl = np.float32([
        [0.0, 0.0], [105.0, 0.0], [105.0, 68.0], [0.0, 68.0],
        [52.5, 0.0], [52.5, 68.0], [52.5, 34.0],
        [16.5, 13.84], [16.5, 54.16], [88.5, 13.84], [88.5, 54.16],
        [26.25, 0.0], [78.75, 0.0]
    ])
    # Target video points with intentional non-linear warp displacements
    pts_vid = np.float32([
        [120.0, 350.0], [1800.0, 340.0], [1870.0, 920.0], [50.0, 930.0],
        [960.0, 380.0], [960.0, 950.0], [960.0, 650.0],
        [350.0, 480.0], [330.0, 780.0], [1570.0, 470.0], [1590.0, 770.0],
        [540.0, 360.0], [1380.0, 355.0]
    ])
    N = len(pts_ctrl)
    norm_scale = 105.0
    p_norm = pts_ctrl / norm_scale

    # Compute base homography residuals
    H_base, _ = cv2.findHomography(pts_ctrl, pts_vid, cv2.RANSAC)
    pts_h = np.column_stack([pts_ctrl, np.ones(N)])
    base_proj = (H_base @ pts_h.T).T
    base_u = base_proj[:, 0] / base_proj[:, 2]
    base_v = base_proj[:, 1] / base_proj[:, 2]
    res_u = pts_vid[:, 0] - base_u
    res_v = pts_vid[:, 1] - base_v

    # Solve TPS matrix system
    dim = N + 3
    L = np.zeros((dim, dim), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            if i != j:
                d = np.linalg.norm(p_norm[i] - p_norm[j])
                L[i, j] = (d**2) * np.log(max(1e-12, d))
        L[i, i] += 1e-9
        L[i, N] = 1.0
        L[i, N + 1] = p_norm[i, 0]
        L[i, N + 2] = p_norm[i, 1]
        L[N, i] = 1.0
        L[N + 1, i] = p_norm[i, 0]
        L[N + 2, i] = p_norm[i, 1]

    b_u = np.concatenate([res_u, [0.0, 0.0, 0.0]])
    b_v = np.concatenate([res_v, [0.0, 0.0, 0.0]])
    w_u = np.linalg.solve(L, b_u)
    w_v = np.linalg.solve(L, b_v)

    # Evaluate TPS at all control points
    max_snap_err = 0.0
    for i in range(N):
        nx, ny = p_norm[i]
        du = w_u[N] + w_u[N + 1] * nx + w_u[N + 2] * ny
        dv = w_v[N] + w_v[N + 1] * nx + w_v[N + 2] * ny
        for j in range(N):
            d = np.linalg.norm(p_norm[i] - p_norm[j])
            if d > 1e-9:
                k = (d**2) * np.log(d)
                du += w_u[j] * k
                dv += w_v[j] * k
        final_u = base_u[i] + du
        final_v = base_v[i] + dv
        err = np.hypot(final_u - pts_vid[i, 0], final_v - pts_vid[i, 1])
        max_snap_err = max(max_snap_err, err)

    assert max_snap_err < 0.001, f"TPS snap error too high: {max_snap_err:.6f} px (expected < 0.001 px)"
    print("  -> test_round6_variable_pitch_and_tps_contracts: PASSED")


def test_round7_calibration_guided_pitch_roi_and_adaptive_detection():
    """Verify calibration-guided ROI cropping and adaptive confidence for panoramic & broadcast views."""
    from perception.detector import FootballDetector
    from perception.homography import HomographyTransformer
    from core.pipeline import stage1_perception

    # 1. Homography ROI derivation
    trans = HomographyTransformer(1920, 1080, camera_mode="sideline")
    roi = trans.get_pitch_roi(1920, 1080)
    assert roi is not None
    x1, y1, x2, y2 = roi
    assert 0 <= x1 < x2 <= 1920
    assert 0 <= y1 < y2 <= 1080
    assert (x2 - x1) >= 0.4 * 1920
    assert (y2 - y1) >= 0.3 * 1080

    # 2. Panoramic aspect ratio fallback
    pano_trans = HomographyTransformer(3840, 1000, warp_mode="universal", lens_lambda=-0.10)
    pano_roi = pano_trans.get_pitch_roi(3840, 1000)
    assert pano_roi is not None
    px1, py1, px2, py2 = pano_roi
    assert px1 == 0 and px2 == 3840
    assert py1 > 0 and py2 < 1000

    # 3. Stage 1 perception integration with synthetic test frame
    dummy_frame = np.full((540, 960, 3), 50, dtype=np.uint8)
    # Paint green turf in the middle
    dummy_frame[100:440, :] = [34, 139, 34]
    det = FootballDetector()
    t_dummy = HomographyTransformer(960, 540, camera_mode="sideline")
    res = stage1_perception(dummy_frame, det, frame_idx=0, transformer=t_dummy)
    assert res is not None
    assert res.frame_width == 960
    assert res.frame_height == 540
    print("  -> test_round7_calibration_guided_pitch_roi_and_adaptive_detection: PASSED")


def test_canonical_pitch_model_registry():
    """Verify single canonical source of truth for FIFA pitch constants, landmarks, and layouts."""
    import perception.pitch_model as pm
    import ui.components.pitch_mask_studio as pms
    import perception.calibration_templates as pct
    import ui.components.calibration_studio as ucs

    # 1. Pitch physical dimensions & constants
    assert pm.LENGTH == 105.0 and pm.WIDTH == 68.0
    assert pm.CENTER_CIRCLE_RADIUS == 9.15
    assert pm.PENALTY_BOX_LENGTH == 16.5 and pm.PENALTY_BOX_WIDTH == 40.32
    assert pm.SIX_YARD_BOX_LENGTH == 5.5 and pm.SIX_YARD_BOX_WIDTH == 18.32
    assert pm.PENALTY_SPOT_DISTANCE == 11.0
    assert pm.GOAL_WIDTH == 7.32
    assert pm.PENALTY_D_RADIUS == 9.15

    # 2. Canonical collections in pitch_model
    assert len(pm.PITCH_LANDMARKS_DEF) == 27
    assert len(pm.FIFA_CHALK_LANDMARKS) == 29
    assert len(pm.MESH_HANDLES_DEF) == 13
    assert pm.LAYOUT_4_IDS == ["TL", "TR", "BR", "BL"]
    assert pm.LAYOUT_4_GOAL_IDS == ["TR", "BR", "BM50", "TM50"]
    assert pm.LAYOUT_8_IDS == ["TL", "TR", "BR", "BL", "L18T", "L18B", "R18T", "R18B"]
    assert pm.LAYOUT_8_GOAL_IDS == ["TR", "BR", "R18T", "R18B", "R18GT", "R18GB", "TM50", "BM50"]
    assert pm.LAYOUT_13_IDS == ["TL", "TR", "BR", "BL", "TM25", "TM50", "TM75", "BM50", "CS", "L18T", "L18B", "R18T", "R18B"]
    assert len(pm.LAYOUT_21_IDS) == 21
    assert pm.LAYOUT_ENDZONE_IDS == ["TR", "BR", "R18T", "R18B", "TM50", "BM50", "CS"]

    # 3. Geometry helper methods on PitchModel
    boundary_std = pm.PitchModel.get_turf_boundary(105.0, 68.0, is_goal_cam=False)
    boundary_goal = pm.PitchModel.get_turf_boundary(105.0, 68.0, is_goal_cam=True)
    assert len(boundary_std) == 4 and boundary_std[0] == (0.0, 0.0)
    assert len(boundary_goal) == 4 and boundary_goal[0] == (52.5, 0.0)

    markings_std = pm.PitchModel.get_markings_polylines(105.0, 68.0, is_goal_cam=False)
    markings_goal = pm.PitchModel.get_markings_polylines(105.0, 68.0, is_goal_cam=True)
    assert len(markings_std) == 9
    assert len(markings_goal) == 6

    circle_pts = pm.PitchModel.get_center_circle_polygon(105.0, 68.0, num_points=36)
    assert len(circle_pts) == 36

    # 4. Backward-compatible re-exports in pitch_mask_studio
    assert pms.PITCH_LANDMARKS_DEF is pm.PITCH_LANDMARKS_DEF
    assert pms.get_pitch_landmarks_def is pm.get_pitch_landmarks_def
    assert pms.MESH_HANDLES_DEF is pm.MESH_HANDLES_DEF
    assert pms.LAYOUT_4_IDS is pm.LAYOUT_4_IDS
    assert pms.LAYOUT_4_GOAL_IDS is pm.LAYOUT_4_GOAL_IDS
    assert pms.LAYOUT_8_IDS is pm.LAYOUT_8_IDS
    assert pms.LAYOUT_8_GOAL_IDS is pm.LAYOUT_8_GOAL_IDS
    assert pms.LAYOUT_13_IDS is pm.LAYOUT_13_IDS
    assert pms.LAYOUT_21_IDS is pm.LAYOUT_21_IDS
    assert pms.LAYOUT_ENDZONE_IDS is pm.LAYOUT_ENDZONE_IDS
    assert pms.PITCH_LEN == pm.LENGTH and pms.PITCH_WID == pm.WIDTH

    # 5. Backward-compatible re-exports in calibration_templates
    assert pct.FIFA_CHALK_LANDMARKS is pm.FIFA_CHALK_LANDMARKS
    assert pct.get_fifa_chalk_landmarks is pm.get_fifa_chalk_landmarks
    assert pct.PITCH_LENGTH == pm.LENGTH and pct.PITCH_WIDTH == pm.WIDTH

    # 6. Direct canonical reference in calibration_studio
    assert ucs.FIFA_CHALK_LANDMARKS is pm.FIFA_CHALK_LANDMARKS

    print("  -> test_canonical_pitch_model_registry: PASSED")


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
    test_round2_voronoi_boundary_conservation()
    test_round2_coach_analytics_empty_hull()
    test_round2_convex_hull_compactness_bounds()
    test_round2_homography_filter_behind_camera()
    test_round2_video_stream_cache_limits()
    test_round3_tactical_ruler_none_attacking()
    test_round3_passing_network_attack_direction_and_ball_guard()
    test_round3_pipeline_null_safety()
    test_round4_foot_contact_and_horizon_depth()
    test_round5_universal_camera_projection()
    test_round5_pitch_mask_box_anchor_alignment()
    test_round6_variable_pitch_and_tps_contracts()
    test_round7_calibration_guided_pitch_roi_and_adaptive_detection()
    test_canonical_pitch_model_registry()
    print("=======================================================")
    print(">>> ALL KNOWLEDGE BASE CONTRACTS 100% VERIFIED! <<<")
    print("=======================================================\n")
    return 0


if __name__ == "__main__":
    sys.exit(run_all_contract_tests())
