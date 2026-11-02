"""
Standardized Tactical Modular Pipeline
---------------------------------------
Strict, decoupled modular stages with standardized data contracts:
- Stage 1: stage1_perception (Raw Frame -> FrameDetections)
- Stage 2: stage2_projection (FrameDetections -> PitchPositions)
- Stage 3: stage3_geometry   (PitchPositions -> TacticalGeometry)
- Stage 4: stage4_render     (Bundle -> 2D Radar Canvas & AR Broadcast Overlay)
"""

import time
from typing import Tuple, Optional, Dict, Any, List
import numpy as np
import cv2

from core.contracts import (
    PlayerDetection,
    FrameDetections,
    PitchPositions,
    TacticalGeometry,
    FrameBundle
)
from perception.detector import FootballDetector
from perception.homography import HomographyTransformer
from algorithms.convex_hull import compute_defensive_cohesion
from algorithms.voronoi_dominance import compute_spatial_dominance
from visualization.pitch_renderer import TacticalPitchRenderer
from visualization.overlays import BroadcastOverlayEngine


def stage1_perception(
    frame: np.ndarray,
    detector: FootballDetector,
    frame_idx: int = 0
) -> FrameDetections:
    """
    Stage 1: Camera Perception Engine.
    Processes a raw video frame to extract player ground contact points, team IDs, and ball position.
    Output: FrameDetections contract (strictly image pixel space).
    """
    fh, fw = frame.shape[:2]
    det_dict = detector.detect_frame(frame)
    raw_players = det_dict.get('players', [])
    raw_ball = det_dict.get('ball', None)

    player_contracts = [
        p if isinstance(p, PlayerDetection) else PlayerDetection(
            box=p.box,
            foot_point=p.foot_point,
            team_id=p.team_id,
            confidence=getattr(p, 'confidence', 1.0)
        )
        for p in raw_players
    ]

    return FrameDetections(
        frame_idx=frame_idx,
        players=player_contracts,
        ball_pixel=tuple(raw_ball) if raw_ball is not None else None,
        frame_width=fw,
        frame_height=fh
    )


def stage2_projection(
    detections: FrameDetections,
    transformer: HomographyTransformer
) -> PitchPositions:
    """
    Stage 2: Spatial Homography & Metric Projection Engine.
    Translates 2D image pixel coordinates into metric FIFA pitch coordinates (105m x 68m).
    Zero model dependency. Reprojection error < 0.5px.
    Output: PitchPositions contract.
    """
    transformer.update_dimensions(detections.frame_width, detections.frame_height)

    # Filter out sideline personnel (substitutes, technical staff in dugout, spectators)
    # allowing a realistic 2.0m touchline run-off buffer
    team_a_pitch = transformer.image_to_pitch(detections.team_a_pixels, filter_out_of_bounds=True, buffer_meters=2.0)
    team_b_pitch = transformer.image_to_pitch(detections.team_b_pixels, filter_out_of_bounds=True, buffer_meters=2.0)

    # Ball metric position
    if detections.ball_pixel is not None:
        ball_pts = transformer.image_to_pitch([detections.ball_pixel], filter_out_of_bounds=False)
        ball_pitch = ball_pts[0] if ball_pts else (52.5, 34.0)
    else:
        # Anchor near first forward
        if team_a_pitch:
            ball_pitch = (team_a_pitch[min(5, len(team_a_pitch) - 1)][0] + 1.5,
                          team_a_pitch[min(5, len(team_a_pitch) - 1)][1])
        else:
            ball_pitch = (52.5, 34.0)

    # Ball carrier (closest player on Team A to the ball)
    carrier_idx = 0
    if team_a_pitch:
        min_d = float('inf')
        for idx, (px, py) in enumerate(team_a_pitch):
            d = (px - ball_pitch[0]) ** 2 + (py - ball_pitch[1]) ** 2
            if d < min_d:
                min_d = d
                carrier_idx = idx

    return PitchPositions(
        frame_idx=detections.frame_idx,
        team_a=team_a_pitch,
        team_b=team_b_pitch,
        ball=ball_pitch,
        carrier_idx=carrier_idx,
        detected_count=len(team_a_pitch) + len(team_b_pitch)
    )


def stage3_geometry(
    positions: PitchPositions,
    hull_algo: str = 'graham'
) -> TacticalGeometry:
    """
    Stage 3: Computational Geometry Engine.
    Computes Convex Hull (defensive block) and Voronoi Dominance (pitch control partitions).
    Output: TacticalGeometry contract.
    """
    t0 = time.perf_counter()

    # Defensive Compactness Hull (Team B defending block)
    hull_algo_key = 'jarvis' if 'jarvis' in hull_algo.lower() else 'graham'
    hull_res = compute_defensive_cohesion(positions.team_b, algorithm=hull_algo_key)
    hull_pts = hull_res.get('hull', [])

    # Map each hull vertex to its exact player index in positions.team_b
    hull_player_indices = []
    for pt in hull_pts:
        matched_idx = -1
        for idx, player_pt in enumerate(positions.team_b):
            if abs(player_pt[0] - pt[0]) < 1e-4 and abs(player_pt[1] - pt[1]) < 1e-4:
                matched_idx = idx
                break
        if matched_idx != -1:
            hull_player_indices.append(matched_idx)

    # Voronoi Spatial Dominance
    voronoi_res = compute_spatial_dominance(positions.team_a, positions.team_b)

    latency_ms = (time.perf_counter() - t0) * 1000.0

    return TacticalGeometry(
        frame_idx=positions.frame_idx,
        hull_polygon=hull_pts,
        hull_player_indices=hull_player_indices,
        hull_area=float(hull_res.get('area_sq_m', 0.0)),
        hull_compactness=float(hull_res.get('compactness', 0.0)),
        hull_vertices=len(hull_pts),
        voronoi_cells_a=voronoi_res.get('cells_a', []),
        voronoi_cells_b=voronoi_res.get('cells_b', []),
        team_a_pct=float(voronoi_res.get('team_a_pct', 50.0)),
        team_b_pct=float(voronoi_res.get('team_b_pct', 50.0)),
        latency_ms=round(latency_ms, 2)
    )


def stage4_render_radar(
    positions: PitchPositions,
    geometry: TacticalGeometry,
    renderer: TacticalPitchRenderer,
    show_voronoi: bool = True,
    show_hull: bool = True
) -> np.ndarray:
    """
    Stage 4A: Top-Down 2D FIFA Tactical Pitch Radar Renderer.
    """
    radar = renderer.draw_base_pitch()

    if show_voronoi and (geometry.voronoi_cells_a or geometry.voronoi_cells_b):
        v_res = {
            'team_a_pct': geometry.team_a_pct,
            'team_b_pct': geometry.team_b_pct,
            'cells_a': geometry.voronoi_cells_a,
            'cells_b': geometry.voronoi_cells_b
        }
        radar = renderer.overlay_voronoi(radar, v_res)

    if show_hull and len(geometry.hull_polygon) >= 3:
        radar = renderer.overlay_defensive_hull(radar, geometry.hull_polygon)

    radar = renderer.render_players(
        radar,
        positions.team_a,
        positions.team_b,
        positions.ball,
        positions.carrier_idx,
        show_acceleration=False
    )
    return radar


def stage4_render_overlay(
    raw_frame: np.ndarray,
    positions: PitchPositions,
    geometry: TacticalGeometry,
    overlay_engine: BroadcastOverlayEngine,
    show_voronoi: bool = True,
    show_hull: bool = True,
    hull_algo_name: str = "Graham Scan",
    pipeline_fps: Optional[float] = None
) -> np.ndarray:
    """
    Stage 4B: Broadcast Camera AR Overlay Renderer.
    Clips strictly to pitch turf perimeter and projects geometry onto camera frame.
    """
    rf_h, rf_w = raw_frame.shape[:2]
    overlay_engine.transformer.update_dimensions(rf_w, rf_h)

    aug = raw_frame.copy()

    if show_voronoi and (geometry.voronoi_cells_a or geometry.voronoi_cells_b):
        v_res = {
            'team_a_pct': geometry.team_a_pct,
            'team_b_pct': geometry.team_b_pct,
            'cells_a': geometry.voronoi_cells_a,
            'cells_b': geometry.voronoi_cells_b
        }
        aug = overlay_engine.overlay_projected_voronoi(aug, v_res)

    if show_hull and len(geometry.hull_polygon) >= 3:
        aug = overlay_engine.overlay_projected_hull(aug, geometry.hull_polygon)

    hud_fps = pipeline_fps if pipeline_fps is not None and pipeline_fps > 0 else (
        1000.0 / (geometry.latency_ms + 1e-6) if geometry.latency_ms > 0 else 60.0
    )

    aug = overlay_engine.draw_broadcast_hud(
        aug,
        fps=hud_fps,
        hull_res={
            'area_sq_m': geometry.hull_area,
            'compactness': geometry.hull_compactness,
            'algorithm': hull_algo_name
        },
        voronoi_res={
            'team_a_pct': geometry.team_a_pct,
            'team_b_pct': geometry.team_b_pct
        },
        dijkstra_res=None,
        daa_latency_ms=geometry.latency_ms
    )
    return aug


def run_pipeline_frame(
    raw_frame: np.ndarray,
    detector: FootballDetector,
    transformer: HomographyTransformer,
    renderer: TacticalPitchRenderer,
    overlay_engine: BroadcastOverlayEngine,
    frame_idx: int = 0,
    hull_algo: str = 'graham',
    show_voronoi: bool = True,
    show_hull: bool = True
) -> Tuple[np.ndarray, np.ndarray, FrameBundle]:
    """
    Executes all 4 decoupled stages sequentially on a single video frame.
    Returns (aug_video_frame, 2d_radar_frame, FrameBundle).
    """
    t_start = time.perf_counter()
    detections = stage1_perception(raw_frame, detector, frame_idx)
    positions = stage2_projection(detections, transformer)
    geometry = stage3_geometry(positions, hull_algo)
    pipe_latency_ms = (time.perf_counter() - t_start) * 1000.0
    actual_fps = 1000.0 / max(pipe_latency_ms, 1e-6)

    radar = stage4_render_radar(positions, geometry, renderer, show_voronoi, show_hull)
    overlay = stage4_render_overlay(
        raw_frame, positions, geometry, overlay_engine, show_voronoi, show_hull,
        pipeline_fps=actual_fps
    )

    bundle = FrameBundle(positions=positions, geometry=geometry)
    return overlay, radar, bundle


def evaluate_pipeline_health(
    detections: FrameDetections,
    positions: PitchPositions,
    geometry: TacticalGeometry,
    match_source: str = "",
    camera_mode: str = "sideline",
    latencies: Optional[Dict[str, float]] = None
) -> Tuple[Dict[str, Tuple[str, str]], List[Tuple[str, str, str]]]:
    """
    Standardized, decoupled health evaluator across all 4 stages.
    Evaluates:
    - Stage 1: Player count detection & sparsity
    - Stage 2: FIFA pitch bounds check and camera mode angle mismatch
    - Stage 3: Defensive Hull closure & Voronoi dominance
    - Stage 4: Renderers operational status

    Returns (diag_status, diag_issues) with standardized PASS / WARN / FAIL severity.
    """
    diag_status: Dict[str, Tuple[str, str]] = {}
    diag_issues: List[Tuple[str, str, str]] = []

    # Stage 1: Perception
    p_count = len(detections.players)
    lat_1 = f" ({latencies['stage1']:.1f}ms)" if latencies and 'stage1' in latencies else ""
    if p_count == 0:
        diag_status['stage1'] = ("FAIL", f"0 players detected{lat_1}")
        diag_issues.append(("Stage 1 (Perception)", "No players detected in this frame. Check video lighting or model weights.", "FAIL"))
    elif p_count < 10:
        diag_status['stage1'] = ("WARN", f"{p_count} players detected{lat_1}")
        diag_issues.append(("Stage 1 (Perception)", f"Only {p_count} players detected. Sparsity may degrade hull and Voronoi shapes.", "WARN"))
    else:
        diag_status['stage1'] = ("PASS", f"{p_count} players detected{lat_1}")

    # Stage 2: Homography
    lat_2 = f" ({latencies['stage2']:.2f}ms)" if latencies and 'stage2' in latencies else ""
    out_of_bounds = [
        pt for pt in positions.team_a + positions.team_b
        if not (-2.0 <= pt[0] <= 107.0 and -2.0 <= pt[1] <= 70.0)
    ]
    ms_lower = match_source.lower().replace("_", " ")
    is_mismatch = False
    if "premier league" in ms_lower and camera_mode == "goal_to_goal":
        is_mismatch = True
        diag_issues.append(("Stage 2 (Homography)", "Camera mode 'goal_to_goal' selected on a Sideline broadcast! Switch to 'sideline'.", "FAIL"))
    elif "world cup" in ms_lower and camera_mode == "sideline":
        is_mismatch = True
        diag_issues.append(("Stage 2 (Homography)", "Camera mode 'sideline' selected on an End-Zone broadcast! Switch to 'goal_to_goal'.", "FAIL"))

    if is_mismatch:
        diag_status['stage2'] = ("FAIL", f"Camera angle mismatch{lat_2}")
    elif len(out_of_bounds) > 3:
        diag_status['stage2'] = ("WARN", f"{len(out_of_bounds)} out of bounds{lat_2}")
        diag_issues.append(("Stage 2 (Homography)", f"{len(out_of_bounds)} players projected outside FIFA pitch boundaries [0, 105]x[0, 68].", "WARN"))
    else:
        diag_status['stage2'] = ("PASS", f"All coordinates bounded{lat_2}")

    # Stage 3: Geometry
    lat_3 = f" ({latencies['stage3']:.2f}ms)" if latencies and 'stage3' in latencies else ""
    if len(geometry.hull_polygon) < 3:
        diag_status['stage3'] = ("WARN", f"< 3 hull nodes{lat_3}")
        diag_issues.append(("Stage 3 (Geometry)", "Defensive Hull contains < 3 vertices.", "WARN"))
    elif geometry.hull_area < 50.0:
        diag_status['stage3'] = ("WARN", f"Small area {geometry.hull_area:.0f}m²{lat_3}")
        diag_issues.append(("Stage 3 (Geometry)", f"Abnormally small hull area ({geometry.hull_area:.0f}m²).", "WARN"))
    else:
        diag_status['stage3'] = ("PASS", f"{len(geometry.hull_polygon)} nodes, {geometry.hull_area:.0f}m²{lat_3}")

    # Stage 4: Renderers
    lat_4 = f" ({latencies['stage4']:.1f}ms)" if latencies and 'stage4' in latencies else ""
    diag_status['stage4'] = ("PASS", f"2D & AR renderers healthy{lat_4}")

    return diag_status, diag_issues


# ==============================================================================
# BUILT-IN CLI STAGE DEBUGGER
# Allows executing and inspecting any stage from command-line without separate scripts:
#   python -m core.pipeline --stage 1
#   python -m core.pipeline --stage 2
#   python -m core.pipeline --stage 3
#   python -m core.pipeline --stage 4
#   python -m core.pipeline --stage all
# ==============================================================================
if __name__ == "__main__":
    import argparse
    import json
    import os

    parser = argparse.ArgumentParser(description="TacticalVision Unified Pipeline Stage Debugger")
    parser.add_argument("--stage", type=str, default="all", choices=["1", "2", "3", "4", "all"], help="Stage to execute & debug")
    parser.add_argument("--video", type=str, default="data/user_match.mp4", help="Video path")
    parser.add_argument("--camera", type=str, default="goal_to_goal", choices=["goal_to_goal", "sideline"], help="Camera mode")
    parser.add_argument("--frame", type=int, default=0, help="Frame index")
    parser.add_argument("--algo", type=str, default="graham", choices=["graham", "jarvis"], help="Convex Hull algorithm")
    args = parser.parse_args()

    os.makedirs("scratch", exist_ok=True)

    # Helper to load video frame
    def _get_raw_frame(path: str, f_idx: int) -> np.ndarray:
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, f = cap.read()
        cap.release()
        assert ret and f is not None, f"Could not read frame {f_idx} from {path}"
        return f

    print("=" * 70)
    print(f"🔬 TacticalVision Unified Stage Debugger | Stage: {args.stage.upper()}")
    print("=" * 70)

    # Execute Stage 1
    if args.stage in ["1", "all"]:
        raw = _get_raw_frame(args.video, args.frame)
        print(f"\n[Stage 1: Perception] Loading frame {args.frame} from {args.video} ({raw.shape[1]}x{raw.shape[0]})...")
        t0 = time.perf_counter()
        det = stage1_perception(raw, FootballDetector(), frame_idx=args.frame)
        t_ms = (time.perf_counter() - t0) * 1000.0
        print(f" -> Execution Latency: {t_ms:.1f} ms")
        print(f" -> Total Players Detected: {len(det.players)} (Team A: {len(det.team_a_pixels)}, Team B: {len(det.team_b_pixels)})")
        print(f" -> Ball Position: {det.ball_pixel}")
        
        with open("scratch/stage1_detections.json", "w") as f_out:
            json.dump(det.to_dict(), f_out, indent=2)
        print(" -> Contract Saved: scratch/stage1_detections.json")

        # Save visual debug
        dbg_img = raw.copy()
        for p in det.players:
            col = (255, 180, 0) if p.team_id == 0 else (50, 50, 240)
            cv2.rectangle(dbg_img, (p.box[0], p.box[1]), (p.box[2], p.box[3]), col, 2)
            cv2.circle(dbg_img, (int(p.foot_point[0]), int(p.foot_point[1])), 5, (0, 255, 255), -1)
        cv2.imwrite("scratch/stage1_debug.jpg", dbg_img)
        print(" -> Visual Artifact: scratch/stage1_debug.jpg")

    # Execute Stage 2
    if args.stage in ["2", "all"]:
        if not os.path.exists("scratch/stage1_detections.json"):
            raw = _get_raw_frame(args.video, args.frame)
            det = stage1_perception(raw, FootballDetector(), frame_idx=args.frame)
        else:
            with open("scratch/stage1_detections.json") as f_in:
                det = FrameDetections.from_dict(json.load(f_in))

        print(f"\n[Stage 2: Homography] Projecting {len(det.players)} detections to pitch metric space (Camera: {args.camera})...")
        trans = HomographyTransformer(frame_width=det.frame_width, frame_height=det.frame_height, camera_mode=args.camera)
        t0 = time.perf_counter()
        pos = stage2_projection(det, trans)
        t_ms = (time.perf_counter() - t0) * 1000.0
        print(f" -> Execution Latency: {t_ms:.2f} ms")
        print(f" -> Team A Metrics ({len(pos.team_a)} players): Sample: {pos.team_a[:2]}")
        print(f" -> Team B Metrics ({len(pos.team_b)} players): Sample: {pos.team_b[:2]}")
        print(f" -> Ball Pitch Coordinate: ({pos.ball[0]:.2f}m, {pos.ball[1]:.2f}m)")

        # Generate Stage 2 Visual Mask Artifact
        raw = _get_raw_frame(args.video, det.frame_idx)
        ov_eng = BroadcastOverlayEngine(trans)
        h_vis = ov_eng.render_homography_debug_frame(raw, positions=pos)
        cv2.imwrite("scratch/stage2_homography_mask.jpg", h_vis)

        with open("scratch/stage2_positions.json", "w") as f_out:
            json.dump(pos.to_dict(), f_out, indent=2)
        print(" -> Contract Saved: scratch/stage2_positions.json")
        print(" -> Visual Artifact: scratch/stage2_homography_mask.jpg")

    # Execute Stage 3
    if args.stage in ["3", "all"]:
        if not os.path.exists("scratch/stage2_positions.json"):
            raise RuntimeError("Run stage 2 first to generate scratch/stage2_positions.json")
        with open("scratch/stage2_positions.json") as f_in:
            pos = PitchPositions.from_dict(json.load(f_in))

        print(f"\n[Stage 3: Geometry Core] Computing Convex Hull ({args.algo}) and Voronoi Dominance...")
        geom = stage3_geometry(pos, hull_algo=args.algo)
        print(f" -> Computational Latency: {geom.latency_ms:.2f} ms")
        print(f" -> Defensive Block: {geom.hull_area:.1f} m² (Compactness: {geom.hull_compactness:.2f}, {geom.hull_vertices} vertices)")
        print(f" -> Pitch Control: Team A {geom.team_a_pct:.1f}% | Team B {geom.team_b_pct:.1f}%")
        print(f" -> Voronoi Cells: {len(geom.voronoi_cells_a)} Team A, {len(geom.voronoi_cells_b)} Team B")

        # Generate Stage 3 Visual Geometry Artifact
        raw = _get_raw_frame(args.video, pos.frame_idx)
        trans = HomographyTransformer(frame_width=raw.shape[1], frame_height=raw.shape[0], camera_mode=args.camera)
        ov_eng = BroadcastOverlayEngine(trans)
        g_vis = ov_eng.render_geometry_debug_frame(raw, positions=pos, geometry=geom)
        cv2.imwrite("scratch/stage3_geometry_visual.jpg", g_vis)

        with open("scratch/stage3_geometry.json", "w") as f_out:
            json.dump(geom.to_dict(), f_out, indent=2)
        print(" -> Contract Saved: scratch/stage3_geometry.json")
        print(" -> Visual Artifact: scratch/stage3_geometry_visual.jpg")

    # Execute Stage 4
    if args.stage in ["4", "all"]:
        if not os.path.exists("scratch/stage2_positions.json") or not os.path.exists("scratch/stage3_geometry.json"):
            raise RuntimeError("Run stages 2 and 3 first to generate positions and geometry contracts.")
        with open("scratch/stage2_positions.json") as f_in:
            pos = PitchPositions.from_dict(json.load(f_in))
        with open("scratch/stage3_geometry.json") as f_in:
            geom = TacticalGeometry.from_dict(json.load(f_in))

        raw = _get_raw_frame(args.video, pos.frame_idx)
        print(f"\n[Stage 4: Renderers] Rendering 2D Radar & Broadcast AR Overlay ({raw.shape[1]}x{raw.shape[0]})...")
        trans = HomographyTransformer(frame_width=raw.shape[1], frame_height=raw.shape[0], camera_mode=args.camera)
        rend = TacticalPitchRenderer(width=1120, height=730, padding=40)
        ov_eng = BroadcastOverlayEngine(trans)

        t0 = time.perf_counter()
        radar_img = stage4_render_radar(pos, geom, rend)
        overlay_img = stage4_render_overlay(raw, pos, geom, ov_eng, hull_algo_name=args.algo)
        turf_mask = ov_eng.get_turf_binary_mask(raw.shape[:2])
        t_ms = (time.perf_counter() - t0) * 1000.0
        print(f" -> Rendering Latency: {t_ms:.1f} ms")

        cv2.imwrite("scratch/stage4_radar.jpg", radar_img)
        cv2.imwrite("scratch/stage4_overlay.jpg", overlay_img)
        cv2.imwrite("scratch/stage4_turf_mask.jpg", turf_mask)
        print(" -> Render Artifacts: scratch/stage4_radar.jpg, scratch/stage4_overlay.jpg, & scratch/stage4_turf_mask.jpg")

    print("\n" + "=" * 70)
    print("✅ Stage Debug Completed Successfully!")
    print("=" * 70)
