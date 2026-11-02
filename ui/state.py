"""
TacticalVision State & Stream Synchronization Layer
===================================================
Manages centralized session state, video streams, precomputed cache synchronization,
and lazy pipeline bundle execution with zero duplicate keys or redundant reruns.
"""

import os
import time
from typing import Dict, Any, Optional
import cv2
import numpy as np
import streamlit as st

from core.contracts import (
    PlayerDetection,
    FrameDetections,
    PitchPositions,
    TacticalGeometry
)
from core.pipeline import (
    stage1_perception,
    stage2_projection,
    stage3_geometry,
    evaluate_pipeline_health
)
from perception.detector import FootballDetector
from perception.homography import HomographyTransformer
from perception.pitch_model import PitchModel
from perception.video_stream import MatchStreamManager
from perception.precompute import load_precomputed_segment
from perception.calibration_templates import CALIBRATION_TEMPLATES
from visualization.pitch_renderer import TacticalPitchRenderer
from visualization.overlays import BroadcastOverlayEngine
from algorithms.coach_analytics import evaluate_coach_analytics
from algorithms.passing_network import compute_dijkstra_passing_path


VIDEO_SOURCES: Dict[str, Dict[str, str]] = {
    "premier_league": {
        "title": "Premier League Broadcast (Sample 30s)",
        "path": "data/sample_broadcast.mp4",
        "cache": "data/precomputed_broadcast.json",
        "default_mode": "sideline"
    },
    "world_cup": {
        "title": "World Cup Tactical Cam (Goal-to-Goal 48s)",
        "path": "data/user_match.mp4",
        "cache": "data/precomputed_user.json",
        "default_mode": "goal_to_goal"
    },
    "custom_upload": {
        "title": "Custom Match Video (Upload MP4)",
        "path": "data/user_match.mp4",
        "cache": "data/precomputed_user.json",
        "default_mode": "goal_to_goal"
    }
}


def init_session_state() -> None:
    """Initializes all necessary session state variables idempotently."""
    if 'detector' not in st.session_state:
        cpu_default = min(8, max(2, (os.cpu_count() or 4) // 2))
        st.session_state.detector = FootballDetector(num_threads=cpu_default)

    if 'transformer' not in st.session_state:
        st.session_state.transformer = HomographyTransformer(camera_mode="sideline")

    if 'renderer' not in st.session_state:
        st.session_state.renderer = TacticalPitchRenderer(width=1000, height=650, padding=35)

    if 'overlay_engine' not in st.session_state:
        st.session_state.overlay_engine = BroadcastOverlayEngine(st.session_state.transformer)

    if 'stream_manager' not in st.session_state:
        st.session_state.stream_manager = MatchStreamManager(detector=st.session_state.detector)

    if 'current_frame' not in st.session_state:
        st.session_state.current_frame = 0

    if 'manual_calib_enabled' not in st.session_state:
        st.session_state.manual_calib_enabled = False
    if 'manual_calib_points' not in st.session_state:
        st.session_state.manual_calib_points = []
    if 'manual_calib_applied' not in st.session_state:
        st.session_state.manual_calib_applied = False
    if 'manual_dst_template' not in st.session_state:
        st.session_state.manual_dst_template = list(CALIBRATION_TEMPLATES.keys())[0]

    if 'precomputed_data' not in st.session_state:
        st.session_state.precomputed_data = None

    if 'active_source_key' not in st.session_state:
        st.session_state.active_source_key = "premier_league"

    if 'setup_confirmed' not in st.session_state:
        st.session_state.setup_confirmed = False

    if 'exec_mode' not in st.session_state:
        st.session_state.exec_mode = "cache"

    if 'show_hull' not in st.session_state:
        st.session_state.show_hull = True
    if 'show_voronoi' not in st.session_state:
        st.session_state.show_voronoi = True
    if 'show_dijkstra' not in st.session_state:
        st.session_state.show_dijkstra = True
    if 'show_passing_graph' not in st.session_state:
        st.session_state.show_passing_graph = False



def sync_video_source() -> Dict[str, Any]:
    """
    Synchronizes the active video source, video capture handle, and precomputed cache.
    Returns stream telemetry: total_frames, width, height, fps, cached_count.
    """
    init_session_state()

    active_key = st.session_state.active_source_key
    if active_key not in VIDEO_SOURCES:
        active_key = "premier_league"
        st.session_state.active_source_key = active_key

    active_info = VIDEO_SOURCES[active_key]
    active_path = active_info["path"]
    active_cache = active_info["cache"]
    default_mode = active_info["default_mode"]

    stream = st.session_state.stream_manager

    # Video file reload if needed
    if stream.current_path != active_path:
        if os.path.exists(active_path):
            stream.load_video_source(active_path)
            if stream.width > 0 and stream.height > 0:
                st.session_state.transformer.update_dimensions(stream.width, stream.height)
        if st.session_state.exec_mode != "live":
            if os.path.exists(active_cache):
                st.session_state.precomputed_data = load_precomputed_segment(active_cache)
            elif os.path.exists(os.path.join("data_backup", os.path.basename(active_cache))):
                backup_p = os.path.join("data_backup", os.path.basename(active_cache))
                st.session_state.precomputed_data = load_precomputed_segment(backup_p)
            else:
                st.session_state.precomputed_data = None
        else:
            st.session_state.precomputed_data = None

        st.session_state.current_frame = 0

        if not st.session_state.manual_calib_enabled:
            st.session_state.transformer.set_camera_mode(default_mode)
            stream.set_camera_mode(default_mode)
            if stream.width > 0 and stream.height > 0:
                st.session_state.transformer.update_dimensions(stream.width, stream.height)
            st.session_state.overlay_engine.transformer = st.session_state.transformer

    elif st.session_state.exec_mode == "live":
        st.session_state.precomputed_data = None
    elif st.session_state.get('precomputed_data') is None and st.session_state.exec_mode != "live":
        if os.path.exists(active_cache):
            st.session_state.precomputed_data = load_precomputed_segment(active_cache)
        elif os.path.exists(os.path.join("data_backup", os.path.basename(active_cache))):
            backup_p = os.path.join("data_backup", os.path.basename(active_cache))
            st.session_state.precomputed_data = load_precomputed_segment(backup_p)

    total_frames = max(1, stream.total_frames)
    vid_w = stream.width or 1280
    vid_h = stream.height or 720
    fps_val = getattr(stream, 'fps', 25.0) or 25.0
    cached_count = len(st.session_state.precomputed_data.get('frames', [])) if st.session_state.precomputed_data else 0

    # Ensure frame is within valid bounds
    st.session_state.current_frame = max(0, min(st.session_state.current_frame, total_frames - 1))

    return {
        "active_key": active_key,
        "title": active_info["title"],
        "path": active_path,
        "cache_path": active_cache,
        "total_frames": total_frames,
        "width": vid_w,
        "height": vid_h,
        "fps": fps_val,
        "cached_count": cached_count
    }


def get_raw_frame(frame_idx: int) -> Optional[np.ndarray]:
    """Retrieves BGR frame from OpenCV stream with thread safety and zero unnecessary seeking."""
    stream = st.session_state.stream_manager

    # Fast-path: Check if background prefetch worker already buffered this raw frame in memory
    if hasattr(stream, 'get_cached_raw_frame'):
        cached_f = stream.get_cached_raw_frame(frame_idx)
        if cached_f is not None:
            return cached_f

    cap = stream.current_cap
    if cap is None or not cap.isOpened():
        active_path = VIDEO_SOURCES[st.session_state.active_source_key]["path"]
        if os.path.exists(active_path):
            stream.load_video_source(active_path)
            cap = stream.current_cap
    if cap is None or not cap.isOpened():
        return None

    with stream._cap_lock:
        last_pos = getattr(stream, '_last_read_pos', -1)
        if last_pos == frame_idx:
            ret, frame = cap.read()
            if ret:
                stream._last_read_pos = frame_idx + 1
                return frame

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            stream._last_read_pos = frame_idx + 1
            return frame
    return None



def execute_pipeline(
    frame_idx: int,
    force_live: bool = False,
    hull_algo: str = "graham"
) -> Dict[str, Any]:
    """
    Executes or loads pipeline data for a given frame index.
    Returns typed contracts, coach metrics, and health diagnostics.
    """
    raw_frame = get_raw_frame(frame_idx)
    vid_w = st.session_state.stream_manager.width or 1280
    vid_h = st.session_state.stream_manager.height or 720

    if raw_frame is None:
        raw_frame = np.zeros((vid_h, vid_w, 3), dtype=np.uint8)

    rf_h, rf_w = raw_frame.shape[:2]
    has_cache = (
        not force_live
        and st.session_state.precomputed_data is not None
        and frame_idx < len(st.session_state.precomputed_data.get('frames', []))
    )

    t0 = time.perf_counter()
    if has_cache:
        f_data = st.session_state.precomputed_data['frames'][frame_idx]
        team_a = f_data['team_a']
        team_b = f_data['team_b']
        ball = f_data['ball']
        carrier = f_data.get('carrier_idx', None)

        hull_poly = f_data['hull']['polygon']
        hull_area = f_data['hull']['area']
        hull_compact = f_data['hull']['compactness']

        voronoi_res = f_data['voronoi']
        latency = f_data.get('latency_ms', 1.8)

        # Invert precomputed metric pitch positions to true camera pixel space
        ref_transformer = st.session_state.get('transformer')
        if ref_transformer is None:
            active_key = st.session_state.active_source_key
            default_mode = VIDEO_SOURCES.get(active_key, {}).get("default_mode", "sideline")
            ref_transformer = HomographyTransformer(frame_width=rf_w, frame_height=rf_h, camera_mode=default_mode)

        team_a_pixels = ref_transformer.pitch_to_image(team_a, filter_behind_camera=True)
        team_b_pixels = ref_transformer.pitch_to_image(team_b, filter_behind_camera=True)
        ball_pixels = ref_transformer.pitch_to_image([ball], filter_behind_camera=True) if ball else []
        ball_pixel = ball_pixels[0] if ball_pixels else None

        # Synthesize realistic player bounding boxes anchored at foot contact points
        player_contracts = []
        bw = max(12, int(rf_w * 0.016))
        bh = max(24, int(rf_h * 0.055))
        for px, py in team_a_pixels:
            box = (max(0, px - bw // 2), max(0, py - bh), min(rf_w, px + bw // 2), min(rf_h, py))
            player_contracts.append(PlayerDetection(box=box, foot_point=(float(px), float(py)), team_id=0))
        for px, py in team_b_pixels:
            box = (max(0, px - bw // 2), max(0, py - bh), min(rf_w, px + bw // 2), min(rf_h, py))
            player_contracts.append(PlayerDetection(box=box, foot_point=(float(px), float(py)), team_id=1))

        detections = FrameDetections(
            frame_idx=frame_idx,
            players=player_contracts,
            ball_pixel=ball_pixel,
            frame_width=rf_w,
            frame_height=rf_h
        )

        is_custom = getattr(st.session_state.transformer, 'is_custom_calibrated', False) or st.session_state.manual_calib_enabled
        if is_custom:
            # Dynamically reproject through user's active manual calibration!
            positions = stage2_projection(detections, st.session_state.transformer)
            geometry = stage3_geometry(positions, hull_algo=hull_algo)
        else:
            positions = PitchPositions(
                frame_idx=frame_idx,
                team_a=team_a,
                team_b=team_b,
                ball=ball,
                carrier_idx=carrier,
                detected_count=len(team_a) + len(team_b)
            )
            hull_player_indices = f_data.get('hull', {}).get('player_indices', [])
            if not hull_player_indices and len(hull_poly) >= 3 and team_b:
                for pt in hull_poly:
                    matched_idx = -1
                    for idx, player_pt in enumerate(team_b):
                        if abs(player_pt[0] - pt[0]) < 1e-3 and abs(player_pt[1] - pt[1]) < 1e-3:
                            matched_idx = idx
                            break
                    if matched_idx != -1:
                        hull_player_indices.append(matched_idx)

            dijkstra_res = f_data.get('dijkstra', f_data.get('dijkstra_res', None))
            if (dijkstra_res is None or 'graph_edges' not in dijkstra_res) and positions.team_a:
                dijkstra_res = compute_dijkstra_passing_path(
                    attacking_players=positions.team_a,
                    defending_players=positions.team_b,
                    carrier_idx=positions.carrier_idx,
                    ball_coord=positions.ball
                )

            geometry = TacticalGeometry(
                frame_idx=frame_idx,
                hull_polygon=hull_poly,
                hull_player_indices=hull_player_indices,
                hull_area=hull_area,
                hull_compactness=hull_compact,
                hull_vertices=len(hull_poly),
                voronoi_cells_a=voronoi_res['cells_a'],
                voronoi_cells_b=voronoi_res['cells_b'],
                team_a_pct=voronoi_res['team_a_pct'],
                team_b_pct=voronoi_res['team_b_pct'],
                dijkstra_res=dijkstra_res,
                latency_ms=latency
            )
    else:
        # Live decoupled execution with O(1) temporal detection cache lookup
        stream = st.session_state.stream_manager
        detections = None
        if hasattr(stream, 'get_cached_detections'):
            detections = stream.get_cached_detections(frame_idx)
        elif hasattr(stream, 'detection_cache') and frame_idx in stream.detection_cache:
            stream.detection_cache.move_to_end(frame_idx)
            detections = stream.detection_cache[frame_idx].get('detections')

        if detections is None:
            detections = stage1_perception(raw_frame, st.session_state.detector, frame_idx=frame_idx)
            if hasattr(stream, 'put_cached_detections'):
                stream.put_cached_detections(frame_idx, detections, raw_frame=raw_frame)
            elif hasattr(stream, 'detection_cache'):
                if len(stream.detection_cache) >= 1000:
                    stream.detection_cache.popitem(last=False)
                stream.detection_cache[frame_idx] = {'detections': detections, 'raw_frame': raw_frame}

        positions = stage2_projection(detections, st.session_state.transformer)
        geometry = stage3_geometry(positions, hull_algo=hull_algo)
        latency = (time.perf_counter() - t0) * 1000.0

        # FW-05: Non-blocking asynchronous background lookahead prefetching for sequential scrubbing
        if hasattr(stream, 'prefetch_lookahead'):
            stream.prefetch_lookahead(frame_idx, lookahead=1)
            stream.prefetch_lookahead(frame_idx, lookahead=2)

    cam_mode_str = "manual_mouse" if st.session_state.manual_calib_enabled else st.session_state.transformer.camera_mode
    diag_status, diag_issues = evaluate_pipeline_health(
        detections=detections,
        positions=positions,
        geometry=geometry,
        match_source=st.session_state.active_source_key,
        camera_mode=cam_mode_str,
        latencies={"stage1": latency, "stage2": 0.1, "stage3": geometry.latency_ms, "stage4": 1.0}
    )

    vor_res_dict = {
        'team_a_pct': geometry.team_a_pct,
        'team_b_pct': geometry.team_b_pct,
        'cells_a': geometry.voronoi_cells_a,
        'cells_b': geometry.voronoi_cells_b
    }
    coach_data = evaluate_coach_analytics(
        positions.team_a, positions.team_b,
        {'hull': geometry.hull_polygon, 'area_sq_m': geometry.hull_area},
        vor_res_dict, geometry.dijkstra_res, None, positions.ball
    )

    return {
        'raw_frame': raw_frame,
        'detections': detections,
        'positions': positions,
        'geometry': geometry,
        'coach_data': coach_data,
        'diag_status': diag_status,
        'diag_issues': diag_issues,
        'latency_ms': latency,
        'is_cached': has_cache
    }
