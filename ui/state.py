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

from ui.media_library import (
    CUSTOM_VIDEOS_DIR,
    CUSTOM_CACHES_DIR,
    LOCAL_VIDEOS_DIR,
    VIDEO_SOURCES,
    get_custom_cache_path,
    build_custom_upload_paths,
    list_available_custom_videos,
    invalidate_custom_user_cache,
    load_video_calibration,
    save_video_calibration,
    delete_video_calibration,
)

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
from perception.precompute import load_precomputed_segment, get_cached_frame_bundle
from perception.calibration_templates import CALIBRATION_TEMPLATES
from visualization.pitch_renderer import TacticalPitchRenderer
from visualization.overlays import BroadcastOverlayEngine
from algorithms.coach_analytics import evaluate_coach_analytics
from algorithms.passing_network import compute_dijkstra_passing_path


def get_active_source_info(source_key: Optional[str] = None) -> Dict[str, str]:
    """Resolves the active video and cache paths without mutating shared globals or built-in sources."""
    key = source_key or st.session_state.get("active_source_key", "premier_league")
    if key not in VIDEO_SOURCES:
        key = "premier_league"
    if key != "custom_upload":
        return dict(VIDEO_SOURCES[key])

    v_path = st.session_state.get("custom_video_path") or ""
    if not v_path or not os.path.exists(v_path):
        available = list_available_custom_videos()
        if available:
            v_path = available[0]
            st.session_state.custom_video_path = v_path
            st.session_state.custom_cache_path = get_custom_cache_path(v_path)
        else:
            v_path = ""

    c_path = st.session_state.get("custom_cache_path") or (
        get_custom_cache_path(v_path) if v_path else ""
    )
    if v_path and not st.session_state.get("custom_cache_path"):
        st.session_state.custom_cache_path = c_path

    title = f"Custom: {os.path.basename(v_path)}" if v_path else VIDEO_SOURCES["custom_upload"]["title"]
    return {
        "title": title,
        "path": v_path,
        "cache": c_path,
        "default_mode": "sideline",
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
        st.session_state.is_playing = False

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
    if 'custom_video_path' not in st.session_state:
        st.session_state.custom_video_path = ""
    if 'custom_cache_path' not in st.session_state:
        st.session_state.custom_cache_path = ""

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
    if 'show_ruler' not in st.session_state:
        st.session_state.show_ruler = False
    if 'ruler_mode' not in st.session_state:
        st.session_state.ruler_mode = "players"
    if 'ruler_p1_idx' not in st.session_state:
        st.session_state.ruler_p1_idx = 0
    if 'ruler_p2_idx' not in st.session_state:
        st.session_state.ruler_p2_idx = 1
    if 'ruler_custom_p1' not in st.session_state:
        st.session_state.ruler_custom_p1 = (35.0, 20.0)
    if 'ruler_custom_p2' not in st.session_state:
        st.session_state.ruler_custom_p2 = (65.0, 45.0)
    if 'ruler_auto_offside' not in st.session_state:
        st.session_state.ruler_auto_offside = True
    if 'ruler_manual_offside_x' not in st.session_state:
        st.session_state.ruler_manual_offside_x = 70.0

    # Automated Native-FPS Frame Playback State
    if 'is_playing' not in st.session_state:
        st.session_state.is_playing = False
    if 'playback_speed' not in st.session_state:
        st.session_state.playback_speed = 1.0
    if 'playback_loop' not in st.session_state:
        st.session_state.playback_loop = True



def _is_cache_compatible(loaded: Optional[Dict[str, Any]], stream_width: int) -> bool:
    """Checks resolution and calibration signature compatibility before hydrating cached frames."""
    if not loaded:
        return False
    if stream_width > 0 and loaded.get('video_width') != stream_width:
        return False
    cache_sig = loaded.get('calibration_sig')
    if getattr(st.session_state.transformer, 'is_custom_calibrated', False):
        if cache_sig != st.session_state.transformer.get_calibration_signature():
            return False
    return True


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

    active_info = get_active_source_info(active_key)
    active_path = active_info["path"]
    active_cache = active_info["cache"]
    default_mode = active_info["default_mode"]

    stream = st.session_state.stream_manager

    # Video file reload if needed
    if not active_path:
        st.session_state.precomputed_data = None
    elif stream.current_path != active_path:
        if os.path.exists(active_path):
            stream.load_video_source(active_path)
        st.session_state.current_frame = 0
        st.session_state.is_playing = False
        st.session_state.tv_calib_points = []
        st.session_state.tv_calib_applied = False
        st.session_state.tv_calib_just_applied = False
        st.session_state.pop("setup_calib_type_radio", None)
        st.session_state.pop("setup_preset_select", None)
        for _pm_k in [k for k in st.session_state.keys() if k.endswith("_pm_cached_frame_key")]:
            st.session_state.pop(_pm_k, None)

        if stream.width > 0 and stream.height > 0:
            st.session_state.transformer.update_dimensions(stream.width, stream.height)

        saved_cal = load_video_calibration(active_path, stream.width, stream.height)
        if saved_cal:
            try:
                src_pts = np.float32(saved_cal["src_points"])
                dst_pts = np.float32(saved_cal["dst_points"])
                w_mode = str(saved_cal.get("warp_mode", "auto"))
                saved_lambda = saved_cal.get("lens_lambda")
                st.session_state.transformer.calibrate(
                    src_pts,
                    dst_pts,
                    frame_width=stream.width or st.session_state.transformer.frame_width,
                    frame_height=stream.height or st.session_state.transformer.frame_height,
                    warp_mode=w_mode,
                    lens_lambda=float(saved_lambda) if saved_lambda is not None else None,
                )
                st.session_state.manual_calib_enabled = True
                st.session_state.manual_calib_applied = True
                st.session_state.tv_calib_enabled = True
                st.session_state.tv_calib_applied = True
                st.session_state.tv_calib_points = [(float(p[0]), float(p[1])) for p in src_pts]
                st.session_state.tv_calib_warp_mode = st.session_state.transformer.warp_mode
                st.session_state.tv_calib_mesh_handles = saved_cal.get("handles")
            except Exception:
                saved_cal = None

        if not saved_cal:
            st.session_state.manual_calib_enabled = False
            st.session_state.manual_calib_applied = False
            st.session_state.tv_calib_enabled = False
            st.session_state.tv_calib_applied = False
            st.session_state.tv_calib_points = []
            st.session_state.tv_calib_mesh_handles = None
            st.session_state.transformer.set_camera_mode(default_mode)
            stream.set_camera_mode(default_mode)
            if stream.width > 0 and stream.height > 0:
                st.session_state.transformer.update_dimensions(stream.width, stream.height)

        if st.session_state.exec_mode != "live":
            if active_cache and os.path.exists(active_cache):
                loaded = load_precomputed_segment(active_cache)
                st.session_state.precomputed_data = loaded if _is_cache_compatible(loaded, stream.width) else None
            elif active_cache and os.path.exists(os.path.join("data_backup", os.path.basename(active_cache))):
                backup_p = os.path.join("data_backup", os.path.basename(active_cache))
                loaded = load_precomputed_segment(backup_p)
                st.session_state.precomputed_data = loaded if _is_cache_compatible(loaded, stream.width) else None
            else:
                st.session_state.precomputed_data = None
        else:
            st.session_state.precomputed_data = None

    else:
        if stream.width > 0 and stream.height > 0:
            st.session_state.transformer.update_dimensions(stream.width, stream.height)

        if st.session_state.exec_mode == "live":
            st.session_state.precomputed_data = None
        elif st.session_state.get('precomputed_data') is None and active_cache:
            if os.path.exists(active_cache):
                loaded = load_precomputed_segment(active_cache)
                st.session_state.precomputed_data = loaded if _is_cache_compatible(loaded, stream.width) else None
            elif os.path.exists(os.path.join("data_backup", os.path.basename(active_cache))):
                backup_p = os.path.join("data_backup", os.path.basename(active_cache))
                loaded = load_precomputed_segment(backup_p)
                st.session_state.precomputed_data = loaded if _is_cache_compatible(loaded, stream.width) else None

    # Keep all subsystem references pointed to the single canonical transformer
    st.session_state.overlay_engine.transformer = st.session_state.transformer
    stream.transformer = st.session_state.transformer

    total_frames = max(1, stream.total_frames)
    vid_w = stream.width or 1280
    vid_h = stream.height or 720
    fps_val = getattr(stream, 'fps', 25.0) or 25.0
    cached_count = st.session_state.precomputed_data.get('total_frames', len(st.session_state.precomputed_data.get('frames', []))) if st.session_state.precomputed_data else 0

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
        "cached_count": cached_count,
        "ring_buffer": stream.get_ring_buffer_telemetry() if hasattr(stream, 'get_ring_buffer_telemetry') else {}
    }


def get_raw_frame(frame_idx: int) -> Optional[np.ndarray]:
    """Retrieves BGR frame with FW-13 in-memory ring buffer caching and zero OpenCV seek overhead."""
    stream = st.session_state.stream_manager

    # 1. Fast-path: Check FW-13 in-memory ring buffer (< 0.05 ms)
    if hasattr(stream, 'get_cached_raw_frame'):
        cached_f = stream.get_cached_raw_frame(frame_idx)
        if cached_f is not None:
            # Trigger lookahead window buffering during active playback or scrubbing
            if st.session_state.get('is_playing', False) and hasattr(stream, 'prefetch_raw_window'):
                stream.prefetch_raw_window(frame_idx, window_size=15)
            return cached_f

    # 2. Slow-path: OpenCV VideoCapture decode from disk
    cap = stream.current_cap
    if cap is None or not cap.isOpened():
        active_path = get_active_source_info().get("path", "")
        if active_path and os.path.exists(active_path):
            stream.load_video_source(active_path)
            cap = stream.current_cap
    if cap is None or not cap.isOpened():
        return None

    frame = None
    with stream._cap_lock:
        last_pos = getattr(stream, '_last_read_pos', -1)
        if last_pos == frame_idx:
            ret, f_read = cap.read()
            if ret and f_read is not None:
                stream._last_read_pos = frame_idx + 1
                frame = f_read

        if frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, f_read = cap.read()
            if ret and f_read is not None:
                stream._last_read_pos = frame_idx + 1
                frame = f_read

    # 3. Store into FW-13 in-memory ring buffer for instant recall & loop playback
    if frame is not None:
        if hasattr(stream, 'put_cached_raw_frame'):
            stream.put_cached_raw_frame(frame_idx, frame)
        if hasattr(stream, 'prefetch_raw_window'):
            stream.prefetch_raw_window(frame_idx, window_size=15)
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
    stream = st.session_state.stream_manager
    raw_frame = get_raw_frame(frame_idx)
    vid_w = stream.width or 1280
    vid_h = stream.height or 720

    if raw_frame is None:
        raw_frame = np.zeros((vid_h, vid_w, 3), dtype=np.uint8)

    rf_h, rf_w = raw_frame.shape[:2]
    f_data = None
    if not force_live and st.session_state.precomputed_data is not None:
        f_data = get_cached_frame_bundle(st.session_state.precomputed_data, frame_idx)

    has_cache = f_data is not None

    t0 = time.perf_counter()
    if has_cache:
        team_a = f_data['team_a']
        team_b = f_data['team_b']
        ball = f_data['ball']
        carrier = f_data.get('carrier_idx', None)

        hull_poly = f_data['hull']['polygon']
        hull_area = f_data['hull']['area']
        hull_compact = f_data['hull']['compactness']

        voronoi_res = f_data['voronoi']
        latency = f_data.get('latency_ms', 1.8)

        # Invert precomputed metric pitch positions to true camera pixel space using cache's original camera mode
        pre_meta = st.session_state.get('precomputed_data') or {}
        orig_cam_mode = pre_meta.get('camera_mode') or get_active_source_info(st.session_state.active_source_key).get("default_mode", "sideline")
        ref_transformer = HomographyTransformer(frame_width=rf_w, frame_height=rf_h, camera_mode=orig_cam_mode)

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

    # FW-06: Interactive Tactical Distance & Offside Ruler
    ruler_data = None
    if st.session_state.get('show_ruler', False):
        try:
            from algorithms.tactical_ruler import compute_tactical_ruler
            team_a = getattr(positions, 'team_a', [])
            team_b = getattr(positions, 'team_b', [])
            r_mode = st.session_state.get('ruler_mode', 'players')

            if r_mode == 'players' and len(team_a) >= 2:
                p1_i = min(len(team_a) - 1, max(0, st.session_state.get('ruler_p1_idx', 0)))
                p2_i = min(len(team_a) - 1, max(0, st.session_state.get('ruler_p2_idx', 1)))
                pt_a = team_a[p1_i]
                pt_b = team_a[p2_i]
                lbl_a = f"Player A#{p1_i + 1}"
                lbl_b = f"Player A#{p2_i + 1}"
            elif r_mode == 'inter_team' and team_a and team_b:
                p1_i = min(len(team_a) - 1, max(0, st.session_state.get('ruler_p1_idx', 0)))
                p2_i = min(len(team_b) - 1, max(0, st.session_state.get('ruler_p2_idx', 0)))
                pt_a = team_a[p1_i]
                pt_b = team_b[p2_i]
                lbl_a = f"Player A#{p1_i + 1}"
                lbl_b = f"Player B#{p2_i + 1}"
            elif r_mode == 'ball' and positions.ball and team_a:
                p1_i = min(len(team_a) - 1, max(0, st.session_state.get('ruler_p1_idx', 0)))
                pt_a = positions.ball
                pt_b = team_a[p1_i]
                lbl_a = "Match Ball"
                lbl_b = f"Player A#{p1_i + 1}"
            else:
                pt_a = st.session_state.get('ruler_custom_p1', (35.0, 20.0))
                pt_b = st.session_state.get('ruler_custom_p2', (65.0, 45.0))
                lbl_a = "Point A"
                lbl_b = "Point B"

            off_x = None if st.session_state.get('ruler_auto_offside', True) else float(st.session_state.get('ruler_manual_offside_x', 70.0))
            ruler_data = compute_tactical_ruler(
                point_a=pt_a,
                point_b=pt_b,
                label_a=lbl_a,
                label_b=lbl_b,
                offside_x=off_x,
                attacking_players=team_a,
                defending_players=team_b
            )
        except Exception:
            ruler_data = None

    return {
        'raw_frame': raw_frame,
        'detections': detections,
        'positions': positions,
        'geometry': geometry,
        'coach_data': coach_data,
        'ruler_data': ruler_data,
        'onnx_telemetry': st.session_state.detector.get_onnx_telemetry() if hasattr(st.session_state, 'detector') and hasattr(st.session_state.detector, 'get_onnx_telemetry') else {},
        'diag_status': diag_status,
        'diag_issues': diag_issues,
        'latency_ms': latency,
        'is_cached': has_cache,
        'ring_buffer': stream.get_ring_buffer_telemetry() if hasattr(stream, 'get_ring_buffer_telemetry') else {}
    }
