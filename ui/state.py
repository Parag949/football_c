"""
TacticalVision State & Stream Synchronization Layer
===================================================
Manages centralized session state, video streams, precomputed cache synchronization,
and lazy pipeline bundle execution with zero duplicate keys or redundant reruns.
"""

import os
import time
import threading
from typing import Dict, Any, Optional
import cv2
import numpy as np
import streamlit as st

from ui.state_models import (
    AppState,
    VideoSourceState,
    PlaybackState,
    CalibrationState,
    AnalyticsDisplayState,
    RulerState,
    StateModelBase,
    coerce_anchor_count,
)
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
    evaluate_pipeline_health,
    stage4_render_radar,
    stage4_render_overlay
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
from algorithms.tactical_ruler import compute_tactical_ruler


def get_active_source_info(source_key: Optional[str] = None) -> Dict[str, str]:
    """Resolves the active video and cache paths without mutating shared globals or built-in sources."""
    default_fallback = list(VIDEO_SOURCES.keys())[0] if VIDEO_SOURCES else "world_cup"
    key = source_key or st.session_state.get("active_source_key", default_fallback)
    if key not in VIDEO_SOURCES:
        key = default_fallback
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


_in_bridge_sync = False


def reset_widget_state(*keys: str) -> None:
    """Cleanly pops Streamlit widget keys to prevent stale radio/slider state on reruns."""
    for k in keys:
        try:
            if not k or not isinstance(k, str) or k.startswith("$$ID"):
                continue
            if k in st.session_state:
                del st.session_state[k]
        except Exception:
            pass


def get_app_state() -> AppState:
    """Returns the canonical typed AppState container from st.session_state."""
    if 'app_state' not in st.session_state or not isinstance(st.session_state.app_state, AppState):
        init_session_state()
    return st.session_state.app_state


def _sync_loose_to_app_state(sess: Any, app: AppState) -> None:
    """Populates AppState from loose session keys if they exist in sess."""
    if 'setup_confirmed' in sess:
        app.setup_confirmed = bool(sess['setup_confirmed'])
    if 'exec_mode' in sess:
        app.exec_mode = str(sess['exec_mode'])
    if 'active_source_key' in sess:
        app.video.active_key = str(sess['active_source_key'])
    if 'custom_video_path' in sess:
        app.video.custom_video_path = str(sess['custom_video_path'])
    if 'custom_cache_path' in sess:
        app.video.custom_cache_path = str(sess['custom_cache_path'])
    if 'last_uploaded_sig' in sess:
        app.video.last_uploaded_sig = str(sess['last_uploaded_sig'])
    if 'last_uploaded_name' in sess:
        app.video.last_uploaded_name = str(sess['last_uploaded_name'])

    if 'current_frame' in sess:
        v = int(sess['current_frame'])
        app.playback.current_frame = v
        app.playback.main_slider_val = v
    elif 'main_slider_val' in sess:
        v = int(sess['main_slider_val'])
        app.playback.current_frame = v
        app.playback.main_slider_val = v
    if 'is_playing' in sess:
        app.playback.is_playing = bool(sess['is_playing'])
    if 'playback_speed' in sess:
        app.playback.speed = float(sess['playback_speed'])
    if 'playback_loop' in sess:
        app.playback.loop = bool(sess['playback_loop'])

    if 'show_hull' in sess:
        app.display.show_hull = bool(sess['show_hull'])
    if 'show_voronoi' in sess:
        app.display.show_voronoi = bool(sess['show_voronoi'])
    if 'show_dijkstra' in sess:
        app.display.show_dijkstra = bool(sess['show_dijkstra'])
    if 'show_passing_graph' in sess:
        app.display.show_passing_graph = bool(sess['show_passing_graph'])
    if 'show_ruler' in sess:
        app.display.show_ruler = bool(sess['show_ruler'])
    if 'attack_direction' in sess:
        app.display.attack_direction = str(sess['attack_direction'])
    if 'ui_hull_algo_val' in sess:
        app.display.hull_algo = str(sess['ui_hull_algo_val'])

    if 'ruler_mode' in sess:
        app.ruler.mode = str(sess['ruler_mode'])
    if 'ruler_p1_idx' in sess:
        app.ruler.p1_idx = int(sess['ruler_p1_idx'])
    if 'ruler_p2_idx' in sess:
        app.ruler.p2_idx = int(sess['ruler_p2_idx'])
    if 'ruler_custom_p1' in sess:
        app.ruler.custom_p1 = tuple(sess['ruler_custom_p1'])
    if 'ruler_custom_p2' in sess:
        app.ruler.custom_p2 = tuple(sess['ruler_custom_p2'])
    if 'ruler_auto_offside' in sess:
        app.ruler.auto_offside = bool(sess['ruler_auto_offside'])
    if 'ruler_manual_offside_x' in sess:
        app.ruler.manual_offside_x = float(sess['ruler_manual_offside_x'])

    is_c = bool(
        sess.get('manual_calib_enabled')
        or sess.get('manual_calib_applied')
        or sess.get('tv_calib_enabled')
        or sess.get('tv_calib_applied')
    )
    if is_c or any(k in sess for k in ('manual_calib_enabled', 'manual_calib_applied', 'tv_calib_enabled', 'tv_calib_applied')):
        app.calibration.is_custom = is_c

    if 'tv_calib_points' in sess:
        app.calibration.points = list(sess['tv_calib_points'])
    elif 'manual_calib_points' in sess:
        app.calibration.points = list(sess['manual_calib_points'])

    if 'tv_calib_template' in sess:
        app.calibration.template = str(sess['tv_calib_template'])
    elif 'manual_dst_template' in sess:
        app.calibration.template = str(sess['manual_dst_template'])

    if 'tv_calib_warp_mode' in sess:
        app.calibration.warp_mode = str(sess['tv_calib_warp_mode'])
    if 'tv_calib_num_anchors' in sess:
        app.calibration.num_anchors = coerce_anchor_count(sess['tv_calib_num_anchors'])
    if 'tv_calib_mesh_handles' in sess:
        app.calibration.mesh_handles = sess['tv_calib_mesh_handles']
    if 'tv_calib_just_applied' in sess:
        app.calibration.just_applied = bool(sess['tv_calib_just_applied'])
    if 'tv_calib_just_reset' in sess:
        app.calibration.just_reset = bool(sess['tv_calib_just_reset'])
    if 'tv_calib_picker_mode' in sess:
        app.calibration.picker_mode = str(sess['tv_calib_picker_mode'])


def _sync_app_state_to_loose(app: AppState, sess: Any) -> None:
    """Populates loose session state variables from AppState."""
    global _in_bridge_sync
    _in_bridge_sync = True
    try:
        sess['setup_confirmed'] = app.setup_confirmed
        sess['exec_mode'] = app.exec_mode

        sess['active_source_key'] = app.video.active_key
        sess['custom_video_path'] = app.video.custom_video_path
        sess['custom_cache_path'] = app.video.custom_cache_path
        sess['last_uploaded_sig'] = app.video.last_uploaded_sig
        sess['last_uploaded_name'] = app.video.last_uploaded_name

        sess['current_frame'] = app.playback.current_frame
        sess['is_playing'] = app.playback.is_playing
        sess['playback_speed'] = app.playback.speed
        sess['playback_loop'] = app.playback.loop
        sess['main_slider_val'] = app.playback.main_slider_val

        sess['show_hull'] = app.display.show_hull
        sess['show_voronoi'] = app.display.show_voronoi
        sess['show_dijkstra'] = app.display.show_dijkstra
        sess['show_passing_graph'] = app.display.show_passing_graph
        sess['show_ruler'] = app.display.show_ruler
        sess['attack_direction'] = app.display.attack_direction
        sess['ui_hull_algo_val'] = app.display.hull_algo

        sess['ruler_mode'] = app.ruler.mode
        sess['ruler_p1_idx'] = app.ruler.p1_idx
        sess['ruler_p2_idx'] = app.ruler.p2_idx
        sess['ruler_custom_p1'] = app.ruler.custom_p1
        sess['ruler_custom_p2'] = app.ruler.custom_p2
        sess['ruler_auto_offside'] = app.ruler.auto_offside
        sess['ruler_manual_offside_x'] = app.ruler.manual_offside_x

        # Collapsed Calibration Keys
        is_c = app.calibration.is_custom
        sess['manual_calib_enabled'] = is_c
        sess['manual_calib_applied'] = is_c
        sess['tv_calib_enabled'] = is_c
        sess['tv_calib_applied'] = is_c
        sess['manual_calib_points'] = list(app.calibration.points)
        sess['tv_calib_points'] = list(app.calibration.points)
        sess['manual_dst_template'] = app.calibration.template
        sess['tv_calib_template'] = app.calibration.template
        sess['tv_calib_warp_mode'] = app.calibration.warp_mode
        sess['tv_calib_num_anchors'] = app.calibration.num_anchors
        sess['tv_calib_mesh_handles'] = app.calibration.mesh_handles
        sess['tv_calib_just_applied'] = app.calibration.just_applied
        sess['tv_calib_just_reset'] = app.calibration.just_reset
        sess['tv_calib_picker_mode'] = app.calibration.picker_mode
    finally:
        _in_bridge_sync = False


def _bind_app_state_to_session(app: AppState, sess: Any) -> None:
    """Attaches a sync observer on AppState so that mutations update st.session_state."""
    def on_app_state_changed(source: StateModelBase, name: str, value: Any):
        global _in_bridge_sync
        if _in_bridge_sync:
            return
        _in_bridge_sync = True
        try:
            if source is app:
                if name in ('setup_confirmed', 'exec_mode'):
                    sess[name] = value
            elif source is app.video:
                key_map = {
                    'active_key': 'active_source_key',
                    'custom_video_path': 'custom_video_path',
                    'custom_cache_path': 'custom_cache_path',
                    'last_uploaded_sig': 'last_uploaded_sig',
                    'last_uploaded_name': 'last_uploaded_name',
                }
                if name in key_map:
                    sess[key_map[name]] = value
            elif source is app.playback:
                if name == 'current_frame':
                    sess['current_frame'] = value
                    sess['main_slider_val'] = value
                elif name == 'main_slider_val':
                    sess['main_slider_val'] = value
                    sess['current_frame'] = value
                elif name == 'is_playing':
                    sess['is_playing'] = value
                elif name == 'speed':
                    sess['playback_speed'] = value
                elif name == 'loop':
                    sess['playback_loop'] = value
            elif source is app.calibration:
                if name == 'is_custom':
                    b = bool(value)
                    sess['manual_calib_enabled'] = b
                    sess['manual_calib_applied'] = b
                    sess['tv_calib_enabled'] = b
                    sess['tv_calib_applied'] = b
                elif name == 'points':
                    pts = list(value)
                    sess['manual_calib_points'] = pts
                    sess['tv_calib_points'] = pts
                elif name == 'template':
                    t = str(value)
                    sess['manual_dst_template'] = t
                    sess['tv_calib_template'] = t
                elif name == 'warp_mode':
                    sess['tv_calib_warp_mode'] = str(value)
                elif name == 'num_anchors':
                    c = coerce_anchor_count(value)
                    sess['tv_calib_num_anchors'] = c
                elif name == 'just_applied':
                    sess['tv_calib_just_applied'] = bool(value)
                elif name == 'just_reset':
                    sess['tv_calib_just_reset'] = bool(value)
                elif name == 'picker_mode':
                    sess['tv_calib_picker_mode'] = str(value)
                elif name == 'mesh_handles':
                    sess['tv_calib_mesh_handles'] = value
            elif source is app.display:
                key_map = {
                    'show_hull': 'show_hull',
                    'show_voronoi': 'show_voronoi',
                    'show_dijkstra': 'show_dijkstra',
                    'show_passing_graph': 'show_passing_graph',
                    'show_ruler': 'show_ruler',
                    'hull_algo': 'ui_hull_algo_val',
                    'attack_direction': 'attack_direction',
                }
                if name in key_map:
                    sess[key_map[name]] = value
            elif source is app.ruler:
                key_map = {
                    'mode': 'ruler_mode',
                    'p1_idx': 'ruler_p1_idx',
                    'p2_idx': 'ruler_p2_idx',
                    'custom_p1': 'ruler_custom_p1',
                    'custom_p2': 'ruler_custom_p2',
                    'auto_offside': 'ruler_auto_offside',
                    'manual_offside_x': 'ruler_manual_offside_x',
                }
                if name in key_map:
                    sess[key_map[name]] = value
        finally:
            _in_bridge_sync = False

    app.set_sync_callback(on_app_state_changed)


def _install_session_state_bridge() -> None:
    """Installs hook on Streamlit's SessionStateProxy for transparent bidirectional synchronization."""
    from streamlit.runtime.state.session_state_proxy import SessionStateProxy
    if getattr(SessionStateProxy, '_bridge_installed', False):
        return

    orig_setitem = SessionStateProxy.__setitem__

    def hooked_setitem(self, key, value):
        global _in_bridge_sync
        orig_setitem(self, key, value)
        if _in_bridge_sync:
            return
        if 'app_state' not in self:
            return

        app = self['app_state']
        if not isinstance(app, AppState):
            return

        _in_bridge_sync = True
        try:
            # Top-level
            if key == 'setup_confirmed':
                app.setup_confirmed = bool(value)
            elif key == 'exec_mode':
                app.exec_mode = str(value)
            # Video
            elif key == 'active_source_key':
                app.video.active_key = str(value)
            elif key == 'custom_video_path':
                app.video.custom_video_path = str(value)
            elif key == 'custom_cache_path':
                app.video.custom_cache_path = str(value)
            elif key == 'last_uploaded_sig':
                app.video.last_uploaded_sig = str(value)
            elif key == 'last_uploaded_name':
                app.video.last_uploaded_name = str(value)
            # Playback
            elif key == 'current_frame':
                v = int(value)
                app.playback.current_frame = v
                app.playback.main_slider_val = v
                self['main_slider_val'] = v
            elif key == 'is_playing':
                app.playback.is_playing = bool(value)
            elif key == 'playback_speed':
                app.playback.speed = float(value)
            elif key == 'playback_loop':
                app.playback.loop = bool(value)
            elif key == 'main_slider_val':
                v = int(value)
                app.playback.main_slider_val = v
                app.playback.current_frame = v
                self['current_frame'] = v
            # Display
            elif key == 'show_hull':
                app.display.show_hull = bool(value)
            elif key == 'show_voronoi':
                app.display.show_voronoi = bool(value)
            elif key == 'show_dijkstra':
                app.display.show_dijkstra = bool(value)
            elif key == 'show_passing_graph':
                app.display.show_passing_graph = bool(value)
            elif key == 'show_ruler':
                app.display.show_ruler = bool(value)
            elif key == 'attack_direction':
                app.display.attack_direction = str(value)
            elif key == 'ui_hull_algo_val':
                app.display.hull_algo = str(value)
            # Ruler
            elif key == 'ruler_mode':
                app.ruler.mode = str(value)
            elif key == 'ruler_p1_idx':
                app.ruler.p1_idx = int(value)
            elif key == 'ruler_p2_idx':
                app.ruler.p2_idx = int(value)
            elif key == 'ruler_custom_p1':
                app.ruler.custom_p1 = tuple(value)
            elif key == 'ruler_custom_p2':
                app.ruler.custom_p2 = tuple(value)
            elif key == 'ruler_auto_offside':
                app.ruler.auto_offside = bool(value)
            elif key == 'ruler_manual_offside_x':
                app.ruler.manual_offside_x = float(value)
            # Collapsed Calibration
            elif key in ('manual_calib_enabled', 'manual_calib_applied', 'tv_calib_enabled', 'tv_calib_applied'):
                b = bool(value)
                app.calibration.is_custom = b
                self['manual_calib_enabled'] = b
                self['manual_calib_applied'] = b
                self['tv_calib_enabled'] = b
                self['tv_calib_applied'] = b
            elif key in ('manual_calib_points', 'tv_calib_points'):
                pts = list(value)
                app.calibration.points = pts
                self['manual_calib_points'] = pts
                self['tv_calib_points'] = pts
            elif key in ('manual_dst_template', 'tv_calib_template'):
                t = str(value)
                app.calibration.template = t
                self['manual_dst_template'] = t
                self['tv_calib_template'] = t
            elif key == 'tv_calib_just_applied':
                app.calibration.just_applied = bool(value)
            elif key == 'tv_calib_just_reset':
                app.calibration.just_reset = bool(value)
            elif key == 'tv_calib_picker_mode':
                app.calibration.picker_mode = str(value)
            elif key == 'tv_calib_warp_mode':
                app.calibration.warp_mode = str(value)
            elif key == 'tv_calib_num_anchors':
                c = coerce_anchor_count(value)
                app.calibration.num_anchors = c
                self['tv_calib_num_anchors'] = c
            elif key == 'tv_calib_mesh_handles':
                app.calibration.mesh_handles = value
        finally:
            _in_bridge_sync = False

    SessionStateProxy.__setitem__ = hooked_setitem
    SessionStateProxy._bridge_installed = True


# Automatically initialize bridge hook on module import
_install_session_state_bridge()


def init_session_state() -> None:
    """Initializes all necessary session state variables idempotently with typed AppState."""
    _install_session_state_bridge()

    if 'app_state' not in st.session_state or not isinstance(st.session_state.app_state, AppState):
        app = AppState()
        _sync_loose_to_app_state(st.session_state, app)
        st.session_state.app_state = app

    app = st.session_state.app_state
    _bind_app_state_to_session(app, st.session_state)
    _sync_app_state_to_loose(app, st.session_state)

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

    if 'precomputed_data' not in st.session_state:
        st.session_state.precomputed_data = None


def sync_subsystems_transformer() -> None:
    """Ensures overlay_engine and stream_manager strictly share the canonical transformer instance."""
    if 'transformer' not in st.session_state:
        return
    canonical = st.session_state.transformer
    if 'overlay_engine' in st.session_state and st.session_state.overlay_engine is not None:
        if hasattr(st.session_state.overlay_engine, '_transformer'):
            st.session_state.overlay_engine._transformer = canonical
        assert st.session_state.overlay_engine.transformer is canonical, (
            "overlay_engine.transformer failed single-source-of-truth verification"
        )
    if 'stream_manager' in st.session_state and st.session_state.stream_manager is not None:
        if hasattr(st.session_state.stream_manager, '_default_transformer'):
            st.session_state.stream_manager._default_transformer = canonical
        assert st.session_state.stream_manager.transformer is canonical, (
            "stream_manager.transformer failed single-source-of-truth verification"
        )



def _is_cache_compatible(loaded: Optional[Dict[str, Any]], stream_width: int) -> bool:
    """Checks resolution, camera mode, and calibration signature compatibility before hydrating cached frames."""
    if not loaded:
        return False
    if stream_width > 0 and loaded.get('video_width') != stream_width:
        return False

    # If the cache contains raw pixel player detections, it is universally compatible with any active calibration/camera mode
    has_raw_players = bool(loaded.get('frames') and len(loaded['frames']) > 0 and 'players' in loaded['frames'][0])
    if has_raw_players:
        return True

    cache_cam = loaded.get('camera_mode')
    current_cam = getattr(st.session_state.transformer, 'camera_mode', 'sideline')
    if cache_cam and cache_cam != current_cam:
        return False
    if not cache_cam and current_cam != 'sideline':
        # Legacy unlabelled caches were precomputed in sideline mode
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
        active_key = list(VIDEO_SOURCES.keys())[0] if VIDEO_SOURCES else "world_cup"
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
        reset_widget_state("setup_calib_type_radio", "setup_preset_select", "setup_exec_mode_radio")
        for _pm_k in [k for k in list(st.session_state.keys()) if k.endswith("_pm_cached_frame_key")]:
            reset_widget_state(_pm_k)

        if stream.width > 0 and stream.height > 0:
            st.session_state.transformer.update_dimensions(stream.width, stream.height)

        saved_cal = load_video_calibration(active_path, stream.width, stream.height)
        if saved_cal:
            try:
                src_pts = np.float32(saved_cal["src_points"])
                dst_pts = np.float32(saved_cal["dst_points"])
                w_mode = str(saved_cal.get("warp_mode", "auto"))
                saved_lambda = saved_cal.get("lens_lambda")
                saved_len = saved_cal.get("pitch_length")
                saved_wid = saved_cal.get("pitch_width")
                st.session_state.transformer.calibrate(
                    src_pts,
                    dst_pts,
                    frame_width=stream.width or st.session_state.transformer.frame_width,
                    frame_height=stream.height or st.session_state.transformer.frame_height,
                    warp_mode=w_mode,
                    lens_lambda=float(saved_lambda) if saved_lambda is not None else None,
                    pitch_length=float(saved_len) if saved_len is not None else None,
                    pitch_width=float(saved_wid) if saved_wid is not None else None,
                )
                pts_list = [(float(p[0]), float(p[1])) for p in src_pts]
                handles = saved_cal.get("handles")
                if 'app_state' in st.session_state:
                    st.session_state.app_state.calibration.apply_points(
                        pts_list,
                        warp=st.session_state.transformer.warp_mode,
                        handles=handles
                    )
                else:
                    st.session_state.manual_calib_enabled = True
                    st.session_state.manual_calib_applied = True
                    st.session_state.tv_calib_enabled = True
                    st.session_state.tv_calib_applied = True
                    st.session_state.tv_calib_points = pts_list
                    st.session_state.tv_calib_warp_mode = st.session_state.transformer.warp_mode
                    st.session_state.tv_calib_mesh_handles = handles
                st.session_state.setup_calib_type_radio = "🎯 Interactive Pitch Mask Alignment"
            except Exception:
                saved_cal = None

        if not saved_cal:
            if 'app_state' in st.session_state:
                st.session_state.app_state.calibration.reset()
            else:
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

        if st.session_state.exec_mode == "live":
            st.session_state.precomputed_data = None
        elif active_cache and os.path.exists(active_cache):
            loaded = load_precomputed_segment(active_cache)
            st.session_state.precomputed_data = loaded if _is_cache_compatible(loaded, stream.width) else None
        elif active_cache and os.path.exists(os.path.join("data_backup", os.path.basename(active_cache))):
            backup_p = os.path.join("data_backup", os.path.basename(active_cache))
            loaded = load_precomputed_segment(backup_p)
            st.session_state.precomputed_data = loaded if _is_cache_compatible(loaded, stream.width) else None
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
    sync_subsystems_transformer()
    if hasattr(stream, 'clear_broadcast_cache'):
        stream.clear_broadcast_cache()

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
    is_custom = getattr(st.session_state.transformer, 'is_custom_calibrated', False) or st.session_state.get('manual_calib_enabled', False)
    f_data = None
    if not force_live and st.session_state.exec_mode != "live" and st.session_state.precomputed_data is not None:
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

        # Standardize player detection caching to strictly raw pixel bounding boxes (u, v)
        if 'players' in f_data and f_data['players'] is not None:
            player_contracts = [
                PlayerDetection(
                    box=(int(round(p['box'][0])), int(round(p['box'][1])), int(round(p['box'][2])), int(round(p['box'][3]))),
                    foot_point=(float(p['foot_point'][0]), float(p['foot_point'][1])) if p.get('foot_point') is not None else None,
                    team_id=int(p.get('team_id', 0)),
                    confidence=float(p.get('confidence', 1.0)),
                    class_id=int(p.get('class_id', 0))
                )
                for p in f_data['players']
            ]
            ball_pixel = (float(f_data['ball_pixel'][0]), float(f_data['ball_pixel'][1])) if f_data.get('ball_pixel') is not None else None
        else:
            # Graceful fallback for legacy caches lacking raw player detections
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

        # Compute all pitch coordinates FORWARD strictly using active calibration:
        positions = stage2_projection(detections, st.session_state.transformer)

        cache_meta = st.session_state.get('precomputed_data') or {}
        cache_cam = cache_meta.get('camera_mode') or 'sideline'
        active_cam = getattr(st.session_state.transformer, 'camera_mode', 'sideline')
        cam_switched = bool(cache_cam != active_cam)
        recompute_geometry = is_custom or cam_switched

        if not is_custom and carrier is not None and len(positions.team_a) > 0:
            positions.carrier_idx = min(carrier, len(positions.team_a) - 1)
        if not is_custom and ball_pixel is None and ball is not None and not cam_switched:
            positions.ball = tuple(ball)

        if recompute_geometry:
            # Dynamically compute pitch geometry forward in ~4.9 ms (>200 FPS)
            geometry = stage3_geometry(positions, hull_algo=hull_algo)
        else:
            hull_player_indices = f_data.get('hull', {}).get('player_indices', [])
            if not hull_player_indices and len(hull_poly) >= 3 and positions.team_b:
                for pt in hull_poly:
                    matched_idx = -1
                    for idx, player_pt in enumerate(positions.team_b):
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
        elif hasattr(stream, 'detection_cache'):
            with getattr(stream, '_cache_lock', threading.Lock()):
                if frame_idx in stream.detection_cache:
                    stream.detection_cache.move_to_end(frame_idx)
                    detections = stream.detection_cache[frame_idx].get('detections')

        if detections is None:
            detections = stage1_perception(
                raw_frame,
                st.session_state.detector,
                frame_idx=frame_idx,
                transformer=st.session_state.transformer
            )
            if hasattr(stream, 'put_cached_detections'):
                stream.put_cached_detections(frame_idx, detections, raw_frame=raw_frame)
            elif hasattr(stream, 'detection_cache'):
                with getattr(stream, '_cache_lock', threading.Lock()):
                    if len(stream.detection_cache) >= 1000:
                        stream.detection_cache.popitem(last=False)
                    stream.detection_cache[frame_idx] = {'detections': detections, 'raw_frame': None}

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
    attack_dir = st.session_state.get('attack_direction', 'right')
    coach_data = evaluate_coach_analytics(
        positions.team_a, positions.team_b,
        {'hull': geometry.hull_polygon, 'area_sq_m': geometry.hull_area},
        vor_res_dict, geometry.dijkstra_res, None, positions.ball,
        attack_direction=attack_dir
    )

    # FW-06: Interactive Tactical Distance & Offside Ruler
    ruler_data = None
    if st.session_state.get('show_ruler', False):
        try:
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
                defending_players=team_b,
                attack_direction=attack_dir,
                ball_coord=getattr(positions, 'ball', None)
            )
        except Exception:
            ruler_data = None

    # Stage 4 Rendering for Execution Bundle Contract Compliance (Knowledge Base §3.2)
    show_h = st.session_state.get('show_hull', True)
    show_v = st.session_state.get('show_voronoi', True)
    show_d = st.session_state.get('show_dijkstra', True)
    show_g = st.session_state.get('show_passing_graph', False)

    renderer = getattr(st.session_state, 'renderer', None)
    if renderer is None:
        renderer = TacticalPitchRenderer(width=1000, height=650, padding=35)
        st.session_state.renderer = renderer

    overlay_engine = getattr(st.session_state, 'overlay_engine', None)
    if overlay_engine is None:
        overlay_engine = BroadcastOverlayEngine(getattr(st.session_state, 'transformer', None) or HomographyTransformer())
        st.session_state.overlay_engine = overlay_engine

    radar_canvas = stage4_render_radar(
        positions, geometry, renderer,
        show_voronoi=show_v, show_hull=show_h, show_dijkstra=show_d,
        show_passing_graph=show_g, ruler_data=ruler_data
    )
    broadcast_overlay = stage4_render_overlay(
        raw_frame, positions, geometry, overlay_engine,
        show_voronoi=show_v, show_hull=show_h, show_dijkstra=show_d,
        show_passing_graph=show_g, ruler_data=ruler_data,
        hull_algo_name="Graham Scan" if hull_algo == "graham" else "Jarvis March",
        pipeline_fps=1000.0 / max(latency, 1e-6),
        transformer=st.session_state.transformer
    )

    return {
        'raw_frame': raw_frame,
        'detections': detections,
        'positions': positions,
        'geometry': geometry,
        'coach_data': coach_data,
        'ruler_data': ruler_data,
        'radar_canvas': radar_canvas,
        'broadcast_overlay': broadcast_overlay,
        'onnx_telemetry': st.session_state.detector.get_onnx_telemetry() if hasattr(st.session_state, 'detector') and hasattr(st.session_state.detector, 'get_onnx_telemetry') else {},
        'diag_status': diag_status,
        'diag_issues': diag_issues,
        'latency_ms': latency,
        'is_cached': has_cache,
        'source_type': 'cache' if has_cache else 'broadcast_stream',
        'frame_idx': frame_idx,
        'ring_buffer': stream.get_ring_buffer_telemetry() if hasattr(stream, 'get_ring_buffer_telemetry') else {}
    }


# Canonical alias for pipeline execution bundle builder
build_pipeline_bundle = execute_pipeline
