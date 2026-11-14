"""
Tactical Pipeline Mandatory Setup Gate
======================================
Enforces precedence: User configures (1) Match Footage Source, (2) Camera Homography
Calibration, and (3) Execution Strategy (Precomputed Cache vs Multiprocess Run vs Live Compute)
before unlocking the full tactical suite.
"""

import os
import shutil
import hashlib
import threading
from typing import Dict, Any, Optional
import cv2
import numpy as np
import streamlit as st

from ui.state import (
    VIDEO_SOURCES,
    get_active_source_info,
    get_custom_cache_path,
    build_custom_upload_paths,
    list_available_custom_videos,
    get_raw_frame,
    invalidate_custom_user_cache,
    reset_widget_state,
    get_app_state,
)
from perception.calibration_templates import (
    CALIBRATION_TEMPLATES,
    get_template_dest_points
)
from ui.components.calibration_studio import render_mouse_calibration_studio
from ui.components.pitch_mask_studio import render_pitch_mask_studio
from perception.precompute import precompute_video_segment, load_precomputed_segment


def compute_upload_signature(filename: str, size: int, source: Any) -> str:
    """Computes a fast content fingerprint from file metadata and head/tail bytes.

    Accepts a seekable/readable stream (avoiding memoryview exports on UploadedFile)
    or a bytes/memoryview buffer.
    """
    hasher = hashlib.md5()
    if hasattr(source, "seek") and hasattr(source, "read"):
        pos = source.tell()
        try:
            if size > 131072:
                source.seek(0)
                hasher.update(source.read(65536))
                source.seek(max(0, size - 65536))
                hasher.update(source.read(65536))
            else:
                source.seek(0)
                hasher.update(source.read())
        finally:
            source.seek(pos)
    else:
        if len(source) > 131072:
            hasher.update(source[:65536])
            hasher.update(source[-65536:])
        else:
            hasher.update(source)
    return f"{filename}_{size}_{hasher.hexdigest()}"


def render_setup_gate(telemetry: Dict[str, Any]) -> bool:
    """
    Renders the setup configuration gate.
    Returns True if setup is confirmed and ready to launch, False otherwise.
    """
    st.markdown("""
    <div style="background: linear-gradient(135deg, #111827 0%, #1e293b 100%); border: 1px solid #1e293b; border-radius: 12px; padding: 18px 24px; margin-bottom: 20px;">
        <div style="font-size: 1.35rem; font-weight: 700; color: #ffffff; display: flex; align-items: center; gap: 10px;">
            <span>⚙️ Match Setup</span>
        </div>
        <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 4px;">
            Choose your match footage, align field markings if needed, and open the match workspace.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_setup1, col_setup2 = st.columns([1.1, 1.2])

    with col_setup1:
        st.markdown("#### 1. Match Video")
        source_keys = list(VIDEO_SOURCES.keys())
        source_labels = [VIDEO_SOURCES[k]["title"] for k in source_keys]
        cur_src_idx = source_keys.index(st.session_state.active_source_key) if st.session_state.active_source_key in source_keys else 0

        selected_source_label = st.radio(
            "Select Match Source",
            source_labels,
            index=cur_src_idx,
            key="setup_source_radio"
        )
        new_key = source_keys[source_labels.index(selected_source_label)]
        if new_key != st.session_state.active_source_key:
            st.session_state.active_source_key = new_key
            st.session_state.current_frame = 0
            reset_widget_state("setup_calib_type_radio", "setup_preset_select", "setup_exec_mode_radio")
            st.rerun()

        # Handle custom upload & persistent local library (never overwrites built-in videos or deletes caches)
        if st.session_state.active_source_key == "custom_upload":
            saved_videos = list_available_custom_videos()
            if saved_videos:
                cur_custom = st.session_state.get("custom_video_path") or ""
                norm_saved = [os.path.normpath(p).replace("\\", "/") for p in saved_videos]
                cur_norm = os.path.normpath(cur_custom).replace("\\", "/") if cur_custom else ""
                sel_idx = norm_saved.index(cur_norm) if cur_norm in norm_saved else 0

                def _fmt_custom_video(p: str) -> str:
                    name = os.path.basename(p)
                    try:
                        mb = os.path.getsize(p) / (1024 * 1024)
                    except OSError:
                        mb = 0.0
                    c_path = get_custom_cache_path(p)
                    tag = "Analyzed" if c_path and os.path.exists(c_path) else "Not Analyzed"
                    return f"{name} ({mb:.1f} MB) — {tag}"

                chosen_video = st.selectbox(
                    "Saved & Uploaded Videos",
                    norm_saved,
                    index=sel_idx,
                    format_func=_fmt_custom_video,
                    key="setup_custom_library_select",
                )
                if chosen_video and chosen_video != cur_norm:
                    st.session_state.custom_video_path = chosen_video
                    st.session_state.custom_cache_path = get_custom_cache_path(chosen_video)
                    invalidate_custom_user_cache(clear_user_cache_file=False)
                    st.session_state.precomputed_data = None
                    st.session_state.current_frame = 0
                    st.session_state.is_playing = False
                    st.session_state.tv_calib_points = []
                    st.session_state.tv_calib_applied = False
                    st.session_state.tv_calib_just_applied = False
                    reset_widget_state("setup_calib_type_radio", "setup_preset_select", "setup_exec_mode_radio")
                    st.rerun()

            custom_up = st.file_uploader(
                "Upload New Custom Video (MP4 / MOV / AVI)",
                type=["mp4", "mov", "avi"],
                key="setup_file_uploader"
            )
            if custom_up is not None:
                upload_sig = compute_upload_signature(custom_up.name, custom_up.size, custom_up)

                if st.session_state.get('last_uploaded_sig') != upload_sig:
                    if custom_up.size <= 0:
                        st.error("Uploaded video file is empty (0 bytes). Please upload a valid MP4 or MOV video.")
                    else:
                        try:
                            target_path, target_cache_path = build_custom_upload_paths(custom_up.name, upload_sig)
                            if not os.path.exists(target_path) or os.path.getsize(target_path) != custom_up.size:
                                custom_up.seek(0)
                                with open(target_path, "wb") as f:
                                    shutil.copyfileobj(custom_up, f, length=4 * 1024 * 1024)
                                custom_up.seek(0)

                            # Verify OpenCV decode capability before accepting upload
                            test_cap = cv2.VideoCapture(target_path)
                            can_read = False
                            if test_cap.isOpened():
                                ret_test, frame_test = test_cap.read()
                                if ret_test and frame_test is not None and frame_test.size > 0:
                                    can_read = True
                            test_cap.release()

                            if not can_read:
                                st.error("Uploaded file could not be decoded as a valid video. Please check video codec or format.")
                                if os.path.exists(target_path):
                                    try:
                                        os.remove(target_path)
                                    except OSError:
                                        pass
                            else:
                                st.session_state.last_uploaded_sig = upload_sig
                                st.session_state.last_uploaded_name = custom_up.name
                                st.session_state.custom_video_path = target_path
                                st.session_state.custom_cache_path = target_cache_path

                                # Only clear derived browser payloads; preserve World Cup & all other videos' caches
                                invalidate_custom_user_cache(clear_user_cache_file=False)
                                static_dir = os.path.abspath("static")
                                static_video_path = os.path.join(static_dir, "active_match.mp4")
                                if os.path.exists(static_video_path) or os.path.islink(static_video_path):
                                    try:
                                        os.remove(static_video_path)
                                    except Exception:
                                        pass

                                st.session_state.precomputed_data = None
                                st.session_state.stream_manager.load_video_source(target_path)
                                st.session_state.transformer.set_camera_mode("sideline")
                                if st.session_state.stream_manager.width > 0 and st.session_state.stream_manager.height > 0:
                                    st.session_state.transformer.update_dimensions(
                                        st.session_state.stream_manager.width,
                                        st.session_state.stream_manager.height
                                    )
                                st.session_state.tv_calib_points = []
                                st.session_state.tv_calib_applied = False
                                st.session_state.tv_calib_just_applied = False
                                st.session_state.tv_calib_mesh_handles = None
                                st.session_state.manual_calib_enabled = False
                                st.session_state.manual_calib_applied = False
                                reset_widget_state("setup_calib_type_radio", "setup_preset_select", "setup_exec_mode_radio")
                                for _pm_k in [k for k in list(st.session_state.keys()) if k.endswith("_pm_cached_frame_key")]:
                                    reset_widget_state(_pm_k)
                                st.session_state.current_frame = 0
                                st.session_state.is_playing = False
                                st.toast(f"Saved & loaded: {custom_up.name} ({custom_up.size / (1024 * 1024):.1f} MB)", icon="🎬")
                                st.rerun()
                        except Exception as e:
                            st.error(f"Error loading custom video: {e}")

        st.markdown("---")
        st.markdown("#### 3. Playback & Analysis Mode")
        cached_n = telemetry.get("cached_count", 0)
        tot_f = telemetry.get("total_frames", 1)
        w_f = telemetry.get("width", 1280)
        h_f = telemetry.get("height", 720)
        fps_f = telemetry.get("fps", 25.0)

        cur_exec_idx = 0 if st.session_state.get('exec_mode') == 'cache' else (2 if st.session_state.get('exec_mode') == 'live' else 1)
        if cur_exec_idx == 0 and cached_n == 0:
            cur_exec_idx = 1

        exec_mode = st.radio(
            "Processing Option",
            [
                "⚡ Instant Replay (Saved Analysis)",
                "🔄 Analyze Full Clip Now",
                "● Live Frame-by-Frame Mode"
            ],
            index=cur_exec_idx,
            key="setup_exec_mode_radio"
        )

        if exec_mode.startswith("⚡"):
            st.session_state.exec_mode = "cache"
            if cached_n > 0:
                st.success(f"✅ Match analysis ready: **{cached_n} of {tot_f} frames** ({int(cached_n/tot_f*100)}%).")
            else:
                st.warning("⚠️ This clip has not been analyzed yet. Choose 'Analyze Full Clip Now' or 'Live Frame-by-Frame Mode'.")

        elif exec_mode.startswith("🔄"):
            st.session_state.exec_mode = "precompute"
            cpu_count = os.cpu_count() or 4

            c_stride1, c_stride2 = st.columns([1, 1])
            with c_stride1:
                stride_val = st.select_slider(
                    "Sampling Rate",
                    options=[1, 2, 3, 5],
                    value=2,
                    format_func=lambda s: "Full (30 FPS)" if s == 1 else f"Every {s} frames (~{max(1, int(round(fps_f/s)))} FPS)",
                    key="setup_stride_slider",
                    help="Sampling every 2 frames cuts analysis time in half while tracking all tactical movement."
                )
            with c_stride2:
                range_options = sorted(list({int(x) for x in [180, 300, 600, tot_f] if x > 0 and x <= tot_f} or {tot_f}))
                def_val = 300 if 300 in range_options else range_options[-1]
                max_f_choice = st.select_slider(
                    "Clip Range",
                    options=range_options,
                    value=def_val,
                    format_func=lambda f: f"First {f} frames (~{max(1, int(round(f/max(1.0, fps_f))))}s)" if f < tot_f else f"Full Clip ({tot_f} frames)",
                    key="setup_max_frames_slider"
                )

            with st.expander("Advanced Analysis Options", expanded=False):
                c_p1, c_p2 = st.columns([1, 1])
                with c_p1:
                    workers = st.slider(
                        "CPU Workers",
                        min_value=1,
                        max_value=cpu_count,
                        value=min(8, max(2, cpu_count // 2)),
                        step=1,
                        key="setup_workers_slider",
                        help="Parallel CPU workers used when preparing match analysis."
                    )
                with c_p2:
                    model_options = {
                        "⚡ Fast & Balanced (Recommended)": "models/best_int8.onnx",
                        "🎯 Maximum Detection Precision": "models/best.onnx",
                        "🚀 Ultra Lightweight": "models/yolov8n.onnx",
                        "🔥 High-Accuracy Football Model": "models/best.pt",
                        "⚡ Standard Lightweight Model": "models/yolov8n.pt"
                    }
                    avail_models = {k: v for k, v in model_options.items() if os.path.exists(v)}
                    if not avail_models:
                        fallback_model = "models/yolov8n.pt" if os.path.exists("models/yolov8n.pt") else "yolov8n.pt"
                        avail_models = {"⚡ Standard Player Detector": fallback_model}
                    sel_model_label = st.selectbox(
                        "Detection Quality",
                        list(avail_models.keys()),
                        index=0,
                        key="setup_model_choice"
                    )
                    selected_model_path = avail_models[sel_model_label]

            if st.button("🔄 Analyze Clip Now", type="secondary", width="stretch", key="setup_run_precompute_btn"):
                active_info = get_active_source_info(st.session_state.active_source_key)
                active_p = active_info["path"]
                cache_p = active_info["cache"]
                if not active_p or not os.path.exists(active_p):
                    st.error("Please select or upload a valid video before analyzing.")
                else:
                    prog_bar = st.progress(0.0)
                    status_text = st.empty()
                    status_text.text("Analyzing players, pitch positions, and passing lanes...")

                    is_manual = (
                        st.session_state.get('manual_calib_enabled', False)
                        and getattr(st.session_state.transformer, 'is_custom_calibrated', False)
                        and hasattr(st.session_state.transformer, 'src_points')
                        and len(getattr(st.session_state.transformer, 'src_points', [])) >= 4
                    )
                    custom_src = st.session_state.transformer.src_points if is_manual else None
                    custom_dst = st.session_state.transformer.dst_points if is_manual else None
                    cur_cam_mode = "manual_mouse" if is_manual else st.session_state.transformer.camera_mode

                    def update_prog(p: float):
                        prog_bar.progress(p)
                        status_text.text(f"Analyzing clip: {int(p * 100)}% complete...")

                    precompute_video_segment(
                        video_path=active_p,
                        output_cache_path=cache_p,
                        max_frames=max_f_choice,
                        stride=stride_val,
                        num_workers=workers,
                        camera_mode=cur_cam_mode,
                        model_name=selected_model_path,
                        progress_callback=update_prog,
                        custom_src_points=custom_src,
                        custom_dst_points=custom_dst,
                        warp_mode=getattr(st.session_state.transformer, 'warp_mode', 'planar')
                    )
                    st.session_state.precomputed_data = load_precomputed_segment(cache_p)
                    st.session_state.exec_mode = "cache"
                    if "setup_exec_mode_radio" in st.session_state:
                        del st.session_state["setup_exec_mode_radio"]
                    st.toast("✅ Match analysis complete and saved.", icon="✅")
                    st.rerun()

        else:
            st.session_state.exec_mode = "live"
            st.session_state.precomputed_data = None
            st.info("● **Live Frame-by-Frame Mode**: Analyzes each frame on demand as you scrub.")

    with col_setup2:
        st.markdown("#### 2. Pitch Alignment")
        st.caption("Match camera view to real-world field dimensions.")

        is_custom_calibrated = getattr(st.session_state.transformer, "is_custom_calibrated", False) or st.session_state.manual_calib_enabled
        radio_opts = [
            "📌 Automatic Preset (Recommended)",
            "🎯 Interactive Pitch Mask Alignment"
        ]
        if "setup_calib_type_radio" not in st.session_state:
            st.session_state["setup_calib_type_radio"] = radio_opts[1] if is_custom_calibrated else radio_opts[0]

        calib_type = st.radio(
            "Alignment Method",
            radio_opts,
            key="setup_calib_type_radio"
        )
        st.session_state.manual_calib_enabled = calib_type.startswith("🎯")

        if not st.session_state.manual_calib_enabled:
            default_mode = get_active_source_info(st.session_state.active_source_key)["default_mode"]
            preset_labels = ["Sideline TV Broadcast", "Goal-to-Goal End Zone"]
            cur_preset_idx = 0 if default_mode == "sideline" else 1

            preset_choice = st.selectbox(
                "Camera View",
                preset_labels,
                index=cur_preset_idx,
                key="setup_preset_select"
            )
            mode = "sideline" if "Sideline" in preset_choice else "goal_to_goal"
            st.session_state.transformer.set_camera_mode(mode)
            if w_f > 0 and h_f > 0:
                st.session_state.transformer.update_dimensions(w_f, h_f)
            from ui.state import sync_subsystems_transformer
            sync_subsystems_transformer()

            friendly_mode = "Sideline Broadcast" if mode == "sideline" else "Goal-to-Goal"
            st.success(f"✅ {friendly_mode} alignment active.")
        else:
            st.info(
                "👇 **Pitch Mask Studio opened below.** "
                "Drag handles to fit field lines, choose `4` / `8` / `13` / `21` points, "
                "or click **+ Add Point** to drop extra anchors anywhere on the pitch."
            )

    if st.session_state.manual_calib_enabled:
        st.markdown("---")
        sample_frame = get_raw_frame(st.session_state.current_frame)
        if sample_frame is None:
            sample_frame = np.zeros((h_f, w_f, 3), dtype=np.uint8)

        active_video_path = get_active_source_info(st.session_state.active_source_key).get("path", "")
        calib_res = render_pitch_mask_studio(
            raw_frame=sample_frame,
            transformer=st.session_state.transformer,
            video_path=active_video_path,
            key_prefix="gate_calib_mask"
        )
        with st.expander("Advanced: Point-by-Point Coordinate Picker", expanded=False):
            pt_res = render_mouse_calibration_studio(
                raw_frame=sample_frame,
                transformer=st.session_state.transformer,
                key_prefix="gate_calib",
                video_path=active_video_path
            )
            if pt_res.get("just_applied"):
                calib_res = pt_res

        if calib_res.get("just_applied"):
            st.session_state.manual_calib_applied = True
            st.session_state.manual_calib_enabled = True
            st.session_state.manual_calib_points = calib_res.get("points", [])
            from ui.state import sync_subsystems_transformer
            sync_subsystems_transformer()
            if hasattr(st.session_state.stream_manager, 'clear_all_caches'):
                st.session_state.stream_manager.clear_all_caches()
            elif hasattr(st.session_state.stream_manager, 'clear_broadcast_cache'):
                st.session_state.stream_manager.clear_broadcast_cache()
            elif hasattr(st.session_state.stream_manager, 'detection_cache'):
                with getattr(st.session_state.stream_manager, '_cache_lock', threading.Lock()):
                    st.session_state.stream_manager.detection_cache.clear()
            st.session_state.precomputed_data = None
            invalidate_custom_user_cache(clear_user_cache_file=False)
            st.rerun()
        elif calib_res.get("just_reset"):
            st.session_state.manual_calib_applied = False
            st.session_state.manual_calib_enabled = False
            st.session_state.tv_calib_applied = False
            st.session_state.tv_calib_enabled = False
            st.session_state.manual_calib_points = []
            st.session_state.tv_calib_points = []
            st.session_state.tv_calib_mesh_handles = None
            reset_widget_state("setup_calib_type_radio", "setup_preset_select")
            if hasattr(st.session_state.stream_manager, 'clear_all_caches'):
                st.session_state.stream_manager.clear_all_caches()
            elif hasattr(st.session_state.stream_manager, 'detection_cache'):
                with getattr(st.session_state.stream_manager, '_cache_lock', threading.Lock()):
                    st.session_state.stream_manager.detection_cache.clear()
            st.session_state.precomputed_data = None
            invalidate_custom_user_cache(clear_user_cache_file=False)
            st.rerun()

    st.markdown("---")
    # Launch CTA Button
    btn_col1, btn_col2, btn_col3 = st.columns([1, 1.8, 1])
    with btn_col2:
        active_p = telemetry.get("path") or get_active_source_info(st.session_state.active_source_key).get("path", "")
        has_valid_source = bool(active_p and os.path.exists(active_p))
        if st.button("🚀 Open Match Workspace", type="primary", width="stretch", disabled=not has_valid_source):
            st.session_state.setup_confirmed = True
            st.rerun()

    return False
