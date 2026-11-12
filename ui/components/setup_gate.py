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
from typing import Dict, Any, Optional
import cv2
import numpy as np
import streamlit as st

from ui.state import VIDEO_SOURCES, get_raw_frame, invalidate_custom_user_cache
from perception.calibration_templates import (
    CALIBRATION_TEMPLATES,
    render_mouse_calibration_studio,
    get_template_dest_points
)
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
            <span>⚙️ Step 1: Tactical Setup & Camera Calibration Gate</span>
        </div>
        <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 4px;">
            Establish video footage input, camera planar homography, and execution acceleration mode before launching analysis.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_setup1, col_setup2 = st.columns([1.1, 1.2])

    with col_setup1:
        st.markdown("#### 1. Select Match Video Source")
        source_keys = list(VIDEO_SOURCES.keys())
        source_labels = [VIDEO_SOURCES[k]["title"] for k in source_keys]
        cur_src_idx = source_keys.index(st.session_state.active_source_key) if st.session_state.active_source_key in source_keys else 0

        selected_source_label = st.radio(
            "Match Video Stream",
            source_labels,
            index=cur_src_idx,
            key="setup_source_radio"
        )
        new_key = source_keys[source_labels.index(selected_source_label)]
        if new_key != st.session_state.active_source_key:
            st.session_state.active_source_key = new_key
            st.session_state.current_frame = 0
            st.rerun()

        # Handle custom upload
        if st.session_state.active_source_key == "custom_upload":
            custom_up = st.file_uploader(
                "Upload Custom Video (MP4 / MOV / AVI)",
                type=["mp4", "mov", "avi"],
                key="setup_file_uploader"
            )
            if custom_up is not None:
                upload_sig = compute_upload_signature(custom_up.name, custom_up.size, custom_up)

                if st.session_state.get('last_uploaded_sig') != upload_sig:
                    try:
                        target_path = "data/user_match.mp4"
                        custom_up.seek(0)
                        with open(target_path, "wb") as f:
                            shutil.copyfileobj(custom_up, f, length=4 * 1024 * 1024)
                        custom_up.seek(0)
                        st.session_state.last_uploaded_sig = upload_sig
                        st.session_state.last_uploaded_name = custom_up.name

                        # Invalidate stale precomputed cache, static symlink, and payload caches
                        invalidate_custom_user_cache(clear_user_cache_file=True)
                        static_dir = os.path.abspath("static")
                        static_video_path = os.path.join(static_dir, "active_match.mp4")
                        if os.path.exists(static_video_path) or os.path.islink(static_video_path):
                            try:
                                os.remove(static_video_path)
                            except Exception:
                                pass

                        st.session_state.precomputed_data = None
                        st.session_state.stream_manager.load_video_source(target_path)
                        if st.session_state.stream_manager.width > 0 and st.session_state.stream_manager.height > 0:
                            st.session_state.transformer.update_dimensions(
                                st.session_state.stream_manager.width,
                                st.session_state.stream_manager.height
                            )
                        st.session_state.stream_manager.transformer = st.session_state.transformer
                        st.session_state.overlay_engine.transformer = st.session_state.transformer
                        st.session_state.tv_calib_points = []
                        st.session_state.tv_calib_applied = False
                        st.session_state.tv_calib_just_applied = False
                        st.session_state.current_frame = 0
                        st.session_state.is_playing = False
                        st.success(f"Loaded: {custom_up.name} ({custom_up.size / (1024 * 1024):.1f} MB)")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error loading custom video: {e}")

        st.markdown("---")
        st.markdown("#### 3. Execution & Acceleration Mode")
        cached_n = telemetry.get("cached_count", 0)
        tot_f = telemetry.get("total_frames", 1)
        w_f = telemetry.get("width", 1280)
        h_f = telemetry.get("height", 720)
        fps_f = telemetry.get("fps", 25.0)

        cur_exec_idx = 0 if st.session_state.get('exec_mode') == 'cache' else (2 if st.session_state.get('exec_mode') == 'live' else 1)
        if cur_exec_idx == 0 and cached_n == 0:
            cur_exec_idx = 1

        exec_mode = st.radio(
            "Select Execution Strategy",
            [
                "⚡ Use Precomputed Disk Cache (Instant 120 FPS, Zero Laptop Heat)",
                "🚀 Run Full Multi-Core Precompute Now (with Progress Bar)",
                "● Pure Live On-the-Fly Compute Mode (Compute When Requested)"
            ],
            index=cur_exec_idx,
            key="setup_exec_mode_radio"
        )

        if exec_mode.startswith("⚡"):
            st.session_state.exec_mode = "cache"
            if cached_n > 0:
                st.success(f"✅ Precomputed cache ready: **{cached_n} of {tot_f} frames** ({int(cached_n/tot_f*100)}%). Instant ~120 FPS playback.")
            else:
                st.warning("⚠️ No precomputed cache file found for this video. Select 'Run Multi-Core Precompute Now' or 'Pure Live Mode'.")

        elif exec_mode.startswith("🚀"):
            st.session_state.exec_mode = "precompute"
            st.info("Uses multi-core parallel multiprocessing for DAA geometry with strictly capped ONNX neural perception.")
            cpu_count = os.cpu_count() or 4

            c_p1, c_p2 = st.columns([1, 1])
            with c_p1:
                workers = st.slider(
                    "CPU Worker Cores / ONNX Threads",
                    min_value=1,
                    max_value=cpu_count,
                    value=min(8, max(2, cpu_count // 2)),
                    step=1,
                    key="setup_workers_slider",
                    help="Strictly caps ONNX Runtime and geometry workers so your CPU does not thermal throttle."
                )
            with c_p2:
                model_options = {
                    "⚡ INT8 Quantized (models/best_int8.onnx — Fast 65MB)": "models/best_int8.onnx",
                    "🎯 Full Precision FP32 (models/best.onnx — 260MB)": "models/best.onnx",
                    "🚀 Ultra-Lightweight (models/yolov8n.onnx — 13MB)": "models/yolov8n.onnx",
                    "🔥 PyTorch Football Model (models/best.pt — 130MB)": "models/best.pt",
                    "⚡ PyTorch Nano (models/yolov8n.pt — 6.5MB)": "models/yolov8n.pt"
                }
                avail_models = {k: v for k, v in model_options.items() if os.path.exists(v)}
                if not avail_models:
                    avail_models = {"⚡ Default COCO Nano (yolov8n.pt)": "yolov8n.pt"}
                sel_model_label = st.selectbox(
                    "Neural Perception Model",
                    list(avail_models.keys()),
                    index=0,
                    key="setup_model_choice"
                )
                selected_model_path = avail_models[sel_model_label]

            c_stride1, c_stride2 = st.columns([1, 1])
            with c_stride1:
                stride_val = st.select_slider(
                    "Tactical Sampling Stride",
                    options=[1, 2, 3, 5],
                    value=2,
                    format_func=lambda s: f"Every {s} frame(s) (~{int(30/s)} FPS)" if s > 1 else "Every 1 frame (Full 30 FPS)",
                    key="setup_stride_slider",
                    help="Stride 2 (15 FPS) captures all tactical shifts while cutting precompute time in half!"
                )
            with c_stride2:
                max_f_choice = st.select_slider(
                    "Frame Range",
                    options=[180, 300, 600, tot_f],
                    value=min(300, tot_f),
                    format_func=lambda f: f"First {f} frames (~{int(f/30)}s)" if f < tot_f else f"Full Video ({tot_f} frames)",
                    key="setup_max_frames_slider"
                )

            if st.button("⚡ Execute Precompute Now", type="secondary", width="stretch", key="setup_run_precompute_btn"):
                prog_bar = st.progress(0.0)
                status_text = st.empty()
                status_text.text(f"Ingesting video frames with {os.path.basename(selected_model_path)} across {workers} capped CPU threads...")

                active_p = VIDEO_SOURCES[st.session_state.active_source_key]["path"]
                cache_p = VIDEO_SOURCES[st.session_state.active_source_key]["cache"]
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
                    status_text.text(f"Processing video: {int(p * 100)}% complete across {workers} CPU threads...")

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
                st.success(f"✅ Precomputation complete! Cached frames to disk.")
                st.rerun()


        else:
            # Pure live compute mode
            st.session_state.exec_mode = "live"
            st.session_state.precomputed_data = None
            st.info("● **Live Mode Active**: Neural perception and DAA geometry compute on-demand when scrubbing frames.")

    with col_setup2:
        st.markdown("#### 2. Camera Homography Calibration (Unified)")
        st.caption("Maps camera pixel coordinates (u, v) to metric FIFA pitch coordinates [0, 105]m x [0, 68]m.")

        calib_type = st.radio(
            "Calibration Approach",
            [
                "📌 Verified Camera Perspective Preset (Fast & Pre-calibrated)",
                "🎯 Interactive Multi-Point / Panorama Mouse Studio (4–12 Points)"
            ],
            index=1 if st.session_state.manual_calib_enabled else 0,
            key="setup_calib_type_radio"
        )
        st.session_state.manual_calib_enabled = calib_type.startswith("🎯")

        if not st.session_state.manual_calib_enabled:
            default_mode = VIDEO_SOURCES[st.session_state.active_source_key]["default_mode"]
            preset_labels = ["Sideline TV Broadcast Perspective", "Goal-to-Goal End-Zone Perspective"]
            cur_preset_idx = 0 if default_mode == "sideline" else 1

            preset_choice = st.selectbox(
                "Camera Perspective Angle",
                preset_labels,
                index=cur_preset_idx,
                key="setup_preset_select"
            )
            mode = "sideline" if "Sideline" in preset_choice else "goal_to_goal"
            st.session_state.transformer.set_camera_mode(mode)
            if w_f > 0 and h_f > 0:
                st.session_state.transformer.update_dimensions(w_f, h_f)
            st.session_state.stream_manager.transformer = st.session_state.transformer
            st.session_state.overlay_engine.transformer = st.session_state.transformer

            H = st.session_state.transformer.H
            cond_H = np.linalg.cond(H)
            st.success(f"✅ Verified `{mode.upper()}` perspective loaded (Matrix Condition κ={cond_H:.1f}).")
        else:
            st.info(
                "👇 **Full-Width Interactive Calibration Studio unlocked below.** "
                "Use the Zone Zoom tabs (`Left` / `Center` / `Right` / `Custom`) for 1:1 native pixel precision, "
                "or enable **Curved Panorama Mode (TPS)** for wide stitched videos."
            )

    if st.session_state.manual_calib_enabled:
        st.markdown("---")
        sample_frame = get_raw_frame(st.session_state.current_frame)
        if sample_frame is None:
            sample_frame = np.zeros((h_f, w_f, 3), dtype=np.uint8)

        calib_res = render_mouse_calibration_studio(
            raw_frame=sample_frame,
            transformer=st.session_state.transformer,
            key_prefix="gate_calib"
        )
        if calib_res.get("just_applied"):
            st.session_state.manual_calib_applied = True
            st.session_state.manual_calib_points = calib_res.get("points", [])
            st.session_state.overlay_engine.transformer = st.session_state.transformer
            st.session_state.stream_manager.transformer = st.session_state.transformer
            st.session_state.stream_manager.detection_cache.clear()
            st.session_state.precomputed_data = None
            invalidate_custom_user_cache(
                clear_user_cache_file=(st.session_state.active_source_key in ("custom_upload", "world_cup"))
            )

    st.markdown("---")
    # Launch CTA Button
    btn_col1, btn_col2, btn_col3 = st.columns([1, 1.8, 1])
    with btn_col2:
        if st.button("🚀 Confirm Configuration & Launch Full Tactical Suite", type="primary", width="stretch"):
            st.session_state.setup_confirmed = True
            st.rerun()

    return False
