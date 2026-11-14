"""
Interactive Pitch Mask Alignment Studio (Streamlit Component v2)
================================================================
Overlays a live-warping FIFA pitch chalk mask directly over the match video frame.
Combines flat perspective homography and curved panorama mesh warping into a single
customizable editor: choose 4, 8, 13, or 21 landmark presets, click "+ Add Point"
to drop unlimited extra landmarks anywhere on the pitch, drag handles with a 2.5x
magnifier loupe, and commit one unified H + TPS residual transform.
"""

import base64
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import streamlit as st

from perception.pitch_model import (
    PitchModel,
    get_pitch_landmarks_def,
    PITCH_LANDMARKS_DEF,
    MESH_HANDLES_DEF,
    LAYOUT_4_IDS,
    LAYOUT_4_GOAL_IDS,
    LAYOUT_8_IDS,
    LAYOUT_8_GOAL_IDS,
    LAYOUT_13_IDS,
    LAYOUT_21_IDS,
    LAYOUT_ENDZONE_IDS,
)

# Backward-compatible pitch dimension aliases
PITCH_LEN = PitchModel.LENGTH
PITCH_WID = PitchModel.WIDTH

from ui.components.pitch_mask_studio_assets import _HTML, _CSS, _JS

_PITCH_MASK_COMPONENT = st.components.v2.component(
    "tactical_pitch_mask_studio",
    html=_HTML,
    css=_CSS,
    js=_JS,
    isolate_styles=True,
)


def _preset_handles_from_transformer(
    transformer: Any,
    w: int,
    h: int,
    active_ids: Optional[List[str]] = None,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
) -> List[Dict[str, Any]]:
    """Projects the 27 canonical pitch landmarks into video pixels using a transformer."""
    act_set = set(active_ids if active_ids is not None else LAYOUT_4_IDS)
    landmarks_def = get_pitch_landmarks_def(pitch_length, pitch_width)
    metric_pts = [(float(d["xm"]), float(d["ym"])) for d in landmarks_def]
    px_pts = transformer.pitch_to_image(metric_pts, filter_behind_camera=False)
    out: List[Dict[str, Any]] = []
    for meta, px in zip(landmarks_def, px_pts):
        out.append({
            "id": meta["id"],
            "label": meta["label"],
            "xm": float(meta["xm"]),
            "ym": float(meta["ym"]),
            "role": meta["role"],
            "u": float(px[0]),
            "v": float(px[1]),
            "active": meta["id"] in act_set,
        })
    return out


def _build_pose_presets(
    w: int,
    h: int,
    current_transformer: Any,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
) -> Dict[str, List[Dict[str, Any]]]:
    """Builds starting handle poses cleanly across all 27 pitch landmarks."""
    from perception.homography import HomographyTransformer

    t_side = HomographyTransformer(w, h, camera_mode="sideline", pitch_length=pitch_length, pitch_width=pitch_width)
    t_end = HomographyTransformer(w, h, camera_mode="goal_to_goal", pitch_length=pitch_length, pitch_width=pitch_width)

    pano_corners_src = np.float32([
        [w * 0.08, h * 0.25],
        [w * 0.92, h * 0.25],
        [w * 0.97, h * 0.86],
        [w * 0.03, h * 0.86],
    ])
    pano_corners_dst = np.float32([[0.0, 0.0], [pitch_length, 0.0], [pitch_length, pitch_width], [0.0, pitch_width]])
    t_pano = HomographyTransformer(w, h, warp_mode="universal", lens_lambda=-0.10, pitch_length=pitch_length, pitch_width=pitch_width)
    t_pano.calibrate(
        pano_corners_src,
        pano_corners_dst,
        frame_width=w,
        frame_height=h,
        warp_mode="universal",
        lens_lambda=-0.10,
        pitch_length=pitch_length,
        pitch_width=pitch_width,
    )

    pano_handles = _preset_handles_from_transformer(t_pano, w, h, active_ids=LAYOUT_13_IDS, pitch_length=pitch_length, pitch_width=pitch_width)

    cur_is_wide = (w / float(max(1, h)) >= 2.2) or (getattr(current_transformer, "warp_mode", "planar") in ("universal", "panorama", "panorama_tps"))
    is_goal = getattr(current_transformer, "camera_mode", "sideline") == "goal_to_goal"
    if is_goal:
        cur_active = LAYOUT_ENDZONE_IDS
    elif cur_is_wide:
        cur_active = LAYOUT_13_IDS
    else:
        cur_active = LAYOUT_4_IDS

    return {
        "current": _preset_handles_from_transformer(current_transformer, w, h, active_ids=cur_active, pitch_length=pitch_length, pitch_width=pitch_width),
        "sideline": _preset_handles_from_transformer(t_side, w, h, active_ids=LAYOUT_8_IDS, pitch_length=pitch_length, pitch_width=pitch_width),
        "endzone": _preset_handles_from_transformer(t_end, w, h, active_ids=LAYOUT_ENDZONE_IDS, pitch_length=pitch_length, pitch_width=pitch_width),
        "panorama": pano_handles,
    }


def _merge_saved_handles(base_handles: List[Dict[str, Any]], saved_handles: Any) -> List[Dict[str, Any]]:
    """Merges arbitrary saved handle subsets and user-created custom points into base landmarks."""
    if not isinstance(saved_handles, list) or len(saved_handles) < 4:
        return base_handles

    has_active_flags = any("active" in item for item in saved_handles if isinstance(item, dict))
    by_id: Dict[str, Dict[str, Any]] = {}
    custom_items: List[Dict[str, Any]] = []
    known_ids = {item["id"] for item in base_handles}

    for item in saved_handles:
        if not isinstance(item, dict) or "id" not in item:
            continue
        hid = str(item["id"])
        if hid in known_ids:
            by_id[hid] = item
        else:
            custom_items.append({
                "id": hid,
                "label": str(item.get("label", hid)),
                "xm": float(item.get("xm", 52.5)),
                "ym": float(item.get("ym", 34.0)),
                "role": "custom",
                "u": float(item.get("u", 0.0)),
                "v": float(item.get("v", 0.0)),
                "active": bool(item.get("active", True)),
            })

    merged: List[Dict[str, Any]] = []
    for base in base_handles:
        copy_item = dict(base)
        if base["id"] in by_id:
            sv = by_id[base["id"]]
            copy_item["u"] = float(sv.get("u", copy_item["u"]))
            copy_item["v"] = float(sv.get("v", copy_item["v"]))
            copy_item["active"] = bool(sv.get("active", True)) if has_active_flags else True
        else:
            copy_item["active"] = False
        merged.append(copy_item)

    merged.extend(custom_items)
    if sum(1 for h in merged if h.get("active")) < 4:
        return base_handles
    return merged


def _encode_frame_data_url(raw_frame: np.ndarray, quality: int = 86) -> str:
    """Encodes a BGR frame cleanly into a base64 JPEG data URL for the browser canvas."""
    ok, buf = cv2.imencode(".jpg", raw_frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return ""
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def render_pitch_mask_studio(
    raw_frame: np.ndarray,
    transformer: Any,
    video_path: str = "",
    key_prefix: str = "mask_calib",
) -> Dict[str, Any]:
    """
    Renders the interactive Draggable Pitch Mask alignment editor over the video frame.
    Supports 4/8/13/21 handle presets, unlimited custom pitch points, Flat Least-Squares
    and Curved Panorama TPS modes, and per-video alignment persistence.
    """
    rf_h, rf_w = raw_frame.shape[:2]
    cur_length = float(getattr(transformer, "pitch_length", 105.0))
    cur_width = float(getattr(transformer, "pitch_width", 68.0))
    presets = _build_pose_presets(rf_w, rf_h, transformer, pitch_length=cur_length, pitch_width=cur_width)

    is_wide = rf_w / float(max(1, rf_h)) >= 2.2
    is_goal = getattr(transformer, "camera_mode", "sideline") == "goal_to_goal"
    if not getattr(transformer, "is_custom_calibrated", False):
        if is_wide:
            default_pose = presets["panorama"]
        elif is_goal:
            default_pose = presets["endzone"]
        else:
            default_pose = presets["current"]
    else:
        default_pose = presets["current"]
    active_handles = _merge_saved_handles(default_pose, st.session_state.get("tv_calib_mesh_handles"))

    n_active = sum(1 for h in active_handles if h.get("active"))
    cur_warp = getattr(transformer, "warp_mode", "planar")
    if cur_warp in ("panorama_tps", "tps") or (is_wide and n_active >= 6 and cur_warp not in ("planar", "universal")):
        force_fit = "tps"
    elif cur_warp == "universal":
        force_fit = "curve"
    elif cur_warp == "planar":
        force_fit = "flat"
    else:
        force_fit = "tps" if is_wide else "flat"

    cur_f = int(st.session_state.get("current_frame", 0))
    frame_cache_key = (video_path, cur_f, rf_w, rf_h)
    if st.session_state.get("_pm_cached_frame_key") != frame_cache_key:
        st.session_state["_pm_cached_frame_key"] = frame_cache_key
        st.session_state["_pm_cached_frame_url"] = _encode_frame_data_url(raw_frame)

    data_url = st.session_state.get("_pm_cached_frame_url", "")
    result = _PITCH_MASK_COMPONENT(
        key=f"{key_prefix}_pitch_mask_v2",
        data={
            "imageDataUrl": data_url,
            "vw": int(rf_w),
            "vh": int(rf_h),
            "cameraMode": getattr(transformer, "camera_mode", "sideline"),
            "handles": active_handles,
            "presets": presets,
            "layout4": LAYOUT_4_GOAL_IDS if is_goal else LAYOUT_4_IDS,
            "layout8": LAYOUT_8_GOAL_IDS if is_goal else LAYOUT_8_IDS,
            "layout13": LAYOUT_13_IDS,
            "layout21": LAYOUT_21_IDS,
            "warpMode": getattr(transformer, "warp_mode", "planar"),
            "lensLambda": float(getattr(transformer, "lens_lambda", -0.10 if is_wide else 0.0)),
            "forceFitMode": force_fit,
            "pitchLength": cur_length,
            "pitchWidth": cur_width,
        },
        on_alignment_change=lambda: None,
    )

    just_applied = False
    just_reset = False
    trigger_val = getattr(result, "alignment", None)
    last_ts_key = f"{key_prefix}_pm_last_ts"

    if isinstance(trigger_val, dict) and trigger_val.get("ts") and trigger_val.get("ts") != st.session_state.get(last_ts_key):
        st.session_state[last_ts_key] = trigger_val["ts"]
        action = trigger_val.get("action")
        if action == "apply":
            raw_h = [
                h for h in (trigger_val.get("handles") or [])
                if isinstance(h, dict) and "u" in h and "v" in h and "xm" in h and "ym" in h
                and h["u"] is not None and h["v"] is not None and h["xm"] is not None and h["ym"] is not None
            ]
            all_h = trigger_val.get("allHandles") or raw_h
            if len(raw_h) >= 4:
                src_pts = np.float32([[float(item["u"]), float(item["v"])] for item in raw_h])
                dst_pts = np.float32([[float(item["xm"]), float(item["ym"])] for item in raw_h])
                req_warp = str(trigger_val.get("warpMode", "auto"))
                req_lambda = float(trigger_val.get("lensLambda", 0.0))
                req_length = float(trigger_val.get("pitchLength", cur_length))
                req_width = float(trigger_val.get("pitchWidth", cur_width))
                transformer.calibrate(
                    src_pts,
                    dst_pts,
                    frame_width=rf_w,
                    frame_height=rf_h,
                    warp_mode=req_warp,
                    lens_lambda=req_lambda if req_warp == "universal" else None,
                    pitch_length=req_length,
                    pitch_width=req_width,
                )

                st.session_state["tv_calib_points"] = [(float(p[0]), float(p[1])) for p in src_pts]
                st.session_state["tv_calib_mesh_handles"] = all_h if len(all_h) >= 4 else None
                st.session_state["tv_calib_applied"] = True
                st.session_state["tv_calib_enabled"] = True
                st.session_state["tv_calib_warp_mode"] = transformer.warp_mode
                st.session_state["manual_calib_enabled"] = True
                st.session_state["manual_calib_applied"] = True
                st.session_state["manual_calib_points"] = [(float(p[0]), float(p[1])) for p in src_pts]

                if video_path:
                    from ui.media_library import save_video_calibration
                    save_video_calibration(video_path, {
                        "video_path": video_path,
                        "frame_width": int(rf_w),
                        "frame_height": int(rf_h),
                        "warp_mode": transformer.warp_mode,
                        "lens_lambda": float(getattr(transformer, "lens_lambda", req_lambda)),
                        "pitch_length": req_length,
                        "pitch_width": req_width,
                        "src_points": [[float(p[0]), float(p[1])] for p in src_pts],
                        "dst_points": [[float(p[0]), float(p[1])] for p in dst_pts],
                        "handles": all_h if len(all_h) >= 4 else None,
                    })

                if "transformer" in st.session_state and st.session_state.transformer is not transformer:
                    st.session_state.transformer = transformer

                just_applied = True
        elif action == "reset":
            st.session_state["tv_calib_points"] = []
            st.session_state["tv_calib_mesh_handles"] = None
            st.session_state["tv_calib_applied"] = False
            st.session_state["tv_calib_enabled"] = False
            st.session_state["manual_calib_applied"] = False
            st.session_state["manual_calib_enabled"] = False
            st.session_state["manual_calib_points"] = []
            from ui.state import get_active_source_info
            active_src = st.session_state.get("active_source_key", "world_cup")
            cam_mode = getattr(transformer, "camera_mode", None) or get_active_source_info(active_src).get("default_mode", "sideline")
            transformer.set_camera_mode(cam_mode)
            if video_path:
                from ui.media_library import delete_video_calibration
                delete_video_calibration(video_path)
            st.session_state.pop("_pm_cached_frame_key", None)
            just_reset = True

    is_custom = bool(getattr(transformer, "is_custom_calibrated", False))
    mean_err = None
    if is_custom and hasattr(transformer, "compute_reprojection_residuals"):
        res_info = transformer.compute_reprojection_residuals()
        if res_info.get("residuals"):
            mean_err = float(res_info["mean_error"])

    return {
        "applied": is_custom,
        "enabled": bool(st.session_state.get("manual_calib_enabled", is_custom)),
        "just_applied": just_applied,
        "just_reset": just_reset,
        "points": st.session_state.get("tv_calib_points", []),
        "mean_error_px": mean_err,
        "warp_mode": getattr(transformer, "warp_mode", "planar"),
        "pitch_length": float(getattr(transformer, "pitch_length", cur_length)),
        "pitch_width": float(getattr(transformer, "pitch_width", cur_width)),
    }
