"""
Interactive Multi-Point & Panorama Calibration Studio UI
========================================================
Decoupled Streamlit component layer separated from pure geometry routines in
`perception/calibration_templates.py`. Organizes state initialization, anchor
builder presets, ROI native-pixel zoom canvas, and reprojection diagnostics into
focused helper functions.
"""

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from perception.calibration_templates import (
    CALIBRATION_TEMPLATES,
    FIFA_CHALK_LANDMARKS,
    GROUND_TRUTH_TEMPLATES,
    compute_calibration_roi_crop,
    map_roi_click_to_native,
    render_landmarks_pitch_thumbnail,
    render_pitch_projection_preview,
)

_DEFAULT_ANCHOR_CATALOG = [
    "Corner Flag — Left Far (Top-Left) [0m, 0m]",
    "Left 18-Yard Box — Infield Far Corner [16.5m, 13.84m]",
    "Left 18-Yard Box — Infield Near Corner [16.5m, 54.16m]",
    "Halfway Line — Far Touchline (Top) [52.5m, 0m]",
    "Center Spot [52.5m, 34m]",
    "Halfway Line — Near Touchline (Bottom) [52.5m, 68m]",
    "Right 18-Yard Box — Infield Far Corner [88.5m, 13.84m]",
    "Right 18-Yard Box — Infield Near Corner [88.5m, 54.16m]",
    "Corner Flag — Right Far (Top-Right) [105m, 0m]",
    "Corner Flag — Right Near (Bottom-Right) [105m, 68m]",
    "Corner Flag — Left Near (Bottom-Left) [0m, 68m]",
    "Right Penalty Arc 'D' — Infield Apex [84.85m, 34m]",
]

_CANON_KEYS = {
    "pts": "tv_calib_points",
    "applied": "tv_calib_applied",
    "enabled": "tv_calib_enabled",
    "just_applied": "tv_calib_just_applied",
    "mode": "tv_calib_picker_mode",
    "tmpl": "tv_calib_template",
    "warp": "tv_calib_warp_mode",
    "anchors": "tv_calib_num_anchors",
}


def coerce_anchor_count(val: Any, default: int = 4) -> int:
    """Safely converts Streamlit widget or session anchor counts into Python int."""
    try:
        if isinstance(val, bool):
            return default
        if isinstance(val, int):
            return val
        if isinstance(val, str):
            digits = "".join(ch for ch in val.split()[0] if ch.isdigit()) if val.strip() else ""
            if digits:
                return int(digits)
        return int(val)
    except Exception:
        return default


def _init_calib_session_keys(transformer: Any, key_prefix: str) -> Tuple[str, str, str]:
    """Ensures shared canonical and per-tab widget session keys exist cleanly."""
    click_key = f"{key_prefix}_last_click"
    time_key = f"{key_prefix}_last_click_time"
    ver_key = f"{key_prefix}_canvas_ver"

    if _CANON_KEYS["pts"] not in st.session_state:
        if getattr(transformer, "is_custom_calibrated", False) and getattr(transformer, "src_points", None) is not None:
            st.session_state[_CANON_KEYS["pts"]] = [(float(p[0]), float(p[1])) for p in transformer.src_points]
        else:
            st.session_state[_CANON_KEYS["pts"]] = []
    if _CANON_KEYS["applied"] not in st.session_state:
        st.session_state[_CANON_KEYS["applied"]] = bool(getattr(transformer, "is_custom_calibrated", False))
    if _CANON_KEYS["enabled"] not in st.session_state:
        st.session_state[_CANON_KEYS["enabled"]] = bool(st.session_state.get("manual_calib_enabled", False))
    if _CANON_KEYS["just_applied"] not in st.session_state:
        st.session_state[_CANON_KEYS["just_applied"]] = False
    if click_key not in st.session_state:
        st.session_state[click_key] = None
    if time_key not in st.session_state:
        st.session_state[time_key] = 0
    if ver_key not in st.session_state:
        st.session_state[ver_key] = 0
    if _CANON_KEYS["tmpl"] not in st.session_state:
        st.session_state[_CANON_KEYS["tmpl"]] = list(GROUND_TRUTH_TEMPLATES.keys())[0]
    if _CANON_KEYS["warp"] not in st.session_state:
        st.session_state[_CANON_KEYS["warp"]] = getattr(transformer, "warp_mode", "planar")
    if _CANON_KEYS["anchors"] not in st.session_state:
        st.session_state[_CANON_KEYS["anchors"]] = 8 if st.session_state[_CANON_KEYS["warp"]] == "panorama_tps" else 4
    else:
        st.session_state[_CANON_KEYS["anchors"]] = coerce_anchor_count(st.session_state[_CANON_KEYS["anchors"]], 4)

    return click_key, time_key, ver_key


def _clear_lm_widget_keys(key_prefix: str) -> None:
    """Drops widget keys before rerun so programmatic preset updates take effect."""
    for i in range(1, 13):
        st.session_state.pop(f"{key_prefix}_l{i}", None)
    st.session_state.pop(f"{key_prefix}_num_anchors_widget", None)
    st.session_state.pop(f"{key_prefix}_warp_mode_radio", None)


def _apply_landmark_preset(landmarks: List[str], num_pts: int, warp_mode: str, key_prefix: str, ver_key: str) -> None:
    """Applies a quick landmark layout preset and clears clicked pixel points."""
    for idx_l, lm_name in enumerate(landmarks):
        st.session_state[f"tv_calib_l{idx_l + 1}"] = lm_name
    st.session_state[_CANON_KEYS["anchors"]] = num_pts
    st.session_state[_CANON_KEYS["warp"]] = warp_mode
    st.session_state[_CANON_KEYS["pts"]] = []
    st.session_state[_CANON_KEYS["applied"]] = False
    st.session_state[_CANON_KEYS["just_applied"]] = False
    st.session_state[ver_key] = st.session_state.get(ver_key, 0) + 1
    _clear_lm_widget_keys(key_prefix)
    st.rerun()


def _render_custom_landmark_builder(
    active_warp_mode: str,
    landmark_names: List[str],
    key_prefix: str,
    ver_key: str,
) -> Tuple[np.ndarray, List[str], str, str, bool, bool]:
    """Renders custom multi-point anchor count selector, quick presets, and landmark dropdowns."""
    anchors_key = _CANON_KEYS["anchors"]
    pts_key = _CANON_KEYS["pts"]
    anchor_choices = [6, 8, 10, 12] if active_warp_mode == "panorama_tps" else [4, 5, 6, 8, 10, 12]
    cur_n = coerce_anchor_count(
        st.session_state.get(anchors_key, 8 if active_warp_mode == "panorama_tps" else 4),
        8 if active_warp_mode == "panorama_tps" else 4,
    )
    if cur_n not in anchor_choices:
        cur_n = anchor_choices[0]
        st.session_state[anchors_key] = cur_n

    widget_n_key = f"{key_prefix}_num_anchors_widget"
    if widget_n_key in st.session_state and st.session_state[widget_n_key] not in anchor_choices:
        cur_n = coerce_anchor_count(st.session_state[widget_n_key], cur_n)
        if cur_n not in anchor_choices:
            cur_n = anchor_choices[0]
        st.session_state[anchors_key] = cur_n
        st.session_state.pop(widget_n_key, None)

    n_col1, n_col2 = st.columns([1.8, 2.2])
    with n_col1:
        selected_n = st.radio(
            "Number of Anchors",
            anchor_choices,
            index=anchor_choices.index(cur_n),
            horizontal=True,
            key=widget_n_key,
        )
        num_anchors = coerce_anchor_count(selected_n, cur_n)
        if num_anchors not in anchor_choices:
            num_anchors = cur_n
        st.session_state[anchors_key] = num_anchors
    with n_col2:
        if active_warp_mode == "panorama_tps":
            solver_hint = f"🌀 <b>Wide-Angle Lens Fit</b> ({num_anchors} points spread across the pitch)"
        elif num_anchors == 4:
            solver_hint = "⚡ <b>4-Point Field Fit</b> (Standard flat perspective)"
        else:
            solver_hint = f"🛡️ <b>Multi-Point Field Fit</b> ({num_anchors} points for higher accuracy)"
        st.markdown(f"<div style='font-size:0.80rem; color:#cbd5e1; margin-top:26px;'>{solver_hint}</div>", unsafe_allow_html=True)

    q_c1, q_c2, q_c3, q_c4, q_c5 = st.columns(5)
    with q_c1:
        if st.button("🌀 Panorama 8-Pt Spread", key=f"{key_prefix}_q_pano8", width="stretch",
                     help="8 anchors spread left-to-right across the entire pitch (recommended for stitched panoramas)"):
            _apply_landmark_preset(_DEFAULT_ANCHOR_CATALOG[:8], 8, "panorama_tps", key_prefix, ver_key)
    with q_c2:
        if st.button("⚡ Right Box + 'D' (4)", key=f"{key_prefix}_q_rbox", width="stretch"):
            _apply_landmark_preset([landmark_names[0], landmark_names[2], landmark_names[3], landmark_names[4]], 4, "planar", key_prefix, ver_key)
    with q_c3:
        if st.button("⚡ Midfield Circle (4)", key=f"{key_prefix}_q_mid", width="stretch"):
            _apply_landmark_preset([
                "Halfway Line — Far Touchline (Top) [52.5m, 0m]",
                "Center Circle — Right Apex [61.65m, 34m]",
                "Halfway Line — Near Touchline (Bottom) [52.5m, 68m]",
                "Center Circle — Left Apex [43.35m, 34m]",
            ], 4, "planar", key_prefix, ver_key)
    with q_c4:
        if st.button("⚡ Left Box + 'D' (4)", key=f"{key_prefix}_q_lbox", width="stretch"):
            _apply_landmark_preset([
                "Left 18-Yard Box — Goal Line Far [0m, 13.84m]",
                "Left 18-Yard Box — Infield Far Corner [16.5m, 13.84m]",
                "Left Penalty Arc 'D' — Infield Apex [20.15m, 34m]",
                "Left 18-Yard Box — Goal Line Near [0m, 54.16m]",
            ], 4, "planar", key_prefix, ver_key)
    with q_c5:
        if st.button("⚡ 4 Pitch Corners", key=f"{key_prefix}_q_flags", width="stretch"):
            _apply_landmark_preset([
                "Corner Flag — Left Far (Top-Left) [0m, 0m]",
                "Corner Flag — Right Far (Top-Right) [105m, 0m]",
                "Corner Flag — Right Near (Bottom-Right) [105m, 68m]",
                "Corner Flag — Left Near (Bottom-Left) [0m, 68m]",
            ], 4, "planar", key_prefix, ver_key)

    cols_per_row = 4 if num_anchors in (4, 8, 12) else (3 if num_anchors == 6 else 5)
    current_lms: List[str] = []
    for row_start in range(0, num_anchors, cols_per_row):
        row_cols = st.columns(min(cols_per_row, num_anchors - row_start))
        for c_idx, col in enumerate(row_cols):
            lm_idx = row_start + c_idx
            canon_l_key = f"tv_calib_l{lm_idx + 1}"
            widget_l_key = f"{key_prefix}_l{lm_idx + 1}"
            def_val = _DEFAULT_ANCHOR_CATALOG[lm_idx % len(_DEFAULT_ANCHOR_CATALOG)]
            cur_val = st.session_state.get(canon_l_key, def_val)
            if cur_val not in landmark_names:
                cur_val = def_val
            sel_idx = landmark_names.index(cur_val)
            with col:
                sel = st.selectbox(f"Landmark {lm_idx + 1} (L{lm_idx + 1})", landmark_names, index=sel_idx, key=widget_l_key)
                st.session_state[canon_l_key] = sel
                current_lms.append(sel)

    last_lms_key = f"{key_prefix}_last_lms"
    last_lms = st.session_state.get(last_lms_key)
    if last_lms is not None and last_lms != current_lms:
        if st.session_state.get(pts_key):
            st.session_state[pts_key] = []
            st.session_state[_CANON_KEYS["applied"]] = False
            st.session_state[_CANON_KEYS["just_applied"]] = False
            st.session_state[ver_key] = st.session_state.get(ver_key, 0) + 1
            st.toast("🔄 Landmark configuration updated. Clicked points reset.", icon="🎯")
    st.session_state[last_lms_key] = current_lms

    has_duplicates = len(set(current_lms)) < num_anchors
    raw_pts = [FIFA_CHALK_LANDMARKS[s] for s in current_lms]

    from perception.homography import HomographyTransformer
    has_collinear, collinear_triplets = HomographyTransformer.check_collinearity(raw_pts, threshold_area=0.5)

    if has_duplicates:
        st.warning(f"⚠️ Duplicate landmarks selected! All {num_anchors} landmark anchors must be distinct pitch features.")
    elif has_collinear and num_anchors == 4:
        t = collinear_triplets[0]
        st.error(f"⚠️ Collinear Warning: Points L{t[0]+1}, L{t[1]+1}, and L{t[2]+1} lie on the exact same straight line! 4-point homography requires non-collinear quad vertices.")

    dst_points = np.float32(raw_pts)
    zone_title = f"Custom {num_anchors}-Point {'Panorama TPS' if active_warp_mode == 'panorama_tps' else 'Planar'} Anchor"
    zone_desc = f"Anchoring L1 ➔ L{num_anchors} across your selected physical field landmarks."
    return dst_points, current_lms, zone_title, zone_desc, has_duplicates, has_collinear


def _render_roi_zoom_canvas(
    raw_frame: np.ndarray,
    dst_points: np.ndarray,
    landmark_labels: List[str],
    key_prefix: str,
    time_key: str,
    ver_key: str,
) -> None:
    """Renders the full-width 1:1 native ROI zoom selector and clickable coordinate canvas."""
    from streamlit_image_coordinates import streamlit_image_coordinates

    pts_key = _CANON_KEYS["pts"]
    rf_h, rf_w = raw_frame.shape[:2]
    target_count = len(dst_points)
    pts_n = len(st.session_state[pts_key])
    next_lbl = f"L{pts_n+1}: {landmark_labels[pts_n]}" if pts_n < target_count else "All landmarks captured!"
    st.markdown(f"###### 📷 Click on Video Frame — Next Point: `{next_lbl}`")

    roi_options = [
        "🌐 Full Pitch",
        "⬅️ Left",
        "🎯 Center",
        "➡️ Right",
        "🔎 Custom Zoom",
    ]
    roi_choice = st.radio(
        "Zoom Region",
        roi_options,
        index=0,
        horizontal=True,
        key=f"{key_prefix}_roi_zone_radio",
        help="Zoom into a section of the pitch for easier clicking.",
    )

    pan_x_pct, pan_y_pct, zoom_val = 0.5, 0.5, 1.0
    if roi_choice.startswith("🌐"):
        z_mode = "full"
    elif roi_choice.startswith("⬅️"):
        z_mode = "left"
    elif roi_choice.startswith("🎯"):
        z_mode = "center"
    elif roi_choice.startswith("➡️"):
        z_mode = "right"
    else:
        z_mode = "custom"
        z_c1, z_c2, z_c3 = st.columns(3)
        with z_c1:
            zoom_val = st.slider("Zoom Factor", min_value=1.0, max_value=4.0, value=2.5, step=0.25, key=f"{key_prefix}_zoom_f")
        with z_c2:
            pan_x_pct = st.slider("Horizontal Pan (%)", min_value=0, max_value=100, value=50, step=2, key=f"{key_prefix}_pan_x") / 100.0
        with z_c3:
            pan_y_pct = st.slider("Vertical Pan (%)", min_value=0, max_value=100, value=55, step=2, key=f"{key_prefix}_pan_y") / 100.0

    roi_box = compute_calibration_roi_crop(rf_w, rf_h, z_mode, pan_x_pct=pan_x_pct, pan_y_pct=pan_y_pct, zoom_factor=zoom_val)
    x0, y0, x1, y1 = roi_box
    crop_w, crop_h = max(1, x1 - x0), max(1, y1 - y0)
    roi_frame = raw_frame[y0:y1, x0:x1]

    max_disp_w = 1920
    if crop_w > max_disp_w:
        disp_w = max_disp_w
        disp_h = max(1, int(round(crop_h * (max_disp_w / float(crop_w)))))
        c_canvas = cv2.resize(roi_frame, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
    else:
        disp_w = crop_w
        disp_h = crop_h
        c_canvas = roi_frame.copy()

    disp_scale_x = disp_w / float(crop_w)
    disp_scale_y = disp_h / float(crop_h)

    colors = [
        (0, 240, 255), (0, 255, 120), (255, 0, 200), (255, 200, 0),
        (255, 150, 50), (100, 255, 255), (200, 100, 255), (255, 255, 100),
    ]
    pts_disp_all = []
    for idx, pt in enumerate(st.session_state[pts_key]):
        dx = int(round((pt[0] - x0) * disp_scale_x))
        dy = int(round((pt[1] - y0) * disp_scale_y))
        pts_disp_all.append((dx, dy))
        if -20 <= dx <= disp_w + 20 and -20 <= dy <= disp_h + 20:
            col = colors[idx % len(colors)]
            cv2.drawMarker(c_canvas, (dx, dy), col, markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2, line_type=cv2.LINE_AA)
            cv2.circle(c_canvas, (dx, dy), 7, col, -1, lineType=cv2.LINE_AA)
            cv2.circle(c_canvas, (dx, dy), 10, (255, 255, 255), 2, lineType=cv2.LINE_AA)
            cv2.putText(
                c_canvas,
                f"L{idx+1} ({pt[0]:.0f},{pt[1]:.0f})",
                (dx + 12, max(16, dy - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    if target_count == 4 and len(pts_disp_all) >= 2:
        is_closed = (len(pts_disp_all) == 4)
        cv2.polylines(c_canvas, [np.array(pts_disp_all, dtype=np.int32)], isClosed=is_closed, color=(0, 240, 255), thickness=2, lineType=cv2.LINE_AA)
        if is_closed:
            overlay_poly = c_canvas.copy()
            cv2.fillPoly(overlay_poly, [np.array(pts_disp_all, dtype=np.int32)], (0, 220, 255))
            cv2.addWeighted(overlay_poly, 0.22, c_canvas, 0.78, 0, c_canvas)

    st.caption(f"View: {crop_w}×{crop_h} px ({roi_choice})")

    pil_canvas = Image.fromarray(cv2.cvtColor(c_canvas, cv2.COLOR_BGR2RGB))
    click_data = streamlit_image_coordinates(
        pil_canvas,
        use_column_width=True,
        image_format="JPEG",
        jpeg_quality=90,
        key=f"{key_prefix}_canvas_v{st.session_state.get(ver_key, 0)}",
    )

    if click_data is not None:
        c_time = click_data.get("unix_time", 0)
        if c_time and c_time != st.session_state.get(time_key, 0):
            st.session_state[time_key] = c_time
            elem_w = float(click_data.get("width", 0)) or float(disp_w)
            elem_h = float(click_data.get("height", 0)) or float(disp_h)

            if "x" in click_data and "y" in click_data and len(st.session_state[pts_key]) < target_count:
                if elem_w > 0 and elem_h > 0:
                    new_pt = map_roi_click_to_native(
                        click_data["x"], click_data["y"], elem_w, elem_h, roi_box, rf_w, rf_h
                    )
                    st.session_state[pts_key].append(new_pt)
                    st.session_state[_CANON_KEYS["applied"]] = False
                    st.session_state[_CANON_KEYS["just_applied"]] = False
                    st.session_state[ver_key] = st.session_state.get(ver_key, 0) + 1
                    st.rerun()


def _render_candidate_preview_and_table(
    raw_frame: np.ndarray,
    dst_points: np.ndarray,
    landmark_labels: List[str],
    active_warp_mode: str,
    zone_title: str,
    key_prefix: str,
) -> None:
    """Renders instant pitch projection preview and per-point reprojection residual table."""
    pts_key = _CANON_KEYS["pts"]
    pts_n = len(st.session_state[pts_key])
    rf_h, rf_w = raw_frame.shape[:2]
    target_count = len(dst_points)

    if pts_n >= 4 and pts_n == target_count:
        st.markdown("---")
        show_cand_preview = st.checkbox(
            "⚡ Preview Full Projected Pitch Mask & Chalk Lines on this Frame",
            value=True,
            key=f"{key_prefix}_preview_cand",
        )
        if show_cand_preview:
            try:
                from perception.homography import HomographyTransformer
                src_cand = np.float32(st.session_state[pts_key])
                eff_warp = active_warp_mode if target_count >= 6 else "planar"
                temp_trans = HomographyTransformer(frame_width=rf_w, frame_height=rf_h, warp_mode=eff_warp)
                temp_trans.calibrate(src_cand, dst_points, frame_width=rf_w, frame_height=rf_h, warp_mode=eff_warp)
                cand_img = render_pitch_projection_preview(raw_frame, temp_trans)
                st.image(
                    cv2.cvtColor(cand_img, cv2.COLOR_BGR2RGB),
                    caption=f"Instant Candidate Projection on Frame ({zone_title} — {target_count} Points, Mode={eff_warp})",
                    width="stretch",
                )
            except Exception as e:
                st.warning(f"Could not compute projection with selected points: {e}")

    if st.session_state[pts_key]:
        residuals: List[float] = []
        inliers = None
        if pts_n >= 4:
            try:
                from perception.homography import HomographyTransformer
                src_cand = np.float32(st.session_state[pts_key])
                dst_cand = np.float32(dst_points[:pts_n])
                eff_warp = active_warp_mode if pts_n >= 6 else "planar"
                temp_trans = HomographyTransformer(frame_width=rf_w, frame_height=rf_h, warp_mode=eff_warp)
                temp_trans.calibrate(src_cand, dst_cand, frame_width=rf_w, frame_height=rf_h, warp_mode=eff_warp)
                res_dict = temp_trans.compute_reprojection_residuals(src_cand, dst_cand)
                residuals = res_dict.get("residuals", [])
                inliers = getattr(temp_trans, "inlier_mask", None)
            except Exception:
                residuals = []

        rows = []
        for idx, pt in enumerate(st.session_state[pts_key]):
            dst = dst_points[idx] if idx < len(dst_points) else (0, 0)
            lbl = landmark_labels[idx] if idx < len(landmark_labels) else f"Target {idx + 1}"
            if idx < len(residuals):
                res_val = residuals[idx]
                is_inlier = bool(inliers[idx][0]) if inliers is not None and idx < len(inliers) else True
                if res_val < 1.5:
                    badge = f"`{res_val:.2f} px` • <span style='color:#10b981; font-weight:700;'>Exact Match</span>"
                elif res_val < 4.0 or is_inlier:
                    badge = f"`{res_val:.2f} px` • <span style='color:#38bdf8; font-weight:700;'>Good Fit</span>"
                else:
                    badge = f"`{res_val:.2f} px` • <span style='color:#ef4444; font-weight:700;'>Recheck Point</span>"
            else:
                badge = "Placed (Need 4+ pts)"
            rows.append(f"| **P{idx+1}: {lbl}** | `({pt[0]:.0f}, {pt[1]:.0f})` | `({dst[0]:.1f}m, {dst[1]:.1f}m)` | {badge} |")

        with st.expander("Alignment Accuracy Details", expanded=False):
            st.markdown(f"""
            | Point | Video Pixel | Pitch Location | Fit Quality |
            | :--- | :---: | :---: | :---: |
            {chr(10).join(rows)}
            """, unsafe_allow_html=True)


def render_mouse_calibration_studio(raw_frame: np.ndarray, transformer, key_prefix: str = "calib") -> Dict[str, Any]:
    """
    Renders an interactive full-width landmark calibration studio supporting:
      - Standard Planar Homography (3x3 perspective, 4–12 points)
      - Curved Panorama Mode (Thin-Plate Spline, 6–12 spread points)
      - Native 1:1 ROI Zone Zoom presets + Custom Pan & Zoom
    """
    pts_key = _CANON_KEYS["pts"]
    applied_key = _CANON_KEYS["applied"]
    enabled_key = _CANON_KEYS["enabled"]
    just_applied_key = _CANON_KEYS["just_applied"]
    mode_key = _CANON_KEYS["mode"]
    tmpl_key = _CANON_KEYS["tmpl"]
    warp_key = _CANON_KEYS["warp"]

    _, time_key, ver_key = _init_calib_session_keys(transformer, key_prefix)
    landmark_names = list(FIFA_CHALK_LANDMARKS.keys())
    rf_h, rf_w = raw_frame.shape[:2]

    st.markdown("#### 🎯 Pitch Alignment Studio")
    st.caption(
        f"Video Frame: **{rf_w} × {rf_h} px** • Click matching field lines below to sync video positions onto the 2D tactical pitch."
    )

    warp_options = [
        "📐 Standard Broadcast Camera",
        "🌀 Wide / Panoramic Camera (Curved Lines)",
    ]
    cur_warp_idx = 1 if st.session_state.get(warp_key, "planar") == "panorama_tps" else 0
    selected_warp = st.radio(
        "Camera Lens Type",
        warp_options,
        index=cur_warp_idx,
        horizontal=True,
        key=f"{key_prefix}_warp_mode_radio",
        help="Use Standard Broadcast for normal TV angles. Use Wide / Panoramic when pitch lines appear curved.",
    )
    new_warp_mode = "panorama_tps" if selected_warp.startswith("🌀") else "planar"
    if new_warp_mode != st.session_state.get(warp_key):
        st.session_state[warp_key] = new_warp_mode
        if new_warp_mode == "panorama_tps" and coerce_anchor_count(st.session_state.get(_CANON_KEYS["anchors"], 4), 4) < 6:
            st.session_state[_CANON_KEYS["anchors"]] = 8
            st.session_state.pop(f"{key_prefix}_num_anchors_widget", None)
        st.session_state[just_applied_key] = False

    active_warp_mode = st.session_state[warp_key]

    calib_mode = st.radio(
        "Alignment Method",
        [
            "🧩 Custom Field Points (4–12)",
            "📌 Quick 4-Point Presets",
        ],
        index=0 if st.session_state.get(mode_key, "custom") == "custom" else 1,
        horizontal=True,
        key=f"{key_prefix}_strategy_radio",
    )
    is_custom_mode = calib_mode.startswith("🧩")
    st.session_state[mode_key] = "custom" if is_custom_mode else "preset"

    if is_custom_mode:
        dst_points, landmark_labels, zone_title, zone_desc, has_duplicates, has_collinear = _render_custom_landmark_builder(
            active_warp_mode, landmark_names, key_prefix, ver_key
        )
    else:
        template_options = list(GROUND_TRUTH_TEMPLATES.keys())
        cur_idx = template_options.index(st.session_state[tmpl_key]) if st.session_state[tmpl_key] in template_options else 0
        template_name = st.selectbox("Quick Pitch Area", template_options, index=cur_idx, key=f"{key_prefix}_dst_tmpl")
        st.session_state[tmpl_key] = template_name
        tmpl_meta = GROUND_TRUTH_TEMPLATES[template_name]
        dst_points = tmpl_meta["points"]
        landmark_labels = tmpl_meta["labels"]
        zone_title = tmpl_meta["title"]
        zone_desc = tmpl_meta["desc"]
        has_duplicates = False
        has_collinear = False
        if active_warp_mode == "panorama_tps":
            st.info("ℹ️ Quick presets use 4 points and flat perspective. Switch to Custom Field Points (6–12) for curved panoramic footage.")

    with st.expander("🗺️ Pitch Guide & Point Sequence", expanded=False):
        col_g1, col_g2 = st.columns([1.15, 1.0])
        with col_g1:
            thumb_canvas = render_landmarks_pitch_thumbnail(
                points_metric=dst_points,
                labels=landmark_labels,
                title=zone_title,
                thumb_width=540,
                thumb_height=330,
            )
            st.image(cv2.cvtColor(thumb_canvas, cv2.COLOR_BGR2RGB), caption=f"Pitch Destination: {zone_title}", width="stretch")
        with col_g2:
            cur_pt_idx = len(st.session_state[pts_key])
            guide_rows = []
            for i, lbl_txt in enumerate(landmark_labels):
                if i < cur_pt_idx:
                    status_tag = "<span style='color:#10b981; font-weight:700;'>✓</span>"
                elif i == cur_pt_idx:
                    status_tag = "<span style='color:#facc15; font-weight:700;'>▶ NEXT</span>"
                else:
                    status_tag = "<span style='color:#64748b;'>•</span>"
                guide_rows.append(f"{status_tag} <b>L{i+1}:</b> {lbl_txt}<br/>")
            st.markdown(f"""
            <div style="background:#0f172a; border-left:4px solid #38bdf8; border-radius:6px; padding:10px 14px;">
                <div style="font-size:0.80rem; font-weight:700; color:#38bdf8;">📌 Ordered Landmark Clicking Sequence ({len(dst_points)} Points):</div>
                <div style="font-size:0.75rem; color:#cbd5e1; margin-top:4px; line-height:1.45;">
                    {''.join(guide_rows)}
                </div>
            </div>
            """, unsafe_allow_html=True)

    _render_roi_zoom_canvas(raw_frame, dst_points, landmark_labels, key_prefix, time_key, ver_key)

    target_count = len(dst_points)
    pts_n = len(st.session_state[pts_key])
    a_c1, a_c2, a_c3, a_c4 = st.columns([1.6, 1.1, 1.1, 2.0])
    with a_c1:
        can_apply = (pts_n == target_count) and not (is_custom_mode and has_duplicates) and not (target_count == 4 and has_collinear)
        if st.button("✅ Apply Alignment", type="primary", disabled=not can_apply, width="stretch", key=f"{key_prefix}_btn_apply"):
            st.session_state[enabled_key] = True
            st.session_state[applied_key] = True
            st.session_state[just_applied_key] = True
            src_pts = np.float32(st.session_state[pts_key])
            eff_warp = active_warp_mode if target_count >= 6 else "planar"
            transformer.calibrate(src_pts, dst_points, frame_width=rf_w, frame_height=rf_h, warp_mode=eff_warp)
            st.success(f"✅ Pitch alignment updated ({target_count} points)!")
            st.rerun()

    with a_c2:
        if st.button("↩ Undo", disabled=len(st.session_state[pts_key]) == 0, width="stretch", key=f"{key_prefix}_btn_undo"):
            st.session_state[pts_key].pop()
            st.session_state[ver_key] = st.session_state.get(ver_key, 0) + 1
            st.session_state[time_key] = 0
            st.session_state[applied_key] = False
            st.session_state[just_applied_key] = False
            st.rerun()

    with a_c3:
        if st.button("🔄 Reset", width="stretch", key=f"{key_prefix}_btn_reset"):
            st.session_state[pts_key] = []
            st.session_state[applied_key] = False
            st.session_state[enabled_key] = False
            st.session_state[just_applied_key] = False
            st.session_state[ver_key] = st.session_state.get(ver_key, 0) + 1
            st.session_state[time_key] = 0
            from ui.state import get_active_source_info
            active_src = st.session_state.get("active_source_key", "sample_broadcast")
            def_mode = get_active_source_info(active_src).get("default_mode", "sideline")
            transformer.set_camera_mode(def_mode)
            st.session_state["tv_calib_just_reset"] = True
            st.rerun()

    with a_c4:
        if pts_n == target_count:
            st.markdown(f"<span class='badge-pass'>{target_count} / {target_count} POINTS READY TO APPLY</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"<span class='badge-warn'>PLACED {pts_n} / {target_count} POINTS</span>", unsafe_allow_html=True)

    _render_candidate_preview_and_table(raw_frame, dst_points, landmark_labels, active_warp_mode, zone_title, key_prefix)

    just_applied = bool(st.session_state.get(just_applied_key, False))
    if just_applied:
        st.session_state[just_applied_key] = False
    just_reset = bool(st.session_state.get("tv_calib_just_reset", False))
    if just_reset:
        st.session_state["tv_calib_just_reset"] = False

    return {
        "applied": st.session_state[applied_key],
        "just_applied": just_applied,
        "just_reset": just_reset,
        "enabled": st.session_state[enabled_key],
        "points": st.session_state[pts_key],
        "dst_points": dst_points,
        "warp_mode": active_warp_mode if target_count >= 6 else "planar",
        "title": zone_title,
    }
