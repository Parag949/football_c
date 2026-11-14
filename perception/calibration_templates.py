"""
TacticalVision — FIFA Metric Chalk Ground-Truth Calibration & Landmark Studio
=============================================================================
Provides standardized ground-truth destination pitch landmarks and templates
based on official FIFA pitch specifications (105m x 68m):
- Point-by-point arbitrary Chalk Landmark Selector (Mix & Match any 4 visible field features)
- Pre-packaged verified Pitch Zone Templates (Quick 1-Click corridor templates)
- Real-time 2D FIFA Pitch Thumbnail Guide with glowing numbered pins (L1 ➔ L2 ➔ L3 ➔ L4)
- Safe perspective projection preview with projective horizon filtering
"""

from typing import Dict, Any, List, Tuple, Optional, Union
import cv2
import numpy as np

from perception.pitch_model import (
    PitchModel,
    FIFA_CHALK_LANDMARKS,
    get_fifa_chalk_landmarks,
)

# Backward-compatible aliases pointing directly to PitchModel canonical constants
PITCH_LENGTH = PitchModel.LENGTH
PITCH_WIDTH = PitchModel.WIDTH

# Exact metric coordinates of standard chalk landmarks
X_HALFWAY = PitchModel.LENGTH / 2.0
Y_CENTER = PitchModel.WIDTH / 2.0
R_CIRCLE = PitchModel.CENTER_CIRCLE_RADIUS

# Penalty spots
X_PENALTY_SPOT_RIGHT = PitchModel.LENGTH - PitchModel.PENALTY_SPOT_DISTANCE
X_PENALTY_SPOT_LEFT = PitchModel.PENALTY_SPOT_DISTANCE

# 18-yard box boundaries
X_BOX_RIGHT = PitchModel.LENGTH - PitchModel.PENALTY_BOX_LENGTH
X_BOX_LEFT = PitchModel.PENALTY_BOX_LENGTH
Y_BOX_TOP = (PitchModel.WIDTH - PitchModel.PENALTY_BOX_WIDTH) / 2.0
Y_BOX_BOTTOM = (PitchModel.WIDTH + PitchModel.PENALTY_BOX_WIDTH) / 2.0

# Penalty "D" Arc Apexes
X_D_APEX_RIGHT = X_PENALTY_SPOT_RIGHT - R_CIRCLE
X_D_APEX_LEFT = X_PENALTY_SPOT_LEFT + R_CIRCLE

# 6-yard goal area boundaries
X_GOAL_AREA_RIGHT = PitchModel.LENGTH - PitchModel.SIX_YARD_BOX_LENGTH
X_GOAL_AREA_LEFT = PitchModel.SIX_YARD_BOX_LENGTH
Y_GOAL_AREA_TOP = (PitchModel.WIDTH - PitchModel.SIX_YARD_BOX_WIDTH) / 2.0
Y_GOAL_AREA_BOTTOM = (PitchModel.WIDTH + PitchModel.SIX_YARD_BOX_WIDTH) / 2.0


# ------------------------------------------------------------------------------
# 2. PRE-PACKAGED VERIFIED GROUND-TRUTH TEMPLATES
# ------------------------------------------------------------------------------
GROUND_TRUTH_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "Right Penalty Box & Penalty Arc 'D' [84.85 ➔ 105.0]m": {
        "points": np.float32([
            [X_BOX_RIGHT, Y_BOX_TOP],
            [105.0, Y_BOX_TOP],
            [105.0, Y_BOX_BOTTOM],
            [X_D_APEX_RIGHT, Y_CENTER]
        ]),
        "title": "Right Penalty Box + Penalty 'D' Apex",
        "desc": "Anchors on 18-yard box corners on goal line, top box corner, and apex of Penalty Arc 'D' (84.85m, 34.0m).",
        "camera_hint": "🎯 Final Third & Goalmouth Camera",
        "chalk_features": ["Penalty Arc 'D' apex", "18-yard box corners", "Goal line intersections"],
        "labels": [
            f"L1: Far 18-Yard Box Corner ({X_BOX_RIGHT:.1f}m, {Y_BOX_TOP:.2f}m)",
            f"L2: Far Goal Line Box Intersection (105.0m, {Y_BOX_TOP:.2f}m)",
            f"L3: Near Goal Line Box Intersection (105.0m, {Y_BOX_BOTTOM:.2f}m)",
            f"L4: Apex of Penalty Arc 'D' ({X_D_APEX_RIGHT:.2f}m, {Y_CENTER:.1f}m)"
        ]
    },
    "Halfway Line to Right Penalty Box Corridor [52.5 ➔ 88.5]m": {
        "points": np.float32([
            [X_HALFWAY, 0.0],
            [X_BOX_RIGHT, Y_BOX_TOP],
            [X_BOX_RIGHT, Y_BOX_BOTTOM],
            [X_HALFWAY, PITCH_WIDTH]
        ]),
        "title": "Halfway Line to Right Box Corridor",
        "desc": "Uses visible chalk lines: Halfway line touchlines (52.5m, 0m & 68m) and two 18-yard box corners.",
        "camera_hint": "📺 Standard TV Sideline Broadcast (Attacking Half)",
        "chalk_features": ["Halfway line touchlines", "18-yard box corners"],
        "labels": [
            f"L1: Halfway Line Far Touchline ({X_HALFWAY:.1f}m, 0.0m)",
            f"L2: Far 18-Yard Box Corner ({X_BOX_RIGHT:.1f}m, {Y_BOX_TOP:.2f}m)",
            f"L3: Near 18-Yard Box Corner ({X_BOX_RIGHT:.1f}m, {Y_BOX_BOTTOM:.2f}m)",
            f"L4: Halfway Line Near Touchline ({X_HALFWAY:.1f}m, {PITCH_WIDTH:.1f}m)"
        ]
    },
    "Left Penalty Box & Penalty Arc 'D' [0.0 ➔ 20.15]m": {
        "points": np.float32([
            [0.0, Y_BOX_TOP],
            [X_BOX_LEFT, Y_BOX_TOP],
            [X_D_APEX_LEFT, Y_CENTER],
            [0.0, Y_BOX_BOTTOM]
        ]),
        "title": "Left Penalty Box + Penalty 'D' Apex",
        "desc": "Anchors on left goal line intersections, 18-yard box corner, and apex of Left Penalty Arc 'D'.",
        "camera_hint": "🥅 Left Goal Area & Goalkeeper Camera",
        "chalk_features": ["Left Penalty Arc 'D' apex", "Left 18-yard box corners"],
        "labels": [
            f"L1: Far Goal Line Box Intersection (0.0m, {Y_BOX_TOP:.2f}m)",
            f"L2: Far 18-Yard Box Corner ({X_BOX_LEFT:.1f}m, {Y_BOX_TOP:.2f}m)",
            f"L3: Apex of Left Penalty Arc 'D' ({X_D_APEX_LEFT:.2f}m, {Y_CENTER:.1f}m)",
            f"L4: Near Goal Line Box Intersection (0.0m, {Y_BOX_BOTTOM:.2f}m)"
        ]
    },
    "Center Circle & Touchline-to-Touchline Band [Y: 0m ➔ 68m]": {
        "points": np.float32([
            [X_HALFWAY, 0.0],
            [X_HALFWAY + R_CIRCLE, Y_CENTER],
            [X_HALFWAY, PITCH_WIDTH],
            [X_HALFWAY - R_CIRCLE, Y_CENTER]
        ]),
        "title": "Center Circle & Full Touchline Band",
        "desc": "Uses the visible touchline-to-touchline halfway line and the right & left outer points of the 9.15m center circle.",
        "camera_hint": "⚽ Midfield Camera (Center Circle Framing)",
        "chalk_features": ["Halfway line touchlines", "Center circle outer apexes"],
        "labels": [
            f"L1: Far Touchline Halfway Mark ({X_HALFWAY:.1f}m, 0.0m)",
            f"L2: Center Circle Right Apex ({X_HALFWAY + R_CIRCLE:.2f}m, {Y_CENTER:.1f}m)",
            f"L3: Near Touchline Halfway Mark ({X_HALFWAY:.1f}m, {PITCH_WIDTH:.1f}m)",
            f"L4: Center Circle Left Apex ({X_HALFWAY - R_CIRCLE:.2f}m, {Y_CENTER:.1f}m)"
        ]
    },
    "Right 18-Yard Penalty Box Complete Chalk [88.5 ➔ 105.0]m": {
        "points": np.float32([
            [X_BOX_RIGHT, Y_BOX_TOP],
            [105.0, Y_BOX_TOP],
            [105.0, Y_BOX_BOTTOM],
            [X_BOX_RIGHT, Y_BOX_BOTTOM]
        ]),
        "title": "Right 18-Yard Box Complete Rectangle",
        "desc": "The 4 visible chalk corners of the right 18-yard box: (88.5m, 13.84m) to (105m, 54.16m).",
        "camera_hint": "📐 Tactical End-Zone & Broadcast Box View",
        "chalk_features": ["18-yard line corners", "Goal line corners"],
        "labels": [
            f"L1: Far 18-Yard Corner ({X_BOX_RIGHT:.1f}m, {Y_BOX_TOP:.2f}m)",
            f"L2: Far Goal Line Box Intersection (105.0m, {Y_BOX_TOP:.2f}m)",
            f"L3: Near Goal Line Box Intersection (105.0m, {Y_BOX_BOTTOM:.2f}m)",
            f"L4: Near 18-Yard Corner ({X_BOX_RIGHT:.1f}m, {Y_BOX_BOTTOM:.2f}m)"
        ]
    },
    "Full Pitch 4 Corner Flags [0.0 ➔ 105.0]m x [0.0 ➔ 68.0]m": {
        "points": np.float32([
            [0.0, 0.0],
            [105.0, 0.0],
            [105.0, 68.0],
            [0.0, 68.0]
        ]),
        "title": "Full Pitch 4 Corner Flags",
        "desc": "The 4 extreme corner flagpoles of the FIFA pitch: (0m, 0m), (105m, 0m), (105m, 68m), (0m, 68m).",
        "camera_hint": "🦅 Ultra-Wide Stadium Panorama / Tactical Gantry",
        "chalk_features": ["4 Corner flags", "Full pitch perimeter"],
        "labels": [
            "L1: Far Left Corner Flag (0.0m, 0.0m)",
            "L2: Far Right Corner Flag (105.0m, 0.0m)",
            "L3: Near Right Corner Flag (105.0m, 68.0m)",
            "L4: Near Left Corner Flag (0.0m, 68.0m)"
        ]
    }
}

# Backward-compatible alias
CALIBRATION_TEMPLATES = GROUND_TRUTH_TEMPLATES


def get_template_dest_points(template_name: str) -> np.ndarray:
    """Returns the 4 destination metric points on the FIFA pitch for the given template."""
    if template_name in GROUND_TRUTH_TEMPLATES:
        return GROUND_TRUTH_TEMPLATES[template_name]["points"]
    for key, val in GROUND_TRUTH_TEMPLATES.items():
        if template_name.lower() in key.lower():
            return val["points"]
    return list(GROUND_TRUTH_TEMPLATES.values())[0]["points"]


# ------------------------------------------------------------------------------
# 3. UNIFIED DYNAMIC 2D PITCH THUMBNAIL RENDERER
# ------------------------------------------------------------------------------
def render_landmarks_pitch_thumbnail(
    points_metric: Union[np.ndarray, List[Tuple[float, float]]],
    labels: Optional[List[str]] = None,
    title: str = "Chalk Landmark Destination Guide",
    thumb_width: int = 500,
    thumb_height: int = 320
) -> np.ndarray:
    """
    Renders an accurate 2D pitch thumbnail illustrating any 4 chosen chalk landmarks.
    Draws full FIFA pitch lines (touchlines, boxes, center circle, 'D' curves, spots)
    and highlights the destination area with numbered glowing pins L1, L2, L3, L4.
    """
    pts_m = np.float32(points_metric)
    pad = 28
    pw = thumb_width - (2 * pad)
    ph = thumb_height - (2 * pad)

    def to_px(x_m: float, y_m: float) -> Tuple[int, int]:
        px = int(pad + (max(0.0, min(PITCH_LENGTH, x_m)) / PITCH_LENGTH) * pw)
        py = int(pad + (max(0.0, min(PITCH_WIDTH, y_m)) / PITCH_WIDTH) * ph)
        return (px, py)

    # Base tactical pitch canvas
    canvas = np.zeros((thumb_height, thumb_width, 3), dtype=np.uint8)
    canvas[:] = (18, 48, 22)  # Stadium emerald turf

    # Pitch boundary chalk
    cv2.rectangle(canvas, (pad, pad), (pad + pw, pad + ph), (220, 240, 220), 2, cv2.LINE_AA)

    # Halfway line & center circle
    half_x = pad + pw // 2
    cv2.line(canvas, (half_x, pad), (half_x, pad + ph), (210, 235, 210), 2, cv2.LINE_AA)
    center_y = pad + ph // 2
    r_cc = int((R_CIRCLE / PITCH_LENGTH) * pw)
    cv2.circle(canvas, (half_x, center_y), r_cc, (210, 235, 210), 2, cv2.LINE_AA)
    cv2.circle(canvas, (half_x, center_y), 4, (210, 235, 210), -1, cv2.LINE_AA)

    # 18-yard penalty boxes
    box_w = int((16.5 / PITCH_LENGTH) * pw)
    box_h = int((40.32 / PITCH_WIDTH) * ph)
    box_top = pad + int((Y_BOX_TOP / PITCH_WIDTH) * ph)
    cv2.rectangle(canvas, (pad, box_top), (pad + box_w, box_top + box_h), (210, 235, 210), 2, cv2.LINE_AA)
    cv2.rectangle(canvas, (pad + pw - box_w, box_top), (pad + pw, box_top + box_h), (210, 235, 210), 2, cv2.LINE_AA)

    # 6-yard goal areas
    gbox_w = int((5.5 / PITCH_LENGTH) * pw)
    gbox_h = int((18.32 / PITCH_WIDTH) * ph)
    gbox_top = pad + int((Y_GOAL_AREA_TOP / PITCH_WIDTH) * ph)
    cv2.rectangle(canvas, (pad, gbox_top), (pad + gbox_w, gbox_top + gbox_h), (210, 235, 210), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (pad + pw - gbox_w, gbox_top), (pad + pw, gbox_top + gbox_h), (210, 235, 210), 1, cv2.LINE_AA)

    # Penalty spots
    p_spot_l = to_px(X_PENALTY_SPOT_LEFT, Y_CENTER)
    p_spot_r = to_px(X_PENALTY_SPOT_RIGHT, Y_CENTER)
    cv2.circle(canvas, p_spot_l, 4, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(canvas, p_spot_r, 4, (255, 255, 255), -1, cv2.LINE_AA)

    # Penalty Arc "D" curves
    r_d = int((R_CIRCLE / PITCH_LENGTH) * pw)
    cv2.ellipse(canvas, p_spot_r, (r_d, r_d), 0, 127, 233, (210, 235, 210), 2, cv2.LINE_AA)
    cv2.ellipse(canvas, p_spot_l, (r_d, r_d), 0, -53, 53, (210, 235, 210), 2, cv2.LINE_AA)

    # Highlighted Translucent Destination Area
    poly_px = np.array([to_px(p[0], p[1]) for p in pts_m], dtype=np.int32)
    overlay = canvas.copy()
    cv2.fillPoly(overlay, [poly_px], (0, 220, 255))
    cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, canvas)
    cv2.polylines(canvas, [poly_px], isClosed=True, color=(0, 255, 255), thickness=3, lineType=cv2.LINE_AA)

    # Direction arrows connecting L1 -> L2 -> ... -> LN
    n_pts = len(poly_px)
    for i in range(n_pts):
        p_start = poly_px[i]
        p_end = poly_px[(i + 1) % n_pts]
        mid_x = int((p_start[0] + p_end[0]) / 2)
        mid_y = int((p_start[1] + p_end[1]) / 2)
        cv2.arrowedLine(canvas, (p_start[0], p_start[1]), (mid_x, mid_y), (255, 255, 255), 2, tipLength=0.35, line_type=cv2.LINE_AA)

    # Numbered Glowing Pins L1, L2, L3, L4
    badge_colors = [(0, 240, 255), (0, 255, 120), (255, 0, 200), (255, 200, 0)]
    for i, (px, py) in enumerate(poly_px):
        col = badge_colors[i % len(badge_colors)]
        cv2.circle(canvas, (px, py), 12, col, -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (px, py), 15, (255, 255, 255), 2, lineType=cv2.LINE_AA)
        cv2.putText(canvas, f"L{i+1}", (px - 9, py + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 2, lineType=cv2.LINE_AA)

        # Coordinate label in meters
        mx, my = pts_m[i]
        lbl = f"L{i+1} ({mx:.1f}m,{my:.1f}m)"
        offset_y = -18 if py < thumb_height // 2 else 24
        offset_x = -35 if px > thumb_width // 2 else 10
        cv2.putText(canvas, lbl, (px + offset_x, py + offset_y), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 255), 1, cv2.LINE_AA)

    # Title Banner
    cv2.putText(canvas, title, (14, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
    return canvas


def render_template_thumbnail(
    template_name: str,
    thumb_width: int = 500,
    thumb_height: int = 320
) -> np.ndarray:
    """Wrapper rendering pre-packaged template thumbnail for backward compatibility."""
    tmpl = GROUND_TRUTH_TEMPLATES.get(template_name, list(GROUND_TRUTH_TEMPLATES.values())[0])
    return render_landmarks_pitch_thumbnail(
        points_metric=tmpl["points"],
        labels=tmpl.get("labels"),
        title=f"Ground Truth: {tmpl['title']}",
        thumb_width=thumb_width,
        thumb_height=thumb_height
    )


# ------------------------------------------------------------------------------
# 4. INSTANT FULL PITCH PROJECTION PREVIEW
# ------------------------------------------------------------------------------
def render_pitch_projection_preview(frame: np.ndarray, H_or_transformer: Any) -> np.ndarray:
    """
    Projects the complete FIFA metric pitch model directly onto the video frame.
    Accepts either a 3x3 homography matrix H (np.ndarray) or a HomographyTransformer
    (supporting both planar homography and curved panorama_tps polylines).
    """
    if H_or_transformer is None:
        return frame.copy()

    transformer = None
    if hasattr(H_or_transformer, "pitch_to_image") and hasattr(H_or_transformer, "H"):
        transformer = H_or_transformer
        H = transformer.H
        is_tps = getattr(transformer, "warp_mode", "planar") in ("panorama_tps", "universal")
    else:
        H = H_or_transformer
        is_tps = False

    if H is None:
        return frame.copy()

    try:
        H_inv = np.linalg.inv(H)
    except Exception:
        return frame.copy()

    # Align projective depth: points on the pitch in front of the camera should have positive w
    if transformer is not None and getattr(transformer, "dst_points", None) is not None and len(transformer.dst_points) > 0:
        cx = float(np.mean(transformer.dst_points[:, 0]))
        cy = float(np.mean(transformer.dst_points[:, 1]))
    else:
        cx, cy = 52.5, 34.0
    center_test = H_inv @ np.array([cx, cy, 1.0], dtype=np.float64)
    if center_test[2] < 0:
        H_inv = -H_inv

    preview = frame.copy()
    h, w = frame.shape[:2]

    def subdivide_path(pts: List[Tuple[float, float]], steps: int = 16, closed: bool = False):
        if not is_tps or len(pts) < 2:
            return pts
        out = []
        n = len(pts)
        edges = n if closed else n - 1
        for i in range(edges):
            p1, p2 = pts[i], pts[(i + 1) % n]
            for s in range(steps):
                t = s / float(steps)
                out.append((p1[0] + (p2[0] - p1[0]) * t, p1[1] + (p2[1] - p1[1]) * t))
        if not closed:
            out.append(pts[-1])
        return out

    def to_img(pts_metric):
        if transformer is not None:
            transformer.update_dimensions(w, h)
            pxs = transformer.pitch_to_image(pts_metric, filter_behind_camera=True)
            return np.array(pxs, dtype=np.int32) if pxs else np.empty((0, 2), dtype=np.int32)
        res = []
        for pt in pts_metric:
            vec = np.array([float(pt[0]), float(pt[1]), 1.0], dtype=np.float64)
            proj = H_inv @ vec
            w_denom = proj[2]
            if w_denom > 1e-4:
                res.append([int(round(proj[0] / w_denom)), int(round(proj[1] / w_denom))])
        return np.array(res, dtype=np.int32)

    def is_safe_poly(pts):
        if len(pts) < 3:
            return False
        return not any(p[0] < -w or p[0] > 2 * w or p[1] < -h or p[1] > 2 * h for p in pts)

    # 1. Pitch perimeter boundary [105m x 68m]
    perimeter = subdivide_path([(0.0, 0.0), (105.0, 0.0), (105.0, 68.0), (0.0, 68.0)], steps=20, closed=True)
    peri_px = to_img(perimeter)

    if is_safe_poly(peri_px):
        overlay = preview.copy()
        cv2.fillPoly(overlay, [peri_px], (0, 180, 80))
        cv2.addWeighted(overlay, 0.20, preview, 0.80, 0, preview)
        cv2.polylines(preview, [peri_px], isClosed=True, color=(0, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    # 2. Halfway Line
    halfway_px = to_img(subdivide_path([(52.5, 0.0), (52.5, 68.0)], steps=16, closed=False))
    if len(halfway_px) >= 2 and not any(p[0] < -w or p[0] > 2*w or p[1] < -h or p[1] > 2*h for p in halfway_px):
        cv2.polylines(preview, [halfway_px], isClosed=False, color=(0, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    # 3. Center Circle
    angles = np.linspace(0, 2 * np.pi, 36)
    circle_metric = [(52.5 + 9.15 * np.cos(a), 34.0 + 9.15 * np.sin(a)) for a in angles]
    circle_px = to_img(circle_metric)
    if is_safe_poly(circle_px):
        cv2.polylines(preview, [circle_px], isClosed=True, color=(0, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    # Center Spot
    c_spot = to_img([(52.5, 34.0)])
    if len(c_spot) == 1 and 0 <= c_spot[0][0] < w and 0 <= c_spot[0][1] < h:
        cv2.circle(preview, tuple(c_spot[0]), 5, (0, 255, 255), -1, lineType=cv2.LINE_AA)

    # 4. Right Penalty Box
    box_r = subdivide_path([(88.5, 13.84), (105.0, 13.84), (105.0, 54.16), (88.5, 54.16)], steps=14, closed=True)
    box_r_px = to_img(box_r)
    if is_safe_poly(box_r_px):
        cv2.polylines(preview, [box_r_px], isClosed=True, color=(0, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    # 5. Left Penalty Box
    box_l = subdivide_path([(0.0, 13.84), (16.5, 13.84), (16.5, 54.16), (0.0, 54.16)], steps=14, closed=True)
    box_l_px = to_img(box_l)
    if is_safe_poly(box_l_px):
        cv2.polylines(preview, [box_l_px], isClosed=True, color=(0, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    # 6. Penalty Spots
    p_spots = to_img([(11.0, 34.0), (94.0, 34.0)])
    for sp in p_spots:
        if 0 <= sp[0] < w and 0 <= sp[1] < h:
            cv2.circle(preview, tuple(sp), 4, (0, 255, 255), -1, lineType=cv2.LINE_AA)

    # HUD Status Banner
    try:
        cond = float(np.linalg.cond(H)) if H is not None else 1.0
    except Exception:
        cond = 1.0
    mode_tag = "PANORAMA TPS" if is_tps else "PLANAR H"
    cv2.rectangle(preview, (10, 10), (510, 42), (10, 15, 25), -1)
    cv2.putText(preview, f"PROJECTED PITCH MODEL ({mode_tag}, cond={cond:.1f})", (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 220), 1, cv2.LINE_AA)

    return preview


def compute_calibration_roi_crop(
    rf_w: int,
    rf_h: int,
    zone_mode: str = "full",
    pan_x_pct: float = 0.5,
    pan_y_pct: float = 0.5,
    zoom_factor: float = 1.0
) -> Tuple[int, int, int, int]:
    """
    Computes a clamped native-resolution crop box (x0, y0, x1, y1) inside [0..rf_w, 0..rf_h]
    so ultrawide/4K/panorama frames can be clicked at 1:1 native pixel precision.
    """
    rf_w = max(1, int(rf_w))
    rf_h = max(1, int(rf_h))

    if zone_mode == "full" or (zone_mode == "custom" and zoom_factor <= 1.05):
        return (0, 0, rf_w, rf_h)

    if zone_mode in ("left", "center", "right"):
        crop_w = min(rf_w, max(640, min(1640, int(round(rf_w * 0.40)))))
        crop_h = rf_h
        if zone_mode == "left":
            x0 = 0
        elif zone_mode == "right":
            x0 = max(0, rf_w - crop_w)
        else:
            x0 = max(0, (rf_w - crop_w) // 2)
        return (x0, 0, min(rf_w, x0 + crop_w), crop_h)

    zf = max(1.0, float(zoom_factor))
    crop_w = max(320, min(rf_w, int(round(rf_w / zf))))
    crop_h = max(240, min(rf_h, int(round(rf_h / zf))))
    cx = max(0.0, min(1.0, float(pan_x_pct))) * rf_w
    cy = max(0.0, min(1.0, float(pan_y_pct))) * rf_h
    x0 = int(round(max(0.0, min(float(rf_w - crop_w), cx - crop_w / 2.0))))
    y0 = int(round(max(0.0, min(float(rf_h - crop_h), cy - crop_h / 2.0))))
    return (x0, y0, min(rf_w, x0 + crop_w), min(rf_h, y0 + crop_h))


def map_roi_click_to_native(
    click_x: float,
    click_y: float,
    elem_w: float,
    elem_h: float,
    roi_box: Tuple[int, int, int, int],
    rf_w: int,
    rf_h: int
) -> Tuple[float, float]:
    """
    Converts a click coordinate on the displayed (possibly cropped & scaled) canvas back into
    exact native full-frame pixel coordinates (u, v) in [0..rf_w-1, 0..rf_h-1].
    """
    x0, y0, x1, y1 = roi_box
    crop_w = max(1.0, float(x1 - x0))
    crop_h = max(1.0, float(y1 - y0))
    safe_ew = max(1.0, float(elem_w))
    safe_eh = max(1.0, float(elem_h))

    native_u = float(x0) + float(click_x) * (crop_w / safe_ew)
    native_v = float(y0) + float(click_y) * (crop_h / safe_eh)
    native_u = max(0.0, min(float(max(1, rf_w) - 1), native_u))
    native_v = max(0.0, min(float(max(1, rf_h) - 1), native_v))
    return (native_u, native_v)


# Re-export UI studio component for backward compatibility with callers importing from perception
try:
    from ui.components.calibration_studio import render_mouse_calibration_studio  # noqa: E402,F401
except ImportError:  # pragma: no cover
    pass
