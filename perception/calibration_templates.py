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
from PIL import Image

# Official FIFA pitch dimensions in meters
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0

# Exact metric coordinates of standard chalk landmarks
X_HALFWAY = 52.5
Y_CENTER = 34.0
R_CIRCLE = 9.15

# Penalty spots
X_PENALTY_SPOT_RIGHT = 105.0 - 11.0  # 94.0m
X_PENALTY_SPOT_LEFT = 11.0          # 11.0m

# 18-yard box boundaries
X_BOX_RIGHT = 105.0 - 16.5          # 88.5m
X_BOX_LEFT = 16.5                   # 16.5m
Y_BOX_TOP = 34.0 - 20.16            # 13.84m
Y_BOX_BOTTOM = 34.0 + 20.16         # 54.16m

# Penalty "D" Arc Apexes
X_D_APEX_RIGHT = X_PENALTY_SPOT_RIGHT - R_CIRCLE  # 94.0 - 9.15 = 84.85m
X_D_APEX_LEFT = X_PENALTY_SPOT_LEFT + R_CIRCLE    # 11.0 + 9.15 = 20.15m

# 6-yard goal area boundaries
X_GOAL_AREA_RIGHT = 105.0 - 5.5     # 99.5m
X_GOAL_AREA_LEFT = 5.5              # 5.5m
Y_GOAL_AREA_TOP = 34.0 - 9.16       # 24.84m
Y_GOAL_AREA_BOTTOM = 34.0 + 9.16    # 43.16m


# ------------------------------------------------------------------------------
# 1. COMPREHENSIVE FIFA CHALK LANDMARK CATALOG (MIX & MATCH ARBITRARY FEATURES)
# ------------------------------------------------------------------------------
FIFA_CHALK_LANDMARKS: Dict[str, Tuple[float, float]] = {
    # Right Goalmouth & Attacking Half (X > 52.5m)
    "Right 18-Yard Box — Infield Far Corner [88.5m, 13.84m]": (X_BOX_RIGHT, Y_BOX_TOP),
    "Right 18-Yard Box — Infield Near Corner [88.5m, 54.16m]": (X_BOX_RIGHT, Y_BOX_BOTTOM),
    "Right 18-Yard Box — Goal Line Far [105m, 13.84m]": (105.0, Y_BOX_TOP),
    "Right 18-Yard Box — Goal Line Near [105m, 54.16m]": (105.0, Y_BOX_BOTTOM),
    "Right Penalty Arc 'D' — Infield Apex [84.85m, 34m]": (X_D_APEX_RIGHT, Y_CENTER),
    "Right Penalty Spot [94m, 34m]": (X_PENALTY_SPOT_RIGHT, Y_CENTER),
    "Right 6-Yard Box — Infield Far Corner [99.5m, 24.84m]": (X_GOAL_AREA_RIGHT, Y_GOAL_AREA_TOP),
    "Right 6-Yard Box — Infield Near Corner [99.5m, 43.16m]": (X_GOAL_AREA_RIGHT, Y_GOAL_AREA_BOTTOM),
    "Corner Flag — Right Far (Top-Right) [105m, 0m]": (105.0, 0.0),
    "Corner Flag — Right Near (Bottom-Right) [105m, 68m]": (105.0, 68.0),
    "Right Goal Center Line [105m, 34m]": (105.0, Y_CENTER),

    # Midfield & Center Circle
    "Halfway Line — Far Touchline (Top) [52.5m, 0m]": (X_HALFWAY, 0.0),
    "Halfway Line — Near Touchline (Bottom) [52.5m, 68m]": (X_HALFWAY, PITCH_WIDTH),
    "Center Spot [52.5m, 34m]": (X_HALFWAY, Y_CENTER),
    "Center Circle — Right Apex [61.65m, 34m]": (X_HALFWAY + R_CIRCLE, Y_CENTER),
    "Center Circle — Left Apex [43.35m, 34m]": (X_HALFWAY - R_CIRCLE, Y_CENTER),
    "Center Circle — Far Apex (Top) [52.5m, 24.85m]": (X_HALFWAY, Y_CENTER - R_CIRCLE),
    "Center Circle — Near Apex (Bottom) [52.5m, 43.15m]": (X_HALFWAY, Y_CENTER + R_CIRCLE),

    # Left Goalmouth & Defending Half (X < 52.5m)
    "Left 18-Yard Box — Infield Far Corner [16.5m, 13.84m]": (X_BOX_LEFT, Y_BOX_TOP),
    "Left 18-Yard Box — Infield Near Corner [16.5m, 54.16m]": (X_BOX_LEFT, Y_BOX_BOTTOM),
    "Left 18-Yard Box — Goal Line Far [0m, 13.84m]": (0.0, Y_BOX_TOP),
    "Left 18-Yard Box — Goal Line Near [0m, 54.16m]": (0.0, Y_BOX_BOTTOM),
    "Left Penalty Arc 'D' — Infield Apex [20.15m, 34m]": (X_D_APEX_LEFT, Y_CENTER),
    "Left Penalty Spot [11m, 34m]": (X_PENALTY_SPOT_LEFT, Y_CENTER),
    "Left 6-Yard Box — Infield Far Corner [5.5m, 24.84m]": (X_GOAL_AREA_LEFT, Y_GOAL_AREA_TOP),
    "Left 6-Yard Box — Infield Near Corner [5.5m, 43.16m]": (X_GOAL_AREA_LEFT, Y_GOAL_AREA_BOTTOM),
    "Corner Flag — Left Far (Top-Left) [0m, 0m]": (0.0, 0.0),
    "Corner Flag — Left Near (Bottom-Left) [0m, 68m]": (0.0, 68.0),
    "Left Goal Center Line [0m, 34m]": (0.0, Y_CENTER),
}


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
def render_pitch_projection_preview(frame: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Projects the complete FIFA metric pitch model directly onto the video frame
    using homography matrix H with projective horizon protection (w > 1e-4).
    """
    if H is None:
        return frame.copy()
        
    try:
        H_inv = np.linalg.inv(H)
    except Exception:
        return frame.copy()

    # Align projective depth: points on the pitch in front of the camera should have positive w
    center_test = H_inv @ np.array([52.5, 34.0, 1.0], dtype=np.float64)
    if center_test[2] < 0:
        H_inv = -H_inv

    preview = frame.copy()
    h, w = frame.shape[:2]

    def to_img(pts_metric):
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
    perimeter = [(0.0, 0.0), (105.0, 0.0), (105.0, 68.0), (0.0, 68.0)]
    peri_px = to_img(perimeter)

    if is_safe_poly(peri_px):
        overlay = preview.copy()
        cv2.fillPoly(overlay, [peri_px], (0, 180, 80))
        cv2.addWeighted(overlay, 0.20, preview, 0.80, 0, preview)
        cv2.polylines(preview, [peri_px], isClosed=True, color=(0, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    # 2. Halfway Line
    halfway_px = to_img([(52.5, 0.0), (52.5, 68.0)])
    if len(halfway_px) == 2 and not any(p[0] < -w or p[0] > 2*w or p[1] < -h or p[1] > 2*h for p in halfway_px):
        cv2.line(preview, tuple(halfway_px[0]), tuple(halfway_px[1]), (0, 255, 255), 2, lineType=cv2.LINE_AA)

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
    box_r = [(88.5, 13.84), (105.0, 13.84), (105.0, 54.16), (88.5, 54.16)]
    box_r_px = to_img(box_r)
    if is_safe_poly(box_r_px):
        cv2.polylines(preview, [box_r_px], isClosed=True, color=(0, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    # 5. Left Penalty Box
    box_l = [(0.0, 13.84), (16.5, 13.84), (16.5, 54.16), (0.0, 54.16)]
    box_l_px = to_img(box_l)
    if is_safe_poly(box_l_px):
        cv2.polylines(preview, [box_l_px], isClosed=True, color=(0, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    # 6. Penalty Spots
    p_spots = to_img([(11.0, 34.0), (94.0, 34.0)])
    for sp in p_spots:
        if 0 <= sp[0] < w and 0 <= sp[1] < h:
            cv2.circle(preview, tuple(sp), 4, (0, 255, 255), -1, lineType=cv2.LINE_AA)

    # HUD Status Banner
    cond = np.linalg.cond(H)
    cv2.rectangle(preview, (10, 10), (450, 42), (10, 15, 25), -1)
    cv2.putText(preview, f"PROJECTED PITCH MODEL (cond={cond:.1f})", (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 220), 1, cv2.LINE_AA)

    return preview


# ------------------------------------------------------------------------------
# 5. INTERACTIVE MOUSE CALIBRATION STUDIO (STREAMLIT ENGINE)
# ------------------------------------------------------------------------------
def render_mouse_calibration_studio(
    raw_frame: np.ndarray,
    transformer: Any,
    key_prefix: str = "calib"
) -> Dict[str, Any]:
    """
    Unified, versatile Interactive Mouse Calibration Studio for Streamlit.
    Empowers users to:
    1. Mix & match ANY 4 visible pitch features/chalk structures using the Landmark Builder.
    2. Or choose from pre-packaged verified pitch zone templates.
    3. View dynamic real-time 2D pitch guide thumbnails with numbered glowing pins.
    4. Click corresponding landmarks on the video canvas with aspect-ratio preservation.
    5. Instantly test homography matrix candidate projection on the frame.
    """
    import streamlit as st
    from streamlit_image_coordinates import streamlit_image_coordinates

    pts_key = f"{key_prefix}_points"
    applied_key = f"{key_prefix}_applied"
    enabled_key = f"{key_prefix}_enabled"
    click_key = f"{key_prefix}_last_click"
    time_key = f"{key_prefix}_last_click_time"
    ver_key = f"{key_prefix}_canvas_ver"
    mode_key = f"{key_prefix}_picker_mode"
    l1_key = f"{key_prefix}_lm1"
    l2_key = f"{key_prefix}_lm2"
    l3_key = f"{key_prefix}_lm3"
    l4_key = f"{key_prefix}_lm4"
    tmpl_key = f"{key_prefix}_template"

    if pts_key not in st.session_state:
        st.session_state[pts_key] = []
    if applied_key not in st.session_state:
        st.session_state[applied_key] = False
    if enabled_key not in st.session_state:
        st.session_state[enabled_key] = False
    if click_key not in st.session_state:
        st.session_state[click_key] = None
    if time_key not in st.session_state:
        st.session_state[time_key] = 0
    if ver_key not in st.session_state:
        st.session_state[ver_key] = 0
    if tmpl_key not in st.session_state:
        st.session_state[tmpl_key] = list(GROUND_TRUTH_TEMPLATES.keys())[0]

    landmark_names = list(FIFA_CHALK_LANDMARKS.keys())
    if l1_key not in st.session_state:
        st.session_state[l1_key] = landmark_names[0]  # Right 18-Yard Box Far Infield
    if l2_key not in st.session_state:
        st.session_state[l2_key] = landmark_names[2]  # Right 18-Yard Box Goal Line Far
    if l3_key not in st.session_state:
        st.session_state[l3_key] = landmark_names[3]  # Right 18-Yard Box Goal Line Near
    if l4_key not in st.session_state:
        st.session_state[l4_key] = landmark_names[4]  # Right Penalty Arc D Apex

    rf_h, rf_w = raw_frame.shape[:2]

    st.markdown("#### 🎯 Interactive Multi-Point Pitch Landmark Calibration Studio")
    st.caption("Identify N visible pitch chalk features (4, 5, 6, or 8 points) on your broadcast screen, map them to metric FIFA coordinates, and compute the homography matrix.")

    calib_mode = st.radio(
        "Chalk Feature Selection Strategy",
        [
            "🧩 Custom Multi-Point Anchor Builder (Select 4, 5, 6, or 8 Landmarks)",
            "📌 Pre-Packaged Pitch Zone Templates (Quick 1-Click Anchors)"
        ],
        index=0 if st.session_state.get(mode_key, "custom") == "custom" else 1,
        horizontal=True,
        key=f"{key_prefix}_strategy_radio"
    )
    is_custom_mode = calib_mode.startswith("🧩")
    st.session_state[mode_key] = "custom" if is_custom_mode else "preset"

    dst_points = np.zeros((4, 2), dtype=np.float32)
    landmark_labels: List[str] = []
    zone_title = ""
    zone_desc = ""
    has_duplicates = False
    has_collinear = False

    if is_custom_mode:
        st.markdown("""
        <div style="background:#0f172a; border:1px solid #1e293b; border-radius:8px; padding:10px 14px; margin-bottom:12px;">
            <div style="font-size:0.80rem; font-weight:700; color:#38bdf8;">🧩 Flexible Multi-Point Landmark Anchor Studio:</div>
            <div style="font-size:0.75rem; color:#94a3b8; margin-top:2px;">Select between 4 and 8 visible chalk landmarks. 4 points provides an exact quad; 5+ points enables robust RANSAC Least-Squares fitting to eliminate human click jitter.</div>
        </div>
        """, unsafe_allow_html=True)

        n_col1, n_col2 = st.columns([1.6, 2.4])
        with n_col1:
            num_anchors = st.radio(
                "Number of Anchors (N >= 4)",
                [4, 5, 6, 8],
                index=0,
                horizontal=True,
                key=f"{key_prefix}_num_anchors"
            )
        with n_col2:
            solver_hint = "⚡ <b>Exact Quad</b> (Analytic cv2.getPerspectiveTransform)" if num_anchors == 4 else f"🛡️ <b>RANSAC Overdetermined</b> ({num_anchors} points Least-Squares)"
            st.markdown(f"<div style='font-size:0.78rem; color:#cbd5e1; margin-top:6px;'>Active Solver: {solver_hint}</div>", unsafe_allow_html=True)

        # Quick preset buttons for 4-point common visible groupings
        if num_anchors == 4:
            q_c1, q_c2, q_c3, q_c4 = st.columns(4)
            with q_c1:
                if st.button("⚡ Right Box + Arc 'D'", key=f"{key_prefix}_q_rbox", use_container_width=True):
                    st.session_state[f"{key_prefix}_l1"] = landmark_names[0]
                    st.session_state[f"{key_prefix}_l2"] = landmark_names[2]
                    st.session_state[f"{key_prefix}_l3"] = landmark_names[3]
                    st.session_state[f"{key_prefix}_l4"] = landmark_names[4]
                    st.session_state[pts_key] = []
                    st.session_state[applied_key] = False
                    st.session_state[ver_key] = st.session_state.get(ver_key, 0) + 1
                    st.rerun()
            with q_c2:
                if st.button("⚡ Midfield & Center Circle", key=f"{key_prefix}_q_mid", use_container_width=True):
                    st.session_state[f"{key_prefix}_l1"] = "Halfway Line — Far Touchline (Top) [52.5m, 0m]"
                    st.session_state[f"{key_prefix}_l2"] = "Center Circle — Right Apex [61.65m, 34m]"
                    st.session_state[f"{key_prefix}_l3"] = "Halfway Line — Near Touchline (Bottom) [52.5m, 68m]"
                    st.session_state[f"{key_prefix}_l4"] = "Center Circle — Left Apex [43.35m, 34m]"
                    st.session_state[pts_key] = []
                    st.session_state[applied_key] = False
                    st.session_state[ver_key] = st.session_state.get(ver_key, 0) + 1
                    st.rerun()
            with q_c3:
                if st.button("⚡ Left Box + Arc 'D'", key=f"{key_prefix}_q_lbox", use_container_width=True):
                    st.session_state[f"{key_prefix}_l1"] = "Left 18-Yard Box — Goal Line Far [0m, 13.84m]"
                    st.session_state[f"{key_prefix}_l2"] = "Left 18-Yard Box — Infield Far Corner [16.5m, 13.84m]"
                    st.session_state[f"{key_prefix}_l3"] = "Left Penalty Arc 'D' — Infield Apex [20.15m, 34m]"
                    st.session_state[f"{key_prefix}_l4"] = "Left 18-Yard Box — Goal Line Near [0m, 54.16m]"
                    st.session_state[pts_key] = []
                    st.session_state[applied_key] = False
                    st.session_state[ver_key] = st.session_state.get(ver_key, 0) + 1
                    st.rerun()
            with q_c4:
                if st.button("⚡ 4 Pitch Corner Flags", key=f"{key_prefix}_q_flags", use_container_width=True):
                    st.session_state[f"{key_prefix}_l1"] = "Corner Flag — Left Far (Top-Left) [0m, 0m]"
                    st.session_state[f"{key_prefix}_l2"] = "Corner Flag — Right Far (Top-Right) [105m, 0m]"
                    st.session_state[f"{key_prefix}_l3"] = "Corner Flag — Right Near (Bottom-Right) [105m, 68m]"
                    st.session_state[f"{key_prefix}_l4"] = "Corner Flag — Left Near (Bottom-Left) [0m, 68m]"
                    st.session_state[pts_key] = []
                    st.session_state[applied_key] = False
                    st.session_state[ver_key] = st.session_state.get(ver_key, 0) + 1
                    st.rerun()

        default_catalog = [
            landmark_names[0],  # Right 18-Yard Box — Infield Far Corner [88.5m, 13.84m]
            landmark_names[2],  # Right 18-Yard Box — Goal Line Far [105m, 13.84m]
            landmark_names[3],  # Right 18-Yard Box — Goal Line Near [105m, 54.16m]
            landmark_names[1],  # Right 18-Yard Box — Infield Near Corner [88.5m, 54.16m]
            "Right Penalty Arc 'D' — Infield Apex [84.85m, 34m]",
            "Center Circle — Right Apex [61.65m, 34m]",
            "Halfway Line — Far Touchline (Top) [52.5m, 0m]",
            "Halfway Line — Near Touchline (Bottom) [52.5m, 68m]"
        ]

        cols_per_row = 4 if num_anchors in [4, 8] else (3 if num_anchors == 6 else 5)
        current_lms = []
        for row_start in range(0, num_anchors, cols_per_row):
            row_cols = st.columns(min(cols_per_row, num_anchors - row_start))
            for c_idx, col in enumerate(row_cols):
                lm_idx = row_start + c_idx
                l_key = f"{key_prefix}_l{lm_idx+1}"
                def_val = default_catalog[lm_idx % len(default_catalog)]
                cur_val = st.session_state.get(l_key, def_val)
                sel_idx = landmark_names.index(cur_val) if cur_val in landmark_names else 0
                with col:
                    sel = st.selectbox(f"Landmark {lm_idx+1} (L{lm_idx+1})", landmark_names, index=sel_idx, key=l_key)
                    current_lms.append(sel)

        last_lms = st.session_state.get(f"{key_prefix}_last_lms")
        if last_lms is not None and (last_lms != current_lms or len(st.session_state.get(pts_key, [])) > num_anchors):
            if st.session_state.get(pts_key):
                st.session_state[pts_key] = []
                st.session_state[applied_key] = False
                st.session_state[ver_key] = st.session_state.get(ver_key, 0) + 1
                st.toast("🔄 Landmark configuration updated. Clicked points reset.", icon="🎯")
        st.session_state[f"{key_prefix}_last_lms"] = current_lms

        # Validation: Check for duplicate landmarks
        has_duplicates = len(set(current_lms)) < num_anchors
        raw_pts = [FIFA_CHALK_LANDMARKS[s] for s in current_lms]

        # Validation: Check for collinearity (3 points on the same line)
        from perception.homography import HomographyTransformer
        has_collinear, collinear_triplets = HomographyTransformer.check_collinearity(raw_pts, threshold_area=0.5)

        if has_duplicates:
            st.warning(f"⚠️ Duplicate landmarks selected! All {num_anchors} landmark anchors must be distinct pitch features.")
        elif has_collinear and num_anchors == 4:
            t = collinear_triplets[0]
            st.error(f"⚠️ Collinear Warning: Points L{t[0]+1}, L{t[1]+1}, and L{t[2]+1} lie on the exact same straight line! 4-point homography requires non-collinear quad vertices.")
        elif has_collinear and num_anchors > 4:
            t = collinear_triplets[0]
            st.warning(f"⚠️ Note: Points L{t[0]+1}, L{t[1]+1}, and L{t[2]+1} lie on a straight line. RANSAC will use all {num_anchors} points to anchor 2D space.")

        dst_points = np.float32(raw_pts)
        landmark_labels = current_lms
        zone_title = f"Custom {num_anchors}-Point Landmark Anchor"
        zone_desc = f"Anchoring L1 ➔ L{num_anchors} across your selected physical field landmarks."

    else:
        # Pre-packaged zone templates
        template_options = list(GROUND_TRUTH_TEMPLATES.keys())
        cur_idx = 0
        if st.session_state[tmpl_key] in template_options:
            cur_idx = template_options.index(st.session_state[tmpl_key])
        template_name = st.selectbox("Packaged Pitch Zone Template", template_options, index=cur_idx, key=f"{key_prefix}_dst_tmpl")
        st.session_state[tmpl_key] = template_name

        tmpl_meta = GROUND_TRUTH_TEMPLATES[template_name]
        dst_points = tmpl_meta["points"]
        landmark_labels = tmpl_meta["labels"]
        zone_title = tmpl_meta["title"]
        zone_desc = tmpl_meta["desc"]

    # Studio Display: Clickable Canvas (Left) + 2D Metric Guide Thumbnail (Right)
    col_left, col_right = st.columns([1.35, 1.0])

    with col_right:
        st.markdown("###### 🗺️ 2D Metric Pitch Destination Guide")
        st.caption(f"**Zone**: {zone_title}<br/>💡 <i>{zone_desc}</i>", unsafe_allow_html=True)

        thumb_canvas = render_landmarks_pitch_thumbnail(
            points_metric=dst_points,
            labels=landmark_labels,
            title=zone_title,
            thumb_width=500,
            thumb_height=320
        )
        st.image(cv2.cvtColor(thumb_canvas, cv2.COLOR_BGR2RGB), caption=f"Pitch Destination: {zone_title}", width="stretch")

        guide_items = "".join([f"<b>{i+1}. L{i+1}: {landmark_labels[i]}</b><br/>" for i in range(len(landmark_labels))])
        st.markdown(f"""
        <div style="background:#0f172a; border-left:4px solid #38bdf8; border-radius:6px; padding:10px 14px; margin-top:8px;">
            <div style="font-size:0.80rem; font-weight:700; color:#38bdf8;">📌 Ordered Landmark Clicking Guide ({len(dst_points)} Points):</div>
            <div style="font-size:0.75rem; color:#cbd5e1; margin-top:4px;">
                {guide_items}
            </div>
            <div style="font-size:0.72rem; color:#94a3b8; margin-top:6px; border-top:1px solid #1e293b; padding-top:4px;">
                Click each of the {len(dst_points)} landmarks on the broadcast footage in this exact sequence.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_left:
        target_count = len(dst_points)
        st.markdown("###### 📷 Camera Clickable Canvas (Image Space)")
        st.caption(f"Click the corresponding {target_count} landmarks on the TV broadcast turf below:")

        # Dynamic canvas size respecting frame aspect ratio
        aspect = rf_w / float(max(1, rf_h))
        calib_w = 960
        calib_h = int(calib_w / aspect)
        c_canvas = cv2.resize(raw_frame, (calib_w, calib_h), interpolation=cv2.INTER_LINEAR)

        # Draw recorded points on the display canvas
        pts_disp = [
            (int(pt[0] / (rf_w / float(calib_w))), int(pt[1] / (rf_h / float(calib_h))))
            for pt in st.session_state[pts_key]
        ]

        colors = [(0, 240, 255), (0, 255, 120), (255, 0, 200), (255, 200, 0), (255, 150, 50), (100, 255, 255), (200, 100, 255), (255, 255, 100)]
        for idx, (px, py) in enumerate(pts_disp):
            col = colors[idx % len(colors)]
            cv2.circle(c_canvas, (px, py), 7, col, -1, lineType=cv2.LINE_AA)
            cv2.circle(c_canvas, (px, py), 10, (255, 255, 255), 2, lineType=cv2.LINE_AA)
            cv2.putText(c_canvas, f"L{idx+1} ({st.session_state[pts_key][idx][0]:.0f},{st.session_state[pts_key][idx][1]:.0f})",
                        (px + 12, py - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, lineType=cv2.LINE_AA)

        if len(pts_disp) >= 2:
            is_closed = (len(pts_disp) == target_count)
            cv2.polylines(c_canvas, [np.array(pts_disp, dtype=np.int32)], isClosed=is_closed, color=(0, 240, 255), thickness=2, lineType=cv2.LINE_AA)

        if len(pts_disp) == target_count and target_count >= 4:
            overlay_poly = c_canvas.copy()
            cv2.fillPoly(overlay_poly, [np.array(pts_disp, dtype=np.int32)], (0, 220, 255))
            cv2.addWeighted(overlay_poly, 0.25, c_canvas, 0.75, 0, c_canvas)

        pil_canvas = Image.fromarray(cv2.cvtColor(c_canvas, cv2.COLOR_BGR2RGB))

        click_data = streamlit_image_coordinates(
            pil_canvas,
            use_column_width=True,
            key=f"{key_prefix}_canvas_v{st.session_state.get(ver_key, 0)}"
        )

        if click_data is not None:
            c_time = click_data.get('unix_time', 0)
            if c_time and c_time != st.session_state.get(time_key, 0):
                st.session_state[time_key] = c_time
                elem_w = float(click_data.get('width', 0)) or float(calib_w)
                elem_h = float(click_data.get('height', 0)) or float(calib_h)
                scale_u = rf_w / elem_w
                scale_v = rf_h / elem_h

                if 'x' in click_data and 'y' in click_data and len(st.session_state[pts_key]) < target_count:
                    new_pt = (float(click_data['x'] * scale_u), float(click_data['y'] * scale_v))
                    st.session_state[pts_key].append(new_pt)
                    st.rerun()

    # Controls Deck
    pts_n = len(st.session_state[pts_key])
    target_count = len(dst_points)
    a_c1, a_c2, a_c3, a_c4 = st.columns([1.5, 1.2, 1.2, 2.0])
    with a_c1:
        can_apply = (pts_n == target_count) and not (is_custom_mode and has_duplicates) and not (target_count == 4 and has_collinear)
        if st.button("💾 Apply Manual Calibration", type="primary", disabled=not can_apply, width="stretch", key=f"{key_prefix}_btn_apply"):
            st.session_state[enabled_key] = True
            st.session_state[applied_key] = True
            src_pts = np.float32(st.session_state[pts_key])
            transformer.calibrate(src_pts, dst_points, frame_width=rf_w, frame_height=rf_h)
            st.success(f"✅ Manual calibration applied successfully ({target_count} points, method={'Exact Quad' if target_count==4 else 'RANSAC Best-Fit'})!")
            st.rerun()

    with a_c2:
        if st.button("↩ Undo Point", disabled=len(st.session_state[pts_key]) == 0, width="stretch", key=f"{key_prefix}_btn_undo"):
            st.session_state[pts_key].pop()
            st.session_state[ver_key] = st.session_state.get(ver_key, 0) + 1
            st.session_state[time_key] = 0
            st.session_state[applied_key] = False
            st.rerun()

    with a_c3:
        if st.button("🔄 Reset Points", width="stretch", key=f"{key_prefix}_btn_reset"):
            st.session_state[pts_key] = []
            st.session_state[applied_key] = False
            st.session_state[enabled_key] = False
            st.session_state[ver_key] = st.session_state.get(ver_key, 0) + 1
            st.session_state[time_key] = 0
            transformer.set_camera_mode('sideline')
            st.rerun()

    with a_c4:
        if pts_n == target_count:
            st.markdown(f"<span class='badge-pass'>{target_count} OF {target_count} POINTS READY TO APPLY</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"<span class='badge-warn'>CAPTURED {pts_n} OF {target_count} POINTS</span>", unsafe_allow_html=True)

    # Instant Candidate Pitch Projection Preview Feature
    if pts_n >= 4 and pts_n == target_count:
        st.markdown("---")
        show_cand_preview = st.checkbox(
            "⚡ Preview Full Projected Pitch Mask & Chalk Lines on this Frame",
            value=True,
            key=f"{key_prefix}_preview_cand"
        )
        if show_cand_preview:
            try:
                from perception.homography import HomographyTransformer
                src_cand = np.float32(st.session_state[pts_key])
                temp_trans = HomographyTransformer(frame_width=rf_w, frame_height=rf_h)
                temp_trans.calibrate(src_cand, dst_points, frame_width=rf_w, frame_height=rf_h)
                cand_img = render_pitch_projection_preview(raw_frame, temp_trans.H)
                st.image(
                    cv2.cvtColor(cand_img, cv2.COLOR_BGR2RGB),
                    caption=f"Instant Candidate Projection on Frame ({zone_title} - {target_count} Points)",
                    width="stretch"
                )
            except Exception as e:
                st.warning(f"Could not compute projection with selected points: {e}")

    # Captured Correspondences Table with Live Reprojection Residuals
    if st.session_state[pts_key]:
        residuals = []
        inliers = None
        if pts_n >= 4:
            try:
                from perception.homography import HomographyTransformer
                src_cand = np.float32(st.session_state[pts_key])
                dst_cand = np.float32(dst_points[:pts_n])
                temp_trans = HomographyTransformer(frame_width=rf_w, frame_height=rf_h)
                temp_trans.calibrate(src_cand, dst_cand, frame_width=rf_w, frame_height=rf_h)
                res_dict = temp_trans.compute_reprojection_residuals(src_cand, dst_cand)
                residuals = res_dict.get('residuals', [])
                inliers = getattr(temp_trans, 'inlier_mask', None)
            except Exception:
                residuals = []

        rows = []
        for idx, pt in enumerate(st.session_state[pts_key]):
            dst = dst_points[idx] if idx < len(dst_points) else (0, 0)
            lbl = landmark_labels[idx] if idx < len(landmark_labels) else f"Target {idx+1}"
            if idx < len(residuals):
                res_val = residuals[idx]
                is_inlier = bool(inliers[idx][0]) if inliers is not None and idx < len(inliers) else True
                if res_val < 1.0:
                    badge = f"`{res_val:.2f} px` • <span style='color:#10b981; font-weight:700;'>Sub-Pixel Fit</span>"
                elif res_val < 3.0 or is_inlier:
                    badge = f"`{res_val:.2f} px` • <span style='color:#38bdf8; font-weight:700;'>Valid Inlier</span>"
                else:
                    badge = f"`{res_val:.2f} px` • <span style='color:#ef4444; font-weight:700;'>High Residual</span>"
            else:
                badge = "Captured (Awaiting 4+ pts)"
            rows.append(f"| **L{idx+1}: {lbl}** | `({pt[0]:.1f}, {pt[1]:.1f})` px | `({dst[0]:.1f}m, {dst[1]:.1f}m)` | {badge} |")

        st.markdown(f"""
        | Landmark | Screen Pixel $(u, v)$ | Metric Target $(X_m, Y_m)$ | Reprojection Residual Error |
        | :--- | :---: | :---: | :---: |
        {chr(10).join(rows)}
        """, unsafe_allow_html=True)

    return {
        "applied": st.session_state[applied_key],
        "enabled": st.session_state[enabled_key],
        "points": st.session_state[pts_key],
        "dst_points": dst_points,
        "title": zone_title
    }
