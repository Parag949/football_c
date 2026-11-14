"""
Standard FIFA Football Pitch Model & Coordinate Specifications
Standard Pitch Dimensions: 105.0 meters length x 68.0 meters width.
Provides tactical keypoint coordinates, metric conversions, canonical landmark
catalogues, and pitch geometry helper functions according to IFAB Law 1.
"""

from typing import Tuple, Dict, List, Any
import numpy as np


class PitchModel:
    """Standard 105m x 68m FIFA Football Pitch Geometry."""
    LENGTH: float = 105.0              # meters
    WIDTH: float = 68.0                # meters
    CENTER_CIRCLE_RADIUS: float = 9.15  # meters
    PENALTY_BOX_LENGTH: float = 16.5    # meters
    PENALTY_BOX_WIDTH: float = 40.32    # meters
    SIX_YARD_BOX_LENGTH: float = 5.5    # meters
    SIX_YARD_BOX_WIDTH: float = 18.32   # meters
    PENALTY_SPOT_DISTANCE: float = 11.0 # meters
    GOAL_WIDTH: float = 7.32           # meters
    PENALTY_D_RADIUS: float = 9.15      # meters
    
    # Landmark coordinates in metric system (x: 0 to 105, y: 0 to 68)
    # Origin (0,0) is top-left corner
    KEYPOINTS: Dict[str, Tuple[float, float]] = {
        'top_left_corner': (0.0, 0.0),
        'top_right_corner': (105.0, 0.0),
        'bottom_right_corner': (105.0, 68.0),
        'bottom_left_corner': (0.0, 68.0),
        'center_spot': (52.5, 34.0),
        'left_penalty_spot': (11.0, 34.0),
        'right_penalty_spot': (94.0, 34.0),
        'left_goal_top': (0.0, 30.34),
        'left_goal_bottom': (0.0, 37.66),
        'right_goal_top': (105.0, 30.34),
        'right_goal_bottom': (105.0, 37.66),
        # Penalty Boxes (16.5m from goal line, 40.32m width)
        'left_box_top_left': (0.0, 13.84),
        'left_box_top_right': (16.5, 13.84),
        'left_box_bottom_right': (16.5, 54.16),
        'left_box_bottom_left': (0.0, 54.16),
        'right_box_top_right': (105.0, 13.84),
        'right_box_top_left': (88.5, 13.84),
        'right_box_bottom_left': (88.5, 54.16),
        'right_box_bottom_right': (105.0, 54.16),
    }

    @classmethod
    def get_keypoints(
        cls,
        length: float = 105.0,
        width: float = 68.0,
    ) -> Dict[str, Tuple[float, float]]:
        """
        Dynamically calculates metric coordinates for all canonical landmarks on a pitch
        of arbitrary dimensions (length x width) according to FIFA IFAB Law 1:
        - 18-yard box is always fixed at 16.5m from goal lines and 40.32m centered on width.
        - 6-yard box is always fixed at 5.5m from goal lines and 18.32m centered on width.
        - Halfway line and center circle are always at length / 2 and width / 2.
        """
        l = float(length)
        w = float(width)
        box_top_y = (w - cls.PENALTY_BOX_WIDTH) / 2.0
        box_bot_y = (w + cls.PENALTY_BOX_WIDTH) / 2.0
        goal_top_y = (w - cls.GOAL_WIDTH) / 2.0
        goal_bot_y = (w + cls.GOAL_WIDTH) / 2.0
        mid_x = l / 2.0
        mid_y = w / 2.0

        return {
            'top_left_corner': (0.0, 0.0),
            'top_right_corner': (l, 0.0),
            'bottom_right_corner': (l, w),
            'bottom_left_corner': (0.0, w),
            'center_spot': (mid_x, mid_y),
            'left_penalty_spot': (cls.PENALTY_SPOT_DISTANCE, mid_y),
            'right_penalty_spot': (l - cls.PENALTY_SPOT_DISTANCE, mid_y),
            'left_goal_top': (0.0, goal_top_y),
            'left_goal_bottom': (0.0, goal_bot_y),
            'right_goal_top': (l, goal_top_y),
            'right_goal_bottom': (l, goal_bot_y),
            'left_box_top_left': (0.0, box_top_y),
            'left_box_top_right': (cls.PENALTY_BOX_LENGTH, box_top_y),
            'left_box_bottom_right': (cls.PENALTY_BOX_LENGTH, box_bot_y),
            'left_box_bottom_left': (0.0, box_bot_y),
            'right_box_top_right': (l, box_top_y),
            'right_box_top_left': (l - cls.PENALTY_BOX_LENGTH, box_top_y),
            'right_box_bottom_left': (l - cls.PENALTY_BOX_LENGTH, box_bot_y),
            'right_box_bottom_right': (l, box_bot_y),
        }

    @classmethod
    def get_turf_boundary(
        cls,
        length: float = 105.0,
        width: float = 68.0,
        is_goal_cam: bool = False
    ) -> List[Tuple[float, float]]:
        """Returns metric perimeter coordinates of the turf boundary."""
        l = float(length)
        w = float(width)
        if is_goal_cam:
            mid_x = l / 2.0
            return [(mid_x, 0.0), (l, 0.0), (l, w), (mid_x, w)]
        return [(0.0, 0.0), (l, 0.0), (l, w), (0.0, w)]

    @classmethod
    def get_markings_polylines(
        cls,
        length: float = 105.0,
        width: float = 68.0,
        is_goal_cam: bool = False
    ) -> List[List[Tuple[float, float]]]:
        """Returns list of metric polylines representing FIFA pitch lines."""
        l = float(length)
        w = float(width)
        mid_x = l / 2.0
        box_top = (w - cls.PENALTY_BOX_WIDTH) / 2.0
        box_bot = (w + cls.PENALTY_BOX_WIDTH) / 2.0
        six_top = (w - cls.SIX_YARD_BOX_WIDTH) / 2.0
        six_bot = (w + cls.SIX_YARD_BOX_WIDTH) / 2.0

        if is_goal_cam:
            return [
                # Visible Touchlines & Goal Line
                [(mid_x, 0.0), (l, 0.0)],
                [(mid_x, w), (l, w)],
                [(l, 0.0), (l, w)],
                # Halfway line
                [(mid_x, 0.0), (mid_x, w)],
                # 18-yard box attacking end (closed)
                [(l - cls.PENALTY_BOX_LENGTH, box_top), (l, box_top), (l, box_bot), (l - cls.PENALTY_BOX_LENGTH, box_bot), (l - cls.PENALTY_BOX_LENGTH, box_top)],
                # 6-yard goal area attacking end (closed)
                [(l - cls.SIX_YARD_BOX_LENGTH, six_top), (l, six_top), (l, six_bot), (l - cls.SIX_YARD_BOX_LENGTH, six_bot), (l - cls.SIX_YARD_BOX_LENGTH, six_top)]
            ]
        else:
            return [
                # Touchlines & Goal lines
                [(0.0, 0.0), (l, 0.0)],
                [(0.0, w), (l, w)],
                [(0.0, 0.0), (0.0, w)],
                [(l, 0.0), (l, w)],
                # Halfway line
                [(mid_x, 0.0), (mid_x, w)],
                # 18-yard box left & right (closed)
                [(0.0, box_top), (cls.PENALTY_BOX_LENGTH, box_top), (cls.PENALTY_BOX_LENGTH, box_bot), (0.0, box_bot), (0.0, box_top)],
                [(l - cls.PENALTY_BOX_LENGTH, box_top), (l, box_top), (l, box_bot), (l - cls.PENALTY_BOX_LENGTH, box_bot), (l - cls.PENALTY_BOX_LENGTH, box_top)],
                # 6-yard goal area left & right (closed)
                [(0.0, six_top), (cls.SIX_YARD_BOX_LENGTH, six_top), (cls.SIX_YARD_BOX_LENGTH, six_bot), (0.0, six_bot), (0.0, six_top)],
                [(l - cls.SIX_YARD_BOX_LENGTH, six_top), (l, six_top), (l, six_bot), (l - cls.SIX_YARD_BOX_LENGTH, six_bot), (l - cls.SIX_YARD_BOX_LENGTH, six_top)]
            ]

    @classmethod
    def get_center_circle_polygon(
        cls,
        length: float = 105.0,
        width: float = 68.0,
        num_points: int = 36
    ) -> List[Tuple[float, float]]:
        """Calculates discrete metric points for the center circle perimeter."""
        mid_x = float(length) / 2.0
        mid_y = float(width) / 2.0
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        return [(float(mid_x + cls.CENTER_CIRCLE_RADIUS * np.cos(a)), float(mid_y + cls.CENTER_CIRCLE_RADIUS * np.sin(a))) for a in angles]

    @classmethod
    def metric_to_pixel(
        cls,
        x: float,
        y: float,
        canvas_width: int,
        canvas_height: int,
        padding: int = 40
    ) -> Tuple[int, int]:
        """
        Converts tactical metric coordinates (meters) to 2D pitch canvas pixels.
        """
        usable_w = max(1.0, float(canvas_width - 2 * padding))
        usable_h = max(1.0, float(canvas_height - 2 * padding))
        scale_x = usable_w / cls.LENGTH
        scale_y = usable_h / cls.WIDTH
        
        px = int(padding + x * scale_x)
        py = int(padding + y * scale_y)
        return (px, py)

    @classmethod
    def pixel_to_metric(
        cls,
        px: int,
        py: int,
        canvas_width: int,
        canvas_height: int,
        padding: int = 40
    ) -> Tuple[float, float]:
        """
        Converts 2D pitch canvas pixels back to tactical metric coordinates.
        """
        usable_w = max(1.0, float(canvas_width - 2 * padding))
        usable_h = max(1.0, float(canvas_height - 2 * padding))
        scale_x = usable_w / cls.LENGTH
        scale_y = usable_h / cls.WIDTH
        
        mx = (px - padding) / scale_x
        my = (py - padding) / scale_y
        return (max(0.0, min(cls.LENGTH, mx)), max(0.0, min(cls.WIDTH, my)))


# Module-level canonical pitch dimension constants
LENGTH: float = PitchModel.LENGTH
WIDTH: float = PitchModel.WIDTH
CENTER_CIRCLE_RADIUS: float = PitchModel.CENTER_CIRCLE_RADIUS
PENALTY_BOX_LENGTH: float = PitchModel.PENALTY_BOX_LENGTH
PENALTY_BOX_WIDTH: float = PitchModel.PENALTY_BOX_WIDTH
SIX_YARD_BOX_LENGTH: float = PitchModel.SIX_YARD_BOX_LENGTH
SIX_YARD_BOX_WIDTH: float = PitchModel.SIX_YARD_BOX_WIDTH
PENALTY_SPOT_DISTANCE: float = PitchModel.PENALTY_SPOT_DISTANCE
GOAL_WIDTH: float = PitchModel.GOAL_WIDTH
PENALTY_D_RADIUS: float = PitchModel.PENALTY_D_RADIUS


def get_pitch_landmarks_def(pitch_length: float = 105.0, pitch_width: float = 68.0) -> List[Dict[str, Any]]:
    """
    Dynamically generates the 27 canonical pitch landmarks for arbitrary pitch dimensions (L x W)
    in accordance with FIFA IFAB Law 1:
    - 18-yard box is locked to 16.5m from goal lines and 40.32m centered on width.
    - 6-yard box is locked to 5.5m from goal lines and 18.32m centered on width.
    - Halfway line, center circle, and penalty marks adapt to length and width.
    """
    l = float(pitch_length)
    w = float(pitch_width)
    box_top_y = round((w - PitchModel.PENALTY_BOX_WIDTH) / 2.0, 2)
    box_bot_y = round((w + PitchModel.PENALTY_BOX_WIDTH) / 2.0, 2)
    six_top_y = round((w - PitchModel.SIX_YARD_BOX_WIDTH) / 2.0, 2)
    six_bot_y = round((w + PitchModel.SIX_YARD_BOX_WIDTH) / 2.0, 2)
    mid_x = round(l / 2.0, 2)
    mid_y = round(w / 2.0, 2)
    q1_x = round(l * 0.25, 2)
    q3_x = round(l * 0.75, 2)

    return [
        # 4 Corners
        {"id": "TL",    "label": "Top-Left Corner",             "xm": 0.0,    "ym": 0.0,   "role": "corner"},
        {"id": "TR",    "label": "Top-Right Corner",            "xm": l,      "ym": 0.0,   "role": "corner"},
        {"id": "BR",    "label": "Bottom-Right Corner",         "xm": l,      "ym": w,     "role": "corner"},
        {"id": "BL",    "label": "Bottom-Left Corner",          "xm": 0.0,    "ym": w,     "role": "corner"},
        # 13-Point Curved Panorama / Touchline + Spine Anchors
        {"id": "TM25",  "label": f"Top Quarter-Left ({q1_x}m)",  "xm": q1_x,   "ym": 0.0,   "role": "edge"},
        {"id": "TM50",  "label": f"Top Halfway ({mid_x}m)",        "xm": mid_x,  "ym": 0.0,   "role": "edge"},
        {"id": "TM75",  "label": f"Top Quarter-Right ({q3_x}m)", "xm": q3_x,   "ym": 0.0,   "role": "edge"},
        {"id": "BM25",  "label": f"Bottom Quarter-Left ({q1_x}m)","xm": q1_x,  "ym": w,     "role": "edge"},
        {"id": "BM50",  "label": f"Bottom Halfway ({mid_x}m)",     "xm": mid_x,  "ym": w,     "role": "edge"},
        {"id": "BM75",  "label": f"Bottom Quarter-Right ({q3_x}m)","xm": q3_x, "ym": w,     "role": "edge"},
        {"id": "LP",    "label": f"Left 18-Yd Line Midpoint (16.5m, {mid_y}m)", "xm": 16.5, "ym": mid_y,  "role": "inner"},
        {"id": "CS",    "label": f"Center Spot ({mid_x}m, {mid_y}m)", "xm": mid_x, "ym": mid_y,  "role": "inner"},
        {"id": "RP",    "label": f"Right 18-Yd Line Midpoint ({round(l - 16.5, 2)}m, {mid_y}m)", "xm": round(l - 16.5, 2), "ym": mid_y,  "role": "inner"},
        # Penalty Boxes, Center Circle & 6-Yard Boxes
        {"id": "L18T",  "label": "Left 18-Yd Box Top Corner",   "xm": 16.5,   "ym": box_top_y, "role": "inner"},
        {"id": "L18B",  "label": "Left 18-Yd Box Bottom Corner","xm": 16.5,   "ym": box_bot_y, "role": "inner"},
        {"id": "R18T",  "label": "Right 18-Yd Box Top Corner",  "xm": round(l - 16.5, 2), "ym": box_top_y, "role": "inner"},
        {"id": "R18B",  "label": "Right 18-Yd Box Bottom Corner","xm": round(l - 16.5, 2), "ym": box_bot_y, "role": "inner"},
        {"id": "CCT",   "label": f"Center Circle Top ({mid_x}m)",   "xm": mid_x,  "ym": round(mid_y - PitchModel.CENTER_CIRCLE_RADIUS, 2), "role": "inner"},
        {"id": "CCB",   "label": f"Center Circle Bottom ({mid_x}m)","xm": mid_x, "ym": round(mid_y + PitchModel.CENTER_CIRCLE_RADIUS, 2), "role": "inner"},
        {"id": "L6T",   "label": "Left 6-Yd Box Top Corner",    "xm": 5.5,    "ym": six_top_y, "role": "inner"},
        {"id": "R6T",   "label": "Right 6-Yd Box Top Corner",   "xm": round(l - 5.5, 2), "ym": six_top_y, "role": "inner"},
        {"id": "L6B",   "label": "Left 6-Yd Box Bottom Corner", "xm": 5.5,    "ym": six_bot_y, "role": "inner"},
        {"id": "R6B",   "label": "Right 6-Yd Box Bottom Corner","xm": round(l - 5.5, 2), "ym": six_bot_y, "role": "inner"},
        {"id": "L18GT", "label": "Left Goal-Line 18-Yd Top",    "xm": 0.0,    "ym": box_top_y, "role": "edge"},
        {"id": "L18GB", "label": "Left Goal-Line 18-Yd Bottom", "xm": 0.0,    "ym": box_bot_y, "role": "edge"},
        {"id": "R18GT", "label": "Right Goal-Line 18-Yd Top",   "xm": l,      "ym": box_top_y, "role": "edge"},
        {"id": "R18GB", "label": "Right Goal-Line 18-Yd Bottom","xm": l,      "ym": box_bot_y, "role": "edge"},
    ]


def get_fifa_chalk_landmarks(
    pitch_length: float = 105.0,
    pitch_width: float = 68.0
) -> Dict[str, Tuple[float, float]]:
    """
    Dynamically generates the 29 canonical FIFA chalk landmarks for arbitrary pitch dimensions
    according to IFAB Law 1 chalk line specifications.
    """
    l = float(pitch_length)
    w = float(pitch_width)
    x_halfway = l / 2.0
    y_center = w / 2.0
    r_circle = PitchModel.CENTER_CIRCLE_RADIUS

    x_spot_right = l - PitchModel.PENALTY_SPOT_DISTANCE
    x_spot_left = PitchModel.PENALTY_SPOT_DISTANCE

    x_box_right = l - PitchModel.PENALTY_BOX_LENGTH
    x_box_left = PitchModel.PENALTY_BOX_LENGTH
    y_box_top = (w - PitchModel.PENALTY_BOX_WIDTH) / 2.0
    y_box_bottom = (w + PitchModel.PENALTY_BOX_WIDTH) / 2.0

    x_d_apex_right = x_spot_right - r_circle
    x_d_apex_left = x_spot_left + r_circle

    x_goal_area_right = l - PitchModel.SIX_YARD_BOX_LENGTH
    x_goal_area_left = PitchModel.SIX_YARD_BOX_LENGTH
    y_goal_area_top = (w - PitchModel.SIX_YARD_BOX_WIDTH) / 2.0
    y_goal_area_bottom = (w + PitchModel.SIX_YARD_BOX_WIDTH) / 2.0

    return {
        # Right Goalmouth & Attacking Half (X > 52.5m)
        "Right 18-Yard Box — Infield Far Corner [88.5m, 13.84m]": (x_box_right, y_box_top),
        "Right 18-Yard Box — Infield Near Corner [88.5m, 54.16m]": (x_box_right, y_box_bottom),
        "Right 18-Yard Box — Goal Line Far [105m, 13.84m]": (l, y_box_top),
        "Right 18-Yard Box — Goal Line Near [105m, 54.16m]": (l, y_box_bottom),
        "Right Penalty Arc 'D' — Infield Apex [84.85m, 34m]": (x_d_apex_right, y_center),
        "Right Penalty Spot [94m, 34m]": (x_spot_right, y_center),
        "Right 6-Yard Box — Infield Far Corner [99.5m, 24.84m]": (x_goal_area_right, y_goal_area_top),
        "Right 6-Yard Box — Infield Near Corner [99.5m, 43.16m]": (x_goal_area_right, y_goal_area_bottom),
        "Corner Flag — Right Far (Top-Right) [105m, 0m]": (l, 0.0),
        "Corner Flag — Right Near (Bottom-Right) [105m, 68m]": (l, w),
        "Right Goal Center Line [105m, 34m]": (l, y_center),

        # Midfield & Center Circle
        "Halfway Line — Far Touchline (Top) [52.5m, 0m]": (x_halfway, 0.0),
        "Halfway Line — Near Touchline (Bottom) [52.5m, 68m]": (x_halfway, w),
        "Center Spot [52.5m, 34m]": (x_halfway, y_center),
        "Center Circle — Right Apex [61.65m, 34m]": (x_halfway + r_circle, y_center),
        "Center Circle — Left Apex [43.35m, 34m]": (x_halfway - r_circle, y_center),
        "Center Circle — Far Apex (Top) [52.5m, 24.85m]": (x_halfway, y_center - r_circle),
        "Center Circle — Near Apex (Bottom) [52.5m, 43.15m]": (x_halfway, y_center + r_circle),

        # Left Goalmouth & Defending Half (X < 52.5m)
        "Left 18-Yard Box — Infield Far Corner [16.5m, 13.84m]": (x_box_left, y_box_top),
        "Left 18-Yard Box — Infield Near Corner [16.5m, 54.16m]": (x_box_left, y_box_bottom),
        "Left 18-Yard Box — Goal Line Far [0m, 13.84m]": (0.0, y_box_top),
        "Left 18-Yard Box — Goal Line Near [0m, 54.16m]": (0.0, y_box_bottom),
        "Left Penalty Arc 'D' — Infield Apex [20.15m, 34m]": (x_d_apex_left, y_center),
        "Left Penalty Spot [11m, 34m]": (x_spot_left, y_center),
        "Left 6-Yard Box — Infield Far Corner [5.5m, 24.84m]": (x_goal_area_left, y_goal_area_top),
        "Left 6-Yard Box — Infield Near Corner [5.5m, 43.16m]": (x_goal_area_left, y_goal_area_bottom),
        "Corner Flag — Left Far (Top-Left) [0m, 0m]": (0.0, 0.0),
        "Corner Flag — Left Near (Bottom-Left) [0m, 68m]": (0.0, w),
        "Left Goal Center Line [0m, 34m]": (0.0, y_center),
    }


# Module-level canonical collections and layout configurations
PITCH_LANDMARKS_DEF: List[Dict[str, Any]] = get_pitch_landmarks_def(LENGTH, WIDTH)
MESH_HANDLES_DEF: List[Dict[str, Any]] = PITCH_LANDMARKS_DEF[:13]
FIFA_CHALK_LANDMARKS: Dict[str, Tuple[float, float]] = get_fifa_chalk_landmarks(LENGTH, WIDTH)

LAYOUT_4_IDS: List[str] = ["TL", "TR", "BR", "BL"]
LAYOUT_4_GOAL_IDS: List[str] = ["TR", "BR", "BM50", "TM50"]
LAYOUT_8_IDS: List[str] = ["TL", "TR", "BR", "BL", "L18T", "L18B", "R18T", "R18B"]
LAYOUT_8_GOAL_IDS: List[str] = ["TR", "BR", "R18T", "R18B", "R18GT", "R18GB", "TM50", "BM50"]
LAYOUT_13_IDS: List[str] = ["TL", "TR", "BR", "BL", "TM25", "TM50", "TM75", "BM50", "CS", "L18T", "L18B", "R18T", "R18B"]
LAYOUT_21_IDS: List[str] = [item["id"] for item in PITCH_LANDMARKS_DEF[:21]]
LAYOUT_ENDZONE_IDS: List[str] = ["TR", "BR", "R18T", "R18B", "TM50", "BM50", "CS"]
