"""
Standard FIFA Football Pitch Model & Coordinate Specifications
Standard Pitch Dimensions: 105.0 meters length x 68.0 meters width.
Provides tactical keypoint coordinates and metric conversions.
"""

from typing import Tuple, Dict, List
import numpy as np


class PitchModel:
    """Standard 105m x 68m FIFA Football Pitch Geometry."""
    LENGTH: float = 105.0  # meters
    WIDTH: float = 68.0    # meters
    
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
    
    CENTER_CIRCLE_RADIUS: float = 9.15  # meters
    PENALTY_BOX_LENGTH: float = 16.5    # meters
    PENALTY_BOX_WIDTH: float = 40.32    # meters
    SIX_YARD_BOX_LENGTH: float = 5.5    # meters
    SIX_YARD_BOX_WIDTH: float = 18.32   # meters

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
        goal_top_y = (w - 7.32) / 2.0
        goal_bot_y = (w + 7.32) / 2.0
        mid_x = l / 2.0
        mid_y = w / 2.0

        return {
            'top_left_corner': (0.0, 0.0),
            'top_right_corner': (l, 0.0),
            'bottom_right_corner': (l, w),
            'bottom_left_corner': (0.0, w),
            'center_spot': (mid_x, mid_y),
            'left_penalty_spot': (11.0, mid_y),
            'right_penalty_spot': (l - 11.0, mid_y),
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
