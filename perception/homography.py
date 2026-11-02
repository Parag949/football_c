"""
Perspective Homography Transformation Matrix Engine
Computes planar perspective projection matrices (cv2.getPerspectiveTransform)
translating 2D broadcast camera perspective angles into flat tactical pitch coordinates (x, y),
and vice-versa for camera-plane visual augmentation overlays.
"""

from typing import List, Tuple, Optional, Any, Union
import numpy as np
import cv2
from perception.pitch_model import PitchModel


class HomographyTransformer:
    """
    Handles perspective projection between broadcast video camera pixels
    and flat 2D pitch coordinates in meters [0, 105] x [0, 68].
    """
    
    def __init__(
        self,
        frame_width: Any = 1280,
        frame_height: int = 720,
        src_points: Optional[np.ndarray] = None,
        dst_points: Optional[np.ndarray] = None,
        camera_mode: str = "sideline"
    ):
        if isinstance(frame_width, str):
            camera_mode = frame_width
            frame_width = 1280
            frame_height = 720
        self.frame_width = int(frame_width)
        self.frame_height = int(frame_height)
        self.camera_mode = camera_mode
        self._init_matrices(src_points, dst_points)

    def _init_matrices(self, src_points: Optional[np.ndarray], dst_points: Optional[np.ndarray]) -> None:
        w, h = self.frame_width, self.frame_height
        if src_points is not None and dst_points is not None:
            self.src_points = np.float32(src_points)
            self.dst_points = np.float32(dst_points)
        elif self.camera_mode == "goal_to_goal":
            # Calibrated End-Zone Tactical Cam (FIFA 105m x 68m landmarks)
            self.src_points = np.float32([
                [int(w * 0.3161), int(h * 0.3481)],  # 18-yard box front-left
                [int(w * 0.7068), int(h * 0.3472)],  # 18-yard box front-right
                [int(w * 0.9474), int(h * 0.5667)],  # Halfway line right
                [int(w * 0.0745), int(h * 0.5787)],  # Halfway line left
            ])
            self.dst_points = np.float32([
                [88.5, 13.84],   # 18-yard box front-left (X=88.5m, Y=13.84m)
                [88.5, 54.16],   # 18-yard box front-right (X=88.5m, Y=54.16m)
                [52.5, 68.00],   # Halfway line right (X=52.5m, Y=68.0m)
                [52.5, 0.00],    # Halfway line left (X=52.5m, Y=0.0m)
            ])
        else:
            # Calibrated Sideline Broadcast camera view (horizontal length, vertical width)
            self.src_points = np.float32([
                [int(w * 0.105), int(h * 0.335)],  # Far touchline left
                [int(w * 0.835), int(h * 0.335)],  # Far touchline right
                [int(w * 0.965), int(h * 0.705)],  # Near touchline right
                [int(w * 0.035), int(h * 0.705)],  # Near touchline left
            ])
            self.dst_points = np.float32([
                [22.0, 3.0],   # Far touchline left (Y=3m, near top sideline)
                [83.0, 3.0],   # Far touchline right (Y=3m, near top sideline)
                [85.0, 65.0],  # Near touchline right (Y=65m, near bottom sideline)
                [20.0, 65.0],  # Near touchline left (Y=65m, near bottom sideline)
            ])
            
        self.is_custom_calibrated = (src_points is not None and dst_points is not None)
        self.H = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        self.H_inv = np.linalg.inv(self.H)
        self._normalize_projective_sign()

    def _normalize_projective_sign(self) -> None:
        """Ensures points on the pitch in front of camera have positive projective depth (w > 0)."""
        try:
            center_test = self.H_inv @ np.array([52.5, 34.0, 1.0], dtype=np.float64)
            if center_test[2] < 0:
                self.H_inv = -self.H_inv
                self.H = -self.H
        except Exception:
            pass

    def set_camera_mode(self, camera_mode: str) -> None:
        """Switches projection preset between 'sideline' and 'goal_to_goal'."""
        self.camera_mode = camera_mode
        self.is_custom_calibrated = False
        self._init_matrices(None, None)

    def update_dimensions(self, frame_width: int, frame_height: int) -> None:
        """Dynamically updates resolution and recalculates projection matrices."""
        if self.frame_width == frame_width and self.frame_height == frame_height:
            return
        old_w, old_h = max(1, self.frame_width), max(1, self.frame_height)
        self.frame_width = frame_width
        self.frame_height = frame_height

        if getattr(self, 'is_custom_calibrated', False):
            # Scale custom source coordinates proportionally
            scale_x = frame_width / float(old_w)
            scale_y = frame_height / float(old_h)
            self.src_points = np.float32([
                [pt[0] * scale_x, pt[1] * scale_y] for pt in self.src_points
            ])
            self.H = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
            self.H_inv = np.linalg.inv(self.H)
            self._normalize_projective_sign()
        else:
            self._init_matrices(None, None)

    def calibrate(
        self,
        src_points: np.ndarray,
        dst_points: np.ndarray,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None
    ) -> None:
        """Updates homography matrix using 4 point correspondences."""
        self.src_points = np.float32(src_points)
        self.dst_points = np.float32(dst_points)
        if frame_width is not None and frame_height is not None:
            self.frame_width = int(frame_width)
            self.frame_height = int(frame_height)
        self.is_custom_calibrated = True
        self.H = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        self.H_inv = np.linalg.inv(self.H)
        self._normalize_projective_sign()

    def image_to_pitch(
        self,
        image_points: List[Tuple[float, float]],
        filter_out_of_bounds: bool = False,
        buffer_meters: float = 2.0
    ) -> List[Tuple[float, float]]:
        """
        Transforms 2D broadcast pixel coordinates (u, v) into metric pitch coordinates (X, Y) in meters.
        If filter_out_of_bounds=True, points landing outside the pitch perimeter + buffer (e.g. dugout/stands)
        are discarded rather than clamped, preventing sideline non-players from corrupting DAA geometry.
        """
        if image_points is None or len(image_points) == 0:
            return []
            
        pts = np.array(image_points, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(pts, self.H)
        coords = []
        for pt in transformed.reshape(-1, 2):
            raw_x, raw_y = float(pt[0]), float(pt[1])
            if filter_out_of_bounds:
                if -buffer_meters <= raw_x <= PitchModel.LENGTH + buffer_meters and -buffer_meters <= raw_y <= PitchModel.WIDTH + buffer_meters:
                    coords.append((raw_x, raw_y))
            else:
                coords.append((raw_x, raw_y))
        return coords

    def pitch_to_image(
        self,
        pitch_points: List[Tuple[float, float]],
        filter_behind_camera: bool = False
    ) -> List[Tuple[int, int]]:
        """
        Transforms flat tactical pitch metric coordinates (X, Y) back into broadcast camera pixels (u, v)
        for neon polygon overlays and passing lane projections.
        Safely checks projective denominator w to prevent inverted wrap-around behind the camera horizon.
        """
        if pitch_points is None or len(pitch_points) == 0:
            return []
            
        pixel_coords = []
        for pt in pitch_points:
            vec = np.array([float(pt[0]), float(pt[1]), 1.0], dtype=np.float64)
            proj = self.H_inv @ vec
            w = proj[2]
            if filter_behind_camera and w <= 1e-4:
                continue
            if abs(w) > 1e-7:
                px = int(round(proj[0] / w))
                py = int(round(proj[1] / w))
                pixel_coords.append((px, py))
            else:
                pixel_coords.append((0, 0))
        return pixel_coords

    def warp_to_pitch_plane(
        self,
        frame: np.ndarray,
        width: int = 1050,
        height: int = 680,
        padding: int = 0
    ) -> np.ndarray:
        """
        Un-warps broadcast camera footage into an orthorectified top-down bird's-eye pitch image.
        Uses scaling and translation matrix S mapping [0, 105]m x [0, 68]m to (width, height) pixels,
        optionally accounting for tactical pitch canvas padding.
        """
        usable_w = width - 2 * padding
        usable_h = height - 2 * padding
        scale_x = usable_w / PitchModel.LENGTH
        scale_y = usable_h / PitchModel.WIDTH
        S = np.array([
            [scale_x, 0.0, float(padding)],
            [0.0, scale_y, float(padding)],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        H_warp = S @ self.H
        return cv2.warpPerspective(frame, H_warp, (width, height))

