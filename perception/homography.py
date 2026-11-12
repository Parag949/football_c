"""
Perspective Homography Transformation Matrix Engine
Computes planar perspective projection matrices (cv2.getPerspectiveTransform)
translating 2D broadcast camera perspective angles into flat tactical pitch coordinates (x, y),
and vice-versa for camera-plane visual augmentation overlays.
"""

import hashlib
import json
from typing import List, Tuple, Optional, Any, Union, Dict
import numpy as np
import cv2
try:
    from scipy.interpolate import RBFInterpolator
except Exception:
    RBFInterpolator = None
from perception.pitch_model import PitchModel


class HomographyTransformer:
    """
    Handles perspective projection between broadcast video camera pixels
    and flat 2D pitch coordinates in meters [0, 105] x [0, 68].
    Supports both planar 3x3 homography ('planar') and non-linear
    Thin-Plate Spline panorama warps ('panorama_tps') for curved wide-angle /
    stitched camera views.
    """

    def __init__(
        self,
        frame_width: Any = 1280,
        frame_height: int = 720,
        src_points: Optional[np.ndarray] = None,
        dst_points: Optional[np.ndarray] = None,
        camera_mode: str = "sideline",
        warp_mode: str = "planar"
    ):
        if isinstance(frame_width, str):
            camera_mode = frame_width
            frame_width = 1280
            frame_height = 720
        self.frame_width = int(frame_width)
        self.frame_height = int(frame_height)
        self.camera_mode = camera_mode
        self.warp_mode = warp_mode if warp_mode in ("planar", "panorama_tps") else "planar"
        self._tps_fwd = None
        self._tps_inv = None
        self._tps_smoothing = 1e-5
        self._init_matrices(src_points, dst_points)

    def _init_matrices(self, src_points: Optional[np.ndarray], dst_points: Optional[np.ndarray]) -> None:
        w, h = self.frame_width, self.frame_height
        if (
            src_points is not None
            and dst_points is not None
            and len(src_points) >= 4
            and len(dst_points) >= 4
        ):
            self.calibrate(src_points, dst_points, warp_mode=self.warp_mode)
            return

        if self.camera_mode == "goal_to_goal":
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

        self.is_custom_calibrated = False
        self.warp_mode = "planar"
        self._tps_fwd = None
        self._tps_inv = None
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
        self.warp_mode = "planar"
        self._tps_fwd = None
        self._tps_inv = None
        self._init_matrices(None, None)

    def update_dimensions(self, frame_width: int, frame_height: int) -> None:
        """Dynamically updates resolution and recalculates projection matrices."""
        frame_width = max(1, int(frame_width))
        frame_height = max(1, int(frame_height))
        if self.frame_width == frame_width and self.frame_height == frame_height:
            return
        old_w, old_h = max(1, self.frame_width), max(1, self.frame_height)
        self.frame_width = frame_width
        self.frame_height = frame_height

        if getattr(self, 'is_custom_calibrated', False) and hasattr(self, 'src_points') and len(self.src_points) >= 4:
            # Scale custom source coordinates proportionally
            scale_x = frame_width / float(old_w)
            scale_y = frame_height / float(old_h)
            new_src = np.float32([
                [pt[0] * scale_x, pt[1] * scale_y] for pt in self.src_points
            ])
            self.calibrate(
                new_src,
                self.dst_points,
                frame_width=frame_width,
                frame_height=frame_height,
                warp_mode=getattr(self, 'warp_mode', 'planar')
            )
        else:
            self._init_matrices(None, None)

    def _fit_tps_models(self, src_pts: np.ndarray, dst_pts: np.ndarray) -> bool:
        """Fits bidirectional normalized Thin-Plate Spline interpolators for curved panoramas."""
        if RBFInterpolator is None or len(src_pts) < 6:
            self._tps_fwd = None
            self._tps_inv = None
            return False

        w = float(max(1, self.frame_width))
        h = float(max(1, self.frame_height))
        src_norm = np.column_stack([src_pts[:, 0] / w, src_pts[:, 1] / h]).astype(np.float64)
        dst_norm = np.column_stack([
            dst_pts[:, 0] / PitchModel.LENGTH,
            dst_pts[:, 1] / PitchModel.WIDTH
        ]).astype(np.float64)

        try:
            self._tps_fwd = RBFInterpolator(
                src_norm,
                dst_norm,
                kernel="thin_plate_spline",
                degree=1,
                smoothing=self._tps_smoothing
            )
            self._tps_inv = RBFInterpolator(
                dst_norm,
                src_norm,
                kernel="thin_plate_spline",
                degree=1,
                smoothing=self._tps_smoothing
            )
            return True
        except Exception:
            self._tps_fwd = None
            self._tps_inv = None
            return False

    def calibrate(
        self,
        src_points: np.ndarray,
        dst_points: np.ndarray,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None,
        warp_mode: str = "planar"
    ) -> None:
        """
        Updates spatial calibration using N >= 4 point correspondences.
        - 'planar' (N == 4 exact quad or N > 4 RANSAC): 3x3 perspective homography.
        - 'panorama_tps' (N >= 6): smooth bidirectional Thin-Plate Spline warp alongside
          best-fit planar H fallback so wide curved panoramas bend along true chalk lines.
        """
        self.src_points = np.float32(src_points)
        self.dst_points = np.float32(dst_points)
        if frame_width is not None and frame_height is not None:
            self.frame_width = max(1, int(frame_width))
            self.frame_height = max(1, int(frame_height))
        self.is_custom_calibrated = True

        n_pts = len(self.src_points)
        if n_pts < 4:
            raise ValueError(f"Planar homography requires at least 4 point correspondences, got {n_pts}.")
        elif n_pts == 4:
            self.H = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
            self.inlier_mask = np.ones((4, 1), dtype=np.uint8)
        else:
            H_mat, mask = cv2.findHomography(self.src_points, self.dst_points, cv2.RANSAC, 3.0)
            if H_mat is not None:
                self.H = H_mat
                self.inlier_mask = mask
            else:
                H_ls, _ = cv2.findHomography(self.src_points, self.dst_points, 0)
                self.H = H_ls if H_ls is not None else cv2.getPerspectiveTransform(self.src_points[:4], self.dst_points[:4])
                self.inlier_mask = np.ones((n_pts, 1), dtype=np.uint8)

        try:
            self.H_inv = np.linalg.inv(self.H)
        except Exception:
            self.H_inv = np.linalg.pinv(self.H)
        self._normalize_projective_sign()

        if warp_mode == "panorama_tps" and n_pts >= 6 and self._fit_tps_models(self.src_points, self.dst_points):
            self.warp_mode = "panorama_tps"
            self.inlier_mask = np.ones((n_pts, 1), dtype=np.uint8)
        else:
            self.warp_mode = "planar"
            self._tps_fwd = None
            self._tps_inv = None

    def compute_reprojection_residuals(
        self,
        src_points: Optional[np.ndarray] = None,
        dst_points: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Calculates pixel reprojection residual errors for each landmark.
        Measures the Euclidean pixel distance between ground truth click (src)
        and where dst reprojects back into the camera image plane via H_inv.
        """
        srcs = self.src_points if src_points is None else np.float32(src_points)
        dsts = self.dst_points if dst_points is None else np.float32(dst_points)
        if srcs is None or dsts is None or len(srcs) == 0:
            return {'residuals': [], 'mean_error': 0.0, 'max_error': 0.0}

        residuals = []
        for s, d in zip(srcs, dsts):
            proj_px = self.pitch_to_image([(float(d[0]), float(d[1]))])
            if proj_px:
                err = float(np.linalg.norm(np.array(s) - np.array(proj_px[0])))
            else:
                err = 999.0
            residuals.append(err)

        return {
            'residuals': residuals,
            'mean_error': float(np.mean(residuals)) if residuals else 0.0,
            'max_error': float(np.max(residuals)) if residuals else 0.0
        }

    @staticmethod
    def check_collinearity(
        points: Union[np.ndarray, List[Tuple[float, float]]],
        threshold_area: float = 0.5
    ) -> Tuple[bool, List[Tuple[int, int, int]]]:
        """
        Checks if any 3 points lie on the same straight line using 2D triangle cross-product area.
        Returns (has_collinear, list_of_collinear_triplet_indices).
        """
        pts = list(points)
        n = len(pts)
        collinear_triplets = []
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    p1, p2, p3 = pts[i], pts[j], pts[k]
                    area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]))
                    if area < threshold_area:
                        collinear_triplets.append((i, j, k))
        return (len(collinear_triplets) > 0, collinear_triplets)

    def image_to_pitch(
        self,
        image_points: List[Tuple[float, float]],
        filter_out_of_bounds: bool = False,
        buffer_meters: float = 2.0
    ) -> List[Tuple[float, float]]:
        """
        Transforms 2D broadcast pixel coordinates (u, v) into metric pitch coordinates (X, Y) in meters.
        Supports both planar homography and Thin-Plate Spline panorama warping.
        """
        if image_points is None or len(image_points) == 0:
            return []

        if getattr(self, 'warp_mode', 'planar') == 'panorama_tps' and getattr(self, '_tps_fwd', None) is not None:
            pts_arr = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
            w = float(max(1, self.frame_width))
            h = float(max(1, self.frame_height))
            norm_in = np.column_stack([pts_arr[:, 0] / w, pts_arr[:, 1] / h])
            norm_out = self._tps_fwd(norm_in)
            mapped = np.column_stack([
                norm_out[:, 0] * PitchModel.LENGTH,
                norm_out[:, 1] * PitchModel.WIDTH
            ])
        else:
            pts = np.array(image_points, dtype=np.float32).reshape(-1, 1, 2)
            mapped = cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)

        coords = []
        for pt in mapped:
            raw_x, raw_y = float(pt[0]), float(pt[1])
            if not (np.isfinite(raw_x) and np.isfinite(raw_y)):
                continue
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
        """
        if pitch_points is None or len(pitch_points) == 0:
            return []

        if getattr(self, 'warp_mode', 'planar') == 'panorama_tps' and getattr(self, '_tps_inv', None) is not None:
            pts_arr = np.asarray(pitch_points, dtype=np.float64).reshape(-1, 2)
            w_f = float(max(1, self.frame_width))
            h_f = float(max(1, self.frame_height))
            norm_in = np.column_stack([
                pts_arr[:, 0] / PitchModel.LENGTH,
                pts_arr[:, 1] / PitchModel.WIDTH
            ])
            norm_out = self._tps_inv(norm_in)
            pixel_coords = []
            for row in norm_out:
                fx = float(row[0] * w_f)
                fy = float(row[1] * h_f)
                if not (np.isfinite(fx) and np.isfinite(fy)):
                    continue
                if filter_behind_camera and (fx < -2.0 * w_f or fx > 3.0 * w_f or fy < -2.0 * h_f or fy > 3.0 * h_f):
                    continue
                pixel_coords.append((int(round(fx)), int(round(fy))))
            return pixel_coords

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
        Uses cv2.remap + Thin-Plate Spline inverse mapping in panorama_tps mode, or cv2.warpPerspective in planar mode.
        """
        usable_w = max(1, width - 2 * padding)
        usable_h = max(1, height - 2 * padding)
        scale_x = usable_w / PitchModel.LENGTH
        scale_y = usable_h / PitchModel.WIDTH

        if getattr(self, 'warp_mode', 'planar') == 'panorama_tps' and getattr(self, '_tps_inv', None) is not None:
            # Evaluate inverse TPS on a compact coarse mesh and upscale bilinearly for sub-5ms orthorectification
            grid_w = min(width, 210)
            grid_h = min(height, 136)
            gx = np.linspace(0, width - 1, grid_w, dtype=np.float64)
            gy = np.linspace(0, height - 1, grid_h, dtype=np.float64)
            xv, yv = np.meshgrid(gx, gy)
            pitch_x = (xv - float(padding)) / scale_x
            pitch_y = (yv - float(padding)) / scale_y
            norm_in = np.column_stack([
                (pitch_x / PitchModel.LENGTH).ravel(),
                (pitch_y / PitchModel.WIDTH).ravel()
            ])
            norm_out = self._tps_inv(norm_in)
            fh, fw = frame.shape[:2]
            map_x_c = (norm_out[:, 0].reshape(grid_h, grid_w) * float(fw)).astype(np.float32)
            map_y_c = (norm_out[:, 1].reshape(grid_h, grid_w) * float(fh)).astype(np.float32)
            map_x = cv2.resize(map_x_c, (width, height), interpolation=cv2.INTER_LINEAR)
            map_y = cv2.resize(map_y_c, (width, height), interpolation=cv2.INTER_LINEAR)
            return cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        S = np.array([
            [scale_x, 0.0, float(padding)],
            [0.0, scale_y, float(padding)],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        H_warp = S @ self.H
        return cv2.warpPerspective(frame, H_warp, (width, height))

    def get_calibration_signature(self) -> str:
        """Deterministic hash of current dimensions, mode, and active calibration geometry."""
        payload = {
            "w": int(self.frame_width),
            "h": int(self.frame_height),
            "camera_mode": str(self.camera_mode),
            "warp_mode": str(getattr(self, "warp_mode", "planar")),
            "custom": bool(getattr(self, "is_custom_calibrated", False)),
            "src": np.round(self.src_points, 2).tolist() if getattr(self, "src_points", None) is not None else [],
            "dst": np.round(self.dst_points, 2).tolist() if getattr(self, "dst_points", None) is not None else [],
        }
        raw = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.md5(raw).hexdigest()[:12]

    def export_client_projection(self, nx: int = 36, ny: int = 24) -> Dict[str, Any]:
        """
        Serializes active projection parameters for browser Hardware Media Engine canvas rendering.
        Includes lightweight bilinear lookup grids when panorama_tps mode is active.
        """
        h_inv_list = [[round(float(val), 9) for val in row] for row in self.H_inv.tolist()]
        h_list = [[round(float(val), 9) for val in row] for row in self.H.tolist()]
        out: Dict[str, Any] = {
            "warp_mode": str(getattr(self, "warp_mode", "planar")),
            "H_inv": h_inv_list,
            "H": h_list,
            "calibration_sig": self.get_calibration_signature(),
            "p2i_lut": None,
            "i2p_lut": None,
        }

        if out["warp_mode"] == "panorama_tps" and getattr(self, "_tps_inv", None) is not None and getattr(self, "_tps_fwd", None) is not None:
            # Pitch [0..105]m x [0..68]m -> Image (u, v) LUT
            px_vals = np.linspace(0.0, PitchModel.LENGTH, nx)
            py_vals = np.linspace(0.0, PitchModel.WIDTH, ny)
            xg, yg = np.meshgrid(px_vals, py_vals)
            p_norm = np.column_stack([
                (xg / PitchModel.LENGTH).ravel(),
                (yg / PitchModel.WIDTH).ravel()
            ])
            uv_norm = self._tps_inv(p_norm)
            u_grid = (uv_norm[:, 0] * float(self.frame_width)).reshape(ny, nx)
            v_grid = (uv_norm[:, 1] * float(self.frame_height)).reshape(ny, nx)
            out["p2i_lut"] = {
                "nx": nx,
                "ny": ny,
                "u": [[round(float(u_grid[r, c]), 2) for c in range(nx)] for r in range(ny)],
                "v": [[round(float(v_grid[r, c]), 2) for c in range(nx)] for r in range(ny)],
            }

            # Image [0..w] x [0..h] -> Pitch (X_m, Y_m) LUT
            ux_vals = np.linspace(0.0, 1.0, nx)
            vy_vals = np.linspace(0.0, 1.0, ny)
            ug, vg = np.meshgrid(ux_vals, vy_vals)
            i_norm = np.column_stack([ug.ravel(), vg.ravel()])
            m_norm = self._tps_fwd(i_norm)
            x_grid = (m_norm[:, 0] * PitchModel.LENGTH).reshape(ny, nx)
            y_grid = (m_norm[:, 1] * PitchModel.WIDTH).reshape(ny, nx)
            out["i2p_lut"] = {
                "nx": nx,
                "ny": ny,
                "x": [[round(float(x_grid[r, c]), 3) for c in range(nx)] for r in range(ny)],
                "y": [[round(float(y_grid[r, c]), 3) for c in range(nx)] for r in range(ny)],
            }

        return out

