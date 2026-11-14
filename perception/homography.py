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
try:
    from scipy.optimize import minimize_scalar
except Exception:
    minimize_scalar = None
from perception.pitch_model import PitchModel


class HomographyTransformer:
    """
    Handles perspective projection between broadcast video camera pixels
    and flat 2D pitch coordinates in meters [0, 105] x [0, 68].
    Supports planar 3x3 homography ('planar'), universal physical camera lens distortion
    modeling ('universal'), and Thin-Plate Spline panorama warps ('panorama_tps')
    for curved wide-angle / stitched camera views.
    """

    def __init__(
        self,
        frame_width: Any = 1280,
        frame_height: int = 720,
        src_points: Optional[np.ndarray] = None,
        dst_points: Optional[np.ndarray] = None,
        camera_mode: str = "sideline",
        warp_mode: str = "planar",
        lens_lambda: float = 0.0,
        pitch_length: Optional[float] = None,
        pitch_width: Optional[float] = None,
    ):
        if isinstance(frame_width, str):
            camera_mode = frame_width
            frame_width = 1280
            frame_height = 720
        self.frame_width = int(frame_width)
        self.frame_height = int(frame_height)
        self.camera_mode = camera_mode
        self.warp_mode = warp_mode if warp_mode in ("planar", "panorama_tps", "universal", "auto") else "planar"
        self.lens_lambda = float(lens_lambda)
        self.pitch_length = float(pitch_length) if pitch_length is not None else float(PitchModel.LENGTH)
        self.pitch_width = float(pitch_width) if pitch_width is not None else float(PitchModel.WIDTH)
        self._update_optical_geometry()
        self._tps_fwd = None
        self._tps_inv = None
        self._tps_smoothing = 1e-5
        self._init_matrices(src_points, dst_points)

    def _update_optical_geometry(self) -> None:
        """Computes principal optical center and radius normalization factor."""
        self.cx = self.frame_width / 2.0
        self.cy = self.frame_height / 2.0
        self.r_norm = float(np.hypot(self.cx, self.cy))

    def undistort_camera_points(
        self,
        points: Union[np.ndarray, List[Tuple[float, float]]],
        lens_lambda: Optional[float] = None
    ) -> np.ndarray:
        """
        Undistorts camera pixel coordinates (u_d, v_d) -> (u_u, v_u) using
        the Fitzgibbon Division Model:
            x_u = x_d / (1 + lambda * (x_d^2 + y_d^2))
        """
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        lam = float(self.lens_lambda if lens_lambda is None else lens_lambda)
        if abs(lam) < 1e-7 or len(pts) == 0:
            return pts.copy()

        xd = (pts[:, 0] - self.cx) / self.r_norm
        yd = (pts[:, 1] - self.cy) / self.r_norm
        rd2 = xd**2 + yd**2
        denom = np.maximum(1e-4, 1.0 + lam * rd2)
        xu = xd / denom
        yu = yd / denom
        uu = self.cx + xu * self.r_norm
        vu = self.cy + yu * self.r_norm
        return np.column_stack([uu, vu])

    def distort_camera_points(
        self,
        points: Union[np.ndarray, List[Tuple[float, float]]],
        lens_lambda: Optional[float] = None
    ) -> np.ndarray:
        """
        Distorts ideal camera pixel coordinates (u_u, v_u) -> (u_d, v_d) using
        the exact closed-form analytical inverse of the Division Model:
            factor = 2 / (1 + sqrt(max(0, 1 - 4 * lambda * r_u^2)))
        """
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        lam = float(self.lens_lambda if lens_lambda is None else lens_lambda)
        if abs(lam) < 1e-7 or len(pts) == 0:
            return pts.copy()

        xu = (pts[:, 0] - self.cx) / self.r_norm
        yu = (pts[:, 1] - self.cy) / self.r_norm
        ru2 = xu**2 + yu**2
        disc = np.maximum(0.0, 1.0 - 4.0 * lam * ru2)
        factor = 2.0 / np.maximum(1e-7, 1.0 + np.sqrt(disc))
        xd = xu * factor
        yd = yu * factor
        ud = self.cx + xd * self.r_norm
        vd = self.cy + yd * self.r_norm
        return np.column_stack([ud, vd])

    def fit_universal_lens_model(
        self,
        src_points: np.ndarray,
        dst_points: np.ndarray,
        lambda_bounds: Tuple[float, float] = (-0.35, 0.15),
        fixed_lambda: Optional[float] = None
    ) -> Tuple[Optional[np.ndarray], float]:
        """
        Jointly estimates optical lens distortion lambda and planar homography H
        minimizing Euclidean reprojection error across all point correspondences.
        """
        src_arr = np.asarray(src_points, dtype=np.float64).reshape(-1, 2)
        dst_arr = np.asarray(dst_points, dtype=np.float64).reshape(-1, 2)
        n_pts = len(src_arr)
        if n_pts < 4:
            return None, 0.0

        if fixed_lambda is not None:
            lam = float(fixed_lambda)
            undist = self.undistort_camera_points(src_arr, lens_lambda=lam)
            H_cand, _ = cv2.findHomography(undist, dst_arr, 0) if n_pts > 4 else (cv2.getPerspectiveTransform(undist.astype(np.float32), dst_arr.astype(np.float32)), None)
            return H_cand, lam

        if minimize_scalar is None or n_pts < 6:
            H_cand, _ = cv2.findHomography(src_arr, dst_arr, 0) if n_pts > 4 else (cv2.getPerspectiveTransform(src_arr.astype(np.float32), dst_arr.astype(np.float32)), None)
            return H_cand, 0.0

        pitch_h = np.column_stack([dst_arr, np.ones(n_pts, dtype=np.float64)])

        def objective(lam: float) -> float:
            undist = self.undistort_camera_points(src_arr, lens_lambda=lam)
            H_m, _ = cv2.findHomography(undist, dst_arr, 0)
            if H_m is None or abs(np.linalg.det(H_m)) < 1e-12:
                return 1e6
            try:
                H_inv = np.linalg.inv(H_m)
            except Exception:
                return 1e6

            proj_u = (H_inv @ pitch_h.T).T
            w = proj_u[:, 2]
            if float(np.mean(w)) < 0.0:
                w = -w
                proj_u = -proj_u
                H_inv = -H_inv
            if np.any(w <= 1e-4):
                return 1e6
            uu = proj_u[:, 0] / w
            vu = proj_u[:, 1] / w
            undist_est = np.column_stack([uu, vu])
            dist_est = self.distort_camera_points(undist_est, lens_lambda=lam)
            return float(np.mean(np.linalg.norm(src_arr - dist_est, axis=1)))

        try:
            res = minimize_scalar(objective, bounds=lambda_bounds, method='bounded')
            best_lam = float(res.x)
            if abs(best_lam) < 0.005:
                best_lam = 0.0
            undist = self.undistort_camera_points(src_arr, lens_lambda=best_lam)
            H_opt, _ = cv2.findHomography(undist, dst_arr, 0)
            if H_opt is None:
                H_opt, _ = cv2.findHomography(src_arr, dst_arr, 0)
                best_lam = 0.0
            return H_opt, best_lam
        except Exception:
            H_opt, _ = cv2.findHomography(src_arr, dst_arr, 0)
            return H_opt, 0.0

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
        self.lens_lambda = 0.0
        self._tps_fwd = None
        self._tps_inv = None
        self.H = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        self.H_inv = np.linalg.inv(self.H)
        self._normalize_projective_sign()

    def _normalize_projective_sign(self) -> None:
        """Ensures points on the pitch in front of camera have positive projective depth (w > 0)."""
        try:
            if getattr(self, "dst_points", None) is not None and len(self.dst_points) > 0:
                cx = float(np.mean(self.dst_points[:, 0]))
                cy = float(np.mean(self.dst_points[:, 1]))
            else:
                cx, cy = 52.5, 34.0
            center_test = self.H_inv @ np.array([cx, cy, 1.0], dtype=np.float64)
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
        self.lens_lambda = 0.0
        self._tps_fwd = None
        self._tps_inv = None
        self._init_matrices(None, None)

    def update_dimensions(self, frame_width: int, frame_height: int) -> None:
        """Dynamically updates resolution and recalculates projection matrices."""
        if frame_width is None or frame_height is None:
            return
        frame_width = max(1, int(frame_width))
        frame_height = max(1, int(frame_height))
        if self.frame_width == frame_width and self.frame_height == frame_height:
            return
        old_w, old_h = max(1, self.frame_width), max(1, self.frame_height)
        self.frame_width = frame_width
        self.frame_height = frame_height
        self._update_optical_geometry()

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
                warp_mode=getattr(self, 'warp_mode', 'planar'),
                lens_lambda=getattr(self, 'lens_lambda', 0.0)
            )
        else:
            self._init_matrices(None, None)

    def _fit_tps_models(self, src_pts: np.ndarray, dst_pts: np.ndarray) -> bool:
        """
        Fits residual Thin-Plate Spline correctors on top of the global 3x3 Homography (H / H_inv).
        Separating global projective perspective (H) from smooth local lens/stitching residuals
        prevents TPS extrapolation blow-ups near frame boundaries on ultrawide panoramas.
        """
        if RBFInterpolator is None or len(src_pts) < 4:
            self._tps_fwd = None
            self._tps_inv = None
            return False

        w = float(max(1, self.frame_width))
        h = float(max(1, self.frame_height))
        src_arr = np.asarray(src_pts, dtype=np.float64).reshape(-1, 2)
        dst_arr = np.asarray(dst_pts, dtype=np.float64).reshape(-1, 2)

        # De-duplicate nearly identical points so RBF matrix never becomes singular
        unique_idx = []
        for idx, pt in enumerate(src_arr):
            if all(np.hypot(pt[0] - src_arr[u][0], pt[1] - src_arr[u][1]) > 1.5 for u in unique_idx):
                unique_idx.append(idx)
        if len(unique_idx) < 6:
            self._tps_fwd = None
            self._tps_inv = None
            return False
        src_arr = src_arr[unique_idx]
        dst_arr = dst_arr[unique_idx]

        # Projective baseline predictions via H and H_inv
        base_pitch = cv2.perspectiveTransform(
            src_arr.astype(np.float32).reshape(-1, 1, 2), self.H
        ).reshape(-1, 2).astype(np.float64)
        base_img = cv2.perspectiveTransform(
            dst_arr.astype(np.float32).reshape(-1, 1, 2), self.H_inv
        ).reshape(-1, 2).astype(np.float64)

        if not (np.all(np.isfinite(base_pitch)) and np.all(np.isfinite(base_img))):
            self._tps_fwd = None
            self._tps_inv = None
            return False

        src_norm = np.column_stack([src_arr[:, 0] / w, src_arr[:, 1] / h])
        dst_norm = np.column_stack([dst_arr[:, 0] / self.pitch_length, dst_arr[:, 1] / self.pitch_width])

        res_pitch_norm = np.column_stack([
            (dst_arr[:, 0] - base_pitch[:, 0]) / self.pitch_length,
            (dst_arr[:, 1] - base_pitch[:, 1]) / self.pitch_width,
        ])
        res_img_norm = np.column_stack([
            (src_arr[:, 0] - base_img[:, 0]) / w,
            (src_arr[:, 1] - base_img[:, 1]) / h,
        ])

        try:
            # Global H captures perspective slope; degree=1 TPS on the small residual stays uniquely solvable and smooth
            self._tps_fwd = RBFInterpolator(
                src_norm,
                res_pitch_norm,
                kernel="thin_plate_spline",
                degree=1,
                smoothing=self._tps_smoothing
            )
            self._tps_inv = RBFInterpolator(
                dst_norm,
                res_img_norm,
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
        warp_mode: str = "planar",
        lens_lambda: Optional[float] = None,
        pitch_length: Optional[float] = None,
        pitch_width: Optional[float] = None,
    ) -> None:
        """
        Updates spatial calibration using N >= 4 point correspondences.
        - 'planar': 3x3 perspective homography (RANSAC / LS).
        - 'universal': Parametric physical camera lens model (Fitzgibbon Division Model / Brown-Conrady)
                       jointly optimizing optical curvature lambda and planar homography H.
        - 'panorama_tps': Global 3x3 perspective H + smooth bounded TPS residual warp.
        - 'auto': Chooses 'planar' when N < 6 or perspective error <= 1.5px, else upgrades to 'universal'
                  or 'panorama_tps'.
        """
        self.src_points = np.float32(src_points)
        self.dst_points = np.float32(dst_points)
        if len(self.src_points) != len(self.dst_points):
            raise ValueError(f"Correspondence count mismatch: src={len(self.src_points)}, dst={len(self.dst_points)}")
        if frame_width is not None and frame_height is not None:
            self.frame_width = max(1, int(frame_width))
            self.frame_height = max(1, int(frame_height))
            self._update_optical_geometry()
        if pitch_length is not None:
            self.pitch_length = float(pitch_length)
        if pitch_width is not None:
            self.pitch_width = float(pitch_width)
        self.is_custom_calibrated = True

        n_pts = len(self.src_points)
        if n_pts < 4:
            raise ValueError(f"Planar homography requires at least 4 point correspondences, got {n_pts}.")

        # Direct Universal Lens Distortion Mode
        if warp_mode == "universal":
            H_mat, best_lam = self.fit_universal_lens_model(
                self.src_points, self.dst_points, fixed_lambda=lens_lambda
            )
            if H_mat is not None and np.all(np.isfinite(H_mat)) and abs(np.linalg.det(H_mat)) > 1e-12:
                self.H = H_mat
                self.lens_lambda = best_lam
                self.warp_mode = "universal"
                self.inlier_mask = np.ones((n_pts, 1), dtype=np.uint8)
                try:
                    self.H_inv = np.linalg.inv(self.H)
                except Exception:
                    self.H_inv = np.linalg.pinv(self.H)
                self._normalize_projective_sign()
                self._tps_fwd = None
                self._tps_inv = None
                return

        H_mat = None
        mask = None
        if n_pts == 4:
            try:
                cand = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
                if cand is not None and np.all(np.isfinite(cand)) and abs(np.linalg.det(cand)) > 1e-12:
                    H_mat = cand
                    mask = np.ones((4, 1), dtype=np.uint8)
            except Exception:
                H_mat = None
        else:
            try:
                H_ls, _ = cv2.findHomography(self.src_points, self.dst_points, 0)
                if H_ls is not None and np.all(np.isfinite(H_ls)) and abs(np.linalg.det(H_ls)) > 1e-12:
                    H_mat = H_ls
                    mask = np.ones((n_pts, 1), dtype=np.uint8)
                else:
                    H_r, r_mask = cv2.findHomography(self.src_points, self.dst_points, cv2.RANSAC, 5.0)
                    if H_r is not None and np.all(np.isfinite(H_r)) and abs(np.linalg.det(H_r)) > 1e-12:
                        H_mat = H_r
                        mask = r_mask
            except Exception:
                H_mat = None

        if H_mat is None:
            # Safe non-singular fallback from default preset if user provided collinear/degenerate points
            self.is_custom_calibrated = False
            self.lens_lambda = 0.0
            self._init_matrices(None, None)
            return

        self.H = H_mat
        self.inlier_mask = mask if mask is not None else np.ones((n_pts, 1), dtype=np.uint8)

        try:
            self.H_inv = np.linalg.inv(self.H)
        except Exception:
            self.H_inv = np.linalg.pinv(self.H)
        self._normalize_projective_sign()

        want_tps = (warp_mode == "panorama_tps")
        if warp_mode == "auto" and n_pts >= 6:
            # Measure global planar fit residual error; upgrade to curved if curve > 1.5 px
            proj_back = cv2.perspectiveTransform(
                self.dst_points.astype(np.float32).reshape(-1, 1, 2), self.H_inv
            ).reshape(-1, 2)
            max_planar_err = float(np.max(np.linalg.norm(proj_back - self.src_points, axis=1))) if len(proj_back) else 0.0
            if max_planar_err > 1.5:
                # First attempt physical camera lens model (Fitzgibbon Division Model)
                H_univ, best_lam = self.fit_universal_lens_model(self.src_points, self.dst_points)
                if H_univ is not None and abs(best_lam) > 0.005:
                    try:
                        H_u_inv = np.linalg.inv(H_univ)
                        pts_h = np.column_stack([self.dst_points, np.ones(n_pts, dtype=np.float64)])
                        proj_u = (H_u_inv @ pts_h.T).T
                        w_u = proj_u[:, 2]
                        if float(np.mean(w_u)) < 0.0:
                            H_u_inv = -H_u_inv
                            proj_u = -proj_u
                            w_u = -w_u
                        if not np.any(w_u <= 1e-4):
                            uu = proj_u[:, 0] / w_u
                            vu = proj_u[:, 1] / w_u
                            undist_est = np.column_stack([uu, vu])
                            dist_est = self.distort_camera_points(undist_est, lens_lambda=best_lam)
                            univ_err = float(np.max(np.linalg.norm(self.src_points - dist_est, axis=1)))
                            if univ_err < max_planar_err:
                                self.H = H_univ
                                self.H_inv = H_u_inv
                                self.lens_lambda = best_lam
                                self.warp_mode = "universal"
                                self.inlier_mask = np.ones((n_pts, 1), dtype=np.uint8)
                                self._normalize_projective_sign()
                                self._tps_fwd = None
                                self._tps_inv = None
                                return
                    except Exception:
                        pass
                want_tps = True

        if want_tps and n_pts >= 4 and self._fit_tps_models(self.src_points, self.dst_points):
            self.warp_mode = "panorama_tps"
            self.lens_lambda = 0.0
            self.inlier_mask = np.ones((n_pts, 1), dtype=np.uint8)
        else:
            self.warp_mode = "planar"
            self.lens_lambda = 0.0
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
        Supports planar homography, universal physical lens model, and Thin-Plate Spline panorama warping.
        """
        if image_points is None or len(image_points) == 0:
            return []

        w_mode = getattr(self, 'warp_mode', 'planar')
        if w_mode == 'universal':
            pts_arr = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
            undist_pts = self.undistort_camera_points(pts_arr)
            coords = []
            for pt in undist_pts:
                vec = self.H @ np.array([pt[0], pt[1], 1.0], dtype=np.float64)
                w = vec[2]
                if filter_out_of_bounds and w <= 1e-4:
                    continue
                if abs(w) > 1e-7:
                    rx = float(vec[0] / w)
                    ry = float(vec[1] / w)
                else:
                    continue
                if not (np.isfinite(rx) and np.isfinite(ry)):
                    continue
                if filter_out_of_bounds:
                    if -buffer_meters <= rx <= self.pitch_length + buffer_meters and -buffer_meters <= ry <= self.pitch_width + buffer_meters:
                        coords.append((rx, ry))
                else:
                    coords.append((rx, ry))
            return coords

        if w_mode == 'panorama_tps' and getattr(self, '_tps_fwd', None) is not None:
            pts_arr = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
            w_frame = float(max(1, self.frame_width))
            h_frame = float(max(1, self.frame_height))
            norm_in = np.column_stack([pts_arr[:, 0] / w_frame, pts_arr[:, 1] / h_frame])
            res_out = self._tps_fwd(norm_in)
            coords = []
            for idx, pt in enumerate(pts_arr):
                vec = self.H @ np.array([pt[0], pt[1], 1.0], dtype=np.float64)
                w = vec[2]
                if filter_out_of_bounds and w <= 1e-4:
                    continue
                if abs(w) > 1e-7:
                    base_x = float(vec[0] / w)
                    base_y = float(vec[1] / w)
                else:
                    continue
                rx = base_x + float(res_out[idx, 0]) * self.pitch_length
                ry = base_y + float(res_out[idx, 1]) * self.pitch_width
                if not (np.isfinite(rx) and np.isfinite(ry)):
                    continue
                if filter_out_of_bounds:
                    if -buffer_meters <= rx <= self.pitch_length + buffer_meters and -buffer_meters <= ry <= self.pitch_width + buffer_meters:
                        coords.append((rx, ry))
                else:
                    coords.append((rx, ry))
            return coords
        else:
            pts_arr = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
            coords = []
            for pt in pts_arr:
                vec = self.H @ np.array([pt[0], pt[1], 1.0], dtype=np.float64)
                w = vec[2]
                if filter_out_of_bounds and w <= 1e-4:
                    continue
                if abs(w) > 1e-7:
                    rx = float(vec[0] / w)
                    ry = float(vec[1] / w)
                else:
                    continue
                if not (np.isfinite(rx) and np.isfinite(ry)):
                    continue
                if filter_out_of_bounds:
                    if -buffer_meters <= rx <= self.pitch_length + buffer_meters and -buffer_meters <= ry <= self.pitch_width + buffer_meters:
                        coords.append((rx, ry))
                else:
                    coords.append((rx, ry))
            return coords

    def pitch_to_image(
        self,
        pitch_points: List[Tuple[float, float]],
        filter_behind_camera: bool = True
    ) -> List[Tuple[int, int]]:
        """
        Transforms flat tactical pitch metric coordinates (X, Y) back into broadcast camera pixels (u, v)
        for neon polygon overlays and passing lane projections.
        """
        if pitch_points is None or len(pitch_points) == 0:
            return []

        w_mode = getattr(self, 'warp_mode', 'planar')
        w_f = float(max(1, self.frame_width))
        h_f = float(max(1, self.frame_height))

        if w_mode == 'universal':
            pts_arr = np.asarray(pitch_points, dtype=np.float64).reshape(-1, 2)
            undist_coords = []
            for pt in pts_arr:
                vec = np.array([float(pt[0]), float(pt[1]), 1.0], dtype=np.float64)
                proj = self.H_inv @ vec
                w = proj[2]
                if filter_behind_camera and w <= 1e-4:
                    continue
                if abs(w) > 1e-7:
                    u_u = proj[0] / w
                    v_u = proj[1] / w
                    undist_coords.append([u_u, v_u])
            if not undist_coords:
                return []
            dist_pts = self.distort_camera_points(np.array(undist_coords, dtype=np.float64))
            pixel_coords = []
            for pt in dist_pts:
                fx, fy = float(pt[0]), float(pt[1])
                if not (np.isfinite(fx) and np.isfinite(fy)):
                    continue
                if filter_behind_camera and (fx < -2.0 * w_f or fx > 3.0 * w_f or fy < -2.0 * h_f or fy > 3.0 * h_f):
                    continue
                px = int(np.clip(round(fx), -200000, 200000))
                py = int(np.clip(round(fy), -200000, 200000))
                pixel_coords.append((px, py))
            return pixel_coords

        if w_mode == 'panorama_tps' and getattr(self, '_tps_inv', None) is not None:
            pts_arr = np.asarray(pitch_points, dtype=np.float64).reshape(-1, 2)
            norm_in = np.column_stack([
                pts_arr[:, 0] / self.pitch_length,
                pts_arr[:, 1] / self.pitch_width
            ])
            res_out = self._tps_inv(norm_in)
            pixel_coords = []
            for idx, pt in enumerate(pts_arr):
                vec = np.array([pt[0], pt[1], 1.0], dtype=np.float64)
                proj = self.H_inv @ vec
                w = proj[2]
                if filter_behind_camera and w <= 1e-4:
                    continue
                if abs(w) > 1e-7:
                    base_u = proj[0] / w
                    base_v = proj[1] / w
                else:
                    base_u, base_v = 0.0, 0.0
                row = res_out[idx]
                fx = float(base_u + row[0] * w_f)
                fy = float(base_v + row[1] * h_f)
                if not (np.isfinite(fx) and np.isfinite(fy)):
                    continue
                if filter_behind_camera and (fx < -2.0 * w_f or fx > 3.0 * w_f or fy < -2.0 * h_f or fy > 3.0 * h_f):
                    continue
                px = int(np.clip(round(fx), -200000, 200000))
                py = int(np.clip(round(fy), -200000, 200000))
                pixel_coords.append((px, py))
            return pixel_coords

        pixel_coords = []
        for pt in pitch_points:
            vec = np.array([float(pt[0]), float(pt[1]), 1.0], dtype=np.float64)
            proj = self.H_inv @ vec
            w = proj[2]
            if filter_behind_camera and w <= 1e-4:
                continue
            if abs(w) > 1e-7:
                px = int(np.clip(round(proj[0] / w), -200000, 200000))
                py = int(np.clip(round(proj[1] / w), -200000, 200000))
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
        Uses cv2.remap for non-linear optical distortion / TPS modes, or cv2.warpPerspective in planar mode.
        """
        usable_w = max(1, width - 2 * padding)
        usable_h = max(1, height - 2 * padding)
        scale_x = usable_w / self.pitch_length
        scale_y = usable_h / self.pitch_width

        if getattr(self, 'warp_mode', 'planar') == 'universal':
            grid_w = min(width, 210)
            grid_h = min(height, 136)
            gx = np.linspace(0, width - 1, grid_w, dtype=np.float64)
            gy = np.linspace(0, height - 1, grid_h, dtype=np.float64)
            xv, yv = np.meshgrid(gx, gy)
            pitch_x = (xv - float(padding)) / scale_x
            pitch_y = (yv - float(padding)) / scale_y
            pitch_pts = np.column_stack([pitch_x.ravel(), pitch_y.ravel()])
            base_uv = cv2.perspectiveTransform(
                pitch_pts.astype(np.float32).reshape(-1, 1, 2), self.H_inv
            ).reshape(-1, 2).astype(np.float64)
            dist_uv = self.distort_camera_points(base_uv)
            fh, fw = frame.shape[:2]
            sx_frame = float(fw) / float(max(1, self.frame_width))
            sy_frame = float(fh) / float(max(1, self.frame_height))
            u_full = dist_uv[:, 0] * sx_frame
            v_full = dist_uv[:, 1] * sy_frame
            map_x_c = u_full.reshape(grid_h, grid_w).astype(np.float32)
            map_y_c = v_full.reshape(grid_h, grid_w).astype(np.float32)
            map_x = cv2.resize(map_x_c, (width, height), interpolation=cv2.INTER_LINEAR)
            map_y = cv2.resize(map_y_c, (width, height), interpolation=cv2.INTER_LINEAR)
            return cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        if getattr(self, 'warp_mode', 'planar') == 'panorama_tps' and getattr(self, '_tps_inv', None) is not None:
            # Evaluate global H_inv + residual TPS on a compact coarse mesh and upscale bilinearly
            grid_w = min(width, 210)
            grid_h = min(height, 136)
            gx = np.linspace(0, width - 1, grid_w, dtype=np.float64)
            gy = np.linspace(0, height - 1, grid_h, dtype=np.float64)
            xv, yv = np.meshgrid(gx, gy)
            pitch_x = (xv - float(padding)) / scale_x
            pitch_y = (yv - float(padding)) / scale_y
            pitch_pts = np.column_stack([pitch_x.ravel(), pitch_y.ravel()])
            base_uv = cv2.perspectiveTransform(
                pitch_pts.astype(np.float32).reshape(-1, 1, 2), self.H_inv
            ).reshape(-1, 2).astype(np.float64)
            norm_in = np.column_stack([
                (pitch_x / self.pitch_length).ravel(),
                (pitch_y / self.pitch_width).ravel()
            ])
            res_out = self._tps_inv(norm_in)
            fh, fw = frame.shape[:2]
            sx_frame = float(fw) / float(max(1, self.frame_width))
            sy_frame = float(fh) / float(max(1, self.frame_height))
            u_full = base_uv[:, 0] * sx_frame + res_out[:, 0] * float(fw)
            v_full = base_uv[:, 1] * sy_frame + res_out[:, 1] * float(fh)
            map_x_c = u_full.reshape(grid_h, grid_w).astype(np.float32)
            map_y_c = v_full.reshape(grid_h, grid_w).astype(np.float32)
            map_x = cv2.resize(map_x_c, (width, height), interpolation=cv2.INTER_LINEAR)
            map_y = cv2.resize(map_y_c, (width, height), interpolation=cv2.INTER_LINEAR)
            return cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        fh, fw = frame.shape[:2]
        sx_frame = float(max(1, self.frame_width)) / float(max(1, fw))
        sy_frame = float(max(1, self.frame_height)) / float(max(1, fh))
        F_scale = np.array([
            [sx_frame, 0.0, 0.0],
            [0.0, sy_frame, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        S = np.array([
            [scale_x, 0.0, float(padding)],
            [0.0, scale_y, float(padding)],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        H_warp = S @ self.H @ F_scale
        return cv2.warpPerspective(frame, H_warp, (width, height))

    def get_pitch_roi(
        self,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None,
        margin_ratio: float = 0.08
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Calculates the bounding region of interest (ROI) [x1, y1, x2, y2] of the playable pitch
        on the video frame based on the active pitch calibration mask.
        Allows object detectors to crop away empty sky/stands and foreground concrete,
        maximizing effective neural pixel resolution on players.
        """
        w = int(frame_width or self.frame_width or 1920)
        h = int(frame_height or self.frame_height or 1080)
        aspect = w / float(max(1, h))

        l = float(getattr(self, "pitch_length", PitchModel.LENGTH))
        pw = float(getattr(self, "pitch_width", PitchModel.WIDTH))

        samples: List[Tuple[float, float]] = []
        steps = 20
        for i in range(steps + 1):
            t = i / float(steps)
            samples.append((t * l, 0.0))       # Top sideline
            samples.append((t * l, pw))        # Bottom sideline
            samples.append((0.0, t * pw))       # Left goal line
            samples.append((l, t * pw))         # Right goal line
            samples.append((t * l, pw * 0.5))  # Pitch spine

        try:
            px_pts = self.pitch_to_image(samples, filter_behind_camera=True)
            valid_pts = [
                (float(p[0]), float(p[1])) for p in px_pts
                if -0.2 * w <= p[0] <= 1.2 * w and -0.2 * h <= p[1] <= 1.2 * h
            ]
            if len(valid_pts) >= 4:
                xs = [p[0] for p in valid_pts]
                ys = [p[1] for p in valid_pts]
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)

                span_x = max_x - min_x
                span_y = max_y - min_y
                pad_x = max(20.0, span_x * margin_ratio)
                # Extra top padding to comfortably encompass standing player body heights along the far touchline
                pad_top = max(35.0, span_y * (margin_ratio + 0.08), 0.08 * h)
                pad_bot = max(24.0, span_y * margin_ratio)

                x1 = max(0, int(min_x - pad_x))
                y1 = max(0, int(min_y - pad_top))
                x2 = min(w, int(max_x + pad_x))
                y2 = min(h, int(max_y + pad_bot))

                if (x2 - x1) >= 0.4 * w and (y2 - y1) >= 0.3 * h:
                    return (x1, y1, x2, y2)
        except Exception:
            pass

        if aspect > 2.2:
            return (0, int(h * 0.12), w, int(h * 0.88))

        return None

    def get_calibration_signature(self) -> str:
        """Deterministic hash of current dimensions, mode, and active calibration geometry."""
        is_custom = bool(getattr(self, "is_custom_calibrated", False))
        payload = {
            "w": int(self.frame_width),
            "h": int(self.frame_height),
            "camera_mode": "custom" if is_custom else str(self.camera_mode),
            "warp_mode": str(getattr(self, "warp_mode", "planar")),
            "lens_lambda": round(float(getattr(self, "lens_lambda", 0.0)), 4),
            "pitch_length": round(float(getattr(self, "pitch_length", PitchModel.LENGTH)), 2),
            "pitch_width": round(float(getattr(self, "pitch_width", PitchModel.WIDTH)), 2),
            "custom": is_custom,
            "src": np.round(self.src_points, 2).tolist() if getattr(self, "src_points", None) is not None else [],
            "dst": np.round(self.dst_points, 2).tolist() if getattr(self, "dst_points", None) is not None else [],
        }
        raw = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.md5(raw).hexdigest()[:12]

    def export_client_projection(self, nx: int = 72, ny: int = 40) -> Dict[str, Any]:
        """
        Serializes active projection parameters for browser Hardware Media Engine canvas rendering.
        Includes dense bilinear lookup grids when panorama_tps or universal mode is active.
        """
        h_inv_list = [[round(float(val), 9) for val in row] for row in self.H_inv.tolist()]
        h_list = [[round(float(val), 9) for val in row] for row in self.H.tolist()]
        out: Dict[str, Any] = {
            "warp_mode": str(getattr(self, "warp_mode", "planar")),
            "lens_lambda": round(float(getattr(self, "lens_lambda", 0.0)), 6),
            "pitch_length": round(float(getattr(self, "pitch_length", PitchModel.LENGTH)), 2),
            "pitch_width": round(float(getattr(self, "pitch_width", PitchModel.WIDTH)), 2),
            "cx": float(getattr(self, "cx", self.frame_width / 2.0)),
            "cy": float(getattr(self, "cy", self.frame_height / 2.0)),
            "r_norm": float(getattr(self, "r_norm", np.hypot(self.frame_width / 2.0, self.frame_height / 2.0))),
            "H_inv": h_inv_list,
            "H": h_list,
            "calibration_sig": self.get_calibration_signature(),
            "p2i_lut": None,
            "i2p_lut": None,
        }

        if out["warp_mode"] in ("panorama_tps", "universal"):
            # Pitch [0..L]m x [0..W]m -> Image (u, v) LUT
            px_vals = np.linspace(0.0, self.pitch_length, nx)
            py_vals = np.linspace(0.0, self.pitch_width, ny)
            xg, yg = np.meshgrid(px_vals, py_vals)
            pitch_pts = [(float(x), float(y)) for x, y in zip(xg.ravel(), yg.ravel())]
            img_pts = self.pitch_to_image(pitch_pts, filter_behind_camera=False)
            u_grid = np.array([p[0] if p else 0 for p in img_pts], dtype=np.float64).reshape(ny, nx)
            v_grid = np.array([p[1] if p else 0 for p in img_pts], dtype=np.float64).reshape(ny, nx)
            out["p2i_lut"] = {
                "nx": nx,
                "ny": ny,
                "u": [[round(float(u_grid[r, c]), 2) for c in range(nx)] for r in range(ny)],
                "v": [[round(float(v_grid[r, c]), 2) for c in range(nx)] for r in range(ny)],
            }

            # Image [0..w] x [0..h] -> Pitch (X_m, Y_m) LUT
            ux_vals = np.linspace(0.0, float(self.frame_width), nx)
            vy_vals = np.linspace(0.0, float(self.frame_height), ny)
            ug, vg = np.meshgrid(ux_vals, vy_vals)
            cam_pts = [(float(u), float(v)) for u, v in zip(ug.ravel(), vg.ravel())]
            metric_pts = self.image_to_pitch(cam_pts, filter_out_of_bounds=False)
            x_grid = np.array([p[0] if p else 52.5 for p in metric_pts], dtype=np.float64).reshape(ny, nx)
            y_grid = np.array([p[1] if p else 34.0 for p in metric_pts], dtype=np.float64).reshape(ny, nx)
            out["i2p_lut"] = {
                "nx": nx,
                "ny": ny,
                "x": [[round(float(x_grid[r, c]), 3) for c in range(nx)] for r in range(ny)],
                "y": [[round(float(y_grid[r, c]), 3) for c in range(nx)] for r in range(ny)],
            }

        return out

