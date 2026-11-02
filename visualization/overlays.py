"""
Augmented Broadcast Video Overlay Engine - Pro Sports AR Edition
Projects the computed algorithmic matrices directly onto raw broadcast video frames:
- Neon Passing Lanes (Dijkstra shortest-path priority queue corridor with chevron dynamics)
- Shifting Voronoi Territory Fields (Fortune's sweep-line clipped zones on pitch turf)
- Glowing Defensive Bounding Shapes (Convex Hull polygon shield with corner reticles)
- High-Tech Broadcast HUD Telemetry Bar (FPS, Area in m^2, Dominance %, DAA Latency, Camera Specs)
"""

from typing import List, Tuple, Dict, Any, Optional
import math
import numpy as np
import cv2
from perception.homography import HomographyTransformer


class BroadcastOverlayEngine:
    """
    Overlays geometric and graph-theoretic tactical analytics onto broadcast video frames.
    """
    
    def __init__(self, transformer: HomographyTransformer):
        self.transformer = transformer

    def draw_broadcast_hud(
        self,
        frame: np.ndarray,
        fps: float,
        hull_res: Optional[Dict[str, Any]],
        voronoi_res: Optional[Dict[str, Any]],
        dijkstra_res: Optional[Dict[str, Any]],
        daa_latency_ms: float,
        kinematics_res: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Draws an ultra-sleek UEFA Champions League / Sky Sports style AR telemetry header.
        """
        h, w = frame.shape[:2]
        banner_h = 58
        
        # Glassmorphic carbon header background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (8, 12, 16), -1)
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
        
        # Cyber neon accent line along bottom of header
        cv2.line(frame, (0, banner_h), (w, banner_h), (0, 240, 255), 1)
        cv2.line(frame, (0, banner_h - 1), (int(w * 0.4), banner_h - 1), (0, 255, 170), 2)
        
        # Responsive scaling for any video resolution (640x360 up to 4K)
        scale = max(0.65, min(1.0, w / 1280.0))
        f_main = max(0.30, 0.38 * scale)
        f_sub = max(0.28, 0.35 * scale)
        y1 = max(18, int(24 * scale))
        y2 = max(34, int(45 * scale))
        
        has_kinematics = kinematics_res is not None
        if has_kinematics:
            c0 = max(8, int(15 * scale))
            c1 = int(w * 0.22)
            c2 = int(w * 0.42)
            c3 = int(w * 0.62)
            c4 = int(w * 0.81)
        else:
            c0 = max(10, int(20 * scale))
            c1 = int(w * 0.28)
            c2 = int(w * 0.53)
            c3 = int(w * 0.77)
        
        # 1. System Status & Camera Watermark
        cv2.circle(frame, (c0 + 4, y1 - 4), max(3, int(5 * scale)), (0, 255, 120), -1, lineType=cv2.LINE_AA)
        cv2.putText(frame, "CAM-01 GANTRY", (c0 + 14, y1), cv2.FONT_HERSHEY_SIMPLEX, f_main, (0, 255, 170), 1, cv2.LINE_AA)
        cv2.putText(frame, f"{daa_latency_ms:.1f}ms | {fps:.0f} FPS", (c0, y2), cv2.FONT_HERSHEY_SIMPLEX, f_sub, (160, 185, 205), 1, cv2.LINE_AA)
        
        # 2. Defensive Cohesion Hull Area
        if hull_res:
            area = hull_res.get('area_sq_m', 0.0)
            compact = hull_res.get('compactness', 0.0)
            algo = hull_res.get('algorithm', 'Hull')
            cv2.putText(frame, f"DEFENSE [{algo.split()[0].upper()}]", (c1, y1), cv2.FONT_HERSHEY_SIMPLEX, f_main, (0, 220, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"{area:.1f} m² | C:{compact:.2f}", (c1, y2), cv2.FONT_HERSHEY_SIMPLEX, f_sub, (230, 235, 245), 1, cv2.LINE_AA)
            
        # 3. Voronoi Spatial Dominance
        if voronoi_res:
            pct_a = voronoi_res.get('team_a_pct', 50.0)
            pct_b = voronoi_res.get('team_b_pct', 50.0)
            cv2.putText(frame, "DOMINANCE [VORONOI]", (c2, y1), cv2.FONT_HERSHEY_SIMPLEX, f_main, (255, 185, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, f"{pct_a:.0f}% H vs {pct_b:.0f}% A", (c2, y2), cv2.FONT_HERSHEY_SIMPLEX, f_sub, (230, 235, 245), 1, cv2.LINE_AA)

        # 4. Dijkstra Optimal Passing Risk
        if dijkstra_res:
            risk = dijkstra_res.get('avg_risk', 0.0)
            gain = dijkstra_res.get('total_progression', 0.0)
            cv2.putText(frame, "OPTIMAL [DIJKSTRA]", (c3, y1), cv2.FONT_HERSHEY_SIMPLEX, f_main, (0, 255, 170), 1, cv2.LINE_AA)
            cv2.putText(frame, f"R:{risk:.1f} | +{gain:.1f}m", (c3, y2), cv2.FONT_HERSHEY_SIMPLEX, f_sub, (230, 235, 245), 1, cv2.LINE_AA)
            
        # 5. Kinematics & Peak Acceleration Telemetry
        if has_kinematics:
            peak_accel = kinematics_res.get('peak_accel_val', 0.0)
            peak_speed = kinematics_res.get('peak_speed_val', 0.0)
            cv2.putText(frame, "BURST [ACCEL]", (c4, y1), cv2.FONT_HERSHEY_SIMPLEX, f_main, (255, 160, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, f"+{peak_accel:.1f} m/s² | {peak_speed:.0f}km/h", (c4, y2), cv2.FONT_HERSHEY_SIMPLEX, f_sub, (230, 235, 245), 1, cv2.LINE_AA)
            
        # Camera corner view reticles for broadcast lens aesthetic
        bracket_col = (100, 140, 130)
        cv2.line(frame, (10, banner_h + 10), (25, banner_h + 10), bracket_col, 1)
        cv2.line(frame, (10, banner_h + 10), (10, banner_h + 25), bracket_col, 1)
        cv2.line(frame, (w - 10, banner_h + 10), (w - 25, banner_h + 10), bracket_col, 1)
        cv2.line(frame, (w - 10, banner_h + 10), (w - 10, banner_h + 25), bracket_col, 1)
        cv2.line(frame, (10, h - 10), (25, h - 10), bracket_col, 1)
        cv2.line(frame, (10, h - 10), (10, h - 25), bracket_col, 1)
        cv2.line(frame, (w - 10, h - 10), (w - 25, h - 10), bracket_col, 1)
        cv2.line(frame, (w - 10, h - 10), (w - 10, h - 25), bracket_col, 1)
        
        return frame

    def overlay_projected_hull(
        self,
        frame: np.ndarray,
        hull_metric_coords: List[Tuple[float, float]],
        color: Tuple[int, int, int] = (0, 220, 255),
        alpha: float = 0.32
    ) -> np.ndarray:
        """
        Projects defensive convex hull polygon back to broadcast camera perspective
        and draws glowing boundary with translucent fill.
        """
        if len(hull_metric_coords) < 3:
            return frame
            
        pixel_pts = self.transformer.pitch_to_image(hull_metric_coords)
        pts_arr = np.array(pixel_pts, dtype=np.int32)
        
        # Translucent shield polygon
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts_arr], color)
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
        
        # Double glowing border lines
        cv2.polylines(frame, [pts_arr], isClosed=True, color=(color[0] // 2, color[1] // 2, color[2] // 2), thickness=5, lineType=cv2.LINE_AA)
        cv2.polylines(frame, [pts_arr], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
        
        # Glowing vertex reticles
        for pt in pixel_pts:
            cv2.circle(frame, pt, 6, color, -1, lineType=cv2.LINE_AA)
            cv2.circle(frame, pt, 3, (255, 255, 255), -1, lineType=cv2.LINE_AA)
            
        return frame

    def overlay_projected_voronoi(
        self,
        frame: np.ndarray,
        voronoi_res: Dict[str, Any],
        alpha: float = 0.20
    ) -> np.ndarray:
        """
        Projects Voronoi territorial partitions onto the broadcast pitch turf,
        bounded strictly within the pitch turf perimeter.
        """
        if not voronoi_res:
            return frame
            
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        # Build visible turf mask to prevent overlay from bleeding into stadium seats
        turf_mask = np.zeros((h, w), dtype=np.uint8)
        if getattr(self.transformer, 'camera_mode', 'sideline') == 'goal_to_goal' and not getattr(self.transformer, 'is_custom_calibrated', False):
            pitch_boundary = [(51.0, 0.0), (105.0, 0.0), (105.0, 68.0), (51.0, 68.0)]
        else:
            pitch_boundary = [(0.0, 0.0), (105.0, 0.0), (105.0, 68.0), (0.0, 68.0)]
            
        proj_boundary = self.transformer.pitch_to_image(pitch_boundary, filter_behind_camera=True)
        if len(proj_boundary) >= 3 and not any(p[0] < -w or p[0] > 2 * w or p[1] < -h or p[1] > 2 * h for p in proj_boundary):
            turf_poly = np.array(proj_boundary, dtype=np.int32)
            cv2.fillPoly(turf_mask, [turf_poly], 255)
        else:
            turf_mask[:] = 255
            
        # Team A cells (Sky Blue tint)
        for cell in voronoi_res.get('cells_a', []):
            poly_pixels = self.transformer.pitch_to_image(cell['polygon'])
            if len(poly_pixels) >= 3:
                pts = np.array(poly_pixels, dtype=np.int32)
                cv2.fillPoly(overlay, [pts], (230, 140, 10))
                cv2.polylines(overlay, [pts], isClosed=True, color=(190, 115, 10), thickness=1, lineType=cv2.LINE_AA)
                
        # Team B cells (Crimson Red tint)
        for cell in voronoi_res.get('cells_b', []):
            poly_pixels = self.transformer.pitch_to_image(cell['polygon'])
            if len(poly_pixels) >= 3:
                pts = np.array(poly_pixels, dtype=np.int32)
                cv2.fillPoly(overlay, [pts], (20, 45, 210))
                cv2.polylines(overlay, [pts], isClosed=True, color=(20, 35, 160), thickness=1, lineType=cv2.LINE_AA)
                
        # In-place blend: avoid allocating 2 additional full-size frame buffers
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, dst=overlay)
        overlay[turf_mask == 0] = frame[turf_mask == 0]
        return overlay

    def draw_dijkstra_broadcast_overlay(
        self,
        frame: np.ndarray,
        corridor_coords: List[Tuple[float, float]]
    ) -> np.ndarray:
        """
        Projects the optimal Dijkstra passing sequence directly onto the broadcast
        camera feed with neon emerald corridor glow, directional chevron arrows, and step tags.
        """
        if len(corridor_coords) < 2 or self.transformer is None:
            return frame

        pxs = self.transformer.pitch_to_image(corridor_coords, filter_behind_camera=True)
        if len(pxs) < 2:
            return frame

        h, w = frame.shape[:2]
        turf_mask = self.get_turf_binary_mask(frame.shape[:2])

        overlay = frame.copy()
        for i in range(len(pxs) - 1):
            p1 = (int(pxs[i][0]), int(pxs[i][1]))
            p2 = (int(pxs[i + 1][0]), int(pxs[i + 1][1]))
            # Skip if wildly outside frame
            if (p1[0] < -w or p1[0] > 2 * w or p1[1] < -h or p1[1] > 2 * h or
                p2[0] < -w or p2[0] > 2 * w or p2[1] < -h or p2[1] > 2 * h):
                continue

            # Outer emerald glow
            cv2.line(overlay, p1, p2, (0, 160, 90), 6, lineType=cv2.LINE_AA)
            # Inner neon emerald core
            cv2.line(overlay, p1, p2, (0, 255, 170), 2, lineType=cv2.LINE_AA)

            # Midpoint directional pulse marker
            mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
            cv2.circle(overlay, mid, 6, (0, 255, 170), -1, lineType=cv2.LINE_AA)
            cv2.circle(overlay, mid, 2, (255, 255, 255), -1, lineType=cv2.LINE_AA)

            # Pass label
            cv2.putText(overlay, f"PASS {i+1}", (mid[0] + 8, mid[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 200), 1, cv2.LINE_AA)

        # Blend on turf only
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, dst=overlay)
        overlay[turf_mask == 0] = frame[turf_mask == 0]
        return overlay

    def get_turf_binary_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """Returns binary turf mask (255 inside pitch perimeter, 0 outside in stands)."""
        h, w = shape[:2]
        turf_mask = np.zeros((h, w), dtype=np.uint8)
        if getattr(self.transformer, 'camera_mode', 'sideline') == 'goal_to_goal' and not getattr(self.transformer, 'is_custom_calibrated', False):
            pitch_boundary = [(51.0, 0.0), (105.0, 0.0), (105.0, 68.0), (51.0, 68.0)]
        else:
            pitch_boundary = [(0.0, 0.0), (105.0, 0.0), (105.0, 68.0), (0.0, 68.0)]
            
        proj_boundary = self.transformer.pitch_to_image(pitch_boundary, filter_behind_camera=True)
        if len(proj_boundary) >= 3 and not any(p[0] < -w or p[0] > 2 * w or p[1] < -h or p[1] > 2 * h for p in proj_boundary):
            turf_poly = np.array(proj_boundary, dtype=np.int32)
            cv2.fillPoly(turf_mask, [turf_poly], 255)
        else:
            turf_mask[:] = 255
        return turf_mask

    def render_homography_debug_frame(
        self,
        raw_frame: np.ndarray,
        show_landmarks: bool = True,
        show_turf_mask: bool = True,
        show_markings: bool = True,
        show_player_pins: bool = True,
        player_detections: Optional[List[Any]] = None,
        positions: Optional[Any] = None
    ) -> np.ndarray:
        """
        Stage 2 Visual Debugger:
        Draws projected pitch turf mask, homography landmark calibration crosshairs,
        reprojected FIFA markings, and metric coordinate pins.
        """
        h, w = raw_frame.shape[:2]
        self.transformer.update_dimensions(w, h)
        dbg = raw_frame.copy()
        overlay = raw_frame.copy()

        # 1. Projected Pitch Turf Mask
        if show_turf_mask:
            if getattr(self.transformer, 'is_custom_calibrated', False):
                # When manual custom calibration is active, the verified calibrated quad
                # defines the valid turf projection zone.
                turf_pts = np.array(self.transformer.src_points, dtype=np.int32)
            elif getattr(self.transformer, 'camera_mode', 'sideline') == 'goal_to_goal':
                boundary = [(52.5, 0.0), (105.0, 0.0), (105.0, 68.0), (52.5, 68.0)]
                turf_pts = np.array(self.transformer.pitch_to_image(boundary, filter_behind_camera=True), dtype=np.int32)
            else:
                boundary = [(0.0, 0.0), (105.0, 0.0), (105.0, 68.0), (0.0, 68.0)]
                turf_pts = np.array(self.transformer.pitch_to_image(boundary, filter_behind_camera=True), dtype=np.int32)

            if len(turf_pts) >= 3 and not any(p[0] < -w or p[0] > 2 * w or p[1] < -h or p[1] > 2 * h for p in turf_pts):
                cv2.fillPoly(overlay, [turf_pts], (0, 160, 80))
                cv2.addWeighted(overlay, 0.22, dbg, 0.78, 0, dbg)
                cv2.polylines(dbg, [turf_pts], isClosed=True, color=(0, 255, 120), thickness=2, lineType=cv2.LINE_AA)

        # 2. Projected FIFA Pitch Markings
        if show_markings:
            markings = [
                # Touchlines & Goal lines
                [(0.0, 0.0), (105.0, 0.0)],
                [(0.0, 68.0), (105.0, 68.0)],
                [(0.0, 0.0), (0.0, 68.0)],
                [(105.0, 0.0), (105.0, 68.0)],
                # Halfway line
                [(52.5, 0.0), (52.5, 68.0)],
                # 18-yard box left & right
                [(0.0, 13.84), (16.5, 13.84), (16.5, 54.16), (0.0, 54.16)],
                [(88.5, 13.84), (105.0, 13.84), (105.0, 54.16), (88.5, 54.16)],
                # 6-yard goal area left & right
                [(0.0, 24.84), (5.5, 24.84), (5.5, 43.16), (0.0, 43.16)],
                [(99.5, 24.84), (105.0, 24.84), (105.0, 43.16), (99.5, 43.16)]
            ]
            for m in markings:
                pxs = self.transformer.pitch_to_image(m, filter_behind_camera=True)
                if len(pxs) != len(m):
                    continue
                pts = np.array(pxs, dtype=np.int32)
                # Discard lines where points are wildly out of frame
                if any(p[0] < -w or p[0] > 2 * w or p[1] < -h or p[1] > 2 * h for p in pts):
                    continue
                if len(pts) == 2:
                    cv2.line(dbg, tuple(pts[0]), tuple(pts[1]), (0, 240, 255), 2, lineType=cv2.LINE_AA)
                else:
                    cv2.polylines(dbg, [pts], isClosed=True, color=(0, 240, 255), thickness=2, lineType=cv2.LINE_AA)

            # Center circle (9.15m radius around 52.5, 34.0)
            circle_pts = [(52.5 + 9.15 * math.cos(math.radians(deg)), 34.0 + 9.15 * math.sin(math.radians(deg))) for deg in range(0, 360, 10)]
            c_pxs = self.transformer.pitch_to_image(circle_pts, filter_behind_camera=True)
            if len(c_pxs) >= 12:
                c_arr = np.array(c_pxs, dtype=np.int32)
                if not any(p[0] < -w or p[0] > 2 * w or p[1] < -h or p[1] > 2 * h for p in c_arr):
                    cv2.polylines(dbg, [c_arr], isClosed=True, color=(0, 240, 255), thickness=2, lineType=cv2.LINE_AA)

        # 3. Homography Calibration Landmarks (4 Quads)
        if show_landmarks:
            src_pts = self.transformer.src_points
            dst_pts = self.transformer.dst_points
            quad_pts = np.array(src_pts, dtype=np.int32)

            # Translucent calibration quad
            q_overlay = dbg.copy()
            cv2.fillPoly(q_overlay, [quad_pts], (255, 200, 0))
            cv2.addWeighted(q_overlay, 0.15, dbg, 0.85, 0, dbg)
            cv2.polylines(dbg, [quad_pts], isClosed=True, color=(0, 255, 255), thickness=2, lineType=cv2.LINE_AA)

            for i in range(len(src_pts)):
                sx, sy = int(src_pts[i][0]), int(src_pts[i][1])
                dx, dy = dst_pts[i][0], dst_pts[i][1]

                # Neon reticle crosshairs
                cv2.drawMarker(dbg, (sx, sy), (0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2, line_type=cv2.LINE_AA)
                cv2.circle(dbg, (sx, sy), 8, (0, 255, 255), 2, lineType=cv2.LINE_AA)
                cv2.circle(dbg, (sx, sy), 4, (0, 0, 0), -1, lineType=cv2.LINE_AA)

                # Text pill
                lbl = f"L{i+1}: ({sx},{sy}) -> ({dx:.1f}m,{dy:.1f}m)"
                (lw, lh), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                bx = max(5, min(w - lw - 10, sx + 12))
                by = max(lh + 5, min(h - 10, sy - 8))
                cv2.rectangle(dbg, (bx - 4, by - lh - 4), (bx + lw + 4, by + 4), (10, 15, 25), -1)
                cv2.rectangle(dbg, (bx - 4, by - lh - 4), (bx + lw + 4, by + 4), (0, 255, 255), 1)
                cv2.putText(dbg, lbl, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)

        # 4. Player Metric Coordinate Pins
        if show_player_pins and positions:
            for team_id, coords in [(0, getattr(positions, 'team_a', [])), (1, getattr(positions, 'team_b', []))]:
                col = (255, 180, 0) if team_id == 0 else (50, 100, 255)
                for pt in coords:
                    img_pts = self.transformer.pitch_to_image([pt], filter_behind_camera=True)
                    if not img_pts:
                        continue
                    ix, iy = img_pts[0]
                    if 0 <= ix < w and 0 <= iy < h:
                        is_in_bounds = (0.0 <= pt[0] <= 105.0 and 0.0 <= pt[1] <= 68.0)
                        pin_col = col if is_in_bounds else (0, 0, 255)

                        cv2.circle(dbg, (ix, iy), 5, pin_col, -1, lineType=cv2.LINE_AA)
                        cv2.circle(dbg, (ix, iy), 7, (255, 255, 255), 1, lineType=cv2.LINE_AA)
                        coord_txt = f"{pt[0]:.1f}m,{pt[1]:.1f}m"
                        cv2.putText(dbg, coord_txt, (ix + 6, iy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

        # Glassmorphic Header Tag
        cv2.rectangle(dbg, (0, 0), (w, 32), (10, 14, 22), -1)
        cv2.line(dbg, (0, 32), (w, 32), (0, 240, 255), 1)
        cv2.putText(dbg, f"STAGE 2: HOMOGRAPHY CALIBRATION & PROJECTED TURF MASK ({self.transformer.camera_mode.upper()})", (14, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 200), 1, cv2.LINE_AA)

        return dbg

    def render_geometry_debug_frame(
        self,
        raw_frame: np.ndarray,
        positions: Any,
        geometry: Any,
        show_hull: bool = True,
        show_voronoi: bool = True,
        show_centroids: bool = True
    ) -> np.ndarray:
        """
        Stage 3 Visual Debugger:
        Visualizes the Convex Hull shield boundary and Voronoi spatial dominance partitions
        projected directly on camera frame with vertex labels and team centroids.
        """
        h, w = raw_frame.shape[:2]
        self.transformer.update_dimensions(w, h)
        dbg = raw_frame.copy()

        # 1. Voronoi Partitions
        if show_voronoi and (getattr(geometry, 'voronoi_cells_a', []) or getattr(geometry, 'voronoi_cells_b', [])):
            v_dict = {
                'cells_a': getattr(geometry, 'voronoi_cells_a', []),
                'cells_b': getattr(geometry, 'voronoi_cells_b', []),
                'team_a_pct': getattr(geometry, 'team_a_pct', 50.0),
                'team_b_pct': getattr(geometry, 'team_b_pct', 50.0)
            }
            dbg = self.overlay_projected_voronoi(dbg, v_dict, alpha=0.25)

        # 2. Render Active Player Markers on Ground Contact Points
        if positions:
            for team_id, coords, col, lbl_prefix in [
                (0, getattr(positions, 'team_a', []), (255, 185, 0), "A"),
                (1, getattr(positions, 'team_b', []), (50, 60, 255), "B")
            ]:
                for idx, pt in enumerate(coords):
                    img_pt = self.transformer.pitch_to_image([pt])[0]
                    ix, iy = int(img_pt[0]), int(img_pt[1])
                    if 0 <= ix < w and 0 <= iy < h:
                        cv2.circle(dbg, (ix, iy), 6, (15, 20, 25), -1)
                        cv2.circle(dbg, (ix, iy), 5, col, -1, lineType=cv2.LINE_AA)
                        cv2.circle(dbg, (ix, iy), 5, (255, 255, 255), 1, lineType=cv2.LINE_AA)
                        cv2.putText(dbg, f"{lbl_prefix}{idx+1}", (ix + 7, iy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)

        # 3. Defensive Convex Hull & Vertex Attribution
        if show_hull and getattr(geometry, 'hull_polygon', None) and len(geometry.hull_polygon) >= 3:
            dbg = self.overlay_projected_hull(dbg, geometry.hull_polygon, color=(0, 200, 255), alpha=0.30)
            
            # Annotate vertices with explicit player attribution
            h_pixels = self.transformer.pitch_to_image(geometry.hull_polygon)
            hull_indices = getattr(geometry, 'hull_player_indices', [])
            for idx, pt in enumerate(h_pixels):
                cv2.circle(dbg, pt, 8, (0, 255, 255), -1, lineType=cv2.LINE_AA)
                cv2.circle(dbg, pt, 10, (0, 0, 0), 1, lineType=cv2.LINE_AA)
                p_tag = f"B#{hull_indices[idx]+1}" if idx < len(hull_indices) else ""
                v_lbl = f"V{idx+1} ({p_tag})" if p_tag else f"V{idx+1}"
                cv2.putText(dbg, v_lbl, (pt[0] + 10, pt[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)

        # 4. Team Centroids (Clearly tagged as mathematical Center of Mass, NOT players)
        if show_centroids and positions:
            for team_name, pts, col in [("Team A", getattr(positions, 'team_a', []), (255, 200, 0)), ("Team B", getattr(positions, 'team_b', []), (50, 80, 255))]:
                if len(pts) >= 2:
                    cx = float(np.mean([p[0] for p in pts]))
                    cy = float(np.mean([p[1] for p in pts]))
                    c_px = self.transformer.pitch_to_image([(cx, cy)])[0]
                    # Distinct diamond marker for centroid to avoid confusing with player circles
                    cv2.drawMarker(dbg, tuple(c_px), col, markerType=cv2.MARKER_DIAMOND, markerSize=18, thickness=2, line_type=cv2.LINE_AA)
                    cv2.circle(dbg, tuple(c_px), 5, (255, 255, 255), -1, lineType=cv2.LINE_AA)
                    cv2.putText(dbg, f"★ {team_name} Mean Centroid ({cx:.1f}m,{cy:.1f}m)", (c_px[0] + 14, c_px[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

        # Header tag
        cv2.rectangle(dbg, (0, 0), (w, 32), (10, 14, 22), -1)
        cv2.line(dbg, (0, 32), (w, 32), (0, 240, 255), 1)
        hull_area = getattr(geometry, 'hull_area', 0.0)
        compact = getattr(geometry, 'hull_compactness', 0.0)
        cv2.putText(dbg, f"STAGE 3: DAA GEOMETRY VISUALIZER | Hull Area: {hull_area:.1f}m² | Compactness: {compact:.2f}", (14, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 200), 1, cv2.LINE_AA)

        return dbg


