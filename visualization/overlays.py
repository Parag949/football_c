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

    def _is_panorama_tps(self) -> bool:
        return getattr(self.transformer, 'warp_mode', 'planar') in ('panorama_tps', 'universal')

    @staticmethod
    def _subdivide_metric_path(
        points: List[Tuple[float, float]],
        steps_per_edge: int = 12,
        closed: bool = False
    ) -> List[Tuple[float, float]]:
        """Inserts intermediate samples along metric line segments so non-linear warps curve smoothly."""
        if len(points) < 2 or steps_per_edge <= 1:
            return list(points)
        dense: List[Tuple[float, float]] = []
        n = len(points)
        edge_count = n if closed else n - 1
        for i in range(edge_count):
            p1 = points[i]
            p2 = points[(i + 1) % n]
            for s in range(steps_per_edge):
                t = s / float(steps_per_edge)
                dense.append((p1[0] + (p2[0] - p1[0]) * t, p1[1] + (p2[1] - p1[1]) * t))
        if not closed:
            dense.append(points[-1])
        return dense

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
        hull_metric_coords: Any,
        color: Tuple[int, int, int] = (0, 220, 255),
        alpha: float = 0.32
    ) -> np.ndarray:
        """
        Projects defensive convex hull polygon back to broadcast camera perspective
        and draws glowing boundary with translucent fill.
        """
        if isinstance(hull_metric_coords, dict):
            hull_metric_coords = hull_metric_coords.get("polygon", hull_metric_coords.get("hull", []))

        if not hull_metric_coords or len(hull_metric_coords) < 3:
            return frame

        is_tps = self._is_panorama_tps()
        poly_coords = self._subdivide_metric_path(hull_metric_coords, steps_per_edge=10, closed=True) if is_tps else hull_metric_coords
        pixel_pts = self.transformer.pitch_to_image(poly_coords, filter_behind_camera=True)
        valid_pts = [p for p in pixel_pts if abs(p[0]) < 200000 and abs(p[1]) < 200000]
        if len(valid_pts) < 3:
            return frame
        pts_arr = np.array(valid_pts, dtype=np.int32)

        # Translucent shield polygon
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts_arr], color)
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)

        # Double glowing border lines
        cv2.polylines(frame, [pts_arr], isClosed=True, color=(color[0] // 2, color[1] // 2, color[2] // 2), thickness=5, lineType=cv2.LINE_AA)
        cv2.polylines(frame, [pts_arr], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)

        # Glowing vertex reticles at true player vertices
        vertex_pixels = self.transformer.pitch_to_image(hull_metric_coords, filter_behind_camera=True)
        for pt in vertex_pixels:
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
        turf_mask = self.get_turf_binary_mask((h, w))
        is_tps = self._is_panorama_tps()

        # Team A cells (Sky Blue tint) and Team B cells (Crimson Red tint)
        for team_key, fill_col, edge_col in [
            ('cells_a', (230, 140, 10), (190, 115, 10)),
            ('cells_b', (20, 45, 210), (20, 35, 160)),
        ]:
            for cell in voronoi_res.get(team_key, []):
                poly_m = cell['polygon']
                if is_tps and len(poly_m) >= 3:
                    poly_m = self._subdivide_metric_path(poly_m, steps_per_edge=6, closed=True)
                poly_pixels = self.transformer.pitch_to_image(poly_m, filter_behind_camera=True)
                valid_poly = [p for p in poly_pixels if abs(p[0]) < 200000 and abs(p[1]) < 200000]
                if len(valid_poly) >= 3:
                    pts = np.array(valid_poly, dtype=np.int32)
                    cv2.fillPoly(overlay, [pts], fill_col)
                    cv2.polylines(overlay, [pts], isClosed=True, color=edge_col, thickness=1, lineType=cv2.LINE_AA)
                
        # In-place blend: avoid allocating 2 additional full-size frame buffers
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, dst=overlay)
        overlay[turf_mask == 0] = frame[turf_mask == 0]
        return overlay

    def draw_passing_graph_broadcast_overlay(
        self,
        frame: np.ndarray,
        graph_edges: List[Dict[str, Any]],
        alpha: float = 0.30
    ) -> np.ndarray:
        """
        Projects the candidate passing network graph G = (V, E) directly onto the broadcast camera feed.
        """
        if not graph_edges or self.transformer is None:
            return frame

        h, w = frame.shape[:2]
        turf_mask = self.get_turf_binary_mask((h, w))
        overlay = frame.copy()

        for edge in graph_edges:
            p1_m = edge.get('source_point')
            p2_m = edge.get('target_point')
            if not p1_m or not p2_m:
                continue

            pxs = self.transformer.pitch_to_image([p1_m, p2_m], filter_behind_camera=True)
            if len(pxs) < 2:
                continue

            pt1 = (int(pxs[0][0]), int(pxs[0][1]))
            pt2 = (int(pxs[1][0]), int(pxs[1][1]))

            if (pt1[0] < -w or pt1[0] > 2 * w or pt1[1] < -h or pt1[1] > 2 * h or
                pt2[0] < -w or pt2[0] > 2 * w or pt2[1] < -h or pt2[1] > 2 * h):
                continue

            pressure = edge.get('pressure', 0.0)
            edge_col = (180, 160, 50) if pressure < 0.8 else (100, 100, 150)
            cv2.line(overlay, pt1, pt2, edge_col, 1, lineType=cv2.LINE_AA)

        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, dst=overlay)
        overlay[turf_mask == 0] = frame[turf_mask == 0]
        return overlay

    def draw_dijkstra_broadcast_overlay(
        self,
        frame: np.ndarray,
        corridor_coords: List[Tuple[float, float]],
        graph_edges: Optional[List[Dict[str, Any]]] = None,
        show_graph: bool = False
    ) -> np.ndarray:
        """
        Projects the optimal Dijkstra passing sequence directly onto the broadcast
        camera feed with neon emerald corridor glow, directional chevron arrows, and step tags.
        If show_graph is True, also projects candidate graph edges in G = (V, E).
        """
        if show_graph and graph_edges:
            frame = self.draw_passing_graph_broadcast_overlay(frame, graph_edges)

        if len(corridor_coords) < 2 or self.transformer is None:
            return frame

        is_tps = self._is_panorama_tps()
        pxs = self.transformer.pitch_to_image(corridor_coords, filter_behind_camera=True)
        if len(pxs) < 2:
            return frame

        h, w = frame.shape[:2]
        turf_mask = self.get_turf_binary_mask(frame.shape[:2])

        overlay = frame.copy()
        for i in range(len(corridor_coords) - 1):
            seg_m = [corridor_coords[i], corridor_coords[i + 1]]
            if is_tps:
                seg_m = self._subdivide_metric_path(seg_m, steps_per_edge=12, closed=False)
            seg_px = self.transformer.pitch_to_image(seg_m, filter_behind_camera=True)
            if len(seg_px) < 2:
                continue
            pts_seg = np.array(seg_px, dtype=np.int32)
            if any(p[0] < -w or p[0] > 2 * w or p[1] < -h or p[1] > 2 * h for p in pts_seg):
                continue

            cv2.polylines(overlay, [pts_seg], isClosed=False, color=(0, 160, 90), thickness=6, lineType=cv2.LINE_AA)
            cv2.polylines(overlay, [pts_seg], isClosed=False, color=(0, 255, 170), thickness=2, lineType=cv2.LINE_AA)

            mid = tuple(pts_seg[len(pts_seg) // 2])
            cv2.circle(overlay, mid, 6, (0, 255, 170), -1, lineType=cv2.LINE_AA)
            cv2.circle(overlay, mid, 2, (255, 255, 255), -1, lineType=cv2.LINE_AA)
            cv2.putText(overlay, f"PASS {i+1}", (mid[0] + 8, mid[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 200), 1, cv2.LINE_AA)

        # Blend on turf only
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, dst=overlay)
        overlay[turf_mask == 0] = frame[turf_mask == 0]
        return overlay

    def draw_tactical_ruler_overlay(
        self,
        frame: np.ndarray,
        ruler_data: Dict[str, Any]
    ) -> np.ndarray:
        """
        FW-06: Projects tactical distance ruler measurement tape and calibrated offside line
        directly onto the broadcast camera video stream clipped to pitch turf.
        """
        if not ruler_data or self.transformer is None:
            return frame

        h, w = frame.shape[:2]
        turf_mask = self.get_turf_binary_mask((h, w))
        overlay = frame.copy()

        pt_a = ruler_data.get('point_a')
        pt_b = ruler_data.get('point_b')
        dist = ruler_data.get('distance', 0.0)
        dx = ruler_data.get('dx', 0.0)
        dy = ruler_data.get('dy', 0.0)
        lbl_a = ruler_data.get('label_a', 'A')
        lbl_b = ruler_data.get('label_b', 'B')
        offside_x = ruler_data.get('offside_x')
        offside_players = ruler_data.get('offside_players', [])

        # 1. Project Calibrated Offside Line (VAR Style)
        if offside_x is not None and 0.0 <= offside_x <= 105.0:
            y_samples = np.linspace(0.0, 68.0, num=15)
            line_pts = [(offside_x, y_val) for y_val in y_samples]
            proj_line = self.transformer.pitch_to_image(line_pts, filter_behind_camera=True)

            if len(proj_line) >= 2:
                for i in range(len(proj_line) - 1):
                    p1 = (int(proj_line[i][0]), int(proj_line[i][1]))
                    p2 = (int(proj_line[i + 1][0]), int(proj_line[i + 1][1]))
                    cv2.line(overlay, p1, p2, (0, 180, 255), 5, lineType=cv2.LINE_AA)
                    cv2.line(overlay, p1, p2, (0, 235, 255), 2, lineType=cv2.LINE_AA)

                p_head = (int(proj_line[0][0]), int(proj_line[0][1]))
                if 0 <= p_head[0] < w and 0 <= p_head[1] < h:
                    cv2.putText(overlay, f"OFFSIDE LINE ({offside_x:.1f}m)", (p_head[0] + 8, p_head[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1, cv2.LINE_AA)

            for op in offside_players:
                op_pt = op.get('point')
                if op_pt:
                    op_proj = self.transformer.pitch_to_image([op_pt], filter_behind_camera=True)
                    if op_proj:
                        ox, oy = int(op_proj[0][0]), int(op_proj[0][1])
                        if 0 <= ox < w and 0 <= oy < h:
                            cv2.circle(overlay, (ox, oy), 14, (0, 0, 255), 2, lineType=cv2.LINE_AA)
                            cv2.putText(overlay, f"OFFSIDE! (+{op['margin_meters']:.1f}m)", (ox + 10, oy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (50, 50, 255), 1, cv2.LINE_AA)

        # 2. Project Distance Measurement Line
        if pt_a and pt_b:
            proj_ab = self.transformer.pitch_to_image([pt_a, pt_b], filter_behind_camera=True)
            if len(proj_ab) == 2:
                p1 = (int(proj_ab[0][0]), int(proj_ab[0][1]))
                p2 = (int(proj_ab[1][0]), int(proj_ab[1][1]))

                cv2.line(overlay, p1, p2, (0, 140, 255), 4, lineType=cv2.LINE_AA)
                cv2.line(overlay, p1, p2, (0, 240, 255), 2, lineType=cv2.LINE_AA)

                for p, lbl in [(p1, lbl_a), (p2, lbl_b)]:
                    cv2.circle(overlay, p, 7, (0, 0, 0), -1)
                    cv2.circle(overlay, p, 5, (0, 220, 255), -1, lineType=cv2.LINE_AA)
                    cv2.circle(overlay, p, 8, (255, 255, 255), 1, lineType=cv2.LINE_AA)
                    cv2.putText(overlay, lbl, (p[0] + 8, p[1] + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

                mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
                bb_txt = f"{dist:.1f}m [dx:{dx:.1f}m, dy:{dy:.1f}m]"
                (tw, th), _ = cv2.getTextSize(bb_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
                cv2.rectangle(overlay, (mid[0] - tw // 2 - 6, mid[1] - th - 6), (mid[0] + tw // 2 + 6, mid[1] + 4), (15, 23, 42), -1)
                cv2.rectangle(overlay, (mid[0] - tw // 2 - 6, mid[1] - th - 6), (mid[0] + tw // 2 + 6, mid[1] + 4), (0, 215, 255), 1)
                cv2.putText(overlay, bb_txt, (mid[0] - tw // 2, mid[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1, cv2.LINE_AA)

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

        if self._is_panorama_tps():
            pitch_boundary = self._subdivide_metric_path(pitch_boundary, steps_per_edge=18, closed=True)

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

        is_tps = self._is_panorama_tps()

        # 1. Projected Pitch Turf Mask
        if show_turf_mask:
            if getattr(self.transformer, 'camera_mode', 'sideline') == 'goal_to_goal' and not getattr(self.transformer, 'is_custom_calibrated', False):
                boundary = [(52.5, 0.0), (105.0, 0.0), (105.0, 68.0), (52.5, 68.0)]
            else:
                boundary = [(0.0, 0.0), (105.0, 0.0), (105.0, 68.0), (0.0, 68.0)]
            if is_tps:
                boundary = self._subdivide_metric_path(boundary, steps_per_edge=18, closed=True)
            turf_pts = np.array(self.transformer.pitch_to_image(boundary, filter_behind_camera=True), dtype=np.int32)

            if len(turf_pts) < 3 or any(p[0] < -w or p[0] > 2 * w or p[1] < -h or p[1] > 2 * h for p in turf_pts):
                if getattr(self.transformer, 'is_custom_calibrated', False) and len(self.transformer.src_points) >= 3:
                    hull_src = cv2.convexHull(np.array(self.transformer.src_points, dtype=np.int32))
                    turf_pts = hull_src.reshape(-1, 2)

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
                # 18-yard box left & right (closed)
                [(0.0, 13.84), (16.5, 13.84), (16.5, 54.16), (0.0, 54.16), (0.0, 13.84)],
                [(88.5, 13.84), (105.0, 13.84), (105.0, 54.16), (88.5, 54.16), (88.5, 13.84)],
                # 6-yard goal area left & right (closed)
                [(0.0, 24.84), (5.5, 24.84), (5.5, 43.16), (0.0, 43.16), (0.0, 24.84)],
                [(99.5, 24.84), (105.0, 24.84), (105.0, 43.16), (99.5, 43.16), (99.5, 24.84)]
            ]
            for m in markings:
                m_pts = self._subdivide_metric_path(m, steps_per_edge=16, closed=False) if is_tps else m
                pxs = self.transformer.pitch_to_image(m_pts, filter_behind_camera=True)
                if len(pxs) < 2:
                    continue
                pts = np.array(pxs, dtype=np.int32)
                if any(p[0] < -w or p[0] > 2 * w or p[1] < -h or p[1] > 2 * h for p in pts):
                    continue
                cv2.polylines(dbg, [pts], isClosed=False, color=(0, 240, 255), thickness=2, lineType=cv2.LINE_AA)

            # Center circle (9.15m radius around 52.5, 34.0)
            circle_pts = [(52.5 + 9.15 * math.cos(math.radians(deg)), 34.0 + 9.15 * math.sin(math.radians(deg))) for deg in range(0, 360, 10)]
            c_pxs = self.transformer.pitch_to_image(circle_pts, filter_behind_camera=True)
            if len(c_pxs) >= 12:
                c_arr = np.array(c_pxs, dtype=np.int32)
                if not any(p[0] < -w or p[0] > 2 * w or p[1] < -h or p[1] > 2 * h for p in c_arr):
                    cv2.polylines(dbg, [c_arr], isClosed=True, color=(0, 240, 255), thickness=2, lineType=cv2.LINE_AA)

        # 3. Homography Calibration Landmarks
        if show_landmarks:
            src_pts = self.transformer.src_points
            dst_pts = self.transformer.dst_points
            if len(src_pts) == 4:
                quad_pts = np.array(src_pts, dtype=np.int32)
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
        show_dijkstra: bool = True,
        show_passing_graph: bool = False,
        show_centroids: bool = True
    ) -> np.ndarray:
        """
        Stage 3 Visual Debugger:
        Visualizes the Convex Hull shield boundary, Voronoi spatial dominance partitions,
        and Dijkstra optimal passing corridor projected directly on camera frame with vertex labels and team centroids.
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
                    img_pts = self.transformer.pitch_to_image([pt], filter_behind_camera=True)
                    if not img_pts:
                        continue
                    img_pt = img_pts[0]
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
            h_pixels = self.transformer.pitch_to_image(geometry.hull_polygon, filter_behind_camera=True)
            hull_indices = getattr(geometry, 'hull_player_indices', [])
            for idx, pt in enumerate(h_pixels):
                cv2.circle(dbg, pt, 8, (0, 255, 255), -1, lineType=cv2.LINE_AA)
                cv2.circle(dbg, pt, 10, (0, 0, 0), 1, lineType=cv2.LINE_AA)
                p_tag = f"B#{hull_indices[idx]+1}" if idx < len(hull_indices) else ""
                v_lbl = f"V{idx+1} ({p_tag})" if p_tag else f"V{idx+1}"
                cv2.putText(dbg, v_lbl, (pt[0] + 10, pt[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)

        # 4. Dijkstra Optimal Passing Corridor & Passing Network Graph
        if show_dijkstra and getattr(geometry, 'dijkstra_res', None) and geometry.dijkstra_res.get('corridor_points'):
            g_edges = geometry.dijkstra_res.get('graph_edges') if show_passing_graph else None
            dbg = self.draw_dijkstra_broadcast_overlay(
                dbg,
                geometry.dijkstra_res['corridor_points'],
                graph_edges=g_edges,
                show_graph=show_passing_graph
            )

        # 5. Team Centroids (Clearly tagged as mathematical Center of Mass, NOT players)
        if show_centroids and positions:
            for team_name, pts, col in [("Team A", getattr(positions, 'team_a', []), (255, 200, 0)), ("Team B", getattr(positions, 'team_b', []), (50, 80, 255))]:
                if len(pts) >= 2:
                    cx = float(np.mean([p[0] for p in pts]))
                    cy = float(np.mean([p[1] for p in pts]))
                    c_px_list = self.transformer.pitch_to_image([(cx, cy)], filter_behind_camera=True)
                    if not c_px_list:
                        continue
                    c_px = c_px_list[0]
                    # Distinct diamond marker for centroid to avoid confusing with player circles
                    cv2.drawMarker(dbg, tuple(c_px), col, markerType=cv2.MARKER_DIAMOND, markerSize=18, thickness=2, line_type=cv2.LINE_AA)
                    cv2.circle(dbg, tuple(c_px), 5, (255, 255, 255), -1, lineType=cv2.LINE_AA)
                    cv2.putText(dbg, f"* {team_name} Mean Centroid ({cx:.1f}m,{cy:.1f}m)", (c_px[0] + 14, c_px[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

        # Header tag
        cv2.rectangle(dbg, (0, 0), (w, 32), (10, 14, 22), -1)
        cv2.line(dbg, (0, 32), (w, 32), (0, 240, 255), 1)
        hull_area = getattr(geometry, 'hull_area', 0.0)
        compact = getattr(geometry, 'hull_compactness', 0.0)
        cv2.putText(dbg, f"STAGE 3: DAA GEOMETRY VISUALIZER | Hull Area: {hull_area:.1f}m2 | Compactness: {compact:.2f}", (14, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 200), 1, cv2.LINE_AA)

        return dbg


