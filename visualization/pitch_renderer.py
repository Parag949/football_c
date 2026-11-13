"""
Tactical 2D Pitch Minimap (Radar) Renderer - High-Aesthetic Pro Sports Edition
Renders an ultra-clean, high-aesthetic UEFA Pro tactical radar board
with pitch markings, glowing player nodes, defensive hulls, Voronoi partitions,
and Dijkstra passing corridors.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import cv2
import math
from perception.pitch_model import PitchModel


class TacticalPitchRenderer:
    """
    Renders flat 2D tactical minimap with dark cyber-tactical aesthetics.
    """
    
    def __init__(self, width: int = 1120, height: int = 730, padding: int = 40):
        self.width = width
        self.height = height
        self.padding = padding
        
        # Color Palette (BGR)
        self.BG_COLOR = (10, 14, 18)          # Ultra-deep obsidian
        self.GRASS_COLOR = (14, 25, 20)       # Dark stadium emerald turf
        self.STRIPE_COLOR = (18, 30, 24)      # Turf mower contrast stripe
        self.LINE_COLOR = (45, 80, 60)        # Pitch markings subtle
        self.LINE_ACCENT = (0, 240, 255)      # Active line accent
        
        # Team Palettes (BGR)
        self.TEAM_A_COLOR = (255, 185, 0)     # Neon Sky Blue (Manchester Home)
        self.TEAM_A_GLOW = (180, 110, 0)
        self.TEAM_B_COLOR = (50, 60, 255)     # Vivid Crimson Red (Atletico Away)
        self.TEAM_B_GLOW = (20, 25, 140)
        
        self.HULL_BORDER = (0, 220, 255)      # Glowing Neon Amber / Gold
        self.HULL_FILL = (0, 160, 220)
        self.DIJKSTRA_COLOR = (0, 255, 170)   # Neon Emerald Core
        self.BALL_COLOR = (255, 255, 255)

    def to_pixel(self, metric_x: float, metric_y: float) -> Tuple[int, int]:
        mx = float(metric_x) if metric_x is not None and math.isfinite(float(metric_x)) else 52.5
        my = float(metric_y) if metric_y is not None and math.isfinite(float(metric_y)) else 34.0
        return PitchModel.metric_to_pixel(mx, my, self.width, self.height, self.padding)

    def draw_base_pitch(self) -> np.ndarray:
        """Draws the standard FIFA pitch lines and markings with mower grass stripes."""
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        canvas[:] = self.BG_COLOR
        
        tl = self.to_pixel(0.0, 0.0)
        br = self.to_pixel(PitchModel.LENGTH, PitchModel.WIDTH)
        
        # Turf rectangle with lawn mower stripes
        cv2.rectangle(canvas, tl, br, self.GRASS_COLOR, -1)
        
        # Alternating lawn stripes (6 vertical tactical zones)
        num_stripes = 10
        stripe_w = (br[0] - tl[0]) / num_stripes
        for s in range(0, num_stripes, 2):
            sx1 = int(tl[0] + s * stripe_w)
            sx2 = int(tl[0] + (s + 1) * stripe_w)
            cv2.rectangle(canvas, (sx1, tl[1]), (sx2, br[1]), self.STRIPE_COLOR, -1)
            
        # Subtle pitch boundary glow
        cv2.rectangle(canvas, (tl[0] - 1, tl[1] - 1), (br[0] + 1, br[1] + 1), (25, 45, 35), 2)
        cv2.rectangle(canvas, tl, br, self.LINE_COLOR, 2)
        
        # Halfway line
        half_top = self.to_pixel(52.5, 0.0)
        half_bot = self.to_pixel(52.5, PitchModel.WIDTH)
        cv2.line(canvas, half_top, half_bot, self.LINE_COLOR, 2)
        
        # Center spot & Center circle
        center = self.to_pixel(52.5, 34.0)
        cv2.circle(canvas, center, 4, (0, 240, 255), -1)
        
        usable_w = max(1.0, float(self.width - 2 * self.padding))
        scale_x = usable_w / PitchModel.LENGTH
        radius_px = max(1, int(round(PitchModel.CENTER_CIRCLE_RADIUS * scale_x)))
        cv2.circle(canvas, center, radius_px, self.LINE_COLOR, 2)
        
        # Penalty Boxes
        box_l_tl = self.to_pixel(0.0, 13.84)
        box_l_br = self.to_pixel(16.5, 54.16)
        cv2.rectangle(canvas, box_l_tl, box_l_br, self.LINE_COLOR, 2)
        
        box_r_tl = self.to_pixel(88.5, 13.84)
        box_r_br = self.to_pixel(105.0, 54.16)
        cv2.rectangle(canvas, box_r_tl, box_r_br, self.LINE_COLOR, 2)
        
        # Six-yard boxes
        six_l_tl = self.to_pixel(0.0, 24.84)
        six_l_br = self.to_pixel(5.5, 43.16)
        cv2.rectangle(canvas, six_l_tl, six_l_br, self.LINE_COLOR, 1)
        
        six_r_tl = self.to_pixel(99.5, 24.84)
        six_r_br = self.to_pixel(105.0, 43.16)
        cv2.rectangle(canvas, six_r_tl, six_r_br, self.LINE_COLOR, 1)
        
        # Penalty Spots
        cv2.circle(canvas, self.to_pixel(11.0, 34.0), 3, (0, 240, 255), -1)
        cv2.circle(canvas, self.to_pixel(94.0, 34.0), 3, (0, 240, 255), -1)
        
        # Goal mouths
        cv2.line(canvas, self.to_pixel(0.0, 30.34), self.to_pixel(0.0, 37.66), (255, 255, 255), 4)
        cv2.line(canvas, self.to_pixel(105.0, 30.34), self.to_pixel(105.0, 37.66), (255, 255, 255), 4)
        
        # Tactical Radar Crosshair Grid Accents
        cv2.drawMarker(canvas, (tl[0], tl[1]), (60, 90, 80), cv2.MARKER_CROSS, 12, 1)
        cv2.drawMarker(canvas, (br[0], tl[1]), (60, 90, 80), cv2.MARKER_CROSS, 12, 1)
        cv2.drawMarker(canvas, (tl[0], br[1]), (60, 90, 80), cv2.MARKER_CROSS, 12, 1)
        cv2.drawMarker(canvas, (br[0], br[1]), (60, 90, 80), cv2.MARKER_CROSS, 12, 1)
        
        return canvas

    def overlay_voronoi(
        self,
        canvas: np.ndarray,
        voronoi_res: Dict[str, Any],
        alpha: float = 0.26
    ) -> np.ndarray:
        """Renders semi-transparent Voronoi dominance territories with illuminated cell seams."""
        if not voronoi_res or 'cells_a' not in voronoi_res:
            return canvas
            
        overlay = canvas.copy()
        
        # Team A cells (Sky Blue tint)
        for cell in voronoi_res.get('cells_a', []):
            poly = cell.get('polygon', [])
            if not poly or len(poly) < 3:
                continue
            pts = np.array([self.to_pixel(x, y) for (x, y) in poly], dtype=np.int32)
            cv2.fillPoly(overlay, [pts], (220, 130, 10))
            cv2.polylines(canvas, [pts], isClosed=True, color=(160, 95, 10), thickness=1, lineType=cv2.LINE_AA)
            
        # Team B cells (Crimson Red tint)
        for cell in voronoi_res.get('cells_b', []):
            poly = cell.get('polygon', [])
            if not poly or len(poly) < 3:
                continue
            pts = np.array([self.to_pixel(x, y) for (x, y) in poly], dtype=np.int32)
            cv2.fillPoly(overlay, [pts], (20, 30, 190))
            cv2.polylines(canvas, [pts], isClosed=True, color=(20, 25, 130), thickness=1, lineType=cv2.LINE_AA)
            
        cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0, canvas)
        return canvas

    def overlay_defensive_hull(
        self,
        canvas: np.ndarray,
        hull_coords: Any,
        team_name: str = "Defense",
        alpha: float = 0.38
    ) -> np.ndarray:
        """Renders the defensive convex hull with dual glowing neon contour."""
        if isinstance(hull_coords, dict):
            hull_coords = hull_coords.get("polygon", hull_coords.get("hull", []))

        if not hull_coords or len(hull_coords) < 3:
            return canvas
            
        pts = np.array([self.to_pixel(x, y) for (x, y) in hull_coords], dtype=np.int32)
        
        # Translucent shield fill
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [pts], (0, 165, 230))
        cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0, canvas)
        
        # Outer ambient glow
        cv2.polylines(canvas, [pts], isClosed=True, color=(0, 100, 150), thickness=5, lineType=cv2.LINE_AA)
        # Inner neon blade
        cv2.polylines(canvas, [pts], isClosed=True, color=(0, 235, 255), thickness=2, lineType=cv2.LINE_AA)
        
        for pt in pts:
            cv2.circle(canvas, tuple(pt), 5, (0, 255, 255), -1, lineType=cv2.LINE_AA)
            cv2.circle(canvas, tuple(pt), 2, (255, 255, 255), -1, lineType=cv2.LINE_AA)
            
        return canvas

    def render_players(
        self,
        canvas: np.ndarray,
        team_a_coords: List[Tuple[float, float]],
        team_b_coords: List[Tuple[float, float]],
        ball_coord: Optional[Tuple[float, float]] = None,
        ball_carrier_idx: Optional[int] = None,
        show_acceleration: bool = False
    ) -> np.ndarray:
        """Renders player nodes with illuminated halos and tactical roles."""
        # Team A (Home, Sky Blue)
        for i, pt in enumerate(team_a_coords or []):
            if pt is None or len(pt) < 2 or not (math.isfinite(pt[0]) and math.isfinite(pt[1])):
                continue
            x, y = pt[0], pt[1]
            px, py = self.to_pixel(x, y)
            is_carrier = (i == ball_carrier_idx)
            
            # Carrier illuminated double halo
            if is_carrier:
                cv2.circle(canvas, (px, py), 15, (0, 240, 255), 2, lineType=cv2.LINE_AA)
                cv2.circle(canvas, (px, py), 20, (0, 180, 255), 1, lineType=cv2.LINE_AA)
                
            # Base node with halo
            cv2.circle(canvas, (px, py), 10, (15, 20, 25), -1)
            cv2.circle(canvas, (px, py), 8, self.TEAM_A_COLOR, -1, lineType=cv2.LINE_AA)
            cv2.circle(canvas, (px, py), 8, (255, 255, 255), 1, lineType=cv2.LINE_AA)
            cv2.putText(canvas, str(i + 1), (px - 4, py + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (10, 15, 20), 1, cv2.LINE_AA)

        # Team B (Away, Crimson Red)
        for j, pt in enumerate(team_b_coords or []):
            if pt is None or len(pt) < 2 or not (math.isfinite(pt[0]) and math.isfinite(pt[1])):
                continue
            x, y = pt[0], pt[1]
            px, py = self.to_pixel(x, y)
            cv2.circle(canvas, (px, py), 10, (25, 15, 18), -1)
            cv2.circle(canvas, (px, py), 8, self.TEAM_B_COLOR, -1, lineType=cv2.LINE_AA)
            cv2.circle(canvas, (px, py), 8, (255, 200, 200), 1, lineType=cv2.LINE_AA)
            cv2.putText(canvas, str(j + 1), (px - 4, py + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

        # Ball with stadium floodlight shadow
        if ball_coord and len(ball_coord) >= 2 and math.isfinite(ball_coord[0]) and math.isfinite(ball_coord[1]):
            bx, by = self.to_pixel(ball_coord[0], ball_coord[1])
            cv2.circle(canvas, (bx + 2, by + 2), 7, (5, 8, 5), -1)
            cv2.circle(canvas, (bx, by), 6, (0, 0, 0), -1)
            cv2.circle(canvas, (bx, by), 5, self.BALL_COLOR, -1, lineType=cv2.LINE_AA)
            
        return canvas

    def render_passing_network_graph(
        self,
        canvas: np.ndarray,
        graph_edges: List[Dict[str, Any]],
        alpha: float = 0.40
    ) -> np.ndarray:
        """
        Renders the complete directed passing network graph G = (V, E) on the 2D pitch radar.
        Candidate edges are rendered as subtle tactical cyan/slate lines with directional indicator pips.
        """
        if not graph_edges:
            return canvas

        overlay = canvas.copy()
        for edge in graph_edges:
            p1_m = edge.get('source_point')
            p2_m = edge.get('target_point')
            if not p1_m or not p2_m:
                continue

            pt1 = self.to_pixel(p1_m[0], p1_m[1])
            pt2 = self.to_pixel(p2_m[0], p2_m[1])

            pressure = edge.get('pressure', 0.0)
            if pressure < 0.5:
                edge_col = (200, 180, 50)   # Teal/cyan (BGR)
            elif pressure < 1.5:
                edge_col = (140, 140, 100)  # Muted slate
            else:
                edge_col = (80, 80, 180)    # Subdued amber (high pressure)

            cv2.line(overlay, pt1, pt2, edge_col, 1, lineType=cv2.LINE_AA)

            # Subtle directional pip at 2/3 along lane
            pip_x = int(pt1[0] * 0.35 + pt2[0] * 0.65)
            pip_y = int(pt1[1] * 0.35 + pt2[1] * 0.65)
            cv2.circle(overlay, (pip_x, pip_y), 2, edge_col, -1, lineType=cv2.LINE_AA)

        cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0, canvas)
        return canvas

    def render_dijkstra_pass_corridor(
        self,
        canvas: np.ndarray,
        corridor_coords: List[Tuple[float, float]],
        alpha: float = 0.85,
        graph_edges: Optional[List[Dict[str, Any]]] = None,
        show_graph: bool = False
    ) -> np.ndarray:
        """
        Renders the optimal Dijkstra passing sequence on the 2D pitch radar
        with glowing neon emerald lines, trajectory arrows, and step indices.
        If show_graph is True, also renders the full candidate passing graph G = (V, E).
        """
        if show_graph and graph_edges:
            canvas = self.render_passing_network_graph(canvas, graph_edges)

        if len(corridor_coords) < 2:
            return canvas

        pts = [self.to_pixel(x, y) for (x, y) in corridor_coords]

        # Draw connecting corridor segments
        for i in range(len(pts) - 1):
            p1, p2 = pts[i], pts[i + 1]
            # Outer neon glow
            cv2.line(canvas, p1, p2, (0, 160, 90), 6, lineType=cv2.LINE_AA)
            # Inner bright emerald core
            cv2.line(canvas, p1, p2, self.DIJKSTRA_COLOR, 2, lineType=cv2.LINE_AA)

            # Midpoint directional pulse marker
            mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
            cv2.circle(canvas, mid, 5, self.DIJKSTRA_COLOR, -1, lineType=cv2.LINE_AA)
            cv2.circle(canvas, mid, 2, (255, 255, 255), -1, lineType=cv2.LINE_AA)

            # Pass step number label
            cv2.putText(canvas, f"P{i+1}", (mid[0] + 6, mid[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 200), 1, cv2.LINE_AA)

        return canvas

    def render_tactical_ruler(
        self,
        canvas: np.ndarray,
        ruler_data: Dict[str, Any]
    ) -> np.ndarray:
        """
        FW-06: Renders interactive tactical distance ruler & calibrated offside laser line
        on the 2D FIFA tactical pitch radar.
        """
        if not ruler_data:
            return canvas

        pt_a = ruler_data.get('point_a')
        pt_b = ruler_data.get('point_b')
        dist = ruler_data.get('distance', 0.0)
        dx = ruler_data.get('dx', 0.0)
        dy = ruler_data.get('dy', 0.0)
        lbl_a = ruler_data.get('label_a', 'Point A')
        lbl_b = ruler_data.get('label_b', 'Point B')
        offside_x = ruler_data.get('offside_x')
        offside_players = ruler_data.get('offside_players', [])

        # 1. Render Calibrated Offside Laser Line (spanning pitch width Y=0..68m)
        if offside_x is not None and 0.0 <= offside_x <= 105.0:
            top_px = self.to_pixel(offside_x, 0.0)
            bot_px = self.to_pixel(offside_x, 68.0)

            # Draw dashed golden-yellow laser line
            dash_len = 10
            y_start = min(top_px[1], bot_px[1])
            y_end = max(top_px[1], bot_px[1])
            x_line = top_px[0]

            for y_pos in range(y_start, y_end, dash_len * 2):
                y_seg_end = min(y_end, y_pos + dash_len)
                cv2.line(canvas, (x_line, y_pos), (x_line, y_seg_end), (0, 215, 255), 2, lineType=cv2.LINE_AA)

            # Offside Line Header Badge
            badge_txt = f"OFFSIDE LINE: {offside_x:.1f}m"
            cv2.putText(canvas, badge_txt, (x_line - 55, top_px[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1, cv2.LINE_AA)

            # Highlight offside players with red warning rings
            for op in offside_players:
                op_pt = op.get('point')
                if op_pt:
                    op_px = self.to_pixel(op_pt[0], op_pt[1])
                    cv2.circle(canvas, op_px, 12, (0, 0, 255), 2, lineType=cv2.LINE_AA)
                    cv2.putText(canvas, f"OFFSIDE! (+{op['margin_meters']:.1f}m)", (op_px[0] + 10, op_px[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (50, 50, 255), 1, cv2.LINE_AA)

        # 2. Render Distance Measurement Tape between Point A and Point B
        if pt_a and pt_b:
            px_a = self.to_pixel(pt_a[0], pt_a[1])
            px_b = self.to_pixel(pt_b[0], pt_b[1])

            # Outer amber glow
            cv2.line(canvas, px_a, px_b, (0, 140, 255), 4, lineType=cv2.LINE_AA)
            # Inner bright yellow measurement core
            cv2.line(canvas, px_a, px_b, (0, 240, 255), 2, lineType=cv2.LINE_AA)

            # End caps & markers
            for px, lbl, tag_col in [(px_a, lbl_a, (0, 255, 255)), (px_b, lbl_b, (0, 200, 255))]:
                cv2.circle(canvas, px, 6, (0, 0, 0), -1)
                cv2.circle(canvas, px, 5, tag_col, -1, lineType=cv2.LINE_AA)
                cv2.circle(canvas, px, 7, (255, 255, 255), 1, lineType=cv2.LINE_AA)
                cv2.putText(canvas, lbl, (px[0] + 8, px[1] + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

            # Midpoint Distance Badge
            mid_x = (px_a[0] + px_b[0]) // 2
            mid_y = (px_a[1] + px_b[1]) // 2
            ruler_txt = f"{dist:.1f}m [dx:{dx:.1f}m, dy:{dy:.1f}m]"
            (tw, th), _ = cv2.getTextSize(ruler_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)

            # Glassmorphic pill badge background
            cv2.rectangle(canvas, (mid_x - tw // 2 - 6, mid_y - th - 6), (mid_x + tw // 2 + 6, mid_y + 4), (15, 23, 42), -1)
            cv2.rectangle(canvas, (mid_x - tw // 2 - 6, mid_y - th - 6), (mid_x + tw // 2 + 6, mid_y + 4), (0, 215, 255), 1)
            cv2.putText(canvas, ruler_txt, (mid_x - tw // 2, mid_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1, cv2.LINE_AA)

        return canvas

    def render_homography_simulation_pitch(
        self,
        positions: Any,
        dst_landmarks: Optional[np.ndarray] = None,
        show_quad: bool = True,
        show_coordinates: bool = True
    ) -> np.ndarray:
        """
        Renders the 2D FIFA metric pitch simulation for Stage 2 Homography debugging.
        Displays calibration destination landmarks (L1-L4), metric coordinates (Xm, Ym),
        and player metric positions in the top-down FIFA coordinate system.
        """
        canvas = self.draw_base_pitch()
        
        # 1. Destination Landmark Calibration Quad on 2D Pitch
        if show_quad and dst_landmarks is not None:
            quad_px = [self.to_pixel(p[0], p[1]) for p in dst_landmarks]
            if len(quad_px) >= 3:
                overlay = canvas.copy()
                cv2.fillPoly(overlay, [np.array(quad_px, dtype=np.int32)], (0, 200, 255))
                cv2.addWeighted(overlay, 0.15, canvas, 0.85, 0, canvas)
                cv2.polylines(canvas, [np.array(quad_px, dtype=np.int32)], isClosed=True, color=(0, 255, 255), thickness=2, lineType=cv2.LINE_AA)
                
            for i, pt in enumerate(dst_landmarks):
                px = self.to_pixel(pt[0], pt[1])
                cv2.drawMarker(canvas, px, (0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=16, thickness=2, line_type=cv2.LINE_AA)
                cv2.circle(canvas, px, 8, (0, 255, 255), 2, lineType=cv2.LINE_AA)
                cv2.circle(canvas, px, 4, (0, 0, 0), -1, lineType=cv2.LINE_AA)
                lbl = f"L{i+1}: ({pt[0]:.1f}m, {pt[1]:.1f}m)"
                cv2.putText(canvas, lbl, (px[0] + 8, px[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)

        # 2. Render Team Players in Metric Pitch Space
        team_a = getattr(positions, 'team_a', [])
        team_b = getattr(positions, 'team_b', [])
        ball = getattr(positions, 'ball', None)
        carrier_idx = getattr(positions, 'carrier_idx', None)
        
        canvas = self.render_players(canvas, team_a, team_b, ball, carrier_idx)
        
        # 3. Optional Coordinate Tags next to each player
        if show_coordinates:
            for coords, col in [(team_a, (255, 220, 100)), (team_b, (150, 150, 255))]:
                for (x, y) in coords:
                    px, py = self.to_pixel(x, y)
                    tag = f"{x:.1f},{y:.1f}m"
                    cv2.putText(canvas, tag, (px + 9, py + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.32, col, 1, cv2.LINE_AA)
                    
        # Header title banner
        cv2.rectangle(canvas, (0, 0), (self.width, 28), (10, 14, 20), -1)
        cv2.line(canvas, (0, 28), (self.width, 28), (0, 240, 255), 1)
        cv2.putText(canvas, "STAGE 2: 2D METRIC SIMULATION PITCH [0, 105]m x [0, 68]m", (14, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 200), 1, cv2.LINE_AA)
        
        return canvas
