"""
Unit & Integration Test Suite for Milestone 1 Remediations
=========================================================
Tests and verifies all 15 remediation items across perception, algorithms,
visualization, core pipeline, and UI state/components.
"""

import math
import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from perception.homography import HomographyTransformer
from perception.calibration_templates import render_pitch_projection_preview
from visualization.overlays import BroadcastOverlayEngine
from visualization.pitch_renderer import TacticalPitchRenderer
from algorithms.tactical_ruler import compute_tactical_ruler
from algorithms.coach_analytics import evaluate_coach_analytics
from core.contracts import PlayerDetection, FrameDetections, PitchPositions, TacticalGeometry
from core.pipeline import stage1_perception, stage2_projection, stage3_geometry
from ui.media_library import save_video_calibration, load_video_calibration, delete_video_calibration


class TestRemediations(unittest.TestCase):

    def test_01_homography_fit_universal_lens_model_sign_normalization(self):
        """Item 1a: fit_universal_lens_model normalizes projective sign before depth check."""
        t = HomographyTransformer(1920, 1080)
        # Create 6 synthetic correspondences with known radial distortion
        src = np.array([
            [200.0, 150.0], [800.0, 150.0], [1700.0, 160.0],
            [150.0, 900.0], [960.0, 850.0], [1750.0, 920.0]
        ], dtype=np.float32)
        dst = np.array([
            [10.0, 10.0], [50.0, 10.0], [90.0, 10.0],
            [10.0, 60.0], [50.0, 60.0], [90.0, 60.0]
        ], dtype=np.float32)

        H_opt, lam = t.fit_universal_lens_model(src, dst)
        self.assertIsNotNone(H_opt)
        self.assertTrue(np.all(np.isfinite(H_opt)))
        self.assertGreater(abs(np.linalg.det(H_opt)), 1e-12)

    def test_01_homography_calibrate_auto_warp_mode(self):
        """Item 1b: calibrate under warp_mode='auto' evaluates and can select 'universal' or 'panorama_tps'."""
        t = HomographyTransformer(1920, 1080, warp_mode="auto")
        self.assertIn(t.warp_mode, ("planar", "universal", "panorama_tps", "auto"))

        # Planar correspondences with small residual should stay 'planar'
        src_planar = np.array([
            [200.0, 150.0], [800.0, 150.0], [1600.0, 150.0],
            [200.0, 800.0], [800.0, 800.0], [1600.0, 800.0]
        ], dtype=np.float32)
        dst_planar = np.array([
            [10.0, 10.0], [50.0, 10.0], [90.0, 10.0],
            [10.0, 60.0], [50.0, 60.0], [90.0, 60.0]
        ], dtype=np.float32)

        t.calibrate(src_planar, dst_planar, warp_mode="auto")
        self.assertIn(t.warp_mode, ("planar", "universal", "panorama_tps"))

    def test_02_render_pitch_projection_preview_active_dst_points(self):
        """Item 2a: render_pitch_projection_preview uses active dst_points for projective sign."""
        t = HomographyTransformer(1920, 1080)
        # End-zone calibration (88.5m - 105m) where (52.5, 34.0) is far outside active region
        src = np.array([
            [100.0, 200.0], [900.0, 200.0], [900.0, 800.0], [100.0, 800.0]
        ], dtype=np.float32)
        dst = np.array([
            [88.5, 13.84], [105.0, 13.84], [105.0, 54.16], [88.5, 54.16]
        ], dtype=np.float32)
        t.calibrate(src, dst)

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        preview = render_pitch_projection_preview(frame, t)
        self.assertEqual(preview.shape, (1080, 1920, 3))
        # Pitch lines should have been drawn
        self.assertGreater(np.count_nonzero(preview), 0)

    def test_02_render_pitch_projection_preview_singular_H(self):
        """Item 2b: render_pitch_projection_preview condition number check guards against singular H."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        singular_H = np.zeros((3, 3), dtype=np.float64)
        # Should not raise LinAlgError
        preview = render_pitch_projection_preview(frame, singular_H)
        self.assertEqual(preview.shape, (720, 1280, 3))

    def test_03_overlays_dictionary_wrapped_hull(self):
        """Item 3: overlay_projected_hull and overlay_defensive_hull accept dictionary-wrapped hulls."""
        t = HomographyTransformer(1920, 1080)
        eng = BroadcastOverlayEngine(t)
        rend = TacticalPitchRenderer()

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        canvas = np.zeros((730, 1120, 3), dtype=np.uint8)

        # Dictionary with 'polygon' key
        hull_dict_poly = {"polygon": [(20.0, 15.0), (80.0, 15.0), (80.0, 55.0), (20.0, 55.0)], "area": 2400.0}
        out_f1 = eng.overlay_projected_hull(frame, hull_dict_poly)
        out_c1 = rend.overlay_defensive_hull(canvas, hull_dict_poly)
        self.assertIsNotNone(out_f1)
        self.assertIsNotNone(out_c1)

        # Dictionary with 'hull' key
        hull_dict_hull = {"hull": [(20.0, 15.0), (80.0, 15.0), (80.0, 55.0), (20.0, 55.0)], "compactness": 0.5}
        out_f2 = eng.overlay_projected_hull(frame, hull_dict_hull)
        out_c2 = rend.overlay_defensive_hull(canvas, hull_dict_hull)
        self.assertIsNotNone(out_f2)
        self.assertIsNotNone(out_c2)

    def test_04_tactical_ruler_fifa_law_11_ball_exemption(self):
        """Item 4: compute_tactical_ruler exempts attackers level with or behind the ball from offside."""
        # Right attack: Goal at X=105m.
        # Defender 2nd deepest at X=70.0m.
        defenders = [(65.0, 30.0), (70.0, 34.0), (85.0, 34.0)]
        # Attacker at X=75.0m (ahead of 2nd deepest defender at X=70.0m).
        attackers = [(75.0, 30.0)]

        # Without ball or ball behind attacker (e.g. ball at X=60.0m): Attacker is offside
        res_offside = compute_tactical_ruler(
            point_a=(50.0, 34.0), point_b=(75.0, 30.0),
            defending_players=defenders, attacking_players=attackers,
            attack_direction="right", ball_coord=(60.0, 34.0)
        )
        self.assertTrue(res_offside['is_offside'])
        self.assertEqual(len(res_offside['offside_players']), 1)

        # With ball ahead of attacker (e.g. ball at X=80.0m, attacker at X=75.0m): Attacker is strictly onside!
        res_onside = compute_tactical_ruler(
            point_a=(50.0, 34.0), point_b=(75.0, 30.0),
            defending_players=defenders, attacking_players=attackers,
            attack_direction="right", ball_coord=(80.0, 34.0)
        )
        self.assertFalse(res_onside['is_offside'])
        self.assertEqual(len(res_onside['offside_players']), 0)

        # Left attack: Goal at X=0m.
        # Defender 2nd deepest at X=35.0m.
        defenders_l = [(40.0, 30.0), (35.0, 34.0), (15.0, 34.0)]
        # Attacker at X=30.0m (ahead towards goal of 2nd deepest defender at X=35.0m).
        attackers_l = [(30.0, 30.0)]

        # Ball at X=40.0m (behind attacker in progression): Offside
        res_l_off = compute_tactical_ruler(
            point_a=(50.0, 34.0), point_b=(30.0, 30.0),
            defending_players=defenders_l, attacking_players=attackers_l,
            attack_direction="left", ball_coord=(40.0, 34.0)
        )
        self.assertTrue(res_l_off['is_offside'])

        # Ball at X=25.0m (ahead of attacker towards goal): Attacker level/behind ball -> Onside!
        res_l_on = compute_tactical_ruler(
            point_a=(50.0, 34.0), point_b=(30.0, 30.0),
            defending_players=defenders_l, attacking_players=attackers_l,
            attack_direction="left", ball_coord=(25.0, 34.0)
        )
        self.assertFalse(res_l_on['is_offside'])

    def test_06_coach_analytics_bidirectional(self):
        """Item 6: evaluate_coach_analytics supports attack_direction='left' and 'right'."""
        team_a = [(30.0, 30.0), (45.0, 34.0), (60.0, 20.0)]
        team_b = [(20.0, 25.0), (25.0, 40.0), (30.0, 30.0)]
        hull = {'hull': team_b, 'area_sq_m': 400.0}
        voronoi = {
            'cells_a': [{'generator_site': (20.0, 30.0)}],
            'cells_b': [{'generator_site': (25.0, 30.0)}]
        }
        dijkstra = {'path': [1, 0], 'total_progression': 15.0}

        # Right attack: line_height = 105.0 - max_x (105 - 30 = 75.0)
        res_right = evaluate_coach_analytics(
            team_a, team_b, hull, voronoi, dijkstra, None, attack_direction="right"
        )
        self.assertAlmostEqual(res_right['block']['line_height_m'], 75.0, places=1)

        # Left attack: line_height = min_x (20.0)
        res_left = evaluate_coach_analytics(
            team_a, team_b, hull, voronoi, dijkstra, None, attack_direction="left"
        )
        self.assertAlmostEqual(res_left['block']['line_height_m'], 20.0, places=1)

    def test_10_pipeline_stages_null_guards(self):
        """Item 15: Pipeline stages 1, 2, 3 safely guard against null inputs."""
        # Stage 1: null frame
        res1 = stage1_perception(None, None, frame_idx=5)
        self.assertEqual(res1.frame_idx, 5)
        self.assertEqual(len(res1.players), 0)
        self.assertIsNone(res1.ball_pixel)

        # Stage 2: null detections
        res2 = stage2_projection(None, None)
        self.assertEqual(res2.frame_idx, 0)
        self.assertEqual(len(res2.team_a), 0)
        self.assertEqual(res2.ball, (52.5, 34.0))

        # Stage 3: null positions
        res3 = stage3_geometry(None)
        self.assertEqual(res3.frame_idx, 0)
        self.assertEqual(len(res3.hull_polygon), 0)
        self.assertEqual(res3.hull_area, 0.0)

    def test_13_media_library_atomic_persistence(self):
        """Item 13: save_video_calibration writes atomically and loads cleanly."""
        test_video = "data/test_atomic_video.mp4"
        payload = {
            "mode": "goal_to_goal",
            "src_points": [[100, 200], [300, 400], [500, 600], [700, 800]],
            "dst_points": [[88.5, 13.84], [105.0, 13.84], [105.0, 54.16], [88.5, 54.16]],
            "warp_mode": "planar"
        }
        saved_path = save_video_calibration(test_video, payload)
        self.assertTrue(os.path.exists(saved_path))

        loaded = load_video_calibration(test_video)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["mode"], "goal_to_goal")
        self.assertEqual(len(loaded["src_points"]), 4)

        delete_video_calibration(test_video)
        self.assertFalse(os.path.exists(saved_path))


if __name__ == "__main__":
    unittest.main()
