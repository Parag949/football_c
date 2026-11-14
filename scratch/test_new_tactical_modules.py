"""
Unit Test Suite for New Advanced Tactical & Perception Modules
==============================================================
Validates:
1. Continuous Pitch Control probability field & heatmap generation (Spearman model)
2. Expected Threat (xT) Karun Singh grid, Delta-xT calculations, and killer pass classification
3. Automated Tactical Formation Detection (4-3-3, 4-2-3-1, 4-4-2, line heights, compactness)
4. Perceptual CIELAB + HSV Jersey Color Classification & Turf Subtraction
"""

import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from algorithms.pitch_control import compute_pitch_control_field, render_pitch_control_heatmap
from algorithms.expected_threat import get_point_xt, evaluate_action_threat, rank_passing_lanes_by_threat, render_xt_heatmap
from algorithms.formation_detector import detect_team_formation
from perception.jersey_classifier import extract_torso_color_feature, classify_player_jerseys
from core.contracts import PlayerDetection


class TestNewTacticalModules(unittest.TestCase):
    def test_01_pitch_control_field(self):
        """Verify continuous pitch control probability surface calculations."""
        # Team A clustered on the left (X ~ 25m), Team B clustered on the right (X ~ 75m)
        team_a = [(20.0, 34.0), (25.0, 20.0), (25.0, 48.0), (35.0, 34.0)]
        team_b = [(70.0, 34.0), (75.0, 20.0), (75.0, 48.0), (80.0, 34.0)]

        res = compute_pitch_control_field(team_a, team_b, grid_size=(50, 32))
        self.assertIn('control_grid', res)
        self.assertEqual(res['control_grid'].shape, (32, 50))
        self.assertTrue(np.all((res['control_grid'] >= 0.0) & (res['control_grid'] <= 1.0)))

        # Team A must dominate left side of pitch (X < 35m)
        left_control = res['control_grid'][:, :15]
        self.assertGreater(float(np.mean(left_control)), 0.60)

        # Team B must dominate right side of pitch (X > 70m)
        right_control = res['control_grid'][:, 35:]
        self.assertLess(float(np.mean(right_control)), 0.40)

        # Percentage sum check
        total_pct = res['team_a_pct'] + res['team_b_pct'] + res['uncontrolled_pct']
        self.assertAlmostEqual(total_pct, 100.0, delta=1.0)

        # Heatmap rendering
        heatmap = render_pitch_control_heatmap(res['control_grid'], width=500, height=320)
        self.assertEqual(heatmap.shape, (320, 500, 3))
        self.assertEqual(heatmap.dtype, np.uint8)

    def test_02_expected_threat_model(self):
        """Verify Expected Threat (xT) model, Delta-xT, and progressive/killer pass identification."""
        # Own half (X=30m) has low threat; opponent 6-yard box (X=100m, Y=34m) has high threat
        xt_own = get_point_xt(20.0, 34.0, attack_direction="right")
        xt_opp_box = get_point_xt(100.0, 34.0, attack_direction="right")
        self.assertGreater(xt_opp_box, xt_own)
        self.assertGreater(xt_opp_box, 0.20)

        # Direction symmetry
        xt_left_attack = get_point_xt(5.0, 34.0, attack_direction="left")
        self.assertAlmostEqual(xt_left_attack, xt_opp_box, delta=0.05)

        # Progressive high-threat pass evaluation
        pass_res = evaluate_action_threat((50.0, 34.0), (95.0, 34.0), attack_direction="right")
        self.assertTrue(pass_res['is_progressive'])
        self.assertTrue(pass_res['is_killer_pass'])
        self.assertGreater(pass_res['delta_xt'], 0.10)

        # Safe back-pass evaluation
        back_pass = evaluate_action_threat((70.0, 34.0), (40.0, 34.0), attack_direction="right")
        self.assertFalse(back_pass['is_progressive'])
        self.assertFalse(back_pass['is_killer_pass'])
        self.assertLess(back_pass['delta_xt'], 0.0)

        # Passing lane ranking
        carrier = (60.0, 34.0)
        targets = [(50.0, 34.0), (90.0, 34.0), (62.0, 10.0)]
        ranked = rank_passing_lanes_by_threat(carrier, targets, attack_direction="right")
        self.assertEqual(ranked[0]['target_pos'], (90.0, 34.0), "Highest threat pass should be ranked #1")

        # Heatmap rendering
        xt_map = render_xt_heatmap(width=400, height=260)
        self.assertEqual(xt_map.shape, (260, 400, 3))

    def test_03_formation_detection(self):
        """Verify automated tactical shape and line height detection."""
        # Construct standard 4-3-3 formation
        # 1 GK, 4 defenders at X=25m, 3 midfielders at X=48m, 3 forwards at X=75m
        four_three_three = [
            (8.0, 34.0),   # GK
            (25.0, 12.0), (25.0, 26.0), (25.0, 42.0), (25.0, 56.0),  # 4 Defenders
            (48.0, 20.0), (48.0, 34.0), (48.0, 48.0),                # 3 Midfielders
            (75.0, 15.0), (75.0, 34.0), (75.0, 53.0)                 # 3 Forwards
        ]

        f_res = detect_team_formation(four_three_three, attack_direction="right")
        self.assertEqual(f_res['formation'], "4-3-3")
        self.assertEqual(f_res['line_counts'], (4, 3, 3))
        self.assertAlmostEqual(f_res['defensive_line_m'], 25.0, delta=1.5)
        self.assertAlmostEqual(f_res['compactness_m'], 50.0, delta=2.0)
        self.assertGreater(f_res['team_width_m'], 40.0)

        # Construct standard 4-4-2 formation
        four_four_two = [
            (6.0, 34.0),   # GK
            (22.0, 10.0), (22.0, 26.0), (22.0, 42.0), (22.0, 58.0),  # 4 Defenders
            (45.0, 12.0), (45.0, 28.0), (45.0, 40.0), (45.0, 56.0),  # 4 Midfielders
            (72.0, 28.0), (72.0, 40.0)                               # 2 Forwards
        ]
        f_res_442 = detect_team_formation(four_four_two, attack_direction="right")
        self.assertEqual(f_res_442['formation'], "4-4-2")
        self.assertEqual(f_res_442['line_counts'], (4, 4, 2))

    def test_04_jersey_color_classification(self):
        """Verify CIELAB + HSV torso feature extraction and K-Means++ kit clustering."""
        frame = np.full((720, 1280, 3), (40, 120, 40), dtype=np.uint8)  # Grass background

        # Team 0: Bright Cyan jerseys (B=255, G=200, R=30)
        # Team 1: Deep Red jerseys (B=20, G=20, R=220)
        players = []
        for i in range(5):
            bx1, by1 = 100 + i * 50, 200
            bx2, by2 = bx1 + 30, by1 + 70
            frame[by1 + 15:by1 + 35, bx1 + 8:bx2 - 8] = [255, 200, 30]  # Cyan torso
            players.append({
                'box': (bx1, by1, bx2, by2),
                'conf': 0.9,
                'cls_id': 0,
                'color_feat': extract_torso_color_feature(frame, (bx1, by1, bx2, by2))
            })

        for i in range(5):
            bx1, by1 = 600 + i * 50, 200
            bx2, by2 = bx1 + 30, by1 + 70
            frame[by1 + 15:by1 + 35, bx1 + 8:bx2 - 8] = [20, 20, 220]  # Red torso
            players.append({
                'box': (bx1, by1, bx2, by2),
                'conf': 0.88,
                'cls_id': 0,
                'color_feat': extract_torso_color_feature(frame, (bx1, by1, bx2, by2))
            })

        classified = classify_player_jerseys(players, frame=frame)
        self.assertEqual(len(classified), 10)

        # Team 0 players must have team_id=0, Team 1 players must have team_id=1
        team_0_assigned = [p for p in classified[:5] if p.team_id == 0]
        team_1_assigned = [p for p in classified[5:] if p.team_id == 1]
        self.assertEqual(len(team_0_assigned), 5, "All 5 cyan players must be assigned to Team 0")
        self.assertEqual(len(team_1_assigned), 5, "All 5 red players must be assigned to Team 1")


if __name__ == "__main__":
    unittest.main()
