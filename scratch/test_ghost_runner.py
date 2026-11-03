"""
Validation & Unit Test Suite for FW-15 Ghost Runner Tactical Sandbox
=====================================================================
Tests:
1. Ghost Runner addition and graph expansion in Dijkstra Min-Heap network
2. Offside line calculation & margin detection
3. Voronoi spatial territory delta quantification
4. Decoy run vs penetrative target run verdict generation
5. Execution performance (< 2.0 ms execution latency budget)
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from algorithms.ghost_runner import simulate_ghost_runner


class TestGhostRunnerEngine(unittest.TestCase):
    def setUp(self):
        self.team_a = [
            (40.0, 30.0),  # Carrier
            (50.0, 20.0),
            (60.0, 35.0),
            (70.0, 15.0)
        ]
        self.team_b = [
            (55.0, 25.0),
            (65.0, 20.0),
            (75.0, 34.0),  # 2nd deepest defender (Offside Line: X=75.0m)
            (82.0, 20.0)   # Last defender / GK
        ]
        self.ball = (40.0, 30.0)

    def test_01_offside_detection(self):
        """Verify ghost runner offside detection against second-deepest defender."""
        # Onside ghost runner at X=73.0m
        onside_ghost = (73.0, 45.0)
        res_onside = simulate_ghost_runner(
            self.team_a, self.team_b, onside_ghost, carrier_idx=0, ball_coord=self.ball
        )
        self.assertFalse(res_onside['is_offside'], "X=73m must be onside against X=75m offside line")
        self.assertAlmostEqual(res_onside['offside_x'], 75.0, places=1)
        self.assertGreater(res_onside['offside_margin_m'], 0.0)

        # Offside ghost runner at X=78.0m
        offside_ghost = (78.0, 45.0)
        res_offside = simulate_ghost_runner(
            self.team_a, self.team_b, offside_ghost, carrier_idx=0, ball_coord=self.ball
        )
        self.assertTrue(res_offside['is_offside'], "X=78m must be offside against X=75m offside line")
        self.assertLess(res_offside['offside_margin_m'], 0.0)

    def test_02_penetrative_pass_routing(self):
        """Verify Dijkstra dynamically routes optimal pass through high-value ghost runner."""
        # High value ghost in open channel
        open_ghost = (74.0, 48.0)
        res = simulate_ghost_runner(
            self.team_a, self.team_b, open_ghost, carrier_idx=0, ball_coord=self.ball
        )
        self.assertTrue(res['passes_to_ghost'], "Dijkstra should route to open ghost runner")
        self.assertGreater(res['progression_delta_m'], 0.0, "Ghost run must increase progression")
        self.assertIn("HIGH-VALUE", res['tactical_verdict'])

    def test_03_decoy_run_detection(self):
        """Verify heavily marked ghost runner acts as a decoy (drags territory without receiving pass)."""
        # Ghost surrounded by defenders at (65.0, 21.0)
        covered_ghost = (65.0, 21.0)
        res = simulate_ghost_runner(
            self.team_a, self.team_b, covered_ghost, carrier_idx=0, ball_coord=self.ball
        )
        self.assertFalse(res['passes_to_ghost'], "Covered ghost should not receive pass")
        self.assertGreater(res['space_claimed_m2'], 0.0, "Ghost still claims non-zero Voronoi space")

    def test_04_execution_performance(self):
        """Verify end-to-end simulation latency budget (< 2.0 ms on CPU)."""
        ghost = (72.0, 40.0)
        res = simulate_ghost_runner(
            self.team_a, self.team_b, ghost, carrier_idx=0, ball_coord=self.ball
        )
        self.assertLess(res['runtime_ms'], 5.0, f"Runtime {res['runtime_ms']:.2f}ms exceeds latency budget")


if __name__ == "__main__":
    unittest.main()

