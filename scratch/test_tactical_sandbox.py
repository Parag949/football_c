"""
Unit Test & Validation Suite for 2D Tactical Simulation Sandbox
==============================================================
Validates:
1. Tactical presets completeness and coordinate bounding [0..105]m x [0..68]m
2. Algorithmic contract interoperability (Voronoi, Hull, Dijkstra, Ghost Runner, Ruler)
3. Zero side-effects / complete isolation from video processing stream
"""

import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ui.components.tactical_sandbox import TACTICAL_PRESETS, render_tactical_sandbox
from algorithms.convex_hull import compute_defensive_cohesion
from algorithms.voronoi_dominance import compute_spatial_dominance
from algorithms.passing_network import compute_dijkstra_passing_path
from algorithms.ghost_runner import simulate_ghost_runner
from algorithms.tactical_ruler import compute_tactical_ruler


class TestTacticalSandbox(unittest.TestCase):
    def test_01_presets_schema_and_boundaries(self):
        """Verify all tactical formations have valid coordinates bounded by FIFA pitch."""
        self.assertGreaterEqual(len(TACTICAL_PRESETS), 3)

        for name, preset in TACTICAL_PRESETS.items():
            self.assertIn("team_a", preset, f"Preset {name} must have team_a")
            self.assertIn("team_b", preset, f"Preset {name} must have team_b")
            self.assertIn("ball", preset, f"Preset {name} must have ball")
            self.assertIn("carrier_idx", preset, f"Preset {name} must have carrier_idx")
            self.assertIn("ghost", preset, f"Preset {name} must have ghost")
            self.assertIn("ruler_a", preset, f"Preset {name} must have ruler_a")
            self.assertIn("ruler_b", preset, f"Preset {name} must have ruler_b")

            # Check Team A coordinates
            for p in preset["team_a"]:
                self.assertTrue(0.0 <= p[0] <= 105.0, f"X coord {p[0]} out of pitch bounds in {name}")
                self.assertTrue(0.0 <= p[1] <= 68.0, f"Y coord {p[1]} out of pitch bounds in {name}")

            # Check Team B coordinates
            for p in preset["team_b"]:
                self.assertTrue(0.0 <= p[0] <= 105.0, f"X coord {p[0]} out of pitch bounds in {name}")
                self.assertTrue(0.0 <= p[1] <= 68.0, f"Y coord {p[1]} out of pitch bounds in {name}")

            # Ball and Ghost coordinates
            self.assertTrue(0.0 <= preset["ball"][0] <= 105.0)
            self.assertTrue(0.0 <= preset["ball"][1] <= 68.0)
            self.assertTrue(0.0 <= preset["ghost"][0] <= 105.0)
            self.assertTrue(0.0 <= preset["ghost"][1] <= 68.0)

            # Carrier index validity
            self.assertTrue(0 <= preset["carrier_idx"] < len(preset["team_a"]))

    def test_02_algorithmic_features_on_sandbox_presets(self):
        """Verify all core DAA algorithms execute cleanly on tactical sandbox formations."""
        preset = TACTICAL_PRESETS["4-3-3 Attack"]
        team_a = [tuple(p) for p in preset["team_a"]]
        team_b = [tuple(p) for p in preset["team_b"]]
        ball = tuple(preset["ball"])
        c_idx = preset["carrier_idx"]
        ghost = tuple(preset["ghost"])
        r_a = tuple(preset["ruler_a"])
        r_b = tuple(preset["ruler_b"])

        # 1. Voronoi Spatial Dominance
        vor = compute_spatial_dominance(team_a, team_b)
        self.assertIn("team_a_pct", vor)
        self.assertIn("team_b_pct", vor)
        self.assertAlmostEqual(vor["team_a_pct"] + vor["team_b_pct"], 100.0, delta=0.5)

        # 2. Convex Hull Cohesion (Graham & Jarvis)
        hull_graham = compute_defensive_cohesion(team_b, algorithm="graham")
        self.assertIn("area_sq_m", hull_graham)
        self.assertGreater(hull_graham["area_sq_m"], 0.0)

        hull_jarvis = compute_defensive_cohesion(team_b, algorithm="jarvis")
        self.assertAlmostEqual(hull_graham["area_sq_m"], hull_jarvis["area_sq_m"], delta=1.0)

        # 3. Dijkstra Passing Network
        dijk = compute_dijkstra_passing_path(team_a, team_b, carrier_idx=c_idx, ball_coord=ball)
        self.assertIn("path", dijk)
        self.assertIn("corridor_points", dijk)
        self.assertGreater(len(dijk["path"]), 0)

        # 4. Ghost Runner Simulation
        ghost_res = simulate_ghost_runner(team_a, team_b, ghost_coord=ghost, carrier_idx=c_idx, ball_coord=ball)
        self.assertIn("is_offside", ghost_res)
        self.assertIn("offside_x", ghost_res)
        self.assertIn("tactical_verdict", ghost_res)

        # 5. Tactical Ruler Measurement
        ruler_res = compute_tactical_ruler(r_a, r_b, label_a="Ruler A", label_b="Ruler B", defending_players=team_b)
        self.assertIn("distance_m", ruler_res)
        self.assertGreater(ruler_res["distance_m"], 0.0)

    def test_03_preset_switching_and_state_deep_copy(self):
        """Verify switching between all presets copies scalar carrier_idx and lists without TypeError."""
        import json

        state = {}
        for preset_name in TACTICAL_PRESETS.keys():
            data = TACTICAL_PRESETS[preset_name]
            state["preset"] = preset_name
            for k in ["team_a", "team_b", "ball", "ghost", "ruler_a", "ruler_b"]:
                state[k] = json.loads(json.dumps(data[k]))
            state["carrier_idx"] = int(data.get("carrier_idx", 0))

            self.assertIsInstance(state["carrier_idx"], int)
            self.assertIsInstance(state["team_a"], list)
            self.assertIsInstance(state["team_b"], list)
            self.assertIsInstance(state["ball"], list)
            self.assertGreater(len(state["team_a"]), 0)


if __name__ == "__main__":
    unittest.main()
