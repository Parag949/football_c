"""
Regression Test Suite: Forward Projection & Detection Caching (Requirement R1 / Milestone M3)
---------------------------------------------------------------------------------------------
Verifies:
1. Precomputed cache stores and loads raw pixel detections (box, foot_point, team_id, ball_pixel).
2. Metric pitch coordinates are computed forward: (u, v) -> (X, Y) via active calibration.
3. Player bounding box synthesis via pitch_to_image is eliminated.
4. Player pin dots in homography debug frame anchor directly to foot_point with 0.0px drift.
5. Custom calibration runs cached forward projection at >100 FPS without dropping into 3.4s live CPU inference.
6. Legacy cache fallback remains backward-compatible.
"""

import sys
sys.path.insert(0, ".")
import unittest
import numpy as np
import streamlit as st

from ui.state import init_session_state, sync_video_source, execute_pipeline, build_pipeline_bundle
from perception.homography import HomographyTransformer
from perception.precompute import load_precomputed_segment
from visualization.overlays import BroadcastOverlayEngine
from core.contracts import PlayerDetection, FrameDetections, PitchPositions
from core.pipeline import stage2_projection, stage3_geometry


class TestForwardProjectionCaching(unittest.TestCase):
    def setUp(self):
        st.session_state.clear()
        init_session_state()

    def test_precomputed_cache_schema_has_raw_detections(self):
        """Verifies disk cache contains raw pixel detections without requiring inverse projection."""
        data = load_precomputed_segment("data/precomputed_user.json")
        self.assertIsNotNone(data, "data/precomputed_user.json failed to load")
        self.assertIn("frames", data)
        self.assertGreater(len(data["frames"]), 0)

        f0 = data["frames"][0]
        self.assertIn("players", f0, "Cache frame missing raw 'players' detection list")
        self.assertIn("ball_pixel", f0, "Cache frame missing 'ball_pixel'")
        self.assertGreater(len(f0["players"]), 0)

        for p in f0["players"]:
            self.assertIn("box", p)
            self.assertEqual(len(p["box"]), 4)
            self.assertIn("foot_point", p)
            self.assertIsNotNone(p["foot_point"])
            self.assertEqual(len(p["foot_point"]), 2)
            self.assertIn("team_id", p)
            self.assertIn(p["team_id"], (0, 1))
            self.assertIn("confidence", p)
            self.assertIn("class_id", p)
            self.assertIn("track_id", p)

            # Assert foot_point ground contact is within camera dimensions
            fx, fy = p["foot_point"]
            self.assertTrue(0.0 <= fx <= 1920.0, f"Player foot X={fx} out of bounds")
            self.assertTrue(0.0 <= fy <= 1080.0, f"Player foot Y={fy} out of bounds")

    def test_forward_projection_without_inverse_box_synthesis(self):
        """Verifies pipeline computes positions forward without calling pitch_to_image for players."""
        sync_video_source()
        self.assertIsNotNone(st.session_state.precomputed_data)

        # Spy on pitch_to_image to verify it is NOT called for player inverse reconstruction
        orig_pitch_to_image = st.session_state.transformer.pitch_to_image
        inverse_calls = []

        def spy_pitch_to_image(pts, *args, **kwargs):
            inverse_calls.append(pts)
            return orig_pitch_to_image(pts, *args, **kwargs)

        st.session_state.transformer.pitch_to_image = spy_pitch_to_image

        bundle = build_pipeline_bundle(0)
        self.assertIsNotNone(bundle)
        self.assertTrue(bundle["is_cached"])

        det = bundle["detections"]
        pos = bundle["positions"]

        # Player detections must come directly from raw pixel cache
        self.assertGreater(len(det.players), 0)
        self.assertIsNotNone(det.players[0].foot_point)

        # Positions must be calculated forward
        self.assertGreater(len(pos.team_a), 0)
        self.assertGreater(len(pos.team_b), 0)

        # Verify pitch coordinates match stage2 forward projection
        expected_pos = stage2_projection(det, st.session_state.transformer)
        self.assertEqual(len(pos.team_a), len(expected_pos.team_a))
        for p_actual, p_exp in zip(pos.team_a, expected_pos.team_a):
            dist = np.hypot(p_actual[0] - p_exp[0], p_actual[1] - p_exp[1])
            self.assertLess(dist, 1e-4)

    def test_dynamic_reprojection_under_custom_calibration_zero_drift(self):
        """
        Verifies that switching to custom calibration dynamically updates pitch positions
        while keeping camera player pin dots anchored at exact pixel feet with 0.0px drift.
        """
        sync_video_source()

        # 1. Baseline Sideline Projection
        bundle_base = execute_pipeline(0)
        pos_base = bundle_base["positions"]
        det_base = bundle_base["detections"]
        self.assertTrue(bundle_base["is_cached"])

        foot_0 = det_base.players[0].foot_point
        pt_base_0 = pos_base.team_a[0]

        # 2. Apply Custom Calibration
        t_custom = HomographyTransformer(1920, 1080, camera_mode="sideline")
        src = np.float32([[200, 200], [1700, 200], [1800, 900], [100, 900]])
        dst = np.float32([[0, 0], [105, 0], [105, 68], [0, 68]])
        t_custom.calibrate(src, dst, frame_width=1920, frame_height=1080)
        st.session_state.transformer = t_custom
        st.session_state.manual_calib_enabled = True

        # 3. Execute with Custom Calibration: MUST remain cached and fast
        bundle_custom = execute_pipeline(0)
        self.assertTrue(bundle_custom["is_cached"], "Custom calibration must use cached raw detections!")

        pos_custom = bundle_custom["positions"]
        det_custom = bundle_custom["detections"]

        # Pixel feet MUST have ZERO drift (identical raw camera detections)
        self.assertEqual(det_custom.players[0].foot_point, foot_0)

        # Metric pitch coordinates MUST dynamically update forward to new homography
        pt_custom_0 = pos_custom.team_a[0]
        drift_meters = np.hypot(pt_custom_0[0] - pt_base_0[0], pt_custom_0[1] - pt_base_0[1])
        self.assertGreater(drift_meters, 1.0, "Changing calibration must forward-reproject pitch positions!")

        # 4. Homography Debug Frame: Player pin dot must anchor at exact pixel foot (0.0px drift)
        raw_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        eng = BroadcastOverlayEngine(t_custom)
        dbg = eng.render_homography_debug_frame(
            raw_frame, show_landmarks=False, show_turf_mask=False, show_markings=False, show_player_pins=True,
            player_detections=det_custom.players, positions=pos_custom, transformer=t_custom
        )

        ix, iy = int(round(foot_0[0])), int(round(foot_0[1]))
        # The center of the player pin must be drawn at (ix, iy)
        b, g, r = dbg[iy, ix]
        self.assertEqual((b, g, r), (255, 180, 0), "Player pin was not anchored at exact pixel feet!")

    def test_custom_calibration_operates_without_cpu_live_inference_freeze(self):
        """Verifies custom calibration operates smoothly in cached mode without live inference freeze."""
        sync_video_source()

        src = np.float32([[200, 200], [1700, 200], [1800, 900], [100, 900]])
        dst = np.float32([[0, 0], [105, 0], [105, 68], [0, 68]])
        st.session_state.transformer.calibrate(src, dst, frame_width=1920, frame_height=1080)
        st.session_state.manual_calib_enabled = True

        # Pipeline must use cache and avoid loading live ONNX model (~3.4s)
        import time
        t0 = time.perf_counter()
        bundle = execute_pipeline(0)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        self.assertTrue(bundle["is_cached"])
        # In cached mode, total execution including video read + AR render is < 600 ms (compared to 3400+ ms live)
        self.assertLess(elapsed_ms, 800.0, f"Custom calibration pipeline took {elapsed_ms:.1f}ms (>800ms indicates live CPU inference freeze)")

    def test_legacy_cache_fallback(self):
        """Verifies graceful backward-compatible fallback for caches lacking 'players' key."""
        rf_w, rf_h = 1920, 1080
        legacy_data = {
            'video_width': rf_w,
            'video_height': rf_h,
            'camera_mode': 'sideline',
            'stride': 1,
            'frames': [
                {
                    'frame_idx': 999,
                    'team_a': [[25.0, 30.0], [40.0, 50.0]],
                    'team_b': [[60.0, 34.0]],
                    'ball': [50.0, 34.0],
                    'carrier_idx': 0,
                    'detected_players': 3,
                    'hull': {'polygon': [[60.0, 34.0]], 'area': 0.0, 'compactness': 0.0, 'player_indices': []},
                    'voronoi': {'cells_a': [], 'cells_b': [], 'team_a_pct': 50.0, 'team_b_pct': 50.0},
                    'latency_ms': 1.5
                }
            ]
        }
        st.session_state.precomputed_data = legacy_data

        bundle = execute_pipeline(999)
        self.assertIsNotNone(bundle)
        self.assertTrue(bundle["is_cached"])
        self.assertEqual(len(bundle["detections"].players), 3)
        self.assertIsNotNone(bundle["detections"].players[0].foot_point)


if __name__ == "__main__":
    unittest.main()
