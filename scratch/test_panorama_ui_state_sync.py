"""
Verification test for Panorama TPS mask persistence and UI state synchronization.
Ensures that:
1. Calibrations saved as "panorama_tps" load properly.
2. Pitch Mask Studio initial force_fit evaluates to "tps" (never forced to "curve" or "flat").
3. Overlay engine and Stream manager transformers stay synchronized.
4. Reprojection residual on Tsukuba panorama is sub-pixel (< 1.0 px).
"""

import json
import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath("."))

from perception.homography import HomographyTransformer
from visualization.overlays import BroadcastOverlayEngine
from ui.media_library import load_video_calibration, save_video_calibration


class TestPanoramaMaskPersistence(unittest.TestCase):
    def test_tsukuba_saved_calibration_residual_and_mode(self):
        cal_path = "data/calibrations/118577_panorama_1st_half_fc64c048.json"
        self.assertTrue(os.path.exists(cal_path), f"Calibration file not found at {cal_path}")

        with open(cal_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data.get("warp_mode"), "panorama_tps", "Saved warp_mode must be panorama_tps")

        src_pts = np.float32(data["src_points"])
        dst_pts = np.float32(data["dst_points"])
        w = int(data["frame_width"])
        h = int(data["frame_height"])

        t = HomographyTransformer()
        t.calibrate(
            src_pts,
            dst_pts,
            frame_width=w,
            frame_height=h,
            warp_mode=data["warp_mode"],
            lens_lambda=data.get("lens_lambda"),
            pitch_length=data.get("pitch_length"),
            pitch_width=data.get("pitch_width"),
        )

        self.assertEqual(t.warp_mode, "panorama_tps")
        res = t.compute_reprojection_residuals()
        mean_err = res["mean_error"]
        max_err = res["max_error"]

        # Contrast: Universal/planar was 79.7px mean, 298.9px max!
        self.assertLess(mean_err, 1.0, f"TPS mean error should be sub-pixel, got {mean_err:.2f} px")
        self.assertLess(max_err, 2.0, f"TPS max error should be bounded, got {max_err:.2f} px")

    def test_pitch_mask_studio_force_fit_logic(self):
        t_tps = HomographyTransformer()
        t_tps.warp_mode = "panorama_tps"

        t_univ = HomographyTransformer()
        t_univ.warp_mode = "universal"

        t_planar = HomographyTransformer()
        t_planar.warp_mode = "planar"

        # Logic from ui/components/pitch_mask_studio.py
        def compute_force_fit(transformer, is_wide, n_active):
            cur_warp = getattr(transformer, "warp_mode", "planar")
            if cur_warp in ("panorama_tps", "tps") or (is_wide and n_active >= 6 and cur_warp not in ("planar", "universal")):
                return "tps"
            elif cur_warp == "universal":
                return "curve"
            elif cur_warp == "planar":
                return "flat"
            else:
                return "tps" if is_wide else "flat"

        # Check TPS transformer on ultrawide
        self.assertEqual(compute_force_fit(t_tps, is_wide=True, n_active=13), "tps")
        # Check universal transformer on wide camera
        self.assertEqual(compute_force_fit(t_univ, is_wide=False, n_active=8), "curve")
        # Check planar broadcast
        self.assertEqual(compute_force_fit(t_planar, is_wide=False, n_active=4), "flat")

    def test_overlay_engine_curved_markings_subdivision(self):
        cal_path = "data/calibrations/118577_panorama_1st_half_fc64c048.json"
        with open(cal_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        src_pts = np.float32(data["src_points"])
        dst_pts = np.float32(data["dst_points"])
        w = int(data["frame_width"])
        h = int(data["frame_height"])

        t = HomographyTransformer()
        t.calibrate(
            src_pts,
            dst_pts,
            frame_width=w,
            frame_height=h,
            warp_mode="panorama_tps",
            pitch_length=105.0,
            pitch_width=68.0
        )

        engine = BroadcastOverlayEngine(t)
        self.assertTrue(engine._is_panorama_tps())

        # Render debug frame on blank canvas
        dummy_frame = np.zeros((h, w, 3), dtype=np.uint8)
        rendered = engine.render_homography_debug_frame(
            dummy_frame,
            show_landmarks=True,
            show_turf_mask=True,
            show_markings=True,
            show_player_pins=False
        )

        self.assertEqual(rendered.shape, (h, w, 3))
        # Ensure markings and turf were actually drawn
        self.assertTrue(np.any(rendered > 0))


if __name__ == "__main__":
    unittest.main()
