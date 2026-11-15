"""
Unit test verifying cache and calibration separation, naming conventions,
slug determinism, and static symlink isolation.
"""

import os
import shutil
import unittest
from ui.media_library import (
    _video_identity_slug,
    get_calibration_path,
    get_custom_cache_path,
    resolve_video_cache_path,
    load_video_calibration,
    save_video_calibration,
    delete_video_calibration,
    invalidate_custom_user_cache,
    CUSTOM_CALIBRATIONS_DIR,
    CUSTOM_CACHES_DIR,
)
from perception.homography import HomographyTransformer
from ui.components.player_assets import prepare_hardware_player_assets


class TestCacheCalibrationIsolation(unittest.TestCase):
    def setUp(self):
        self.test_video = "data/user_match.mp4"
        self.test_calib_payload = {
            "frame_width": 1920,
            "frame_height": 1080,
            "src_points": [[100, 100], [1800, 100], [1800, 1000], [100, 1000]],
            "dst_points": [[0, 0], [105, 0], [105, 68], [0, 68]],
            "warp_mode": "planar",
        }

    def test_distinct_file_extensions(self):
        calib_path = get_calibration_path("data/test_clip_abc.mp4")
        cache_path = get_custom_cache_path("data/test_clip_abc.mp4")
        
        self.assertTrue(calib_path.endswith("_calib.json"), f"Expected _calib.json, got {calib_path}")
        self.assertTrue(cache_path.endswith("_cache.json"), f"Expected _cache.json, got {cache_path}")
        self.assertIn("data/calibrations", calib_path)
        self.assertIn("data/custom_caches", cache_path)
        self.assertNotEqual(os.path.basename(calib_path), os.path.basename(cache_path))

    def test_slug_invariance_across_path_formats(self):
        slug1 = _video_identity_slug("data/user_match.mp4")
        slug2 = _video_identity_slug("./data/user_match.mp4")
        slug3 = _video_identity_slug(os.path.abspath("data/user_match.mp4"))
        self.assertEqual(slug1, slug2)
        self.assertEqual(slug1, slug3)
        self.assertEqual(slug1, "user_match_e49e9dab")

    def test_legacy_calibration_loading(self):
        # user_match_e49e9dab.json exists as legacy
        cal = load_video_calibration("data/user_match.mp4")
        self.assertIsNotNone(cal)
        self.assertIn("src_points", cal)
        self.assertIn("dst_points", cal)

    def test_resolve_video_cache_path_priority(self):
        # When custom cache doesn't exist, falls back to default cache
        resolved = resolve_video_cache_path("data/user_match.mp4", "data/precomputed_user.json")
        self.assertEqual(resolved, "data/precomputed_user.json")

    def test_player_assets_unique_static_video(self):
        transformer = HomographyTransformer()
        meta = prepare_hardware_player_assets("data/user_match.mp4", "data/precomputed_user.json", transformer)
        self.assertIn("static_video_url", meta)
        self.assertIn("active_match_user_match_e49e9dab.mp4", meta["static_video_url"])
        self.assertTrue(os.path.exists("static/active_match_user_match_e49e9dab.mp4"))
        self.assertTrue(os.path.exists("static/active_match.mp4"))


if __name__ == "__main__":
    unittest.main()
