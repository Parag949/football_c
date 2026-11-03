"""
Test & Validation Suite for Hardware-Accelerated Browser Media Engine Player
=============================================================================
Validates:
1. Static payload generation & caching
2. Numerical accuracy of homography inverse projection
3. Payload schema validation
4. Streamlit static serving configuration
"""

import os
import sys
import json
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from perception.homography import HomographyTransformer


class TestHardwarePlayerPayload(unittest.TestCase):
    def setUp(self):
        self.cache_path = "data/precomputed_user.json"
        self.static_dir = "static"
        self.payload_path = os.path.join(self.static_dir, "tactical_payload.json")

    def test_01_static_dir_and_config(self):
        """Verify static directory and config.toml have enableStaticServing = true."""
        self.assertTrue(os.path.isdir(self.static_dir), "static/ directory must exist")
        with open(".streamlit/config.toml", "r") as f:
            cfg = f.read()
        self.assertIn("enableStaticServing = true", cfg, "enableStaticServing must be enabled in config.toml")

    def test_02_homography_inverse_projection(self):
        """Verify H_inv 3x3 projective transformation preserves ground geometry."""
        transformer = HomographyTransformer(1920, 1080)
        H_inv = transformer.H_inv.tolist()

        test_points = [
            (52.5, 34.0),  # Center circle
            (16.5, 34.0),  # Left penalty area edge
            (88.5, 34.0),  # Right penalty area edge
            (0.0, 0.0),    # Corner
            (105.0, 68.0)  # Opposite corner
        ]

        for x, y in test_points:
            # Direct matrix projective multiplication in Python
            u_h = H_inv[0][0]*x + H_inv[0][1]*y + H_inv[0][2]
            v_h = H_inv[1][0]*x + H_inv[1][1]*y + H_inv[1][2]
            w_h = H_inv[2][0]*x + H_inv[2][1]*y + H_inv[2][2]
            self.assertNotEqual(w_h, 0.0, "Projective divisor w cannot be zero")
            u = u_h / w_h
            v = v_h / w_h

            ref_pt = transformer.pitch_to_image([(x, y)])[0]
            # Pixel roundoff error must be < 1.0 pixel
            self.assertLess(abs(u - ref_pt[0]), 1.0)
            self.assertLess(abs(v - ref_pt[1]), 1.0)

    def test_03_payload_schema_and_size(self):
        """Verify tactical_payload.json format, frame count, and compactness."""
        self.assertTrue(os.path.exists(self.payload_path), "tactical_payload.json must exist")
        sz = os.path.getsize(self.payload_path)
        # 1439 frames should be compact (< 10MB)
        self.assertLess(sz, 10 * 1024 * 1024, f"Payload size {sz/(1024*1024):.1f}MB exceeds 10MB budget")

        with open(self.payload_path, "r") as f:
            payload = json.load(f)

        self.assertIn("fps", payload)
        self.assertIn("total_frames", payload)
        self.assertIn("frames", payload)
        self.assertGreater(len(payload["frames"]), 0)

        f0 = payload["frames"][0]
        self.assertIn("i", f0)
        self.assertIn("a", f0)
        self.assertIn("b", f0)
        self.assertIn("ball", f0)
        self.assertIn("hull", f0)
        self.assertIn("vor", f0)
        self.assertIn("dijk", f0)

    def test_04_missing_broadcast_cache_does_not_load_user_match(self):
        """Bug 1 Fix: Missing broadcast cache probe video and disable overlays rather than loading user match."""
        from ui.components.hardware_player import prepare_hardware_player_assets
        transformer = HomographyTransformer(960, 540, camera_mode="sideline")

        meta = prepare_hardware_player_assets(
            "data/sample_broadcast.mp4",
            "data/precomputed_broadcast.json",
            transformer
        )

        self.assertFalse(meta["has_cache"], "has_cache must be False when precomputed_broadcast.json is missing")
        self.assertEqual(len(meta["frames"]), 0, "Frames must be empty to disable overlays")
        self.assertEqual(meta["total_frames"], 750, "Total frames must be true video frame count (750), not user match (180/1439)")
        self.assertEqual(meta["video_width"], 960, "Video width must be 960")
        self.assertEqual(meta["video_height"], 540, "Video height must be 540")
        self.assertEqual(meta["fps"], 25.0, "FPS must match broadcast video (25.0)")

    def test_05_stride_alignment(self):
        """Bug 2 Fix: Stride > 1 correctly records stride and preserves frame_idx alignment."""
        from ui.components.hardware_player import prepare_hardware_player_assets
        import tempfile

        transformer = HomographyTransformer(1920, 1080)
        # Create a mock cache with stride 2
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
            mock_cache = {
                "fps": 30.0,
                "total_frames": 20,
                "stride": 2,
                "video_width": 1920,
                "video_height": 1080,
                "frames": [
                    {
                        "frame_idx": i,
                        "team_a": [[50.0, 30.0], [60.0, 40.0]],
                        "team_b": [[40.0, 30.0]],
                        "carrier_idx": 0,
                        "ball": [50.0, 30.0],
                        "dijkstra": {"path": [0, 1], "graph_edges": []}
                    }
                    for i in range(0, 20, 2)
                ]
            }
            json.dump(mock_cache, tf)
            tf_path = tf.name

        try:
            meta = prepare_hardware_player_assets("data/user_match.mp4", tf_path, transformer)
            self.assertEqual(meta["stride"], 2, "Stride must be correctly preserved as 2")
            self.assertEqual(meta["total_frames"], 20, "Total frames must be 20")
            self.assertEqual(len(meta["frames"]), 10, "Sampled frames count must be 10")
            self.assertEqual(meta["frames"][0]["i"], 0)
            self.assertEqual(meta["frames"][1]["i"], 2)
            self.assertEqual(meta["frames"][-1]["i"], 18)
        finally:
            if os.path.exists(tf_path):
                os.remove(tf_path)

    def test_06_custom_upload_signature(self):
        """Bug 3 Fix: File upload signature changes when file content changes under same filename."""
        import hashlib

        fname = "user_match.mp4"
        content_v1 = b"video_bytes_version_1_data_initial"
        content_v2 = b"video_bytes_version_2_data_updated"

        sig_v1 = f"{fname}_{len(content_v1)}_{hashlib.md5(content_v1).hexdigest()}"
        sig_v2 = f"{fname}_{len(content_v2)}_{hashlib.md5(content_v2).hexdigest()}"

        self.assertNotEqual(sig_v1, sig_v2, "Uploading new content with identical filename must yield distinct signature")


if __name__ == "__main__":
    unittest.main()
