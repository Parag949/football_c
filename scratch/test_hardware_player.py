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


if __name__ == "__main__":
    unittest.main()
