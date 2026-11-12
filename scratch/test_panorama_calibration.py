"""
Unit & Integration Tests for Option 1 Native-Resolution ROI Calibration,
Curved Panorama Thin-Plate Spline (panorama_tps) Homography, and
Setup/Live vs Diagnostics Calibration Synchronization.
"""

import json
import os
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from perception.homography import HomographyTransformer
from perception.calibration_templates import (
    compute_calibration_roi_crop,
    map_roi_click_to_native,
    render_pitch_projection_preview,
)
from visualization.overlays import BroadcastOverlayEngine
from ui.components.hardware_player import prepare_hardware_player_assets


def _synthetic_panorama_point(x_m: float, y_m: float, w: int = 4096, h: int = 1080):
    """
    Simulate a stitched dual-camera 4096x1080 wide panorama with barrel/bowing distortion
    along the horizontal touchlines (sidelines curve upward toward the wings).
    """
    u_norm = x_m / 105.0  # 0..1
    v_norm = y_m / 68.0   # 0..1

    # Non-linear perspective trapezoid + quadratic panoramic bow
    bow_px = -85.0 * ((2.0 * u_norm - 1.0) ** 2 - 1.0)  # +85px sag in center vs wings
    top_y = 260.0 + 0.45 * bow_px
    bot_y = 940.0 + bow_px
    v_px = top_y + v_norm * (bot_y - top_y)

    # Slight horizontal pincushion / perspective taper near far touchline
    x_margin_top = 420.0
    x_margin_bot = 140.0
    margin = x_margin_top + v_norm * (x_margin_bot - x_margin_top)
    u_px = margin + u_norm * (w - 2.0 * margin)
    return float(u_px), float(v_px)


class TestPanoramaCalibrationAndSync(unittest.TestCase):
    def test_01_panorama_tps_accuracy_beats_planar_and_roundtrips(self):
        """TPS warp fits curved 4096x1080 touchlines <1px and outperforms 3x3 planar."""
        pitch_anchors = [
            (0.0, 0.0), (52.5, 0.0), (105.0, 0.0),
            (0.0, 34.0), (52.5, 34.0), (105.0, 34.0),
            (0.0, 68.0), (52.5, 68.0), (105.0, 68.0),
            (16.5, 13.84), (88.5, 54.16),
        ]
        src_pts = [_synthetic_panorama_point(px, py) for px, py in pitch_anchors]

        tps_t = HomographyTransformer(
            4096, 1080,
            src_points=src_pts,
            dst_points=pitch_anchors,
            warp_mode="panorama_tps",
        )
        planar_t = HomographyTransformer(
            4096, 1080,
            src_points=src_pts,
            dst_points=pitch_anchors,
            warp_mode="planar",
        )

        # Verify forward & inverse anchor residuals in TPS mode
        rec_px = tps_t.pitch_to_image(pitch_anchors)
        max_px_err = max(
            np.hypot(rx - sx, ry - sy)
            for (rx, ry), (sx, sy) in zip(rec_px, src_pts)
        )
        self.assertLess(max_px_err, 1.0, f"TPS inverse anchor residual {max_px_err:.3f}px >= 1.0px")

        rec_m = tps_t.image_to_pitch(src_pts)
        max_m_err = max(
            np.hypot(rmx - px, rmy - py)
            for (rmx, rmy), (px, py) in zip(rec_m, pitch_anchors)
        )
        self.assertLess(max_m_err, 0.25, f"TPS forward anchor residual {max_m_err:.3f}m >= 0.25m")

        # Verify held-out curved midpoint accuracy (touchline quarter-points & center wings)
        held_out = [(26.25, 0.0), (78.75, 0.0), (26.25, 68.0), (78.75, 68.0), (26.25, 34.0)]
        held_src = [_synthetic_panorama_point(px, py) for px, py in held_out]

        tps_held_m = tps_t.image_to_pitch(held_src)
        planar_held_m = planar_t.image_to_pitch(held_src)

        tps_mean_err = np.mean([
            np.hypot(rx - px, ry - py) for (rx, ry), (px, py) in zip(tps_held_m, held_out)
        ])
        planar_mean_err = np.mean([
            np.hypot(rx - px, ry - py) for (rx, ry), (px, py) in zip(planar_held_m, held_out)
        ])
        self.assertLess(
            tps_mean_err, 0.75,
            f"TPS held-out error {tps_mean_err:.3f}m too large (planar={planar_mean_err:.3f}m)"
        )
        self.assertLess(
            tps_mean_err, planar_mean_err * 0.3,
            f"Expected TPS ({tps_mean_err:.3f}m) << planar ({planar_mean_err:.3f}m)"
        )

    def test_02_warp_to_pitch_plane_and_client_lut(self):
        """cv2.remap panoramic bird's-eye warp & client LUT bilinear interpolation work cleanly."""
        pitch_anchors = [
            (0.0, 0.0), (52.5, 0.0), (105.0, 0.0),
            (0.0, 68.0), (52.5, 68.0), (105.0, 68.0),
            (52.5, 34.0), (16.5, 34.0),
        ]
        src_pts = [_synthetic_panorama_point(px, py) for px, py in pitch_anchors]
        tps_t = HomographyTransformer(
            4096, 1080, src_points=src_pts, dst_points=pitch_anchors, warp_mode="panorama_tps"
        )

        fake_frame = np.zeros((1080, 4096, 3), dtype=np.uint8)
        fake_frame[200:1000, 100:3996] = (40, 140, 60)
        topdown = tps_t.warp_to_pitch_plane(fake_frame, width=1050, height=680)
        self.assertEqual(topdown.shape, (680, 1050, 3))
        self.assertGreater(int(np.count_nonzero(topdown)), 400000)

        meta = tps_t.export_client_projection(nx=36, ny=24)
        self.assertEqual(meta["warp_mode"], "panorama_tps")
        lut = meta["p2i_lut"]
        self.assertEqual(lut["nx"], 36)
        self.assertEqual(lut["ny"], 24)
        self.assertEqual(len(lut["u"]), 24)
        self.assertEqual(len(lut["u"][0]), 36)

        # Bilinear LUT interpolation at center (52.5, 34.0) vs exact TPS pitch_to_image
        fx = (52.5 / 105.0) * (lut["nx"] - 1)
        fy = (34.0 / 68.0) * (lut["ny"] - 1)
        ix = min(lut["nx"] - 2, max(0, int(fx)))
        iy = min(lut["ny"] - 2, max(0, int(fy)))
        tx, ty = fx - ix, fy - iy
        u_arr = np.array(lut["u"])
        v_arr = np.array(lut["v"])
        u_lut = (
            (1 - tx) * (1 - ty) * u_arr[iy, ix]
            + tx * (1 - ty) * u_arr[iy, ix + 1]
            + (1 - tx) * ty * u_arr[iy + 1, ix]
            + tx * ty * u_arr[iy + 1, ix + 1]
        )
        v_lut = (
            (1 - tx) * (1 - ty) * v_arr[iy, ix]
            + tx * (1 - ty) * v_arr[iy, ix + 1]
            + (1 - tx) * ty * v_arr[iy + 1, ix]
            + tx * ty * v_arr[iy + 1, ix + 1]
        )
        u_ref, v_ref = tps_t.pitch_to_image([(52.5, 34.0)])[0]
        self.assertLess(np.hypot(u_lut - u_ref, v_lut - v_ref), 3.0)

    def test_03_roi_crop_and_native_click_roundtrip(self):
        """ROI zoom presets map clicks back to native 4096x1080 coordinates accurately."""
        w, h = 4096, 1080
        for zone in ("full", "left", "center", "right", "custom"):
            rx0, ry0, rx1, ry1 = compute_calibration_roi_crop(
                w, h, zone_mode=zone, pan_x_pct=0.37, pan_y_pct=0.45, zoom_factor=2.5
            )
            self.assertGreaterEqual(rx0, 0)
            self.assertGreaterEqual(ry0, 0)
            self.assertLessEqual(rx1, w)
            self.assertLessEqual(ry1, h)
            self.assertGreater(rx1, rx0)
            self.assertGreater(ry1, ry0)

            crop_w = rx1 - rx0
            crop_h = ry1 - ry0
            # Simulate browser displaying crop in arbitrary scaled box (e.g. width=920 CSS px)
            elem_w = 920.0
            elem_h = elem_w * (crop_h / crop_w)
            target_nx = rx0 + 0.63 * crop_w
            target_ny = ry0 + 0.41 * crop_h

            click_x = 0.63 * elem_w
            click_y = 0.41 * elem_h
            mapped_x, mapped_y = map_roi_click_to_native(
                click_x, click_y, elem_w, elem_h, (rx0, ry0, rx1, ry1), w, h
            )
            self.assertAlmostEqual(mapped_x, target_nx, delta=1e-4)
            self.assertAlmostEqual(mapped_y, target_ny, delta=1e-4)

    def test_04_overlays_and_preview_render_with_tps_and_planar(self):
        """Overlay engine and calibration preview handle planar and panorama_tps with >4 anchors."""
        pitch_anchors = [
            (0.0, 0.0), (52.5, 0.0), (105.0, 0.0),
            (0.0, 68.0), (52.5, 68.0), (105.0, 68.0),
            (52.5, 34.0), (16.5, 34.0),
        ]
        src_pts = [_synthetic_panorama_point(px, py) for px, py in pitch_anchors]
        frame = np.zeros((1080, 4096, 3), dtype=np.uint8)

        for mode in ("planar", "panorama_tps"):
            t = HomographyTransformer(
                4096, 1080, src_points=src_pts, dst_points=pitch_anchors, warp_mode=mode
            )
            preview = render_pitch_projection_preview(frame, t)
            self.assertEqual(preview.shape, frame.shape)
            self.assertGreater(int(np.count_nonzero(preview)), 0)

            eng = BroadcastOverlayEngine(t)
            mask = eng.get_turf_binary_mask(frame.shape)
            self.assertEqual(mask.shape, (1080, 4096))
            self.assertGreater(int(np.count_nonzero(mask)), 0)

            out = eng.overlay_projected_hull(
                frame,
                {"polygon": [(20.0, 15.0), (80.0, 15.0), (80.0, 55.0), (20.0, 55.0)], "area": 2400.0},
            )
            self.assertEqual(out.shape, frame.shape)

            from types import SimpleNamespace
            dbg = eng.render_homography_debug_frame(
                frame,
                positions=SimpleNamespace(team_a=[(52.5, 34.0)], team_b=[(40.0, 20.0)]),
                show_player_pins=True,
            )
            self.assertEqual(dbg.shape, frame.shape)

    def test_05_signature_and_hardware_player_cache_sync(self):
        """Calibration signature changes across dimensions/modes and rejects stale custom caches."""
        t1 = HomographyTransformer(4096, 1080)
        sig1 = t1.get_calibration_signature()

        t1.update_dimensions(1920, 1080)
        sig2 = t1.get_calibration_signature()
        self.assertNotEqual(sig1, sig2)

        pts_p = [(0.0, 0.0), (105.0, 0.0), (105.0, 68.0), (0.0, 68.0), (52.5, 34.0), (52.5, 0.0)]
        pts_i = [_synthetic_panorama_point(x, y, 1920, 1080) for x, y in pts_p]
        t1.calibrate(pts_i, pts_p, frame_width=1920, frame_height=1080, warp_mode="panorama_tps")
        sig3 = t1.get_calibration_signature()
        self.assertNotEqual(sig2, sig3)

        # Verify prepare_hardware_player_assets rejects stale cache lacking matching calibration_sig
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
            json.dump({
                "fps": 30.0,
                "total_frames": 10,
                "stride": 1,
                "video_width": 1920,
                "video_height": 1080,
                "calibration_sig": "stale_old_sig",
                "frames": [
                    {
                        "frame_idx": 0,
                        "team_a": [[50.0, 30.0], [60.0, 40.0]],
                        "team_b": [[40.0, 30.0]],
                        "carrier_idx": 0,
                        "ball": [50.0, 30.0],
                        "dijkstra": {"path": [0, 1], "graph_edges": []},
                    }
                ],
            }, tf)
            cache_path = tf.name

        try:
            meta = prepare_hardware_player_assets("data/sample_broadcast.mp4", cache_path, t1)
            self.assertFalse(meta["has_cache"], "Stale cache with mismatched sig must disable overlays")
            self.assertEqual(meta["warp_mode"], "panorama_tps")
            self.assertIsNotNone(meta["p2i_lut"])
            self.assertEqual(t1.frame_width, 1920)
            self.assertEqual(t1.frame_height, 1080)
        finally:
            if os.path.exists(cache_path):
                os.remove(cache_path)


if __name__ == "__main__":
    unittest.main()
