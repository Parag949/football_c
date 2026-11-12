"""
Streamlit AppTest Integration Suite
===================================
Executes real Streamlit script runs (`app.py` -> Setup Gate -> Calibration Studio
-> Hardware Player / Workstation) inside `streamlit.testing.v1.AppTest` so widget
type mismatches, session-state key errors, and HTML template errors fail in CI
rather than in the user's browser.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app.py"))


class TestStreamlitUIAppTest(unittest.TestCase):
    def _assert_clean(self, at: AppTest, ctx: str) -> None:
        if at.exception:
            msgs = [str(e.value) for e in at.exception]
            self.fail(f"[{ctx}] Uncaught Streamlit exception(s): {msgs}")

    def test_01_setup_gate_boots_cleanly(self):
        """App boots to Setup Gate with zero uncaught widget or session-state exceptions."""
        at = AppTest.from_file(APP_PATH, default_timeout=20)
        at.run()
        self._assert_clean(at, "initial boot")
        self.assertEqual(at.session_state["active_source_key"], "premier_league")
        self.assertFalse(at.session_state["setup_confirmed"])

    def test_02_custom_upload_library_selection_preserves_builtin_sources(self):
        """Switching to Custom Match Video lists saved library videos cleanly without errors."""
        at = AppTest.from_file(APP_PATH, default_timeout=20)
        at.run()
        self._assert_clean(at, "boot for custom")

        src_radio = next(r for r in at.radio if r.key == "setup_source_radio")
        custom_opt = next(o for o in src_radio.options if "Custom Match Video" in str(o))
        src_radio.set_value(custom_opt).run()
        self._assert_clean(at, "switch to custom_upload")
        self.assertEqual(at.session_state["active_source_key"], "custom_upload")

        # Saved library selectbox should appear if data/videos or data/custom_videos has footage
        from ui.media_library import list_available_custom_videos
        lib_boxes = [sb for sb in at.selectbox if sb.key == "setup_custom_library_select"]
        avail = list_available_custom_videos()
        if lib_boxes:
            self.assertGreater(len(lib_boxes[0].options), 0)
            if len(avail) > 1:
                lib_boxes[0].set_value(avail[1]).run()
                self._assert_clean(at, "switch library video")

        self.assertTrue(os.path.exists("data/user_match.mp4"))
        self.assertTrue(os.path.exists("data/precomputed_user.json"))

    def test_03_calibration_studio_coerces_string_state_and_runs_panorama_modes(self):
        """Calibration Studio handles legacy str anchor counts, Panorama TPS, presets, and ROI zoom."""
        at = AppTest.from_file(APP_PATH, default_timeout=25)
        at.run()
        self._assert_clean(at, "boot for calib")

        # Seed dirty string state like a stale browser session before opening Calibration Studio
        at.session_state["tv_calib_num_anchors"] = "4"
        at.session_state["gate_calib_num_anchors_widget"] = "4"

        calib_radio = next(r for r in at.radio if r.key == "setup_calib_type_radio")
        manual_opt = next(o for o in calib_radio.options if "Interactive" in str(o))
        calib_radio.set_value(manual_opt).run()
        self._assert_clean(at, "open calibration studio with dirty str anchor state")
        self.assertIsInstance(at.session_state["tv_calib_num_anchors"], int)

        # Switch to Curved Panorama Mode
        warp_radio = next(r for r in at.radio if r.key == "gate_calib_warp_mode_radio")
        tps_opt = next(o for o in warp_radio.options if "Curved Panorama" in str(o))
        warp_radio.set_value(tps_opt).run()
        self._assert_clean(at, "enable panorama_tps")
        self.assertEqual(at.session_state["tv_calib_warp_mode"], "panorama_tps")
        self.assertGreaterEqual(at.session_state["tv_calib_num_anchors"], 6)

        # Trigger Panorama 8-Pt Spread button
        pano_btn = next(b for b in at.button if b.key == "gate_calib_q_pano8")
        pano_btn.click().run()
        self._assert_clean(at, "click panorama 8pt preset")
        self.assertEqual(at.session_state["tv_calib_num_anchors"], 8)

        # Exercise ROI Zoom radio zones (Left, Center, Right, Custom Pan & Zoom)
        roi_radio = next(r for r in at.radio if r.key == "gate_calib_roi_zone_radio")
        for opt in roi_radio.options:
            roi_radio = next(r for r in at.radio if r.key == "gate_calib_roi_zone_radio")
            roi_radio.set_value(opt).run()
            self._assert_clean(at, f"roi zoom {opt}")

    def test_04_launch_workstation_renders_live_player_cleanly(self):
        """Confirming setup launches into Workstation mode and renders Hardware Player cleanly."""
        at = AppTest.from_file(APP_PATH, default_timeout=25)
        at.run()
        self._assert_clean(at, "boot for launch")

        launch_btn = next(b for b in at.button if "Confirm Configuration" in b.label)
        launch_btn.click().run()
        self._assert_clean(at, "launch workstation")
        self.assertTrue(at.session_state["setup_confirmed"])


if __name__ == "__main__":
    unittest.main()
