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
        manual_opt = next(o for o in calib_radio.options if "Interactive Pitch Mask" in str(o))
        calib_radio.set_value(manual_opt).run()
        self._assert_clean(at, "open calibration studio with dirty str anchor state")
        self.assertIsInstance(at.session_state["tv_calib_num_anchors"], int)

        # Switch to Universal Camera Mode
        warp_radio = next(r for r in at.radio if r.key == "gate_calib_warp_mode_radio")
        opt = next(o for o in warp_radio.options if "Universal Optical" in str(o) or "Wide / Panoramic" in str(o))
        warp_radio.set_value(opt).run()
        self._assert_clean(at, "enable universal lens mode")
        self.assertIn(at.session_state["tv_calib_warp_mode"], ("universal", "panorama_tps"))

        # Trigger Panorama 8-Pt Spread button
        pano_btn = next(b for b in at.button if b.key == "gate_calib_q_pano8")
        pano_btn.click().run()
        self._assert_clean(at, "click panorama 8pt preset")
        self.assertEqual(at.session_state["tv_calib_num_anchors"], 8)

        # Exercise ROI Zoom radio zones (Left, Center, Right, Custom Zoom)
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

        launch_btn = next(b for b in at.button if "Open Match Workspace" in b.label)
        launch_btn.click().run()
        self._assert_clean(at, "launch workstation")
        self.assertTrue(at.session_state["setup_confirmed"])

    def test_05_workstation_inspector_and_tactics_board(self):
        """Workstation Live Player, Control Deck, all 5 Inspector stages, and 2D Tactics Board run cleanly."""
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        self._assert_clean(at, "boot workstation suite")

        launch_btn = next(b for b in at.button if "Open Match Workspace" in b.label)
        launch_btn.click().run()
        self._assert_clean(at, "enter workstation")

        # Switch to Frame Inspector & Alignment via segmented_control and exercise Control Deck + all 5 tabs
        wm_ctrl = next(sc for sc in at.segmented_control if sc.key == "tv_workstation_mode")
        wm_ctrl.set_value("🔍 Frame Inspector & Alignment").run()
        self._assert_clean(at, "enter frame inspector")

        step_p1 = [b for b in at.button if b.key == "step_p1"]
        if step_p1:
            step_p1[0].click().run()
            self._assert_clean(at, "step_p1")

        # Exercise Pitch Alignment tab and all 4 alignment sub-views (including Pitch Mask Studio)
        stage_ctrl = next(sc for sc in at.segmented_control if sc.key == "active_stage_tab")
        stage_ctrl.set_value("📐 Pitch Alignment").run()
        self._assert_clean(at, "inspector tab Pitch Alignment")

        s2_radio = next(r for r in at.radio if any("Edit Pitch Alignment" in str(o) for o in r.options))
        for opt in s2_radio.options:
            s2_radio = next(r for r in at.radio if any("Edit Pitch Alignment" in str(o) for o in r.options))
            s2_radio.set_value(opt).run()
            self._assert_clean(at, f"inspector stage2 {opt}")

        # Exercise remaining Inspector tabs (Player Detection, Tactical Metrics, Performance)
        for tab_val in ["👁️ Player Detection", "📊 Tactical Metrics", "⏱️ Performance"]:
            stage_ctrl = next(sc for sc in at.segmented_control if sc.key == "active_stage_tab")
            stage_ctrl.set_value(tab_val).run()
            self._assert_clean(at, f"inspector tab {tab_val}")

        bench_btn = [b for b in at.button if b.key == "btn_run_bench"]
        if bench_btn:
            bench_btn[0].click().run()
            self._assert_clean(at, "inspector stage5 benchmark")

        # Switch to 2D Tactics Board via segmented_control
        wm_ctrl = next(sc for sc in at.segmented_control if sc.key == "tv_workstation_mode")
        wm_ctrl.set_value("📋 2D Tactics Board").run()
        self._assert_clean(at, "enter 2D tactics board")

    def test_06_per_video_calibration_isolation(self):
        """Per-video saved calibrations isolate by video path and resolution."""
        from ui.media_library import (
            save_video_calibration,
            load_video_calibration,
            delete_video_calibration,
        )

        dummy_path = "data/custom_videos/__test_isolation_dummy__.mp4"
        try:
            payload = {
                "video_path": dummy_path,
                "frame_width": 4096,
                "frame_height": 1080,
                "warp_mode": "panorama_tps",
                "src_points": [[200, 300], [3800, 300], [3900, 900], [100, 900], [2048, 250], [2048, 850]],
                "dst_points": [[0, 0], [105, 0], [105, 68], [0, 68], [52.5, 0], [52.5, 68]],
            }
            save_video_calibration(dummy_path, payload)
            loaded_ok = load_video_calibration(dummy_path, 4096, 1080)
            self.assertIsNotNone(loaded_ok)
            self.assertEqual(loaded_ok["warp_mode"], "panorama_tps")

            # Mismatched video dimensions must not apply stale coordinates
            loaded_wrong_res = load_video_calibration(dummy_path, 1920, 1080)
            self.assertIsNone(loaded_wrong_res)

            # Different video key must not inherit dummy calibration
            loaded_other = load_video_calibration("data/user_match.mp4", 4096, 1080)
            if loaded_other is not None:
                self.assertNotEqual(loaded_other.get("video_path"), dummy_path)
        finally:
            delete_video_calibration(dummy_path)
            self.assertIsNone(load_video_calibration(dummy_path, 4096, 1080))


if __name__ == "__main__":
    unittest.main()
