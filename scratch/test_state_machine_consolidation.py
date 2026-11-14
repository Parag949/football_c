"""
Regression Test Suite: State Machine Consolidation (Requirement R4 / Milestone M4)
----------------------------------------------------------------------------------
Verifies:
1. AppState and typed sub-states initialization, strong typing, and default contracts.
2. Bidirectional state bridge synchronization (loose session keys <-> AppState).
3. Elimination of duplicate calibration split-brain (manual_calib_* vs tv_calib_*).
4. reset_widget_state helper functionality and error tolerance.
5. Dynamic geometry recalculation when switching camera presets (active_cam != cache_cam).
6. get_app_state idempotency across full session lifecycle.
"""

import sys
import os
import unittest
import numpy as np
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ui.state_models import (
    AppState,
    VideoSourceState,
    PlaybackState,
    CalibrationState,
    AnalyticsDisplayState,
    RulerState,
    coerce_anchor_count,
)
from ui.state import (
    init_session_state,
    sync_video_source,
    execute_pipeline,
    reset_widget_state,
    get_app_state,
)
from perception.homography import HomographyTransformer
from core.contracts import TacticalGeometry


class TestStateMachineConsolidation(unittest.TestCase):
    def setUp(self):
        st.session_state.clear()
        init_session_state()

    def test_01_app_state_and_substates_initialization_and_typing(self):
        """Verifies AppState and its sub-dataclasses initialize with clean typed defaults."""
        app = AppState()
        self.assertIsInstance(app.video, VideoSourceState)
        self.assertIsInstance(app.playback, PlaybackState)
        self.assertIsInstance(app.calibration, CalibrationState)
        self.assertIsInstance(app.display, AnalyticsDisplayState)
        self.assertIsInstance(app.ruler, RulerState)

        # VideoSourceState defaults
        self.assertEqual(app.video.active_key, "world_cup")
        self.assertEqual(app.video.custom_video_path, "")

        # PlaybackState defaults & methods
        self.assertEqual(app.playback.current_frame, 0)
        self.assertFalse(app.playback.is_playing)
        self.assertEqual(app.playback.speed, 1.0)
        self.assertTrue(app.playback.loop)
        self.assertEqual(app.playback.step(5, total_frames=100), 5)
        self.assertEqual(app.playback.step(-2, total_frames=100), 3)
        self.assertEqual(app.playback.seek(42, total_frames=100), 42)
        self.assertTrue(app.playback.toggle_play(total_frames=100))
        self.assertFalse(app.playback.toggle_play(total_frames=100))

        # CalibrationState defaults & methods
        self.assertFalse(app.calibration.is_custom)
        self.assertEqual(app.calibration.points, [])
        self.assertEqual(app.calibration.warp_mode, "planar")
        self.assertEqual(app.calibration.picker_mode, "custom")
        app.calibration.apply_points([(10.0, 20.0), (30.0, 40.0)], warp="universal")
        self.assertTrue(app.calibration.is_custom)
        self.assertEqual(len(app.calibration.points), 2)
        self.assertEqual(app.calibration.warp_mode, "universal")
        self.assertTrue(app.calibration.just_applied)
        app.calibration.reset()
        self.assertFalse(app.calibration.is_custom)
        self.assertEqual(app.calibration.points, [])
        self.assertTrue(app.calibration.just_reset)

        # AnalyticsDisplayState defaults
        self.assertTrue(app.display.show_hull)
        self.assertTrue(app.display.show_voronoi)
        self.assertTrue(app.display.show_dijkstra)
        self.assertFalse(app.display.show_passing_graph)
        self.assertFalse(app.display.show_ruler)
        self.assertEqual(app.display.hull_algo, "graham")

        # RulerState defaults
        self.assertEqual(app.ruler.mode, "players")
        self.assertEqual(app.ruler.p1_idx, 0)
        self.assertEqual(app.ruler.p2_idx, 1)
        self.assertTrue(app.ruler.auto_offside)

        # Serialization to dict
        d = app.to_dict()
        self.assertIsInstance(d, dict)
        self.assertIn("video", d)
        self.assertIn("playback", d)
        self.assertIn("calibration", d)

    def test_02_bidirectional_bridge_session_to_app_state(self):
        """Writing loose session_state variables immediately updates AppState."""
        app = st.session_state.app_state
        self.assertIsInstance(app, AppState)

        # Playback sync
        st.session_state.current_frame = 37
        self.assertEqual(app.playback.current_frame, 37)
        self.assertEqual(app.playback.main_slider_val, 37)
        self.assertEqual(st.session_state.main_slider_val, 37)

        st.session_state.is_playing = True
        self.assertTrue(app.playback.is_playing)

        st.session_state.playback_speed = 2.0
        self.assertEqual(app.playback.speed, 2.0)

        # Display sync
        st.session_state.show_hull = False
        self.assertFalse(app.display.show_hull)

        st.session_state.show_voronoi = False
        self.assertFalse(app.display.show_voronoi)

        st.session_state.ui_hull_algo_val = "jarvis"
        self.assertEqual(app.display.hull_algo, "jarvis")

        # Ruler sync
        st.session_state.ruler_mode = "inter_team"
        self.assertEqual(app.ruler.mode, "inter_team")

        st.session_state.ruler_p1_idx = 4
        self.assertEqual(app.ruler.p1_idx, 4)

        # Video source & workflow sync
        st.session_state.active_source_key = "custom_upload"
        self.assertEqual(app.video.active_key, "custom_upload")

        st.session_state.setup_confirmed = True
        self.assertTrue(app.setup_confirmed)

    def test_03_bidirectional_bridge_app_state_to_session(self):
        """Mutating AppState directly immediately writes through to loose session_state keys."""
        app = st.session_state.app_state

        # Mutate playback in AppState
        app.playback.current_frame = 84
        self.assertEqual(st.session_state.current_frame, 84)
        self.assertEqual(st.session_state["current_frame"], 84)
        self.assertEqual(st.session_state.main_slider_val, 84)

        app.playback.is_playing = True
        self.assertTrue(st.session_state.is_playing)

        # Mutate display in AppState
        app.display.show_ruler = True
        self.assertTrue(st.session_state.show_ruler)
        self.assertTrue(st.session_state["show_ruler"])

        app.display.hull_algo = "jarvis"
        self.assertEqual(st.session_state.ui_hull_algo_val, "jarvis")

        # Mutate workflow in AppState
        app.setup_confirmed = True
        self.assertTrue(st.session_state.setup_confirmed)

        app.video.custom_video_path = "data/custom_videos/match1.mp4"
        self.assertEqual(st.session_state.custom_video_path, "data/custom_videos/match1.mp4")

    def test_04_calibration_split_brain_elimination(self):
        """Eliminates dual-key split-brain: manual_calib_* and tv_calib_* collapse onto CalibrationState."""
        app = st.session_state.app_state

        # Setting manual_calib_enabled sets all 4 calibration active flags
        st.session_state.manual_calib_enabled = True
        self.assertTrue(st.session_state.tv_calib_enabled)
        self.assertTrue(st.session_state.manual_calib_applied)
        self.assertTrue(st.session_state.tv_calib_applied)
        self.assertTrue(app.calibration.is_custom)

        # Clearing via app_state clears all 4 keys
        app.calibration.is_custom = False
        self.assertFalse(st.session_state.manual_calib_enabled)
        self.assertFalse(st.session_state.tv_calib_enabled)
        self.assertFalse(st.session_state.manual_calib_applied)
        self.assertFalse(st.session_state.tv_calib_applied)

        # Point synchronization between manual and tv keys
        sample_pts = [(100.0, 150.0), (800.0, 150.0), (900.0, 700.0), (50.0, 700.0)]
        st.session_state.tv_calib_points = sample_pts
        self.assertEqual(st.session_state.manual_calib_points, sample_pts)
        self.assertEqual(app.calibration.points, sample_pts)

        # Anchor count dirty string coercion
        st.session_state["tv_calib_num_anchors"] = "8 Points"
        self.assertEqual(st.session_state.tv_calib_num_anchors, 8)
        self.assertIsInstance(st.session_state.tv_calib_num_anchors, int)
        self.assertEqual(app.calibration.num_anchors, 8)

        st.session_state["tv_calib_num_anchors"] = "4"
        self.assertEqual(st.session_state.tv_calib_num_anchors, 4)
        self.assertIsInstance(st.session_state.tv_calib_num_anchors, int)

    def test_05_reset_widget_state_helper(self):
        """Verifies reset_widget_state cleanly pops target widget keys without raising exceptions."""
        st.session_state["setup_calib_type_radio"] = "🎯 Interactive Pitch Mask Alignment"
        st.session_state["setup_preset_select"] = "Sideline TV Broadcast"
        st.session_state["setup_exec_mode_radio"] = "⚡ Instant Replay"
        st.session_state["persistent_flag"] = "keep_me"
        st.session_state["valid_key"] = "to_be_deleted"

        # Edge-case hardening: dirty widget IDs ($$ID-...), empty strings, None, and nonexistent keys
        try:
            reset_widget_state(
                "",
                "$$ID-dirty-widget-123",
                None,
                "setup_calib_type_radio",
                "setup_preset_select",
                "setup_exec_mode_radio",
                "valid_key",
                "non_existent_widget_key"  # Must tolerate missing keys safely
            )
        except Exception as e:
            self.fail(f"reset_widget_state raised an unexpected exception: {e}")

        self.assertNotIn("setup_calib_type_radio", st.session_state)
        self.assertNotIn("setup_preset_select", st.session_state)
        self.assertNotIn("setup_exec_mode_radio", st.session_state)
        self.assertNotIn("valid_key", st.session_state)
        self.assertIn("persistent_flag", st.session_state)
        self.assertEqual(st.session_state["persistent_flag"], "keep_me")

    def test_06_dynamic_geometry_recalculation_on_camera_preset_switch(self):
        """
        Verifies dynamic tactical geometry recalculation when switching camera mode presets
        (active_cam != cache_cam), preventing stale sideline hull/voronoi in endzone/panorama modes.
        """
        sync_video_source()
        self.assertIsNotNone(st.session_state.precomputed_data)

        # Baseline: sideline mode (active_cam == "sideline", cache_cam == "sideline")
        st.session_state.transformer.set_camera_mode("sideline")
        bundle_side = execute_pipeline(0)
        self.assertIsNotNone(bundle_side)
        geom_side = bundle_side["geometry"]
        self.assertIsInstance(geom_side, TacticalGeometry)

        # Switch camera mode preset to "goal_to_goal" (active_cam != cache_cam)
        st.session_state.transformer.set_camera_mode("goal_to_goal")
        self.assertEqual(st.session_state.transformer.camera_mode, "goal_to_goal")

        bundle_endzone = execute_pipeline(0)
        self.assertIsNotNone(bundle_endzone)
        geom_endzone = bundle_endzone["geometry"]
        self.assertIsInstance(geom_endzone, TacticalGeometry)

        # In goal_to_goal camera mode, projected player coordinates shift significantly
        # Dynamic geometry recalculation must compute new hull polygon and voronoi
        pos_side = bundle_side["positions"]
        pos_endzone = bundle_endzone["positions"]
        self.assertNotEqual(pos_side.team_a[0], pos_endzone.team_a[0], "Projections should differ across camera modes")

        # Dynamic geometry must reflect new player coordinates, not stale sideline polygon
        self.assertNotEqual(geom_side.hull_polygon, geom_endzone.hull_polygon,
                            "Tactical geometry must dynamically recompute when active_cam != cache_cam")

        # Verify custom calibration dynamic recalculation
        src_custom = np.float32([[200, 200], [1700, 200], [1800, 900], [100, 900]])
        dst_custom = np.float32([[0, 0], [105, 0], [105, 68], [0, 68]])
        st.session_state.transformer.calibrate(src_custom, dst_custom, warp_mode="planar")
        st.session_state.manual_calib_enabled = True
        bundle_custom = execute_pipeline(0)
        geom_custom = bundle_custom["geometry"]
        self.assertNotEqual(geom_custom.hull_polygon, geom_side.hull_polygon)
        self.assertNotEqual(geom_custom.hull_polygon, geom_endzone.hull_polygon)

    def test_07_get_app_state_idempotence(self):
        """Verifies get_app_state provides reliable idempotent access across full lifecycle."""
        app1 = get_app_state()
        self.assertIsInstance(app1, AppState)

        app2 = get_app_state()
        self.assertIs(app1, app2)

        # Clear session state and verify self-healing initialization
        st.session_state.clear()
        app3 = get_app_state()
        self.assertIsInstance(app3, AppState)
        self.assertIn("app_state", st.session_state)


if __name__ == "__main__":
    res = unittest.main(exit=False)
    sys.exit(0 if res.result.wasSuccessful() else 1)
