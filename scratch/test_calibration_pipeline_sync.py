import sys
sys.path.insert(0, ".")
import unittest
import numpy as np
import streamlit as st

from ui.state import init_session_state, sync_subsystems_transformer, execute_pipeline
from perception.homography import HomographyTransformer
from core.contracts import PlayerDetection, FrameDetections, PitchPositions

class TestCalibrationPipelineSync(unittest.TestCase):
    def setUp(self):
        init_session_state()

    def test_subsystem_synchronization(self):
        t = HomographyTransformer(1920, 1080, camera_mode="sideline")
        src = np.float32([[200, 200], [1700, 200], [1800, 900], [100, 900]])
        dst = np.float32([[0, 0], [105, 0], [105, 68], [0, 68]])
        t.calibrate(src, dst, frame_width=1920, frame_height=1080, warp_mode="planar")

        st.session_state.transformer = t
        sync_subsystems_transformer()

        self.assertIs(st.session_state.overlay_engine.transformer, t)
        self.assertIs(st.session_state.stream_manager.transformer, t)
        self.assertTrue(st.session_state.transformer.is_custom_calibrated)

    def test_custom_calib_bypasses_cache_in_pipeline(self):
        st.session_state.manual_calib_enabled = True
        st.session_state.transformer.is_custom_calibrated = True

        bundle = execute_pipeline(0, force_live=False)
        self.assertIsNotNone(bundle)
        self.assertIn('positions', bundle)
        self.assertIn('detections', bundle)

    def test_pin_to_foot_alignment(self):
        t = HomographyTransformer(1920, 1080, camera_mode="sideline")
        st.session_state.transformer = t
        sync_subsystems_transformer()

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        players = [
            PlayerDetection(box=(500, 400, 560, 520), team_id=0),
            PlayerDetection(box=(800, 600, 850, 720), team_id=1),
        ]
        pos = PitchPositions(frame_idx=0, team_a=[(20.0, 30.0)], team_b=[(50.0, 40.0)], ball=(52.5, 34.0))

        dbg = st.session_state.overlay_engine.render_homography_debug_frame(
            frame,
            show_player_pins=True,
            player_detections=players,
            positions=pos
        )
        self.assertEqual(dbg.shape, (1080, 1920, 3))

if __name__ == "__main__":
    unittest.main()
