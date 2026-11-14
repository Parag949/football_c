import sys
sys.path.insert(0, ".")
import unittest
import numpy as np
import streamlit as st

from ui.state import init_session_state, sync_subsystems_transformer, execute_pipeline, sync_video_source
from perception.homography import HomographyTransformer
from core.contracts import PlayerDetection, FrameDetections, PitchPositions

class TestCalibrationPipelineSync(unittest.TestCase):
    def setUp(self):
        st.session_state.clear()
        init_session_state()
        sync_video_source()

    def test_subsystem_synchronization(self):
        t = HomographyTransformer(1920, 1080, camera_mode="sideline")
        src = np.float32([[200, 200], [1700, 200], [1800, 900], [100, 900]])
        dst = np.float32([[0, 0], [105, 0], [105, 68], [0, 68]])
        t.calibrate(src, dst, frame_width=1920, frame_height=1080, warp_mode="planar")

        st.session_state.transformer = t
        # Single source of truth: property dynamically resolves to canonical transformer immediately
        self.assertIs(st.session_state.overlay_engine.transformer, t)
        self.assertIs(st.session_state.stream_manager.transformer, t)

        # Verification check passes
        sync_subsystems_transformer()

        self.assertIs(st.session_state.overlay_engine.transformer, t)
        self.assertIs(st.session_state.stream_manager.transformer, t)
        self.assertTrue(st.session_state.transformer.is_custom_calibrated)

    def test_custom_calib_uses_cached_raw_detections_in_pipeline(self):
        st.session_state.manual_calib_enabled = True
        st.session_state.transformer.is_custom_calibrated = True

        bundle = execute_pipeline(0, force_live=False)
        self.assertIsNotNone(bundle)
        self.assertIn('positions', bundle)
        self.assertIn('detections', bundle)
        self.assertTrue(bundle.get('is_cached', False))
        self.assertGreater(len(bundle['detections'].players), 0)
        self.assertIsNotNone(bundle['detections'].players[0].foot_point)

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

    def test_stage4_render_overlay_respects_explicit_transformer(self):
        from core.pipeline import stage4_render_overlay
        from core.contracts import TacticalGeometry

        t_session = HomographyTransformer(1920, 1080, camera_mode="sideline")
        t_explicit = HomographyTransformer(1920, 1080, camera_mode="goal_to_goal")
        st.session_state.transformer = t_session

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        pos = PitchPositions(frame_idx=0, team_a=[(20.0, 30.0)], team_b=[(50.0, 40.0)], ball=(52.5, 34.0))
        geom = TacticalGeometry(
            frame_idx=0,
            hull_polygon=[(20.0, 30.0), (30.0, 40.0), (25.0, 50.0)],
            voronoi_cells_a=[{'polygon': [(10.0, 10.0), (20.0, 10.0), (20.0, 20.0)]}],
            voronoi_cells_b=[]
        )

        calls_session = []
        calls_explicit = []

        orig_proj_session = t_session.pitch_to_image
        orig_proj_explicit = t_explicit.pitch_to_image

        def spy_session(*args, **kwargs):
            calls_session.append(True)
            return orig_proj_session(*args, **kwargs)

        def spy_explicit(*args, **kwargs):
            calls_explicit.append(True)
            return orig_proj_explicit(*args, **kwargs)

        t_session.pitch_to_image = spy_session
        t_explicit.pitch_to_image = spy_explicit

        # When transformer=t_explicit is passed, stage4_render_overlay must use t_explicit
        res_explicit = stage4_render_overlay(
            frame, pos, geom, st.session_state.overlay_engine,
            show_voronoi=True, show_hull=True,
            transformer=t_explicit
        )
        self.assertEqual(res_explicit.shape, (1080, 1920, 3))
        self.assertTrue(len(calls_explicit) > 0, "stage4_render_overlay did not invoke explicit transformer!")
        self.assertEqual(len(calls_session), 0, "stage4_render_overlay mistakenly invoked session transformer instead of explicit transformer!")

        # When transformer=None, stage4_render_overlay falls back to overlay_engine.transformer (t_session)
        calls_explicit.clear()
        calls_session.clear()
        res_fallback = stage4_render_overlay(
            frame, pos, geom, st.session_state.overlay_engine,
            show_voronoi=True, show_hull=True,
            transformer=None
        )
        self.assertEqual(res_fallback.shape, (1080, 1920, 3))
        self.assertTrue(len(calls_session) > 0, "stage4_render_overlay fallback did not invoke overlay_engine transformer!")
        self.assertEqual(len(calls_explicit), 0)

if __name__ == "__main__":
    unittest.main()
