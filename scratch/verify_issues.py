"""
Verification script for all 12 reported issues.
Demonstrates and confirms the existence of each issue prior to fixing.
"""
import sys
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import cv2

def verify_issue_1():
    print("--- [Issue 1] Camera Angle Mismatch Health Check ---")
    from core.contracts import FrameDetections, PitchPositions, TacticalGeometry
    from core.pipeline import evaluate_pipeline_health
    
    dets = FrameDetections(frame_idx=0, players=[])
    pos = PitchPositions(frame_idx=0, team_a=[], team_b=[])
    geom = TacticalGeometry(frame_idx=0)
    
    # ui/state.py passes st.session_state.active_source_key: "premier_league" and "goal_to_goal"
    status, issues = evaluate_pipeline_health(
        detections=dets,
        positions=pos,
        geometry=geom,
        match_source="premier_league",
        camera_mode="goal_to_goal"
    )
    # If the bug exists, status['stage2'][0] will NOT be "FAIL" (it will be "PASS")
    # and no camera mismatch issue will be in issues.
    has_mismatch_alert = any("Camera mode 'goal_to_goal' selected on a Sideline broadcast" in iss[1] for iss in issues)
    print(f"  Passed match_source='premier_league' with camera_mode='goal_to_goal'.")
    print(f"  Stage 2 Status: {status.get('stage2')}")
    print(f"  Camera mismatch alert detected: {has_mismatch_alert}")
    assert not has_mismatch_alert, "Expected failure to detect mismatch due to case/key difference!"
    print("  -> CONFIRMED: Bug exists. The mismatch alert was silently skipped.\n")

def verify_issue_2():
    print("--- [Issue 2] Unhandled Qhull Error in Voronoi Tessellation ---")
    from scipy.spatial import Voronoi
    from scipy.spatial._qhull import QhullError
    import algorithms.voronoi_dominance as vd
    
    # 3 collinear points
    collinear_pts = [(10.0, 10.0), (20.0, 20.0), (30.0, 30.0)]
    try:
        # scipy Voronoi directly on collinear points throws QhullError
        Voronoi(np.array(collinear_pts))
        print("  Voronoi did not raise QhullError directly.")
    except QhullError as qe:
        print(f"  scipy.spatial.Voronoi raised QhullError: {type(qe).__name__}")
        
    # Check if compute_bounded_voronoi has try/except
    import inspect
    src = inspect.getsource(vd.compute_bounded_voronoi)
    has_try_except = "except" in src and "Voronoi" in src
    print(f"  compute_bounded_voronoi has try/except around Voronoi: {has_try_except}")
    assert not has_try_except, "Expected Voronoi call to lack try/except protection."
    print("  -> CONFIRMED: Bug exists. Voronoi call is unprotected against QhullError.\n")

def verify_issue_3():
    print("--- [Issue 3] test_landmark_anchor.py import crash ---")
    import subprocess
    cmd = [sys.executable, os.path.join(REPO_ROOT, "scratch/test_landmark_anchor.py")]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(f"  Exit code: {res.returncode}")
    print(f"  Stderr: {res.stderr.strip()[:100]}...")
    assert res.returncode != 0 and "ModuleNotFoundError" in res.stderr
    print("  -> CONFIRMED: Bug exists. test_landmark_anchor.py crashes on execution.\n")

def verify_issue_4():
    print("--- [Issue 4] audit_suite.py missing precomputed cache crash ---")
    import subprocess
    cmd = [sys.executable, os.path.join(REPO_ROOT, "scratch/audit_suite.py")]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(f"  Exit code: {res.returncode}")
    print(f"  Stderr / Output contains FileNotFoundError: {'FileNotFoundError' in res.stderr or 'FileNotFoundError' in res.stdout}")
    assert res.returncode != 0
    print("  -> CONFIRMED: Bug exists. audit_suite.py crashes when cache files are missing.\n")

def verify_issue_5():
    print("--- [Issue 5] Timeline Slider Visual Desynchronization ---")
    import inspect
    from ui.components import control_deck
    src = inspect.getsource(control_deck.render_control_deck)
    has_sync = "st.session_state.main_slider_val =" in src or "main_slider_val" in src.split("def step_frame")[1].split("with b1:")[0]
    print(f"  step_frame updates st.session_state.main_slider_val: {has_sync}")
    assert not has_sync, "Expected step_frame to miss main_slider_val update."
    print("  -> CONFIRMED: Bug exists. Stepper buttons do not synchronize main_slider_val.\n")

def verify_issue_6():
    print("--- [Issue 6] Custom Mouse Calibration Discarded in Precompute ---")
    import inspect
    from perception import precompute
    sig = inspect.signature(precompute.precompute_video_segment)
    has_custom_params = "src_points" in sig.parameters or "transformer" in sig.parameters
    print(f"  precompute_video_segment accepts custom calibration points/transformer: {has_custom_params}")
    assert not has_custom_params, "Expected precompute_video_segment to lack custom calibration params."
    print("  -> CONFIRMED: Bug exists. Precompute pipeline drops manual homography.\n")

def verify_issue_7():
    print("--- [Issue 7] Missing Defending Player Attribution in Cached Playback ---")
    import inspect
    from perception import precompute
    from ui import state
    src_precompute = inspect.getsource(precompute._worker_compute_daa_batch)
    saves_indices = "'player_indices'" in src_precompute
    src_state = inspect.getsource(state.execute_pipeline)
    loads_indices = "hull_player_indices=" in src_state
    print(f"  _worker_compute_daa_batch serializes player_indices: {saves_indices}")
    print(f"  state.execute_pipeline sets hull_player_indices in cached mode: {loads_indices}")
    assert not saves_indices or not loads_indices
    print("  -> CONFIRMED: Bug exists. Hull vertex player attribution is lost in cache mode.\n")

def verify_issue_8():
    print("--- [Issue 8] Calibration Studio Undo Retains Component Key ---")
    import inspect
    from perception import calibration_templates
    src = inspect.getsource(calibration_templates.render_mouse_calibration_studio)
    # Check lines around btn_undo
    undo_block = src.split('key=f"{key_prefix}_btn_undo"')[0].split('key=f"{key_prefix}_btn_apply"')[-1]
    increments_ver = "ver_key" in undo_block and ("st.session_state[ver_key] =" in undo_block or "+= 1" in undo_block)
    print(f"  Undo button increments canvas ver_key: {increments_ver}")
    assert not increments_ver
    print("  -> CONFIRMED: Bug exists. Undo button does not invalidate component key.\n")

def verify_issue_9():
    print("--- [Issue 9] Missing Projective Depth (w) Sign Normalization in calibrate() ---")
    import inspect
    from perception import homography
    src = inspect.getsource(homography.HomographyTransformer.calibrate)
    has_w_check = "center_test" in src or "H_inv = -H_inv" in src or "w < 0" in src
    print(f"  HomographyTransformer.calibrate normalizes w sign: {has_w_check}")
    assert not has_w_check
    print("  -> CONFIRMED: Bug exists. calibrate() lacks projective horizon sign check.\n")

def verify_issue_10():
    print("--- [Issue 10] Dead Temporal LRU Cache in Stream Manager ---")
    from ui import state
    import inspect
    src = inspect.getsource(state.execute_pipeline)
    uses_cache = "detection_cache" in src
    print(f"  state.execute_pipeline queries stream_manager.detection_cache in live mode: {uses_cache}")
    assert not uses_cache
    print("  -> CONFIRMED: Bug exists. Live mode never uses or populates detection_cache.\n")

def verify_issue_11():
    print("--- [Issue 11] Duplicate Player Circles in Stage 2 View ---")
    import inspect
    from ui.views import diagnostics_view
    src = inspect.getsource(diagnostics_view.render_diagnostics_view)
    s2_block = src.split("elif active_stage == stage_options[1]:")[1].split("elif active_stage == stage_options[2]:")[0]
    has_duplicate_drawing = "for p in det.players:" in s2_block and "cv2.circle(cam_dbg, fp" in s2_block
    print(f"  Stage 2 inspection loops over det.players drawing circles over cam_dbg: {has_duplicate_drawing}")
    assert has_duplicate_drawing
    print("  -> CONFIRMED: Bug exists. Redundant duplicate player circles are drawn in Stage 2.\n")

def verify_issue_12():
    print("--- [Issue 12] Setup Gate Model Options Lack PyTorch Fallbacks ---")
    import inspect
    from ui.components import setup_gate
    src = inspect.getsource(setup_gate.render_setup_gate)
    has_pt_fallbacks = "models/best.pt" in src or "models/yolov8n.pt" in src
    print(f"  setup_gate.py includes PyTorch (.pt) model fallback options: {has_pt_fallbacks}")
    assert not has_pt_fallbacks
    print("  -> CONFIRMED: Bug exists. If only .pt models exist, avail_models is empty and crashes selectbox.\n")

if __name__ == "__main__":
    verify_issue_1()
    verify_issue_2()
    verify_issue_3()
    verify_issue_4()
    verify_issue_5()
    verify_issue_6()
    verify_issue_7()
    verify_issue_8()
    verify_issue_9()
    verify_issue_10()
    verify_issue_11()
    verify_issue_12()
    print("=" * 60)
    print(">>> ALL 12 BUGS RE-VERIFIED AND CONFIRMED TO EXIST! <<<")
    print("=" * 60)
