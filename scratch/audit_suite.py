import os
import sys

# Ensure repository root is in sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import json
import cv2
import numpy as np

def run_audit():
    print("=== TACTICAL VISION PRO: COMPREHENSIVE PROJECT AUDIT ===")
    errors = []

    # 1. Syntax & Imports
    print("\n[1/7] Checking module imports & compilation...")
    try:
        import core.contracts as contracts
        import core.pipeline as pipeline
        import perception.detector as detector
        import perception.homography as homography
        import perception.pitch_model as pitch_model
        import perception.precompute as precompute
        import perception.video_stream as video_stream
        import algorithms.voronoi_dominance as voronoi_dominance
        import algorithms.convex_hull as convex_hull
        import algorithms.coach_analytics as coach_analytics
        import algorithms.daa_benchmarks as daa_benchmarks
        import visualization.pitch_renderer as pitch_renderer
        import visualization.overlays as overlays
        import app
        print("  -> ALL CORE, PERCEPTION, ALGORITHM, VISUALIZATION, AND APP MODULES IMPORTED CLEANLY")
    except Exception as e:
        errors.append(f"Import failure: {e}")
        print(f"  -> ERROR: {e}")

    # 2. Check Removed Files (Confirm No Ghost Imports)
    print("\n[2/7] Verifying removal of redundant files...")
    ghost_files = [
        "algorithms/kinematics.py",
        "scripts/debug_stage1_yolo.py",
        "scripts/debug_stage2_projection.py",
        "scripts/debug_stage3_daa.py",
        "scripts/debug_stage4_render.py",
        "scripts"
    ]
    for gf in ghost_files:
        full_p = os.path.join(REPO_ROOT, gf)
        if os.path.exists(full_p):
            errors.append(f"Ghost file/folder still present: {gf}")
            print(f"  -> FAIL: {gf} still exists!")
        else:
            print(f"  -> OK: {gf} removed")

    # 3. Check Data Files & Dimensions
    print("\n[3/7] Verifying sample media & precomputed caches...")
    sample_broadcast = os.path.join(REPO_ROOT, "data/sample_broadcast.mp4")
    cache_broadcast = os.path.join(REPO_ROOT, "data/precomputed_broadcast.json")
    user_match = os.path.join(REPO_ROOT, "data/user_match.mp4")
    cache_user = os.path.join(REPO_ROOT, "data/precomputed_user.json")

    for path in [sample_broadcast, user_match]:
        if not os.path.exists(path):
            errors.append(f"Missing expected media file: {path}")
            print(f"  -> FAIL: {path} not found")
        else:
            print(f"  -> OK: {os.path.basename(path)} exists ({os.path.getsize(path)/1024:.1f} KB)")

    for c_path in [cache_broadcast, cache_user]:
        if os.path.exists(c_path):
            print(f"  -> OK: Optional cache {os.path.basename(c_path)} exists ({os.path.getsize(c_path)/1024:.1f} KB)")
        else:
            print(f"  -> INFO: {os.path.basename(c_path)} not precomputed on disk (generable on demand via Setup Gate)")

    # Verify cache dimensions match video resolution if cache exists
    if os.path.exists(cache_broadcast):
        cap_b = cv2.VideoCapture(sample_broadcast)
        bw = int(cap_b.get(cv2.CAP_PROP_FRAME_WIDTH))
        bh = int(cap_b.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_b.release()

        with open(cache_broadcast, "r") as f:
            data_b = json.load(f)
        print(f"  -> Broadcast video: {bw}x{bh}, Cache metadata: {data_b.get('video_width')}x{data_b.get('video_height')}")
        if data_b.get('video_width') != bw or data_b.get('video_height') != bh:
            errors.append(f"Broadcast cache dims mismatch: cache has {data_b.get('video_width')}x{data_b.get('video_height')} vs video {bw}x{bh}")

    # 4. End-to-End Pipeline Contract Validation
    print("\n[4/7] Verifying core/pipeline.py contracts & stage execution...")
    try:
        from core.contracts import FrameDetections, PitchPositions, TacticalGeometry, FrameBundle
        from core.pipeline import (
            stage1_perception,
            stage2_projection,
            stage3_geometry,
            stage4_render_radar,
            stage4_render_overlay,
            run_pipeline_frame
        )
        from perception.detector import FootballDetector

        test_frame = np.zeros((540, 960, 3), dtype=np.uint8)
        det_engine = FootballDetector()

        # Test stage 1
        dets = stage1_perception(test_frame, det_engine, frame_idx=0)
        assert isinstance(dets, FrameDetections), "stage1 must return FrameDetections"
        print(f"  -> Stage 1 Perception Contract: PASSED (detections: {len(dets.players)})")

        # Test stage 2
        transformer = homography.HomographyTransformer("broadcast")
        transformer.update_dimensions(960, 540)
        pitch_pos = stage2_projection(dets, transformer)
        assert isinstance(pitch_pos, PitchPositions), "stage2 must return PitchPositions"
        print(f"  -> Stage 2 Homography Projection Contract: PASSED")

        # Test stage 3
        geom = stage3_geometry(pitch_pos)
        assert isinstance(geom, TacticalGeometry), "stage3 must return TacticalGeometry"
        print(f"  -> Stage 3 DAA Tactical Geometry Contract: PASSED")

        # Test stage 4
        from visualization.pitch_renderer import TacticalPitchRenderer
        from visualization.overlays import BroadcastOverlayEngine

        pitch_rend = TacticalPitchRenderer()
        overlay_eng = BroadcastOverlayEngine(transformer)

        radar_img = stage4_render_radar(pitch_pos, geom, pitch_rend)
        assert radar_img.shape[0] > 0 and radar_img.shape[1] > 0, "Radar render failed"
        print(f"  -> Stage 4a 2D Radar Canvas Render Contract: PASSED ({radar_img.shape})")

        overlay_img = stage4_render_overlay(test_frame, pitch_pos, geom, overlay_eng)
        assert overlay_img.shape == test_frame.shape, "Overlay render shape mismatch"
        print(f"  -> Stage 4b Broadcast AR Overlay Render Contract: PASSED ({overlay_img.shape})")

        # Test full pipeline bundle
        aug_out, radar_out, bundle = run_pipeline_frame(
            test_frame,
            detector=det_engine,
            transformer=transformer,
            renderer=pitch_rend,
            overlay_engine=overlay_eng,
            frame_idx=0
        )
        assert isinstance(bundle, FrameBundle), "run_pipeline_frame must return FrameBundle"
        bundle_dict = bundle.to_dict()
        assert 'player_indices' in bundle_dict.get('hull', {}), "Missing player_indices in bundle hull serialization"
        assert 'dijkstra' in bundle_dict, "Missing dijkstra in bundle serialization"
        bundle_reconstructed = FrameBundle.from_dict(bundle_dict)
        assert isinstance(bundle_reconstructed, FrameBundle), "FrameBundle.from_dict reconstruction failed"
        print(f"  -> Unified End-to-End FrameBundle Contract & Serialization Round-Trip: PASSED")
    except Exception as e:
        errors.append(f"Pipeline contract validation error: {e}")
        print(f"  -> ERROR in Pipeline: {e}")

    # 5. Multiprocessing Core Config Validation
    print("\n[5/7] Verifying Multiprocess Worker Configuration & DAA Benchmarks...")
    try:
        from perception.precompute import precompute_video_segment
        import algorithms.daa_benchmarks as daa_benchmarks
        # Verify DAA benchmarks work
        df_hull = daa_benchmarks.benchmark_convex_hull([5, 11], iterations=2)
        df_vor = daa_benchmarks.benchmark_voronoi_dominance([6, 11], iterations=2)
        assert len(df_hull) == 2 and len(df_vor) == 2
        print(f"  -> DAA Benchmark Suite: PASSED ({len(df_hull)} hull levels, {len(df_vor)} voronoi levels tested)")
    except Exception as e:
        errors.append(f"DAA / Multiprocess verification error: {e}")
        print(f"  -> ERROR in DAA / Multiprocess: {e}")

    # 6. Streamlit Unified Application Check
    print("\n[6/7] Checking Streamlit Unified Application Structure...")
    app_entry = os.path.join(REPO_ROOT, "app.py")
    if os.path.exists(app_entry):
        print(f"  -> OK: Unified entry point {os.path.basename(app_entry)} verified")
    else:
        errors.append("app.py is missing!")

    # 7. Standalone Test Verification Integration
    print("\n[7/7] Verifying Standalone Test Suites Integration...")
    import subprocess
    standalone_tests = [
        "scratch/test_all_6_bugfixes.py",
        "scratch/test_ring_buffer_fw13.py",
        "scratch/test_hardware_player.py",
        "scratch/test_async_prefetch.py",
        "scratch/test_dijkstra_passing.py",
        "scratch/test_multipoint_calib.py",
        "scratch/test_tactical_ruler.py",
        "scratch/verify_coordinates.py"
    ]
    for test_file in standalone_tests:
        test_path = os.path.join(REPO_ROOT, test_file)
        if os.path.exists(test_path):
            res = subprocess.run([sys.executable, test_path], capture_output=True, text=True)
            if res.returncode != 0:
                err_msg = (res.stderr or res.stdout).strip().splitlines()[-1] if (res.stderr or res.stdout) else f"code {res.returncode}"
                errors.append(f"Standalone test failure in {test_file}: {err_msg}")
                print(f"  -> FAIL: {test_file} ({err_msg})")
            else:
                print(f"  -> PASS: {test_file} executed cleanly with 0 errors")

    print("\n=======================================================")
    if not errors:
        print(">>> ALL CHECKS PASSED: 100% OPERATIONAL & VERIFIED <<<")
        print("=======================================================")
        return 0
    else:
        print(f">>> AUDIT FAILED WITH {len(errors)} ERROR(S):")
        for err in errors:
            print(f"  - {err}")
        print("=======================================================")
        return 1

if __name__ == "__main__":
    sys.exit(run_audit())
