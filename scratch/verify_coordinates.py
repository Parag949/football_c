"""
End-to-End Coordinate Consistency & Precision Verification Suite
Validates mathematical integrity across all pipeline stages:
- Stage 1: Frame pixels
- Stage 2: Homography metric projection & round-trip precision
- Stage 3: Voronoi territorial partitioning & Convex Hull geometry
- Stage 4: 2D Radar & AR Broadcast Overlay
- UI: Coordinate probe & boundary checks
"""

import sys
import os
import numpy as np

# Ensure repository root is in python path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from perception.homography import HomographyTransformer
from perception.pitch_model import PitchModel
from algorithms.voronoi_dominance import compute_spatial_dominance, compute_bounded_voronoi
from algorithms.convex_hull import compute_defensive_cohesion
from visualization.pitch_renderer import TacticalPitchRenderer
from visualization.overlays import BroadcastOverlayEngine
from core.pipeline import (
    stage1_perception,
    stage2_projection,
    stage3_geometry,
    stage4_render_radar,
    stage4_render_overlay,
    run_pipeline_frame
)
from core.contracts import PlayerDetection, FrameDetections, PitchPositions


def test_roundtrip_precision():
    print("Test 1: Homography Round-Trip Precision (image -> pitch -> image)...")
    transformer = HomographyTransformer(1920, 1080, camera_mode='sideline')
    test_pixels = [
        (960.0, 540.0),   # screen center
        (200.0, 360.0),   # top left pitch
        (1600.0, 360.0),  # top right pitch
        (1850.0, 760.0),  # near touchline right
        (100.0, 760.0),   # near touchline left
        (960.0, 850.0),   # near apron (outside pitch line)
    ]
    pitch_pts = transformer.image_to_pitch(test_pixels, filter_out_of_bounds=False)
    reproj_px = transformer.pitch_to_image(pitch_pts)
    
    max_err = 0.0
    for orig, rep in zip(test_pixels, reproj_px):
        err = np.linalg.norm(np.array(orig) - np.array(rep))
        if err > max_err:
            max_err = err
    assert max_err < 0.05, f"Reprojection error too high: {max_err:.4f}px"
    print(f"  PASS: Max round-trip reprojection error across all points is {max_err:.4f}px (< 0.05px)!")


def test_touchline_buffer_retention():
    print("Test 2: Touchline Buffer Retention vs Discard Threshold...")
    transformer = HomographyTransformer(1920, 1080, camera_mode='sideline')
    # A point slightly outside pitch boundary (e.g., Y=69.0m, within 2m buffer)
    # Project to image and back with filter_out_of_bounds=True
    pts = transformer.pitch_to_image([(52.5, 69.2), (52.5, 75.0)])
    filtered = transformer.image_to_pitch(pts, filter_out_of_bounds=True, buffer_meters=2.0)
    
    # 69.2m is within 2.0m buffer (<= 70.0m) -> retained with true coordinate
    # 75.0m is outside 2.0m buffer (> 70.0m) -> discarded
    assert len(filtered) == 1, f"Expected 1 point retained, got {len(filtered)}"
    retained_x, retained_y = filtered[0]
    assert abs(retained_y - 69.2) < 0.05, f"Expected unclamped Y ~ 69.2, got {retained_y}"
    print(f"  PASS: Point at 69.2m retained unclamped at {retained_y:.2f}m; point at 75m discarded!")


def test_voronoi_offpitch_stability():
    print("Test 3: Voronoi Partitioning with Touchline Buffer Coordinates...")
    team_a = [(20.0, 30.0), (30.0, 40.0), (52.5, 34.0), (70.0, 69.2)]  # 69.2m touchline run-off
    team_b = [(80.0, 20.0), (85.0, 34.0), (90.0, 50.0)]
    
    res = compute_spatial_dominance(team_a, team_b)
    assert res['team_a_pct'] + res['team_b_pct'] > 99.9, "Territory % must sum to 100%"
    assert len(res['cells_a']) > 0 and len(res['cells_b']) > 0, "Voronoi cells must be generated"
    # Ensure all Voronoi polygon vertices remain strictly bounded to pitch dimensions
    for cell in res['cells_a'] + res['cells_b']:
        for vx, vy in cell['polygon']:
            assert -1e-4 <= vx <= 105.0 + 1e-4, f"Voronoi vertex X out of pitch: {vx}"
            assert -1e-4 <= vy <= 68.0 + 1e-4, f"Voronoi vertex Y out of pitch: {vy}"
    print(f"  PASS: Voronoi dominance cleanly calculated ({res['team_a_pct']:.1f}% vs {res['team_b_pct']:.1f}%), bounded strictly to turf!")


def test_overlay_turf_mask_projection():
    print("Test 4: AR Overlay Turf Mask Under Custom Calibration...")
    transformer = HomographyTransformer(1920, 1080)
    # Simulate custom 4-point calibration on 18-yard box
    src = np.float32([[600, 375], [1350, 375], [1850, 760], [100, 760]])
    dst = np.float32([[88.5, 13.84], [105.0, 13.84], [105.0, 54.16], [88.5, 54.16]])
    transformer.calibrate(src, dst, frame_width=1920, frame_height=1080)
    
    overlay_engine = BroadcastOverlayEngine(transformer)
    mask = overlay_engine.get_turf_binary_mask((1080, 1920))
    # Turf mask should not be artificially restricted to only the 18-yard box
    turf_pixels = np.count_nonzero(mask)
    total_pixels = 1080 * 1920
    assert turf_pixels > 0.20 * total_pixels, f"Turf mask too small: {turf_pixels} / {total_pixels}"
    print(f"  PASS: Custom calibration turf mask covers {turf_pixels/total_pixels*100:.1f}% of frame without 18-yard box clipping!")


def test_full_pipeline_bundle():
    print("Test 5: Full Decoupled Pipeline (Stage 1 -> 2 -> 3 -> 4)...")
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    # Mock detector that produces detections
    class MockDetector:
        def detect_frame(self, f):
            return {
                'players': [
                    PlayerDetection(box=(500, 350, 530, 420), team_id=0),
                    PlayerDetection(box=(600, 360, 630, 430), team_id=0),
                    PlayerDetection(box=(700, 370, 730, 440), team_id=0),
                    PlayerDetection(box=(800, 350, 830, 420), team_id=1),
                    PlayerDetection(box=(900, 360, 930, 430), team_id=1),
                    PlayerDetection(box=(950, 370, 980, 440), team_id=1),
                ],
                'ball': (650, 420)
            }
            
    transformer = HomographyTransformer(1280, 720, camera_mode='sideline')
    renderer = TacticalPitchRenderer(width=800, height=520)
    overlay_engine = BroadcastOverlayEngine(transformer)
    
    aug, radar, bundle = run_pipeline_frame(
        raw_frame=frame,
        detector=MockDetector(),
        transformer=transformer,
        renderer=renderer,
        overlay_engine=overlay_engine,
        frame_idx=1
    )
    
    assert radar.shape == (520, 800, 3), f"Radar shape mismatch: {radar.shape}"
    assert aug.shape == (720, 1280, 3), f"AR Overlay shape mismatch: {aug.shape}"
    assert len(bundle.positions.team_a) == 3
    assert len(bundle.positions.team_b) == 3
    assert len(bundle.geometry.hull_polygon) == 3
    assert len(bundle.geometry.hull_player_indices) == 3
    print("  PASS: End-to-end pipeline executes flawlessly with 100% contract compliance!")


if __name__ == '__main__':
    print("=" * 60)
    print("RUNNING TACTICAL PIPELINE COORDINATE INTEGRITY TESTS")
    print("=" * 60)
    test_roundtrip_precision()
    test_touchline_buffer_retention()
    test_voronoi_offpitch_stability()
    test_overlay_turf_mask_projection()
    test_full_pipeline_bundle()
    print("=" * 60)
    print("ALL 5 COORDINATE INTEGRITY CHECKS PASSED WITH 100% ACCURACY!")
    print("=" * 60)
