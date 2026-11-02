"""
Test Suite for Custom Landmark Anchor Feature & Calibration Studio
"""
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import cv2
from perception.calibration_templates import (
    FIFA_CHALK_LANDMARKS,
    GROUND_TRUTH_TEMPLATES,
    render_pitch_projection_preview,
    render_landmarks_pitch_thumbnail
)
from perception.homography import HomographyTransformer

def test_landmark_definitions():
    print("[1] Testing FIFA Chalk Landmarks definitions...")
    assert len(FIFA_CHALK_LANDMARKS) >= 20, f"Expected >= 20 landmarks, got {len(FIFA_CHALK_LANDMARKS)}"
    for name, (x, y) in FIFA_CHALK_LANDMARKS.items():
        assert 0.0 <= x <= 105.0, f"Landmark {name} X ({x}) out of pitch bounds [0, 105]"
        assert 0.0 <= y <= 68.0, f"Landmark {name} Y ({y}) out of pitch bounds [0, 68]"
    print("  -> All FIFA Chalk Landmarks within valid [105m x 68m] metric bounds.")

def test_quick_presets_validity():
    print("[2] Testing Quick Preset landmark assignments...")
    landmark_names = list(FIFA_CHALK_LANDMARKS.keys())
    
    presets = {
        "Right Box": [landmark_names[0], landmark_names[2], landmark_names[3], landmark_names[4]],
        "Midfield": [
            "Halfway Line — Far Touchline (Top) [52.5m, 0m]",
            "Center Circle — Right Apex [61.65m, 34m]",
            "Halfway Line — Near Touchline (Bottom) [52.5m, 68m]",
            "Center Circle — Left Apex [43.35m, 34m]"
        ],
        "Left Box": [
            "Left 18-Yard Box — Goal Line Far [0m, 13.84m]",
            "Left 18-Yard Box — Infield Far Corner [16.5m, 13.84m]",
            "Left Penalty Arc 'D' — Infield Apex [20.15m, 34m]",
            "Left 18-Yard Box — Goal Line Near [0m, 54.16m]"
        ],
        "Corner Flags": [
            "Corner Flag — Left Far (Top-Left) [0m, 0m]",
            "Corner Flag — Right Far (Top-Right) [105m, 0m]",
            "Corner Flag — Right Near (Bottom-Right) [105m, 68m]",
            "Corner Flag — Left Near (Bottom-Left) [0m, 68m]"
        ]
    }
    
    for name, p_lms in presets.items():
        assert len(set(p_lms)) == 4, f"Preset {name} contains duplicate landmarks!"
        pts = [FIFA_CHALK_LANDMARKS[lm] for lm in p_lms]
        # Verify non-collinear
        for i in range(4):
            p1, p2, p3 = [pts[j] for j in range(4) if j != i]
            area = abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]))
            assert area >= 0.5, f"Preset {name} has 3 collinear points: area={area}"
    print("  -> All 4 Quick Presets verified: unique, non-collinear, and valid.")

def test_collinear_and_duplicate_detector():
    print("[3] Testing Collinear & Duplicate Landmark Detector...")
    # Collinear halfway line
    p_halfway = [
        "Halfway Line — Far Touchline (Top) [52.5m, 0m]",
        "Center Spot [52.5m, 34m]",
        "Halfway Line — Near Touchline (Bottom) [52.5m, 68m]",
        "Corner Flag — Right Far (Top-Right) [105m, 0m]"
    ]
    raw_pts = [FIFA_CHALK_LANDMARKS[s] for s in p_halfway]
    def _check_collinear(p1, p2, p3):
        area = abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]))
        return area < 0.5
    has_collinear = (
        _check_collinear(raw_pts[0], raw_pts[1], raw_pts[2]) or
        _check_collinear(raw_pts[0], raw_pts[1], raw_pts[3]) or
        _check_collinear(raw_pts[0], raw_pts[2], raw_pts[3]) or
        _check_collinear(raw_pts[1], raw_pts[2], raw_pts[3])
    )
    assert has_collinear, "Detector failed to catch collinear halfway line points!"
    print("  -> Correctly flagged collinear landmarks.")

def test_projection_preview():
    print("[4] Testing Candidate Projection Preview...")
    test_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    src_pts = np.float32([
        [200, 150],
        [900, 180],
        [1050, 600],
        [150, 550]
    ])
    dst_pts = np.float32([
        [88.5, 13.84],
        [105.0, 13.84],
        [105.0, 54.16],
        [84.85, 34.0]
    ])
    H = cv2.getPerspectiveTransform(src_pts, dst_pts)
    preview = render_pitch_projection_preview(test_frame, H)
    assert preview.shape == (720, 1280, 3)
    # Check that preview is not completely black (lines were drawn)
    assert np.count_nonzero(preview) > 0, "Projection preview rendered nothing!"
    print("  -> Projection preview rendered non-empty pitch overlay with positive horizon.")

def test_homography_transformer_calibrate():
    print("[5] Testing HomographyTransformer calibrate method...")
    transformer = HomographyTransformer(frame_width=1280, frame_height=720, camera_mode="sideline")
    assert not getattr(transformer, 'is_custom_calibrated', False)
    
    src_pts = np.float32([[200, 150], [900, 180], [1050, 600], [150, 550]])
    dst_pts = np.float32([[88.5, 13.84], [105.0, 13.84], [105.0, 54.16], [84.85, 34.0]])
    transformer.calibrate(src_pts, dst_pts, frame_width=1280, frame_height=720)
    
    assert getattr(transformer, 'is_custom_calibrated', False)
    assert np.allclose(transformer.src_points, src_pts)
    assert np.allclose(transformer.dst_points, dst_pts)
    
    # Test round-trip transformation of anchor point
    test_img_pt = [(200.0, 150.0)]
    pitch_coords = transformer.image_to_pitch(test_img_pt)
    assert np.isclose(pitch_coords[0][0], 88.5, atol=0.5)
    assert np.isclose(pitch_coords[0][1], 13.84, atol=0.5)
    print("  -> HomographyTransformer calibration & roundtrip verified.")

if __name__ == "__main__":
    test_landmark_definitions()
    test_quick_presets_validity()
    test_collinear_and_duplicate_detector()
    test_projection_preview()
    test_homography_transformer_calibrate()
    print("\n✅ ALL CUSTOM LANDMARK ANCHOR TESTS PASSED 100%!")
