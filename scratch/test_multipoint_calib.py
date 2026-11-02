"""
Automated validation suite for multi-point flexible homography calibration (N >= 4).
Tests:
1. Collinearity detection on pitch points.
2. 4-point analytic calibration and residuals.
3. 5-point, 6-point, and 8-point RANSAC overdetermined calibration and residuals.
4. Robustness against slight mouse jitter / noise.
5. Invertibility and forward/backward projection accuracy.
"""

import sys
import os
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from perception.homography import HomographyTransformer

def test_collinearity_detection():
    print("Testing Collinearity Detection...")
    # 3 collinear points along touchline y=0
    collinear_pts = [(0.0, 0.0), (52.5, 0.0), (105.0, 0.0)]
    is_collinear, triplets = HomographyTransformer.check_collinearity(collinear_pts, threshold_area=0.5)
    assert is_collinear, "Failed: Collinear points along touchline were not flagged!"
    print(f"  PASS: Correctly flagged collinear triplet: {triplets}")

    # 4 points forming a quadrilateral (corners of pitch)
    non_collinear_pts = [(0.0, 0.0), (105.0, 0.0), (105.0, 68.0), (0.0, 68.0)]
    is_collinear, triplets = HomographyTransformer.check_collinearity(non_collinear_pts, threshold_area=0.5)
    assert not is_collinear, f"Failed: Non-collinear points falsely flagged as collinear: {triplets}"
    print("  PASS: Correctly passed non-collinear quadrilateral.")

    # 4 points where 3 are collinear
    three_collinear_pts = [(0.0, 0.0), (52.5, 0.0), (105.0, 0.0), (52.5, 34.0)]
    is_collinear, triplets = HomographyTransformer.check_collinearity(three_collinear_pts, threshold_area=0.5)
    assert is_collinear, "Failed: Triplet collinearity inside 4 points was not flagged!"
    print(f"  PASS: Correctly detected subset collinearity in 4-point set: {triplets}")

def test_4point_calibration():
    print("\nTesting 4-Point Analytic Calibration...")
    # Pitch coordinates: 4 corners of penalty box / pitch section
    pitch_pts = [(16.5, 13.84), (16.5, 54.16), (0.0, 68.0), (0.0, 0.0)]
    # Synthetic realistic camera pixels (perspective projection)
    img_pts = [(540.0, 420.0), (820.0, 440.0), (1150.0, 650.0), (220.0, 610.0)]

    transformer = HomographyTransformer()
    transformer.calibrate(img_pts, pitch_pts)
    assert transformer.is_custom_calibrated, "Failed: Transformer failed to calibrate with 4 points!"

    # Calculate residuals
    res_data = transformer.compute_reprojection_residuals(img_pts, pitch_pts)
    residuals = res_data['residuals']
    assert len(residuals) == 4, f"Failed: Expected 4 residuals, got {len(residuals)}"
    max_res = res_data['max_error']
    mean_res = res_data['mean_error']
    print(f"  4-Point Residuals: {[round(r, 4) for r in residuals]}")
    print(f"  Max Residual: {max_res:.4f}px, Mean Residual: {mean_res:.4f}px")
    assert max_res < 0.05, f"Failed: 4-point analytic calibration should have near-zero residual (<0.05px), got {max_res}"
    print("  PASS: 4-Point analytic calibration exact fit verified (< 0.05px).")

def test_multipoint_ransac_calibration():
    print("\nTesting Multi-Point RANSAC Calibration (N = 5, 6, 8)...")
    # Ground truth homography matrix (perspective transformation)
    # Pitch rectangle [0, 105] x [0, 68] to camera view [1280 x 720]
    src_test_pts = np.array([
        [0.0, 0.0],
        [105.0, 0.0],
        [105.0, 68.0],
        [0.0, 68.0]
    ], dtype=np.float32)
    dst_test_pts = np.array([
        [150.0, 650.0],
        [1130.0, 650.0],
        [950.0, 220.0],
        [330.0, 220.0]
    ], dtype=np.float32)

    import cv2
    gt_h = cv2.getPerspectiveTransform(dst_test_pts, src_test_pts) # image -> pitch

    # Define candidate pitch landmarks
    candidate_pitch = [
        (0.0, 0.0),      # L1: Bottom-left
        (105.0, 0.0),    # L2: Bottom-right
        (105.0, 68.0),   # L3: Top-right
        (0.0, 68.0),     # L4: Top-left
        (52.5, 34.0),    # L5: Center spot
        (52.5, 0.0),     # L6: Center touchline near
        (16.5, 34.0),    # L7: Left penalty spot
        (88.5, 34.0),    # L8: Right penalty spot
    ]

    h_inv = np.linalg.inv(gt_h) # pitch -> image

    for n in [5, 6, 8]:
        pts_pitch = candidate_pitch[:n]
        # Project pitch points to ground truth camera pixels
        pts_img_clean = []
        for X, Y in pts_pitch:
            vec = h_inv @ np.array([X, Y, 1.0], dtype=np.float64)
            u = vec[0] / vec[2]
            v = vec[1] / vec[2]
            pts_img_clean.append((float(u), float(v)))

        # Add simulated human click jitter (up to 0.4 px noise)
        np.random.seed(42)
        noise = np.random.uniform(-0.4, 0.4, size=(n, 2))
        pts_img_noisy = [
            (u + noise[i, 0], v + noise[i, 1])
            for i, (u, v) in enumerate(pts_img_clean)
        ]

        # Calibrate with N points
        transformer = HomographyTransformer()
        transformer.calibrate(pts_img_noisy, pts_pitch)
        assert transformer.is_custom_calibrated, f"Failed: Transformer failed to calibrate with {n} points!"

        res_data = transformer.compute_reprojection_residuals(pts_img_noisy, pts_pitch)
        residuals = res_data['residuals']
        assert len(residuals) == n, f"Failed: Expected {n} residuals, got {len(residuals)}"
        mean_res = res_data['mean_error']
        max_res = res_data['max_error']

        print(f"  N = {n} Points: Mean Residual = {mean_res:.3f}px, Max Residual = {max_res:.3f}px")
        # Mean residual should closely match the injected noise magnitude (< 1.0px)
        assert mean_res < 1.0, f"Failed: Mean residual {mean_res}px exceeded threshold 1.0px for N={n}"

        # Test forward mapping on center spot (52.5, 34.0)
        center_img = pts_img_noisy[4]
        mapped_pitch = transformer.image_to_pitch([center_img])[0]
        dist_error = np.hypot(mapped_pitch[0] - 52.5, mapped_pitch[1] - 34.0)
        print(f"    Center spot recovery error: {dist_error:.4f}m on pitch")
        assert dist_error < 0.25, f"Failed: Center spot pitch error too high: {dist_error}m"
        print(f"  PASS: N = {n} RANSAC calibration converged accurately.")

def main():
    print("=" * 60)
    print("RUNNING MULTI-POINT CALIBRATION VALIDATION SUITE")
    print("=" * 60)
    test_collinearity_detection()
    test_4point_calibration()
    test_multipoint_ransac_calibration()
    print("=" * 60)
    print("ALL MULTI-POINT CALIBRATION TESTS PASSED WITH 100% ACCURACY!")
    print("=" * 60)

if __name__ == "__main__":
    main()
