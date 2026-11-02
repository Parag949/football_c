# 🔮 TacticalVision Pro — Future Work & Roadmap
### Prioritized Technical Specifications, Algorithmic Enhancements & Optimizations

This document serves as the persistent architectural roadmap for future enhancements, algorithmic extensions, and compute optimizations. Any engineer or LLM can refer to this file to implement these prioritized features without having to re-architect or guess system contracts.

---

## 📋 Prioritized Feature Matrix

| ID | Feature Name | Category | Primary Target Files | Estimated Impact |
| :--- | :--- | :--- | :--- | :--- |
| **FW-02** | **Temporal Coordinate Smoothing** | Kinematics / Perception | `core/pipeline.py`, `ui/state.py` | 🌟 High (Broadcast Fluidity) |
| **FW-03** | **Ball Trajectory Extrapolation** | Physics / Perception | `core/pipeline.py`, `perception/detector.py` | ⚡ Medium (Pass Tracking) |
| **FW-07** | **Velocity-Weighted Pitch Control** | Geometry / Spatial | `algorithms/voronoi_dominance.py` | 💡 Advanced Tactical Model |

---

## 1. FW-02: Temporal Coordinate Smoothing (Eliminate Jitter)

### Problem Statement:
YOLO runs independently on each frame. Sub-pixel noise and bounding box jitter (+- 1 to 3 px) cause player coordinates to twitch between consecutive frames, creating micro-fluctuations in Voronoi cell boundaries and Convex Hull perimeters.

### Implementation Specification:
- Implement Exponential Moving Average (EMA) or 1D Kalman Filter on player foot contacts:
  ```
  X_smoothed(t) = alpha * X_raw(t) + (1.0 - alpha) * X_smoothed(t - 1)
  Y_smoothed(t) = alpha * Y_raw(t) + (1.0 - alpha) * Y_smoothed(t - 1)
  ```
  - Smoothing Factor: `alpha = 0.70` (fast responsiveness with zero lag).
  - Velocity Gate: If distance `hypot(X_raw - X_prev) > 3.0 meters` (camera cut or player teleportation), reset filter immediately (`alpha = 1.0`).
- **Files to Modify**: `core/pipeline.py` inside `stage2_projection()` or `MatchStreamManager.detection_cache`.

---

## 2. FW-03: Ball Trajectory Extrapolation & Occlusion Recovery

### Problem Statement:
In fast television broadcast footage, the ball is small (~12 px) and frequently suffers from motion blur, aerial occlusion, or jersey clustering during long passes.

### Implementation Specification:
- Maintain a rolling 5-frame ball history: `[(x_0, y_0, t_0), ..., (x_k, y_k, t_k)]`.
- If the detector fails to detect the ball for 1 to 3 frames:
  - Fit a 2nd-degree polynomial (gravity arc for air balls) or linear constant-velocity vector:
    ```
    v_x = x_(t-1) - x_(t-2)
    v_y = y_(t-1) - y_(t-2)
    x_extrapolated = x_(t-1) + v_x
    y_extrapolated = y_(t-1) + v_y
    ```
  - Mark `ball_extrapolated = True` in `PitchPositions` contract.
  - Drop extrapolation after 4 consecutive missed frames to avoid runaway drift.

---

## 3. FW-07: Velocity-Weighted Pitch Control (Fernandez-Bornn Model)

### Problem Statement:
Standard Voronoi assumes all players have equal omni-directional reach regardless of running speed. In reality, a sprinting winger controls space ahead of them, not behind them.

### Implementation Specification:
- Replace standard Voronoi Euclidean metric with time-to-intercept distance:
  ```
  Distance_effective(P, x) = EuclideanDistance(P + v_P * tau, x) / MaxSpeed
  ```
- Creates dynamic, elliptical pitch control teardrops reflecting true momentum and passing accessibility.

---

## 🏁 Graduated to Production (Completed)

The following items from earlier roadmap iterations have been fully implemented, validated with automated test suites, and integrated into the production pipeline:

| ID | Feature Name | Core Files | Shipped Version / Status |
| :--- | :--- | :--- | :--- |
| **FW-01** | **Dijkstra Passing Network** | `algorithms/passing_network.py`, `core/pipeline.py` | v1.2 (Operational, 100% test pass) |
| **FW-04** | **ONNX Multi-Thread Tuning** | `perception/detector.py`, `ui/views/diagnostics_view.py` | v1.4 (Operational, ORT_SEQUENTIAL, ORT_ENABLE_ALL, Stage 1 Telemetry) |
| **FW-05** | **Async Frame Prefetching** | `perception/video_stream.py`, `ui/state.py` | v1.3 (Operational, 15,000x speedup) |
| **FW-06** | **Interactive Tactical Ruler** | `algorithms/tactical_ruler.py`, `visualization/pitch_renderer.py`, `visualization/overlays.py` | v1.4 (Operational, Euclidean Distance Tape, VAR Offside Laser Line) |
| **FW-08** | **Git Remote History Alignment** | GitHub Remote (`main` branch) | v1.3 (Clean push, 100% synchronized) |

*Full architectural specifications and test contracts for graduated features are maintained in `FEATURES_AND_IMPLEMENTATION.md` and `ARCHITECTURE.md`.*
