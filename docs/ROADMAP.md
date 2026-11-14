# 🔮 TacticalVision Pro — Technical Roadmap & Future Work

This document maintains the persistent architectural roadmap for future enhancements, algorithmic extensions, and compute optimizations.

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
YOLO runs independently on each frame. Sub-pixel noise and bounding box jitter ($\pm 1\text{--}3\text{ px}$) cause player coordinates to twitch between consecutive frames, creating micro-fluctuations in Voronoi cell boundaries and Convex Hull perimeters.

### Implementation Specification:
- Implement Exponential Moving Average (EMA) or 1D Kalman Filter on player foot contacts:
  $$\begin{aligned}
  X_{\text{smoothed}}(t) &= \alpha \cdot X_{\text{raw}}(t) + (1.0 - \alpha) \cdot X_{\text{smoothed}}(t - 1) \\
  Y_{\text{smoothed}}(t) &= \alpha \cdot Y_{\text{raw}}(t) + (1.0 - \alpha) \cdot Y_{\text{smoothed}}(t - 1)
  \end{aligned}$$
  - **Smoothing Factor**: $\alpha = 0.70$ (fast responsiveness with zero lag).
  - **Velocity Gate**: If Euclidean distance $\text{hypot}(X_{\text{raw}} - X_{\text{prev}}) > 3.0\text{ meters}$ (camera cut or player teleportation), reset filter immediately ($\alpha = 1.0$).
- **Target Files**: `core/pipeline.py` inside `stage2_projection()` or `MatchStreamManager.detection_cache`.

---

## 2. FW-03: Ball Trajectory Extrapolation & Occlusion Recovery

### Problem Statement:
In fast television broadcast footage, the ball is small ($\approx 12\text{ px}$) and frequently suffers from motion blur, aerial occlusion, or player kit clustering during long passes.

### Implementation Specification:
- Maintain a rolling 5-frame ball history: $[(x_0, y_0, t_0), \dots, (x_k, y_k, t_k)]$.
- If the detector fails to detect the ball for 1 to 3 frames:
  - Fit a 2nd-degree polynomial (gravity arc for air balls) or linear constant-velocity vector:
    $$\begin{aligned}
    v_x &= x(t-1) - x(t-2) \\
    v_y &= y(t-1) - y(t-2) \\
    x_{\text{extrapolated}} &= x(t-1) + v_x \\
    y_{\text{extrapolated}} &= y(t-1) + v_y
    \end{aligned}$$
  - Mark `ball_extrapolated = True` in `PitchPositions` contract.
  - Drop extrapolation after 4 consecutive missed frames to avoid runaway drift.

---

## 3. FW-07: Velocity-Weighted Pitch Control (Fernandez-Bornn Model)

### Problem Statement:
Standard Voronoi assumes all players have equal omni-directional reach regardless of running speed. In reality, a sprinting winger controls space ahead of them, not behind them.

### Implementation Specification:
- Replace standard Voronoi Euclidean metric with time-to-intercept distance:
  $$\text{Dist}_{\text{effective}}(P, x) = \frac{\text{EuclideanDistance}(P + \mathbf{v}_P \cdot \tau, x)}{\text{MaxSpeed}}$$
- Creates dynamic, elliptical pitch control teardrops reflecting true momentum and passing accessibility.

---

## 🏁 Graduated to Production (Completed & Verified)

The following capabilities have been fully implemented, validated with automated test suites, and integrated into production:

| ID | Feature Name | Core Files | Shipped Capabilities |
| :--- | :--- | :--- | :--- |
| **FW-04** | **ONNX Multi-Thread Tuning** | `perception/detector.py`, `ui/views/diagnostics_view.py` | `ORT_SEQUENTIAL`, `ORT_ENABLE_ALL`, Stage 1 detection telemetry |
| **FW-05** | **Async Frame Prefetching** | `perception/video_stream.py`, `ui/state.py` | Background prefetch executor with lock synchronization |
| **FW-06** | **Interactive Tactical Ruler** | `algorithms/tactical_ruler.py`, `visualization/pitch_renderer.py` | Euclidean distance tape, offside breach radar |
| **FW-09** | **CIELAB Kit Clustering & Referee Isolation** | `perception/jersey_classifier.py`, `perception/detector.py` | Perceptual 4D contrast, chromatic referee outlier isolation (`team_id = -1`) |
| **FW-10** | **Native Source-FPS Playback** | `app.py`, `ui/components/control_deck.py` | Source-rate matching, play/pause, scrub slider |
| **FW-11** | **Multi-Regime Camera Calibration** | `perception/homography.py`, `ui/components/pitch_mask_studio.py` | Planar, Universal radial lens distortion, and Panorama Thin-Plate Spline (TPS) |
| **FW-12** | **Cache Acceleration Engine** | `perception/precompute.py`, `ui/state.py` | Stride $O(1)$ index mapping, deterministic per-video cache storage |
| **FW-13** | **Decoded Frame Ring Buffer** | `perception/video_stream.py` | 120-frame bounded memory buffer, thread-safe prefetching |
| **FW-14** | **Hardware WebGL Media Player** | `ui/components/hardware_player.py`, `hardware_player.js` | 60 FPS HTML5 dual-canvas synchronizer with hardware video acceleration |
| **FW-15** | **Tactical Whiteboard & Simulation** | `ui/components/tactical_sandbox.py`, `tactical_sandbox_canvas.py` | Interactive drag-and-drop 2D board with live formation, xT, and pitch control |
| **FW-16** | **Continuous Pitch Control & xT** | `algorithms/pitch_control.py`, `algorithms/expected_threat.py` | Radial player dominance fields and $16 \times 12$ spatial threat transition matrix |
| **FW-17** | **Panoramic Dynamic Detection** | `perception/detector.py` | Ultrawide aspect ratio scaling (`(480, 1536)`), relaxed box filtering |
