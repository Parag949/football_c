# 🔮 TacticalVision Pro — Future Work & Roadmap
### Prioritized Technical Specifications, Algorithmic Enhancements & Optimizations

This document serves as the persistent architectural roadmap for future enhancements, algorithmic extensions, and compute optimizations. Any engineer or LLM can refer to this file to implement these prioritized features without having to re-architect or guess system contracts.

---

## 📋 Prioritized Feature Matrix

| ID | Feature Name | Category | Primary Target Files | Estimated Impact |
| :--- | :--- | :--- | :--- | :--- |
| **FW-01** | **Dijkstra Passing Network** | DAA / Graph Theory | `algorithms/passing_network.py`, `core/pipeline.py` | 🌟 High (Visuals & DAA Viva) |
| **FW-02** | **Temporal Coordinate Smoothing** | Kinematics / Perception | `core/pipeline.py`, `ui/state.py` | 🌟 High (Broadcast Fluidity) |
| **FW-03** | **Ball Trajectory Extrapolation** | Physics / Perception | `core/pipeline.py`, `perception/detector.py` | ⚡ Medium (Pass Tracking) |
| **FW-04** | **ONNX Multi-Thread Tuning** | Compute / Latency | `perception/detector.py` | ⚡ Medium (50% CPU Speedup) |
| **FW-05** | **Async Frame Prefetching** | Concurrency / UX | `perception/video_stream.py`, `ui/state.py` | ⚡ Medium (Instant Scrubbing) |
| **FW-06** | **Interactive Tactical Ruler** | Analytics Tool | `visualization/pitch_renderer.py`, `ui/` | 💡 Feature (Coach Analytics) |
| **FW-07** | **Velocity-Weighted Pitch Control**| Geometry / Spatial | `algorithms/voronoi_dominance.py` | 💡 Advanced Tactical Model |
| **FW-08** | **Git Remote History Alignment** | DevOps / Versioning | Git CLI (`git pull --rebase`) | 🔧 Operational |

---

## 1. FW-01: Dijkstra Optimal Passing Network (Graph Theory)

### Problem Statement:
The visualization engine (`pitch_renderer.py`, `overlays.py`) and coach cards (`coach_analytics.py`) already contain complete rendering logic for `dijkstra_res` (Neon Emerald passing lanes with chevron dynamics, risk vs progression index). Currently, `dijkstra_res=None` is passed in `core/pipeline.py`.

### Mathematical & Algorithmic Formulation:
- **Graph Construction**: Directed complete graph G = (V, E) where:
  - Vertices V: Attacking outfield players (Team A, |V| <= 10).
  - Directed Edge (u, v): Possible ground or air pass from player u to player v.
- **Edge Weight Function**:
  ```
  Weight(u, v) = [ EuclideanDistance(u, v) / (DefensivePressure(u, v) + 1.0) ] - alpha * ForwardProgression(u, v)
  ```
  Where:
  - `ForwardProgression(u, v) = x_v - x_u` (positive towards opponent's goal).
  - `DefensivePressure(u, v)` = Inverse distance to nearest defender along pass corridor.
  - `alpha` = Tactical aggression coefficient (default: 0.35).
- **Solver**: Dijkstra's algorithm with a binary Min-Heap (`heapq`):
  - Time Complexity: O((V + E) log V). For V = 10, execution time is < 0.1 ms.
  - Returns: `{'path': [start_idx, ..., target_idx], 'avg_risk': float, 'total_progression': float}`.
- **Expected Visual Result**:
  - Automatically illuminates the optimal passing corridor in Neon Emerald on both the 2D pitch radar and the broadcast AR feed.
  - Updates Coach Tactical Card with "Line-Breaking Corridor (Penetrative)" status.

---

## 2. FW-02: Temporal Coordinate Smoothing (Eliminate Jitter)

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

## 3. FW-03: Ball Trajectory Extrapolation & Occlusion Recovery

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

## 4. FW-04: ONNX Runtime Multi-Thread Concurrency Tuning

### Problem Statement:
ONNX Runtime defaults to unconstrained CPU threading, causing thread thrashing and memory bus contention on 4-core and 8-core laptops.

### Implementation Specification:
In `perception/detector.py`:
```python
import os
import onnxruntime as ort

opts = ort.SessionOptions()
cpu_count = os.cpu_count() or 4
opts.intra_op_num_threads = max(1, min(cpu_count // 2, 4))
opts.inter_op_num_threads = 1
opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

session = ort.InferenceSession("models/best.onnx", sess_options=opts, providers=['CPUExecutionProvider'])
```
- **Expected Benchmark**: Reduces Stage 1 perception latency from ~180 ms down to ~85-95 ms on Intel/AMD CPUs.

---

## 5. FW-05: Asynchronous Frame Prefetching in Live Mode

### Problem Statement:
When scrubbing video sequentially in Live Mode, the user waits ~150 ms per frame step for sequential OpenCV frame decode and YOLO perception.

### Implementation Specification:
- Create a background worker thread in `MatchStreamManager`:
  - When the user views Frame N, the thread pre-fetches and pre-processes Frame N + 1 into a 1-frame lookahead queue.
  - Stepping to Frame N + 1 retrieves the pre-computed bundle with 0 ms perceived latency.

---

## 6. FW-06: Interactive Tactical Distance & Offside Ruler

### Problem Statement:
Analysts frequently need to measure metric distances between specific players (e.g. defensive line width, distance between center-backs, or gap between last defender and goalkeeper).

### Implementation Specification:
- Add a "Tactical Ruler" toggle in the Control Deck or Stage 2 view:
  - Click Point A and Point B on the 2D Pitch Radar or Camera view.
  - Calculates Euclidean metric distance: `d = hypot(x_B - x_A, y_B - y_A)`.
  - Computes vertical offset: `dx = x_B - x_A` (compactness depth) and lateral offset: `dy = y_B - y_A` (compactness width).
  - Renders a calibrated offside projection line across pitch width y in [0, 68] m.

---

## 7. FW-07: Velocity-Weighted Pitch Control (Fernandez-Bornn Model)

### Problem Statement:
Standard Voronoi assumes all players have equal omni-directional reach regardless of running speed. In reality, a sprinting winger controls space ahead of them, not behind them.

### Implementation Specification:
- Replace standard Voronoi Euclidean metric with time-to-intercept distance:
  ```
  Distance_effective(P, x) = EuclideanDistance(P + v_P * tau, x) / MaxSpeed
  ```
- Creates dynamic, elliptical pitch control teardrops reflecting true momentum and passing accessibility.

---

## 8. FW-08: Git Remote History Alignment & Push

### Problem Statement:
Local `main` and `origin/main` diverged because remote originated from an earlier independent root commit.

### Execution Procedure:
```bash
# 1. Fetch remote changes
git fetch origin

# 2. Rebase local commits onto origin/main
git pull --rebase origin main

# 3. Push verified commits to GitHub
git push -u origin main
```
*(Alternatively, force local state as single source of truth: `git push -u origin main --force-with-lease`).*
