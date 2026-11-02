# 🔮 TacticalVision Pro — Future Work & Roadmap
### Prioritized Technical Specifications, Algorithmic Enhancements & Optimizations

This document serves as the persistent architectural roadmap for future enhancements, algorithmic extensions, and compute optimizations. Any engineer or LLM can refer to this file to implement these prioritized features without having to re-architect or guess system contracts.

---

## 📋 Prioritized Feature Matrix

| ID | Feature Name | Category | Primary Target Files | Status / Estimated Impact |
| :--- | :--- | :--- | :--- | :--- |
| **FW-01** | **Dijkstra Passing Network** | DAA / Graph Theory | `algorithms/passing_network.py`, `core/pipeline.py` | ✅ COMPLETED & OPERATIONAL (v1.2) |
| **FW-02** | **Temporal Coordinate Smoothing** | Kinematics / Perception | `core/pipeline.py`, `ui/state.py` | 🌟 High (Broadcast Fluidity) |
| **FW-03** | **Ball Trajectory Extrapolation** | Physics / Perception | `core/pipeline.py`, `perception/detector.py` | ⚡ Medium (Pass Tracking) |
| **FW-04** | **ONNX Multi-Thread Tuning** | Compute / Latency | `perception/detector.py` | ⚡ Medium (50% CPU Speedup) |
| **FW-05** | **Async Frame Prefetching** | Concurrency / UX | `perception/video_stream.py`, `ui/state.py` | ✅ COMPLETED & OPERATIONAL (v1.3) |
| **FW-06** | **Interactive Tactical Ruler** | Analytics Tool | `visualization/pitch_renderer.py`, `ui/` | 💡 Feature (Coach Analytics) |
| **FW-07** | **Velocity-Weighted Pitch Control**| Geometry / Spatial | `algorithms/voronoi_dominance.py` | 💡 Advanced Tactical Model |
| **FW-08** | **Git Remote History Alignment** | DevOps / Versioning | Git CLI (`git pull --rebase`) | 🔧 Operational |

---

## 1. FW-01: Dijkstra Optimal Passing Network (Graph Theory) [COMPLETED & OPERATIONAL]

### Implementation Status:
- **Status**: FULLY IMPLEMENTED & OPERATIONAL (Validated by `scratch/test_dijkstra_passing.py`).
- **Core Engine**: `algorithms/passing_network.py`
- **Data Contract**: Added `dijkstra_res: Optional[Dict[str, Any]] = None` to `TacticalGeometry` in `core/contracts.py`.
- **Pipeline Integration**: Stages 3 and 4 in `core/pipeline.py`, multi-core precomputing in `perception/precompute.py`, and live benchmark suite in `algorithms/daa_benchmarks.py`.

### Mathematical & Algorithmic Formulation:
- **Graph Construction**: Directed complete graph G = (V, E) where:
  - Vertices V: Attacking outfield players (Team A, |V| <= 10).
  - Directed Edge (u, v): Possible pass corridor from player u to player v.
- **Edge Weight Function**:
  ```
  Weight(u, v) = max(0.1, [d * (1.0 + 2.5 * DefensivePressure(u, v))] - alpha * ForwardProgression(u, v) + 10.0)
  ```
  Where:
  - `ForwardProgression(u, v) = x_v - x_u` (positive towards opponent's goal at x = 105m).
  - `DefensivePressure(u, v)` = Inverse perpendicular distance to nearest defender within a 4.0m pass corridor radius.
  - Non-negative guarantee: Adding `+ 10.0` and `max(0.1, ...)` strictly guarantees edge weights >= 0.1, preserving Dijkstra's greedy correctness invariant.
- **Solver**: Dijkstra's algorithm with a binary Min-Heap (`heapq`):
  - Time Complexity: O((V + E) log V). For V = 10, execution latency is 0.25 ms.
  - Returns: `{'path': [start_idx, ..., target_idx], 'path_points': [...], 'total_weight': float, 'avg_risk': float, 'total_progression': float, 'edge_count': int, 'target_idx': int}`.
- **Visual & UI Integration**:
  - Automatically illuminates optimal passing corridors in Neon Emerald on both the 2D pitch radar canvas (with animated forward chevrons) and the broadcast AR perspective turf overlay.
  - Integrated into Coach Tactical Card producing penetrative line-breaking directives.
  - Interactive live comparison benchmark in Stage 5 Diagnostics View comparing Dijkstra Min-Heap vs Greedy search.

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

## 5. FW-05: Asynchronous Frame Prefetching in Live Mode [COMPLETED & OPERATIONAL]

### Problem Statement:
When scrubbing video sequentially in Live Mode, users experienced ~80-150 ms latency per frame step due to sequential OpenCV frame decoding and YOLO perception inference on the main thread.

### Implementation Details:
- **Dedicated Lookahead Worker**: Integrated `ThreadPoolExecutor(max_workers=1)` in [video_stream.py](file:///home/parag/Documents/football/perception/video_stream.py) to manage non-blocking lookahead decoding and inference.
- **Dual Capture Handle Architecture**: Allocated an isolated `_prefetch_cap` handle with `_prefetch_lock` exclusively for background workers. This eliminates race conditions, seek collisions, and FFmpeg thread deadlocks with the primary UI playback stream (`current_cap`).
- **Memory Buffer Caches**:
  - `_prefetch_raw_cache`: Thread-safe dictionary storing prefetched raw BGR frames.
  - `_prefetch_det_cache`: Thread-safe dictionary caching prefetched YOLO detection bundles.
- **Fast-Path UI Integration**:
  - In [state.py](file:///home/parag/Documents/football/ui/state.py), `get_raw_frame(frame_idx)` checks `stream.get_cached_raw_frame(frame_idx)` before falling back to disk/capture seek.
  - `execute_pipeline()` consults `stream.get_cached_detections(frame_idx)`, bypassing YOLO inference on hits, and schedules background prefetching for `frame_idx + 1` and `frame_idx + 2`.
- **Empirical Latency Benchmarks**:
  - Worker dispatch latency: 0.63 ms (strictly non-blocking).
  - Prefetched frame retrieval: 0.0057 ms (< 1.0 ms requirement, representing > 15,000x speedup over synchronous seek & decode).
- **Automated Verification**:
  - Verified via [test_async_prefetch.py](file:///home/parag/Documents/football/scratch/test_async_prefetch.py). Run:
  ```bash
  python3 scratch/test_async_prefetch.py
  ```

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
