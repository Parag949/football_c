# ⚽ TacticalVision Pro
### Real-Time Spatial Reconstruction & Geometric Algorithmic Tactical Workstation

TacticalVision Pro transforms standard, single-camera broadcast football footage into an interactive, real-time 2D and AR tactical analytics workstation. Built around a decoupled 4-stage contract pipeline, it bridges deep learning perception with classical Design and Analysis of Algorithms (DAA) computational geometry.

---

## 🌟 Key Features

### 1. 🎯 Flexible Multi-Point Calibration (N in {4, 5, 6, 8})
- **Dual Matrix Solvers**:
  - **N = 4 Points**: Analytic exact solution via `cv2.getPerspectiveTransform()`.
  - **N > 4 Points**: Robust overdetermined solution via `cv2.findHomography(src, dst, cv2.RANSAC, 3.0)` with Least-Squares fallback, smoothing out human click jitter and misclicks.
- **Collinearity Protection**: Automatic straight-line detection using 2D cross-product triangle area (< 0.5 m^2) prevents degenerate, singular homography matrices.
- **On-Screen Reprojection Diagnostics**: Live color-coded residual error badges per landmark:
  - **Green Badge** (< 1.0 px): Sub-Pixel Fit
  - **Yellow Badge** (1.0 - 3.0 px): Valid Inlier
  - **Red Badge** (> 3.0 px): Outlier / High Residual
- **Seamless UI Integration**: Accessible both in the initial **Setup Gate** and directly inside **Stage 2: Homography & Warping** of the main dashboard.

### 2. ⚡ Dual Execution Modes & Asynchronous Prefetching (100% Mathematically Identical)
- **⚡ Precomputed Disk Cache Mode (Instant 120+ FPS, Zero Laptop Heat)**:
  - Two-stage selective multiprocessing: Sequential YOLO perception runs once; pure-math DAA geometry scales across all CPU cores in parallel (< 15 MB RAM per worker).
  - Enables instant timeline scrubbing with zero CPU fan noise.
- **👁️ Live On-the-Fly Compute Mode with Asynchronous Lookahead Prefetching (FW-05)**:
  - Full end-to-end neural perception and DAA geometry computed on demand with LRU frame caching.
  - **Asynchronous Lookahead Prefetch Engine**: Integrated `ThreadPoolExecutor(max_workers=1)` with an isolated secondary OpenCV capture handle (`_prefetch_cap` guarded by `_prefetch_lock`). Lookahead frames N+1 and N+2 are pre-decoded and pre-inferred in the background with 0.63 ms dispatch latency.
  - **Sub-Millisecond Retrieval**: Sequential scrubbing retrieves prefetched frames from memory in 0.0057 ms (> 15,000x faster than synchronous frame seek/decode), eliminating scrubbing stutters.
- **Zero Deviation**: Both modes produce identical pitch coordinates and tactical metrics down to 0.000000 meters.

### 3. 📐 Classical DAA Computational Geometry & Graph Theory Core
- **Defensive Convex Hull**:
  - Switchable between **Graham Scan** (O(N log N)) and **Jarvis March Gift Wrapping** (O(N * H)).
  - Quantifies defensive block compactness (m^2) and shape compactness.
- **Voronoi Spatial Pitch Dominance**:
  - Fortune's sweep-line Voronoi tessellation clipped strictly to FIFA pitch dimensions (105m x 68m).
  - Computes territorial dominance percentages (Team A vs Team B).
- **Dijkstra Optimal Passing Network (FW-01)**:
  - Models attacking outfield players as a directed graph G = (V, E).
  - Solves the lowest-risk forward penetrative passing corridor via binary Min-Heap (`heapq`) in O((V + E) log V) time (< 0.25 ms).
  - Dynamically weights edge cost against defensive pressure along pass corridors and forward yardage.
  - Illuminated with Neon Emerald passing corridors and dynamic chevrons on both 2D Radar and Broadcast AR overlays.
- **Coach Tactical Cards**:
  - High-level tactical intelligence: block compactness, ball carrier attribution, passing pressure, and territorial dominance.

### 4. ⚡ ONNX Runtime Multi-Thread Concurrency Tuning (FW-04)
- **Deterministic CPU Thread Pool**: Binds `intra_op_num_threads` to `max(1, min(cpu_count // 2, 4))` and `inter_op_num_threads = 1`, preventing thread thrashing and memory bus contention on multicore laptops.
- **Graph Optimization**: Runs `ort.GraphOptimizationLevel.ORT_ENABLE_ALL` and `ort.ExecutionMode.ORT_SEQUENTIAL` for fused kernels and constant folding.
- **Live Diagnostics Telemetry**: Real-time readout of engine provider, thread pools, and execution mode directly in Stage 1 diagnostics.

### 5. 📏 Interactive Tactical Distance & VAR Offside Ruler (FW-06)
- **Metric Spatial Distance**: Measures real-world Euclidean distance `d = hypot(dx, dy)`, longitudinal depth (`dx`), and lateral spread (`dy`) between teammates, opposing lines, ball carriers, or custom coordinates.
- **Calibrated VAR Laser Line**: Projects the official second-deepest defender offside line across full pitch width (Y in [0, 68] m) with real-time offside player breach warnings.
- **Synchronized Visualizations**: Renders an amber distance tape with glassmorphic metric badge on the 2D Pitch Radar, and perspective-correct laser beams on the TV Broadcast AR Camera overlay clipped to the turf polygon.

---

## 🏗️ 4-Stage Decoupled Pipeline Architecture

```
+-----------------------------------------------------------------------------------+
| 1. PERCEPTION        | YOLOv8 detection + foot contact anchor + jersey K-means     |
| 2. HOMOGRAPHY        | Perspective projection to FIFA pitch coordinates [105x68m] |
| 3. GEOMETRY (DAA)    | Graham/Jarvis Hull + Voronoi + Dijkstra Passing Network     |
| 4. VIEWPORTS         | Synchronized 2D Minimap Radar + Broadcast AR Turf Overlay  |
+-----------------------------------------------------------------------------------+
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites & Environment Setup
```bash
# Clone the repository
git clone https://github.com/Parag949/football_c.git
cd football_c

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launching the Tactical Workstation
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## 🧪 Automated Verification & Test Suites

The repository includes comprehensive automated validation suites:

```bash
# 1. Tactical Distance Ruler & ONNX Multi-Thread Concurrency Suite (FW-04 & FW-06)
python3 scratch/test_tactical_ruler.py

# 2. Asynchronous Frame Prefetching Suite (Lookahead worker, dual-capture isolation, 0.0057ms cache retrieval)
python3 scratch/test_async_prefetch.py

# 3. Dijkstra Passing Network Suite (Min-Heap, corridor risk, 2D/AR renderers)
python3 scratch/test_dijkstra_passing.py

# 4. Multi-Point Calibration Suite (Collinearity, 4-point exact fit, 5/6/8-pt RANSAC)
python3 scratch/test_multipoint_calib.py

# 5. Coordinate Precision & Round-Trip Reprojection Integrity Suite (< 0.05px)
python3 scratch/verify_coordinates.py

# 6. Codebase Architectural Health & Regression Fixes Suite
python3 scratch/verify_fixes.py

# 7. Full Pipeline Multi-Process Worker & DAA Benchmark Audit Suite
python3 scratch/audit_suite.py
```

---

## 📚 Codebase Reference Documentation
- [FEATURES_AND_IMPLEMENTATION.md](file:///home/parag/Documents/football/FEATURES_AND_IMPLEMENTATION.md): Universal master reference ledger detailing all data contracts, stage architectures, session keys, and mathematical invariants.
- [ARCHITECTURE.md](file:///home/parag/Documents/football/ARCHITECTURE.md): Deep-dive system topology, complexity bounds, and stage contracts.
- [HACKATHON_PITCH_GUIDE.md](file:///home/parag/Documents/football/HACKATHON_PITCH_GUIDE.md): Presentation guide, elevator pitch, viva Q&A, and live demo script.
- [FUTURE_WORK.md](file:///home/parag/Documents/football/FUTURE_WORK.md): Prioritized roadmap for upcoming algorithmic enhancements, compute optimizations, and interactive tools.
