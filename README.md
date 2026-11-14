# TacticalVision Pro

TacticalVision Pro is a football tactical analysis workstation built with Python and Streamlit. It takes standard single-camera match footage, detects players and the ball with YOLOv8, projects them onto a 105m x 68m pitch using planar homography, and runs classical computational geometry and graph algorithms (Convex Hull, Voronoi partitions, Dijkstra passing networks) for tactical analysis.

---

## Why This Project?

Professional optical tracking systems (like TRACAB or Second Spectrum) rely on expensive multi-camera stadium setups. This project explores how much tactical depth you can extract from a **single standard video feed** using computer vision and Design and Analysis of Algorithms (DAA) fundamentals.

Instead of treating computer vision as a black box that just draws bounding boxes, this system:
1. Projects image pixel coordinates `(u, v)` into metric pitch coordinates `(x, y)` in meters.
2. Separates players into teams using jersey color clustering in CIELAB color space.
3. Applies classical algorithms to analyze team structure: defensive block compactness, pitch control percentage, and forward passing corridors.
4. Provides an interactive 2D whiteboard where coaches can tinker with player positioning, test what-if scenarios, and evaluate offside lines.

---

## Core Features

### 1. Computer Vision & Metric Projection
- **YOLOv8 Detection**: Detects players, goalkeepers, referees, and the ball using an ONNX Runtime backend.
- **Foot-Contact Ground Anchoring**: Uses bottom-center bounding box points rather than object centroids so player pitch coordinates accurately reflect where the player's feet touch the turf.
- **Jersey Color Clustering (CIELAB Space)**: Samples the player's upper torso with an HSV green-pitch mask to eliminate turf bleed, then clusters colors in CIELAB space to distinguish Team A vs Team B. Referees are filtered out, and goalkeepers are assigned to their defending goal based on team pitch centroids.
- **Planar Homography Calibration**:
  - Maps camera perspective pixels to standard FIFA pitch dimensions (105m x 68m).
  - Supports 4-point exact perspective transform or multi-point (5, 6, 8 points) RANSAC fitting to handle manual landmark jitter.
  - Collinearity guard rejects degenerate point selections (triangle area < 0.5 m²).
  - Inverse homography (`H_inv`) re-projects tactical graphics back onto the broadcast camera view.

### 2. Classical DAA Algorithms (custom implementations in `/algorithms` without external spatial wrappers)
- **Defensive Convex Hull (Block Compactness)**:
  - Supports both **Graham Scan** (Andrew's Monotone Chain, `O(N log N)`) and **Jarvis March** (`O(N * H)`).
  - Computes total defensive area (m²) and isoperimetric compactness ratio to measure how tightly a team compresses space.
- **Voronoi Spatial Dominance (Pitch Control)**:
  - Computes Voronoi polygons bounded strictly within the 105m x 68m pitch using Sutherland-Hodgman halfplane clipping.
  - Uses the Shoelace formula to measure exact territorial control percentages for each team.
- **Dijkstra Optimal Passing Network**:
  - Builds a weighted directed graph `G = (V, E)` connecting attacking players.
  - Edge weights account for physical distance, forward progression, and opponent defensive pressure along the passing lane.
  - Solves the lowest-risk forward path using a binary min-heap (`heapq`) in `O((V + E) log V)` time.
- **Ghost Runner What-If Simulation**:
  - Allows placing or dragging a virtual attacker anywhere on the pitch.
  - Evaluates offside status against the opponent's second-to-last defender and ball position (strict FIFA Law 11 compliance).
  - Evaluates progressive corridors and checks if Dijkstra passing paths re-route to exploit open space.
- **Tactical Distance Ruler**:
  - Measures Euclidean distance, longitudinal depth (ΔX), lateral width (ΔY), and pass angle in degrees between any two points.

---

## The Three Workstation Modes

The application provides three operating modes in the top navigation:

### 1. ⚡ Hardware Media Engine (60 FPS Playback)
- **Problem**: Running YOLOv8, jersey clustering, and geometric math in pure Python on every frame caps video playback at 8–15 FPS on laptop CPUs and causes Streamlit UI freezes.
- **Solution**: The video is decoded directly by the browser's HTML5 engine using GPU hardware acceleration (AMD VCN, Intel QuickSync, NVIDIA NVDEC). Precomputed frame telemetry is stored in a lightweight JSON payload.
- **Synchronization**: Uses the browser's `requestVideoFrameCallback` API to synchronize canvas overlays with video presentation timestamps in under 1 ms per frame, delivering smooth 60 FPS playback with zero Python compute overhead.
- Features dual viewports: a Broadcast AR overlay and a synchronized 2D tactical radar.

### 2. 🔬 Deep Pipeline Diagnostics (5-Stage DAA Inspector)
- A frame-by-frame diagnostic tool for developers and analysts to inspect intermediate data across every pipeline stage:
  - **Stage 1**: Perception & Bounding Boxes (YOLO detections, class filters, jersey clustering).
  - **Stage 2**: Homography Calibration & Warping (matrix coefficients, landmark residuals, bird's-eye pitch warp).
  - **Stage 3**: DAA Tactical Geometry (convex hull polygon vertices, Voronoi cells, Dijkstra path weights).
  - **Stage 4a**: 2D Tactical Radar rendering.
  - **Stage 4b**: Broadcast AR Perspective overlay rendering.
- Includes interactive multi-point calibration studio, residual error badges, and live algorithm performance benchmarks.

### 3. 🎯 2D Tactical Simulation Sandbox (Coach's Whiteboard)
- A standalone interactive coaching whiteboard that runs completely in the browser without interfering with video playback.
- **Drag-and-Drop Entities**: Drag players, the match ball, ghost runner, or ruler pins freely across the pitch. Click any teammate to pass the ball and assign them as the ball carrier.
- **Formation Presets**: One-click loading of standard structures: 4-3-3 Attack, 4-4-2 Mid-Block, 3-5-2 High Press, Corner Overload.
- **Live Match Frame Import**: Snapshot actual player coordinates from the current video frame into the whiteboard.
- **Real-Time Client Geometry**: Voronoi dominance, convex hull, Dijkstra lanes, and offside checks recompute instantly (under 2 ms) on every mouse drag.
- **Export PNG**: One-click export of tactical setups directly as image files for coaching presentations.

---

## 📐 Interactive Pitch Mask Studio & Camera Regimes

TacticalVision includes an interactive, draggable pitch chalk mask studio overlaid directly over match footage. It automatically adapts to arbitrary pitch dimensions ($L \in [90, 120]\text{m}$, $W \in [45, 90]\text{m}$ while strictly locking FIFA IFAB Law 1 penalty box dimensions) and provides three specialized camera calibration regimes:

| Mode | When to Choose | Pitch Shape on Screen |
| :--- | :--- | :--- |
| **`Normal Broadcast`** | Standard TV angle (16:9, single camera in the stands with tilt & pan). | **Trapezoid / Keystone**: Near edge is wide, far edge is narrow. Pitch lines stay straight. |
| **`Wide-Angle Lens`** | Fisheye or wide-angle tactical camera. | **Curved Trapezoid**: Near edge is wide, but straight lines bend into arcs (barrel distortion). |
| **`Panorama Multi-Cam`** | 2 to 4 cameras stitched together ($> 2:1$ or $4:1$ ultra-wide video). | **Wavy / S-curve / Cylindrical**: Multiple vanishing points because of camera stitching seams. |

### Understanding the Broadcast Trapezoid
In a standard TV broadcast, the camera is positioned high in the stadium stands, tilted downwards ($15^\circ - 30^\circ$). Because the near touchline is physically closer to the lens than the far touchline, perspective foreshortening makes the near touchline appear significantly wider on screen than the far touchline, forming a classic **keystone trapezoid**:

```text
Far sideline (far from camera):    TR •───────────────────────────• TL  (narrower pixel width)
                                       \                         /
                                        \                       /
Near sideline (close to camera):   BR   •───────────────────────•   BL  (wider pixel width)
```

- In **`Normal Broadcast`** mode, standard perspective homography models this elevation tilt, keeping pitch lines straight.
- In **`Wide-Angle Lens`** mode, an adjustable optical lens parameter $\lambda \in [-0.30, +0.10]$ straightens barrel curvature.
- In **`Panorama Multi-Cam`** mode, a 2D Thin-Plate Spline (TPS) solver mathematically pins mask polylines directly through the anchor handles with **$0.00\text{ px}$ residual**, completely eliminating anchor-to-mask drift.

### 👁️ Panoramic Perception & Calibration-Guided ROI (CPU-Optimized)
On ultra-wide panoramic footage ($3.8:1$), feeding full frames into standard $640 \times 640$ YOLO crushes distant players into $12 - 15\text{ px}$ tall silhouettes, causing missed bounding boxes.

* **Industry Gold Standard**: Enterprise servers (e.g. Spiideo) run **Slicing Aided Hyper Inference (SAHI)**, splitting frames into 3–4 overlapping tiles. However, running $3\times$ YOLO passes per frame drops a laptop CPU to $< 4\text{ FPS}$.
* **Our Lightweight CPU Solution**: TacticalVision extracts the bounding box of the active **calibration pitch mask** (`transformer.get_pitch_roi`), automatically cropping out empty sky and concrete embankments before inference. This **boosts effective player resolution by nearly $40\%$** with **$1\times$ inference overhead (zero extra CPU passes)**, paired with adaptive turf-zone confidence ($0.18$ on-pitch vs $0.35$ off-pitch) and human aspect ratio filtering ($1.3 \le H/W \le 5.0$) to capture every player without false positives.

---

## System Architecture

```text
+-----------------------------------------------------------------------------------+
| 1. VIDEO INGESTION    | OpenCV stream manager with prefetch cache                 |
| 2. PERCEPTION         | YOLOv8 detection + foot anchor + CIELAB jersey clustering |
| 3. HOMOGRAPHY         | Planar projection from camera (u,v) to pitch (X,Y) in m   |
| 4. DAA GEOMETRY       | Graham/Jarvis Hull + Bounded Voronoi + Dijkstra Network   |
| 5. CLIENT VIEWPORTS   | Browser Media Engine (60 FPS) + Interactive Sandbox       |
+-----------------------------------------------------------------------------------+
```

### Repository Layout
```text
football_c/
├── app.py                         # Main Streamlit application entry point
├── algorithms/
│   ├── convex_hull.py             # Graham scan and Jarvis march implementations
│   ├── voronoi_dominance.py       # Bounded Voronoi and pitch control calculation
│   ├── passing_network.py         # Dijkstra shortest passing path and graph edges
│   ├── ghost_runner.py            # What-if off-ball runner simulation & Law 11
│   ├── tactical_ruler.py          # Spatial distance, coordinate offsets, and angles
│   └── coach_analytics.py         # Tactical metrics aggregation
├── core/
│   ├── contracts.py               # Typed dataclass contracts between pipeline stages
│   └── pipeline.py                # 4-stage decoupled processing pipeline
├── perception/
│   ├── detector.py                # YOLOv8 inference wrapper (ONNX Runtime)
│   ├── homography.py              # Camera-to-pitch homography matrix & calibration
│   ├── pitch_model.py             # FIFA standard pitch dimensions (105m x 68m)
│   ├── precompute.py              # Offline video processing & JSON cache generator
│   └── video_stream.py            # OpenCV video capture and frame caching
├── ui/
│   ├── components/
│   │   ├── hardware_player.py     # Browser-based 60 FPS video & canvas player
│   │   ├── tactical_sandbox.py    # Interactive 2D coaching simulation whiteboard
│   │   ├── setup_gate.py          # Initial calibration and source video selection
│   │   └── control_deck.py        # Algorithmic parameter controls
│   └── views/
│       └── diagnostics_view.py    # 5-stage pipeline diagnostics interface
├── data/
│   ├── sample_broadcast.mp4       # Sample broadcast camera footage
│   ├── user_match.mp4             # Sample tactical wide-angle footage
│   └── precomputed_user.json      # Precomputed telemetry cache for user_match.mp4
└── scratch/                       # Test suites and regression verification scripts
```

---

## Quickstart

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12
- Linux, macOS, or Windows
- A modern web browser (Chrome, Firefox, Edge, Safari)

### 2. Setup
```bash
# Clone the repository
git clone https://github.com/Parag949/football_c.git
cd football_c

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Running the App
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

- Select a sample match (`user_match.mp4` or `sample_broadcast.mp4`) or upload your own video.
- Choose between:
  - **Hardware Media Engine** for smooth 60 FPS playback with tactical overlays.
  - **Deep Pipeline Diagnostics** to inspect individual pipeline stages.
  - **2D Tactical Sandbox** to freely position players and test formations.

---

## Automated Test Suites

The project includes unit tests and an end-to-end audit suite to verify all pipeline stages and algorithms:

```bash
# Run the complete end-to-end project audit (all 7 stages + 10 test suites)
python3 scratch/audit_suite.py

# Run individual component test suites:
python3 scratch/test_tactical_sandbox.py   # Tactical Sandbox & Voronoi area conservation
python3 scratch/test_dijkstra_passing.py   # Dijkstra passing graph and min-heap routes
python3 scratch/test_ghost_runner.py       # Ghost runner and FIFA Law 11 offside checks
python3 scratch/test_multipoint_calib.py   # Homography solvers and collinearity protection
python3 scratch/test_tactical_ruler.py     # Distance ruler geometry calculations
python3 scratch/verify_coordinates.py      # Sub-pixel reprojection and round-trip precision
```

---

## Engineering Trade-offs & Limitations

Being clear about what this system can and cannot do:

1. **Planar Homography Assumption**:
   The projection assumes the football pitch is a flat 2D plane (`Z = 0`). While accurate for ground-level foot contact, balls or players in the air will appear slightly displaced due to parallax.
2. **Camera Motion**:
   The single static homography matrix works best on fixed tactical or wide-angle cameras (like `user_match.mp4`). For broadcast cameras that pan, tilt, or zoom across the pitch (like `sample_broadcast.mp4`), a single matrix causes coordinate drift over time. Handling panning cameras requires dynamic inter-frame keypoint tracking or recalibration.
3. **Precompute vs. Live Mode**:
   Live on-the-fly compute runs YOLO and full pipeline stages on each frame. On standard laptop CPUs without a discrete GPU, live mode runs at ~10 FPS. Precomputing the cache once lets the browser play footage at 60 FPS with zero lag.

---

## Detailed References

For in-depth architectural notes, mathematical derivations, and project design details:
- [ARCHITECTURE.md](./ARCHITECTURE.md): Mathematical formulations, stage contracts, and pipeline topology.
- [FEATURES_AND_IMPLEMENTATION.md](./FEATURES_AND_IMPLEMENTATION.md): Complete ledger of stage contracts, data schemas, and edge case fixes.
- [HACKATHON_PITCH_GUIDE.md](./HACKATHON_PITCH_GUIDE.md): Project presentation notes and technical Q&A preparation.
- [FUTURE_WORK.md](./FUTURE_WORK.md): Planned roadmap items including dynamic homography tracking for panning cameras.
