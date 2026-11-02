# ⚽ TacticalVision Pro
### Real-Time Spatial Reconstruction & Geometric Algorithmic Tactical Workstation

TacticalVision Pro transforms standard, single-camera broadcast football footage into an interactive, real-time 2D and AR tactical analytics workstation. Built around a decoupled 4-stage contract pipeline, it bridges deep learning perception with classical Design and Analysis of Algorithms (DAA) computational geometry.

---

## 🌟 Key Features

### 1. 🎯 Flexible Multi-Point Calibration ($N \in \{4, 5, 6, 8\}$)
- **Dual Matrix Solvers**:
  - **$N = 4$ Points**: Analytic exact solution via `cv2.getPerspectiveTransform()`.
  - **$N > 4$ Points**: Robust overdetermined solution via `cv2.findHomography(src, dst, cv2.RANSAC, 3.0)` with Least-Squares fallback, smoothing out human click jitter and misclicks.
- **Collinearity Protection**: Automatic straight-line detection using 2D cross-product triangle area ($< 0.5\,\text{m}^2$) prevents degenerate, singular homography matrices.
- **On-Screen Reprojection Diagnostics**: Live color-coded residual error badges per landmark:
  - **Green Badge** ($< 1.0\,\text{px}$): Sub-Pixel Fit
  - **Yellow Badge** ($1.0 - 3.0\,\text{px}$): Valid Inlier
  - **Red Badge** ($> 3.0\,\text{px}$): Outlier / High Residual
- **Seamless UI Integration**: Accessible both in the initial **Setup Gate** and directly inside **Stage 2: Homography & Warping** of the main dashboard.

### 2. ⚡ Dual Execution Modes (100% Mathematically Identical)
- **⚡ Precomputed Disk Cache Mode (Instant 120+ FPS, Zero Laptop Heat)**:
  - Two-stage selective multiprocessing: Sequential YOLO perception runs once; pure-math DAA geometry scales across all CPU cores in parallel ($< 15\,\text{MB}$ RAM per worker).
  - Enables instant timeline scrubbing with zero CPU fan noise.
- **👁️ Live On-the-Fly Compute Mode**:
  - Full end-to-end neural perception and DAA geometry computed on demand with LRU frame caching.
  - Ideal for testing live camera calibrations and instant frame-by-frame probing.
- **Zero Deviation**: Both modes produce identical pitch coordinates and tactical metrics down to $0.000000\,\text{meters}$.

### 3. 📐 Classical DAA Computational Geometry Core
- **Defensive Convex Hull**:
  - Switchable between **Graham Scan** ($O(N \log N)$) and **Jarvis March Gift Wrapping** ($O(N \cdot H)$).
  - Quantifies defensive block compactness ($m^2$) and shape compactness.
- **Voronoi Spatial Pitch Dominance**:
  - Fortune's sweep-line Voronoi tessellation clipped strictly to FIFA pitch dimensions ($105\,\text{m} \times 68\,\text{m}$).
  - Computes territorial dominance percentages (Team A vs Team B).
- **Coach Tactical Cards**:
  - High-level tactical intelligence: block compactness, ball carrier attribution, passing pressure, and territorial dominance.

---

## 🏗️ 4-Stage Decoupled Pipeline Architecture

```
+-----------------------------------------------------------------------------------+
| 1. PERCEPTION        | YOLOv8 detection + foot contact anchor + jersey K-means     |
| 2. HOMOGRAPHY        | Perspective projection to FIFA pitch coordinates [105x68m] |
| 3. GEOMETRY (DAA)    | Graham Scan / Jarvis March Convex Hull + Voronoi dominance  |
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
# 1. Multi-Point Calibration Suite (Collinearity, 4-point exact fit, 5/6/8-pt RANSAC)
python3 scratch/test_multipoint_calib.py

# 2. Coordinate Precision & Round-Trip Reprojection Integrity Suite (< 0.05px)
python3 scratch/verify_coordinates.py

# 3. Codebase Architectural Health & Regression Fixes Suite
python3 scratch/verify_fixes.py

# 4. Full Pipeline Multi-Process Worker & DAA Benchmark Audit Suite
python3 scratch/audit_suite.py
```

---

## 📚 Codebase Reference Documentation
- [FEATURES_AND_IMPLEMENTATION.md](file:///home/parag/Documents/football/FEATURES_AND_IMPLEMENTATION.md): Universal master reference ledger detailing all data contracts, stage architectures, session keys, and mathematical invariants.
- [ARCHITECTURE.md](file:///home/parag/Documents/football/ARCHITECTURE.md): Deep-dive system topology, complexity bounds, and stage contracts.
- [HACKATHON_PITCH_GUIDE.md](file:///home/parag/Documents/football/HACKATHON_PITCH_GUIDE.md): Presentation guide, elevator pitch, viva Q&A, and live demo script.
