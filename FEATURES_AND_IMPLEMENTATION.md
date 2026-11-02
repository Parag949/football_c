# FEATURES AND IMPLEMENTATION MAP (LLM REFERENCE LEDGER)
================================================================================
LAST UPDATED: November 2026
STATUS: All 12 bugs fixed | All 5 coordinate stages aligned (< 0.05px error)
PURPOSE: This document provides an exhaustive, single-source-of-truth reference
for any AI coding assistant or developer. Read this file first instead of
scanning the entire codebase.
================================================================================

--------------------------------------------------------------------------------
TABLE OF CONTENTS
--------------------------------------------------------------------------------
1. System Overview & Technology Stack
2. Complete Directory & File Architecture
3. Data Contracts & Type Schemas (core/contracts.py)
4. Pipeline Stages & Execution Flow (core/pipeline.py)
5. Perception & Homography Engine (perception/)
6. DAA Computational Geometry Core (algorithms/)
7. Visualization & Renderers (visualization/)
8. Streamlit UI, State & Viewports (ui/ & app.py)
9. Precomputation & Multiprocessing Engine (perception/precompute.py)
10. Coordinate Systems & Mathematical Conventions
11. Verification Suites & Test Commands (scratch/)
12. Critical Invariants & Rules (DO NOT BREAK)
13. Upcoming Planned Feature: N-Point Flexible Calibration

================================================================================
1. SYSTEM OVERVIEW & TECHNOLOGY STACK
================================================================================
TacticalVision Pro is a real-time football AI analytics and tactical radar
studio that reconstructs the tactical state of a match directly from single-
camera television broadcast footage without in-stadium sensors or GPS vests.

Core Tech Stack:
- Python 3.10+
- PyTorch & Ultralytics YOLOv8 (Player & ball bounding box detection)
- OpenCV (cv2) (Planar homography, image drawing, video decoding)
- SciPy (scipy.spatial.Voronoi for territorial partitioning)
- NumPy (Vectorized coordinate transformations)
- Streamlit (Web dashboard, dynamic viewports, interactive calibration studio)

Key Performance Metrics:
- DAA Pure-Math Geometry Latency: < 2.5 ms per frame (Graham Scan + Voronoi)
- Homography Reprojection Error: < 0.05 px round-trip (Image <-> Pitch)
- Playback Frame Rate: 60+ FPS on cached/precomputed streams, ~5 FPS live CPU

================================================================================
2. COMPLETE DIRECTORY & FILE ARCHITECTURE
================================================================================
football/
│
├── app.py                         # Application entrypoint & multi-viewport studio router
├── requirements.txt               # Dependencies (torch, ultralytics, opencv, streamlit, scipy)
├── README.md                      # Public project documentation & quickstart
├── ARCHITECTURE.md                # Mathematical formulations & DAA complexity specifications
├── HACKATHON_PITCH_GUIDE.md       # 30s pitch, 2-min demo script, evaluator viva Q&A
├── FEATURES_AND_IMPLEMENTATION.md # THIS FILE: Exhaustive LLM reference ledger
│
├── core/
│   ├── __init__.py
│   ├── contracts.py               # Typed dataclasses for inter-stage data contracts
│   └── pipeline.py                # Decoupled 4-stage pipeline functions & CLI stage runner
│
├── perception/
│   ├── __init__.py
│   ├── pitch_model.py             # FIFA standard 105m x 68m pitch metric specifications & keypoints
│   ├── detector.py                # FootballDetector: YOLOv8 inference & K-Means++ kit clustering
│   ├── homography.py              # HomographyTransformer: 3x3 planar perspective projection matrix H
│   ├── calibration_templates.py   # Landmark catalog (FIFA_CHALK_LANDMARKS), 2D guide, mouse studio
│   ├── precompute.py              # Multiprocessing batch precomputation engine (ProcessPoolExecutor)
│   └── video_stream.py            # MatchStreamManager: OpenCV video stream & LRU frame cache
│
├── algorithms/
│   ├── __init__.py
│   ├── convex_hull.py             # Defensive compactness: Graham Scan O(N log N) & Jarvis March O(NH)
│   ├── voronoi_dominance.py       # Spatial dominance: Bounded Voronoi & Sutherland-Hodgman clipping
│   ├── coach_analytics.py         # Tactical insights: High/Mid/Low block & territorial control
│   └── daa_benchmarks.py          # Empirical latency & operation benchmark generation
│
├── visualization/
│   ├── __init__.py
│   ├── pitch_renderer.py          # TacticalPitchRenderer: 2D minimap radar canvas (1120x730 BGR)
│   └── overlays.py                # BroadcastOverlayEngine: AR video graphics & broadcast HUD header
│
├── ui/
│   ├── __init__.py
│   ├── theme.py                   # Obsidian cyber-tactical CSS tokens & styling injection
│   ├── state.py                   # Session state initialization, frame pipeline execution, cache loader
│   ├── components/
│   │   ├── __init__.py
│   │   ├── header.py              # Top cyber-tactical header bar & telemetry status
│   │   ├── control_deck.py        # Frame scrubbing controls (-10, -1, play, +1, +10) & sliders
│   │   ├── setup_gate.py          # Initial configuration modal (camera preset / landmark studio)
│   │   └── coach_cards.py         # Tactical telemetry cards (compactness, block line, dominance)
│   └── views/
│       ├── __init__.py
│       └── diagnostics_view.py    # 6-tab pipeline debugger, coordinate probe, and DAA benchmark view
│
├── data/
│   ├── sample_broadcast.mp4       # Premier League footage (960x540, sideline broadcast view)
│   ├── precomputed_broadcast.json # Precomputed cached frames for sample_broadcast.mp4
│   ├── user_match.mp4             # Tactical footage (1920x1080, goal-to-goal end-zone view)
│   └── precomputed_user.json      # Precomputed cached frames for user_match.mp4
│
├── models/
│   ├── yolov8n.pt                 # PyTorch YOLOv8 Nano weights
│   ├── yolov8n.onnx               # ONNX export for optimized CPU inference
│   ├── best.pt                    # Fine-tuned football detector weights (optional)
│   └── best.onnx                  # Fine-tuned ONNX weights (optional)
│
└── scratch/
    ├── verify_coordinates.py      # Automated 5-point coordinate precision & stage consistency test
    ├── verify_fixes.py            # Automated 12-point system regression verification test
    └── audit_suite.py             # 6-point architectural & data contract audit suite

================================================================================
3. DATA CONTRACTS & TYPE SCHEMAS (core/contracts.py)
================================================================================
Every pipeline stage produces and consumes standardized typed dataclasses:

1. PlayerDetection:
   - box: Tuple[int, int, int, int]           # [x1, y1, x2, y2] bounding box in pixels
   - foot_point: Tuple[float, float]          # Ground contact: ((x1+x2)/2.0, float(y2))
   - team_id: int                             # 0: Team A (Sky Blue), 1: Team B (Crimson Red)
   - confidence: float                        # YOLO detection confidence (0.0 to 1.0)
   - class_id: int = 0                        # 0: player, 32: ball
   - jersey_color: Optional[Tuple[int,int,int]]# RGB kit color sampled from upper torso

2. FrameDetections (Stage 1 Output):
   - frame_idx: int                           # Video frame index
   - players: List[PlayerDetection]           # All detected players
   - ball_pixel: Optional[Tuple[float, float]]# Ball pixel coordinates (u, v)
   - frame_width: int                         # Video frame pixel width
   - frame_height: int                        # Video frame pixel height
   - Properties: .team_a_pixels, .team_b_pixels (returns List of (u, v) foot points)

3. PitchPositions (Stage 2 Output):
   - frame_idx: int
   - team_a: List[Tuple[float, float]]        # Metric positions (X, Y) in meters on [0,105]x[0,68]
   - team_b: List[Tuple[float, float]]        # Metric positions (X, Y) in meters
   - ball: Tuple[float, float]                # Ball metric position (default: 52.5, 34.0)
   - carrier_idx: int                         # Index of closest Team A player to ball
   - detected_count: int                      # Total outfield players projected

4. TacticalGeometry (Stage 3 Output):
   - frame_idx: int
   - hull_polygon: List[Tuple[float, float]]  # Metric perimeter vertices of defending block
   - hull_player_indices: List[int]           # 1-to-1 map linking each hull vertex to positions.team_b
   - hull_area: float                         # Defensive footprint in m^2 (Shoelace formula)
   - hull_compactness: float                  # Isoperimetric quotient 4*pi*Area / Perimeter^2
   - hull_vertices: int                       # Number of vertices on the hull perimeter
   - voronoi_cells_a: List[Dict]              # Team A polygonal cells, generator sites, and areas
   - voronoi_cells_b: List[Dict]              # Team B polygonal cells, generator sites, and areas
   - team_a_pct: float                        # Territory pitch dominance percentage
   - team_b_pct: float                        # Territory pitch dominance percentage
   - latency_ms: float                        # Asymptotic computation time in milliseconds

5. FrameBundle (Unified Container):
   - detections: Optional[FrameDetections]
   - positions: PitchPositions
   - geometry: TacticalGeometry
   - raw_frame: Optional[np.ndarray]
   - latencies: Dict[str, float]              # Per-stage latency breakdown

================================================================================
4. PIPELINE STAGES & EXECUTION FLOW (core/pipeline.py)
================================================================================
The execution pipeline is split into 4 independent, decoupled functions:

- stage1_perception(frame, detector, frame_idx=0) -> FrameDetections:
  Runs YOLOv8, computes foot contact ((x1+x2)/2, y2), samples upper torso patch,
  runs K-Means++ clustering to assign team_id (0 vs 1).

- stage2_projection(detections, transformer) -> PitchPositions:
  Updates transformer resolution. Projects team_a_pixels and team_b_pixels using
  image_to_pitch(filter_out_of_bounds=True, buffer_meters=2.0). Projects ball
  without clamping (filter_out_of_bounds=False). Finds closest player to ball.

- stage3_geometry(positions, hull_algo='graham') -> TacticalGeometry:
  Runs Graham Scan (or Jarvis March) on positions.team_b. Maps hull vertices back
  to team_b indices via coordinate proximity (< 1e-4 m). Computes bounded
  Voronoi tessellation on combined positions.team_a and team_b.

- stage4_render_radar(positions, geometry, renderer, show_voronoi, show_hull) -> np.ndarray:
  Renders flat 2D pitch canvas (1120x730), shaded Voronoi cells, defensive hull,
  and illuminated player nodes.

- stage4_render_overlay(raw_frame, positions, geometry, overlay_engine, ...) -> np.ndarray:
  Projects Voronoi cells and convex hull onto the raw video using H_inv, masked
  by the pitch turf boundary, and draws the broadcast telemetry HUD.

- run_pipeline_frame(...) -> Tuple[overlay, radar, bundle]:
  Executes all 4 stages end-to-end and returns the composite bundle.

- evaluate_pipeline_health(...) -> Tuple[diag_status, diag_issues]:
  Decoupled health audit checking player count, angle mismatch, and geometry closure.

================================================================================
5. PERCEPTION & HOMOGRAPHY ENGINE (perception/)
================================================================================
1. perception/pitch_model.py (PitchModel):
   - Standard FIFA pitch dimensions: LENGTH = 105.0 meters, WIDTH = 68.0 meters.
   - KEYPOINTS dictionary contains exact metric coordinates of:
     top/bottom corner flags, penalty box corners, 6-yard boxes, penalty spots,
     center spot (52.5, 34.0), and goal mouths.
   - metric_to_pixel(x, y, w, h, pad) and pixel_to_metric(px, py, w, h, pad).

2. perception/homography.py (HomographyTransformer):
   - Maintains 3x3 matrix H and inverse H_inv.
   - Presets: 'sideline' (broadcast cam) and 'goal_to_goal' (end-zone cam).
   - calibrate(src_points, dst_points, frame_width, frame_height):
     Computes H using cv2.getPerspectiveTransform.
   - image_to_pitch(image_points, filter_out_of_bounds=False, buffer_meters=2.0):
     Transforms (u, v) -> (X, Y). When filter_out_of_bounds=False, returns raw
     unclamped coordinates (zero round-trip reprojection error). When True,
     discards points outside [-buffer, 105+buffer] x [-buffer, 68+buffer].
   - pitch_to_image(pitch_points, filter_behind_camera=False):
     Transforms (X, Y) -> (u, v) using H_inv. Discards points with projective
     scale w <= 1e-4 behind camera horizon.
   - _normalize_projective_sign(): Ensures pitch center has positive depth (w > 0).

3. perception/calibration_templates.py:
   - FIFA_CHALK_LANDMARKS: Comprehensive catalog of 20+ FIFA ground-truth marks.
   - GROUND_TRUTH_TEMPLATES: Packaged templates (18-yard box, midfield, 4 corners).
   - render_landmarks_pitch_thumbnail(): 2D guide thumbnail with numbered pins L1-L4.
   - render_mouse_calibration_studio(): Interactive clicking canvas with dynamic
     aspect ratio scaling and instant projection verification overlay.

4. perception/detector.py (FootballDetector):
   - Loads models/yolov8n.onnx (or .pt) using ONNXRuntime / Ultralytics.
   - detect_frame(frame): Extracts boxes for person (class 0) and ball (class 32).
   - K-Means++ clustering with cv2.KMEANS_PP_CENTERS for deterministic kit colors.

================================================================================
6. DAA COMPUTATIONAL GEOMETRY CORE (algorithms/)
================================================================================
1. algorithms/convex_hull.py:
   - graham_scan(points): O(N log N) time complexity.
     Finds lowest y-pivot point, sorts by polar angle using atan2, and processes
     via stack discarding clockwise turns using 2D cross-product:
     cross_product(o, a, b) = (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0]).
     Collinear point filtering uses cross-product check < 1e-9 (zero trig calls).
   - jarvis_march(points): O(N * H) output-sensitive gift-wrapping.
   - polygon_area_shoelace(pts): Gauss's shoelace formula for polygon area.
   - polygon_perimeter(pts) & compactness: 4 * pi * Area / Perimeter^2.

2. algorithms/voronoi_dominance.py:
   - compute_bounded_voronoi(points, pitch_length=105.0, pitch_width=68.0):
     Uses scipy.spatial.Voronoi with 4 ghost mirror sites across boundaries.
     Safely clamps generator sites to [0, L] x [0, W] when building ghost mirrors
     so off-pitch points never reflect ghost sites inside the field.
     Includes collinearity fallback with deterministic micro-jitter (< 0.1 mm)
     to prevent Qhull crashes.
   - sutherland_hodgman_clip(subject_poly, clip_poly, pitch_length, pitch_width):
     Numerically robust axis-aligned polygon clipper against all 4 pitch edges.
     Guarantees every cell vertex lies strictly within [0, 105] x [0, 68].
   - compute_spatial_dominance(team_a, team_b):
     Calculates territorial control percentages and per-player dominance cells.

3. algorithms/coach_analytics.py:
   - Classifies defensive block line: High Block (X > 45m), Mid Block (32-45m),
     Low Block (X < 32m).
   - Evaluates defensive compactness and territorial dominance.

4. algorithms/daa_benchmarks.py:
   - Benchmarks Graham Scan vs Jarvis March and Voronoi scaling across N nodes.

================================================================================
7. VISUALIZATION & RENDERERS (visualization/)
================================================================================
1. visualization/pitch_renderer.py (TacticalPitchRenderer):
   - Default dimensions: 1120x730 pixels, 40px margin padding.
   - Obsidian dark palette: BG (10, 14, 18), Turf (14, 25, 20), Stripes (18, 30, 24).
   - Draws FIFA markings: halfway line, center circle (9.15m), penalty boxes,
     6-yard boxes, penalty spots (11m), goal mouths, and crosshairs.
   - overlay_voronoi: Renders translucent polygons (alpha 0.26).
   - overlay_defensive_hull: Glowing double contour (outer neon, inner core).
   - render_players: Sky blue home nodes, crimson red away nodes, ball carrier halo.

2. visualization/overlays.py (BroadcastOverlayEngine):
   - draw_broadcast_hud: UEFA/Sky Sports glassmorphic header bar showing true
     end-to-end FPS, DAA latency (ms), hull area (m^2), and dominance (%).
   - overlay_projected_hull: Inverse homography projection of defensive block.
   - overlay_projected_voronoi: Inverse projection of Voronoi partitions.
   - get_turf_binary_mask: Projects full pitch boundary [(0,0),(105,0),(105,68),(0,68)]
     via H_inv to mask overlay within grass, ensuring custom calibration does not
     clip Voronoi to arbitrary landmark quadrilaterals.
   - In-place blending (cv2.addWeighted dst=overlay) reduces RAM allocations.

================================================================================
8. STREAMLIT UI, STATE & VIEWPORTS (ui/ & app.py)
================================================================================
1. app.py (Entrypoint):
   - Multi-viewport layout routing selectable from sidebar:
     * Dual Synchronized Studio (AR Video + 2D Radar)
     * Tri-View Studio (Raw Video + AR Video + 2D Radar)
     * Broadcast Camera AR Studio (Full screen AR)
     * Original Raw Video Studio (Baseline inspection)
     * 2D Tactical Radar Studio (Full screen radar board)
   - Setup Gate guard: forces calibration setup before entering main dashboard.
   - Top right button toggles 6-tab Pipeline Debugger (diagnostics_view.py).

2. ui/state.py:
   - Stores global session state: active_source_key, current_frame, is_playing,
     transformer, stream_manager, precomputed_data, manual_calib_points.
   - execute_pipeline():
     * Live Mode: Queries stream_manager.get_frame(), runs stage1-stage4 live.
     * Cached Mode: Reads precomputed JSON, inverts coordinates back to pixels
       using active session transformer (respecting custom calibration), and
       synthesizes bounding boxes.

3. ui/views/diagnostics_view.py:
   - Streamlit st.segmented_control for non-eager on-demand stage rendering.
   - Stage 1: Player detections table, foot contact coordinates, team clustering.
   - Stage 2: Matrix H values, determinant, condition number, coordinate probe.
     Coordinate probe correctly reports ON PITCH vs OUT OF BOUNDS.
   - Stage 3: Convex hull vertex-to-player attribution table and DAA complexity.
   - Stage 4: 2D Radar and AR Overlay inspection.
   - Coach AI & DAA Benchmarks: Live empirical runtime graphs.

================================================================================
9. PRECOMPUTATION & MULTIPROCESSING ENGINE (perception/precompute.py)
================================================================================
Selective Multiprocessing Architecture:
- Phase 1 (Single Process): Sequential video ingestion and YOLOv8 inference.
  Keeps PyTorch weights loaded in RAM exactly once (~500 MB total).
- Phase 2 (Selective Multiprocessing): Dispatches purely numerical coordinate
  payloads to concurrent.futures.ProcessPoolExecutor across 4 to 8 CPU cores.
  Each geometry worker consumes < 15 MB RAM with zero PyTorch duplication.
- Output: Compact JSON cache (data/precomputed_*.json) with serialized hull
  polygons, player_indices, and Voronoi territories.

================================================================================
10. COORDINATE SYSTEMS & MATHEMATICAL CONVENTIONS
================================================================================
There are 3 coordinate spaces across the system:

1. Camera Image Pixel Space:
   - Coordinates: (u, v)
   - Origin: Top-Left corner of image (0, 0)
   - Bounds: u in [0, frame_width], v in [0, frame_height]
   - Ground contact anchor: u = (box[0] + box[2]) / 2.0, v = float(box[3])

2. Metric Pitch Space:
   - Coordinates: (X, Y) in meters
   - Origin: Top-Left corner flag (0.0, 0.0) in standard sideline view
   - Length (X axis): 0.0 to 105.0 meters (Goal line to Goal line)
   - Width (Y axis): 0.0 to 68.0 meters (Far touchline to Near touchline)
   - Center spot: (52.5, 34.0)

3. 2D Radar Canvas Pixel Space:
   - Coordinates: (px, py)
   - Dimensions: 1120 x 730 pixels with padding = 40 px
   - px = padding + X * (usable_w / 105.0)
   - py = padding + Y * (usable_h / 68.0)

Round-Trip Reprojection Consistency:
- image_to_pitch(pitch_to_image(P)) error is < 0.05 px across the full camera view.
- When filter_out_of_bounds=False, image_to_pitch does NOT clamp coordinates.

================================================================================
11. VERIFICATION SUITES & TEST COMMANDS (scratch/)
================================================================================
Run all three automated suites to verify project health:

1. Coordinate Precision Suite:
   $ python3 scratch/verify_coordinates.py
   - Test 1: Homography round-trip precision (< 0.05 px).
   - Test 2: Sideline buffer retention (unclamped within 2.0m, discarded > 2.0m).
   - Test 3: Voronoi pitch boundary containment (0 <= x <= 105, 0 <= y <= 68).
   - Test 4: Custom calibration turf mask coverage (> 20% of frame, no quad clip).
   - Test 5: Full decoupled pipeline contract compliance.

2. System-Wide Regression Suite:
   $ python3 scratch/verify_fixes.py
   - Validates all 12 core fixes (Qhull collinearity, timeline sync, angle casing,
     landmark anchor test imports, cache file checks, player indices, etc.).

3. Architectural Contract Audit:
   $ python3 scratch/audit_suite.py
   - Validates multi-core worker scaling, clean imports, media dimension sync.

================================================================================
12. CRITICAL INVARIANTS & RULES (DO NOT BREAK)
================================================================================
1. Zero Phantom Players:
   Never inject fake, synthetic, or fallback players if detection is empty.
   If len(team_b) < 3, hull_polygon MUST be empty list [].

2. Do Not Clamp image_to_pitch Unconditionally:
   image_to_pitch() must return raw (raw_x, raw_y) when filter_out_of_bounds=False.
   Clamping breaks round-trip reprojection and coordinate probing.

3. Keep Inter-Stage Contracts Strictly Decoupled:
   Stage 4 renderers must never call YOLO detectors or compute homography matrices;
   they must strictly consume PitchPositions and TacticalGeometry.

4. Deterministic Frame Locking:
   Stream manager must seek and decode frame index N strictly without frame drops.

5. Clean Text Formatting:
   Never output broken raw LaTeX environments (\begin{bmatrix}) or dollar signs ($)
   in user responses when the client UI does not support MathJax rendering.

================================================================================
13. UPCOMING PLANNED FEATURE: N-POINT FLEXIBLE CALIBRATION
================================================================================
Target: Upgrade perception/calibration_templates.py and perception/homography.py
Requirements:
- Support selecting and clicking N points (4 <= N <= 10) instead of strictly 4.
- If N == 4: solve analytically via cv2.getPerspectiveTransform().
- If N > 4: solve via cv2.findHomography(src, dst, cv2.RANSAC).
- Live collinearity detection: alert user if any 3 points lie on a straight line.
- Live residual display: show per-point reprojection error in pixels on screen.
================================================================================
