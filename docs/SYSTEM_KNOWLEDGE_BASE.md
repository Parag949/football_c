# TacticalVision System Knowledge Base & Architectural Reference
**Version:** 2.1 (Engineering Reference & Contract Specification)  
**Scope:** Core perception, homography projection, algorithmic geometry, UI state synchronization, and hardware-accelerated playback.

---

## 1. Metric Pitch Coordinate System & Dimensions

1. **Standard Pitch Geometry (FIFA Law 1)**:
   - Pitch Length (X-axis): `0.0m` to `105.0m`.
   - Pitch Width (Y-axis): `0.0m` to `68.0m`.
   - Center Mark: `(52.5, 34.0)`.
   - Center Circle Radius: `9.15m`.
   - Penalty Areas:
     - Left Box: `[0.0, 16.5] x [13.84, 54.16]`.
     - Right Box: `[88.5, 105.0] x [13.84, 54.16]`.
   - Goal Posts: `X = 0.0` and `X = 105.0`, `Y` between `30.34m` and `37.66m` (width = `7.32m`).
   - Total Pitch Area: `105.0 * 68.0 = 7140.0 m²`.

2. **Valid Bounds & Buffering**:
   - Standard Turf Margin: `buffer_meters = 2.0m`.
   - Valid Player Metric Box: `[-2.0, 107.0] x [-2.0, 70.0]`. Coordinates outside this envelope represent false detections off-pitch (stands/benches).
   - Ball Coordinate: Can travel outside turf boundary (throw-ins/corners/stands); projection should not discard valid out-of-bounds ball locations.

3. **Camera Pixel Coordinates**:
   - `(0, 0)` is top-left, `(width - 1, height - 1)` is bottom-right.
   - Projected foot point is anchored inside player bounding box: `(box[0] + box[2]) / 2.0`, `box[3] - min(6.0, 0.04 * box_h)`.

4. **Projective Depth Convention**:
   - For any projective homography `x_cam = H_inv @ X_pitch` and `X_pitch = H @ x_cam`, projective scale factor `w` represents projective depth relative to the camera focal plane.
   - **In Front of Camera**: `w > 1e-4`. Points with `w > 0` are in front of the camera and can be safely mapped to image pixels.
   - **Behind Camera**: `w <= 0`. Points behind the camera principal plane must be filtered (`filter_behind_camera=True`) and never projected inverted across the horizon line.

---

## 2. Calibration & Homography Contracts

1. **Calibration Signature (`get_calibration_signature`)**:
   - Purpose: Deterministic 12-char MD5 cache key ensuring that stored precomputations, payloads, and detections match the active camera projection.
   - Invariant:
     - When `is_custom_calibrated` is `True`, the signature is uniquely determined by `(width, height, warp_mode, src_points, dst_points)`. The preset `camera_mode` (`"sideline"`, `"goal_to_goal"`, `"manual_mouse"`) must NOT alter the signature when custom points are active.
     - When `is_custom_calibrated` is `False`, the signature is determined by `(width, height, camera_mode, warp_mode)`.

2. **Warp Modes**:
   - `"planar"`: Standard 3x3 projective homography matrix `H` via `cv2.findHomography` / `cv2.getPerspectiveTransform`.
   - `"panorama_tps"`: Thin-Plate Spline deformation grid over dense field points `(p2i_lut, i2p_lut)`.
   - `"auto"`: Evaluates planar reprojection residual. If max residual `> 1.5 px` and point count `>= 6`, automatically selects `"panorama_tps"`; otherwise stays `"planar"`. `HomographyTransformer.__init__` must accept `"auto"`.

3. **Projective Sign Normalization (`_normalize_projective_sign`)**:
   - Must determine camera frontality using the center of the active calibration region (e.g. mean of `dst_points` or visible landmarks), not a hardcoded pitch center `(52.5, 34.0)`. For zoomed cameras focused on end-zones, `(52.5, 34.0)` may lie behind the camera while all calibrated points are in front.

4. **Behind-Camera Filtering in TPS & Planar Modes**:
   - `pitch_to_image` must discard `w <= 1e-4` in both planar and TPS modes.
   - Projected integer coordinates must be verified finite and clamped to prevent NumPy `int32` overflows before OpenCV rendering.

---

## 3. Data Schemas & Contracts

### 3.1. Core Contracts (`core/contracts.py`)
- `PlayerDetection`:
  - `box`: `Tuple[int, int, int, int]` (`(x1, y1, x2, y2)`).
  - `foot_point`: `Tuple[float, float]`.
  - `team_id`: `int` (`0` = Team A, `1` = Team B, `-1` = referee/unknown).
- `FrameDetections`:
  - `frame_idx`: `int`.
  - `players`: `List[PlayerDetection]`.
  - `ball_pixel`: `Optional[Tuple[float, float]]`.
  - `frame_width`: `int`, `frame_height`: `int`.
- `PitchPositions`:
  - `frame_idx`: `int`.
  - `team_a`: `List[Tuple[float, float]]` (metric pitch coordinates).
  - `team_b`: `List[Tuple[float, float]]`.
  - `ball`: `Optional[Tuple[float, float]]`.
  - `carrier_idx`: `Optional[int]`.
  - `detected_count`: `int`.
  - Must tolerate `{"ball": None}`, `{"team_a": None}`, etc. in `from_dict`.
- `TacticalGeometry`:
  - `frame_idx`: `int`.
  - `hull_polygon`: `List[Tuple[float, float]]`.
  - `hull_area`: `float`, `hull_compactness`: `float`.
  - `voronoi_cells_a`: `List[Dict[str, Any]]`, `voronoi_cells_b`: `List[Dict[str, Any]]`.
  - `team_a_pct`: `float`, `team_b_pct`: `float`.
  - `dijkstra_res`: `Optional[Dict[str, Any]]`.
  - Must tolerate `{"hull": None}`, `{"voronoi": None}`, `{"dijkstra": None}` in `from_dict`.

### 3.2. Execution Bundle (`ui/state.py:execute_pipeline`)
The return dict of `execute_pipeline` must contain:
```python
{
    "raw_frame": np.ndarray,             # uint8 BGR frame
    "detections": FrameDetections,
    "positions": PitchPositions,         # Note: team_a, team_b, ball, carrier_idx are properties of positions
    "geometry": TacticalGeometry,
    "coach_data": Dict[str, Any],        # Analytics metrics (see 3.3)
    "ruler_data": Dict[str, Any],        # Measurement data
    "radar_canvas": np.ndarray,          # 2D radar image
    "broadcast_overlay": np.ndarray,     # Broadcast AR frame
    "latency_ms": float,
    "source_type": str,
    "frame_idx": int
}
```

### 3.3. Coach Analytics Schema (`algorithms/coach_analytics.py`)
`evaluate_coach_analytics()` must supply both top-level and structured `'passing'` subdict:
```python
{
    "block": str,                        # e.g. "Mid Block", "High Press", "Low Block"
    "final_third_pct_home": float,
    "final_third_verdict": str,
    "is_line_breaking": bool,
    "pass_directive": str,               # e.g. "Through Ball", "Circulate"
    "active_pressers": int,
    "pressing_state": str,
    "peak_accel": float,
    "line_height_m": float,
    "passing": {
        "progression_m": float,
        "is_line_breaking": bool,
        "directive": str,
        "passes_count": int,
        "path": List[int]
    }
}
```

---

## 4. Algorithmic Invariants

1. **Convex Hull (`algorithms/convex_hull.py`)**:
   - Must accept list-of-lists `[[x, y], ...]` as well as tuples without raising `TypeError`.
   - Must filter non-finite (`NaN`, `Inf`) points before computing the hull.
   - Polar sorting in `graham_scan` must normalize vectors or use robust cross-product angular comparison so points close to the pivot (`< 1e-4m`) are not dropped.
   - Compactness (isoperimetric quotient `(4 * pi * area) / perimeter^2`) must be strictly clamped to `[0.0, 1.0]`.

2. **Voronoi Dominance (`algorithms/voronoi_dominance.py`)**:
   - Player coordinates must be deduplicated (or slightly perturbed if coincident) before Voronoi tessellation.
   - Boundary Ghost Mirror Sites: Ghost sites must be offset by `>= 1.0m` outside physical turf boundaries (`[-x - 1.0, y]`, `[2*L - x + 1.0, y]`, `[x, -y - 1.0]`, `[x, 2*W - y + 1.0]`). Reflecting points with distance `0` when players stand directly on touchlines creates coincident sites and drops boundary Voronoi cells.
   - Area Conservation: When partition territory is measured, `team_a_area + team_b_area == 7140.0 m²` (the physical area of the pitch) via proportional normalization across both teams.

3. **Coach Analytics (`algorithms/coach_analytics.py`)**:
   - `line_height_m` must always be initialized to default baseline `35.0m` at function entry so that empty defender sets or collinear hull inputs do not raise `UnboundLocalError`.

4. **Dijkstra Passing Network (`algorithms/passing_network.py`)**:
   - If no route exists to the target node, `compute_dijkstra_passing_path` must return `path = []` (or a single carrier node `[carrier_idx]`), with `total_weight = 0.0` and `total_progression = 0.0`. It must NEVER invent a direct edge between non-adjacent nodes with `0.0` cost.
   - Points with `NaN` coordinates must be discarded and never allowed to form `0.1` cost edges.
   - Attack direction: support both left-to-right (`X -> 105`) and right-to-left (`X -> 0`).

5. **Ghost Runner (`algorithms/ghost_runner.py`)**:
   - Offside Rule (FIFA Law 11): A player is NOT offside if they are behind or level with the ball at the moment of the pass, even if beyond the second-deepest defender. `simulate_ghost_runner` must evaluate `ball_coord` when checking offside.

6. **DAA Benchmarks (`algorithms/daa_benchmarks.py`)**:
   - Guard against `n <= 0` and empty lists (`node_counts = []`).

---

## 5. Thread Safety, Cache Integrity & Memory (`perception/video_stream.py`)

1. **Memory Budget & Cache Separation**:
   - `detection_cache`: strictly stores lightweight Stage 1 detection meta (`'detections': FrameDetections, 'raw_frame': None`). Storing raw 1080p numpy arrays in this 1000-entry cache consumed 6-12 GB RAM; raw frames must route strictly to `raw_frame_cache`.
   - `raw_frame_cache`: thread-safe in-memory ring buffer with fixed capacity `_raw_cache_capacity = 120` frames (~745 MB RAM).
   - `broadcast_state_cache`: bounded to `_raw_cache_capacity` (120 entries) instead of 1000 entries.
   - All accesses to caches must be synchronized under `_cache_lock`.

2. **Seek Failure & EOF Resilience**:
   - If OpenCV seek or frame read fails (e.g. at EOF), `get_broadcast_frame_state` must return a blank fallback state and reset `_last_read_pos = -1`. It must NEVER seek to frame 0 and cache frame 0 under future frame indices.
   - On successful read, `_last_read_pos` must record `frame_idx + 1` to align with OpenCV's internal playhead.

3. **Source Switch Lifecycle & Resource Cleanup**:
   - `load_video_source()` must increment an atomic generation counter `_stream_generation` to discard any in-flight background worker threads from previous videos.
   - `release()` / `close()` methods release OpenCV handles and shut down the prefetch thread pool executor cleanly.
   - Detector execution (`FootballDetector.detect_frame`) must be guarded by a thread lock so the background prefetch thread and the UI main thread do not run concurrent inference on shared model predictor state.

---

## 6. Frontend Browser Player (`ui/components/assets/hardware_player.js`)

1. **Lifecycle & Playback Callbacks**:
   - `playVideo()` / `pauseVideo()` / `seekToFrame()` must track `activeRvfcId` and invoke `cancelVideoFrameCallback(activeRvfcId)` to prevent multiple stacked 60fps draw loops.
   - A `seeked` event listener on `<video>` synchronizes overlays immediately during paused scrub operations.

2. **Bounds & Math Guards**:
   - `renderRadar2D` must guard `pw <= 0 || ph <= 0` before drawing circles to avoid `DOMException: IndexSizeError: The radius provided is negative`.
   - `getPitchCoordsFromEvent` must guard `!rect.width || !rect.height || pw <= 0 || ph <= 0` to prevent division by zero / NaN coordinates.
   - `bilinearLut` must check `nx >= 2 && ny >= 2` and coordinate finiteness.

3. **Variables & Scope**:
   - Variables such as `grpt` and `ghostColor` must be declared at function scope in `renderRadar2D`, ensuring they are valid even when `bundle.a` is empty or missing.

4. **Bundle Nullability**:
   - All bundle accesses (`bundle.dijk.path`, `bundle.hull`, `bundle.vor`, `bundle.c`) must be null-guarded.
   - Pass count metric: `(bundle.dijk.path.length > 1 ? bundle.dijk.path.length - 1 : 0)`.
   - Carrier index check: `bundle.c !== null && bundle.c !== undefined && bundle.c >= 0`.

5. **Projective Horizon Guard**:
   - `projectPitchToImage` and `projectImageToPitch` must check `w_h > 1e-4` to reject behind-camera points before screen clipping.

---

## 7. Streamlit UI & State Lifecycle

1. **Namespaced Calibration Tracking**:
   - Landmark tracking key in `calibration_studio.py` must be namespaced per instance (`f"{key_prefix}_last_lms"`), ensuring that switching between Setup Gate (4 points) and Diagnostics (8 points) tabs does not wipe user-clicked calibration points.
   - Adding a point must increment `ver_key` to force a clean component re-render.

2. **Upload Integrity & Decoder Verification (`setup_gate.py`)**:
   - Discards 0-byte uploads immediately and verifies that OpenCV `VideoCapture` can successfully decode at least one frame before updating session state, preventing infinite error-reset loops.
   - Uses `_atomic_json_dump` (atomic file replacement) and module-level locks in `player_assets.py` to prevent `JSONDecodeError` during concurrent browser tab loads.

3. **Interactive Sandbox**:
   - "Import Video Frame" clamps imported coordinates to `[0.5, 104.5] x [0.5, 67.5]` so that borderline off-pitch players never disrupt SVG drag handles.
