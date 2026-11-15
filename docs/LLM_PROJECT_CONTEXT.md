# TacticalVision Pro: LLM Project Context and Report Brief


## Instructions for a report-writing LLM

Use this document as a map, not as proof that every described feature works in every scenario. Prefer the current implementation in the source files over older prose in the README, architecture diagrams, hackathon guide, and knowledge base. When evidence is incomplete:

- Describe a capability as implemented only when the code path or a relevant test supports that description.
- Do not turn target performance, illustrative examples, mathematical assumptions, or roadmap proposals into measured results.
- Do not claim player identity tracking, event recognition, broadcast-camera motion compensation, or production-grade offside adjudication.
- State that the pitch-control implementation uses a simplified, shared-speed arrival-time model; it does not estimate individual player velocity.
- State that xT is a fixed spatial lookup grid, not a model trained on this project's match data.
- Distinguish algorithm modules that can be called from the default pipeline. In particular, the Dijkstra passing module exists, but `stage3_geometry` sets `dijkstra_res` to `None`.
- Cite source paths by their repository-relative names, and identify any assertion that has not been validated by a test or benchmark.

## Executive synopsis

TacticalVision Pro is a Python/Streamlit football video analysis workstation. Its goal is to turn a single match-video feed into a calibrated 2D pitch view and broadcast-frame overlays, then expose geometric and spatial analytics through an interactive interface. The application combines YOLO-based object detection and jersey-color grouping with camera-to-pitch projection, computational geometry, rendering, saved frame data, and an interactive tactics board.

The central pipeline has four conceptual stages: frame perception, pixel-to-pitch projection, tactical geometry, and visualization. The main coordinates are measured in metres on a configurable football pitch (105 m by 68 m by default). Current geometry includes a defensive convex hull and pitch-bounded Voronoi territory. Additional modules provide a simplified pitch-control field, a static 16-by-12 expected-threat grid, heuristic formation detection, passing-path calculations, offside/ghost-runner simulations, and a tactical ruler. The Streamlit UI supports match setup and calibration, replay, diagnostics, and a 2D tactics sandbox. Browser-side video and canvas rendering can use precomputed frame payloads to avoid running Python inference for each displayed frame.

This is best described as a **prototype/engineering workstation** or **computer-vision-assisted tactical visualization project**, not as a validated professional tracking system. Detection and projection quality depend on the video, model weights, team kits, and calibration. The repository's own documentation identifies camera motion and planar-pitch assumptions as limitations; advertised frame rates and accuracy should be treated as claims unless independently measured for the chosen setup.

### Short synopsis

> TacticalVision Pro is a Streamlit-based football analysis prototype that detects players and the ball in video, assigns team labels from kit appearance, maps image locations to pitch coordinates through camera calibration, and visualizes team shape and territorial space. It combines computer vision with computational geometry and lightweight tactical metrics, offering replay, calibration/diagnostics, and a browser-based tactics board. Its principal constraints are single-view detection quality, calibration dependence, limited temporal tracking, and simplified analytical models.

## Project goals and intended users

- Explore lower-cost tactical visualization from a single broadcast, panoramic, or user-supplied video rather than a dedicated stadium tracking installation.
- Support coaches, analysts, students, and developers inspecting spatial shape, territory, and tactical scenarios.
- Demonstrate a hybrid of machine-learning-based perception, classical geometry/graph algorithms, and interactive visualization.
- Provide both a frame-by-frame diagnostic workflow and a replay/sandbox-oriented workflow.

The project is also framed in its supporting docs as a DAA (Design and Analysis of Algorithms) demonstration. That framing is appropriate for the implemented hull, Voronoi, graph, and benchmarking modules, but the live application is more accurately characterized as an integrated prototype than as a pure algorithms package.

## How the application works

### Application flow

`app.py` initializes Streamlit configuration, theme and session state, synchronizes the selected video source, then gates entry to the workstation on setup confirmation. Workstation views are:

1. **Match Video & Tactical Replay** — the browser-oriented hardware player, using generated assets/payloads and canvas overlays.
2. **Frame Inspector & Alignment** — controls, per-frame pipeline diagnostics, calibration tools, and performance/algorithm views.
3. **2D Tactics Board** — an interactive browser canvas for hypothetical player/ball placement and scenarios.

The UI state and media helpers are chiefly in `ui/state.py`, `ui/state_models.py`, `ui/media_library.py`, and `ui/components/`.

### Four pipeline stages

1. **Perception — `core.pipeline.stage1_perception`**
   - Calls `FootballDetector` on an OpenCV frame, optionally using the calibrated pitch region as a region of interest.
   - Produces `FrameDetections`: player boxes, foot-contact anchors, team labels, optional ball pixel, frame dimensions, and frame index.
   - `perception/detector.py` wraps Ultralytics YOLO, preferring available local model files and supporting ONNX or PyTorch weights. It also contains class handling, ROI/aspect-ratio logic, confidence settings, and inference locking.
   - `perception/jersey_classifier.py` extracts torso color features and groups kits; referee/unknown labels can be kept separate.

2. **Projection — `core.pipeline.stage2_projection`**
   - Uses `HomographyTransformer` to transform foot anchors from image pixels to pitch metres and filter player positions outside a run-off buffer.
   - The ball may remain out of bounds. If no ball is detected, the code supplies a synthetic/default location rather than measuring possession from video.
   - The current carrier heuristic selects the Team A player nearest to the projected ball; this is not persistent identity tracking.

3. **Geometry — `core.pipeline.stage3_geometry`**
   - Calculates the Team B defensive hull using Graham Scan by default, with Jarvis March as an option.
   - Calculates the combined-team bounded Voronoi partition and Team A/Team B area shares.
   - The Voronoi implementation uses `scipy.spatial.Voronoi` and clips cells with Sutherland-Hodgman. It is not a custom Fortune sweep-line implementation despite that phrase in some documentation.
   - It explicitly assigns `dijkstra_res = None`; passing network code is available separately but is not run by this default stage.

4. **Visualization and analytics — `core.pipeline.execute_frame_bundle` and renderers**
   - Renders a 2D pitch radar and a projected broadcast overlay.
   - Also computes coach summaries, the arrival-time pitch-control field, heuristic formations, and ball-location xT in the bundle path.
   - `visualization/pitch_renderer.py` draws the 2D pitch; `visualization/overlays.py` projects geometry and HUD elements onto the video.
   - `algorithms/expected_threat.py`, `algorithms/pitch_control.py`, and `algorithms/formation_detector.py` provide additional analytic data consumed by UI/rendering paths.

`core/contracts.py` contains dataclasses for detections, projected positions, geometry, and frame bundles. The public frame bundle shape is serialized into JSON-like data for caches and browser assets.

### Camera calibration and coordinate convention

- `perception/pitch_model.py` defines pitch landmarks and geometry; standard dimensions are 105 m by 68 m, with helper support for variable dimensions.
- `perception/homography.py` implements planar projective calibration, an optical lens-distortion mode (`universal`), and a thin-plate-spline residual mode (`panorama_tps`), plus an `auto` selection path.
- Calibration tools in `ui/components/calibration_studio.py` and `ui/components/pitch_mask_studio.py` collect and persist user alignment points/handles.
- `ui/media_library.py` stores per-video calibration and cache paths. Custom uploads and calibration files are runtime/user data, not a guarantee of included sample assets.
- Projection assumes a flat ground plane. A homography does not correctly recover the ground location of airborne objects or account for arbitrary camera motion without recalibration or additional tracking.

### Playback, precomputation, and state

- `perception/video_stream.py` handles OpenCV video capture, frame seeking, detection caching, and prefetch/ring-buffer behavior.
- `perception/precompute.py` performs sequential video reading and perception, then can parallelize numerical geometry work and save frame bundles.
- `ui/state.py` chooses cached versus live data, checks some cache compatibility metadata, and builds the application bundle. `ui/state_models.py` groups video, playback, calibration, display, and ruler settings.
- `ui/components/player_assets.py`, `hardware_player.py`, and the corresponding HTML/CSS/JavaScript implement the browser playback/rendering payload path.
- A “60 FPS” browser display is not the same as 60 FPS end-to-end video inference. Cached/browser playback may be smooth while producing no new detections during playback.

## Analytics and algorithm modules

| Module | What the source implements | Important qualification |
|---|---|---|
| `algorithms/convex_hull.py` | Graham Scan and Jarvis March hulls, polygon area/centroid, defensive area/compactness | Complexity depends on player count and hull size; timings are runtime-dependent. |
| `algorithms/voronoi_dominance.py` | SciPy Voronoi cells, boundary mirror sites, polygon clipping, team territory areas/percentages | Equal-speed nearest-site territory, not a full dynamic control/arrival model. |
| `algorithms/pitch_control.py` | Grid over pitch; each side's minimum arrival time is based on distance, common `v_max`, and reaction time, then mapped through a sigmoid | No player-specific velocity or acceleration is used in this function despite references in its module text. |
| `algorithms/expected_threat.py` | Fixed 16 x 12 spatial lookup grid, start/end xT and action-threat ranking | Not trained or calibrated from the included match videos or local event data. |
| `algorithms/formation_detector.py` | Groups longitudinal player positions into bands and matches line counts to common formation templates | A positional heuristic; does not infer a formation from temporal tracking or player roles. |
| `algorithms/passing_network.py` | Builds a directed candidate passing graph weighted by pass length/progression and defender distance to the pass segment; solves a path with Dijkstra | Available as a utility, but not run in `stage3_geometry`; the default pipeline geometry contract leaves its result empty. |
| `algorithms/ghost_runner.py` | Simulates a hypothetical player, offside line, and before/after territory/path deltas | Scenario analysis from supplied coordinates; not referee-grade rule adjudication or observed player tracking. |
| `algorithms/tactical_ruler.py` | Euclidean distance, signed/absolute offsets, angle, and offside breach summaries | Depends on the supplied positions and attack direction. |
| `algorithms/coach_analytics.py` | Converts hull/Voronoi/path/kinematic inputs into labels and summaries | Its result is a heuristic interpretation; current pipeline passes no kinematics and no Dijkstra result. |
| `algorithms/daa_benchmarks.py` | Generates random pitch nodes and times hull, Voronoi, and passing routines | Synthetic microbenchmarks, not end-to-end video throughput or a reproducible hardware-independent guarantee. |

## Repository map

### Root and runtime configuration

- `app.py` — Streamlit entry point and workstation routing.
- `requirements.txt` — Python package requirements.
- `.streamlit/config.toml` — Streamlit server/upload settings.
- `.gitignore` — excludes generated artifacts, local user video/calibration/cache directories, runtime secrets, and some model/static patterns.
- `README.md` — public overview, quickstart, feature descriptions, limitations, and example test commands.

### `core/`

- `contracts.py` — typed pipeline schemas and serialization helpers.
- `pipeline.py` — stage functions, frame execution, analytics composition, and pipeline health summaries.

### `perception/`

- `detector.py` — YOLO model loading, inference, and detection postprocessing.
- `jersey_classifier.py` — torso color feature extraction, team grouping, referee isolation.
- `homography.py` — calibration, coordinate conversion, and projection export.
- `pitch_model.py` — pitch dimensions, markings, landmarks, and coordinate helpers.
- `calibration_templates.py` — calibration landmark/preset rendering and ROI coordinate conversion.
- `video_stream.py` — video ingest, decoding, caches, prefetch, and stream lifecycle.
- `precompute.py` — disk precomputation and cached-frame access.
- `__init__.py` — package marker.

### `algorithms/`

- `convex_hull.py` — hull algorithms and derived polygon metrics.
- `voronoi_dominance.py` — SciPy Voronoi construction, pitch clipping, territorial shares.
- `passing_network.py` — passing graph, defensive pressure, and Dijkstra path.
- `pitch_control.py` — gridded arrival-time control and heatmap.
- `expected_threat.py` — static xT grid and action-threat helpers.
- `formation_detector.py` — heuristic formation and line metrics.
- `ghost_runner.py` — hypothetical runner and offside simulation.
- `tactical_ruler.py` — distance and offside measurement.
- `coach_analytics.py` — tactical summary generation.
- `daa_benchmarks.py` — synthetic algorithm benchmarks.
- `__init__.py` — package exports.

### `ui/`

- `state.py`, `state_models.py` — session orchestration, source/cache selection, and structured state.
- `media_library.py` — built-in sources, custom video paths, calibration and cache persistence.
- `theme.py` — Streamlit styling.
- `components/setup_gate.py` — initial source/calibration/execution setup.
- `components/calibration_studio.py` — manual multi-point camera calibration UI.
- `components/pitch_mask_studio.py` and `pitch_mask_studio_assets.py` — interactive pitch mask/calibration interface and its HTML/CSS/JS.
- `components/control_deck.py` — diagnostic controls, playback and analysis options.
- `components/hardware_player.py`, `player_assets.py` — browser player asset/payload preparation.
- `components/header.py` — application header and stream telemetry.
- `components/coach_cards.py` — coaching metrics presentation.
- `components/tactical_sandbox.py`, `tactical_sandbox_canvas.py` — interactive 2D scenario board.
- `components/assets/hardware_player.html`, `hardware_player.css`, `hardware_player.js` — browser player and rendering implementation.
- `views/diagnostics_view.py` — pipeline inspector, calibration controls, analytics, and benchmark UI.
- Package `__init__.py` files provide module exports.

### `visualization/`

- `pitch_renderer.py` — 2D pitch, player, territory, hull, and heatmap rendering.
- `overlays.py` — camera-space AR overlays, pitch projection, and HUD.
- `__init__.py` — renderer exports.

### `docs/`

- `ARCHITECTURE.md` — intended system layout and data contracts.
- `DATA_SOURCES.md` — directory of external football datasets and platforms; it is reference material, not evidence that these sources are integrated.
- `HACKATHON_GUIDE.md` — pitch/demo script and algorithm explanations. Some claims are promotional or illustrative.
- `ROADMAP.md` — proposed work and a table of items described as completed; roadmap prose is not itself test evidence.
- `SYSTEM_KNOWLEDGE_BASE.md` — detailed contracts and invariants, including prior bug-fix expectations. Verify implementation before treating an invariant as guaranteed.
- `LLM_PROJECT_CONTEXT.md` — this report-writing reference.

### `data/`, `models/`, `static/`, and generated/local files

- The repository includes sample video files and local model artifacts (`best.onnx`, `best.pt`, `yolov8n.onnx`, `yolov8n.pt`). Treat these as binary assets, not source code; their mere presence does not establish model accuracy, licensing, or suitability.
- `data/precomputed_user.json` is named by source/docs as a cache associated with the sample match. It is generated/derived data whose availability may vary by checkout; the Streamlit AppTest currently asserts that this path exists, so test results depend on that asset being available.
- User-supplied videos, saved calibrations, cache files, and browser payloads may be ignored/untracked runtime assets. Their availability differs between machines.
- `static/` is used for generated/browser-facing assets. The tracked `.gitkeep` does not represent a usable match video.

### `scratch/`

This directory contains runnable test/regression scripts, verification/audit helpers, and some generated debug images/media. These are engineering artifacts rather than application modules. Tracked Python test scripts include:

- `test_all_6_bugfixes.py`
- `test_async_prefetch.py`
- `test_calibration_pipeline_sync.py`
- `test_dijkstra_passing.py`
- `test_forward_projection_caching.py`
- `test_ghost_runner.py`
- `test_hardware_player.py`
- `test_knowledge_base_contracts.py`
- `test_landmark_anchor.py`
- `test_multipoint_calib.py`
- `test_new_tactical_modules.py`
- `test_panorama_calibration.py`
- `test_panorama_ui_state_sync.py`
- `test_remediation_fixes.py`
- `test_ring_buffer_fw13.py`
- `test_state_machine_consolidation.py`
- `test_streamlit_apptest.py`
- `test_tactical_ruler.py`
- `test_tactical_sandbox.py`
- `verify_coordinates.py`
- `verify_fixes.py`
- `audit_suite.py`
- `deep_bug_hunter.py`

The test files cover contract/null safety, calibration/projection, geometry, analytics, caching/prefetch, browser assets, Streamlit navigation, and sandbox behavior. Their existence is not equivalent to a passing run. The analysis for this document did not execute the suite.

## Evidence-aware limitations and claim checks

Use these distinctions in any formal synopsis or report:

1. **Detection is not tracking.** The inspected pipeline detects objects on frames and groups jerseys; it does not establish persistent player identity across time.
2. **Calibration is a dependency.** Metric coordinates and overlays depend on correctly selected landmarks and an appropriate camera mode. A static transform may drift when a broadcast camera pans, tilts, or zooms.
3. **Ground-plane geometry has parallax limits.** Foot-contact points are the intended projection anchors. The ball and airborne players are not made metrically accurate by a ground-plane homography.
4. **The algorithms have different meanings.** The bounded Voronoi output is nearest-site territory; the pitch-control grid is a separate approximate arrival-time model. Neither alone proves actual control or possession.
5. **The xT and formation outputs are simplified.** xT uses a fixed matrix and formation uses coordinate-band heuristics.
6. **Passing-route availability differs from default execution.** Dijkstra code and UI/what-if uses exist, but the central geometry stage disables it and produces no path result by default. Avoid presenting a live passing route as a default pipeline output without checking the active code path.
7. **Performance figures need measurement context.** Browser presentation, synthetic algorithm timings, stage latency, and full-video inference throughput are distinct metrics. Do not reuse “60 FPS,” “under 2 ms,” or similar values as validated end-to-end results without a fresh, reproducible benchmark.
8. **Supporting docs contain stale or aspirational wording.** Examples include Fortune sweep-line terminology for SciPy Voronoi, velocity/acceleration language for a constant-speed pitch-control implementation, and strong real-time/accuracy claims. Source code and fresh test results take precedence.
9. **Inputs and generated assets are machine-specific.** Data, caches, calibration files, and installed model weights may be absent or differ. Report the exact analyzed configuration when making empirical statements.

## Suggested project report structure

An LLM can use the following outline, omitting sections that the requested report does not need:

1. **Title and concise synopsis** — what it does, target user, and central idea.
2. **Problem and motivation** — the challenge of extracting tactical spatial information from ordinary video.
3. **Objectives and scope** — video ingestion, calibration, player/team perception, spatial metrics, visualization, and interactive scenarios.
4. **System architecture** — Streamlit application, four-stage pipeline, contracts, cache/precompute path, browser player.
5. **Methods** — detection and team color grouping; calibration; convex hull; bounded Voronoi; optional pitch control, xT, formation, passing, offside, and ruler modules.
6. **Implementation and tools** — Python, Streamlit, OpenCV, Ultralytics/YOLO, NumPy/SciPy, ONNX Runtime, and browser HTML/CSS/JavaScript, as supported by `requirements.txt` and source.
7. **User interface and workflow** — setup/calibration, replay, diagnostics, tactics board.
8. **Testing and evaluation** — list the exact tests run, input/video/model/camera mode, and actual outputs. Separate synthetic tests from real-footage evaluation.
9. **Results** — include only measurements obtained in the report's own validation run; cite setup and method.
10. **Limitations and risks** — calibration, camera motion, occlusion, no identity tracking, simplified models, assets and reproducibility.
11. **Conclusion and future work** — summarize demonstrated scope and identify roadmap items as future work, not completed features.

## Recommended evidence to collect before claiming results

- Record Python/platform/dependency versions and the exact model file used.
- Identify video dimensions, source frame rate, camera behavior, calibration mode/landmarks, and whether frame data came from cache or live inference.
- Run the relevant scripts from `scratch/` and record their exit status and output. Note that tests may assume sample assets (including the precomputed cache) that are absent in the current snapshot.
- Measure detector recall/precision only against annotated frames; count false positives and missed detections rather than relying on visual impressions.
- Measure calibration reprojection error on held-out landmarks or frames, not only the calibration points used to fit the transform.
- Benchmark full pipeline throughput separately from isolated geometry functions and browser playback FPS.
- Capture example output frames and describe any manual calibration or preprocessing needed to obtain them.

## Documentation map and precedence

- For the public project narrative and usage entry point, see `README.md`.
- For intended topology/contracts, see `docs/ARCHITECTURE.md`.
- For detailed stated invariants, inspect `docs/SYSTEM_KNOWLEDGE_BASE.md` and verify against source/tests.
- For actual behavior, inspect `app.py`, `core/`, `perception/`, `algorithms/`, `ui/`, and `visualization/`.
- For proposed work, see `docs/ROADMAP.md`.
- For example defenses of the project, see `docs/HACKATHON_GUIDE.md`, but label its scenario numbers and marketing claims as illustrative unless independently measured.

---

## Expanded handoff for a report-writing LLM

This section is deliberately detailed so another person can provide this file to an LLM on a different computer and get a useful report without assuming the LLM has access to this repository. It adds explanatory context, a recommended evidence hierarchy, reusable writing prompts, and a cross-machine reproducibility checklist.

### 1. Stable project identity

| Item | Grounded description |
|---|---|
| Project name | TacticalVision Pro (also called TacticalVision in source comments and README headings). |
| Repository | `Parag949/football_c`, a Python application repository. |
| Application type | Streamlit web application/workstation; it is run locally and viewed in a browser. |
| Domain | Football/soccer video analysis, spatial visualization, tactical scenario exploration. |
| Main input | Video file frames, either a built-in/sample source or a user-selected/uploaded video. |
| Core coordinate output | Approximate player and ball coordinates in a two-dimensional pitch coordinate system, normally metres on a 105 m by 68 m field. |
| Main user-facing outputs | Broadcast-frame overlays, 2D pitch radar, diagnostic metrics, and an interactive tactics sandbox. |
| Technical character | Integration prototype combining object detection, camera calibration, geometry, analytics, Python UI, and browser assets. |
| Not established by repository evidence | A product-level accuracy guarantee, official tracking certification, player identity system, automated event classifier, or representative field-trial result. |

### 2. Problem framing and motivation

Football performance analysis often benefits from knowing where players are, how a team is arranged, which spaces are occupied, and whether a hypothetical movement changes the tactical picture. Dedicated optical tracking systems can provide spatial data, but this project explores a more accessible workflow: start with a single video image stream, estimate player/ball locations, calibrate the camera view to a pitch, and render spatial summaries.

The important engineering problem is that raw video is in **image coordinates**, while most tactical quantities are easier to explain in **pitch coordinates**. The project therefore has to join several different technical disciplines:

1. **Perception:** locate candidate objects in a camera frame.
2. **Appearance grouping:** separate teams by the detected players' shirt colors.
3. **Geometric calibration:** associate visible image points with known pitch landmarks.
4. **Metric projection:** express image locations as positions on a flat pitch.
5. **Spatial analysis:** derive hulls, nearest-player territory, and tactical estimates from those positions.
6. **Presentation:** render a synchronized pitch graphic, an annotated video frame, and tools for exploring alternatives.

This is a useful educational and prototype problem because it demonstrates how errors can propagate through a pipeline: a missed detection or an incorrect foot anchor can alter the metric coordinates; a calibration error can distort every downstream geometry; and even correct coordinates do not make a simplified tactical model equivalent to actual player intent or match events.

### 3. Objectives, deliverables, and non-goals

#### Objectives evidenced by the repository

- Accept or select a match video through a Streamlit setup flow.
- Configure and persist a camera-to-pitch alignment for a video.
- Run YOLO-based object detection and produce typed detection records.
- Estimate player team membership from jersey appearance.
- Convert image-space ground-contact anchors to pitch-space coordinates.
- Calculate a defensive convex hull and a pitch-bounded Voronoi partition.
- Provide additional tactical utilities, including pitch control, xT lookup, formation heuristics, passing paths, offside/ghost-runner simulation, and distance measurement.
- Display the results in Python-rendered and browser-rendered interfaces.
- Support analysis caching/precomputation and responsive frame playback.
- Include regression scripts for a range of geometry, calibration, state, UI, and browser concerns.

#### Explicit non-goals or capabilities not established

- Reconstructing a complete, ground-truth 22-player trajectory with persistent player identities.
- Replacing a certified tracking or officiating system.
- Determining possession, pass completion, shot outcomes, or match events with validated accuracy.
- Proving tactical intent from spatial configuration alone.
- Providing a camera-motion tracker for arbitrary broadcast pan/tilt/zoom.
- Making all computations execute at real-time frame rate on every machine.
- Learning a project-specific xT model or a player-specific physics model from the checked-in clips.
- Guaranteeing that listed sample videos, weights, or cached analysis files are installed or accessible in another checkout.

An LLM can phrase these as scope boundaries, not project failures: the work demonstrates a video-to-spatial-analysis path and provides an extensible interactive prototype, while leaving robust tracking and empirical validation as future work.

### 4. A detailed end-to-end narrative

The following walk-through can be adapted to a “System Design” or “Implementation” chapter.

#### 4.1 Video source selection and frame access

The app's setup flow selects a predefined source or a custom upload/library item. `ui/media_library.py` resolves source paths and deterministic per-video paths for saved calibration/cache data. `ui/state.py` owns application-level source and analysis state. `perception/video_stream.py` wraps OpenCV capture and includes synchronized caches/prefetch mechanisms so playback and seeking do not have to decode and process every requested frame from scratch.

There are two conceptually different frame products:

- The **raw image frame** is pixel data used for display and, in live mode, inference.
- The **analysis record** is a compact frame-indexed bundle of detections/positions/geometry that can be stored and reused.

Precomputation (`perception/precompute.py`) separates expensive perception from numerical work: it reads the video sequentially and runs detection/projection, then can send coordinate-only batches to worker processes for geometry calculations. This design aims to avoid loading a neural network into every numerical worker. It is an architectural design choice; actual performance still depends on video resolution, model runtime, CPU/GPU availability, and worker configuration.

#### 4.2 Detection and appearance-based team assignment

`perception/detector.py` wraps the Ultralytics YOLO API. The class chooses a model path based on available local files, runs inference with a confidence threshold and optional region of interest, then maps model classes into player, goalkeeper, referee, and ball-like outputs. Model class names can differ, so any report describing class semantics should identify the exact model used.

For people detections, the project derives a ground-contact anchor from the box near its lower edge. This is a more appropriate point for a ground-plane transform than the bounding-box center, but remains an estimate: detector boxes can include shadows, motion blur, occlusion, or partial bodies.

`perception/jersey_classifier.py` samples a central torso region from a player crop, excludes selected green/skin-like pixels, and builds a color feature from CIELAB and saturation statistics. OpenCV k-means groups the outfield candidates into two color clusters. The assigned “Team 0” versus “Team 1” names are canonical labels for display, not semantic club/team identities unless separately configured. Explicitly tagged referees and chromatic outliers can be assigned a separate team ID.

This grouping is per-frame appearance classification, not a temporal tracker. Two teams with similar kits, poor lighting, severe occlusion, very small detections, or an unusual referee kit can confuse the assignment. The report should avoid asserting perfect team separation.

#### 4.3 Calibration, projection, and coordinate semantics

Calibration links clicked or otherwise selected source-image landmarks to known metric pitch landmarks. `perception/pitch_model.py` provides standard markings and coordinates. `perception/homography.py` supports:

- A **planar projective transform** represented by a 3 x 3 homography.
- A **universal/lens-distortion mode** that combines an optical radial distortion model with a projective transform.
- A **panorama TPS mode** that adds a thin-plate-spline residual model to accommodate non-planar image deformation.
- An **auto mode** that may retain a planar solution or attempt a curved-camera mode based on the number of correspondences and planar residual.

The familiar homography relation can be described using homogeneous coordinates:

`[X', Y', w]^T = H [u, v, 1]^T`, with `(X, Y) = (X'/w, Y'/w)`.

Here, `(u, v)` are pixels and `(X, Y)` are pitch coordinates, subject to the source/destination convention used by the transformer. The inverse mapping is used to paint pitch-space polygons back onto the image. When calibration points are noisy or poorly distributed, projection error can grow away from the clicked points. A low fitting residual at the calibration anchors alone does not establish accuracy on unseen pitch locations.

In the standard model, `X` runs along the 105 m length and `Y` along the 68 m width. The origin is at one pitch corner and the center is `(52.5, 34.0)`. Team attack direction is configurable for some tactical modules; coordinate `X` itself does not change direction.

Only points intended to lie on the pitch ground plane should be interpreted as metrically projected by a planar homography. A player's feet can approximately meet that assumption. The center of a ball in the air does not; the same limitation applies to airborne parts of a player. The transform is also tied to the camera pose at calibration time, so moving/zooming cameras can make it stale.

#### 4.4 Tactical geometry and interpretation

`algorithms/convex_hull.py` computes the smallest convex polygon containing the input positions. In the pipeline, this is applied to Team B and used as a broad defensive-block footprint. The shoelace formula yields polygon area. The source also computes an isoperimetric-style compactness ratio based on area and perimeter. A hull ignores concavities and players inside the boundary, so it is a coarse team-shape descriptor, not a full measure of defensive organization.

`algorithms/voronoi_dominance.py` gathers both teams' sites, computes a Voronoi diagram using SciPy, constructs extra mirrored boundary sites, clips cells to the pitch rectangle, and aggregates the cell areas by team. A player's Voronoi cell consists of points closer to that player than to another input site under Euclidean distance. It can serve as an intuitive space-partition visualization; it does not incorporate speed, body orientation, reaction time, the ball, pressure, or actual possession. Some documentation calls this a pitch-control model, but the report should distinguish it from the separate arrival-time grid.

`algorithms/pitch_control.py` samples a grid across the pitch. For a point `q`, the estimated arrival time of player `i` is approximately:

`T_i(q) = reaction_time + distance(player_i, q) / max(v_max, 0.1)`.

It takes the minimum time for each team and passes the difference through a logistic function to create a continuous value between zero and one. The defaults are shared by all players; this implementation does not estimate individual speed or velocity. The share of grid samples above/below thresholds produces reported team percentages. These are outputs of the specified approximation, not measured possession percentages.

#### 4.5 xT and formation utilities

`algorithms/expected_threat.py` contains a numeric 12-row by 16-column array. Coordinates are clamped to the pitch, transformed if attacking left, and mapped to a cell. Action threat is calculated by comparing the start and end cell values. “Progressive” and “killer pass” labels are threshold-based classifications in source code. The repository does not establish that this matrix was fitted to the checked-in videos or calibrated for a particular league or competition.

`algorithms/formation_detector.py` sorts player coordinates along the attack axis, identifies large longitudinal gaps, divides players into bands, and compares the resulting line counts to a set of common templates. It may attempt to exclude a goalkeeper near the own goal. The output is a heuristic label from a single spatial arrangement; it does not use a sequence, role classifier, or tactical event data. Explain it as a rough positional grouping tool rather than definitive formation recognition.

#### 4.6 Passing, offside, and measurement utilities

`algorithms/passing_network.py` builds candidate directed links between attacking players within configured pass-distance bounds. It estimates defensive pressure using defenders' distance to the pass segment and assigns a non-negative edge weight influenced by distance, pressure, and forward progress. Dijkstra's algorithm then searches for a least-cost route between a source and target selection.

The module is callable directly and the tactical sandbox invokes algorithm utilities for hypothetical formations. However, the default `stage3_geometry` path sets the Dijkstra result to `None`, and `execute_frame_bundle` passes that geometry result into downstream analytics. Therefore:

- It is accurate to say the **repository includes** a Dijkstra passing-path module and sandbox scenarios can use passing utilities.
- It is not accurate to say a live Dijkstra route is always produced in the normal video pipeline.

`algorithms/ghost_runner.py` evaluates a proposed hypothetical coordinate against the second-deepest defender and ball X-coordinate, with separate left/right attack handling. It also compares before/after graph and territory results. This is a simplified simulation of relevant spatial conditions; it does not process a real pass event or adjudicate whether a player interfered with play.

`algorithms/tactical_ruler.py` calculates distance, X/Y offsets, angle, and candidate offside-breach measurements from supplied coordinates. It is a geometry utility, not an officiating tool.

#### 4.7 UI, browser rendering, and sandbox

The Streamlit app is not just a command-line analytics script. The UI provides a staged workflow:

1. The setup gate selects a source and relevant camera/calibration/execution settings.
2. The match-player path prepares media assets and frame-analysis payloads for a browser player.
3. The frame inspector exposes underlying detections, alignment, tactical geometry, and diagnostics.
4. The tactics board uses presets, supports importing the current frame's coordinates, and lets the user explore hypothetical arrangements.

`ui/components/assets/` contains the browser player implementation. `ui/components/tactical_sandbox_canvas.py` generates the interactive canvas document. `visualization/pitch_renderer.py` and `visualization/overlays.py` provide Python/OpenCV rendering routes. These paths make the system a human-in-the-loop analysis interface: users need to configure calibration and may need to interpret or correct the underlying assumptions.

The codebase contains both browser-payload/replay workflows and a live pipeline. Browser-native video playback can be smoother than live inference because analysis can be precomputed. A report should specify which path was used when describing latency or visual behavior.

### 5. Data and artifact lifecycle

#### Input data categories

1. **Video bytes:** built-in or user-provided match footage.
2. **Frame pixels:** BGR arrays read by OpenCV and displayed/processed.
3. **Detection records:** boxes, optional foot points, class/team labels, optional ball pixel.
4. **Pitch records:** teams' metric positions, ball coordinate, carrier index, detection count.
5. **Geometry records:** hull, area/compactness, Voronoi cells and team shares, optional graph result, latency.
6. **Derived analytics:** coach labels, pitch control grid, formation summaries, xT values, ruler measurements.
7. **Presentation payloads:** rendered frames/canvases and JSON-like data consumed by browser assets.

#### Persisted and generated artifacts

- Per-video calibration JSON is stored under the calibration directory chosen by `ui/media_library.py`.
- Custom-video precompute caches are stored under a deterministic path in the custom cache directory.
- Browser-facing payloads and prepared assets are generated as needed.
- The precomputed sample cache referenced in documentation is derived data. A second computer may not have it; the application may need to recompute it or use live analysis.
- User videos, calibrations, and generated cache payloads can be excluded by `.gitignore`. Copying only tracked Python files may therefore not reproduce a particular demonstration.
- Model files are binary inputs. If a desired weight file is not present locally, the detector's fallback behavior may try a different weight path or trigger model/package download behavior depending on installed Ultralytics configuration and network access. For repeatable offline runs, record and supply the exact weight file.

#### Privacy and data stewardship prompts for a report

Before sharing sample clips or screenshots, check their source, license, and whether they contain personal data. This document does not establish permissions for included or user-uploaded video. A report can discuss that videos are processed locally by the app, but should verify any network/model-download behavior and deployment configuration before making broad privacy or offline-operation guarantees.

### 6. Dependency and environment reference

`requirements.txt` lists these direct Python packages with the displayed minimum versions where present:

| Package | Role in the project |
|---|---|
| `numpy` | Arrays, coordinate math, grids, payload preparation. |
| `scipy` | Spatial Voronoi implementation and interpolation/optimization functionality used by selected code paths. |
| `pandas` | Benchmark result data frames. |
| `opencv-python` | Video decoding, image/color processing, calibration, rendering. |
| `ultralytics` | YOLO model API. |
| `streamlit` | Local web UI and session state. |
| `torch`, `torchvision` | PyTorch model/runtime dependencies; no fixed version is specified in this file. |
| `streamlit-image-coordinates` | UI coordinate selection interaction. |
| `onnx`, `onnxruntime` | ONNX model/session support. |

The listed requirements are not a complete lockfile: they do not pin every package to a precise version, and transitive dependencies may change. The README suggests Python 3.10, 3.11, or 3.12. Confirm actual interpreter compatibility and model execution backend on the target machine before presenting a “reproduced” result.

#### Reproduction setup

The README's basic workflow is:

```bash
git clone https://github.com/Parag949/football_c.git
cd football_c
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Then open `http://localhost:8501`. For a useful report validation run, additionally:

1. Confirm which video is actually available and record its filename, resolution, duration, and native FPS.
2. Confirm the selected model path/format and whether inference is ONNX Runtime or PyTorch.
3. Calibrate or load a saved calibration appropriate to the camera/video.
4. Record whether processing is live, precomputed, or reading cached frame bundles.
5. Record hardware, operating system, Python version, dependency versions, and the exact commands used.
6. Capture frame examples and benchmark outputs together with those configuration details.

This is a reproduction recipe, not a guarantee of success: video codecs, optional model weights, Streamlit/browser versions, and generated data availability can differ across PCs.

### 7. Source-of-truth hierarchy

When project materials disagree, a report-writing LLM should apply this order:

1. **Current executable source code** for what a function actually does.
2. **Fresh tests run against that source** for verified behavior on those test inputs.
3. **Current configuration and dependency manifests** for runtime requirements.
4. **Architecture and knowledge-base documentation** for intended design/contracts, checked against source.
5. **README and roadmap** for project framing, user workflow, and planned scope.
6. **Hackathon/demo copy** for presentation examples only; treat performance numbers, claims of universal real-time operation, and named example values as unverified unless independently measured.

Tests can establish a behavior for their test fixture; they do not prove general field performance. Likewise, a class or function named “pitch control” does not validate the model behind it.

### 8. Feature-to-evidence matrix

Use the matrix below to decide how strongly each feature can be described. The references point to source files rather than to marketing copy.

| Capability | Main evidence location | Safe report wording | Do not imply without further evidence |
|---|---|---|---|
| Streamlit application | `app.py`, `ui/` | “The project provides a Streamlit analysis interface.” | Public hosted product or multi-user production service. |
| Video input/decoding | `perception/video_stream.py`, `ui/media_library.py`, `ui/components/setup_gate.py` | “The app reads selected video files via OpenCV.” | Support for every codec/container/camera source. |
| YOLO object detection | `perception/detector.py` | “A YOLO-based detector produces object candidates.” | Stated precision/recall, robust ball detection, or reliable 22-player coverage without evaluation. |
| Team color assignment | `perception/jersey_classifier.py` | “Torso color features and clustering are used to group players.” | Stable player IDs or guaranteed correct team membership. |
| Pixel-to-pitch calibration | `perception/homography.py`, `perception/pitch_model.py` | “Interactive correspondences define a metric pitch projection.” | Globally accurate projection outside measured/validated points. |
| Panoramic correction modes | `perception/homography.py`, `scratch/test_panorama_calibration.py` | “The code includes TPS and optical distortion calibration options, with synthetic panorama tests.” | Accuracy on all real panoramic cameras or stitched feeds. |
| Defensive convex hull | `algorithms/convex_hull.py`, `core/pipeline.py` | “A selected team's positions are summarized by a convex hull.” | Complete defensive shape or a universally meaningful compactness threshold. |
| Voronoi area partition | `algorithms/voronoi_dominance.py` | “Nearest-site cells are clipped to pitch bounds and aggregated by team.” | Actual possession/control probability or Fortune's algorithm implementation. |
| Pitch control field | `algorithms/pitch_control.py` | “A grid-based approximation compares minimum arrival times under shared parameters.” | Player-specific velocity-aware control or empirically calibrated control probability. |
| Expected threat | `algorithms/expected_threat.py` | “A fixed spatial grid supports location/action threat comparisons.” | xT trained from this project's matches or predictive validation. |
| Formation labels | `algorithms/formation_detector.py` | “Longitudinal coordinate bands are matched to common line-count templates.” | Tactical role recognition or robust formation inference across time. |
| Passing graph | `algorithms/passing_network.py` | “An optional weighted graph and Dijkstra utility are implemented.” | Default pipeline always returns a live route; its `stage3_geometry` path sets none. |
| Ghost runner / offside | `algorithms/ghost_runner.py` | “The sandbox evaluates hypothetical coordinates against a simplified offside line.” | Official/legal decision-making or full Law 11 interpretation. |
| Tactical ruler | `algorithms/tactical_ruler.py` | “The app can calculate distances and coordinate differences between points.” | Measurement precision independent of calibration accuracy. |
| Browser replay | `ui/components/hardware_player.py`, `ui/components/assets/` | “A browser player synchronizes video and analytical overlays, including precomputed payload paths.” | End-to-end inference at 60 FPS on arbitrary hardware. |
| Precomputation | `perception/precompute.py` | “The project can prepare analysis bundles for later playback.” | A specific throughput or latency without measured run logs. |
| Algorithm benchmarks | `algorithms/daa_benchmarks.py`, diagnostics view | “The app includes synthetic-input benchmark routines.” | Benchmark values as real-match throughput or portable guarantees. |
| Automated testing | `scratch/test_*.py` | “Regression scripts cover multiple subsystems.” | Tests passed unless the exact commands were run and results recorded. |

### 9. Testing and evaluation guide

#### Test categories visible in the repository

| Area | Representative scripts | What the scripts appear intended to exercise |
|---|---|---|
| Contracts and broad invariants | `scratch/test_knowledge_base_contracts.py` | Null safety, serialization, projection, hull/Voronoi bounds, offside, cache/state contracts, browser syntax checks. |
| Core geometry/pipeline | `scratch/verify_coordinates.py`, `scratch/test_all_6_bugfixes.py`, `scratch/test_remediation_fixes.py` | Coordinate round trips, filtering, rendering, and regression cases. |
| New analytic modules | `scratch/test_new_tactical_modules.py` | Pitch-control field, xT lookup, heuristic formations, jersey colors/referee isolation. |
| Calibration | `scratch/test_landmark_anchor.py`, `scratch/test_multipoint_calib.py`, `scratch/test_panorama_calibration.py` | Landmark validity, multipoint calibration, synthetic curved-panorama fitting and round trips. |
| UI/state | `scratch/test_state_machine_consolidation.py`, `scratch/test_panorama_ui_state_sync.py`, `scratch/test_streamlit_apptest.py` | Session-state behavior, calibration synchronization, and app navigation/rendering. |
| Playback and cache | `scratch/test_async_prefetch.py`, `scratch/test_ring_buffer_fw13.py`, `scratch/test_forward_projection_caching.py`, `scratch/test_hardware_player.py` | Prefetch, bounded frame cache, projection payload/cache behavior, HTML/JS assets. |
| Passing/offside/ruler/sandbox | `scratch/test_dijkstra_passing.py`, `scratch/test_ghost_runner.py`, `scratch/test_tactical_ruler.py`, `scratch/test_tactical_sandbox.py` | Algorithm utility behavior on synthetic coordinates and preset scenarios. |
| Audit helpers | `scratch/audit_suite.py`, `scratch/deep_bug_hunter.py`, `scratch/verify_fixes.py` | Additional diagnostic or static checks; inspect script behavior before treating outputs as formal tests. |

#### How to report tests accurately

For every executed script, report:

- exact command and working directory;
- date/time, OS, Python version, and key package versions;
- exit code and pass/fail count, including skipped or expected failures;
- whether the test used synthetic fixtures, sample footage, a user clip, or cached results;
- model weights and calibration used, where applicable;
- missing assets or setup steps that changed the test path;
- what the test actually asserts and what it does not establish.

A test that synthesizes a curved panorama can support a statement about fitting that synthetic test geometry, not a claim about all real panoramic cameras. A test that checks a valid app render does not establish detection quality. A synthetic benchmark measures the configured synthetic function call, not full video throughput.

#### Example commands

The README explicitly lists:

```bash
python3 scratch/test_new_tactical_modules.py
python3 scratch/test_knowledge_base_contracts.py
python3 scratch/test_state_machine_consolidation.py
python3 scratch/test_all_6_bugfixes.py
python3 scratch/test_streamlit_apptest.py
```

Additional scripts in the test table may be useful for a targeted evaluation. Check each test's imports and asset requirements before running. In particular, UI tests may depend on the presence of bundled video/cache files and a functioning model setup; do not interpret a test failure caused by missing local data as an algorithmic failure without diagnosing it.

### 10. Evaluation plan for a credible project report

If a report needs actual results, collect them on the target machine rather than copying figures from the README or demo guide.

#### 10.1 Detection evaluation

- Select representative frames across near/far pitch regions, different kits, occlusions, camera zooms, and lighting.
- Annotate ground-truth player/ball boxes or points.
- Compare predicted objects with labels using a stated matching rule (for example IoU threshold for boxes).
- Report precision, recall, and F1 separately for players and ball; disclose ignored/ambiguous objects.
- Evaluate team classification separately from detection.
- State the exact model weight, confidence threshold, image size, ROI use, and video resolution.

#### 10.2 Calibration evaluation

- Record all clicked image points and their known pitch correspondences.
- Reserve additional visible landmarks/frames as hold-out checks if possible.
- Report mean, median, and maximum pixel reprojection error for image-space checks, and metric coordinate error only if independent ground-truth pitch points are available.
- Compare planar and nonlinear modes on the same points and test locations.
- Include camera movement, lens type, calibration point count/distribution, and any points excluded.
- Avoid claiming zero error or sub-pixel general accuracy merely because calibration points are fit/interpolated.

#### 10.3 Tactical metric evaluation

- For hull and Voronoi, use hand-constructed cases with known expected properties (e.g. rectangle area, boundary clipping, symmetric arrangement).
- For pitch-control, explain `v_max`, reaction time, grid dimensions, threshold, and the fact that shared values are assumed.
- For formation detection, prepare representative labeled arrangements and report confusion cases; do not assess only the presets that match the built-in templates.
- For xT, describe the static grid and thresholds; an accuracy study would need a separate labeled event dataset and appropriate scoring design.
- For offside, compare only the stated geometric model; actual Law 11 decisions require more context than a single coordinate check.

#### 10.4 Runtime and usability evaluation

- Separate model inference latency, geometry time, Python end-to-end frame processing, precomputation throughput, and browser display FPS.
- Warm up model/runtime before measuring and state whether timing includes image decode, rendering, and serialization.
- Report median and tail latency over a defined number of frames, not a single best frame.
- Record CPU/GPU model, thread settings, ONNX/PyTorch backend, resolution, and batch/stride options.
- For interactive usability, define tasks (select video, calibrate, inspect frame, import to sandbox) and record completion observations rather than calling a UI “intuitive” without evidence.

### 11. Recommended report outline with writing notes

This expanded outline is suitable for a university project report, capstone report, or detailed technical synopsis. It can be shortened for a one-page abstract.

#### Front matter

- **Title:** “TacticalVision Pro: Computer-Vision-Assisted Spatial Analytics for Football Video”
- Student/author, course/department, institution, supervisor, date: obtain from the user; do not invent.
- Abstract: motivation, objective, implemented method, validation actually performed, and limitation in roughly 150–250 words (adapt to submission rules).
- Keywords: football analytics, computer vision, camera calibration, homography, computational geometry, Streamlit, tactical visualization.

#### Chapter 1 — Introduction

- Explain the interest in spatial team analysis and the accessibility gap addressed by the prototype.
- State the project aim with bounded language: create a workflow that estimates and visualizes tactical geometry from a single video.
- List concrete objectives, such as detection, calibration, metric projection, geometric analysis, and interactive visualization.
- State the scope and non-goals (no robust identity tracking, no official decisions, no proven universal accuracy).

#### Chapter 2 — Background and related concepts

- Briefly explain object detection and why boxes/anchors are needed.
- Explain the difference between image coordinates and pitch coordinates.
- Introduce planar homography and why calibration correspondences are needed.
- Explain convex hull and Voronoi diagrams in plain language.
- Distinguish geometric nearest-site territory from arrival-time pitch control.
- Present xT as a fixed spatial-value lookup and explain the generic idea without asserting project-specific model training.
- If comparing commercial systems or datasets, use independent citations and verify license/attribution. `docs/DATA_SOURCES.md` is only a list of possible sources, not evidence of integration.

#### Chapter 3 — Requirements and design

- Functional requirements: choose/load video, configure calibration, analyze a frame, view metrics/overlays, and run sandbox scenarios.
- Non-functional goals: modular stages, reuse of precomputed analysis, responsive interaction, and testable contracts.
- Explain `core/contracts.py` as the boundary between stage-specific representations.
- Include a pipeline diagram based on current source, not a stale diagram that suggests every analytic module runs in the main stage.
- Document design tradeoffs: Python/Streamlit iteration speed, OpenCV processing, browser rendering, and model inference cost.

#### Chapter 4 — Implementation

- Describe the app boot sequence and setup gate.
- Describe frame loading, detection, jersey grouping, foot anchoring, and the limitations of detections.
- Describe calibration modes and coordinate conventions.
- Describe geometry and analytic modules with formulas/assumptions.
- Describe caches and precompute separately from live execution.
- Describe rendering and the sandbox as distinct interaction modes.
- State exact dependencies/runtime/model choices for the tested configuration.

#### Chapter 5 — Testing methodology

- Enumerate scripts actually executed and their outcomes.
- Classify each test as unit-style, synthetic integration, UI integration, or real-footage evaluation.
- Describe hardware, software environment, videos, calibration, model, and frame selection.
- Clearly say if only automated/synthetic tests were run and no annotated real-match benchmark was performed.

#### Chapter 6 — Results and discussion

- Insert only measured outputs from the evaluation plan.
- Include figures/screenshots with captions, frame IDs, camera mode, and whether data is cached or live.
- Discuss failure cases (occlusion, small ball, team kit similarity, moving camera, distant players).
- Compare expected and observed behavior; do not present one demo screenshot as statistical accuracy.

#### Chapter 7 — Limitations and future work

Potential future work described in `docs/ROADMAP.md` includes:

- **FW-02 temporal coordinate smoothing** — reduce frame-to-frame jitter with a temporal filter.
- **FW-03 ball trajectory extrapolation** — estimate short gaps during missed ball detections.
- **FW-07 velocity-weighted pitch control** — account for player motion/momentum.

The roadmap lists other items as “graduated” or completed. Treat those as roadmap assertions and verify the current source/tests before repeating their “production” status. Other reasonable future-work topics that a report may recommend (but must label as recommendations, not existing features) include persistent player tracking, dynamic camera-motion compensation, annotated real-footage evaluation, robust ball tracking, and dependency/model reproducibility.

#### Chapter 8 — Conclusion

- Summarize the implemented pipeline and interfaces.
- State what the evaluation demonstrated, scoped to test inputs.
- End with limitations and the next validation/engineering milestone rather than claiming professional readiness.

### 12. Report style guide and terminology

- Prefer **“estimates,” “projects,” “visualizes,” “approximates,” “supports,”** and **“prototype”** over **“tracks accurately,” “guarantees,” “proves,”** or **“real-time”** unless evidence justifies the stronger word.
- Use **football** or **soccer** consistently based on the intended audience; define the choice once.
- Use **Team A/Team B** or **attacking/defending side** unless actual team identities are known.
- Distinguish **detection** (per-frame object candidates) from **tracking** (identity/trajectory association over time).
- Distinguish **calibration anchors** (points used to fit a transform) from **validation points** (independent points used to test it).
- Distinguish **pitch-space location** from image-space bounding boxes.
- Distinguish **Voronoi territory share**, **pitch-control grid share**, **ball possession**, and **actual tactical control**; these are not interchangeable.
- Use SI units: metres (m), square metres (m²), pixels (px), milliseconds (ms), frames per second (FPS).
- Treat “FIFA” and “IFAB” dimensions as standard pitch reference values in code; avoid implying official certification by those organizations.
- Avoid saying the application works with “any video” without qualification. Formats and results depend on decoder, frame geometry, camera view, model, and calibration.
- Do not describe a Python `scipy.spatial.Voronoi` call as a hand-coded Fortune sweep-line implementation.
- Do not call the implemented pitch-control formula “Gaussian” or “radial influence” unless referring to another explicit implementation; the inspected `compute_pitch_control_field` uses distance-based arrival times and a logistic mapping.
- Do not describe xT as a learned Markov model in this repository unless an independent source confirms how the fixed grid was produced. Source code here demonstrates a lookup array and lookup/delta operations.

### 13. Reusable prompt for generating a formal report

The person using this file can paste the following prompt into another LLM and fill in the bracketed information:

> You are writing a technically accurate report about the software project described in the attached `LLM_PROJECT_CONTEXT.md`. Use that document as the primary project reference. Do not invent author, institution, hardware, test results, model accuracy, timing results, datasets, or citations. If an item is unknown, mark it `[NEEDS USER INPUT]` or omit it. Clearly distinguish implemented code, test-verified behavior, documentation claims, and future work. In particular, do not claim persistent player identity tracking; do not conflate nearest-site Voronoi territory with the separate simplified arrival-time pitch-control field; describe xT as a fixed lookup grid; and note that the default pipeline sets the Dijkstra result to `None`, even though the passing-network module exists. Treat performance claims from promotional documents as unverified unless I provide measurements. Write the report for [AUDIENCE/COURSE], at approximately [WORD COUNT] words, using [CITATION STYLE] and this structure: [SECTIONS]. Before drafting, list any missing information that would materially affect the report, and then proceed using explicit placeholders for those details.

### 14. Reusable prompt for a synopsis or abstract

> Using the attached project context, write (1) a one-sentence project description, (2) a 100–150 word synopsis, and (3) a 200–250 word abstract for TacticalVision Pro. Describe it as a prototype Streamlit football video analysis workstation combining YOLO-based detections, manual camera calibration, metric pitch projection, geometry, and visual analytics. Keep claims bounded by the code evidence. Mention at least one limitation: calibration dependence, lack of persistent identity tracking, moving-camera limitations, or simplified analytics. Do not invent results, names, affiliations, benchmark numbers, or citations. Return the three pieces under separate headings and include a short “facts to confirm” list.

### 15. Reusable prompt for a presentation or viva

> Prepare a [NUMBER]-slide presentation and a short viva script using the attached context. Explain the pipeline in plain language and then give a technically accurate version. Separate the core live pipeline from optional analytics and sandbox utilities. Include one slide on assumptions/limitations, one on the tests actually run (leave results blank unless supplied), and one on future work. Replace any unsupported numerical claims with `[MEASURE LOCALLY]`. Add likely examiner questions about calibration, homography, Voronoi, pitch control, xT, detection versus tracking, and reproducibility.

### 16. Information a future report author should request from the project owner

The source repository cannot fill in all report-specific details. Ask the owner for:

- author names, IDs, course, department, institution, supervisor, and submission date;
- required report length, template, chapter headings, citation format, and academic-integrity requirements;
- which branch/commit or release should be described;
- target audience and whether to emphasize computer vision, DAA, software engineering, or coaching use;
- which video(s) were actually tested, their source/permission, and whether they may be reproduced in a report;
- exact model file, model provenance/license, inference backend, and thresholds;
- calibration mode and points for each tested clip;
- machine specifications, environment versions, measured latencies, and full test logs;
- annotated ground truth or independent validation data, if any;
- screenshots/results and which UI mode generated them;
- whether the desired document should be a synopsis, abstract, full report, proposal, poster, or viva notes.

If the owner cannot provide these details, write only a repository-based technical description and explicitly state that empirical evaluation and project metadata were not supplied.

### 17. Final fact-check checklist for a generated report

Before handing in a report made from this context, verify each item:

- [ ] Project name and application type are correct.
- [ ] The work is described as a prototype unless evidence supports a stronger maturity claim.
- [ ] The source video, model, calibration, and execution mode are specified for any results.
- [ ] Detection and persistent tracking are not conflated.
- [ ] Team color clustering is not presented as guaranteed team identity.
- [ ] Metric projection is described as calibration-dependent and ground-plane-based.
- [ ] The homography mode and validation method are stated, if calibration accuracy is claimed.
- [ ] SciPy Voronoi is not mislabeled as a custom Fortune implementation.
- [ ] Voronoi territory, simplified pitch control, possession, and tactical control are not treated as synonyms.
- [ ] xT is described as a fixed grid unless external provenance is provided.
- [ ] Formation output is described as heuristic.
- [ ] Dijkstra is not claimed to run in the default geometry stage.
- [ ] Sandbox simulations are distinguished from observations of a real match.
- [ ] Performance values come from a reproducible measurement and distinguish browser playback from inference.
- [ ] Test claims list exact commands and observed outcomes; test presence is not described as test success.
- [ ] Roadmap items are called planned work unless source and tests verify implementation.
- [ ] Data-source directory links are not misrepresented as data integrations.
- [ ] No authorship, institutional details, external citations, licenses, or results have been fabricated.

### 18. Project-specific engineering choices: what is distinctive and why

This section answers a different question from “what algorithms are in the project?” It records decisions that make this particular implementation more tailored than a generic YOLO-on-video demo. These are **project-specific design contributions and integration choices**, not claims that the underlying ideas are new to computer vision or sports analytics as a whole. A formal report may call them “the project's distinguishing implementation choices,” “engineering contributions,” or “features tailored to this application.” Do not call them globally novel research unless a separate literature review establishes that.

#### 18.1 Calibration-guided detection region instead of blindly processing the whole frame

**The specific problem:** In wide or panoramic footage, much of a frame can be sky, stands, advertising, or foreground outside the playable surface. If a fixed-size detector resizes the full wide frame, players can become very small. Running several overlapping image tiles (a common high-recall strategy) may improve resolution but requires multiple model passes.

**What this project does:** `stage1_perception` passes the active transform to the detector. `HomographyTransformer.get_pitch_roi()` samples the pitch boundary/interior, projects those samples into image space, and calculates a padded rectangular crop. The detector prefers this calibrated ROI; for an uncalibrated very wide image it has a central-height fallback crop. It then maps crop-local detections back to full-frame coordinates using the crop offsets.

**Why it was chosen:** It uses information already entered for camera calibration to focus the detector where players are expected to be, while retaining one inference call rather than multiplying inference through tiled passes. The crop includes extra vertical padding so player bodies near the far touchline are not cut off merely because their feet lie on the pitch boundary.

**What is specific here:** Perception is explicitly coupled to the current pitch calibration: calibration is not only for converting detections after inference; it can also guide where inference is run.

**Caveats:** The ROI is a rectangle around a projected pitch region, not a perfect semantic pitch mask. A poor or stale calibration can crop out valid players. The fallback wide-frame crop is a heuristic. Do not claim a measured recall increase unless one has been measured on annotated footage.

**Safe report phrasing:** “The implementation reuses the active pitch calibration to crop the detector's input region, aiming to spend image resolution on the playing surface without requiring multiple tiled inference passes.”

#### 18.2 A laptop-oriented panoramic detector path with guarded lower confidence

**The specific problem:** Players far from a wide-angle camera occupy few pixels and may receive lower detector confidence than large nearby players. A single high threshold can drop them; lowering the threshold without safeguards can add many false positives.

**What this project does:** In `perception/detector.py`, aspect ratios above a threshold select a panoramic inference size `(480, 1536)` instead of the standard `640`. Adaptive inference can lower the confidence threshold to at most `0.16`. Lower-confidence player candidates receive shape/size filtering, and the detector's processing path handles the ROI and coordinate offsets.

**Why it was chosen:** This is a compromise for a CPU/laptop workflow: increase the effective resolution for ultrawide sources and admit faint small-player candidates, while applying inexpensive geometric checks to suppress some non-player detections. It is an alternative to simply running many heavyweight crops.

**What is specific here:** The code chooses resolution and candidate filtering from the source aspect ratio and combines that with calibration-guided cropping. It is a camera-regime-specific path, rather than a single fixed detector setting for every input.

**Caveats:** These thresholds and dimensions are engineering heuristics, not universal optima. Detection quality depends on model training, video encoding, actual crop, object scale, and kits/backgrounds. Raising recall can lower precision. The repository does not establish a quantitative recall/precision improvement.

**Safe report phrasing:** “An aspect-ratio-aware inference path and lower-confidence candidate filters were added to make small players in panoramic inputs more recoverable under a constrained compute budget; their accuracy should be validated per model and video.”

#### 18.3 Multiple calibration regimes for different camera distortions

**The specific problem:** A single planar homography is appropriate for a pinhole-like view of a flat pitch, but real user footage can include wide-angle radial distortion or stitching curvature. Simply adding more planar points cannot fully model every non-projective deformation.

**What this project does:** `perception/homography.py` supports planar, universal lens-distortion, and panorama TPS calibration. The `auto` path starts with a planar fit and may attempt a nonlinear mode when enough correspondences are available and the planar residual is high. The TPS implementation combines a global projective transform with a smooth residual rather than discarding the projective structure. ROI sampling, round-trip conversion, and projection guards are also implemented around these modes.

**Why it was chosen:** The target input category includes more than one camera regime: a standard sideline broadcast, a goal-oriented view, and wide or stitched panorama footage. A mode switch makes the transformation model match the camera more closely instead of expecting one transform to handle every lens/view.

**What is specific here:** Calibration is treated as a selectable/automatic multi-regime component integrated with UI landmark presets, saved per-video calibration, pitch-mask editing, projected overlays, and the browser payload format.

**Caveats:** “Auto” is a fitting heuristic, not an oracle that identifies every camera model. Extra model flexibility can overfit sparse/noisy points. Synthetic panorama tests demonstrate behavior on constructed geometry, not general real-camera accuracy. Each mode needs independent validation on held-out points.

**Safe report phrasing:** “The system offers projective and nonlinear calibration options to accommodate different camera geometries; mode choice and accuracy remain dependent on calibration data and camera behavior.”

#### 18.4 Browser lookup tables bridge nonlinear Python calibration and interactive rendering

**The specific problem:** A nonlinear transform such as TPS is straightforward to evaluate in Python/SciPy but less convenient to reproduce exactly in a browser canvas, especially for every player, polygon edge, and mouse interaction.

**What this project does:** `HomographyTransformer.export_client_projection()` serializes matrix parameters and, for nonlinear modes, samples pitch-to-image and image-to-pitch mappings into lookup grids. The browser player uses these lookup tables and interpolation for client-side projection.

**Why it was chosen:** It preserves a shared calibration result across server-side analysis and browser-side display without requiring the browser to rebuild the Python/SciPy interpolation model. It also lets the browser handle repeated overlay drawing and coordinate conversion from compact precomputed projection data.

**What is specific here:** The project does not only store player positions for the browser; it exports a representation of the active camera transform so nonlinear overlays can be drawn in the same coordinate system as the Python pipeline.

**Caveats:** A finite lookup grid approximates a continuous mapping; grid density trades payload size against interpolation precision. It must be regenerated when the calibration or video dimensions change. LUT interpolation tests are not equivalent to a full real-video overlay accuracy study.

**Safe report phrasing:** “For nonlinear calibration modes, the server exports sampled projection lookup tables so browser-side rendering can approximate the same transform without duplicating the Python interpolation implementation.”

#### 18.5 Two-phase precomputation: model inference first, numerical geometry second

**The specific problem:** Neural model objects and their dependencies consume memory and are not good payloads to repeatedly serialize into worker processes. At the same time, a whole clip can contain many independent frames whose coordinate-only geometry calculations can be parallelized.

**What this project does:** `perception/precompute.py` reads frames sequentially and performs YOLO perception/projection in one process. It accumulates lightweight player/ball/position records, releases the detector, and then dispatches batches of coordinate payloads to a `ProcessPoolExecutor` for geometric calculations. It can write the resulting frame bundles to a disk cache.

**Why it was chosen:** Sequential video reading avoids seeking for each sampled frame; keeping the detector in one process avoids intentionally replicating the model in every geometry worker; dispatching small coordinate records reduces worker communication compared with sending full image frames.

**What is specific here:** This is a selective split at the perception/geometry boundary. It parallelizes the pure numerical part while keeping model execution and video decoding centralized, rather than attempting to parallelize the entire application indiscriminately.

**Caveats:** Perception can remain the bottleneck, worker startup and serialization have costs, and the best worker count depends on hardware. The code includes a sequential fallback. The comments' RAM and speed figures are design claims, not portable measured guarantees.

**Safe report phrasing:** “Offline analysis separates sequential video/model work from parallel coordinate-based geometry, aiming to limit model duplication while allowing independent frame computations to use multiple processes.”

#### 18.6 Separate caches for pixels, detections, and calibration-dependent broadcast state

**The specific problem:** A decoded high-resolution frame is much larger than its detection metadata. Keeping raw images in a large general-purpose cache can consume excessive memory. Also, image detections can often survive a camera recalibration, but projected pitch positions and overlays cannot.

**What this project does:** `MatchStreamManager` has distinct caches for raw decoded frames, detection records, and broadcast/projected state. The decoded-frame cache is a bounded ring/LRU-like store, while detection and broadcast data have separate bounds and locks. Loading a new video clears state, and stream-generation checks prevent stale background jobs from a prior source from populating the newly active stream's caches.

**Why it was chosen:** Separate cache lifetimes match the data dependencies:

- Raw pixels depend on the video and frame index.
- Image-space detections depend on the video/model/detector settings.
- Pitch-space projections and overlays additionally depend on the calibration.

This avoids retaining large image arrays in the same cache as compact metadata and makes it possible to invalidate camera-dependent results without necessarily discarding all work.

**What is specific here:** The cache design reflects the pipeline's coordinate stages and video-switch lifecycle rather than using one undifferentiated frame cache. It addresses both memory pressure and stale asynchronous work.

**Caveats:** Cache bounds are implementation choices, not a memory guarantee for every resolution or process. Correct invalidation depends on every relevant setting being represented in keys/signatures and invalidation events.

**Safe report phrasing:** “The stream manager separates decoded frames from detection and projection caches, with bounded storage and source-generation checks to manage memory and avoid stale prefetch results after a video switch.”

#### 18.7 Calibration signatures and per-video persistence to prevent cross-video alignment reuse

**The specific problem:** A cached analysis or calibration can look structurally valid but be wrong if reused for a different video, resolution, camera, or set of points. This is especially dangerous because the error can produce plausible-looking but spatially incorrect overlays.

**What this project does:** `ui/media_library.py` derives isolated cache/calibration paths from a video path identity. The transformer computes a calibration signature from dimensions, camera/warp parameters, pitch dimensions, and active correspondences. Cache and browser-payload code checks dimensions/signatures before reusing data.

**Why it was chosen:** Calibration belongs to a particular camera/video setup. Persisting it with a deterministic per-video identity reduces accidental reuse between unrelated uploaded clips, while a signature lets consumers detect a mismatch after calibration changes.

**What is specific here:** The signature is part of the end-to-end handoff: Python cache loading, UI video selection, and browser payload preparation all use source/calibration metadata rather than assuming that a filename or frame index alone makes a cache reusable.

**Caveats:** A signature check is only as complete as its included fields. File identity based on path/name is not cryptographic content identity. Reports should not describe it as a universal cache integrity or authenticity guarantee.

**Safe report phrasing:** “Per-video paths and calibration signatures are used to reduce stale or cross-source cache reuse; they are compatibility checks, not proof of data correctness.”

#### 18.8 Sparse frame analysis with stride-aware cache lookup

**The specific problem:** Full inference on every frame is costly, particularly on a laptop. A user may accept analysis at selected frames and still want to seek through a video smoothly.

**What this project does:** Precomputation supports a frame sampling stride. Cached bundles preserve their original frame indices; Python and browser paths build index maps and align playback requests to nearby sampled analysis records, with browser-side interpolation/alignment support in the payload path.

**Why it was chosen:** It allows a compute-versus-temporal-resolution tradeoff: increasing stride reduces the number of frames that need perception, while preserving the original timeline for video playback and UI controls.

**What is specific here:** The sampled analysis timeline is explicitly kept distinct from the source video timeline. The cache stores frame indices and stride metadata rather than treating sampled record number as video frame number.

**Caveats:** Sparse samples can make tactical markers lag, jump, or be interpolated between observations. Interpolation does not create new detections or guarantee physically correct trajectories. A report should state the stride and explain how displayed analysis frames are selected.

**Safe report phrasing:** “The precompute/cache path records frame indices and stride metadata so playback can align video time with sampled analysis, trading temporal detail for reduced compute.”

#### 18.9 Browser-native playback separated from Python inference

**The specific problem:** Streamlit reruns and Python-side frame inference are not ideal for smooth video playback. Rendering video in a browser while synchronizing overlays can give a more responsive interface, but it does not make the underlying analysis faster.

**What this project does:** The hardware player prepares a video URL and compact tactical payload for HTML/JavaScript rendering. The client can synchronize canvas drawing with video presentation callbacks and use cached frame data plus exported projection parameters.

**Why it was chosen:** Video decoding and repeated canvas drawing can be handled by the browser's media/rendering stack, reducing the need to send every displayed frame through a Python rerun. Precomputed analysis then supplies the matching annotations.

**What is specific here:** The design combines browser-native video playback, a synchronized tactical canvas, precomputed analysis, frame-index mapping, and camera projection metadata. The browser is an analysis display client, not the YOLO inference engine.

**Caveats:** Smooth playback FPS measures presentation cadence, not detector throughput or metric accuracy. A browser can play at the source rate while overlays are sparse, stale, or based on precomputed frames.

**Safe report phrasing:** “The browser player decouples video presentation from Python inference and can display precomputed annotations; playback FPS should not be reported as live analysis FPS.”

#### 18.10 Interactive sandbox as a separate what-if workspace

**The specific problem:** A coach or student may want to test formations and player movement independently of a video, without making every interaction wait on frame decoding, detector inference, or camera calibration.

**What this project does:** `ui/components/tactical_sandbox.py` provides formation presets and imports a selected frame's positions when desired. The canvas supports dragging entities and calculating tactical utilities from the edited arrangement. The code and `scratch/test_tactical_sandbox.py` cover preset schemas and synthetic geometry scenarios.

**Why it was chosen:** Separating the whiteboard from the video pipeline provides a quick way to explore hypothetical spatial arrangements and makes the algorithms testable on known coordinates without needing a video/model in every test.

**What is specific here:** It bridges two modes of reasoning: observed (or estimated) coordinates from an analyzed video and manually edited scenario coordinates. The imported frame is a starting point, while subsequent placements are hypotheses.

**Caveats:** A sandbox result is conditional on the coordinates and model assumptions supplied. It is not a prediction of what real players will do, and a synthetic preset does not validate the detector or camera calibration.

**Safe report phrasing:** “A separate browser-based tactics board supports manual what-if analysis and optional import of a video-frame snapshot, keeping hypothetical scenarios distinct from observed data.”

#### 18.11 Robustness work is focused on boundary conditions, not only the ideal frame

**The specific problem:** Geometry and UI pipelines often fail on empty teams, duplicate/collinear points, invalid coordinates, degenerate calibration, EOF seeks, null cache fields, or switching sources while prefetch is active. These edge cases are common in real interactive use even if they are rare in a curated demo frame.

**What this project does:** Contracts and tests cover cases such as non-finite hull coordinates, coincident Voronoi sites, empty inputs, behind-camera projection, singular/degenerate calibration, cache switching, null serialized records, and UI navigation/state synchronization.

**Why it was chosen:** This workstream prioritizes a usable workstation that tolerates incomplete detections and iterative calibration, rather than assuming every frame contains two complete teams and a valid transform.

**What is specific here:** Several regression scripts are organized around named bug-fix/remediation rounds and knowledge-base contracts. This reveals an engineering process focused on repeated integration failures across Python geometry, Streamlit session state, asynchronous video loading, and browser rendering.

**Caveats:** A robustness test proves only the tested condition. The report should state which scripts were actually run, and should not call the whole app “robust” based only on null-safety or synthetic tests.

**Safe report phrasing:** “The regression suite includes edge cases spanning geometry, calibration, caching, browser assets, and Streamlit state; the report should list observed test outcomes rather than infer universal reliability.”

#### 18.12 Putting the distinguishing choices together

The strongest project-specific story is not that any single component is unprecedented. It is the **combination tailored to calibrated football video on a local workstation**:

1. A calibration interface supplies geometry used both to project positions and to constrain detector input.
2. The detector has an aspect-ratio-aware path for wide footage rather than one fixed input treatment.
3. Multiple camera models are bridged into both Python analysis and browser rendering.
4. Precomputation parallelizes geometry after model inference instead of copying the model into every worker.
5. Cache types and invalidation follow the distinction between video pixels, image detections, and calibration-dependent projections.
6. A separate tactics board lets users explore hypotheses without conflating them with video observations.
7. Tests and diagnostics span not only geometric math but also calibration state, asynchronous stream behavior, and browser payloads.

This combination may be presented as the project's **integration contribution** or **application-specific engineering contribution**. To make a stronger academic novelty claim, the report author would need a literature review comparing these choices with existing sports-analytics and panorama-video systems. To make a performance claim, the author would need a controlled benchmark. To make an accuracy claim, the author would need annotated footage and independent calibration ground truth.

### 19. Suggested “contribution” text with evidence-aware wording

The following draft may be adapted for an introduction or conclusion. Keep it only if it matches the version being reported:

> The project contribution is an integrated, locally runnable workflow that connects single-video football perception to calibrated pitch-space visualization and interactive tactical exploration. Rather than treating detection as the final output, the system converts player foot-contact estimates into metric coordinates, applies geometric summaries, and presents the results in both a video-aligned view and a separate tactics board. Several implementation choices target practical constraints in the intended workstation setting: a calibration-derived region of interest and aspect-aware detection path for wide footage; selectable planar and nonlinear camera transforms; lookup-table export for client-side projection; two-phase precomputation that separates model inference from parallel geometry; and cache separation/signatures for video-, detection-, and calibration-dependent data. These are application-specific engineering choices, not claims of globally novel algorithms. Their practical accuracy and performance should be evaluated on the target hardware and representative annotated footage.

### 20. Optional contribution table for a report

| Contribution area | Project-specific implementation | Rationale | Evidence to cite in the report |
|---|---|---|---|
| Perception and camera geometry | Calibration-derived pitch ROI is passed to the detector; wide aspect ratios use a dedicated inference setting. | Focus limited compute on the field and preserve distant player detail without automatically multiplying inference passes. | `core/pipeline.py`, `perception/homography.py`, `perception/detector.py`; run a controlled detection comparison before claiming improvement. |
| Calibration coverage | Planar, universal optical, panorama TPS, and auto paths share a transformer API. | One workflow can handle more than one camera regime. | `perception/homography.py`, calibration UI modules, calibration tests. |
| Cross-runtime projection | Nonlinear projection is sampled into browser lookup grids. | Reuse the Python calibration in client-side overlay drawing. | `perception/homography.py`, `ui/components/player_assets.py`, `ui/components/assets/hardware_player.js`. |
| Compute decomposition | Inference/projection happen sequentially; coordinate-only geometry can be distributed across worker processes. | Avoid intentional model duplication while parallelizing independent numerical work. | `perception/precompute.py`; measure on the actual machine. |
| Cache lifecycle | Raw frames, detections, and projected broadcast state are separated and bounded; stream generations guard video switches. | Reduce unnecessary image-memory use and prevent stale background work. | `perception/video_stream.py`; inspect behavior under source-switch tests. |
| Cache compatibility | Per-video paths and calibration signatures are checked before reusing derived payloads. | Avoid applying an old projection to a different source or calibration. | `ui/media_library.py`, `perception/homography.py`, `ui/components/player_assets.py`. |
| Interactive exploration | Imported frame coordinates can seed a separate editable tactics board. | Support scenario analysis without claiming edited positions were observed in the match. | `ui/components/tactical_sandbox.py`, `ui/components/tactical_sandbox_canvas.py`, sandbox tests. |
| Integration validation | Regression scripts cover edge cases in math, state, calibration, streams, and browser payloads. | Detect failures at module boundaries, not only in idealized algorithm examples. | `scratch/test_*.py`; list exact run results, not just filenames. |
