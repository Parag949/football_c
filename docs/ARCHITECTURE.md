# 📐 TacticalVision Pro: End-to-End System Architecture

This document defines the technical architecture, data contracts, mathematical models, stage inputs/outputs, algorithmic complexities, and execution modes of the **TacticalVision Pro** platform.

---

## 1. High-Level Architecture & Pipeline Topology

The platform is strictly organized around a 4-stage decoupled contract pipeline with a unified multi-regime calibration gate. Each stage accepts a strongly-typed contract and produces an immutable downstream contract.

```mermaid
flowchart TD
    subgraph INGEST ["0. Video Ingest Layer"]
        A1["Broadcast Footage (MP4)"] --> B1["OpenCV MatchStreamManager\n(Dual-Capture + Async Prefetch Worker)"]
        A2["User Upload (MP4/MOV)"] --> B1
        B1 -->|"raw_frame: np.ndarray (H, W, 3)\nframe_idx: int"| S1
    end

    subgraph CALIB ["Unified Multi-Regime Calibration Architecture"]
        G_MODE["Setup Gate & Stage 2: Camera Calibration"] --> P_PRESET["Verified Presets (Sideline / Goal-to-Goal)"]
        G_MODE --> P_STUDIO["Interactive Multi-Point Studio (N in {4, 5, 6, 8, 13})"]
        P_STUDIO --> P_MODES["Mode Selection:\n• Planar Homography (3x3)\n• Universal Lens Distortion (Fitzgibbon)\n• Panorama Thin-Plate Spline (TPS)"]
        P_MODES --> RESID["Residual Diagnostics (< 1.0px Sub-Pixel Fit)"]
        RESID --> S2
        P_PRESET --> S2
    end

    subgraph STAGE1 ["Stage 1: Camera Perception Engine"]
        S1["stage1_perception()"] -->|"YOLOv8 Inference\n(Dynamic Aspect for Panoramas)"| P1["Bounding Boxes (x1, y1, x2, y2)"]
        P1 -->|"Ground Contact Anchor"| P2["Cleat Foot Points (u, v)"]
        P2 -->|"Torso CIELAB Clustering\n& Referee Isolation"| P3["Team Assignment (0=A, 1=B, -1=Ref)"]
        P3 --> C1[("FrameDetections Contract\n• players: List[PlayerDetection]\n• ball_pixel: (u, v)\n• frame_width, frame_height")]
    end

    subgraph STAGE2 ["Stage 2: Homography & Projection Engine"]
        C1 --> S2["stage2_projection()"]
        M1["FIFA Pitch Model (105m x 68m)"] --> S2
        S2 -->|"Perspective Matrix H / Universal / TPS"| PR1["Metric Projection: (u, v) ➔ (X_m, Y_m)"]
        PR1 -->|"Touchline Horizon Filter"| PR2["Filter Off-Pitch Dugout & Behind-Camera Points"]
        PR2 --> C2[("PitchPositions Contract\n• team_a: List[(X_m, Y_m)]\n• team_b: List[(X_m, Y_m)]\n• ball: (X_m, Y_m)\n• carrier_idx: int")]
    end

    subgraph STAGE3 ["Stage 3: Computational Geometry & Spatial Intelligence"]
        C2 --> S3["stage3_geometry()"]
        S3 -->|"Graham Scan / Jarvis March"| G1["Defensive Block Compactness"]
        S3 -->|"Fortune's Sweep-line"| G2["Voronoi Spatial Pitch Control"]
        S3 -->|"Gaussian Radial Influence"| G3["Continuous Pitch Control Field"]
        S3 -->|"16x12 Threat Matrix"| G4["Expected Threat (xT)"]
        G1 & G2 & G3 & G4 --> C3[("TacticalGeometry Contract\n• hull_polygon: List[(X_m, Y_m)]\n• hull_area, hull_compactness\n• voronoi_cells_a, voronoi_cells_b\n• team_a_pct, team_b_pct\n• pitch_control: Dict\n• expected_threat: Dict")]
    end

    subgraph STAGE4 ["Stage 4: Visualization & Rendering Engine"]
        C2 & C3 --> R1["stage4_render_radar()"]
        C2 & C3 --> R2["stage4_render_overlay()"]
        R1 --> V1["2D Tactical Radar Canvas\n(Continuous Influence, Threat Cells, Formations)"]
        R2 --> V2["Broadcast AR Perspective Overlay\n(On-pitch player tags, block perimeters)"]
    end

    subgraph APPS ["Application Consumers"]
        V1 & V2 --> APP1["Hardware-Accelerated Dual Canvas (hardware_player)"]
        APP1 --> APP2["Interactive 2D Tactics Board (tactical_sandbox)"]
        C1 & C2 & C3 --> PRE["Multi-Core Precompute Engine (precompute.py)"]
    end
```

---

## 2. Standardized Data Contracts

All stage interfaces are formalized in `core/contracts.py`:

```python
@dataclass
class PlayerDetection:
    box: Tuple[int, int, int, int]         # [x1, y1, x2, y2] in pixels
    foot_point: Tuple[float, float]        # (u, v) ground contact in pixels
    team_id: int                           # 0: Team A, 1: Team B, -1: Referee
    confidence: float                      # YOLO detection confidence score (0.0 to 1.0)
    class_id: int = 0                      # 0: Player, 3: Referee, 32: Sports Ball

@dataclass
class FrameDetections:
    frame_idx: int                         # Timeline frame number
    players: List[PlayerDetection]         # Detected outfield players, goalkeepers & referees
    ball_pixel: Optional[Tuple[float, float]] = None # Ball ground pixel (u, v)
    frame_width: int = 1280                # Frame pixel width
    frame_height: int = 720                # Frame pixel height

@dataclass
class PitchPositions:
    frame_idx: int
    team_a: List[Tuple[float, float]]      # Metric positions (X_m, Y_m) in [0..105]x[0..68]
    team_b: List[Tuple[float, float]]      # Metric positions (X_m, Y_m) in [0..105]x[0..68]
    ball: Optional[Tuple[float, float]] = (52.5, 34.0)
    carrier_idx: Optional[int] = 0         # Index of closest possession player to ball
    detected_count: int = 0

@dataclass
class TacticalGeometry:
    frame_idx: int
    hull_polygon: List[Tuple[float, float]] = field(default_factory=list) # Defensive perimeter
    hull_area: float = 0.0                 # Defensive block footprint (m²)
    hull_compactness: float = 0.0          # Isoperimetric quotient (0.0 to 1.0)
    hull_vertices: int = 0                 # Count of perimeter players
    hull_player_indices: List[int] = field(default_factory=list)
    voronoi_cells_a: List[List[Tuple[float, float]]] = field(default_factory=list)
    voronoi_cells_b: List[List[Tuple[float, float]]] = field(default_factory=list)
    team_a_pct: float = 50.0               # Territory pitch dominance percentage
    team_b_pct: float = 50.0               # Territory pitch dominance percentage
    centroid_team_a: Tuple[float, float] = (52.5, 34.0) # Mathematical Center of Mass
    centroid_team_b: Tuple[float, float] = (52.5, 34.0)
    dijkstra_res: Optional[Dict[str, Any]] = None       # Backwards-compatible contract field
    latency_ms: float = 0.0
```

---

## 3. Execution Modes & Performance

TacticalVision Pro supports three unified execution modes:

1. **`Instant Replay (Saved Analysis)`**:
   - High-throughput client playback at 60 FPS utilizing the browser hardware player (`hardware_player.html` / WebGL).
   - Reads serialized frame bundles (`positions`, `detections`, `geometry`) from persistent JSON caches (`data/custom_caches/` or `data/precomputed_user.json`).
   - Zero CPU inference overhead during scrubbing or playback.

2. **`Analyze Full Clip Now (Precomputation)`**:
   - Parallel batch analysis engine using Python's `ProcessPoolExecutor` with configurable CPU workers ($N \in [1, \text{cpu\_count}]$).
   - Computes player detections, kit clustering, foot anchoring, and tactical geometry with an $O(1)$ sampling stride ($S \in [1, 5]$).
   - Automatically saves results to disk for instant future sessions.

3. **`Live Frame-by-Frame Mode`**:
   - Live on-demand pipeline execution as frames are requested.
   - Ideal for interactive calibration tweaking and live camera inspection.
