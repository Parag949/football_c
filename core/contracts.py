"""
Standardized Data Contracts (Pipeline API Schemas)
---------------------------------------------------
Decouples the system into strict, independently testable stages:
- Stage 1: FrameDetections (pixel coordinate space)
- Stage 2: PitchPositions (metric pitch meters: FIFA 105m x 68m)
- Stage 3: TacticalGeometry (Convex Hull & Voronoi partitions)
- Stage 4: Rendered Visualizations (2D Tactical Radar & AR Broadcast Overlay)
"""

from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict, Any, Optional
import json


@dataclass
class PlayerDetection:
    """Single player detection in image pixel space."""
    box: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    foot_point: Optional[Tuple[float, float]] = None  # (px_x, px_y) ground contact
    team_id: int = 0  # 0 for Team A, 1 for Team B
    confidence: float = 1.0
    class_id: int = 0
    jersey_color: Optional[Tuple[int, int, int]] = None

    def __post_init__(self):
        if self.foot_point is None and self.box is not None:
            self.foot_point = ((self.box[0] + self.box[2]) / 2.0, float(self.box[3]))



@dataclass
class FrameDetections:
    """Stage 1 Output: Raw detections in camera image pixel space."""
    frame_idx: int
    players: List[PlayerDetection] = field(default_factory=list)
    ball_pixel: Optional[Tuple[float, float]] = None
    frame_width: int = 1920
    frame_height: int = 1080

    @property
    def team_a_pixels(self) -> List[Tuple[float, float]]:
        return [p.foot_point for p in self.players if p.team_id == 0]

    @property
    def team_b_pixels(self) -> List[Tuple[float, float]]:
        return [p.foot_point for p in self.players if p.team_id == 1]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'frame_idx': self.frame_idx,
            'players': [asdict(p) for p in self.players],
            'ball_pixel': self.ball_pixel,
            'frame_width': self.frame_width,
            'frame_height': self.frame_height
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FrameDetections':
        players = [PlayerDetection(**p) for p in data.get('players', [])]
        return cls(
            frame_idx=data.get('frame_idx', 0),
            players=players,
            ball_pixel=tuple(data['ball_pixel']) if data.get('ball_pixel') else None,
            frame_width=data.get('frame_width', 1920),
            frame_height=data.get('frame_height', 1080)
        )


@dataclass
class PitchPositions:
    """Stage 2 Output: Projected metric coordinates on FIFA 105m x 68m pitch."""
    frame_idx: int
    team_a: List[Tuple[float, float]] = field(default_factory=list)  # (X, Y) in meters
    team_b: List[Tuple[float, float]] = field(default_factory=list)
    ball: Tuple[float, float] = (52.5, 34.0)
    carrier_idx: int = 0
    detected_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PitchPositions':
        return cls(
            frame_idx=data.get('frame_idx', 0),
            team_a=[tuple(pt) for pt in data.get('team_a', [])],
            team_b=[tuple(pt) for pt in data.get('team_b', [])],
            ball=tuple(data.get('ball', (52.5, 34.0))),
            carrier_idx=data.get('carrier_idx', 0),
            detected_count=data.get('detected_count', len(data.get('team_a', [])) + len(data.get('team_b', [])))
        )


@dataclass
class TacticalGeometry:
    """Stage 3 Output: Asymptotic computational geometry results."""
    frame_idx: int
    hull_polygon: List[Tuple[float, float]] = field(default_factory=list)
    hull_player_indices: List[int] = field(default_factory=list)
    hull_area: float = 0.0
    hull_compactness: float = 0.0
    hull_vertices: int = 0
    voronoi_cells_a: List[Dict[str, Any]] = field(default_factory=list)
    voronoi_cells_b: List[Dict[str, Any]] = field(default_factory=list)
    team_a_pct: float = 50.0
    team_b_pct: float = 50.0
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TacticalGeometry':
        hull = data.get('hull', {}) if 'hull' in data else data
        voronoi = data.get('voronoi', {}) if 'voronoi' in data else data
        return cls(
            frame_idx=data.get('frame_idx', 0),
            hull_polygon=[tuple(pt) for pt in hull.get('polygon', hull.get('hull_polygon', []))],
            hull_player_indices=list(hull.get('player_indices', hull.get('hull_player_indices', []))),
            hull_area=float(hull.get('area', hull.get('hull_area', 0.0))),
            hull_compactness=float(hull.get('compactness', hull.get('hull_compactness', 0.0))),
            hull_vertices=int(hull.get('vertices', hull.get('hull_vertices', 0))),
            voronoi_cells_a=voronoi.get('cells_a', voronoi.get('voronoi_cells_a', [])),
            voronoi_cells_b=voronoi.get('cells_b', voronoi.get('voronoi_cells_b', [])),
            team_a_pct=float(voronoi.get('team_a_pct', 50.0)),
            team_b_pct=float(voronoi.get('team_b_pct', 50.0)),
            latency_ms=float(data.get('latency_ms', 0.0))
        )


@dataclass
class FrameBundle:
    """Unified contract bundle for a frame, holding positions and geometry."""
    positions: PitchPositions
    geometry: TacticalGeometry

    def to_dict(self) -> Dict[str, Any]:
        return {
            'frame_idx': self.positions.frame_idx,
            'team_a': self.positions.team_a,
            'team_b': self.positions.team_b,
            'ball': self.positions.ball,
            'carrier_idx': self.positions.carrier_idx,
            'detected_players': self.positions.detected_count,
            'hull': {
                'polygon': self.geometry.hull_polygon,
                'area': self.geometry.hull_area,
                'compactness': self.geometry.hull_compactness,
                'vertices': self.geometry.hull_vertices
            },
            'voronoi': {
                'team_a_pct': self.geometry.team_a_pct,
                'team_b_pct': self.geometry.team_b_pct,
                'cells_a': self.geometry.voronoi_cells_a,
                'cells_b': self.geometry.voronoi_cells_b
            },
            'latency_ms': self.geometry.latency_ms
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FrameBundle':
        positions = PitchPositions.from_dict(data)
        geometry = TacticalGeometry.from_dict(data)
        return cls(positions=positions, geometry=geometry)
