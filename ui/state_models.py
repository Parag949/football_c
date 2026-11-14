"""
TacticalVision State Models
===========================
Typed dataclasses defining structured application, video source, playback,
calibration, analytics display, and tactical ruler state containers.
Eliminates loose session state key collisions, stale radio states, and rerun desynchronization.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional, Any, Dict, Callable


def coerce_anchor_count(val: Any, default: int = 4) -> int:
    """Safely converts Streamlit widget or session anchor counts into Python int."""
    try:
        if isinstance(val, bool):
            return default
        if isinstance(val, int):
            return val
        if isinstance(val, str):
            digits = "".join(ch for ch in val.split()[0] if ch.isdigit()) if val.strip() else ""
            if digits:
                return int(digits)
        return int(val)
    except Exception:
        return default


class StateModelBase:
    """Base class for state models providing transparent change observation and synchronization."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "_sync_cb", None)

    def set_sync_callback(self, cb: Optional[Callable[["StateModelBase", str, Any], None]]) -> None:
        object.__setattr__(self, "_sync_cb", cb)

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        cb = getattr(self, "_sync_cb", None)
        if cb is not None and not name.startswith("_"):
            try:
                cb(self, name, value)
            except Exception:
                pass

    def to_dict(self) -> Dict[str, Any]:
        """Serializes dataclass fields to standard dictionary."""
        return asdict(self)


@dataclass
class VideoSourceState(StateModelBase):
    """Encapsulates video source selection and custom upload file paths."""
    active_key: str = "world_cup"
    custom_video_path: str = ""
    custom_cache_path: str = ""
    last_uploaded_sig: str = ""
    last_uploaded_name: str = ""


@dataclass
class PlaybackState(StateModelBase):
    """Encapsulates video playback position, frame indices, and play/pause timeline state."""
    current_frame: int = 0
    is_playing: bool = False
    speed: float = 1.0
    loop: bool = True
    main_slider_val: int = 0

    def step(self, delta: int, total_frames: int) -> int:
        """Steps current frame by delta within [0, total_frames - 1]."""
        new_f = max(0, min(max(1, total_frames) - 1, self.current_frame + delta))
        self.current_frame = new_f
        self.main_slider_val = new_f
        return new_f

    def seek(self, frame_idx: int, total_frames: int) -> int:
        """Seeks to target absolute frame index within [0, total_frames - 1]."""
        new_f = max(0, min(max(1, total_frames) - 1, frame_idx))
        self.current_frame = new_f
        self.main_slider_val = new_f
        return new_f

    def toggle_play(self, total_frames: int = 1) -> bool:
        """Toggles play/pause state; rewinds if at end of clip."""
        if not self.is_playing:
            if self.current_frame >= max(1, total_frames) - 1:
                self.current_frame = 0
                self.main_slider_val = 0
            self.is_playing = True
        else:
            self.is_playing = False
        return self.is_playing


@dataclass
class CalibrationState(StateModelBase):
    """
    Unified calibration container collapsing legacy manual_calib_* and tv_calib_* keys.
    Single canonical source of truth for homography alignment status and anchor points.
    """
    is_custom: bool = False
    points: List[Tuple[float, float]] = field(default_factory=list)
    warp_mode: str = "planar"
    lens_lambda: float = 0.0
    num_anchors: int = 4
    mesh_handles: Optional[Any] = None
    template: str = "standard_half"
    just_applied: bool = False
    just_reset: bool = False
    picker_mode: str = "custom"

    def reset(self) -> None:
        """Resets custom calibration back to default planar preset."""
        self.is_custom = False
        self.points = []
        self.warp_mode = "planar"
        self.lens_lambda = 0.0
        self.num_anchors = 4
        self.mesh_handles = None
        self.just_applied = False
        self.just_reset = True

    def apply_points(
        self,
        pts: List[Tuple[float, float]],
        warp: str = "planar",
        handles: Optional[Any] = None
    ) -> None:
        """Applies active calibration points and updates state flags."""
        self.is_custom = True
        self.points = [(float(p[0]), float(p[1])) for p in pts]
        self.warp_mode = warp
        self.mesh_handles = handles
        self.num_anchors = len(pts)
        self.just_applied = True
        self.just_reset = False


@dataclass
class AnalyticsDisplayState(StateModelBase):
    """Encapsulates layer visibility toggles and tactical analytics algorithms."""
    show_hull: bool = True
    show_voronoi: bool = True
    show_dijkstra: bool = True
    show_passing_graph: bool = False
    show_ruler: bool = False
    hull_algo: str = "graham"
    attack_direction: str = "right"


@dataclass
class RulerState(StateModelBase):
    """Encapsulates tactical distance and offside line ruler tool configurations."""
    mode: str = "players"
    p1_idx: int = 0
    p2_idx: int = 1
    custom_p1: Tuple[float, float] = (35.0, 20.0)
    custom_p2: Tuple[float, float] = (65.0, 45.0)
    auto_offside: bool = True
    manual_offside_x: float = 70.0


@dataclass
class AppState(StateModelBase):
    """
    Top-level structured application state container.
    Consolidates 49 loose session state variables into strongly-typed sub-states.
    """
    setup_confirmed: bool = False
    exec_mode: str = "cache"
    video: VideoSourceState = field(default_factory=VideoSourceState)
    playback: PlaybackState = field(default_factory=PlaybackState)
    calibration: CalibrationState = field(default_factory=CalibrationState)
    display: AnalyticsDisplayState = field(default_factory=AnalyticsDisplayState)
    ruler: RulerState = field(default_factory=RulerState)
    sandbox_state: Dict[str, Any] = field(default_factory=lambda: {"imported_frame": None})

    def __post_init__(self):
        object.__setattr__(self, "_sync_cb", None)
        self._bind_children()

    def _bind_children(self) -> None:
        """Attaches notification callbacks to all child sub-states."""
        for child in (self.video, self.playback, self.calibration, self.display, self.ruler):
            if isinstance(child, StateModelBase):
                child.set_sync_callback(self._on_child_changed)

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        if name in ("video", "playback", "calibration", "display", "ruler"):
            if isinstance(value, StateModelBase):
                value.set_sync_callback(self._on_child_changed)
        cb = getattr(self, "_sync_cb", None)
        if cb is not None and not name.startswith("_"):
            try:
                cb(self, name, value)
            except Exception:
                pass

    def _on_child_changed(self, child: StateModelBase, name: str, value: Any) -> None:
        """Propagates child state mutations to the top-level sync observer."""
        cb = getattr(self, "_sync_cb", None)
        if cb is not None:
            try:
                cb(child, name, value)
            except Exception:
                pass

    def reset_playback(self) -> None:
        """Resets playback state to frame 0 and stopped."""
        self.playback.current_frame = 0
        self.playback.main_slider_val = 0
        self.playback.is_playing = False

    def reset_calibration(self) -> None:
        """Resets calibration state back to defaults."""
        self.calibration.reset()
