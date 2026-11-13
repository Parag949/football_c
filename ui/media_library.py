"""
TacticalVision Media Library & Isolated Storage Helpers
=======================================================
Manages built-in match sources, persistent custom uploaded video paths,
local library discovery, and per-video precomputed cache path resolution
without importing Streamlit or mutating shared source registries.
"""

import hashlib
import json
import os
import re
import threading
from typing import Any, Dict, List, Optional, Tuple


CUSTOM_VIDEOS_DIR = "data/custom_videos"
CUSTOM_CACHES_DIR = "data/custom_caches"
CUSTOM_CALIBRATIONS_DIR = "data/calibrations"
LOCAL_VIDEOS_DIR = "data/videos"
_SUPPORTED_VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv")

VIDEO_SOURCES: Dict[str, Dict[str, str]] = {
    "premier_league": {
        "title": "Premier League Broadcast (30s Sample)",
        "path": "data/sample_broadcast.mp4",
        "cache": "data/precomputed_broadcast.json",
        "default_mode": "sideline",
    },
    "world_cup": {
        "title": "World Cup Tactical Cam (48s Sample)",
        "path": "data/user_match.mp4",
        "cache": "data/precomputed_user.json",
        "default_mode": "goal_to_goal",
    },
    "custom_upload": {
        "title": "Custom Match Video (Upload / Saved Library)",
        "path": "",
        "cache": "",
        "default_mode": "sideline",
    },
}


def _video_identity_slug(video_path: str) -> str:
    stem = os.path.splitext(os.path.basename(video_path))[0]
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-") or "video"
    rel_norm = os.path.relpath(os.path.abspath(video_path), os.path.abspath(".")).replace("\\", "/")
    path_hash = hashlib.md5(rel_norm.encode("utf-8")).hexdigest()[:8]
    return f"{safe_stem}_{path_hash}"


def get_custom_cache_path(video_path: str) -> str:
    """Returns a deterministic, per-video precomputed cache path without touching built-in caches."""
    if not video_path:
        return ""
    os.makedirs(CUSTOM_CACHES_DIR, exist_ok=True)
    return os.path.join(CUSTOM_CACHES_DIR, f"{_video_identity_slug(video_path)}.json").replace("\\", "/")


def get_calibration_path(video_path: str) -> str:
    """Returns a deterministic per-video saved pitch alignment JSON path."""
    if not video_path:
        return ""
    os.makedirs(CUSTOM_CALIBRATIONS_DIR, exist_ok=True)
    return os.path.join(CUSTOM_CALIBRATIONS_DIR, f"{_video_identity_slug(video_path)}.json").replace("\\", "/")


def _atomic_json_dump(data: Any, target_path: str, indent: Optional[int] = 2) -> None:
    """Atomically writes JSON payload using temporary file and atomic filesystem replace."""
    target_dir = os.path.dirname(target_path)
    if target_dir:
        os.makedirs(target_dir, exist_ok=True)
    tmp_path = f"{target_path}.tmp.{os.getpid()}.{threading.get_ident()}"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent)
        os.replace(tmp_path, target_path)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise


def save_video_calibration(video_path: str, payload: Dict[str, Any]) -> str:
    """Persists pitch mask handle / landmark alignment for a specific video to disk atomically."""
    path = get_calibration_path(video_path)
    if not path:
        return ""
    _atomic_json_dump(payload, path, indent=2)
    return path


def load_video_calibration(
    video_path: str,
    expected_width: int = 0,
    expected_height: int = 0,
) -> Optional[Dict[str, Any]]:
    """Loads saved pitch alignment for a video if present and resolution-compatible."""
    if not video_path:
        return None
    path = os.path.join(CUSTOM_CALIBRATIONS_DIR, f"{_video_identity_slug(video_path)}.json").replace("\\", "/")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            return None
        w = int(data.get("frame_width", 0) or 0)
        h = int(data.get("frame_height", 0) or 0)
        if expected_width > 0 and expected_height > 0 and (w != expected_width or h != expected_height):
            return None
        src_pts = data.get("src_points", [])
        dst_pts = data.get("dst_points", [])
        if len(src_pts) < 4 or len(src_pts) != len(dst_pts):
            return None
        return data
    except Exception:
        return None


def delete_video_calibration(video_path: str) -> None:
    """Removes saved custom alignment for the specified video."""
    if not video_path:
        return
    path = os.path.join(CUSTOM_CALIBRATIONS_DIR, f"{_video_identity_slug(video_path)}.json").replace("\\", "/")
    if os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            pass


def build_custom_upload_paths(filename: str, upload_sig: str) -> Tuple[str, str]:
    """Generates unique, non-overwriting storage and cache paths for an uploaded custom video."""
    os.makedirs(CUSTOM_VIDEOS_DIR, exist_ok=True)
    os.makedirs(CUSTOM_CACHES_DIR, exist_ok=True)
    stem, ext = os.path.splitext(os.path.basename(filename or "custom_match.mp4"))
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-") or "custom_video"
    ext_clean = ext.lower() if ext.lower() in _SUPPORTED_VIDEO_EXTS else ".mp4"
    sig_hash = hashlib.md5(upload_sig.encode("utf-8")).hexdigest()[:8]
    video_path = os.path.join(CUSTOM_VIDEOS_DIR, f"{safe_stem}_{sig_hash}{ext_clean}").replace("\\", "/")
    cache_path = get_custom_cache_path(video_path)
    return video_path, cache_path


def list_available_custom_videos() -> List[str]:
    """Lists saved uploaded videos and local library videos, newest first."""
    found: List[Tuple[float, str]] = []
    for folder in (CUSTOM_VIDEOS_DIR, LOCAL_VIDEOS_DIR):
        if not os.path.isdir(folder):
            continue
        for fname in os.listdir(folder):
            if fname.lower().endswith(_SUPPORTED_VIDEO_EXTS):
                full_p = os.path.join(folder, fname).replace("\\", "/")
                try:
                    mtime = os.path.getmtime(full_p)
                except OSError:
                    mtime = 0.0
                found.append((mtime, full_p))
    found.sort(key=lambda item: item[0], reverse=True)
    return [p for _, p in found]


def invalidate_custom_user_cache(
    clear_user_cache_file: bool = False,
    target_cache_path: Optional[str] = None,
) -> None:
    """Clears derived browser payload caches without deleting unrelated videos or built-in caches."""
    if clear_user_cache_file and target_cache_path and os.path.exists(target_cache_path):
        try:
            os.remove(target_cache_path)
        except Exception:
            pass

    static_dir = os.path.abspath("static")
    if os.path.exists(static_dir):
        for fname in os.listdir(static_dir):
            if fname.startswith("tactical_payload") and fname.endswith(".json"):
                try:
                    os.remove(os.path.join(static_dir, fname))
                except Exception:
                    pass
