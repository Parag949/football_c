"""
TacticalVision Media Library & Isolated Storage Helpers
=======================================================
Manages built-in match sources, persistent custom uploaded video paths,
local library discovery, and per-video precomputed cache path resolution
without importing Streamlit or mutating shared source registries.
"""

import glob
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
    if not video_path:
        return "video"
    norm_path = os.path.normpath(video_path).replace("\\", "/")
    stem = os.path.splitext(os.path.basename(norm_path))[0]
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-") or "video"

    rel_norm = os.path.relpath(os.path.abspath(video_path), os.path.abspath(".")).replace("\\", "/")
    legacy_hash = hashlib.md5(rel_norm.encode("utf-8")).hexdigest()[:8]
    exact_slug = f"{safe_stem}_{legacy_hash}"

    # Check existing calibrations and caches to preserve exact existing slugs
    for folder in (CUSTOM_CALIBRATIONS_DIR, CUSTOM_CACHES_DIR):
        if os.path.isdir(folder):
            if os.path.exists(os.path.join(folder, f"{exact_slug}.json")) or \
               os.path.exists(os.path.join(folder, f"{exact_slug}_calib.json")) or \
               os.path.exists(os.path.join(folder, f"{exact_slug}_cache.json")):
                return exact_slug
            matches = glob.glob(os.path.join(folder, f"{safe_stem}_*.json"))
            if matches:
                # If exact didn't match, return the newest existing slug for this stem
                newest = max(matches, key=os.path.getmtime)
                base_newest = os.path.splitext(os.path.basename(newest))[0]
                clean_slug = base_newest.replace("_calib", "").replace("_cache", "")
                return clean_slug

    return exact_slug


def get_custom_cache_path(video_path: str) -> str:
    """Returns a deterministic, per-video precomputed cache path without touching built-in caches."""
    if not video_path:
        return ""
    os.makedirs(CUSTOM_CACHES_DIR, exist_ok=True)
    slug = _video_identity_slug(video_path)
    stem = os.path.splitext(os.path.basename(video_path))[0]
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-") or "video"

    candidates = [
        os.path.join(CUSTOM_CACHES_DIR, f"{slug}_cache.json"),
        os.path.join(CUSTOM_CACHES_DIR, f"{slug}.json"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c.replace("\\", "/")

    stem_matches = sorted(
        glob.glob(os.path.join(CUSTOM_CACHES_DIR, f"{safe_stem}_*.json")),
        key=os.path.getmtime,
        reverse=True
    )
    if stem_matches:
        return stem_matches[0].replace("\\", "/")

    return os.path.join(CUSTOM_CACHES_DIR, f"{slug}_cache.json").replace("\\", "/")


def resolve_video_cache_path(video_path: str, default_cache: str = "") -> str:
    """
    Resolves the canonical precomputed cache path for a video:
    1. Checks if a custom analysis cache exists in CUSTOM_CACHES_DIR.
    2. Falls back to default_cache if it exists.
    3. Returns the dedicated custom cache path for future analysis.
    """
    if not video_path:
        return default_cache or ""
    custom_p = get_custom_cache_path(video_path)
    if custom_p and os.path.exists(custom_p):
        return custom_p
    if default_cache and os.path.exists(default_cache):
        return default_cache
    return custom_p


def get_calibration_path(video_path: str) -> str:
    """Returns a deterministic per-video saved pitch alignment JSON path."""
    if not video_path:
        return ""
    os.makedirs(CUSTOM_CALIBRATIONS_DIR, exist_ok=True)
    slug = _video_identity_slug(video_path)
    stem = os.path.splitext(os.path.basename(video_path))[0]
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-") or "video"

    candidates = [
        os.path.join(CUSTOM_CALIBRATIONS_DIR, f"{slug}_calib.json"),
        os.path.join(CUSTOM_CALIBRATIONS_DIR, f"{slug}.json"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c.replace("\\", "/")

    stem_matches = sorted(
        glob.glob(os.path.join(CUSTOM_CALIBRATIONS_DIR, f"{safe_stem}_*.json")),
        key=os.path.getmtime,
        reverse=True
    )
    if stem_matches:
        return stem_matches[0].replace("\\", "/")

    return os.path.join(CUSTOM_CALIBRATIONS_DIR, f"{slug}_calib.json").replace("\\", "/")


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
    # If path was legacy without _calib, also ensure new _calib standard path is updated
    slug = _video_identity_slug(video_path)
    calib_std_path = os.path.join(CUSTOM_CALIBRATIONS_DIR, f"{slug}_calib.json").replace("\\", "/")
    if os.path.normpath(path) != os.path.normpath(calib_std_path):
        try:
            _atomic_json_dump(payload, calib_std_path, indent=2)
        except Exception:
            pass
    return path


def load_video_calibration(
    video_path: str,
    expected_width: int = 0,
    expected_height: int = 0,
) -> Optional[Dict[str, Any]]:
    """Loads saved pitch alignment for a video if present and resolution-compatible."""
    if not video_path:
        return None
    path = get_calibration_path(video_path)
    if not path or not os.path.exists(path):
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
    slug = _video_identity_slug(video_path)
    targets = [
        os.path.join(CUSTOM_CALIBRATIONS_DIR, f"{slug}_calib.json"),
        os.path.join(CUSTOM_CALIBRATIONS_DIR, f"{slug}.json"),
    ]
    for target in targets:
        if os.path.exists(target):
            try:
                os.remove(target)
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
            if (fname.startswith("tactical_payload") and fname.endswith(".json")) or \
               (fname.startswith("active_match_") and fname.endswith(".mp4")):
                try:
                    os.remove(os.path.join(static_dir, fname))
                except Exception:
                    pass
