"""
Hardware-Accelerated Browser Media Engine & Synchronized Tactical Canvas
=========================================================================
Thin Streamlit UI wrapper around extracted asset/cache preparation
(`ui/components/player_assets.py`) and standalone HTML/CSS/JS templates
(`ui/components/assets/hardware_player.{html,css,js}`).
"""

import json
import os
from functools import lru_cache
from typing import Any, Dict

import streamlit as st

from ui.components.player_assets import prepare_hardware_player_assets

_ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


@lru_cache(maxsize=4)
def _load_player_templates() -> Dict[str, str]:
    """Loads standalone HTML, CSS, and JS player template files from disk."""
    out: Dict[str, str] = {}
    for ext in ("html", "css", "js"):
        with open(os.path.join(_ASSETS_DIR, f"hardware_player.{ext}"), "r", encoding="utf-8") as f:
            out[ext] = f.read()
    return out


def render_hardware_player(telemetry: Dict[str, Any], transformer) -> None:
    """Renders the high-performance browser hardware media engine playback studio."""
    active_video = (
        telemetry.get("path")
        or telemetry.get("source_path")
        or st.session_state.get("active_video_path", "data/user_match.mp4")
    )
    fallback_cache = (
        "data/precomputed_user.json"
        if os.path.basename(active_video) == "user_match.mp4"
        else (
            "data/precomputed_broadcast.json"
            if os.path.basename(active_video) == "sample_broadcast.mp4"
            else ""
        )
    )
    cache_path = telemetry.get("cache_path") or fallback_cache

    meta = prepare_hardware_player_assets(active_video, cache_path, transformer)

    stride_val = int(meta.get("stride", 1))
    has_cache = bool(meta.get("has_cache", len(meta.get("frames", [])) > 0))
    fps_val = float(meta["fps"])
    total_f = max(1, int(meta["total_frames"]))

    templates = _load_player_templates()
    html_code = (
        templates["html"]
        .replace("__STYLE_BLOCK__", templates["css"])
        .replace("__SCRIPT_BLOCK__", templates["js"])
        .replace("__V_URL__", str(meta["video_url"]))
        .replace("__NO_CACHE_DISPLAY__", "flex" if not has_cache else "none")
        .replace("__FPS_FMT__", f"{fps_val:.2f}")
        .replace("__MAX_FRAME_IDX__", str(total_f - 1))
        .replace("__H_INV_JSON__", json.dumps(meta["H_inv"]))
        .replace("__H_JSON__", json.dumps(meta.get("H", meta["H_inv"])))
        .replace("__WARP_MODE_JSON__", json.dumps(meta.get("warp_mode", "planar")))
        .replace("__P2I_LUT_JSON__", json.dumps(meta.get("p2i_lut")))
        .replace("__I2P_LUT_JSON__", json.dumps(meta.get("i2p_lut")))
        .replace("__FPS_VAL__", str(fps_val))
        .replace("__TOTAL_F__", str(total_f))
        .replace("__STRIDE_VAL__", str(stride_val))
        .replace("__HAS_CACHE__", "true" if has_cache else "false")
        .replace("__VW__", str(meta["video_width"]))
        .replace("__VH__", str(meta["video_height"]))
        .replace("__FRAMES_JSON__", json.dumps(meta["frames"]))
    )

    if hasattr(st, "iframe"):
        st.iframe(html_code, height=920, width="stretch")
    else:
        st.components.v1.html(html_code, height=920, scrolling=False)


__all__ = ["prepare_hardware_player_assets", "render_hardware_player"]
