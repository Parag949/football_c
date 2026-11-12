"""
Hardware Media Player Asset & Payload Preparation
=================================================
Synchronizes video symlinks, Streamlit MediaFileManager URLs, and signature-checked
compact tactical JSON payloads without embedding UI markup.
"""

import os
import json
import time
import threading
from typing import Dict, Any, List, Optional
import numpy as np

_PAYLOAD_LOCK = threading.Lock()


def _atomic_json_dump(data: Any, target_path: str) -> None:
    """Atomically writes JSON payload using temporary file and atomic filesystem replace."""
    target_dir = os.path.dirname(target_path)
    if target_dir:
        os.makedirs(target_dir, exist_ok=True)
    tmp_path = f"{target_path}.tmp.{os.getpid()}.{threading.get_ident()}"
    try:
        with open(tmp_path, "w") as f:
            json.dump(data, f)
        os.replace(tmp_path, target_path)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise


def prepare_hardware_player_assets(
    video_path: str,
    precomputed_cache_path: str,
    transformer
) -> Dict[str, Any]:
    """Prepares and validates static assets and in-memory compact frames for browser hardware media engine playback."""
    static_dir = os.path.abspath("static")
    os.makedirs(static_dir, exist_ok=True)

    abs_src_video = os.path.abspath(video_path)
    static_video_path = os.path.join(static_dir, "active_match.mp4")

    # 1. Synchronize static symlink (replace stale copies or mismatched symlink targets)
    need_sync = False
    if abs_src_video and os.path.exists(abs_src_video):
        if not os.path.exists(static_video_path) and not os.path.islink(static_video_path):
            need_sync = True
        elif not os.path.islink(static_video_path):
            need_sync = True
        elif os.path.realpath(static_video_path) != os.path.realpath(abs_src_video):
            need_sync = True

    if need_sync:
        try:
            if os.path.exists(static_video_path) or os.path.islink(static_video_path):
                os.remove(static_video_path)
            os.symlink(abs_src_video, static_video_path)
        except Exception:
            import shutil
            shutil.copyfile(abs_src_video, static_video_path)

    # 2. Register with Streamlit MediaFileManager (Bypasses all static route / proxy / port issues)
    video_url = ""
    try:
        from streamlit.runtime import Runtime
        if Runtime.exists() and abs_src_video and os.path.exists(abs_src_video):
            video_url = Runtime.instance().media_file_mgr.add(
                abs_src_video, "video/mp4", "hardware_player_active_video"
            )
    except Exception:
        video_url = ""

    if not video_url:
        video_url = "/app/static/active_match.mp4"

    # 3. Load or generate compact tactical payload
    cache_base = os.path.splitext(os.path.basename(precomputed_cache_path))[0] if precomputed_cache_path else "none"
    vid_base = os.path.splitext(os.path.basename(video_path))[0] if video_path else "none"
    payload_filename = f"tactical_payload_{cache_base}_{vid_base}.json"
    payload_path = os.path.join(static_dir, payload_filename)
    legacy_payload_path = os.path.join(static_dir, "tactical_payload.json")

    compact_frames: List[Dict[str, Any]] = []
    fps = 25.0
    total_frames = 1
    stride = 1
    v_w, v_h = 1280, 720
    has_cache = False

    if not precomputed_cache_path or not os.path.exists(precomputed_cache_path):
        # Cache file is missing: Never load an unrelated payload!
        # Probe true source video dimensions, frame count and framerate via OpenCV.
        import cv2
        cap = cv2.VideoCapture(abs_src_video)
        if cap.isOpened():
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
            total_frames = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1))
            v_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
            v_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
            cap.release()
        compact_frames = []
        has_cache = False
    else:
        has_cache = True
        needs_rebuild = True
        is_custom_calib = bool(getattr(transformer, "is_custom_calibrated", False))

        def _matching_sig(vw: int, vh: int, payload_sig: Optional[str]) -> bool:
            if hasattr(transformer, "update_dimensions") and vw > 0 and vh > 0:
                try:
                    transformer.update_dimensions(vw, vh)
                except Exception:
                    pass
            if not hasattr(transformer, "get_calibration_signature"):
                return True
            cur_sig = transformer.get_calibration_signature()
            if is_custom_calib:
                return payload_sig == cur_sig
            if payload_sig:
                return payload_sig == cur_sig
            return True

        if os.path.exists(payload_path):
            if os.path.getmtime(payload_path) >= os.path.getmtime(precomputed_cache_path):
                try:
                    with open(payload_path, "r") as f:
                        cached_payload = json.load(f)
                    c_vw = int(cached_payload.get("video_width", 1920))
                    c_vh = int(cached_payload.get("video_height", 1080))
                    frames_list = cached_payload.get("frames") or []
                    f0 = frames_list[0] if frames_list else {}
                    f0_dijk = f0.get("dijk") if isinstance(f0, dict) else None
                    has_edges = isinstance(f0_dijk, dict) and ("edges" in f0_dijk)
                    if frames_list and has_edges and _matching_sig(
                        c_vw, c_vh, cached_payload.get("calibration_sig")
                    ):
                        needs_rebuild = False
                        fps = float(cached_payload.get("fps", 29.97))
                        total_frames = int(cached_payload.get("total_frames", 180))
                        stride = int(cached_payload.get("stride", 1))
                        v_w = c_vw
                        v_h = c_vh
                        compact_frames = cached_payload.get("frames", [])
                except Exception:
                    needs_rebuild = True

        if needs_rebuild:
            with _PAYLOAD_LOCK:
                # Re-check under lock in case another thread/tab already built the payload
                if os.path.exists(payload_path) and os.path.getsize(payload_path) > 0:
                    try:
                        with open(payload_path, "r") as f:
                            cached_payload = json.load(f)
                        c_vw = int(cached_payload.get("video_width", 1920))
                        c_vh = int(cached_payload.get("video_height", 1080))
                        frames_list = cached_payload.get("frames") or []
                        f0 = frames_list[0] if frames_list else {}
                        f0_dijk = f0.get("dijk") if isinstance(f0, dict) else None
                        has_edges = isinstance(f0_dijk, dict) and ("edges" in f0_dijk)
                        if frames_list and has_edges and _matching_sig(
                            c_vw, c_vh, cached_payload.get("calibration_sig")
                        ):
                            needs_rebuild = False
                            fps = float(cached_payload.get("fps", 29.97))
                            total_frames = int(cached_payload.get("total_frames", 180))
                            stride = int(cached_payload.get("stride", 1))
                            v_w = c_vw
                            v_h = c_vh
                            compact_frames = cached_payload.get("frames", [])
                    except Exception:
                        needs_rebuild = True

                if needs_rebuild:
                    raw_cache = None
                    try:
                        with open(precomputed_cache_path, "r") as f:
                            raw_cache = json.load(f)
                    except Exception:
                        has_cache = False
                        needs_rebuild = False

                    if raw_cache is not None:
                        fps = float(raw_cache.get("fps", 29.97))
                        total_frames = int(raw_cache.get("total_frames", len(raw_cache.get("frames", []))))
                        stride = int(raw_cache.get("stride", 1))
                        raw_frames_list = raw_cache.get("frames", [])
                        if stride == 1 and len(raw_frames_list) > 1:
                            f0_idx = raw_frames_list[0].get("frame_idx", 0)
                            f1_idx = raw_frames_list[1].get("frame_idx", 1)
                            if f1_idx - f0_idx > 1:
                                stride = max(1, f1_idx - f0_idx)
                        v_w = int(raw_cache.get("video_width", 1920))
                        v_h = int(raw_cache.get("video_height", 1080))

                        if not _matching_sig(v_w, v_h, raw_cache.get("calibration_sig")):
                            # Stale cache from a previous calibration — disable overlays in memory without modifying disk
                            has_cache = False
                            compact_frames = []
                            raw_frames_list = []

                        if has_cache:
                            from algorithms.passing_network import compute_dijkstra_passing_path

                            for f in raw_frames_list:
                                if not isinstance(f, dict):
                                    continue
                                h_raw = f.get("hull") or {}
                                v_raw = f.get("voronoi") or {}
                                d_raw = f.get("dijkstra") or {}
                                team_a = f.get("team_a") or []
                                team_b = f.get("team_b") or []
                                c_idx = f.get("carrier_idx")
                                c_idx = int(c_idx) if c_idx is not None and c_idx >= 0 else -1
                                ball = f.get("ball") or [52.5, 34.0]

                                # If precomputed cache was created without Dijkstra or graph edges, compute dynamically
                                if not d_raw or not d_raw.get("graph_edges"):
                                    if len(team_a) >= 2:
                                        d_raw = compute_dijkstra_passing_path(
                                            team_a, team_b, carrier_idx=c_idx, ball_coord=ball
                                        )
                                    else:
                                        d_raw = {}

                                cf = {
                                    "i": int(f.get("frame_idx", 0) or 0),
                                    "a": [[round(x, 2), round(y, 2)] for x, y in team_a if x is not None and y is not None],
                                    "b": [[round(x, 2), round(y, 2)] for x, y in team_b if x is not None and y is not None],
                                    "ball": [round(float(ball[0]), 2), round(float(ball[1]), 2)] if len(ball) >= 2 else [52.5, 34.0],
                                    "c": c_idx,
                                    "hull": {
                                        "p": [[round(x, 2), round(y, 2)] for x, y in (h_raw.get("polygon") or []) if x is not None and y is not None],
                                        "area": round(float(h_raw.get("area", 0.0) or 0.0), 1),
                                        "comp": round(float(h_raw.get("compactness", 0.0) or 0.0), 2),
                                    },
                                    "vor": {
                                        "a_pct": round(float(v_raw.get("team_a_pct", 50.0) or 50.0), 1),
                                        "b_pct": round(float(v_raw.get("team_b_pct", 50.0) or 50.0), 1),
                                        "ca": [[[round(x, 1), round(y, 1)] for x, y in (c.get("polygon") or []) if x is not None and y is not None] for c in (v_raw.get("cells_a") or []) if isinstance(c, dict)],
                                        "cb": [[[round(x, 1), round(y, 1)] for x, y in (c.get("polygon") or []) if x is not None and y is not None] for c in (v_raw.get("cells_b") or []) if isinstance(c, dict)],
                                    },
                                    "dijk": {
                                        "path": d_raw.get("path") or [],
                                        "pts": [[round(x, 2), round(y, 2)] for x, y in (d_raw.get("corridor_points") or []) if x is not None and y is not None],
                                        "prog": round(float(d_raw.get("total_progression", 0.0) or 0.0), 1),
                                        "cost": round(float(d_raw.get("total_weight", 0.0) or 0.0), 1),
                                        "risk": round(float(d_raw.get("avg_risk", 0.0) or 0.0), 2),
                                        "edges": [
                                            {
                                                "s": int(e.get("source_idx", 0) or 0),
                                                "t": int(e.get("target_idx", 0) or 0),
                                                "sp": [round(float(e.get("source_point", [0, 0])[0]), 1), round(float(e.get("source_point", [0, 0])[1]), 1)],
                                                "tp": [round(float(e.get("target_point", [0, 0])[0]), 1), round(float(e.get("target_point", [0, 0])[1]), 1)],
                                                "w": round(float(e.get("weight", 0.0) or 0.0), 1)
                                            }
                                            for e in (d_raw.get("graph_edges") or [])[:35] if isinstance(e, dict)
                                        ]
                                    }
                                }
                                compact_frames.append(cf)

                            cur_sig = (
                                transformer.get_calibration_signature()
                                if hasattr(transformer, "get_calibration_signature")
                                else ""
                            )
                            payload = {
                                "fps": fps,
                                "total_frames": total_frames,
                                "stride": stride,
                                "video_width": v_w,
                                "video_height": v_h,
                                "calibration_sig": raw_cache.get("calibration_sig") or cur_sig,
                                "frames": compact_frames
                            }

                            _atomic_json_dump(payload, payload_path)

                            # Sync legacy path for built-in World Cup user_match compatibility
                            if os.path.basename(precomputed_cache_path) == "precomputed_user.json":
                                try:
                                    _atomic_json_dump(payload, legacy_payload_path)
                                except Exception:
                                    pass

    # Ensure transformer resolution always matches true video dimensions before serializing projection
    if hasattr(transformer, "update_dimensions") and v_w > 0 and v_h > 0:
        try:
            transformer.update_dimensions(v_w, v_h)
        except Exception:
            pass

    if hasattr(transformer, "export_client_projection"):
        proj_meta = transformer.export_client_projection()
        h_inv_list = proj_meta["H_inv"]
        h_list = proj_meta["H"]
        warp_mode = proj_meta.get("warp_mode", "planar")
        p2i_lut = proj_meta.get("p2i_lut")
        i2p_lut = proj_meta.get("i2p_lut")
    else:
        h_inv_list = [[round(float(val), 9) for val in row] for row in transformer.H_inv.tolist()]
        h_list = [[round(float(val), 9) for val in row] for row in transformer.H.tolist()]
        warp_mode = getattr(transformer, "warp_mode", "planar")
        p2i_lut = None
        i2p_lut = None

    return {
        "video_url": video_url,
        "fps": fps,
        "total_frames": total_frames,
        "stride": stride,
        "video_width": v_w,
        "video_height": v_h,
        "H_inv": h_inv_list,
        "H": h_list,
        "warp_mode": warp_mode,
        "p2i_lut": p2i_lut,
        "i2p_lut": i2p_lut,
        "frames": compact_frames,
        "has_cache": has_cache and len(compact_frames) > 0,
    }


