"""
Unified Video Stream & Perception Manager
Handles ingestion, decoding, frame caching, and YOLOv8 perception pipeline for:
1. Real Broadcast Video Footage (data/sample_broadcast.mp4)
2. User-Uploaded Match Videos (MP4 / MOV / AVI)
3. Synthetic High-Fidelity Match Simulator
"""

import os
import cv2
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from collections import OrderedDict
from perception.detector import FootballDetector
from perception.homography import HomographyTransformer
from perception.pitch_model import PitchModel


import threading
import time
from concurrent.futures import ThreadPoolExecutor

class MatchStreamManager:
    """
    Manages live match video feeds, frame extraction, and player detection.
    Includes FW-05 Asynchronous Frame Prefetching for 0 ms perceived latency during scrubbing.
    """
    
    def __init__(
        self,
        sample_video_path: str = "data/sample_broadcast.mp4",
        detector: Optional[FootballDetector] = None
    ):
        self.sample_video_path = sample_video_path
        self.detector = detector if detector is not None else FootballDetector()
        self.current_cap = None
        self.current_path = None
        self._cap_lock = threading.Lock()
        self.total_frames = 180
        self.fps = 25.0
        self.width = 1280
        self.height = 720
        self._default_transformer = HomographyTransformer(frame_width=self.width, frame_height=self.height)
        
        # LRU Cache for detected Stage 1 states & separated broadcast states
        self.detection_cache: OrderedDict[int, Dict[str, Any]] = OrderedDict()
        self.broadcast_state_cache: OrderedDict[int, Dict[str, Any]] = OrderedDict()
        self._cache_lock = threading.Lock()
        self._stream_generation: int = 0
        self._last_read_pos: int = -1
        
        # FW-05: Dedicated asynchronous lookahead prefetch pipeline
        self._prefetch_lock = threading.Lock()
        self._prefetch_cap: Optional[cv2.VideoCapture] = None
        self._prefetch_executor: Optional[ThreadPoolExecutor] = None
        self._prefetch_hits: int = 0
        self._prefetch_pending: set = set()
        self._last_prefetch_error: Optional[str] = None
        
        # FW-13: In-Memory Ring Buffer Decoded Frame Cache (Zero-OpenCV-Seek Playback)
        self._raw_cache_capacity: int = 120
        self.raw_frame_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._raw_cache_lock = threading.Lock()
        self._raw_cache_hits: int = 0
        self._raw_cache_misses: int = 0
        self._raw_prefetch_pending: set = set()
        
        if os.path.exists(sample_video_path):
            self.load_video_source(sample_video_path)
        
    def load_video_source(self, video_path: str) -> bool:
        """Loads and initializes a video capture object from file path."""
        with self._cap_lock:
            if self.current_cap is not None:
                self.current_cap.release()
                self.current_cap = None
                
            if not os.path.exists(video_path):
                return False
                
            self.current_cap = cv2.VideoCapture(video_path)
            if not self.current_cap.isOpened():
                self.current_cap = None
                return False
                
            self.current_path = video_path
            self._stream_generation += 1
            self._last_read_pos = -1
            self.total_frames = max(1, int(self.current_cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            raw_fps = float(self.current_cap.get(cv2.CAP_PROP_FPS) or 0.0)
            self.fps = raw_fps if raw_fps > 0.0 else 25.0
            self.width = int(self.current_cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
            self.height = int(self.current_cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
            if self._default_transformer is not None:
                self._default_transformer.update_dimensions(self.width, self.height)
            try:
                import streamlit as st
                if hasattr(st, "session_state") and "transformer" in st.session_state and st.session_state.transformer is not None:
                    st.session_state.transformer.update_dimensions(self.width, self.height)
            except Exception:
                pass
            self._last_prefetch_error = None
            
            with self._cache_lock:
                self.detection_cache.clear()
                self.broadcast_state_cache.clear()
                self._prefetch_pending.clear()

            with self._raw_cache_lock:
                self.raw_frame_cache.clear()
                self._raw_prefetch_pending.clear()
                self._raw_cache_hits = 0
                self._raw_cache_misses = 0

            with self._prefetch_lock:
                if self._prefetch_cap is not None:
                    self._prefetch_cap.release()
                self._prefetch_cap = cv2.VideoCapture(video_path)
                if self._prefetch_executor is None:
                    self._prefetch_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="prefetch_worker")
            return True

    def _get_executor(self) -> ThreadPoolExecutor:
        with self._prefetch_lock:
            if self._prefetch_executor is None:
                self._prefetch_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="prefetch_worker")
            return self._prefetch_executor

    def clear_broadcast_cache(self) -> None:
        """Flushes cached Stage 2 broadcast states when calibration changes."""
        with self._cache_lock:
            self.broadcast_state_cache.clear()
            self._prefetch_pending.clear()

    def clear_all_caches(self) -> None:
        """Flushes all caches (detection, broadcast, and raw frame ring buffer) thread-safely."""
        with self._cache_lock:
            self.detection_cache.clear()
            self.broadcast_state_cache.clear()
            self._prefetch_pending.clear()
        with self._raw_cache_lock:
            self.raw_frame_cache.clear()
            self._raw_prefetch_pending.clear()
            self._raw_cache_hits = 0
            self._raw_cache_misses = 0

    @property
    def transformer(self) -> HomographyTransformer:
        """
        Dynamically resolves to the canonical st.session_state.transformer in Streamlit context,
        or falls back to self._default_transformer in headless/test context.
        """
        try:
            import streamlit as st
            if hasattr(st, "session_state") and "transformer" in st.session_state and st.session_state.transformer is not None:
                if st.session_state.get("stream_manager") is self or self._default_transformer is None:
                    return st.session_state.transformer
        except Exception:
            pass
        return self._default_transformer

    @transformer.setter
    def transformer(self, value: HomographyTransformer) -> None:
        self._default_transformer = value
        try:
            import streamlit as st
            if hasattr(st, "session_state") and (st.session_state.get("stream_manager") is self or "transformer" not in st.session_state):
                st.session_state.transformer = value
        except Exception:
            pass

    def set_camera_mode(self, camera_mode: str) -> None:
        """Sets the camera perspective mode on the transformer and flushes cached positions."""
        active_t = self.transformer
        active_t.set_camera_mode(camera_mode)
        if self._default_transformer is not None and self._default_transformer is not active_t:
            self._default_transformer.set_camera_mode(camera_mode)
        with self._cache_lock:
            self.broadcast_state_cache.clear()
            self._prefetch_pending.clear()

    def release(self) -> None:
        """Cleanly releases all OpenCV video capture handles and shuts down prefetch worker threads."""
        with self._cap_lock:
            if self.current_cap is not None:
                self.current_cap.release()
                self.current_cap = None
        with self._prefetch_lock:
            if self._prefetch_cap is not None:
                self._prefetch_cap.release()
                self._prefetch_cap = None
            if self._prefetch_executor is not None:
                try:
                    self._prefetch_executor.shutdown(wait=False, cancel_futures=True)
                except TypeError:
                    self._prefetch_executor.shutdown(wait=False)
                self._prefetch_executor = None

    def close(self) -> None:
        self.release()

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            pass

    def get_cached_raw_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """
        FW-13: Returns cached decoded BGR frame from in-memory ring buffer if present.
        Achieves sub-0.05 ms retrieval latency without OpenCV disk seek overhead.
        """
        with self._raw_cache_lock:
            if frame_idx in self.raw_frame_cache:
                self.raw_frame_cache.move_to_end(frame_idx)
                self._raw_cache_hits += 1
                return self.raw_frame_cache[frame_idx]

        # Check secondary lookahead detection cache
        with self._cache_lock:
            entry = self.detection_cache.get(frame_idx)
            if entry and 'raw_frame' in entry and entry['raw_frame'] is not None:
                raw_f = entry['raw_frame']
                with self._raw_cache_lock:
                    if len(self.raw_frame_cache) >= self._raw_cache_capacity:
                        self.raw_frame_cache.popitem(last=False)
                    self.raw_frame_cache[frame_idx] = raw_f
                    self._raw_cache_hits += 1
                return raw_f

        with self._raw_cache_lock:
            self._raw_cache_misses += 1
        return None

    def put_cached_raw_frame(self, frame_idx: int, raw_frame: np.ndarray) -> None:
        """Stores a decoded BGR frame into the in-memory ring buffer with LRU eviction."""
        if raw_frame is None or not isinstance(raw_frame, np.ndarray) or raw_frame.size == 0:
            return
        with self._raw_cache_lock:
            if len(self.raw_frame_cache) >= self._raw_cache_capacity:
                self.raw_frame_cache.popitem(last=False)
            self.raw_frame_cache[frame_idx] = raw_frame

    def prefetch_raw_window(self, start_frame_idx: int, window_size: int = 15) -> None:
        """
        FW-13: Asynchronously decodes upcoming raw video frames into the in-memory ring buffer
        without running YOLO inference. Keeps the buffer continuously populated during playback.
        """
        executor = self._get_executor()

        target_frames = []
        with self._raw_cache_lock:
            for offset in range(1, window_size + 1):
                f_idx = start_frame_idx + offset
                if 0 <= f_idx < self.total_frames:
                    if f_idx not in self.raw_frame_cache and f_idx not in self._raw_prefetch_pending:
                        target_frames.append(f_idx)
                        self._raw_prefetch_pending.add(f_idx)

        if not target_frames:
            return

        gen = self._stream_generation

        def _do_prefetch_raw(indices: List[int], job_gen: int):
            if job_gen != self._stream_generation:
                return
            try:
                with self._prefetch_lock:
                    if job_gen != self._stream_generation:
                        return
                    if self._prefetch_cap is None or not self._prefetch_cap.isOpened():
                        if self.current_path and os.path.exists(self.current_path):
                            self._prefetch_cap = cv2.VideoCapture(self.current_path)
                    if self._prefetch_cap is None or not self._prefetch_cap.isOpened():
                        return

                    indices_sorted = sorted(indices)
                    self._prefetch_cap.set(cv2.CAP_PROP_POS_FRAMES, indices_sorted[0])
                    cur_pos = indices_sorted[0]

                    for f_idx in indices_sorted:
                        if job_gen != self._stream_generation:
                            break
                        if cur_pos != f_idx:
                            self._prefetch_cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                            cur_pos = f_idx
                        ret, frame = self._prefetch_cap.read()
                        cur_pos += 1
                        if ret and frame is not None and job_gen == self._stream_generation:
                            with self._raw_cache_lock:
                                if len(self.raw_frame_cache) >= self._raw_cache_capacity:
                                    self.raw_frame_cache.popitem(last=False)
                                self.raw_frame_cache[f_idx] = frame
            except Exception as e:
                import logging
                logging.getLogger(__name__).debug("Raw frame window prefetch failed: %s", e)
            finally:
                with self._raw_cache_lock:
                    for f_idx in indices:
                        self._raw_prefetch_pending.discard(f_idx)

        try:
            executor.submit(_do_prefetch_raw, target_frames, gen)
        except RuntimeError:
            pass

    def get_ring_buffer_telemetry(self) -> Dict[str, Any]:
        """Returns FW-13 raw frame ring buffer statistics."""
        with self._raw_cache_lock:
            total_reqs = self._raw_cache_hits + self._raw_cache_misses
            hit_ratio = (self._raw_cache_hits / total_reqs * 100.0) if total_reqs > 0 else 0.0
            total_bytes = sum(f.nbytes for f in self.raw_frame_cache.values())
            mem_mb = total_bytes / (1024.0 * 1024.0)
            return {
                'count': len(self.raw_frame_cache),
                'capacity': self._raw_cache_capacity,
                'hits': self._raw_cache_hits,
                'misses': self._raw_cache_misses,
                'hit_ratio_pct': round(hit_ratio, 1),
                'memory_mb': round(mem_mb, 2),
                'pending_prefetch': len(self._raw_prefetch_pending)
            }

    def get_cached_detections(self, frame_idx: int) -> Optional[Any]:
        """Retrieves cached Stage 1 detections from LRU memory with LRU update."""
        with self._cache_lock:
            if frame_idx in self.detection_cache:
                self.detection_cache.move_to_end(frame_idx)
                entry = self.detection_cache[frame_idx]
                self._prefetch_hits += 1
                return entry.get('detections')
        return None

    def put_cached_detections(self, frame_idx: int, detections: Any, raw_frame: Optional[np.ndarray] = None) -> None:
        """Stores Stage 1 detections in LRU cache and routes raw frame to raw ring buffer."""
        if raw_frame is not None:
            self.put_cached_raw_frame(frame_idx, raw_frame)
        with self._cache_lock:
            if len(self.detection_cache) >= 1000:
                self.detection_cache.popitem(last=False)
            self.detection_cache[frame_idx] = {
                'detections': detections,
                'raw_frame': None
            }

    def prefetch_lookahead(self, current_frame_idx: int, lookahead: int = 1) -> None:
        """
        FW-05: Non-blocking asynchronous background lookahead prefetching.
        Pre-decodes and pre-processes frame (current_frame_idx + lookahead) into memory
        so sequential scrubbing achieves ~0 ms perceived latency.
        """
        target_idx = current_frame_idx + lookahead
        if target_idx < 0 or target_idx >= self.total_frames:
            return

        with self._cache_lock:
            if target_idx in self.detection_cache or target_idx in self._prefetch_pending:
                return
            self._prefetch_pending.add(target_idx)

        executor = self._get_executor()
        gen = self._stream_generation
        active_trans = self.transformer

        def _do_prefetch(f_idx: int, job_gen: int, trans_model: Any = active_trans):
            if job_gen != self._stream_generation:
                return
            try:
                # 1. Read frame via dedicated secondary capture handle
                with self._prefetch_lock:
                    if job_gen != self._stream_generation:
                        return
                    if self._prefetch_cap is None or not self._prefetch_cap.isOpened():
                        if self.current_path and os.path.exists(self.current_path):
                            self._prefetch_cap = cv2.VideoCapture(self.current_path)
                    if self._prefetch_cap is None or not self._prefetch_cap.isOpened():
                        self._last_prefetch_error = f"Frame {f_idx}: Prefetch capture handle not opened"
                        return
                    self._prefetch_cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                    ret, frame = self._prefetch_cap.read()
                    if not ret or frame is None:
                        self._last_prefetch_error = f"Frame {f_idx}: Read returned empty frame (ret={ret})"
                        return

                if job_gen != self._stream_generation:
                    return

                # 2. Execute Stage 1 Perception asynchronously
                from core.pipeline import stage1_perception
                detections = stage1_perception(frame, self.detector, frame_idx=f_idx, transformer=trans_model)

                if job_gen != self._stream_generation:
                    return

                # 3. Store into shared thread-safe detection cache & FW-13 raw ring buffer
                with self._cache_lock:
                    if job_gen != self._stream_generation:
                        return
                    if len(self.detection_cache) >= 1000:
                        self.detection_cache.popitem(last=False)
                    self.detection_cache[f_idx] = {
                        'detections': detections,
                        'raw_frame': None
                    }
                with self._raw_cache_lock:
                    if job_gen != self._stream_generation:
                        return
                    if len(self.raw_frame_cache) >= self._raw_cache_capacity:
                        self.raw_frame_cache.popitem(last=False)
                    self.raw_frame_cache[f_idx] = frame
                self._last_prefetch_error = None
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning("Async prefetch failed for frame %d: %s", f_idx, e, exc_info=True)
                self._last_prefetch_error = f"Frame {f_idx}: {e}"
            finally:
                with self._cache_lock:
                    self._prefetch_pending.discard(f_idx)

        try:
            executor.submit(_do_prefetch, target_idx, gen)
        except RuntimeError:
            pass

    def get_prefetch_telemetry(self) -> Dict[str, Any]:
        """Returns prefetch telemetry metrics."""
        with self._cache_lock:
            return {
                'hits': self._prefetch_hits,
                'cached_count': len(self.detection_cache),
                'pending_count': len(self._prefetch_pending),
                'last_error': getattr(self, '_last_prefetch_error', None)
            }

    def get_broadcast_frame_state(self, frame_idx: int) -> Dict[str, Any]:
        """
        Extracts video frame at frame_idx and runs standardized core.pipeline Stage 1 and Stage 2.
        """
        if self.current_cap is None:
            blank = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            return self._build_fallback_state(frame_idx, blank)
            
        # Check broadcast state LRU cache first under thread lock
        with self._cache_lock:
            if frame_idx in self.broadcast_state_cache:
                self.broadcast_state_cache.move_to_end(frame_idx)
                return self.broadcast_state_cache[frame_idx]

        # Check if Stage 1 detections and raw frame were prefetched
        pre_det = None
        pre_frame = self.get_cached_raw_frame(frame_idx)
        with self._cache_lock:
            if frame_idx in self.detection_cache:
                entry = self.detection_cache[frame_idx]
                pre_det = entry.get('detections')
                if pre_frame is None:
                    pre_frame = entry.get('raw_frame')

        if pre_det is not None and pre_frame is not None:
            frame = pre_frame
            detections = pre_det
        else:
            # Deterministic seek with thread lock to avoid race conditions in FFmpeg
            with self._cap_lock:
                cur_pos = int(self.current_cap.get(cv2.CAP_PROP_POS_FRAMES))
                if cur_pos != frame_idx:
                    self.current_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = self.current_cap.read()
                
                if not ret or frame is None:
                    self._last_read_pos = -1
                    blank = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                    return self._build_fallback_state(frame_idx, blank)
                self._last_read_pos = frame_idx + 1

            self.put_cached_raw_frame(frame_idx, frame)
            from core.pipeline import stage1_perception
            detections = stage1_perception(frame, self.detector, frame_idx=frame_idx, transformer=self.transformer)
            with self._cache_lock:
                if len(self.detection_cache) >= 1000:
                    self.detection_cache.popitem(last=False)
                self.detection_cache[frame_idx] = {
                    'detections': detections,
                    'raw_frame': None
                }

        from core.pipeline import stage2_projection
        positions = stage2_projection(detections, self.transformer)

        state = {
            'frame_idx': frame_idx,
            'raw_frame': frame,
            'team_a': positions.team_a,
            'team_b': positions.team_b,
            'ball': positions.ball,
            'ball_carrier_idx': positions.carrier_idx,
            'detected_players': positions.detected_count,
            'detected_ball': detections.ball_pixel is not None,
            'source_type': 'broadcast_stream',
            'video_width': self.width,
            'video_height': self.height
        }
        
        with self._cache_lock:
            if len(self.broadcast_state_cache) >= self._raw_cache_capacity:
                self.broadcast_state_cache.popitem(last=False)
            self.broadcast_state_cache[frame_idx] = state
        return state

    def _build_fallback_state(self, frame_idx: int, frame: np.ndarray) -> Dict[str, Any]:
        """Builds default tactical state if frame reading fails without injecting phantom players."""
        return {
            'frame_idx': frame_idx,
            'raw_frame': frame,
            'team_a': [],
            'team_b': [],
            'ball': (52.5, 34.0),
            'ball_carrier_idx': 0,
            'detected_players': 0,
            'detected_ball': False,
            'source_type': 'fallback',
            'video_width': self.width,
            'video_height': self.height
        }
