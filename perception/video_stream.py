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
        self.transformer = HomographyTransformer(frame_width=self.width, frame_height=self.height)
        
        # LRU Cache for detected states to avoid re-running YOLO on already analyzed frames
        self.detection_cache: OrderedDict[int, Dict[str, Any]] = OrderedDict()
        self._cache_lock = threading.Lock()
        
        # FW-05: Dedicated asynchronous lookahead prefetch pipeline
        self._prefetch_lock = threading.Lock()
        self._prefetch_cap: Optional[cv2.VideoCapture] = None
        self._prefetch_executor: Optional[ThreadPoolExecutor] = None
        self._prefetch_hits: int = 0
        self._prefetch_pending: set = set()
        
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
            self.total_frames = max(10, int(self.current_cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            self.fps = max(10.0, float(self.current_cap.get(cv2.CAP_PROP_FPS) or 25.0))
            self.width = int(self.current_cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
            self.height = int(self.current_cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
            self.transformer = HomographyTransformer(frame_width=self.width, frame_height=self.height)
            
            with self._cache_lock:
                self.detection_cache.clear()
                self._prefetch_pending.clear()

            with self._prefetch_lock:
                if self._prefetch_cap is not None:
                    self._prefetch_cap.release()
                self._prefetch_cap = cv2.VideoCapture(video_path)

            if self._prefetch_executor is None:
                self._prefetch_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="prefetch_worker")
            return True

    def set_camera_mode(self, camera_mode: str) -> None:
        """Sets the camera perspective mode on the transformer and flushes cached positions."""
        self.transformer.set_camera_mode(camera_mode)
        with self._cache_lock:
            self.detection_cache.clear()
            self._prefetch_pending.clear()

    def get_cached_raw_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """Returns prefetched raw BGR frame from memory if available, bypassing OpenCV decode."""
        with self._cache_lock:
            entry = self.detection_cache.get(frame_idx)
            if entry and 'raw_frame' in entry and entry['raw_frame'] is not None:
                return entry['raw_frame']
        return None

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
        """Stores Stage 1 detections and optional raw frame in LRU cache."""
        with self._cache_lock:
            if len(self.detection_cache) >= 1000:
                self.detection_cache.popitem(last=False)
            self.detection_cache[frame_idx] = {
                'detections': detections,
                'raw_frame': raw_frame
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

        if self._prefetch_executor is None:
            self._prefetch_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="prefetch_worker")

        def _do_prefetch(f_idx: int):
            try:
                # 1. Read frame via dedicated secondary capture handle
                with self._prefetch_lock:
                    if self._prefetch_cap is None or not self._prefetch_cap.isOpened():
                        if self.current_path and os.path.exists(self.current_path):
                            self._prefetch_cap = cv2.VideoCapture(self.current_path)
                    if self._prefetch_cap is None or not self._prefetch_cap.isOpened():
                        return
                    self._prefetch_cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                    ret, frame = self._prefetch_cap.read()
                    if not ret or frame is None:
                        return

                # 2. Execute Stage 1 Perception asynchronously
                from core.pipeline import stage1_perception
                detections = stage1_perception(frame, self.detector, frame_idx=f_idx)

                # 3. Store into shared thread-safe detection cache
                with self._cache_lock:
                    if len(self.detection_cache) >= 1000:
                        self.detection_cache.popitem(last=False)
                    self.detection_cache[f_idx] = {
                        'detections': detections,
                        'raw_frame': frame
                    }
            except Exception:
                pass
            finally:
                with self._cache_lock:
                    self._prefetch_pending.discard(f_idx)

        self._prefetch_executor.submit(_do_prefetch, target_idx)

    def get_prefetch_telemetry(self) -> Dict[str, Any]:
        """Returns prefetch telemetry metrics."""
        with self._cache_lock:
            return {
                'hits': self._prefetch_hits,
                'cached_count': len(self.detection_cache),
                'pending_count': len(self._prefetch_pending)
            }

    def get_broadcast_frame_state(self, frame_idx: int) -> Dict[str, Any]:
        """
        Extracts video frame at frame_idx and runs standardized core.pipeline Stage 1 and Stage 2.
        """
        if self.current_cap is None:
            blank = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            return self._build_fallback_state(frame_idx, blank)
            
        # Check LRU cache first for instant responsiveness
        if frame_idx in self.detection_cache:
            self.detection_cache.move_to_end(frame_idx)
            return self.detection_cache[frame_idx]
            
        # Deterministic seek with thread lock to avoid race conditions in FFmpeg
        with self._cap_lock:
            cur_pos = int(self.current_cap.get(cv2.CAP_PROP_POS_FRAMES))
            if cur_pos != frame_idx:
                self.current_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.current_cap.read()
            
            if not ret or frame is None:
                self.current_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.current_cap.read()
                if not ret or frame is None:
                    blank = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                    return self._build_fallback_state(frame_idx, blank)

        # Standardized Stage 1 & Stage 2 Execution
        from core.pipeline import stage1_perception, stage2_projection
        detections = stage1_perception(frame, self.detector, frame_idx=frame_idx)
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
        
        # Evict oldest entry (LRU) instead of clearing all 1000 entries
        if len(self.detection_cache) >= 1000:
            self.detection_cache.popitem(last=False)
        self.detection_cache[frame_idx] = state
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
