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

class MatchStreamManager:
    """
    Manages live match video feeds, frame extraction, and player detection.
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
            self.detection_cache.clear()
            return True

    def set_camera_mode(self, camera_mode: str) -> None:
        """Sets the camera perspective mode on the transformer and flushes cached positions."""
        self.transformer.set_camera_mode(camera_mode)
        self.detection_cache.clear()

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
