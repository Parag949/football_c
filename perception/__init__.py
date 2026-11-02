"""
Perception Layer package:
- Pitch Model (FIFA 105m x 68m specifications)
- Homography Perspective Transformers
- Object Detector (YOLOv8 & Jersey Clustering)
- Match Stream Manager (Broadcast Video & Upload Ingest)
"""

from .pitch_model import PitchModel
from .homography import HomographyTransformer
from .detector import FootballDetector, PlayerDetection
from .video_stream import MatchStreamManager

__all__ = [
    'PitchModel',
    'HomographyTransformer',
    'FootballDetector',
    'PlayerDetection',
    'MatchStreamManager'
]
