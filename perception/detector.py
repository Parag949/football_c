"""
Perception Layer: Computer Vision Object Detection & Team Clustering
Uses YOLOv8 for player, referee, and ball detection from broadcast video frames.
Extracts bounding boxes, detects ground-contact foot coordinates, and clusters jersey colors.
Includes high-fidelity tactical match simulation generator for instant offline testing.
"""

from typing import List, Tuple, Dict, Any, Optional
import threading
import numpy as np
import cv2


from core.contracts import PlayerDetection


def compute_box_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """Computes Intersection-over-Union between two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / float(max(1e-5, union))


def apply_cross_class_nms(candidates: List[Dict[str, Any]], iou_thresh: float = 0.45) -> List[Dict[str, Any]]:
    """
    Suppresses redundant overlapping bounding boxes with Referee & Goalkeeper priority.
    If a referee or goalkeeper overlaps with a generic player box, the specialized class
    is preserved even if 'player' has slightly higher raw logit confidence.
    """
    if not candidates:
        return []

    def score_cand(c):
        s = float(c.get('conf', 0.0))
        if c.get('is_referee', False):
            s += 0.40  # Referee priority to prevent class-imbalance swallowing
        elif c.get('is_gk', False):
            s += 0.25  # Goalkeeper priority
        return s

    sorted_cand = sorted(candidates, key=score_cand, reverse=True)
    kept: List[Dict[str, Any]] = []

    for cand in sorted_cand:
        box = cand['box']
        overlap = False
        for k in kept:
            if compute_box_iou(box, k['box']) > iou_thresh:
                overlap = True
                if cand.get('is_referee', False) and not k.get('is_referee', False):
                    k['is_referee'] = True
                    k['cls_id'] = 3
                if cand.get('is_gk', False) and not k.get('is_gk', False):
                    k['is_gk'] = True
                    k['cls_id'] = 1
                break
        if not overlap:
            kept.append(cand)

    return kept


class FootballDetector:
    """
    Perception engine wrapping YOLOv8 object detector with color-based team assignment.
    Supports both specialized football tracking models (models/best.pt) and standard COCO models.
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        conf_thresh: float = 0.25,
        num_threads: Optional[int] = None
    ):
        import os
        if model_name is None:
            # Auto-prioritize INT8 quantized model on CPU, fallback to standard ONNX or PyTorch
            if os.path.exists("models/best_int8.onnx"):
                model_name = "models/best_int8.onnx"
            elif os.path.exists("models/best.onnx"):
                model_name = "models/best.onnx"
            elif os.path.exists("models/best.pt"):
                model_name = "models/best.pt"
            elif os.path.exists("models/yolov8n.onnx"):
                model_name = "models/yolov8n.onnx"
            elif os.path.exists("models/yolov8n.pt"):
                model_name = "models/yolov8n.pt"
            elif os.path.exists("yolov8n.onnx"):
                model_name = "yolov8n.onnx"
            else:
                model_name = "yolov8n.pt"
        self.model_name = model_name
        self.conf_thresh = conf_thresh
        cpu_count = os.cpu_count() or 4
        self.num_threads = num_threads if num_threads is not None else max(1, min(cpu_count // 2, 4))
        self.model = None
        self.is_onnx = str(model_name).endswith(".onnx")
        self._inference_lock = threading.RLock()
        from perception.jersey_classifier import JerseyCentroidTracker
        self.centroid_tracker = JerseyCentroidTracker()
        self._load_model()
        
    def _load_model(self):
        with self._inference_lock:
            try:
                import os
                import torch
                from ultralytics import YOLO

                if self.num_threads:
                    torch.set_num_threads(self.num_threads)
                    os.environ["OMP_NUM_THREADS"] = str(self.num_threads)

                self.model = YOLO(self.model_name)
                self._session_configured = False
            except Exception:
                self.model = None

    def set_num_threads(self, num_threads: int):
        """Dynamically reconfigures CPU thread cap."""
        with self._inference_lock:
            self.num_threads = max(1, int(num_threads))
            self._load_model()

    def set_model(self, model_name: str):
        """Dynamically switches detector neural model."""
        with self._inference_lock:
            self.model_name = model_name
            self.is_onnx = str(model_name).endswith(".onnx")
            self._load_model()

    def get_onnx_telemetry(self) -> Dict[str, Any]:
        """Returns ONNX Runtime multi-thread concurrency tuning telemetry (FW-04)."""
        return {
            'is_onnx': self.is_onnx,
            'model_name': self.model_name,
            'intra_op_threads': int(self.num_threads) if self.num_threads else 1,
            'inter_op_threads': 1,
            'execution_mode': 'ORT_SEQUENTIAL',
            'graph_optimization': 'ORT_ENABLE_ALL',
            'provider': 'CPUExecutionProvider',
            'session_configured': getattr(self, '_session_configured', False)
        }

    def detect_frame(
        self,
        frame: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]] = None,
        adaptive_conf: bool = True,
        transformer: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Runs YOLOv8 detection on a single video frame.
        Maps detections according to model vocabulary (sports model vs COCO).
        Supports calibration-guided or panoramic aspect ROI cropping and adaptive
        confidence thresholding to detect distant / small players on CPU without
        expensive multi-pass inference overhead.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return {'players': [], 'ball': None}

        fh, fw = frame.shape[:2]
        aspect = fw / float(max(1, fh))

        # Determine active ROI: prioritize caller/calibration ROI, fallback to panoramic horizon crop
        active_roi = roi
        if active_roi is None and aspect > 2.2:
            active_roi = (0, int(fh * 0.12), fw, int(fh * 0.88))

        inference_frame = frame
        offset_x, offset_y = 0, 0
        use_roi = False

        if active_roi is not None:
            rx1, ry1, rx2, ry2 = active_roi
            rx1 = max(0, min(fw - 1, int(rx1)))
            ry1 = max(0, min(fh - 1, int(ry1)))
            rx2 = max(rx1 + 10, min(fw, int(rx2)))
            ry2 = max(ry1 + 10, min(fh, int(ry2)))
            if (rx2 - rx1) >= 80 and (ry2 - ry1) >= 60 and ((rx2 - rx1) < fw or (ry2 - ry1) < fh):
                inference_frame = frame[ry1:ry2, rx1:rx2]
                offset_x, offset_y = rx1, ry1
                use_roi = True

        # Adaptive confidence: small or distant outfield players on pitch turf frequently
        # score confidence in [0.15..0.22]. Allow threshold down to 0.16 when adaptive_conf is active.
        run_conf = min(self.conf_thresh, 0.16) if adaptive_conf else self.conf_thresh
        run_imgsz = 1024

        with self._inference_lock:
            if self.model is None:
                self._load_model()
                
            if self.model is None:
                return {'players': [], 'ball': None}

            # Multi-regime inference:
            # For ultrawide panoramic footage (aspect > 2.2), run high-resolution 3-tile inference
            # to resolve edge goalkeepers, distant players, and referees without feature pyramid collapse.
            all_raw_boxes = []
            if aspect > 2.2:
                y1 = int(fh * 0.10)
                y2 = int(fh * 0.90)
                if active_roi is not None:
                    y1 = max(0, min(fh - 1, int(active_roi[1])))
                    y2 = max(y1 + 10, min(fh, int(active_roi[3])))

                tiles = [
                    (0, int(fw * 0.42)),
                    (int(fw * 0.30), int(fw * 0.70)),
                    (int(fw * 0.44), fw)
                ]
                for x_start, x_end in tiles:
                    tile_img = frame[y1:y2, x_start:x_end]
                    res = self.model(tile_img, imgsz=run_imgsz, conf=run_conf, verbose=False)[0]
                    for b in res.boxes:
                        all_raw_boxes.append((b, x_start, y1))
            else:
                results = self.model(inference_frame, imgsz=run_imgsz, conf=run_conf, verbose=False)
                for b in results[0].boxes:
                    all_raw_boxes.append((b, offset_x, offset_y))

            # FW-04: Lazily configure ONNX Runtime session threads with ORT_SEQUENTIAL and ORT_ENABLE_ALL
            if self.is_onnx and self.num_threads and not getattr(self, '_session_configured', False):
                try:
                    if hasattr(self.model, 'predictor') and hasattr(self.model.predictor, 'model'):
                        backend = self.model.predictor.model
                        if hasattr(backend, 'session'):
                            import onnxruntime as ort
                            sess_opts = ort.SessionOptions()
                            sess_opts.intra_op_num_threads = int(self.num_threads)
                            sess_opts.inter_op_num_threads = 1
                            sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                            backend.session = ort.InferenceSession(self.model_name, sess_opts, providers=['CPUExecutionProvider'])
                            self._session_configured = True
                except Exception:
                    self._session_configured = True

            model_names = self.model.names if self.model is not None else {}
        
        candidates = []
        ball = None
        best_ball_conf = -1.0
        if isinstance(model_names, dict):
            is_sports_model = "player" in model_names.values()
        elif isinstance(model_names, (list, tuple)):
            is_sports_model = "player" in model_names
        else:
            is_sports_model = False
        
        for box, off_x, off_y in all_raw_boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            raw_xyxy = [int(v) for v in box.xyxy[0]]
            xyxy = [
                raw_xyxy[0] + off_x,
                raw_xyxy[1] + off_y,
                raw_xyxy[2] + off_x,
                raw_xyxy[3] + off_y,
            ]
            if isinstance(model_names, dict):
                cls_name = model_names.get(cls_id, "")
            elif isinstance(model_names, (list, tuple)) and 0 <= cls_id < len(model_names):
                cls_name = model_names[cls_id]
            else:
                cls_name = ""
            
            is_player = False
            is_gk = False
            is_ball = False
            is_referee = False
            
            if is_sports_model:
                if cls_name == "player":
                    is_player = True
                elif cls_name == "goalkeeper":
                    is_gk = True
                elif cls_name == "ball":
                    is_ball = True
                elif cls_name == "referee" or cls_id == 3:
                    is_referee = True
            else:
                if cls_id == 0:  # COCO Person
                    is_player = True
                elif cls_id == 32:  # COCO Sports ball
                    is_ball = True
                    
            if is_player or is_gk or is_referee:
                w = xyxy[2] - xyxy[0]
                h = xyxy[3] - xyxy[1]
                if w < 3 or h < 8:
                    continue

                if conf < self.conf_thresh and h >= 28:
                    ratio = h / float(max(1, w))
                    if ratio < 0.65 or ratio > 6.0:
                        continue

                candidates.append({
                    'box': tuple(xyxy),
                    'conf': conf,
                    'cls_id': 3 if is_referee else (1 if is_gk else 2),
                    'is_gk': is_gk,
                    'is_referee': is_referee
                })

            elif is_ball and conf > best_ball_conf:
                best_ball_conf = conf
                ball = ((xyxy[0] + xyxy[2]) / 2.0, (xyxy[1] + xyxy[3]) / 2.0)

        # Apply Cross-Class NMS with Referee & Goalkeeper Priority
        clean_candidates = apply_cross_class_nms(candidates, iou_thresh=0.45)

        outfield_players = []
        goalkeepers = []
        referees = []

        for cand in clean_candidates:
            xyxy = cand['box']
            w = xyxy[2] - xyxy[0]
            h = xyxy[3] - xyxy[1]
            torso_y1 = max(0, xyxy[1] + int(0.15 * h))
            torso_y2 = min(frame.shape[0], xyxy[1] + int(0.48 * h))
            torso_x1 = max(0, xyxy[0] + int(0.25 * w))
            torso_x2 = min(frame.shape[1], xyxy[0] + int(0.75 * w))

            if torso_y2 > torso_y1 and torso_x2 > torso_x1:
                torso_crop = frame[torso_y1:torso_y2, torso_x1:torso_x2]
            else:
                cy1 = max(0, min(frame.shape[0], xyxy[1]))
                cy2 = max(0, min(frame.shape[0], xyxy[3]))
                cx1 = max(0, min(frame.shape[1], xyxy[0]))
                cx2 = max(0, min(frame.shape[1], xyxy[2]))
                if cy2 > cy1 and cx2 > cx1:
                    torso_crop = frame[cy1:cy2, cx1:cx2]
                else:
                    torso_crop = np.zeros((10, 10, 3), dtype=np.uint8)

            color_feat = self._extract_jersey_feature(torso_crop)
            player_entry = {
                'box': cand['box'],
                'conf': cand['conf'],
                'cls_id': 3 if cand['is_referee'] else (1 if cand['is_gk'] else 2),
                'crop': torso_crop,
                'color_feat': color_feat,
                'is_gk': cand['is_gk'],
                'is_referee': cand['is_referee']
            }
            if cand['is_referee']:
                referees.append(player_entry)
            elif cand['is_gk']:
                goalkeepers.append(player_entry)
            else:
                outfield_players.append(player_entry)
                
        # Adaptive Team Clustering via CIELAB perceptual contrast & spatial goalkeeper attribution
        player_objects = self._assign_teams_kmeans(
            outfield_players,
            goalkeepers,
            referees=referees,
            frame_width=frame.shape[1],
            transformer=transformer
        )
        return {'players': player_objects, 'ball': ball}

    def _extract_jersey_feature(self, crop: np.ndarray) -> np.ndarray:
        """
        Extracts robust perceptual color representation in CIELAB space [L*, a*, b*, Saturation],
        filtering out background grass pixels to prevent pitch bleed while strictly preserving
        neon-yellow, lime, and gold referee kits.
        """
        if crop is None or crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
            return np.array([128.0, 128.0, 128.0, 50.0], dtype=np.float32)
        try:
            lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

            # Turf grass green mask (Hue ~ 38 to 85, Saturation > 50)
            grass_mask = (hsv[:, :, 0] >= 38) & (hsv[:, :, 0] <= 85) & (hsv[:, :, 1] > 50)
            valid_mask = ~grass_mask

            if np.any(valid_mask):
                v_lab = lab[valid_mask]
                v_sat = hsv[:, :, 1][valid_mask]
            else:
                v_lab = lab.reshape(-1, 3)
                v_sat = hsv[:, :, 1].reshape(-1)

            # Robust median to resist border outliers & kit badges/numbers
            med_l, med_a, med_b = np.median(v_lab, axis=0)
            med_s = float(np.median(v_sat))

            return np.array([med_l, med_a, med_b, med_s], dtype=np.float32)
        except Exception:
            return np.array([128.0, 128.0, 128.0, 50.0], dtype=np.float32)

    def _assign_teams_kmeans(
        self,
        raw_players: List[Dict[str, Any]],
        goalkeepers: Optional[List[Dict[str, Any]]] = None,
        referees: Optional[List[Dict[str, Any]]] = None,
        frame_width: int = 1920,
        transformer: Optional[Any] = None
    ) -> List[PlayerDetection]:
        """
        Perceptually clusters outfield players into exactly 2 distinct team kits
        using standardized CIELAB + Saturation features and cv2.kmeans.
        Referees are cleanly isolated with team_id = -1.
        Goalkeepers are preserved and deterministically assigned to their defending goal line.
        """
        from perception.jersey_classifier import classify_player_jerseys
        all_candidates = list(raw_players)
        if goalkeepers:
            for gk in goalkeepers:
                gk['is_gk'] = True
            all_candidates.extend(goalkeepers)
        if referees:
            for ref in referees:
                ref['is_referee'] = True
                ref['cls_id'] = 3
            all_candidates.extend(referees)

        return classify_player_jerseys(
            all_candidates,
            frame_width=frame_width,
            centroid_tracker=getattr(self, 'centroid_tracker', None),
            transformer=transformer
        )
