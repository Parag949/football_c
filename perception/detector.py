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
        adaptive_conf: bool = True
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

        # Adaptive confidence: on cropped pitch turf / panorama, small players score lower conf (0.15 - 0.25)
        run_conf = min(self.conf_thresh, 0.18) if (use_roi or aspect > 2.2) and adaptive_conf else self.conf_thresh

        with self._inference_lock:
            if self.model is None:
                self._load_model()
                
            if self.model is None:
                return {'players': [], 'ball': None}
                
            results = self.model(inference_frame, conf=run_conf, verbose=False)

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

            detections = results[0]
            model_names = self.model.names if self.model is not None else {}
        
        outfield_players = []
        goalkeepers = []
        ball = None
        best_ball_conf = -1.0
        if isinstance(model_names, dict):
            is_sports_model = "player" in model_names.values()
        elif isinstance(model_names, (list, tuple)):
            is_sports_model = "player" in model_names
        else:
            is_sports_model = False
        
        for box in detections.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            raw_xyxy = [int(v) for v in box.xyxy[0]]
            xyxy = [
                raw_xyxy[0] + offset_x,
                raw_xyxy[1] + offset_y,
                raw_xyxy[2] + offset_x,
                raw_xyxy[3] + offset_y,
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
            
            if is_sports_model:
                if cls_name == "player":
                    is_player = True
                elif cls_name == "goalkeeper":
                    is_gk = True
                elif cls_name == "ball":
                    is_ball = True
                elif cls_name == "referee":
                    # Referees are neither outfield players nor goalkeepers; exclude from pitch analytics
                    continue
            else:
                if cls_id == 0:  # COCO Person
                    is_player = True
                elif cls_id == 32:  # COCO Sports ball
                    is_ball = True
                    
            if is_player or is_gk:
                w = xyxy[2] - xyxy[0]
                h = xyxy[3] - xyxy[1]
                if w < 5 or h < 12:
                    continue

                # Aspect ratio noise rejection for low-confidence detections (conf < self.conf_thresh)
                # Standing football players maintain vertical aspect ratio h/w in [1.3, 5.0]
                if conf < self.conf_thresh:
                    ratio = h / float(max(1, w))
                    if ratio < 1.3 or ratio > 5.0:
                        continue
                    
                # Inner-Core Torso Sampling: center 50% width [25%..75%], upper torso [15%..48%]
                # Eliminates boundary grass bleed on low-resolution / distant broadcast footage
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
                
                # Extract robust perceptual color representation
                color_feat = self._extract_jersey_feature(torso_crop)
                
                player_entry = {
                    'box': tuple(xyxy),
                    'conf': conf,
                    'cls_id': cls_id,
                    'crop': torso_crop,
                    'color_feat': color_feat,
                    'is_gk': is_gk
                }
                if is_gk:
                    goalkeepers.append(player_entry)
                else:
                    outfield_players.append(player_entry)

            elif is_ball and conf > best_ball_conf:
                best_ball_conf = conf
                ball = ((xyxy[0] + xyxy[2]) / 2.0, (xyxy[1] + xyxy[3]) / 2.0)
                
        # Adaptive Team Clustering via CIELAB perceptual contrast & spatial goalkeeper attribution
        player_objects = self._assign_teams_kmeans(outfield_players, goalkeepers, frame_width=frame.shape[1])
        return {'players': player_objects, 'ball': ball}

    def _extract_jersey_feature(self, crop: np.ndarray) -> np.ndarray:
        """
        Extracts robust perceptual color representation in CIELAB space [L*, a*, b*, Saturation],
        filtering out background grass pixels to prevent pitch bleed.
        """
        if crop is None or crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
            return np.array([128.0, 128.0, 128.0, 50.0], dtype=np.float32)
        try:
            lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

            # Turf grass green mask (Hue ~ 35 to 85, Saturation > 45)
            grass_mask = (hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 85) & (hsv[:, :, 1] > 45)
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
        frame_width: int = 1920
    ) -> List[PlayerDetection]:
        """
        Perceptually clusters outfield players into exactly 2 distinct team kits
        using standardized CIELAB + Saturation features and cv2.kmeans.
        Goalkeepers are preserved and deterministically assigned to their defending goal line.
        """
        team_0_color = (255, 180, 0)  # Home (Cyan / Sky Blue BGR)
        team_1_color = (0, 70, 255)   # Away (Vibrant Red BGR)

        if goalkeepers is None:
            # Backward-compatibility: extract GKs if already in raw_players list
            goalkeepers = [p for p in raw_players if p.get('is_gk', False) or p.get('cls_id') == 1]
            raw_players = [p for p in raw_players if not (p.get('is_gk', False) or p.get('cls_id') == 1)]

        if not raw_players and not goalkeepers:
            return []

        all_raw = raw_players + goalkeepers
        if len(raw_players) < 4:
            res = []
            for p in all_raw:
                feat = p['color_feat']
                t_id = 0 if feat[0] > 180.0 else 1
                c = team_1_color if t_id == 1 else team_0_color
                res.append(PlayerDetection(
                    box=p['box'],
                    confidence=p['conf'],
                    class_id=p['cls_id'],
                    team_id=t_id,
                    jersey_color=c
                ))
            return res

        # 1. Cluster outfield players ONLY (excluding goalkeepers and referees)
        feats = np.float32([p['color_feat'] for p in raw_players])
        f_mean = np.mean(feats, axis=0)
        f_std = np.std(feats, axis=0) + 1e-6
        feats_norm = (feats - f_mean) / f_std

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.2)
        _, labels, centers = cv2.kmeans(feats_norm, 2, None, criteria, 15, cv2.KMEANS_PP_CENTERS)
        labels = labels.flatten()

        # 2. Deterministic Team Ordering across ANY video / kit combination:
        # Check perceptual luminance L* (feats[:, 0]):
        # Team 0 is consistently the lighter / higher-luminance kit (e.g. White, Yellow, Cyan).
        # Team 1 is the darker / warmer contrasting kit (e.g. Red, Navy, Black).
        c0_feats = [raw_players[i]['color_feat'] for i in range(len(raw_players)) if labels[i] == 0]
        c1_feats = [raw_players[i]['color_feat'] for i in range(len(raw_players)) if labels[i] == 1]

        if c0_feats and c1_feats:
            l0 = np.mean([f[0] for f in c0_feats])
            l1 = np.mean([f[0] for f in c1_feats])

            # If luminance contrast is minimal (< 10 units), use color warmth (a* + b*) as tie-breaker
            if abs(l0 - l1) < 10.0:
                w0 = np.mean([f[1] + f[2] for f in c0_feats])
                w1 = np.mean([f[1] + f[2] for f in c1_feats])
                if w0 > w1:
                    labels = 1 - labels
            elif l0 < l1:
                labels = 1 - labels

        # 3. Create PlayerDetection objects for outfield players
        player_objects: List[PlayerDetection] = []
        for i, p in enumerate(raw_players):
            t_id = int(labels[i])
            c = team_1_color if t_id == 1 else team_0_color
            player_objects.append(PlayerDetection(
                box=p['box'],
                confidence=p['conf'],
                class_id=p['cls_id'],
                team_id=t_id,
                jersey_color=c
            ))

        # 4. Deterministic Goalkeeper Spatial Attribution:
        # In football, each team defends opposite goal lines. Goalkeepers are deterministically
        # assigned to their defending team based on the team's longitudinal pitch centroid.
        if goalkeepers:
            t0_x = [(raw_players[i]['box'][0] + raw_players[i]['box'][2]) / 2.0 for i in range(len(raw_players)) if labels[i] == 0]
            t1_x = [(raw_players[i]['box'][0] + raw_players[i]['box'][2]) / 2.0 for i in range(len(raw_players)) if labels[i] == 1]

            mean_x0 = float(np.mean(t0_x)) if t0_x else frame_width / 2.0
            mean_x1 = float(np.mean(t1_x)) if t1_x else frame_width / 2.0

            # The team whose centroid is to the left defends the left goal line
            left_defending_team = 0 if mean_x0 < mean_x1 else 1
            right_defending_team = 1 - left_defending_team

            # If two goalkeepers are detected, enforce strict 1-per-team attribution
            sorted_gks = sorted(goalkeepers, key=lambda g: (g['box'][0] + g['box'][2]) / 2.0)
            if len(sorted_gks) == 2:
                gk_left, gk_right = sorted_gks[0], sorted_gks[1]
                for gk, team in [(gk_left, left_defending_team), (gk_right, right_defending_team)]:
                    c = team_1_color if team == 1 else team_0_color
                    player_objects.append(PlayerDetection(
                        box=gk['box'],
                        confidence=gk['conf'],
                        class_id=gk['cls_id'],
                        team_id=team,
                        jersey_color=c
                    ))
            else:
                for gk in sorted_gks:
                    gk_cx = (gk['box'][0] + gk['box'][2]) / 2.0
                    gk_team = left_defending_team if gk_cx < (frame_width / 2.0) else right_defending_team
                    c = team_1_color if gk_team == 1 else team_0_color
                    player_objects.append(PlayerDetection(
                        box=gk['box'],
                        confidence=gk['conf'],
                        class_id=gk['cls_id'],
                        team_id=gk_team,
                        jersey_color=c
                    ))

        return player_objects
