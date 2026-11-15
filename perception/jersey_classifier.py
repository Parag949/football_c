"""
Automated Team Jersey Color Classification Engine
=================================================
Perceptual CIELAB + HSV Torso Clustering for Multi-Team Kit Attribution.

Features:
- Inner-Core Torso Sampling (isolates chest kit fabric, excludes hair, neck, shorts, and cleats)
- Adaptive Turf Grass Subtraction (filters green pitch reflection and camera motion blur)
- Weighted CIELAB (L*, a*, b*) + Saturation Color Space
- K-Means++ Chromatic Clustering with Outlier / Referee Rejection
- Dynamic Jersey Color Assignment for UI Overlays
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import cv2

from core.contracts import PlayerDetection

# Standard tactical overlay kit colors
TEAM_0_HEX = (255, 180, 0)   # Home: Vibrant Cyan / Sky Blue (BGR)
TEAM_1_HEX = (0, 70, 255)    # Away: Electric Crimson / Red (BGR)
REFEREE_HEX = (0, 255, 255)  # Referee: Bright Yellow (BGR)


class JerseyCentroidTracker:
    """
    Temporal Jersey Centroid Memory and Team Kit Stabilization Engine.
    
    Maintains Exponential Moving Average (EMA) color centroids across consecutive video frames
    to prevent frame-to-frame team identity flipping, kit color drift, and misclassification.
    """
    def __init__(self, momentum: float = 0.85):
        self.momentum = float(np.clip(momentum, 0.50, 0.98))
        self.team_0_center: Optional[np.ndarray] = None  # (4,) [L*, a*, b*, Sat]
        self.team_1_center: Optional[np.ndarray] = None  # (4,) [L*, a*, b*, Sat]
        self.weights = np.array([0.7, 1.8, 1.8, 0.9], dtype=np.float32)

    def reset(self) -> None:
        """Flushes tracked kit centroids (e.g. on video change or stream restart)."""
        self.team_0_center = None
        self.team_1_center = None

    def update_and_match(
        self,
        c0_unweighted: np.ndarray,
        c1_unweighted: np.ndarray,
        default_flip: bool = False
    ) -> Tuple[bool, np.ndarray, np.ndarray]:
        """
        Permutation-invariant temporal matching against stabilized reference centroids.
        
        Returns:
        --------
        (flip_teams, active_team_0_center, active_team_1_center)
        where flip_teams indicates whether raw K-Means cluster 0 should be mapped to Team 1.
        """
        if self.team_0_center is None or self.team_1_center is None:
            # First frame initialization: break symmetry deterministically
            if default_flip:
                self.team_0_center = c1_unweighted.copy()
                self.team_1_center = c0_unweighted.copy()
                return True, self.team_0_center, self.team_1_center
            else:
                self.team_0_center = c0_unweighted.copy()
                self.team_1_center = c1_unweighted.copy()
                return False, self.team_0_center, self.team_1_center

        # Compute permutation distances to stored reference centroids
        d00 = float(np.linalg.norm((c0_unweighted - self.team_0_center) * self.weights))
        d01 = float(np.linalg.norm((c0_unweighted - self.team_1_center) * self.weights))
        d10 = float(np.linalg.norm((c1_unweighted - self.team_0_center) * self.weights))
        d11 = float(np.linalg.norm((c1_unweighted - self.team_1_center) * self.weights))

        if (d00 + d11) <= (d01 + d10):
            # Cluster 0 matches Team 0, Cluster 1 matches Team 1
            flip_teams = False
            matched_0 = c0_unweighted
            matched_1 = c1_unweighted
        else:
            # Cluster 0 matches Team 1, Cluster 1 matches Team 0
            flip_teams = True
            matched_0 = c1_unweighted
            matched_1 = c0_unweighted

        # Smooth EMA centroid update
        self.team_0_center = self.momentum * self.team_0_center + (1.0 - self.momentum) * matched_0
        self.team_1_center = self.momentum * self.team_1_center + (1.0 - self.momentum) * matched_1

        return flip_teams, self.team_0_center, self.team_1_center


def extract_torso_color_feature(
    frame: np.ndarray,
    box: Tuple[int, int, int, int]
) -> np.ndarray:
    """
    Extracts a robust perceptual color representation from a player's chest core.
    
    Parameters:
    -----------
    frame : Full BGR video frame
    box : (x1, y1, x2, y2) player bounding box
    
    Returns:
    --------
    np.ndarray: [L*, a*, b*, Saturation] feature vector of the kit fabric
    """
    x1, y1, x2, y2 = box
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)

    # Sample inner chest: upper 18%..46% height, inner 25%..75% width
    # Avoids head, neck, shorts, and grass border bleed
    tx1 = max(0, min(frame.shape[1] - 1, x1 + int(0.25 * w)))
    tx2 = max(tx1 + 2, min(frame.shape[1], x1 + int(0.75 * w)))
    ty1 = max(0, min(frame.shape[0] - 1, y1 + int(0.18 * h)))
    ty2 = max(ty1 + 2, min(frame.shape[0], y1 + int(0.46 * h)))

    crop = frame[ty1:ty2, tx1:tx2]
    if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
        return np.array([128.0, 128.0, 128.0, 50.0], dtype=np.float32)

    try:
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        # 1. Pitch turf green subtraction (Hue ~ 38 to 85 in OpenCV 0..180, Saturation > 50)
        # Calibrated strictly to protect neon-yellow, gold, and lime referee jerseys (Hue 28..36)
        grass_mask = (hsv[:, :, 0] >= 38) & (hsv[:, :, 0] <= 85) & (hsv[:, :, 1] > 50)
        # 2. Skin tone subtraction (Hue ~ 4 to 22, low Saturation or high Brightness)
        skin_mask = (hsv[:, :, 0] >= 4) & (hsv[:, :, 0] <= 22) & (hsv[:, :, 1] > 50) & (hsv[:, :, 2] > 110)

        exclude_mask = grass_mask | skin_mask
        valid_mask = ~exclude_mask

        if np.count_nonzero(valid_mask) >= 6:
            v_lab = lab[valid_mask]
            v_sat = hsv[:, :, 1][valid_mask]
        else:
            # Fallback to whole torso crop if kit itself happens to match green/skin hue
            v_lab = lab.reshape(-1, 3)
            v_sat = hsv[:, :, 1].reshape(-1)

        # Robust median resists shirt sponsors, numbers, and sleeve cuffs
        med_l, med_a, med_b = np.median(v_lab, axis=0)
        med_s = float(np.median(v_sat))

        return np.array([float(med_l), float(med_a), float(med_b), med_s], dtype=np.float32)
    except Exception:
        return np.array([128.0, 128.0, 128.0, 50.0], dtype=np.float32)


def classify_player_jerseys(
    players: List[Dict[str, Any]],
    frame: Optional[np.ndarray] = None,
    color_team_0: Tuple[int, int, int] = TEAM_0_HEX,
    color_team_1: Tuple[int, int, int] = TEAM_1_HEX,
    frame_width: int = 1920,
    centroid_tracker: Optional[JerseyCentroidTracker] = None,
    transformer: Optional[Any] = None
) -> List[PlayerDetection]:
    """
    Perceptually clusters detected players into exactly two teams using CIELAB K-Means++.
    Separates Goalkeepers and Referees with temporal centroid tracking.
    
    Parameters:
    -----------
    players : List of raw player dicts containing 'box', 'conf', 'color_feat' (or 'crop')
    frame : Optional video frame for on-the-fly torso feature extraction
    color_team_0 : Tactical BGR color for Team 0
    color_team_1 : Tactical BGR color for Team 1
    frame_width : Video frame pixel width (for horizontal coordinate centering)
    centroid_tracker : Optional persistent JerseyCentroidTracker for inter-frame temporal stability
    transformer : Optional HomographyTransformer for pitch metric coordinate projection
    
    Returns:
    --------
    List[PlayerDetection]: Fully typed PlayerDetection contract objects with assigned team_id
    """
    if not players:
        return []

    # Separate Referees, Goalkeepers, and Outfield players
    outfield = []
    goalkeepers = []
    referees = []
    for p in players:
        if p.get('is_referee', False) or p.get('cls_id') == 3:
            referees.append(p)
        elif p.get('is_gk', False) or p.get('cls_id') == 1:
            goalkeepers.append(p)
        else:
            outfield.append(p)

    # Ensure all outfield players have color features
    for p in outfield:
        if 'color_feat' not in p or p['color_feat'] is None:
            if frame is not None and 'box' in p:
                p['color_feat'] = extract_torso_color_feature(frame, p['box'])
            else:
                p['color_feat'] = np.array([128.0, 128.0, 128.0, 50.0], dtype=np.float32)

    mid_x = float(frame_width) / 2.0
    pitch_len = float(getattr(transformer, "pitch_length", 105.0)) if transformer is not None else 105.0
    mid_pitch_x = pitch_len / 2.0

    # For tiny sample sizes (< 4 outfield players)
    if len(outfield) < 4:
        all_players = outfield + goalkeepers
        res = []
        for p in all_players:
            feat = p.get('color_feat', np.array([128.0, 128.0, 128.0, 50.0]))
            t_id = 0 if feat[0] > 140.0 else 1
            c = color_team_0 if t_id == 0 else color_team_1
            res.append(PlayerDetection(
                box=p['box'],
                confidence=p.get('conf', 1.0),
                class_id=p.get('cls_id', 0),
                team_id=t_id,
                jersey_color=c
            ))
        for ref in referees:
            res.append(PlayerDetection(
                box=ref['box'],
                confidence=ref.get('conf', 1.0),
                class_id=3,
                team_id=-1,
                jersey_color=REFEREE_HEX
            ))
        return res

    # Extract feature matrix for K-Means: (N, 4) -> [L*, a*, b*, Saturation]
    feats = np.array([p['color_feat'] for p in outfield], dtype=np.float32)
    weights = np.array([0.7, 1.8, 1.8, 0.9], dtype=np.float32)
    weighted_feats = feats * weights

    # K-Means++ 2-cluster separation
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    flags = cv2.KMEANS_PP_CENTERS
    _, labels, centers = cv2.kmeans(weighted_feats, 2, None, criteria, 10, flags)

    flat_labels = labels.ravel()

    # Determine canonical Team 0 vs Team 1
    c0_unweighted = centers[0] / weights
    c1_unweighted = centers[1] / weights
    default_flip = bool(c1_unweighted[0] > c0_unweighted[0])

    if centroid_tracker is not None:
        flip_teams, active_c0, active_c1 = centroid_tracker.update_and_match(c0_unweighted, c1_unweighted, default_flip)
        eval_centers = [active_c0 * weights, active_c1 * weights] if not flip_teams else [active_c1 * weights, active_c0 * weights]
    else:
        flip_teams = default_flip
        eval_centers = [centers[0], centers[1]]

    # Outlier / Chromatic Referee & Goalkeeper Detection:
    dists_0 = np.linalg.norm(weighted_feats - eval_centers[0], axis=1)
    dists_1 = np.linalg.norm(weighted_feats - eval_centers[1], axis=1)
    min_dists = np.minimum(dists_0, dists_1)
    mean_dist = float(np.mean(min_dists)) if len(min_dists) > 0 else 20.0
    outlier_cutoff = max(52.0, mean_dist * 2.3)

    # Determine spatial pitch defending halves for outfield teams
    t0_xs = [float(p['box'][0] + p['box'][2]) / 2.0 for idx, p in enumerate(outfield) if (flat_labels[idx] == 0 and not flip_teams) or (flat_labels[idx] == 1 and flip_teams)]
    t1_xs = [float(p['box'][0] + p['box'][2]) / 2.0 for idx, p in enumerate(outfield) if (flat_labels[idx] == 1 and not flip_teams) or (flat_labels[idx] == 0 and flip_teams)]
    team_0_left_defending = True
    if t0_xs and t1_xs:
        team_0_left_defending = (np.mean(t0_xs) <= np.mean(t1_xs))

    result: List[PlayerDetection] = []

    # Assign Outfield Players (isolating chromatic outliers as referees or distinct goalkeepers)
    for idx, p in enumerate(outfield):
        m_dist = float(min_dists[idx])
        c_feat = p.get('color_feat', np.array([128.0, 128.0, 128.0, 50.0]))
        # Distinct referee kits: high saturation fluorescent/neon (Sat > 85), very bright (L > 195), or stark all-black (L < 45)
        is_ref_profile = bool(c_feat[3] > 85.0 or c_feat[0] > 195.0 or c_feat[0] < 45.0)
        is_chromatic_outlier = bool(m_dist > outlier_cutoff and m_dist > 52.0 and (is_ref_profile or m_dist > 75.0))

        if is_chromatic_outlier:
            # Check if this outlier is near a goal line (indicating a goalkeeper wearing an outlier kit)
            box = p['box']
            px = (float(box[0]) + float(box[2])) / 2.0
            py = float(box[3])
            is_in_goal_zone = False

            if transformer is not None and hasattr(transformer, 'image_to_pitch'):
                try:
                    pitch_pts = transformer.image_to_pitch([(px, py)], filter_out_of_bounds=False)
                    if pitch_pts and len(pitch_pts) > 0:
                        mx = pitch_pts[0][0]
                        if mx < 18.0 or mx > (pitch_len - 18.0):
                            is_in_goal_zone = True
                except Exception:
                    pass
            if not is_in_goal_zone:
                if px < (frame_width * 0.15) or px > (frame_width * 0.85):
                    is_in_goal_zone = True

            if is_in_goal_zone:
                # Distinct GK kit: attribute to defending goal team
                is_left = (px < mid_x)
                if transformer is not None and hasattr(transformer, 'image_to_pitch'):
                    try:
                        pts = transformer.image_to_pitch([(px, py)], filter_out_of_bounds=False)
                        if pts: is_left = (pts[0][0] < mid_pitch_x)
                    except Exception:
                        pass
                gk_team = 0 if ((is_left and team_0_left_defending) or (not is_left and not team_0_left_defending)) else 1
                result.append(PlayerDetection(
                    box=p['box'],
                    confidence=p.get('conf', 1.0),
                    class_id=1,
                    team_id=gk_team,
                    jersey_color=color_team_0 if gk_team == 0 else color_team_1
                ))
                continue
            else:
                # True chromatic referee in outfield area
                result.append(PlayerDetection(
                    box=p['box'],
                    confidence=p.get('conf', 1.0),
                    class_id=3,
                    team_id=-1,
                    jersey_color=REFEREE_HEX
                ))
                continue

        raw_cluster = int(flat_labels[idx])
        assigned_team = (1 - raw_cluster) if flip_teams else raw_cluster
        assigned_color = color_team_1 if assigned_team == 1 else color_team_0

        result.append(PlayerDetection(
            box=p['box'],
            confidence=p.get('conf', 1.0),
            class_id=p.get('cls_id', 0),
            team_id=assigned_team,
            jersey_color=assigned_color
        ))

    # Append explicitly tagged Referees (neural class 3)
    for ref in referees:
        result.append(PlayerDetection(
            box=ref['box'],
            confidence=ref.get('conf', 1.0),
            class_id=3,
            team_id=-1,
            jersey_color=REFEREE_HEX
        ))

    # Assign Goalkeepers deterministically by spatial defending half
    for gk in goalkeepers:
        box = gk['box']
        gx = (float(box[0]) + float(box[2])) / 2.0
        gy = float(box[3])

        is_left_side = (gx < mid_x)
        if transformer is not None and hasattr(transformer, 'image_to_pitch'):
            try:
                pitch_pts = transformer.image_to_pitch([(gx, gy)], filter_out_of_bounds=False)
                if pitch_pts and len(pitch_pts) > 0:
                    is_left_side = (pitch_pts[0][0] < mid_pitch_x)
            except Exception:
                is_left_side = (gx < mid_x)

        if is_left_side:
            gk_team = 0 if team_0_left_defending else 1
        else:
            gk_team = 1 if team_0_left_defending else 0

        gk_color = color_team_0 if gk_team == 0 else color_team_1
        result.append(PlayerDetection(
            box=box,
            confidence=gk.get('conf', 1.0),
            class_id=gk.get('cls_id', 1),
            team_id=gk_team,
            jersey_color=gk_color
        ))

    return result
