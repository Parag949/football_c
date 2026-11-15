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
) -> List[PlayerDetection]:
    """
    Perceptually clusters detected players into exactly two teams using CIELAB K-Means++.
    Separates Goalkeepers and Referees.
    
    Parameters:
    -----------
    players : List of raw player dicts containing 'box', 'conf', 'color_feat' (or 'crop')
    frame : Optional video frame for on-the-fly torso feature extraction
    
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

    # Determine canonical Team 0 vs Team 1: Home team is assigned the cluster with higher lightness/contrast
    c0_unweighted = centers[0] / weights
    c1_unweighted = centers[1] / weights
    flip_teams = bool(c1_unweighted[0] > c0_unweighted[0])

    # Outlier / Chromatic Referee Detection:
    # Compute Euclidean distance from each candidate to both kit centroids
    dists_0 = np.linalg.norm(weighted_feats - centers[0], axis=1)
    dists_1 = np.linalg.norm(weighted_feats - centers[1], axis=1)
    min_dists = np.minimum(dists_0, dists_1)
    mean_dist = float(np.mean(min_dists)) if len(min_dists) > 0 else 20.0
    outlier_cutoff = max(52.0, mean_dist * 2.3)

    result: List[PlayerDetection] = []

    # Assign Outfield Players (isolating chromatic outliers as referees)
    for idx, p in enumerate(outfield):
        m_dist = float(min_dists[idx])
        # Chromatic outlier test: distance to both kit clusters exceeds outlier threshold
        is_chromatic_ref = bool(m_dist > outlier_cutoff and m_dist > 45.0)

        if is_chromatic_ref:
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

    # Append explicitly tagged Referees
    for ref in referees:
        result.append(PlayerDetection(
            box=ref['box'],
            confidence=ref.get('conf', 1.0),
            class_id=3,
            team_id=-1,
            jersey_color=REFEREE_HEX
        ))

    # Assign Goalkeepers deterministically by spatial defending half
    # GK closest to left goal line (X < frame_center) defends for Team 0; GK on right defends for Team 1
    for gk in goalkeepers:
        box = gk['box']
        gx = (box[0] + box[2]) / 2.0
        gk_team = 0 if gx < 960 else 1
        gk_color = color_team_0 if gk_team == 0 else color_team_1
        result.append(PlayerDetection(
            box=box,
            confidence=gk.get('conf', 1.0),
            class_id=gk.get('cls_id', 1),
            team_id=gk_team,
            jersey_color=gk_color
        ))

    return result
