"""
Interactive Pitch Mask Alignment Studio (Streamlit Component v2)
================================================================
Overlays a live-warping FIFA pitch chalk mask directly over the match video frame.
Combines flat perspective homography and curved panorama mesh warping into a single
customizable editor: choose 4, 8, 13, or 21 landmark presets, click "+ Add Point"
to drop unlimited extra landmarks anywhere on the pitch, drag handles with a 2.5x
magnifier loupe, and commit one unified H + TPS residual transform.
"""

import base64
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import streamlit as st

PITCH_LEN = 105.0
PITCH_WID = 68.0

def get_pitch_landmarks_def(pitch_length: float = 105.0, pitch_width: float = 68.0) -> List[Dict[str, Any]]:
    """
    Dynamically generates the 27 canonical pitch landmarks for arbitrary pitch dimensions (L x W)
    in accordance with FIFA IFAB Law 1:
    - 18-yard box is locked to 16.5m from goal lines and 40.32m centered on width.
    - 6-yard box is locked to 5.5m from goal lines and 18.32m centered on width.
    - Halfway line, center circle, and penalty marks adapt to length and width.
    """
    l = float(pitch_length)
    w = float(pitch_width)
    box_top_y = round((w - 40.32) / 2.0, 2)
    box_bot_y = round((w + 40.32) / 2.0, 2)
    six_top_y = round((w - 18.32) / 2.0, 2)
    six_bot_y = round((w + 18.32) / 2.0, 2)
    mid_x = round(l / 2.0, 2)
    mid_y = round(w / 2.0, 2)
    q1_x = round(l * 0.25, 2)
    q3_x = round(l * 0.75, 2)

    return [
        # 4 Corners
        {"id": "TL",    "label": "Top-Left Corner",             "xm": 0.0,    "ym": 0.0,   "role": "corner"},
        {"id": "TR",    "label": "Top-Right Corner",            "xm": l,      "ym": 0.0,   "role": "corner"},
        {"id": "BR",    "label": "Bottom-Right Corner",         "xm": l,      "ym": w,     "role": "corner"},
        {"id": "BL",    "label": "Bottom-Left Corner",          "xm": 0.0,    "ym": w,     "role": "corner"},
        # 13-Point Curved Panorama / Touchline + Spine Anchors
        {"id": "TM25",  "label": f"Top Quarter-Left ({q1_x}m)",  "xm": q1_x,   "ym": 0.0,   "role": "edge"},
        {"id": "TM50",  "label": f"Top Halfway ({mid_x}m)",        "xm": mid_x,  "ym": 0.0,   "role": "edge"},
        {"id": "TM75",  "label": f"Top Quarter-Right ({q3_x}m)", "xm": q3_x,   "ym": 0.0,   "role": "edge"},
        {"id": "BM25",  "label": f"Bottom Quarter-Left ({q1_x}m)","xm": q1_x,  "ym": w,     "role": "edge"},
        {"id": "BM50",  "label": f"Bottom Halfway ({mid_x}m)",     "xm": mid_x,  "ym": w,     "role": "edge"},
        {"id": "BM75",  "label": f"Bottom Quarter-Right ({q3_x}m)","xm": q3_x, "ym": w,     "role": "edge"},
        {"id": "LP",    "label": f"Left 18-Yd Line Midpoint (16.5m, {mid_y}m)", "xm": 16.5, "ym": mid_y,  "role": "inner"},
        {"id": "CS",    "label": f"Center Spot ({mid_x}m, {mid_y}m)", "xm": mid_x, "ym": mid_y,  "role": "inner"},
        {"id": "RP",    "label": f"Right 18-Yd Line Midpoint ({round(l - 16.5, 2)}m, {mid_y}m)", "xm": round(l - 16.5, 2), "ym": mid_y,  "role": "inner"},
        # Penalty Boxes, Center Circle & 6-Yard Boxes
        {"id": "L18T",  "label": "Left 18-Yd Box Top Corner",   "xm": 16.5,   "ym": box_top_y, "role": "inner"},
        {"id": "L18B",  "label": "Left 18-Yd Box Bottom Corner","xm": 16.5,   "ym": box_bot_y, "role": "inner"},
        {"id": "R18T",  "label": "Right 18-Yd Box Top Corner",  "xm": round(l - 16.5, 2), "ym": box_top_y, "role": "inner"},
        {"id": "R18B",  "label": "Right 18-Yd Box Bottom Corner","xm": round(l - 16.5, 2), "ym": box_bot_y, "role": "inner"},
        {"id": "CCT",   "label": f"Center Circle Top ({mid_x}m)",   "xm": mid_x,  "ym": round(mid_y - 9.15, 2), "role": "inner"},
        {"id": "CCB",   "label": f"Center Circle Bottom ({mid_x}m)","xm": mid_x, "ym": round(mid_y + 9.15, 2), "role": "inner"},
        {"id": "L6T",   "label": "Left 6-Yd Box Top Corner",    "xm": 5.5,    "ym": six_top_y, "role": "inner"},
        {"id": "R6T",   "label": "Right 6-Yd Box Top Corner",   "xm": round(l - 5.5, 2), "ym": six_top_y, "role": "inner"},
        {"id": "L6B",   "label": "Left 6-Yd Box Bottom Corner", "xm": 5.5,    "ym": six_bot_y, "role": "inner"},
        {"id": "R6B",   "label": "Right 6-Yd Box Bottom Corner","xm": round(l - 5.5, 2), "ym": six_bot_y, "role": "inner"},
        {"id": "L18GT", "label": "Left Goal-Line 18-Yd Top",    "xm": 0.0,    "ym": box_top_y, "role": "edge"},
        {"id": "L18GB", "label": "Left Goal-Line 18-Yd Bottom", "xm": 0.0,    "ym": box_bot_y, "role": "edge"},
        {"id": "R18GT", "label": "Right Goal-Line 18-Yd Top",   "xm": l,      "ym": box_top_y, "role": "edge"},
        {"id": "R18GB", "label": "Right Goal-Line 18-Yd Bottom","xm": l,      "ym": box_bot_y, "role": "edge"},
    ]


# 27 canonical FIFA pitch landmarks [0..105m] x [0..68m] + named layout sets
PITCH_LANDMARKS_DEF: List[Dict[str, Any]] = get_pitch_landmarks_def(105.0, 68.0)

# Backward-compatible alias for any callers referencing the 13 core handles
MESH_HANDLES_DEF = PITCH_LANDMARKS_DEF[:13]

LAYOUT_4_IDS = ["TL", "TR", "BR", "BL"]
LAYOUT_4_GOAL_IDS = ["TR", "BR", "BM50", "TM50"]
LAYOUT_8_IDS = ["TL", "TR", "BR", "BL", "L18T", "L18B", "R18T", "R18B"]
LAYOUT_8_GOAL_IDS = ["TR", "BR", "R18T", "R18B", "R18GT", "R18GB", "TM50", "BM50"]
LAYOUT_13_IDS = ["TL", "TR", "BR", "BL", "TM25", "TM50", "TM75", "BM50", "CS", "L18T", "L18B", "R18T", "R18B"]
LAYOUT_21_IDS = [item["id"] for item in PITCH_LANDMARKS_DEF[:21]]
LAYOUT_ENDZONE_IDS = ["TR", "BR", "R18T", "R18B", "TM50", "BM50", "CS"]

_HTML = """
<div class="pm-shell">
  <div class="pm-toolbar">
    <div class="pm-group">
      <button type="button" id="pm-fit-flat" class="pm-btn pm-btn-mode" title="Normal Broadcast (16:9): Flat perspective with tilt/pan compensation">Normal Broadcast</button>
      <button type="button" id="pm-fit-curve" class="pm-btn pm-btn-mode" title="Wide-Angle Lens: Optical lens curvature compensation (Division Model)">Wide-Angle Lens</button>
      <button type="button" id="pm-fit-tps" class="pm-btn pm-btn-mode" title="Panorama Multi-Cam: Thin-Plate Spline interpolation locking mask directly to anchor handles (0.00 px error)">Panorama Multi-Cam</button>
      <label class="pm-slider-lbl" id="pm-curve-wrap" style="display:none;">Lens Curvature <input type="range" id="pm-curve-slider" min="-30" max="10" value="-10" step="1" title="Adjust optical barrel distortion" /><span id="pm-curve-val" style="font-family:monospace; margin-left:4px;">-0.10</span></label>
      <span class="pm-divider"></span>
      <label class="pm-slider-lbl" title="Pitch Length in meters (FIFA: 90m - 120m, Standard: 105m)">L: <input type="number" id="pm-pitch-l" min="90" max="120" step="0.5" value="105.0" style="width:50px; background:#0f172a; border:1px solid #334155; color:#38bdf8; border-radius:4px; padding:2px 4px; font-family:monospace; font-weight:bold; font-size:11px;" />m</label>
      <label class="pm-slider-lbl" title="Pitch Width in meters (FIFA: 45m - 90m, Standard: 68m)">W: <input type="number" id="pm-pitch-w" min="45" max="90" step="0.5" value="68.0" style="width:50px; background:#0f172a; border:1px solid #334155; color:#38bdf8; border-radius:4px; padding:2px 4px; font-family:monospace; font-weight:bold; font-size:11px;" />m</label>
      <select id="pm-layout" class="pm-select" title="Switch number of anchor points">
        <option value="4">4 Corners (Basic Pitch)</option>
        <option value="8">8 Points (Corners &amp; 18-Yd Boxes)</option>
        <option value="13">13 Points (Boxes &amp; Midfield)</option>
        <option value="21">21 Points (Full Pitch)</option>
        <option value="custom" disabled>Custom Point Set</option>
      </select>
      <select id="pm-preset" class="pm-select" title="Load a starting mask pose">
        <option value="current">Pose: Current Alignment</option>
        <option value="panorama">Pose: Wide / Panorama</option>
        <option value="sideline">Pose: Broadcast Sideline</option>
        <option value="endzone">Pose: Goal-to-Goal End Zone</option>
      </select>
    </div>
    <div class="pm-group">
      <button type="button" id="pm-add-mode" class="pm-btn" title="Click on the pitch mask (or a faint + landmark) to add a point">+ Add Point</button>
      <select id="pm-add-landmark" class="pm-select" title="Add a specific FIFA pitch landmark"></select>
      <button type="button" id="pm-remove-pt" class="pm-btn" title="Remove currently selected point (or right-click any handle)">Remove Point</button>
      <button type="button" id="pm-move-all" class="pm-btn" title="Drag anywhere to shift the entire mask">Move All</button>
    </div>
    <div class="pm-group">
      <button type="button" id="pm-zoom-out" class="pm-btn" title="Zoom Out">-</button>
      <button type="button" id="pm-zoom-reset" class="pm-btn" title="Fit Frame">Fit</button>
      <button type="button" id="pm-zoom-in" class="pm-btn" title="Zoom In">+</button>
      <label class="pm-slider-lbl">Opacity <input type="range" id="pm-opacity" min="20" max="100" value="80" /></label>
      <select id="pm-color" class="pm-select" title="Mask Line Color">
        <option value="#00f0ff">Cyan</option>
        <option value="#22c55e">Green</option>
        <option value="#facc15">Yellow</option>
        <option value="#ffffff">White</option>
      </select>
    </div>
    <div class="pm-group pm-actions">
      <span id="pm-badge" class="pm-badge">13 pts</span>
      <button type="button" id="pm-reset" class="pm-btn">Reset</button>
      <button type="button" id="pm-apply" class="pm-btn pm-btn-primary">Apply Alignment</button>
    </div>
  </div>
  <div class="pm-canvas-wrap" id="pm-wrap">
    <canvas id="pm-canvas"></canvas>
    <div class="pm-hint" id="pm-hint">
      Drag colored handles onto pitch lines &bull; Switch 4/8/13/21 points anytime or click "+ Add Point" to drop extra pins &bull; Right-click a point to remove &bull; Scroll to zoom
    </div>
  </div>
</div>
"""

_CSS = """
.pm-shell {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  background: #0b1220;
  border: 1px solid #1e293b;
  border-radius: 10px;
  padding: 10px;
  color: #e2e8f0;
  user-select: none;
}
.pm-toolbar {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 8px;
}
.pm-group {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  align-items: center;
}
.pm-divider {
  width: 1px;
  height: 20px;
  background: #334155;
  margin: 0 4px;
}
.pm-btn, .pm-select {
  background: #162033;
  color: #e2e8f0;
  border: 1px solid #334155;
  border-radius: 6px;
  padding: 5px 9px;
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
}
.pm-btn:hover:not(:disabled) { background: #1e293b; border-color: #475569; }
.pm-btn:disabled { opacity: 0.45; cursor: not-allowed; }
.pm-btn.active {
  background: #0e7490;
  border-color: #22d3ee;
  color: #ffffff;
}
.pm-btn-primary {
  background: #0284c7;
  border-color: #38bdf8;
  color: #ffffff;
}
.pm-btn-primary:hover:not(:disabled) { background: #0369a1; }
.pm-badge {
  background: #1e293b;
  color: #38bdf8;
  border: 1px solid #334155;
  border-radius: 999px;
  padding: 3px 9px;
  font-size: 11px;
  font-weight: 700;
}
.pm-slider-lbl {
  display: flex;
  align-items: center;
  gap: 5px;
  font-size: 11px;
  color: #94a3b8;
}
.pm-slider-lbl input { width: 68px; accent-color: #38bdf8; }
.pm-canvas-wrap {
  position: relative;
  width: 100%;
  background: #050811;
  border: 1px solid #1e293b;
  border-radius: 8px;
  overflow: hidden;
}
#pm-canvas {
  display: block;
  width: 100%;
  height: auto;
  cursor: grab;
  touch-action: none;
}
#pm-canvas.add-cursor { cursor: crosshair; }
#pm-canvas.dragging { cursor: grabbing; }
.pm-hint {
  padding: 5px 10px;
  font-size: 11px;
  color: #94a3b8;
  background: #0f172a;
  border-top: 1px solid #1e293b;
}
"""

_JS = """
export default function(component) {
  const { data, parentElement, setTriggerValue } = component;
  if (!data || !data.imageDataUrl) return;

  const root = parentElement;
  const canvas = root.getElementById("pm-canvas");
  const ctx = canvas.getContext("2d");
  const hintEl = root.getElementById("pm-hint");
  const btnFlat = root.getElementById("pm-fit-flat");
  const btnCurve = root.getElementById("pm-fit-curve");
  const btnTps = root.getElementById("pm-fit-tps");
  const curveWrap = root.getElementById("pm-curve-wrap");
  const curveSlider = root.getElementById("pm-curve-slider");
  const curveVal = root.getElementById("pm-curve-val");
  const inputPitchL = root.getElementById("pm-pitch-l");
  const inputPitchW = root.getElementById("pm-pitch-w");
  const selLayout = root.getElementById("pm-layout");
  const selPreset = root.getElementById("pm-preset");
  const btnAddMode = root.getElementById("pm-add-mode");
  const selAddLandmark = root.getElementById("pm-add-landmark");
  const btnRemovePt = root.getElementById("pm-remove-pt");
  const btnMoveAll = root.getElementById("pm-move-all");
  const selColor = root.getElementById("pm-color");
  const opInput = root.getElementById("pm-opacity");
  const badgeEl = root.getElementById("pm-badge");

  const vw = Math.max(1, Number(data.vw || 1280));
  const vh = Math.max(1, Number(data.vh || 720));
  let pitchL = Math.max(80.0, Math.min(130.0, Number(data.pitchLength || 105.0)));
  let pitchW = Math.max(40.0, Math.min(100.0, Number(data.pitchWidth || 68.0)));

  const LAYOUT_SETS = {
    "4": data.layout4 || ["TL", "TR", "BR", "BL"],
    "8": data.layout8 || ["TL", "TR", "BR", "BL", "L18T", "L18B", "R18T", "R18B"],
    "13": data.layout13 || [],
    "21": data.layout21 || [],
  };

  let lensLambda = Number(data.lensLambda !== undefined ? data.lensLambda : (data.warpMode === "universal" ? -0.10 : -0.08));
  let fitMode = (data.warpMode === "panorama_tps" || data.forceFitMode === "tps")
    ? "tps"
    : ((data.warpMode === "universal" || data.forceFitMode === "curve") ? "curve" : "flat");
  let moveAllActive = false;
  let addPointMode = false;
  let selectedHandleId = null;
  let customCounter = 1;
  let maskColor = "#00f0ff";
  let maskOpacity = 0.80;

  function updateLandmarkMetricPositions() {
    const boxTopY = (pitchW - 40.32) / 2.0;
    const boxBotY = (pitchW + 40.32) / 2.0;
    const sixTopY = (pitchW - 18.32) / 2.0;
    const sixBotY = (pitchW + 18.32) / 2.0;
    const midX = pitchL / 2.0;
    const midY = pitchW / 2.0;

    const metricMap = {
      TL: [0.0, 0.0],
      TR: [pitchL, 0.0],
      BR: [pitchL, pitchW],
      BL: [0.0, pitchW],
      TM25: [pitchL * 0.25, 0.0],
      TM50: [midX, 0.0],
      TM75: [pitchL * 0.75, 0.0],
      BM25: [pitchL * 0.25, pitchW],
      BM50: [midX, pitchW],
      BM75: [pitchL * 0.75, pitchW],
      LP: [16.5, midY],
      CS: [midX, midY],
      RP: [pitchL - 16.5, midY],
      L18T: [16.5, boxTopY],
      L18B: [16.5, boxBotY],
      R18T: [pitchL - 16.5, boxTopY],
      R18B: [pitchL - 16.5, boxBotY],
      CCT: [midX, midY - 9.15],
      CCB: [midX, midY + 9.15],
      L6T: [5.5, sixTopY],
      L6B: [5.5, sixBotY],
      R6T: [pitchL - 5.5, sixTopY],
      R6B: [pitchL - 5.5, sixBotY],
      L18GT: [0.0, boxTopY],
      L18GB: [0.0, boxBotY],
      R18GT: [pitchL, boxTopY],
      R18GB: [pitchL, boxBotY],
    };

    handles.forEach((h) => {
      if (metricMap[h.id]) {
        h.xm = Number(metricMap[h.id][0].toFixed(2));
        h.ym = Number(metricMap[h.id][1].toFixed(2));
      }
    });
  }

  const MAX_CSS_W = 1280;
  const aspect = vw / vh;
  const dispW = Math.min(MAX_CSS_W, vw);
  const dispH = Math.max(220, Math.round(dispW / aspect));
  canvas.width = dispW;
  canvas.height = dispH;

  let viewZoom = 1.0;
  let viewPanX = 0.0;
  let viewPanY = 0.0;

  function cloneHandles(arr) {
    return (arr || []).map((h) => ({
      id: String(h.id),
      label: String(h.label || h.id),
      xm: Number(h.xm),
      ym: Number(h.ym),
      role: String(h.role || "edge"),
      u: Number(h.u),
      v: Number(h.v),
      active: Boolean(h.active),
    }));
  }

  let handles = cloneHandles(data.handles);
  const initialHandles = cloneHandles(data.handles);
  const presetMap = data.presets || {};

  // Track next U# custom counter cleanly
  handles.forEach((h) => {
    const m = /^U(\\d+)$/.exec(h.id);
    if (m) customCounter = Math.max(customCounter, Number(m[1]) + 1);
  });

  function activeHandles() {
    return handles.filter((h) => h.active);
  }

  function inactivePredefinedHandles() {
    return handles.filter((h) => !h.active && h.role !== "custom");
  }

  // ---- Linear Algebra (Normalized Least-Squares Homography + Damped TPS Residual) ----
  function solveLinear(A, b) {
    const n = b.length;
    const M = A.map((row, i) => [...row, b[i]]);
    for (let col = 0; col < n; col++) {
      let piv = col;
      let maxVal = Math.abs(M[col][col]);
      for (let r = col + 1; r < n; r++) {
        if (Math.abs(M[r][col]) > maxVal) {
          maxVal = Math.abs(M[r][col]);
          piv = r;
        }
      }
      if (maxVal < 1e-12) return null;
      if (piv !== col) {
        const tmp = M[col];
        M[col] = M[piv];
        M[piv] = tmp;
      }
      const diag = M[col][col];
      for (let j = col; j <= n; j++) M[col][j] /= diag;
      for (let r = 0; r < n; r++) {
        if (r === col) continue;
        const f = M[r][col];
        if (Math.abs(f) < 1e-15) continue;
        for (let j = col; j <= n; j++) M[r][j] -= f * M[col][j];
      }
    }
    return M.map((row) => row[n]);
  }

  function invert3x3(m) {
    const a = m[0][0], b = m[0][1], c = m[0][2];
    const d = m[1][0], e = m[1][1], f = m[1][2];
    const g = m[2][0], h = m[2][1], i = m[2][2];
    const A = e * i - f * h;
    const B = -(d * i - f * g);
    const C = d * h - e * g;
    const D = -(b * i - c * h);
    const E = a * i - c * g;
    const F = -(a * h - b * g);
    const G = b * f - c * e;
    const H = -(a * f - c * d);
    const I = a * e - b * d;
    const det = a * A + b * B + c * C;
    if (Math.abs(det) < 1e-14) return null;
    const invDet = 1.0 / det;
    return [
      [A * invDet, D * invDet, G * invDet],
      [B * invDet, E * invDet, H * invDet],
      [C * invDet, F * invDet, I * invDet],
    ];
  }

  const cx = vw * 0.5;
  const cy = vh * 0.5;
  const rNorm = Math.hypot(cx, cy);

  function undistortPixel(u, v, lam) {
    if (!lam || Math.abs(lam) < 1e-6) return [u, v];
    const xd = (u - cx) / rNorm;
    const yd = (v - cy) / rNorm;
    const rd2 = xd * xd + yd * yd;
    const denom = Math.max(1e-4, 1.0 + lam * rd2);
    return [cx + (xd / denom) * rNorm, cy + (yd / denom) * rNorm];
  }

  function distortPixel(uu, vu, lam) {
    if (!lam || Math.abs(lam) < 1e-6) return [uu, vu];
    const xu = (uu - cx) / rNorm;
    const yu = (vu - cy) / rNorm;
    const ru2 = xu * xu + yu * yu;
    const disc = Math.max(0.0, 1.0 - 4.0 * lam * ru2);
    const factor = 2.0 / Math.max(1e-7, 1.0 + Math.sqrt(disc));
    return [cx + xu * factor * rNorm, cy + yu * factor * rNorm];
  }

  function computeHomographyFromHandles(pts) {
    if (!pts || pts.length < 4) return null;

    const M = Array.from({ length: 8 }, () => new Array(8).fill(0.0));
    const rhs = new Array(8).fill(0.0);
    const lam = (fitMode === "curve") ? lensLambda : 0.0;

    for (let k = 0; k < pts.length; k++) {
      const x = pts[k].xm / pitchL;
      const y = pts[k].ym / pitchW;
      const [uu, vu] = undistortPixel(pts[k].u, pts[k].v, lam);
      const u = uu / vw;
      const v = vu / vh;

      const r1 = [x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y];
      const r2 = [0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y];

      for (let i = 0; i < 8; i++) {
        rhs[i] += r1[i] * u + r2[i] * v;
        for (let j = i; j < 8; j++) {
          M[i][j] += r1[i] * r1[j] + r2[i] * r2[j];
        }
      }
    }
    for (let i = 0; i < 8; i++) {
      for (let j = 0; j < i; j++) M[i][j] = M[j][i];
      M[i][i] += 1e-9;
    }

    const sol = solveLinear(M, rhs);
    if (!sol) return null;

    // Un-normalize from [0..1]x[0..1] -> pixel (u, v) from metric (xm, ym)
    const Hinv = [
      [(vw * sol[0]) / pitchL, (vw * sol[1]) / pitchW, vw * sol[2]],
      [(vh * sol[3]) / pitchL, (vh * sol[4]) / pitchW, vh * sol[5]],
      [sol[6] / pitchL, sol[7] / pitchW, 1.0],
    ];
    return Hinv;
  }

  function applyH(Hmat, x, y) {
    if (!Hmat) return [vw * 0.5, vh * 0.5];
    const uh = Hmat[0][0] * x + Hmat[0][1] * y + Hmat[0][2];
    const vh2 = Hmat[1][0] * x + Hmat[1][1] * y + Hmat[1][2];
    const wh = Hmat[2][0] * x + Hmat[2][1] * y + Hmat[2][2];
    if (Math.abs(wh) < 1e-8) return [vw * 0.5, vh * 0.5];
    return [uh / wh, vh2 / wh];
  }

  let cachedHinv = null;
  let cachedHfwd = null;
  let tpsWu = null;
  let tpsWv = null;
  let tpsCtrl = null;

  function rebuildTransform() {
    const pts = activeHandles();
    cachedHinv = computeHomographyFromHandles(pts);
    cachedHfwd = cachedHinv ? invert3x3(cachedHinv) : null;

    // Build Thin-Plate Spline residual for exact anchor handle snapping
    // Active in "tps" / "panorama_tps" mode, or whenever 8+ points are active in curve mode
    const wantTps = (fitMode === "tps" || fitMode === "panorama_tps" || (fitMode === "curve" && pts.length >= 8));
    if (wantTps && pts.length >= 4 && cachedHinv) {
      const n = pts.length;
      const normScale = Math.max(pitchL, pitchW);
      const pNorm = pts.map((p) => [p.xm / normScale, p.ym / normScale]);
      const resU = [];
      const resV = [];

      for (let i = 0; i < n; i++) {
        const base = applyH(cachedHinv, pts[i].xm, pts[i].ym);
        const [bu, bv] = (fitMode === "curve" && Math.abs(lensLambda) > 1e-6)
          ? distortPixel(base[0], base[1], lensLambda)
          : base;
        resU.push(pts[i].u - bu);
        resV.push(pts[i].v - bv);
      }

      function rbfKernel(r) {
        if (r < 1e-7) return 0.0;
        return (r * r) * Math.log(r);
      }

      const dim = n + 3;
      const Lmat = Array.from({ length: dim }, () => new Array(dim).fill(0.0));
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          if (i !== j) {
            const d = Math.hypot(pNorm[i][0] - pNorm[j][0], pNorm[i][1] - pNorm[j][1]);
            Lmat[i][j] = rbfKernel(d);
          }
        }
        Lmat[i][i] += 1e-8; // micro-regularization for exact interpolation
        Lmat[i][n] = 1.0;
        Lmat[i][n + 1] = pNorm[i][0];
        Lmat[i][n + 2] = pNorm[i][1];
        Lmat[n][i] = 1.0;
        Lmat[n + 1][i] = pNorm[i][0];
        Lmat[n + 2][i] = pNorm[i][1];
      }

      const bU = [...resU, 0.0, 0.0, 0.0];
      const bV = [...resV, 0.0, 0.0, 0.0];
      tpsWu = solveLinear(Lmat, bU);
      tpsWv = solveLinear(Lmat, bV);
      tpsCtrl = pNorm;
    } else {
      tpsWu = null;
      tpsWv = null;
      tpsCtrl = null;
    }
  }

  function pitchToVideo(xm, ym) {
    const base = applyH(cachedHinv, xm, ym);
    let curU = base[0], curV = base[1];
    if (fitMode === "curve" && Math.abs(lensLambda) > 1e-6) {
      const dist = distortPixel(base[0], base[1], lensLambda);
      curU = dist[0];
      curV = dist[1];
    }
    if (tpsWu && tpsWv && tpsCtrl) {
      const normScale = Math.max(pitchL, pitchW);
      const nx = xm / normScale, ny = ym / normScale;
      let du = tpsWu[tpsCtrl.length] + tpsWu[tpsCtrl.length + 1] * nx + tpsWu[tpsCtrl.length + 2] * ny;
      let dv = tpsWv[tpsCtrl.length] + tpsWv[tpsCtrl.length + 1] * nx + tpsWv[tpsCtrl.length + 2] * ny;
      for (let i = 0; i < tpsCtrl.length; i++) {
        const d = Math.hypot(nx - tpsCtrl[i][0], ny - tpsCtrl[i][1]);
        if (d > 1e-7) {
          const k = (d * d) * Math.log(d);
          du += tpsWu[i] * k;
          dv += tpsWv[i] * k;
        }
      }
      curU += du;
      curV += dv;
    }
    return [curU, curV];
  }

  function videoToPitch(u, v) {
    if (!cachedHfwd) return [pitchL * 0.5, pitchW * 0.5];
    let uu = u, vu = v;
    if (fitMode === "curve" && Math.abs(lensLambda) > 1e-6) {
      const und = undistortPixel(u, v, lensLambda);
      uu = und[0];
      vu = und[1];
    }
    const pt = applyH(cachedHfwd, uu, vu);
    return [
      Math.max(-2.0, Math.min(pitchL + 2.0, pt[0])),
      Math.max(-2.0, Math.min(pitchW + 2.0, pt[1])),
    ];
  }

  // ---- Coordinate Conversions (Video Native <-> Display Canvas) ----
  function videoToCanvas(u, v) {
    const baseX = (u / vw) * dispW;
    const baseY = (v / vh) * dispH;
    return [baseX * viewZoom + viewPanX, baseY * viewZoom + viewPanY];
  }

  function canvasToVideo(cx, cy) {
    const baseX = (cx - viewPanX) / viewZoom;
    const baseY = (cy - viewPanY) / viewZoom;
    return [(baseX / dispW) * vw, (baseY / dispH) * vh];
  }

  function clampPan() {
    const maxW = dispW * viewZoom;
    const maxH = dispH * viewZoom;
    if (viewZoom <= 1.001) {
      viewZoom = 1.0;
      viewPanX = 0.0;
      viewPanY = 0.0;
      return;
    }
    viewPanX = Math.min(60, Math.max(dispW - maxW - 60, viewPanX));
    viewPanY = Math.min(60, Math.max(dispH - maxH - 60, viewPanY));
  }

  // ---- FIFA Pitch Mask Geometry Polylines ----
  function drawWarpedPolyline(ptsMetric, closed, steps, lineWidth, dashed) {
    ctx.save();
    ctx.beginPath();
    if (dashed) ctx.setLineDash([6, 5]);
    let started = false;
    const n = ptsMetric.length;
    const segs = closed ? n : n - 1;
    for (let i = 0; i < segs; i++) {
      const a = ptsMetric[i];
      const b = ptsMetric[(i + 1) % n];
      for (let s = 0; s <= steps; s++) {
        if (s === 0 && i > 0) continue;
        const t = s / steps;
        const xm = a[0] + (b[0] - a[0]) * t;
        const ym = a[1] + (b[1] - a[1]) * t;
        const uv = pitchToVideo(xm, ym);
        const cp = videoToCanvas(uv[0], uv[1]);
        if (!started) { ctx.moveTo(cp[0], cp[1]); started = true; }
        else { ctx.lineTo(cp[0], cp[1]); }
      }
    }
    if (closed) ctx.closePath();
    ctx.strokeStyle = maskColor;
    ctx.lineWidth = lineWidth;
    ctx.globalAlpha = maskOpacity;
    ctx.stroke();
    ctx.restore();
  }

  function drawWarpedCircle(cxm, cym, rad, nPts, lw) {
    const pts = [];
    for (let i = 0; i < nPts; i++) {
      const a = (i / nPts) * Math.PI * 2;
      pts.push([cxm + rad * Math.cos(a), cym + rad * Math.sin(a)]);
    }
    drawWarpedPolyline(pts, true, 1, lw, false);
  }

  function drawWarpedArc(cxm, cym, rad, aStart, aEnd, nPts, lw) {
    const pts = [];
    for (let i = 0; i <= nPts; i++) {
      const a = aStart + (i / nPts) * (aEnd - aStart);
      pts.push([cxm + rad * Math.cos(a), cym + rad * Math.sin(a)]);
    }
    drawWarpedPolyline(pts, false, 1, lw, false);
  }

  function drawPitchFill() {
    const isGoal = data.cameraMode === "goal_to_goal";
    const startX = isGoal ? (pitchL * 0.5) : 0.0;
    const ring = [];
    const steps = 20;
    for (let s = 0; s < steps; s++) ring.push([startX + (s / steps) * (pitchL - startX), 0.0]);
    for (let s = 0; s < steps; s++) ring.push([pitchL, (s / steps) * pitchW]);
    for (let s = 0; s < steps; s++) ring.push([pitchL - (s / steps) * (pitchL - startX), pitchW]);
    for (let s = 0; s < steps; s++) ring.push([startX, pitchW - (s / steps) * pitchW]);
    ctx.save();
    ctx.beginPath();
    ring.forEach((m, idx) => {
      const uv = pitchToVideo(m[0], m[1]);
      const cp = videoToCanvas(uv[0], uv[1]);
      if (idx === 0) ctx.moveTo(cp[0], cp[1]);
      else ctx.lineTo(cp[0], cp[1]);
    });
    ctx.closePath();
    ctx.fillStyle = maskColor;
    ctx.globalAlpha = Math.max(0.05, maskOpacity * 0.14);
    ctx.fill();
    ctx.restore();
  }

  // ---- Render Loop ----
  const bgImg = new Image();
  let bgLoaded = false;
  let activeDragIdx = -1;
  let dragAllMode = false;
  let panDrag = false;
  let lastPointer = null;

  function render() {
    ctx.clearRect(0, 0, dispW, dispH);
    ctx.fillStyle = "#050811";
    ctx.fillRect(0, 0, dispW, dispH);

    if (bgLoaded) {
      ctx.save();
      ctx.translate(viewPanX, viewPanY);
      ctx.scale(viewZoom, viewZoom);
      ctx.drawImage(bgImg, 0, 0, dispW, dispH);
      ctx.restore();
    }

    drawPitchFill();

    const halfL = pitchL * 0.5;
    const halfW = pitchW * 0.5;
    const boxDepth = 16.5;
    const boxHalfW = 20.16; // 40.32m width / 2
    const boxTopY = halfW - boxHalfW;
    const boxBotY = halfW + boxHalfW;
    const sixDepth = 5.5;
    const sixHalfW = 9.16; // 18.32m width / 2
    const sixTopY = halfW - sixHalfW;
    const sixBotY = halfW + sixHalfW;
    const penDist = 11.0;
    const rightPenX = pitchL - penDist;

    const isGoal = data.cameraMode === "goal_to_goal";
    if (isGoal) {
      // Attacking half boundary: halfway line to far goal line
      drawWarpedPolyline([[halfL, 0], [pitchL, 0], [pitchL, pitchW], [halfL, pitchW]], true, 24, 2.4, false);
      // Halfway line + center circle + center spot
      drawWarpedPolyline([[halfL, 0], [halfL, pitchW]], false, 14, 2.0, false);
      drawWarpedCircle(halfL, halfW, 9.15, 40, 1.8);
      // Right 18-yard & 6-yard boxes + penalty arc (attacking end)
      drawWarpedPolyline([[pitchL, boxTopY], [pitchL - boxDepth, boxTopY], [pitchL - boxDepth, boxBotY], [pitchL, boxBotY]], false, 10, 2.0, false);
      drawWarpedPolyline([[pitchL, sixTopY], [pitchL - sixDepth, sixTopY], [pitchL - sixDepth, sixBotY], [pitchL, sixBotY]], false, 6, 1.5, false);
      drawWarpedArc(rightPenX, halfW, 9.15, Math.PI - 0.927, Math.PI + 0.927, 20, 1.8);
    } else {
      // Outer pitch boundary (subdivided heavily so curved panoramas flex smoothly)
      drawWarpedPolyline([[0, 0], [pitchL, 0], [pitchL, pitchW], [0, pitchW]], true, 24, 2.4, false);
      // Halfway line + center circle + center spot
      drawWarpedPolyline([[halfL, 0], [halfL, pitchW]], false, 14, 2.0, false);
      drawWarpedCircle(halfL, halfW, 9.15, 40, 1.8);
      // Left 18-yard & 6-yard boxes + penalty arc
      drawWarpedPolyline([[0, boxTopY], [boxDepth, boxTopY], [boxDepth, boxBotY], [0, boxBotY]], false, 10, 2.0, false);
      drawWarpedPolyline([[0, sixTopY], [sixDepth, sixTopY], [sixDepth, sixBotY], [0, sixBotY]], false, 6, 1.5, false);
      drawWarpedArc(penDist, halfW, 9.15, -0.927, 0.927, 20, 1.8);
      // Right 18-yard & 6-yard boxes + penalty arc
      drawWarpedPolyline([[pitchL, boxTopY], [pitchL - boxDepth, boxTopY], [pitchL - boxDepth, boxBotY], [pitchL, boxBotY]], false, 10, 2.0, false);
      drawWarpedPolyline([[pitchL, sixTopY], [pitchL - sixDepth, sixTopY], [pitchL - sixDepth, sixBotY], [pitchL, sixBotY]], false, 6, 1.5, false);
      drawWarpedArc(rightPenX, halfW, 9.15, Math.PI - 0.927, Math.PI + 0.927, 20, 1.8);
    }

    // Draw clickable ghost "+" pins for inactive landmarks while Add Point mode is active
    if (addPointMode) {
      inactivePredefinedHandles().forEach((h) => {
        const uv = pitchToVideo(h.xm, h.ym);
        const cp = videoToCanvas(uv[0], uv[1]);
        ctx.save();
        ctx.beginPath();
        ctx.arc(cp[0], cp[1], 6, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(15, 23, 42, 0.72)";
        ctx.fill();
        ctx.lineWidth = 1.5;
        ctx.setLineDash([2, 2]);
        ctx.strokeStyle = "#f8fafc";
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.beginPath();
        ctx.moveTo(cp[0] - 3, cp[1]); ctx.lineTo(cp[0] + 3, cp[1]);
        ctx.moveTo(cp[0], cp[1] - 3); ctx.lineTo(cp[0], cp[1] + 3);
        ctx.strokeStyle = "#38bdf8";
        ctx.lineWidth = 1.4;
        ctx.stroke();
        ctx.restore();
      });
    }

    // Draw active handles
    activeHandles().forEach((h) => {
      const cp = videoToCanvas(h.u, h.v);
      const isSelected = (h.id === selectedHandleId) || (activeDragIdx >= 0 && handles[activeDragIdx] && handles[activeDragIdx].id === h.id);
      const r = h.role === "corner" ? 8 : 6;
      const fill = h.role === "corner" ? "#facc15" : (h.role === "custom" ? "#f472b6" : (h.role === "inner" ? "#22c55e" : "#38bdf8"));

      ctx.save();
      ctx.beginPath();
      if (h.role === "corner") {
        ctx.rect(cp[0] - r, cp[1] - r, r * 2, r * 2);
      } else if (h.role === "custom") {
        ctx.moveTo(cp[0], cp[1] - r - 1);
        ctx.lineTo(cp[0] + r + 1, cp[1]);
        ctx.lineTo(cp[0], cp[1] + r + 1);
        ctx.lineTo(cp[0] - r - 1, cp[1]);
        ctx.closePath();
      } else {
        ctx.arc(cp[0], cp[1], r, 0, Math.PI * 2);
      }
      ctx.fillStyle = fill;
      ctx.fill();
      ctx.lineWidth = isSelected ? 3 : 2;
      ctx.strokeStyle = isSelected ? "#ffffff" : "#0f172a";
      ctx.stroke();

      ctx.font = "bold 10px monospace";
      const lbl = h.id;
      const tw = ctx.measureText(lbl).width + 8;
      ctx.fillStyle = isSelected ? "rgba(2, 132, 199, 0.92)" : "rgba(15, 23, 42, 0.85)";
      ctx.fillRect(cp[0] + 10, cp[1] - 10, tw, 15);
      ctx.fillStyle = "#f8fafc";
      ctx.fillText(lbl, cp[0] + 14, cp[1] + 1);
      ctx.restore();
    });

    // Precision 2.5x Magnifier Loupe while dragging a handle
    if (activeDragIdx >= 0 && handles[activeDragIdx] && bgLoaded) {
      const dh = handles[activeDragIdx];
      const lr = 58;
      const lx = dispW - lr - 16;
      const ly = lr + 16;
      const srcPatch = 90;
      ctx.save();
      ctx.beginPath();
      ctx.arc(lx, ly, lr, 0, Math.PI * 2);
      ctx.clip();
      ctx.fillStyle = "#050811";
      ctx.fillRect(lx - lr, ly - lr, lr * 2, lr * 2);
      ctx.drawImage(
        bgImg,
        dh.u - srcPatch, dh.v - srcPatch, srcPatch * 2, srcPatch * 2,
        lx - lr, ly - lr, lr * 2, lr * 2
      );
      ctx.strokeStyle = "#facc15";
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.moveTo(lx - lr, ly); ctx.lineTo(lx + lr, ly);
      ctx.moveTo(lx, ly - lr); ctx.lineTo(lx, ly + lr);
      ctx.stroke();
      ctx.restore();

      ctx.save();
      ctx.beginPath();
      ctx.arc(lx, ly, lr, 0, Math.PI * 2);
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = "#38bdf8";
      ctx.stroke();
      ctx.fillStyle = "rgba(15,23,42,0.88)";
      ctx.fillRect(lx - 45, ly + lr - 18, 90, 16);
      ctx.fillStyle = "#f8fafc";
      ctx.font = "bold 10px monospace";
      ctx.textAlign = "center";
      ctx.fillText(Math.round(dh.u) + "," + Math.round(dh.v) + " px", lx, ly + lr - 6);
      ctx.restore();
    }
  }

  function refreshLandmarkSelect() {
    const curVal = selAddLandmark.value;
    selAddLandmark.innerHTML = "";
    const defOpt = document.createElement("option");
    defOpt.value = "";
    defOpt.textContent = "+ Add Landmark...";
    selAddLandmark.appendChild(defOpt);
    inactivePredefinedHandles().forEach((h) => {
      const opt = document.createElement("option");
      opt.value = h.id;
      opt.textContent = h.id + " — " + h.label;
      selAddLandmark.appendChild(opt);
    });
    if (curVal) selAddLandmark.value = "";
  }

  function detectLayoutChoice() {
    const actIds = new Set(activeHandles().map((h) => h.id));
    for (const key of ["4", "8", "13", "21"]) {
      const target = LAYOUT_SETS[key];
      if (target.length === actIds.size && target.every((id) => actIds.has(id))) {
        return key;
      }
    }
    return "custom";
  }

  function syncUiButtons() {
    const nAct = activeHandles().length;
    btnFlat.classList.toggle("active", fitMode === "flat");
    btnCurve.classList.toggle("active", fitMode === "curve");
    if (btnTps) btnTps.classList.toggle("active", fitMode === "tps");
    if (curveWrap) {
      curveWrap.style.display = (fitMode === "curve") ? "inline-flex" : "none";
    }
    if (curveSlider) {
      curveSlider.value = String(Math.round(lensLambda * 100));
    }
    if (curveVal) {
      curveVal.textContent = (lensLambda >= 0 ? "+" : "") + lensLambda.toFixed(2);
    }
    btnAddMode.classList.toggle("active", addPointMode);
    btnMoveAll.classList.toggle("active", moveAllActive);
    canvas.classList.toggle("add-cursor", addPointMode);

    selLayout.value = detectLayoutChoice();
    const canRemove = Boolean(selectedHandleId && nAct > 4);
    btnRemovePt.disabled = !canRemove;
    let modeLabel = "Normal Broadcast (16:9)";
    if (fitMode === "tps") modeLabel = "Panorama Multi-Cam (TPS Snap)";
    else if (fitMode === "curve") modeLabel = "Wide-Angle Lens (λ=" + (lensLambda >= 0 ? "+" : "") + lensLambda.toFixed(2) + ")";
    badgeEl.textContent = nAct + " pts • " + modeLabel + " • " + pitchL.toFixed(1) + "×" + pitchW.toFixed(1) + "m";
    refreshLandmarkSelect();
  }

  if (curveSlider) {
    curveSlider.oninput = () => {
      lensLambda = Number(curveSlider.value) / 100.0;
      if (curveVal) curveVal.textContent = (lensLambda >= 0 ? "+" : "") + lensLambda.toFixed(2);
      rebuildTransform();
      syncUiButtons();
      render();
    };
  }

  function activateHandleSet(targetIds) {
    const targetSet = new Set(targetIds);
    // Project inactive handles using user's CURRENT warp before toggling active so existing alignment never jumps!
    handles.forEach((h) => {
      if (!h.active && targetSet.has(h.id)) {
        const uv = pitchToVideo(h.xm, h.ym);
        h.u = uv[0];
        h.v = uv[1];
      }
    });
    handles.forEach((h) => {
      if (h.role !== "custom") {
        h.active = targetSet.has(h.id);
      } else {
        // Keep custom points active unless switching explicitly to 4/8/13/21? Keep user custom pins if explicit preset chosen? No, switching layout cleanly selects that layout, OR keep custom pins active? Better: cleanly activate targetSet + deactivate inactive predefined; remove custom only if explicitly reset. Wait: switching to "4 Corners" should mean 4 Corners!
        h.active = false;
      }
    });
    selectedHandleId = null;
    rebuildTransform();
    syncUiButtons();
    render();
  }

  function removeHandleById(id) {
    const nAct = activeHandles().length;
    if (nAct <= 4) {
      hintEl.textContent = "Minimum 4 active anchors required.";
      return;
    }
    const idx = handles.findIndex((h) => h.id === id);
    if (idx < 0) return;
    if (handles[idx].role === "custom") {
      handles.splice(idx, 1);
    } else {
      handles[idx].active = false;
    }
    if (selectedHandleId === id) selectedHandleId = null;
    rebuildTransform();
    syncUiButtons();
    render();
  }

  function pointerCoords(evt) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = dispW / Math.max(1, rect.width);
    const scaleY = dispH / Math.max(1, rect.height);
    return [(evt.clientX - rect.left) * scaleX, (evt.clientY - rect.top) * scaleY];
  }

  canvas.addEventListener("contextmenu", (evt) => {
    evt.preventDefault();
    const [cx, cy] = pointerCoords(evt);
    let bestId = null;
    let bestDist = 18;
    activeHandles().forEach((h) => {
      const cp = videoToCanvas(h.u, h.v);
      const d = Math.hypot(cp[0] - cx, cp[1] - cy);
      if (d < bestDist) {
        bestDist = d;
        bestId = h.id;
      }
    });
    if (bestId) removeHandleById(bestId);
  });

  canvas.addEventListener("pointerdown", (evt) => {
    if (evt.button === 2) return;
    const [cx, cy] = pointerCoords(evt);
    canvas.setPointerCapture(evt.pointerId);
    lastPointer = [cx, cy];

    if (moveAllActive || evt.shiftKey) {
      dragAllMode = true;
      canvas.classList.add("dragging");
      return;
    }

    // 1. Check active handles first so user can drag existing handles even in Add mode
    let bestIdx = -1;
    let bestDist = 18;
    activeHandles().forEach((h) => {
      const cp = videoToCanvas(h.u, h.v);
      const d = Math.hypot(cp[0] - cx, cp[1] - cy);
      if (d < bestDist) {
        bestDist = d;
        bestIdx = handles.findIndex((item) => item.id === h.id);
      }
    });

    if (bestIdx >= 0) {
      selectedHandleId = handles[bestIdx].id;
      activeDragIdx = bestIdx;
      canvas.classList.add("dragging");
      hintEl.textContent = "Dragging " + handles[bestIdx].label + " (" + Math.round(handles[bestIdx].u) + ", " + Math.round(handles[bestIdx].v) + " px)";
      syncUiButtons();
      render();
      return;
    }

    // 2. If "+ Add Point" mode is on, snap to nearest ghost landmark OR drop a new custom point
    if (addPointMode) {
      let ghostIdx = -1;
      let ghostDist = 20;
      inactivePredefinedHandles().forEach((h) => {
        const uv = pitchToVideo(h.xm, h.ym);
        const cp = videoToCanvas(uv[0], uv[1]);
        const d = Math.hypot(cp[0] - cx, cp[1] - cy);
        if (d < ghostDist) {
          ghostDist = d;
          ghostIdx = handles.findIndex((item) => item.id === h.id);
        }
      });

      const uvClick = canvasToVideo(cx, cy);
      if (ghostIdx >= 0) {
        handles[ghostIdx].active = true;
        handles[ghostIdx].u = uvClick[0];
        handles[ghostIdx].v = uvClick[1];
        selectedHandleId = handles[ghostIdx].id;
        activeDragIdx = ghostIdx;
        addPointMode = false;
        canvas.classList.add("dragging");
        hintEl.textContent = "Added landmark " + handles[ghostIdx].label + " — drag to fine-tune.";
      } else {
        const mxy = videoToPitch(uvClick[0], uvClick[1]);
        const cid = "U" + String(customCounter++);
        handles.push({
          id: cid,
          label: "Custom Pitch Point " + cid + " [" + mxy[0].toFixed(1) + "m, " + mxy[1].toFixed(1) + "m]",
          xm: Number(mxy[0].toFixed(2)),
          ym: Number(mxy[1].toFixed(2)),
          role: "custom",
          u: uvClick[0],
          v: uvClick[1],
          active: true,
        });
        selectedHandleId = cid;
        activeDragIdx = handles.length - 1;
        addPointMode = false;
        canvas.classList.add("dragging");
        hintEl.textContent = "Added custom point " + cid + " at [" + mxy[0].toFixed(1) + "m, " + mxy[1].toFixed(1) + "m].";
      }
      rebuildTransform();
      syncUiButtons();
      render();
      return;
    }

    // 3. Otherwise pan background
    panDrag = true;
    canvas.classList.add("dragging");
  });

  canvas.addEventListener("pointermove", (evt) => {
    const [cx, cy] = pointerCoords(evt);
    if (activeDragIdx >= 0) {
      const uv = canvasToVideo(cx, cy);
      handles[activeDragIdx].u = Math.max(-0.25 * vw, Math.min(1.25 * vw, uv[0]));
      handles[activeDragIdx].v = Math.max(-0.25 * vh, Math.min(1.25 * vh, uv[1]));
      rebuildTransform();
      hintEl.textContent = handles[activeDragIdx].label + ": (" + Math.round(handles[activeDragIdx].u) + ", " + Math.round(handles[activeDragIdx].v) + " px)";
      render();
    } else if (dragAllMode && lastPointer) {
      const uv0 = canvasToVideo(lastPointer[0], lastPointer[1]);
      const uv1 = canvasToVideo(cx, cy);
      const du = uv1[0] - uv0[0];
      const dv = uv1[1] - uv0[1];
      handles.forEach((h) => { h.u += du; h.v += dv; });
      lastPointer = [cx, cy];
      rebuildTransform();
      render();
    } else if (panDrag && lastPointer) {
      viewPanX += cx - lastPointer[0];
      viewPanY += cy - lastPointer[1];
      lastPointer = [cx, cy];
      clampPan();
      render();
    }
  });

  function stopDrag() {
    if (activeDragIdx >= 0 || dragAllMode || panDrag) {
      activeDragIdx = -1;
      dragAllMode = false;
      panDrag = false;
      canvas.classList.remove("dragging");
      render();
    }
  }
  canvas.addEventListener("pointerup", stopDrag);
  canvas.addEventListener("pointercancel", stopDrag);

  canvas.addEventListener("wheel", (evt) => {
    evt.preventDefault();
    const [cx, cy] = pointerCoords(evt);
    const factor = evt.deltaY < 0 ? 1.18 : (1.0 / 1.18);
    const nextZoom = Math.max(1.0, Math.min(5.0, viewZoom * factor));
    const ratio = nextZoom / viewZoom;
    viewPanX = cx - (cx - viewPanX) * ratio;
    viewPanY = cy - (cy - viewPanY) * ratio;
    viewZoom = nextZoom;
    clampPan();
    render();
  }, { passive: false });

  btnFlat.onclick = () => {
    fitMode = "flat";
    rebuildTransform();
    syncUiButtons();
    render();
  };

  btnCurve.onclick = () => {
    fitMode = "curve";
    if (Math.abs(lensLambda) < 1e-4) {
      lensLambda = -0.10;
    }
    rebuildTransform();
    syncUiButtons();
    render();
  };

  if (btnTps) {
    btnTps.onclick = () => {
      fitMode = "tps";
      rebuildTransform();
      syncUiButtons();
      render();
    };
  }

  if (inputPitchL) {
    inputPitchL.onchange = () => {
      const val = parseFloat(inputPitchL.value);
      if (!isNaN(val) && val >= 90.0 && val <= 120.0) {
        pitchL = val;
        updateLandmarkMetricPositions();
        rebuildTransform();
        syncUiButtons();
        render();
      } else {
        inputPitchL.value = String(pitchL);
      }
    };
  }

  if (inputPitchW) {
    inputPitchW.onchange = () => {
      const val = parseFloat(inputPitchW.value);
      if (!isNaN(val) && val >= 45.0 && val <= 90.0) {
        pitchW = val;
        updateLandmarkMetricPositions();
        rebuildTransform();
        syncUiButtons();
        render();
      } else {
        inputPitchW.value = String(pitchW);
      }
    };
  }

  selLayout.onchange = () => {
    const key = selLayout.value;
    if (LAYOUT_SETS[key]) {
      activateHandleSet(LAYOUT_SETS[key]);
    }
  };

  btnAddMode.onclick = () => {
    addPointMode = !addPointMode;
    if (addPointMode) {
      hintEl.textContent = "Click any faint + landmark or anywhere inside the pitch mask to drop a new anchor.";
    }
    syncUiButtons();
    render();
  };

  selAddLandmark.onchange = () => {
    const id = selAddLandmark.value;
    if (!id) return;
    const h = handles.find((item) => item.id === id);
    if (h) {
      const uv = pitchToVideo(h.xm, h.ym);
      h.u = uv[0];
      h.v = uv[1];
      h.active = true;
      selectedHandleId = h.id;
      hintEl.textContent = "Added " + h.label + " — drag it onto the video pitch.";
      rebuildTransform();
      syncUiButtons();
      render();
    }
    selAddLandmark.value = "";
  };

  btnRemovePt.onclick = () => {
    if (selectedHandleId) removeHandleById(selectedHandleId);
  };

  btnMoveAll.onclick = () => { moveAllActive = !moveAllActive; syncUiButtons(); };

  root.getElementById("pm-zoom-in").onclick = () => {
    const cx = dispW * 0.5, cy = dispH * 0.5;
    const nextZoom = Math.min(5.0, viewZoom * 1.35);
    const ratio = nextZoom / viewZoom;
    viewPanX = cx - (cx - viewPanX) * ratio;
    viewPanY = cy - (cy - viewPanY) * ratio;
    viewZoom = nextZoom;
    clampPan();
    render();
  };
  root.getElementById("pm-zoom-out").onclick = () => {
    const cx = dispW * 0.5, cy = dispH * 0.5;
    const nextZoom = Math.max(1.0, viewZoom / 1.35);
    const ratio = nextZoom / viewZoom;
    viewPanX = cx - (cx - viewPanX) * ratio;
    viewPanY = cy - (cy - viewPanY) * ratio;
    viewZoom = nextZoom;
    clampPan();
    render();
  };
  root.getElementById("pm-zoom-reset").onclick = () => { viewZoom = 1.0; viewPanX = 0.0; viewPanY = 0.0; render(); };

  opInput.oninput = () => { maskOpacity = Number(opInput.value) / 100.0; render(); };
  selColor.onchange = () => { maskColor = selColor.value; render(); };

  selPreset.onchange = () => {
    const key = selPreset.value;
    if (key === "current") {
      handles = cloneHandles(initialHandles);
    } else if (presetMap[key]) {
      handles = cloneHandles(presetMap[key]);
      if (key === "panorama") {
        fitMode = "tps";
        lensLambda = -0.10;
      } else {
        fitMode = "flat";
      }
    }
    selectedHandleId = null;
    rebuildTransform();
    syncUiButtons();
    render();
  };

  root.getElementById("pm-reset").onclick = () => {
    setTriggerValue("alignment", { ts: Date.now(), action: "reset" });
  };

  root.getElementById("pm-apply").onclick = () => {
    const act = activeHandles();
    const outHandles = act.map((h) => ({
      id: h.id,
      label: h.label,
      xm: Number(h.xm),
      ym: Number(h.ym),
      u: Number(h.u.toFixed(2)),
      v: Number(h.v.toFixed(2)),
      role: h.role,
      active: true,
    }));
    const allOut = handles.map((h) => ({
      id: h.id,
      label: h.label,
      xm: Number(h.xm),
      ym: Number(h.ym),
      u: Number(h.u.toFixed(2)),
      v: Number(h.v.toFixed(2)),
      role: h.role,
      active: Boolean(h.active),
    }));
    const warpOut = (fitMode === "tps") ? "panorama_tps" : ((fitMode === "curve") ? "universal" : "planar");
    setTriggerValue("alignment", {
      ts: Date.now(),
      action: "apply",
      warpMode: warpOut,
      lensLambda: Number(lensLambda),
      pitchLength: Number(pitchL),
      pitchWidth: Number(pitchW),
      handles: outHandles,
      allHandles: allOut,
    });
  };

  bgImg.onload = () => {
    bgLoaded = true;
    rebuildTransform();
    syncUiButtons();
    render();
  };
  bgImg.src = data.imageDataUrl;
  rebuildTransform();
  syncUiButtons();
  render();
}
"""

_PITCH_MASK_COMPONENT = st.components.v2.component(
    "tactical_pitch_mask_studio",
    html=_HTML,
    css=_CSS,
    js=_JS,
    isolate_styles=True,
)


def _preset_handles_from_transformer(
    transformer: Any,
    w: int,
    h: int,
    active_ids: Optional[List[str]] = None,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
) -> List[Dict[str, Any]]:
    """Projects the 27 canonical pitch landmarks into video pixels using a transformer."""
    act_set = set(active_ids if active_ids is not None else LAYOUT_4_IDS)
    landmarks_def = get_pitch_landmarks_def(pitch_length, pitch_width)
    metric_pts = [(float(d["xm"]), float(d["ym"])) for d in landmarks_def]
    px_pts = transformer.pitch_to_image(metric_pts, filter_behind_camera=False)
    out: List[Dict[str, Any]] = []
    for meta, px in zip(landmarks_def, px_pts):
        out.append({
            "id": meta["id"],
            "label": meta["label"],
            "xm": float(meta["xm"]),
            "ym": float(meta["ym"]),
            "role": meta["role"],
            "u": float(px[0]),
            "v": float(px[1]),
            "active": meta["id"] in act_set,
        })
    return out


def _build_pose_presets(
    w: int,
    h: int,
    current_transformer: Any,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
) -> Dict[str, List[Dict[str, Any]]]:
    """Builds starting handle poses cleanly across all 27 pitch landmarks."""
    from perception.homography import HomographyTransformer

    t_side = HomographyTransformer(w, h, camera_mode="sideline", pitch_length=pitch_length, pitch_width=pitch_width)
    t_end = HomographyTransformer(w, h, camera_mode="goal_to_goal", pitch_length=pitch_length, pitch_width=pitch_width)

    pano_corners_src = np.float32([
        [w * 0.08, h * 0.25],
        [w * 0.92, h * 0.25],
        [w * 0.97, h * 0.86],
        [w * 0.03, h * 0.86],
    ])
    pano_corners_dst = np.float32([[0.0, 0.0], [pitch_length, 0.0], [pitch_length, pitch_width], [0.0, pitch_width]])
    t_pano = HomographyTransformer(w, h, warp_mode="universal", lens_lambda=-0.10, pitch_length=pitch_length, pitch_width=pitch_width)
    t_pano.calibrate(
        pano_corners_src,
        pano_corners_dst,
        frame_width=w,
        frame_height=h,
        warp_mode="universal",
        lens_lambda=-0.10,
        pitch_length=pitch_length,
        pitch_width=pitch_width,
    )

    pano_handles = _preset_handles_from_transformer(t_pano, w, h, active_ids=LAYOUT_13_IDS, pitch_length=pitch_length, pitch_width=pitch_width)

    cur_is_wide = (w / float(max(1, h)) >= 2.2) or (getattr(current_transformer, "warp_mode", "planar") in ("universal", "panorama", "panorama_tps"))
    is_goal = getattr(current_transformer, "camera_mode", "sideline") == "goal_to_goal"
    if is_goal:
        cur_active = LAYOUT_ENDZONE_IDS
    elif cur_is_wide:
        cur_active = LAYOUT_13_IDS
    else:
        cur_active = LAYOUT_4_IDS

    return {
        "current": _preset_handles_from_transformer(current_transformer, w, h, active_ids=cur_active, pitch_length=pitch_length, pitch_width=pitch_width),
        "sideline": _preset_handles_from_transformer(t_side, w, h, active_ids=LAYOUT_8_IDS, pitch_length=pitch_length, pitch_width=pitch_width),
        "endzone": _preset_handles_from_transformer(t_end, w, h, active_ids=LAYOUT_ENDZONE_IDS, pitch_length=pitch_length, pitch_width=pitch_width),
        "panorama": pano_handles,
    }


def _merge_saved_handles(base_handles: List[Dict[str, Any]], saved_handles: Any) -> List[Dict[str, Any]]:
    """Merges arbitrary saved handle subsets and user-created custom points into base landmarks."""
    if not isinstance(saved_handles, list) or len(saved_handles) < 4:
        return base_handles

    has_active_flags = any("active" in item for item in saved_handles if isinstance(item, dict))
    by_id: Dict[str, Dict[str, Any]] = {}
    custom_items: List[Dict[str, Any]] = []
    known_ids = {item["id"] for item in base_handles}

    for item in saved_handles:
        if not isinstance(item, dict) or "id" not in item:
            continue
        hid = str(item["id"])
        if hid in known_ids:
            by_id[hid] = item
        else:
            custom_items.append({
                "id": hid,
                "label": str(item.get("label", hid)),
                "xm": float(item.get("xm", 52.5)),
                "ym": float(item.get("ym", 34.0)),
                "role": "custom",
                "u": float(item.get("u", 0.0)),
                "v": float(item.get("v", 0.0)),
                "active": bool(item.get("active", True)),
            })

    merged: List[Dict[str, Any]] = []
    for base in base_handles:
        copy_item = dict(base)
        if base["id"] in by_id:
            sv = by_id[base["id"]]
            copy_item["u"] = float(sv.get("u", copy_item["u"]))
            copy_item["v"] = float(sv.get("v", copy_item["v"]))
            copy_item["active"] = bool(sv.get("active", True)) if has_active_flags else True
        else:
            copy_item["active"] = False
        merged.append(copy_item)

    merged.extend(custom_items)
    if sum(1 for h in merged if h.get("active")) < 4:
        return base_handles
    return merged


def _encode_frame_data_url(raw_frame: np.ndarray, quality: int = 86) -> str:
    """Encodes a BGR frame cleanly into a base64 JPEG data URL for the browser canvas."""
    ok, buf = cv2.imencode(".jpg", raw_frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return ""
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def render_pitch_mask_studio(
    raw_frame: np.ndarray,
    transformer: Any,
    video_path: str = "",
    key_prefix: str = "mask_calib",
) -> Dict[str, Any]:
    """
    Renders the interactive Draggable Pitch Mask alignment editor over the video frame.
    Supports 4/8/13/21 handle presets, unlimited custom pitch points, Flat Least-Squares
    and Curved Panorama TPS modes, and per-video alignment persistence.
    """
    rf_h, rf_w = raw_frame.shape[:2]
    cur_length = float(getattr(transformer, "pitch_length", 105.0))
    cur_width = float(getattr(transformer, "pitch_width", 68.0))
    presets = _build_pose_presets(rf_w, rf_h, transformer, pitch_length=cur_length, pitch_width=cur_width)

    is_wide = rf_w / float(max(1, rf_h)) >= 2.2
    is_goal = getattr(transformer, "camera_mode", "sideline") == "goal_to_goal"
    if not getattr(transformer, "is_custom_calibrated", False):
        if is_wide:
            default_pose = presets["panorama"]
        elif is_goal:
            default_pose = presets["endzone"]
        else:
            default_pose = presets["current"]
    else:
        default_pose = presets["current"]
    active_handles = _merge_saved_handles(default_pose, st.session_state.get("tv_calib_mesh_handles"))

    n_active = sum(1 for h in active_handles if h.get("active"))
    cur_warp = getattr(transformer, "warp_mode", "planar")
    if cur_warp in ("panorama_tps", "tps") or (is_wide and n_active >= 6 and cur_warp not in ("planar", "universal")):
        force_fit = "tps"
    elif cur_warp == "universal":
        force_fit = "curve"
    elif cur_warp == "planar":
        force_fit = "flat"
    else:
        force_fit = "tps" if is_wide else "flat"

    cur_f = int(st.session_state.get("current_frame", 0))
    frame_cache_key = (video_path, cur_f, rf_w, rf_h)
    if st.session_state.get("_pm_cached_frame_key") != frame_cache_key:
        st.session_state["_pm_cached_frame_key"] = frame_cache_key
        st.session_state["_pm_cached_frame_url"] = _encode_frame_data_url(raw_frame)

    data_url = st.session_state.get("_pm_cached_frame_url", "")
    result = _PITCH_MASK_COMPONENT(
        key=f"{key_prefix}_pitch_mask_v2",
        data={
            "imageDataUrl": data_url,
            "vw": int(rf_w),
            "vh": int(rf_h),
            "cameraMode": getattr(transformer, "camera_mode", "sideline"),
            "handles": active_handles,
            "presets": presets,
            "layout4": LAYOUT_4_GOAL_IDS if is_goal else LAYOUT_4_IDS,
            "layout8": LAYOUT_8_GOAL_IDS if is_goal else LAYOUT_8_IDS,
            "layout13": LAYOUT_13_IDS,
            "layout21": LAYOUT_21_IDS,
            "warpMode": getattr(transformer, "warp_mode", "planar"),
            "lensLambda": float(getattr(transformer, "lens_lambda", -0.10 if is_wide else 0.0)),
            "forceFitMode": force_fit,
            "pitchLength": cur_length,
            "pitchWidth": cur_width,
        },
        on_alignment_change=lambda: None,
    )

    just_applied = False
    just_reset = False
    trigger_val = getattr(result, "alignment", None)
    last_ts_key = f"{key_prefix}_pm_last_ts"

    if isinstance(trigger_val, dict) and trigger_val.get("ts") and trigger_val.get("ts") != st.session_state.get(last_ts_key):
        st.session_state[last_ts_key] = trigger_val["ts"]
        action = trigger_val.get("action")
        if action == "apply":
            raw_h = [
                h for h in (trigger_val.get("handles") or [])
                if isinstance(h, dict) and "u" in h and "v" in h and "xm" in h and "ym" in h
                and h["u"] is not None and h["v"] is not None and h["xm"] is not None and h["ym"] is not None
            ]
            all_h = trigger_val.get("allHandles") or raw_h
            if len(raw_h) >= 4:
                src_pts = np.float32([[float(item["u"]), float(item["v"])] for item in raw_h])
                dst_pts = np.float32([[float(item["xm"]), float(item["ym"])] for item in raw_h])
                req_warp = str(trigger_val.get("warpMode", "auto"))
                req_lambda = float(trigger_val.get("lensLambda", 0.0))
                req_length = float(trigger_val.get("pitchLength", cur_length))
                req_width = float(trigger_val.get("pitchWidth", cur_width))
                transformer.calibrate(
                    src_pts,
                    dst_pts,
                    frame_width=rf_w,
                    frame_height=rf_h,
                    warp_mode=req_warp,
                    lens_lambda=req_lambda if req_warp == "universal" else None,
                    pitch_length=req_length,
                    pitch_width=req_width,
                )

                st.session_state["tv_calib_points"] = [(float(p[0]), float(p[1])) for p in src_pts]
                st.session_state["tv_calib_mesh_handles"] = all_h if len(all_h) >= 4 else None
                st.session_state["tv_calib_applied"] = True
                st.session_state["tv_calib_enabled"] = True
                st.session_state["tv_calib_warp_mode"] = transformer.warp_mode
                st.session_state["manual_calib_enabled"] = True
                st.session_state["manual_calib_applied"] = True
                st.session_state["manual_calib_points"] = [(float(p[0]), float(p[1])) for p in src_pts]

                if video_path:
                    from ui.media_library import save_video_calibration
                    save_video_calibration(video_path, {
                        "video_path": video_path,
                        "frame_width": int(rf_w),
                        "frame_height": int(rf_h),
                        "warp_mode": transformer.warp_mode,
                        "lens_lambda": float(getattr(transformer, "lens_lambda", req_lambda)),
                        "pitch_length": req_length,
                        "pitch_width": req_width,
                        "src_points": [[float(p[0]), float(p[1])] for p in src_pts],
                        "dst_points": [[float(p[0]), float(p[1])] for p in dst_pts],
                        "handles": all_h if len(all_h) >= 4 else None,
                    })

                if "overlay_engine" in st.session_state and st.session_state.overlay_engine is not None:
                    st.session_state.overlay_engine.transformer = transformer
                if "stream_manager" in st.session_state and st.session_state.stream_manager is not None:
                    st.session_state.stream_manager.transformer = transformer

                just_applied = True
        elif action == "reset":
            st.session_state["tv_calib_points"] = []
            st.session_state["tv_calib_mesh_handles"] = None
            st.session_state["tv_calib_applied"] = False
            st.session_state["tv_calib_enabled"] = False
            st.session_state["manual_calib_applied"] = False
            st.session_state["manual_calib_enabled"] = False
            st.session_state["manual_calib_points"] = []
            from ui.state import get_active_source_info
            active_src = st.session_state.get("active_source_key", "world_cup")
            cam_mode = getattr(transformer, "camera_mode", None) or get_active_source_info(active_src).get("default_mode", "sideline")
            transformer.set_camera_mode(cam_mode)
            if video_path:
                from ui.media_library import delete_video_calibration
                delete_video_calibration(video_path)
            st.session_state.pop("_pm_cached_frame_key", None)
            just_reset = True

    is_custom = bool(getattr(transformer, "is_custom_calibrated", False))
    mean_err = None
    if is_custom and hasattr(transformer, "compute_reprojection_residuals"):
        res_info = transformer.compute_reprojection_residuals()
        if res_info.get("residuals"):
            mean_err = float(res_info["mean_error"])

    return {
        "applied": is_custom,
        "enabled": bool(st.session_state.get("manual_calib_enabled", is_custom)),
        "just_applied": just_applied,
        "just_reset": just_reset,
        "points": st.session_state.get("tv_calib_points", []),
        "mean_error_px": mean_err,
        "warp_mode": getattr(transformer, "warp_mode", "planar"),
        "pitch_length": float(getattr(transformer, "pitch_length", cur_length)),
        "pitch_width": float(getattr(transformer, "pitch_width", cur_width)),
    }
