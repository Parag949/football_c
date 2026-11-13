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

# 27 canonical FIFA pitch landmarks [0..105m] x [0..68m] + named layout sets
PITCH_LANDMARKS_DEF: List[Dict[str, Any]] = [
    # 4 Corners
    {"id": "TL",    "label": "Top-Left Corner",             "xm": 0.0,    "ym": 0.0,   "role": "corner"},
    {"id": "TR",    "label": "Top-Right Corner",            "xm": 105.0,  "ym": 0.0,   "role": "corner"},
    {"id": "BR",    "label": "Bottom-Right Corner",         "xm": 105.0,  "ym": 68.0,  "role": "corner"},
    {"id": "BL",    "label": "Bottom-Left Corner",          "xm": 0.0,    "ym": 68.0,  "role": "corner"},
    # 13-Point Curved Panorama / Touchline + Spine Anchors
    {"id": "TM25",  "label": "Top Quarter-Left (26.25m)",   "xm": 26.25,  "ym": 0.0,   "role": "edge"},
    {"id": "TM50",  "label": "Top Halfway (52.5m)",         "xm": 52.5,   "ym": 0.0,   "role": "edge"},
    {"id": "TM75",  "label": "Top Quarter-Right (78.75m)",  "xm": 78.75,  "ym": 0.0,   "role": "edge"},
    {"id": "BM25",  "label": "Bottom Quarter-Left (26.25m)","xm": 26.25,  "ym": 68.0,  "role": "edge"},
    {"id": "BM50",  "label": "Bottom Halfway (52.5m)",      "xm": 52.5,   "ym": 68.0,  "role": "edge"},
    {"id": "BM75",  "label": "Bottom Quarter-Right (78.75m)","xm": 78.75, "ym": 68.0,  "role": "edge"},
    {"id": "LP",    "label": "Left 18-Yd Center (16.5m)",   "xm": 16.5,   "ym": 34.0,  "role": "inner"},
    {"id": "CS",    "label": "Center Spot (52.5m, 34m)",    "xm": 52.5,   "ym": 34.0,  "role": "inner"},
    {"id": "RP",    "label": "Right 18-Yd Center (88.5m)",  "xm": 88.5,   "ym": 34.0,  "role": "inner"},
    # Penalty Boxes, Center Circle & 6-Yard Boxes (14 additional selectable landmarks)
    {"id": "L18T",  "label": "Left 18-Yd Box Top Corner",   "xm": 16.5,   "ym": 13.84, "role": "inner"},
    {"id": "L18B",  "label": "Left 18-Yd Box Bottom Corner","xm": 16.5,   "ym": 54.16, "role": "inner"},
    {"id": "R18T",  "label": "Right 18-Yd Box Top Corner",  "xm": 88.5,   "ym": 13.84, "role": "inner"},
    {"id": "R18B",  "label": "Right 18-Yd Box Bottom Corner","xm": 88.5,  "ym": 54.16, "role": "inner"},
    {"id": "CCT",   "label": "Center Circle Top (52.5m)",   "xm": 52.5,   "ym": 24.85, "role": "inner"},
    {"id": "CCB",   "label": "Center Circle Bottom (52.5m)","xm": 52.5,   "ym": 43.15, "role": "inner"},
    {"id": "L6T",   "label": "Left 6-Yd Box Top Corner",    "xm": 5.5,    "ym": 24.84, "role": "inner"},
    {"id": "R6T",   "label": "Right 6-Yd Box Top Corner",   "xm": 99.5,   "ym": 24.84, "role": "inner"},
    {"id": "L6B",   "label": "Left 6-Yd Box Bottom Corner", "xm": 5.5,    "ym": 43.16, "role": "inner"},
    {"id": "R6B",   "label": "Right 6-Yd Box Bottom Corner","xm": 99.5,   "ym": 43.16, "role": "inner"},
    {"id": "L18GT", "label": "Left Goal-Line 18-Yd Top",    "xm": 0.0,    "ym": 13.84, "role": "edge"},
    {"id": "L18GB", "label": "Left Goal-Line 18-Yd Bottom", "xm": 0.0,    "ym": 54.16, "role": "edge"},
    {"id": "R18GT", "label": "Right Goal-Line 18-Yd Top",   "xm": 105.0,  "ym": 13.84, "role": "edge"},
    {"id": "R18GB", "label": "Right Goal-Line 18-Yd Bottom","xm": 105.0,  "ym": 54.16, "role": "edge"},
]

# Backward-compatible alias for any callers referencing the 13 core handles
MESH_HANDLES_DEF = PITCH_LANDMARKS_DEF[:13]

LAYOUT_4_IDS = ["TL", "TR", "BR", "BL"]
LAYOUT_8_IDS = ["TL", "TR", "BR", "BL", "TM50", "BM50", "LP", "RP"]
LAYOUT_13_IDS = [item["id"] for item in PITCH_LANDMARKS_DEF[:13]]
LAYOUT_21_IDS = [item["id"] for item in PITCH_LANDMARKS_DEF[:21]]

_HTML = """
<div class="pm-shell">
  <div class="pm-toolbar">
    <div class="pm-group">
      <button type="button" id="pm-fit-flat" class="pm-btn pm-btn-mode" title="Least-squares flat perspective across all active points">Flat Perspective</button>
      <button type="button" id="pm-fit-curve" class="pm-btn pm-btn-mode" title="Universal physical camera optical lens curvature">Optical Lens Curve</button>
      <label class="pm-slider-lbl" id="pm-curve-wrap" style="display:none;">Lens Curvature <input type="range" id="pm-curve-slider" min="-30" max="10" value="-10" step="1" title="Adjust optical barrel distortion" /><span id="pm-curve-val" style="font-family:monospace; margin-left:4px;">-0.10</span></label>
      <select id="pm-layout" class="pm-select" title="Switch number of anchor points">
        <option value="4">4 Corners (Basic)</option>
        <option value="8">8 Points (Lines &amp; Boxes)</option>
        <option value="13">13 Points (Wide Field)</option>
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
  const curveWrap = root.getElementById("pm-curve-wrap");
  const curveSlider = root.getElementById("pm-curve-slider");
  const curveVal = root.getElementById("pm-curve-val");
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
  const PITCH_L = 105.0;
  const PITCH_W = 68.0;

  const LAYOUT_SETS = {
    "4": data.layout4 || ["TL", "TR", "BR", "BL"],
    "8": data.layout8 || ["TL", "TR", "BR", "BL", "TM50", "BM50", "LP", "RP"],
    "13": data.layout13 || [],
    "21": data.layout21 || [],
  };

  let lensLambda = Number(data.lensLambda !== undefined ? data.lensLambda : (data.warpMode === "universal" ? -0.10 : -0.08));
  let fitMode = (data.forceFitMode === "curve" || data.forceFitMode === "flat")
    ? data.forceFitMode
    : (data.warpMode === "universal" || data.warpMode === "panorama_tps" ? "curve" : "flat");
  let moveAllActive = false;
  let addPointMode = false;
  let selectedHandleId = null;
  let customCounter = 1;
  let maskColor = "#00f0ff";
  let maskOpacity = 0.80;

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
      const x = pts[k].xm / PITCH_L;
      const y = pts[k].ym / PITCH_W;
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
      [(vw * sol[0]) / PITCH_L, (vw * sol[1]) / PITCH_W, vw * sol[2]],
      [(vh * sol[3]) / PITCH_L, (vh * sol[4]) / PITCH_W, vh * sol[5]],
      [sol[6] / PITCH_L, sol[7] / PITCH_W, 1.0],
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

  function rebuildTransform() {
    const pts = activeHandles();
    cachedHinv = computeHomographyFromHandles(pts);
    cachedHfwd = cachedHinv ? invert3x3(cachedHinv) : null;
  }

  function pitchToVideo(xm, ym) {
    const base = applyH(cachedHinv, xm, ym);
    if (fitMode === "curve" && Math.abs(lensLambda) > 1e-6) {
      return distortPixel(base[0], base[1], lensLambda);
    }
    return base;
  }

  function videoToPitch(u, v) {
    if (!cachedHfwd) return [PITCH_L * 0.5, PITCH_W * 0.5];
    let uu = u, vu = v;
    if (fitMode === "curve" && Math.abs(lensLambda) > 1e-6) {
      const und = undistortPixel(u, v, lensLambda);
      uu = und[0];
      vu = und[1];
    }
    const pt = applyH(cachedHfwd, uu, vu);
    return [
      Math.max(-2.0, Math.min(PITCH_L + 2.0, pt[0])),
      Math.max(-2.0, Math.min(PITCH_W + 2.0, pt[1])),
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
    const ring = [];
    const steps = 20;
    for (let s = 0; s < steps; s++) ring.push([(s / steps) * 105.0, 0.0]);
    for (let s = 0; s < steps; s++) ring.push([105.0, (s / steps) * 68.0]);
    for (let s = 0; s < steps; s++) ring.push([105.0 - (s / steps) * 105.0, 68.0]);
    for (let s = 0; s < steps; s++) ring.push([0.0, 68.0 - (s / steps) * 68.0]);
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

    // Outer pitch boundary (subdivided heavily so curved panoramas flex smoothly)
    drawWarpedPolyline([[0, 0], [105, 0], [105, 68], [0, 68]], true, 24, 2.4, false);
    // Halfway line + center circle + center spot
    drawWarpedPolyline([[52.5, 0], [52.5, 68]], false, 14, 2.0, false);
    drawWarpedCircle(52.5, 34.0, 9.15, 40, 1.8);
    // Left 18-yard & 6-yard boxes + penalty arc
    drawWarpedPolyline([[0, 13.84], [16.5, 13.84], [16.5, 54.16], [0, 54.16]], false, 10, 2.0, false);
    drawWarpedPolyline([[0, 24.84], [5.5, 24.84], [5.5, 43.16], [0, 43.16]], false, 6, 1.5, false);
    drawWarpedArc(11.0, 34.0, 9.15, -0.927, 0.927, 20, 1.8);
    // Right 18-yard & 6-yard boxes + penalty arc
    drawWarpedPolyline([[105, 13.84], [88.5, 13.84], [88.5, 54.16], [105, 54.16]], false, 10, 2.0, false);
    drawWarpedPolyline([[105, 24.84], [99.5, 24.84], [99.5, 43.16], [105, 43.16]], false, 6, 1.5, false);
    drawWarpedArc(94.0, 34.0, 9.15, Math.PI - 0.927, Math.PI + 0.927, 20, 1.8);

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
    badgeEl.textContent = nAct + " pts • " + (fitMode === "curve" ? "Optical Curve (λ=" + (lensLambda >= 0 ? "+" : "") + lensLambda.toFixed(2) + ")" : "Flat");
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
      if (key === "panorama") fitMode = "curve";
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
    setTriggerValue("alignment", {
      ts: Date.now(),
      action: "apply",
      warpMode: (fitMode === "curve") ? "universal" : "planar",
      lensLambda: Number(lensLambda),
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
) -> List[Dict[str, Any]]:
    """Projects the 27 canonical pitch landmarks into video pixels using a transformer."""
    act_set = set(active_ids if active_ids is not None else LAYOUT_4_IDS)
    metric_pts = [(float(d["xm"]), float(d["ym"])) for d in PITCH_LANDMARKS_DEF]
    px_pts = transformer.pitch_to_image(metric_pts, filter_behind_camera=False)
    out: List[Dict[str, Any]] = []
    for meta, px in zip(PITCH_LANDMARKS_DEF, px_pts):
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


def _build_pose_presets(w: int, h: int, current_transformer: Any) -> Dict[str, List[Dict[str, Any]]]:
    """Builds starting handle poses cleanly across all 27 pitch landmarks."""
    from perception.homography import HomographyTransformer

    t_side = HomographyTransformer(w, h, camera_mode="sideline")
    t_end = HomographyTransformer(w, h, camera_mode="goal_to_goal")

    pano_corners_src = np.float32([
        [w * 0.08, h * 0.25],
        [w * 0.92, h * 0.25],
        [w * 0.97, h * 0.86],
        [w * 0.03, h * 0.86],
    ])
    pano_corners_dst = np.float32([[0.0, 0.0], [105.0, 0.0], [105.0, 68.0], [0.0, 68.0]])
    t_pano = HomographyTransformer(w, h, src_points=pano_corners_src, dst_points=pano_corners_dst, warp_mode="planar")

    pano_handles = _preset_handles_from_transformer(t_pano, w, h, active_ids=LAYOUT_13_IDS)
    for item in pano_handles:
        if item["id"] == "TM50":
            item["v"] = max(8.0, item["v"] - h * 0.035)
        elif item["id"] == "BM50":
            item["v"] = min(h - 8.0, item["v"] + h * 0.035)

    cur_is_wide = (w / float(max(1, h)) >= 2.2) or (getattr(current_transformer, "warp_mode", "planar") == "panorama_tps")
    cur_active = LAYOUT_13_IDS if cur_is_wide else LAYOUT_4_IDS

    return {
        "current": _preset_handles_from_transformer(current_transformer, w, h, active_ids=cur_active),
        "sideline": _preset_handles_from_transformer(t_side, w, h, active_ids=LAYOUT_4_IDS),
        "endzone": _preset_handles_from_transformer(t_end, w, h, active_ids=LAYOUT_4_IDS),
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
    presets = _build_pose_presets(rf_w, rf_h, transformer)

    is_wide = rf_w / float(max(1, rf_h)) >= 2.2
    default_pose = presets["panorama"] if (is_wide and not getattr(transformer, "is_custom_calibrated", False)) else presets["current"]
    active_handles = _merge_saved_handles(default_pose, st.session_state.get("tv_calib_mesh_handles"))

    n_active = sum(1 for h in active_handles if h.get("active"))
    force_fit = "curve" if (getattr(transformer, "warp_mode", "planar") in ("universal", "panorama", "panorama_tps") or (is_wide and n_active >= 6)) else "flat"

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
            "handles": active_handles,
            "presets": presets,
            "layout4": LAYOUT_4_IDS,
            "layout8": LAYOUT_8_IDS,
            "layout13": LAYOUT_13_IDS,
            "layout21": LAYOUT_21_IDS,
            "warpMode": getattr(transformer, "warp_mode", "planar"),
            "lensLambda": float(getattr(transformer, "lens_lambda", -0.10 if is_wide else 0.0)),
            "forceFitMode": force_fit,
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
                transformer.calibrate(
                    src_pts,
                    dst_pts,
                    frame_width=rf_w,
                    frame_height=rf_h,
                    warp_mode=req_warp,
                    lens_lambda=req_lambda if req_warp == "universal" else None,
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
                        "src_points": [[float(p[0]), float(p[1])] for p in src_pts],
                        "dst_points": [[float(p[0]), float(p[1])] for p in dst_pts],
                        "handles": all_h if len(all_h) >= 4 else None,
                    })
                just_applied = True
        elif action == "reset":
            st.session_state["tv_calib_points"] = []
            st.session_state["tv_calib_mesh_handles"] = None
            st.session_state["tv_calib_applied"] = False
            st.session_state["tv_calib_enabled"] = False
            st.session_state["manual_calib_applied"] = False
            st.session_state["manual_calib_enabled"] = False
            st.session_state["manual_calib_points"] = []
            transformer.set_camera_mode("sideline")
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
    }
