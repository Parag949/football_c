"""
Interactive Pitch Mask Alignment Studio (Streamlit Component v2)
================================================================
Overlays a live-warping FIFA pitch chalk mask directly over the match video frame.
Users drag 4 corner handles (flat broadcast perspective) or 13 mesh handles (wide /
curved panorama cameras) directly onto visible field lines with live zoom, pan, and
a precision magnifier loupe—then commit one unified H + TPS residual transform.
"""

import base64
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import streamlit as st

PITCH_LEN = 105.0
PITCH_WID = 68.0

# 13 well-spaced handles on the FIFA pitch [0..105m] x [0..68m]
MESH_HANDLES_DEF: List[Dict[str, Any]] = [
    {"id": "TL",   "label": "Top-Left Corner",        "xm": 0.0,    "ym": 0.0,   "role": "corner"},
    {"id": "TR",   "label": "Top-Right Corner",       "xm": 105.0,  "ym": 0.0,   "role": "corner"},
    {"id": "BR",   "label": "Bottom-Right Corner",    "xm": 105.0,  "ym": 68.0,  "role": "corner"},
    {"id": "BL",   "label": "Bottom-Left Corner",     "xm": 0.0,    "ym": 68.0,  "role": "corner"},
    {"id": "TM25", "label": "Top Left-Mid (26m)",     "xm": 26.25,  "ym": 0.0,   "role": "edge"},
    {"id": "TM50", "label": "Top Halfway (52.5m)",    "xm": 52.5,   "ym": 0.0,   "role": "edge"},
    {"id": "TM75", "label": "Top Right-Mid (79m)",    "xm": 78.75,  "ym": 0.0,   "role": "edge"},
    {"id": "BM25", "label": "Bottom Left-Mid (26m)",  "xm": 26.25,  "ym": 68.0,  "role": "edge"},
    {"id": "BM50", "label": "Bottom Halfway (52.5m)", "xm": 52.5,   "ym": 68.0,  "role": "edge"},
    {"id": "BM75", "label": "Bottom Right-Mid (79m)", "xm": 78.75,  "ym": 68.0,  "role": "edge"},
    {"id": "LP",   "label": "Left 18-Yd Top Mid",     "xm": 16.5,   "ym": 34.0,  "role": "inner"},
    {"id": "CS",   "label": "Center Spot",            "xm": 52.5,   "ym": 34.0,  "role": "inner"},
    {"id": "RP",   "label": "Right 18-Yd Top Mid",    "xm": 88.5,   "ym": 34.0,  "role": "inner"},
]

_HTML = """
<div class="pm-shell">
  <div class="pm-toolbar">
    <div class="pm-group">
      <button type="button" id="pm-mode-flat" class="pm-btn pm-btn-mode">4-Corner Flat</button>
      <button type="button" id="pm-mode-mesh" class="pm-btn pm-btn-mode">13-Point Curved Panorama</button>
    </div>
    <div class="pm-group">
      <select id="pm-preset" class="pm-select" title="Load a starting mask pose">
        <option value="current">Preset: Current Alignment</option>
        <option value="panorama">Preset: Wide / Panorama</option>
        <option value="sideline">Preset: Broadcast Sideline</option>
        <option value="endzone">Preset: Goal-to-Goal End Zone</option>
      </select>
      <button type="button" id="pm-move-all" class="pm-btn" title="Drag anywhere to reposition the entire mask at once">Move Whole Mask</button>
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
      <button type="button" id="pm-reset" class="pm-btn">Reset</button>
      <button type="button" id="pm-apply" class="pm-btn pm-btn-primary">Apply Alignment</button>
    </div>
  </div>
  <div class="pm-canvas-wrap" id="pm-wrap">
    <canvas id="pm-canvas"></canvas>
    <div class="pm-hint" id="pm-hint">
      Drag colored handles to stretch the pitch lines over the grass lines &bull; Scroll to zoom &bull; Drag background to pan &bull; Hold Shift + Drag to shift entire mask
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
  padding: 5px 10px;
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
}
.pm-btn:hover { background: #1e293b; border-color: #475569; }
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
.pm-btn-primary:hover { background: #0369a1; }
.pm-slider-lbl {
  display: flex;
  align-items: center;
  gap: 5px;
  font-size: 11px;
  color: #94a3b8;
}
.pm-slider-lbl input { width: 72px; accent-color: #38bdf8; }
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
  const btnFlat = root.getElementById("pm-mode-flat");
  const btnMesh = root.getElementById("pm-mode-mesh");
  const btnMoveAll = root.getElementById("pm-move-all");
  const selPreset = root.getElementById("pm-preset");
  const selColor = root.getElementById("pm-color");
  const opInput = root.getElementById("pm-opacity");

  const vw = Math.max(1, Number(data.vw || 1280));
  const vh = Math.max(1, Number(data.vh || 720));
  const PITCH_L = 105.0;
  const PITCH_W = 68.0;

  let meshMode = data.warpMode === "panorama_tps" ? "mesh" : (data.handles && data.handles.length > 4 && data.meshActive ? "mesh" : "flat");
  if (data.forceMeshMode === "mesh" || data.forceMeshMode === "flat") {
    meshMode = data.forceMeshMode;
  }
  let moveAllActive = false;
  let maskColor = "#00f0ff";
  let maskOpacity = 0.80;

  // Viewport zoom/pan in canvas display pixels
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
      id: h.id,
      label: h.label,
      xm: Number(h.xm),
      ym: Number(h.ym),
      role: h.role || "edge",
      u: Number(h.u),
      v: Number(h.v),
    }));
  }

  let handles = cloneHandles(data.handles);
  const initialHandles = cloneHandles(data.handles);
  const presetMap = data.presets || {};

  // ---- Linear Algebra Helpers (Exact 8x8 DLT Homography + 16x16 TPS Residual) ----
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

  function computeCornerHomography() {
    // Use 4 corner handles (TL, TR, BR, BL) to compute 3x3 H_inv: metric (xm, ym) -> video (u, v)
    const corners = handles.filter((h) => h.role === "corner");
    const pts = corners.length === 4 ? corners : handles.slice(0, 4);
    if (pts.length < 4) return null;

    const A = [];
    const b = [];
    for (let i = 0; i < 4; i++) {
      const x = pts[i].xm, y = pts[i].ym;
      const u = pts[i].u, v = pts[i].v;
      A.push([x, y, 1, 0, 0, 0, -u * x, -u * y]);
      b.push(u);
      A.push([0, 0, 0, x, y, 1, -v * x, -v * y]);
      b.push(v);
    }
    const sol = solveLinear(A, b);
    if (!sol) return null;
    return [
      [sol[0], sol[1], sol[2]],
      [sol[3], sol[4], sol[5]],
      [sol[6], sol[7], 1.0],
    ];
  }

  function applyH(Hinv, xm, ym) {
    if (!Hinv) return [vw * 0.5, vh * 0.5];
    const uh = Hinv[0][0] * xm + Hinv[0][1] * ym + Hinv[0][2];
    const vh2 = Hinv[1][0] * xm + Hinv[1][1] * ym + Hinv[1][2];
    const wh = Hinv[2][0] * xm + Hinv[2][1] * ym + Hinv[2][2];
    if (Math.abs(wh) < 1e-6) return [vw * 0.5, vh * 0.5];
    return [uh / wh, vh2 / wh];
  }

  function fitResidualTps(Hinv) {
    if (!Hinv || handles.length < 6) return null;
    const n = handles.length;
    const normPts = [];
    const resU = [];
    const resV = [];
    for (let i = 0; i < n; i++) {
      const h = handles[i];
      const base = applyH(Hinv, h.xm, h.ym);
      normPts.push([h.xm / PITCH_L, h.ym / PITCH_W]);
      resU.push((h.u - base[0]) / vw);
      resV.push((h.v - base[1]) / vh);
    }
    const m = n + 3;
    const K = Array.from({ length: m }, () => new Array(m).fill(0));
    const bu = new Array(m).fill(0);
    const bv = new Array(m).fill(0);
    const reg = 1e-4;
    for (let i = 0; i < n; i++) {
      K[i][i] = reg;
      bu[i] = resU[i];
      bv[i] = resV[i];
      for (let j = i + 1; j < n; j++) {
        const dx = normPts[i][0] - normPts[j][0];
        const dy = normPts[i][1] - normPts[j][1];
        const r2 = dx * dx + dy * dy;
        const val = r2 > 1e-12 ? 0.5 * r2 * Math.log(r2) : 0.0;
        K[i][j] = val;
        K[j][i] = val;
      }
      K[i][n] = 1.0;
      K[i][n + 1] = normPts[i][0];
      K[i][n + 2] = normPts[i][1];
      K[n][i] = 1.0;
      K[n + 1][i] = normPts[i][0];
      K[n + 2][i] = normPts[i][1];
    }
    // Slight ridge on affine tail so solver stays unconditionally non-singular
    K[n][n] = 1e-9; K[n + 1][n + 1] = 1e-9; K[n + 2][n + 2] = 1e-9;
    const wu = solveLinear(K, bu);
    const wv = solveLinear(K, bv);
    if (!wu || !wv) return null;
    return { normPts, wu, wv, n };
  }

  let cachedHinv = null;
  let cachedTps = null;

  function rebuildTransform() {
    cachedHinv = computeCornerHomography();
    if (meshMode === "mesh") {
      cachedTps = fitResidualTps(cachedHinv);
    } else {
      cachedTps = null;
      // Keep non-corner handles snapped onto flat perspective plane
      for (let i = 0; i < handles.length; i++) {
        if (handles[i].role !== "corner") {
          const p = applyH(cachedHinv, handles[i].xm, handles[i].ym);
          handles[i].u = p[0];
          handles[i].v = p[1];
        }
      }
    }
  }

  function pitchToVideo(xm, ym) {
    const base = applyH(cachedHinv, xm, ym);
    if (!cachedTps) return base;
    const nx = xm / PITCH_L;
    const ny = ym / PITCH_W;
    const { normPts, wu, wv, n } = cachedTps;
    let du = wu[n] + wu[n + 1] * nx + wu[n + 2] * ny;
    let dv = wv[n] + wv[n + 1] * nx + wv[n + 2] * ny;
    for (let i = 0; i < n; i++) {
      const dx = nx - normPts[i][0];
      const dy = ny - normPts[i][1];
      const r2 = dx * dx + dy * dy;
      if (r2 > 1e-12) {
        const k = 0.5 * r2 * Math.log(r2);
        du += wu[i] * k;
        dv += wv[i] * k;
      }
    }
    return [base[0] + du * vw, base[1] + dv * vh];
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
    // Translucent tinted field mask polygon along outer boundary
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
      ctx.drawImage(bgImg, viewPanX, viewPanY, dispW * viewZoom, dispH * viewZoom);
    }

    drawPitchFill();

    // Subtle interior guide grid (quarters + lengthwise thirds)
    const steps = meshMode === "mesh" ? 28 : 8;
    [26.25, 78.75].forEach((gx) => drawWarpedPolyline([[gx, 0], [gx, 68]], false, steps, 1.0, true));
    [17.0, 34.0, 51.0].forEach((gy) => drawWarpedPolyline([[0, gy], [105, gy]], false, steps, 1.0, true));

    // Outer pitch boundary
    drawWarpedPolyline([[0, 0], [105, 0], [105, 68], [0, 68]], true, steps, 2.5, false);
    // Halfway line
    drawWarpedPolyline([[52.5, 0], [52.5, 68]], false, 12, 2.2, false);
    // Center circle & spot
    drawWarpedCircle(52.5, 34.0, 9.15, 48, 2.0);
    // Left 18-yard & 6-yard boxes + penalty arc
    drawWarpedPolyline([[0, 13.84], [16.5, 13.84], [16.5, 54.16], [0, 54.16]], false, 10, 2.0, false);
    drawWarpedPolyline([[0, 24.84], [5.5, 24.84], [5.5, 43.16], [0, 43.16]], false, 6, 1.5, false);
    drawWarpedArc(11.0, 34.0, 9.15, -0.927, 0.927, 20, 1.8);
    // Right 18-yard & 6-yard boxes + penalty arc
    drawWarpedPolyline([[105, 13.84], [88.5, 13.84], [88.5, 54.16], [105, 54.16]], false, 10, 2.0, false);
    drawWarpedPolyline([[105, 24.84], [99.5, 24.84], [99.5, 43.16], [105, 43.16]], false, 6, 1.5, false);
    drawWarpedArc(94.0, 34.0, 9.15, Math.PI - 0.927, Math.PI + 0.927, 20, 1.8);

    // Draw control handles
    const activeHandles = meshMode === "mesh" ? handles : handles.filter((h) => h.role === "corner");
    activeHandles.forEach((h) => {
      const cp = videoToCanvas(h.u, h.v);
      const isSelected = activeDragIdx >= 0 && handles[activeDragIdx] && handles[activeDragIdx].id === h.id;
      const r = h.role === "corner" ? 8 : 6;
      const fill = h.role === "corner" ? "#facc15" : (h.role === "inner" ? "#22c55e" : "#38bdf8");

      ctx.save();
      ctx.beginPath();
      if (h.role === "corner") {
        ctx.rect(cp[0] - r, cp[1] - r, r * 2, r * 2);
      } else {
        ctx.arc(cp[0], cp[1], r, 0, Math.PI * 2);
      }
      ctx.fillStyle = fill;
      ctx.fill();
      ctx.lineWidth = isSelected ? 3 : 2;
      ctx.strokeStyle = isSelected ? "#ffffff" : "#0f172a";
      ctx.stroke();

      // Short handle label tag
      ctx.font = "bold 10px monospace";
      const lbl = h.id;
      const tw = ctx.measureText(lbl).width + 8;
      ctx.fillStyle = "rgba(15, 23, 42, 0.85)";
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
      const srcPatch = 90; // native px radius sampled from frame
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

  function syncUiButtons() {
    btnFlat.classList.toggle("active", meshMode === "flat");
    btnMesh.classList.toggle("active", meshMode === "mesh");
    btnMoveAll.classList.toggle("active", moveAllActive);
  }

  function pointerCoords(evt) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = dispW / Math.max(1, rect.width);
    const scaleY = dispH / Math.max(1, rect.height);
    return [(evt.clientX - rect.left) * scaleX, (evt.clientY - rect.top) * scaleY];
  }

  canvas.addEventListener("pointerdown", (evt) => {
    const [cx, cy] = pointerCoords(evt);
    canvas.setPointerCapture(evt.pointerId);
    lastPointer = [cx, cy];

    if (moveAllActive || evt.shiftKey) {
      dragAllMode = true;
      canvas.classList.add("dragging");
      return;
    }

    const activeSet = meshMode === "mesh" ? handles : handles.filter((h) => h.role === "corner");
    let bestIdx = -1;
    let bestDist = 18;
    activeSet.forEach((h) => {
      const cp = videoToCanvas(h.u, h.v);
      const d = Math.hypot(cp[0] - cx, cp[1] - cy);
      if (d < bestDist) {
        bestDist = d;
        bestIdx = handles.findIndex((item) => item.id === h.id);
      }
    });

    if (bestIdx >= 0) {
      activeDragIdx = bestIdx;
      canvas.classList.add("dragging");
      hintEl.textContent = "Dragging " + handles[bestIdx].label + " (" + Math.round(handles[bestIdx].u) + ", " + Math.round(handles[bestIdx].v) + " px)";
      render();
    } else {
      panDrag = true;
      canvas.classList.add("dragging");
    }
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

  function stopDrag(evt) {
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

  btnFlat.onclick = () => { meshMode = "flat"; rebuildTransform(); syncUiButtons(); render(); };
  btnMesh.onclick = () => { meshMode = "mesh"; rebuildTransform(); syncUiButtons(); render(); };
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
      if (key === "panorama") meshMode = "mesh";
    }
    rebuildTransform();
    syncUiButtons();
    render();
  };

  root.getElementById("pm-reset").onclick = () => {
    setTriggerValue("alignment", { ts: Date.now(), action: "reset" });
  };

  root.getElementById("pm-apply").onclick = () => {
    const outHandles = (meshMode === "mesh" ? handles : handles.filter((h) => h.role === "corner")).map((h) => ({
      id: h.id,
      label: h.label,
      xm: Number(h.xm),
      ym: Number(h.ym),
      u: Number(h.u.toFixed(2)),
      v: Number(h.v.toFixed(2)),
      role: h.role,
    }));
    setTriggerValue("alignment", {
      ts: Date.now(),
      action: "apply",
      warpMode: meshMode === "mesh" ? "auto" : "planar",
      handles: outHandles,
      allHandles: handles.map((h) => ({
        id: h.id, label: h.label, xm: Number(h.xm), ym: Number(h.ym),
        u: Number(h.u.toFixed(2)), v: Number(h.v.toFixed(2)), role: h.role,
      })),
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


def _preset_handles_from_transformer(transformer: Any, w: int, h: int) -> List[Dict[str, Any]]:
    """Projects the 13 mesh handles into video pixels using a transformer instance."""
    metric_pts = [(float(d["xm"]), float(d["ym"])) for d in MESH_HANDLES_DEF]
    px_pts = transformer.pitch_to_image(metric_pts, filter_behind_camera=False)
    out: List[Dict[str, Any]] = []
    for meta, px in zip(MESH_HANDLES_DEF, px_pts):
        out.append({
            "id": meta["id"],
            "label": meta["label"],
            "xm": float(meta["xm"]),
            "ym": float(meta["ym"]),
            "role": meta["role"],
            "u": float(px[0]),
            "v": float(px[1]),
        })
    return out


def _build_pose_presets(w: int, h: int, current_transformer: Any) -> Dict[str, List[Dict[str, Any]]]:
    """Builds starting handle poses purely as editor initializers (no runtime mode dependency)."""
    from perception.homography import HomographyTransformer

    t_side = HomographyTransformer(w, h, camera_mode="sideline")
    t_end = HomographyTransformer(w, h, camera_mode="goal_to_goal")

    # Wide panorama initial pose across full canvas width with gentle touchline bow
    pano_corners_src = np.float32([
        [w * 0.08, h * 0.25],
        [w * 0.92, h * 0.25],
        [w * 0.97, h * 0.86],
        [w * 0.03, h * 0.86],
    ])
    pano_corners_dst = np.float32([[0.0, 0.0], [105.0, 0.0], [105.0, 68.0], [0.0, 68.0]])
    t_pano = HomographyTransformer(w, h, src_points=pano_corners_src, dst_points=pano_corners_dst, warp_mode="planar")

    pano_handles = _preset_handles_from_transformer(t_pano, w, h)
    # Give wide panorama midpoint handles a mild realistic touchline arch as a starting hint
    for item in pano_handles:
        if item["id"] == "TM50":
            item["v"] = max(8.0, item["v"] - h * 0.035)
        elif item["id"] == "BM50":
            item["v"] = min(h - 8.0, item["v"] + h * 0.035)

    return {
        "current": _preset_handles_from_transformer(current_transformer, w, h),
        "sideline": _preset_handles_from_transformer(t_side, w, h),
        "endzone": _preset_handles_from_transformer(t_end, w, h),
        "panorama": pano_handles,
    }


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
    Handles apply & reset triggers cleanly and persists per-video alignment to disk.
    """
    rf_h, rf_w = raw_frame.shape[:2]
    presets = _build_pose_presets(rf_w, rf_h, transformer)

    saved_handles = st.session_state.get("tv_calib_mesh_handles")
    if isinstance(saved_handles, list) and len(saved_handles) == len(MESH_HANDLES_DEF):
        active_handles = saved_handles
    else:
        active_handles = presets["panorama"] if (rf_w / float(max(1, rf_h)) >= 2.2 and not getattr(transformer, "is_custom_calibrated", False)) else presets["current"]

    force_mesh = "mesh" if getattr(transformer, "warp_mode", "planar") == "panorama_tps" or (rf_w / float(max(1, rf_h)) >= 2.2) else "flat"

    # Encode JPEG frame (cache per frame shape + video path + frame index in session state to avoid re-encoding on unrelated reruns)
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
            "warpMode": getattr(transformer, "warp_mode", "planar"),
            "forceMeshMode": force_mesh,
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
            raw_h = trigger_val.get("handles") or []
            all_h = trigger_val.get("allHandles") or raw_h
            if len(raw_h) >= 4:
                src_pts = np.float32([[float(item["u"]), float(item["v"])] for item in raw_h])
                dst_pts = np.float32([[float(item["xm"]), float(item["ym"])] for item in raw_h])
                req_warp = str(trigger_val.get("warpMode", "auto"))
                transformer.calibrate(src_pts, dst_pts, frame_width=rf_w, frame_height=rf_h, warp_mode=req_warp)

                st.session_state["tv_calib_points"] = [(float(p[0]), float(p[1])) for p in src_pts]
                st.session_state["tv_calib_mesh_handles"] = all_h if len(all_h) == len(MESH_HANDLES_DEF) else None
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
                        "src_points": [[float(p[0]), float(p[1])] for p in src_pts],
                        "dst_points": [[float(p[0]), float(p[1])] for p in dst_pts],
                        "handles": all_h if len(all_h) == len(MESH_HANDLES_DEF) else None,
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
