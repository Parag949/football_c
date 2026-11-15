const H_inv = __H_INV_JSON__;
const H = __H_JSON__;
const WARP_MODE = __WARP_MODE_JSON__;
const P2I_LUT = __P2I_LUT_JSON__;
const I2P_LUT = __I2P_LUT_JSON__;
const FPS = __FPS_VAL__;
const TOTAL_FRAMES = __TOTAL_F__;
const STRIDE = __STRIDE_VAL__;
const HAS_CACHE = __HAS_CACHE__;
const VID_W = __VW__;
const VID_H = __VH__;
const PAYLOAD_URL = __PAYLOAD_URL_JSON__;
const PAYLOAD_FILENAME = __PAYLOAD_FILENAME_JSON__;

// Precomputed tactical frames data (Inlined or stream-loaded for large clips)
let framesData = __FRAMES_JSON__;

// Frame index map for O(1) lookup and stride-aware alignment
const frameMap = new Map();
const sortedIndices = [];

function buildFrameMap() {
    frameMap.clear();
    sortedIndices.length = 0;
    if (framesData && framesData.length > 0) {
        for (let k = 0; k < framesData.length; k++) {
            const fd = framesData[k];
            if (fd && fd.i !== undefined) {
                frameMap.set(fd.i, fd);
                sortedIndices.push(fd.i);
            }
        }
        sortedIndices.sort((a, b) => a - b);
    }
}
buildFrameMap();

function loadAsyncPayload() {
    if ((!framesData || framesData.length === 0) && HAS_CACHE) {
        const hudFormations = document.getElementById('hud-formations');
        if (hudFormations) {
            hudFormations.innerText = "⏳ Loading tactical telemetry frames...";
        }
        const tryFetch = (url) => {
            if (!url) return Promise.reject(new Error("Empty URL"));
            return fetch(url).then(res => {
                if (!res.ok) throw new Error("HTTP " + res.status);
                return res.json();
            });
        };
        const primaryUrl = PAYLOAD_URL || ("/app/static/" + PAYLOAD_FILENAME);
        tryFetch(primaryUrl)
            .catch(() => tryFetch("/app/static/" + PAYLOAD_FILENAME))
            .catch(() => tryFetch("./static/" + PAYLOAD_FILENAME))
            .catch(() => tryFetch("/static/" + PAYLOAD_FILENAME))
            .then(payload => {
                if (payload && payload.frames) {
                    framesData = payload.frames;
                    buildFrameMap();
                    renderFrame(currentFrameIdx);
                    if (hudFormations) {
                        hudFormations.innerText = "";
                    }
                }
            })
            .catch(err => {
                console.warn("Async payload loading failed:", err);
                if (hudFormations) {
                    hudFormations.innerText = "⚠️ Telemetry offline";
                }
            });
    }
}
loadAsyncPayload();

function _interpPlayerList(pts0, pts1, alpha) {
    if (!pts0 || pts0.length === 0) return pts1 || [];
    if (!pts1 || pts1.length === 0) return pts0;
    if (pts0.length === pts1.length) {
        let maxStep = 0;
        const out = new Array(pts0.length);
        for (let i = 0; i < pts0.length; i++) {
            const dx = pts1[i][0] - pts0[i][0];
            const dy = pts1[i][1] - pts0[i][1];
            const d = Math.hypot(dx, dy);
            if (d > maxStep) maxStep = d;
            out[i] = [pts0[i][0] + alpha * dx, pts0[i][1] + alpha * dy];
        }
        // Only use index-paired lerp if IDs weren't reshuffled across frames
        if (maxStep < 10.0) return out;
    }
    return pts0.map(p0 => {
        let best = null;
        let bestD = 8.0;
        for (let j = 0; j < pts1.length; j++) {
            const d = Math.hypot(pts1[j][0] - p0[0], pts1[j][1] - p0[1]);
            if (d < bestD) { bestD = d; best = pts1[j]; }
        }
        return best ? [p0[0] + alpha * (best[0] - p0[0]), p0[1] + alpha * (best[1] - p0[1])] : p0;
    });
}

function getFrameBundle(targetIdx) {
    if (!framesData || framesData.length === 0) return null;
    // 1. Direct O(1) exact frame hit
    if (frameMap.has(targetIdx)) {
        return frameMap.get(targetIdx);
    }
    // 2. Direct array lookup if dense stride 1
    if (framesData[targetIdx] && framesData[targetIdx].i === targetIdx) {
        return framesData[targetIdx];
    }
    // 3. Stride alignment: interpolate between bracketing keyframes when within stride tolerance
    if (sortedIndices.length === 0) return null;
    const minF = sortedIndices[0];
    const maxF = sortedIndices[sortedIndices.length - 1];
    const effStride = Math.max(1, STRIDE || 1);
    if (targetIdx < minF || targetIdx > maxF + (effStride - 1)) {
        return null;
    }

    let low = 0, high = sortedIndices.length - 1;
    let best = 0;
    while (low <= high) {
        const mid = (low + high) >> 1;
        if (sortedIndices[mid] <= targetIdx) {
            best = mid;
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }

    const prevIdx = sortedIndices[best];
    const nextIdx = (best + 1 < sortedIndices.length) ? sortedIndices[best + 1] : prevIdx;
    const nearestIdx = (Math.abs(nextIdx - targetIdx) < Math.abs(prevIdx - targetIdx)) ? nextIdx : prevIdx;

    if (prevIdx !== nextIdx && targetIdx > prevIdx && targetIdx < nextIdx && (nextIdx - prevIdx) <= Math.max(2, effStride * 2)) {
        const f0 = frameMap.get(prevIdx);
        const f1 = frameMap.get(nextIdx);
        if (f0 && f1) {
            const alpha = (targetIdx - prevIdx) / (nextIdx - prevIdx);
            const anchor = alpha < 0.5 ? f0 : f1;
            const lerpBall = (f0.ball && f1.ball && Math.hypot(f1.ball[0] - f0.ball[0], f1.ball[1] - f0.ball[1]) < 8.0)
                ? [f0.ball[0] + alpha * (f1.ball[0] - f0.ball[0]), f0.ball[1] + alpha * (f1.ball[1] - f0.ball[1])]
                : anchor.ball;
            return {
                i: targetIdx,
                a: _interpPlayerList(f0.a, f1.a, alpha),
                b: _interpPlayerList(f0.b, f1.b, alpha),
                ball: lerpBall,
                c: anchor.c,
                hull: anchor.hull,
                vor: anchor.vor,
                dijk: anchor.dijk,
            };
        }
    }

    if (Math.abs(nearestIdx - targetIdx) <= effStride) {
        return frameMap.get(nearestIdx) || null;
    }
    return null;
}

let currentFrameIdx = 0;
let isPlaying = false;
let playbackSpeed = 1.0;

let showHull = true;
let showVoronoi = true;
let showPitchControl = false;
let showXt = true;
let showLabels = true;
let showGhost = false;
let showGraph = false;
let ghostPos = [78.0, 22.0];
let isDraggingGhost = false;

const XT_GRID_16x12 = [
    [0.006, 0.008, 0.008, 0.010, 0.011, 0.014, 0.018, 0.021, 0.025, 0.030, 0.035, 0.042, 0.051, 0.063, 0.078, 0.091],
    [0.007, 0.008, 0.009, 0.011, 0.013, 0.016, 0.020, 0.024, 0.029, 0.035, 0.042, 0.052, 0.065, 0.082, 0.108, 0.134],
    [0.008, 0.009, 0.010, 0.012, 0.014, 0.018, 0.023, 0.028, 0.035, 0.043, 0.053, 0.067, 0.086, 0.117, 0.162, 0.214],
    [0.008, 0.010, 0.011, 0.013, 0.016, 0.020, 0.026, 0.033, 0.041, 0.052, 0.065, 0.084, 0.112, 0.158, 0.231, 0.312],
    [0.009, 0.010, 0.012, 0.014, 0.017, 0.022, 0.029, 0.037, 0.047, 0.060, 0.077, 0.102, 0.138, 0.201, 0.298, 0.384],
    [0.009, 0.010, 0.012, 0.014, 0.018, 0.023, 0.031, 0.040, 0.052, 0.066, 0.086, 0.115, 0.159, 0.236, 0.342, 0.422],
    [0.009, 0.010, 0.012, 0.014, 0.018, 0.023, 0.031, 0.040, 0.052, 0.066, 0.086, 0.115, 0.159, 0.236, 0.342, 0.422],
    [0.009, 0.010, 0.012, 0.014, 0.017, 0.022, 0.029, 0.037, 0.047, 0.060, 0.077, 0.102, 0.138, 0.201, 0.298, 0.384],
    [0.008, 0.010, 0.011, 0.013, 0.016, 0.020, 0.026, 0.033, 0.041, 0.052, 0.065, 0.084, 0.112, 0.158, 0.231, 0.312],
    [0.008, 0.009, 0.010, 0.012, 0.014, 0.018, 0.023, 0.028, 0.035, 0.043, 0.053, 0.067, 0.086, 0.117, 0.162, 0.214],
    [0.007, 0.008, 0.009, 0.011, 0.013, 0.016, 0.020, 0.024, 0.029, 0.035, 0.042, 0.052, 0.065, 0.082, 0.108, 0.134],
    [0.006, 0.008, 0.008, 0.010, 0.011, 0.014, 0.018, 0.021, 0.025, 0.030, 0.035, 0.042, 0.051, 0.063, 0.078, 0.091]
];

const video = document.getElementById('tactical-video');
const overlayCanvas = document.getElementById('overlay-canvas');
const overlayCtx = overlayCanvas.getContext('2d');
const radarCanvas = document.getElementById('radar-canvas');
const radarCtx = radarCanvas.getContext('2d');
const timeline = document.getElementById('timeline');
const curTimeEl = document.getElementById('cur-time');
const totalTimeEl = document.getElementById('total-time');
const btnPlay = document.getElementById('btn-play');
const hudLatency = document.getElementById('hud-latency');
const hudFrame = document.getElementById('hud-frame');
const hudFormations = document.getElementById('hud-formations');
const hudXt = document.getElementById('hud-xt');
const hudFps = document.getElementById('hud-fps');
const studioGrid = document.getElementById('studio-grid');
const cardBroadcast = document.getElementById('card-broadcast');
const cardRadar = document.getElementById('card-radar');

function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    const fr = Math.floor((seconds % 1) * 100);
    return String(mins).padStart(2, '0') + ':' +
           String(secs).padStart(2, '0') + '.' +
           String(fr).padStart(2, '0');
}

function getGraphEdges(bundle) {
    if (bundle && bundle.dijk && bundle.dijk.edges && bundle.dijk.edges.length > 0) {
        return bundle.dijk.edges;
    }
    const fallbackEdges = [];
    const pts = (bundle && bundle.a) ? bundle.a : [];
    for (let i = 0; i < pts.length; i++) {
        for (let j = i + 1; j < pts.length; j++) {
            const p1 = pts[i];
            const p2 = pts[j];
            const d = Math.hypot(p2[0] - p1[0], p2[1] - p1[1]);
            if (d >= 2.5 && d <= 42.0) {
                fallbackEdges.push({ sp: p1, tp: p2, w: d });
            }
        }
    }
    return fallbackEdges;
}

function drawTacticalBadge(ctx, x, y, text, color, bgColor) {
    ctx.save();
    ctx.font = "bold 11px monospace";
    const tw = ctx.measureText(text).width + 16;
    const th = 20;
    const bx = x - tw / 2;
    const by = y - th;
    ctx.fillStyle = bgColor || "rgba(15, 23, 42, 0.90)";
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    if (typeof ctx.roundRect === 'function') {
        ctx.roundRect(bx, by, tw, th, 4);
    } else {
        ctx.rect(bx, by, tw, th);
    }
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = color;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(text, x, by + th / 2);
    ctx.restore();
}

function bilinearLut(gridA, gridB, nx, ny, normX, normY) {
    if (!gridA || !gridB || nx < 2 || ny < 2) return [0, 0];
    const safeX = Number.isFinite(normX) ? normX : 0.0;
    const safeY = Number.isFinite(normY) ? normY : 0.0;
    const cx = Math.max(0.0, Math.min(1.0, safeX)) * (nx - 1);
    const cy = Math.max(0.0, Math.min(1.0, safeY)) * (ny - 1);
    const x0 = Math.min(nx - 2, Math.max(0, Math.floor(cx)));
    const y0 = Math.min(ny - 2, Math.max(0, Math.floor(cy)));
    const tx = cx - x0;
    const ty = cy - y0;
    if (!gridA[y0] || !gridA[y0 + 1] || !gridB[y0] || !gridB[y0 + 1]) return [0, 0];
    const a00 = gridA[y0][x0], a10 = gridA[y0][x0 + 1], a01 = gridA[y0 + 1][x0], a11 = gridA[y0 + 1][x0 + 1];
    const b00 = gridB[y0][x0], b10 = gridB[y0][x0 + 1], b01 = gridB[y0 + 1][x0], b11 = gridB[y0 + 1][x0 + 1];
    const va = (1 - tx) * (1 - ty) * a00 + tx * (1 - ty) * a10 + (1 - tx) * ty * a01 + tx * ty * a11;
    const vb = (1 - tx) * (1 - ty) * b00 + tx * (1 - ty) * b10 + (1 - tx) * ty * b01 + tx * ty * b11;
    return [va, vb];
}

function subdivideMetricPath(pts, stepsPerEdge, closed) {
    if ((WARP_MODE !== "panorama_tps" && WARP_MODE !== "universal") || !pts || pts.length < 2 || stepsPerEdge <= 1) return pts;
    const dense = [];
    const n = pts.length;
    const edges = closed ? n : (n - 1);
    for (let i = 0; i < edges; i++) {
        const p1 = pts[i];
        const p2 = pts[(i + 1) % n];
        for (let s = 0; s < stepsPerEdge; s++) {
            const t = s / stepsPerEdge;
            dense.push([p1[0] + (p2[0] - p1[0]) * t, p1[1] + (p2[1] - p1[1]) * t]);
        }
    }
    if (!closed) dense.push(pts[n - 1]);
    return dense;
}

function projectPitchToImage(x, y) {
    if (typeof x !== 'number' || typeof y !== 'number' || !Number.isFinite(x) || !Number.isFinite(y)) return null;
    if ((WARP_MODE === "panorama_tps" || WARP_MODE === "universal") && P2I_LUT && P2I_LUT.u && P2I_LUT.v) {
        return bilinearLut(P2I_LUT.u, P2I_LUT.v, P2I_LUT.nx, P2I_LUT.ny, x / 105.0, y / 68.0);
    }
    const u_h = H_inv[0][0]*x + H_inv[0][1]*y + H_inv[0][2];
    const v_h = H_inv[1][0]*x + H_inv[1][1]*y + H_inv[1][2];
    const w_h = H_inv[2][0]*x + H_inv[2][1]*y + H_inv[2][2];
    if (w_h <= 1e-4) return null;
    const px = u_h / w_h;
    const py = v_h / w_h;
    if (!Number.isFinite(px) || !Number.isFinite(py)) return null;
    return [px, py];
}

function projectImageToPitch(u, v) {
    let px = 0.0, py = 0.0;
    if ((WARP_MODE === "panorama_tps" || WARP_MODE === "universal") && I2P_LUT && I2P_LUT.x && I2P_LUT.y) {
        const res = bilinearLut(I2P_LUT.x, I2P_LUT.y, I2P_LUT.nx, I2P_LUT.ny, u / Math.max(1, VID_W), v / Math.max(1, VID_H));
        px = res[0];
        py = res[1];
    } else {
        const x_h = H[0][0]*u + H[0][1]*v + H[0][2];
        const y_h = H[1][0]*u + H[1][1]*v + H[1][2];
        const w_h = H[2][0]*u + H[2][1]*v + H[2][2];
        if (w_h <= 1e-4) return null;
        px = x_h / w_h;
        py = y_h / w_h;
    }
    if (px < -5 || px > 110 || py < -5 || py > 73) return null;
    return [
        Math.max(0.5, Math.min(104.5, px)),
        Math.max(0.5, Math.min(67.5, py))
    ];
}

function getOffsideX(bundle) {
    if (!bundle || !bundle.b || bundle.b.length === 0) return 52.5;
    const sortedB = bundle.b.map(p => p[0]).sort((a, b) => b - a);
    const secondDeepest = sortedB.length >= 2 ? sortedB[1] : sortedB[0];
    let offX = Math.max(52.5, secondDeepest);
    if (bundle.ball && typeof bundle.ball[0] === 'number' && isFinite(bundle.ball[0])) {
        if (bundle.ball[0] > offX) {
            offX = bundle.ball[0];
        }
    }
    return offX;
}

function getGhostTelemetry(bundle) {
    const offsideX = getOffsideX(bundle);
    let isOffside = ghostPos[0] > (offsideX + 0.1);
    if (ghostPos[0] <= 52.5) {
        isOffside = false;
    } else if (bundle && bundle.ball && typeof bundle.ball[0] === 'number' && isFinite(bundle.ball[0])) {
        if (ghostPos[0] <= bundle.ball[0] + 0.1) {
            isOffside = false;
        }
    }
    const carrierIdx = (bundle && bundle.c !== undefined && bundle.c >= 0) ? bundle.c : 0;
    const carrier = (bundle && bundle.a && bundle.a[carrierIdx]) ? bundle.a[carrierIdx] : [52.5, 34.0];
    const dx = ghostPos[0] - carrier[0];
    const dy = ghostPos[1] - carrier[1];
    const dist = Math.sqrt(dx * dx + dy * dy);
    const prog = dx;
    let verdict = "Optimal Corridor";
    if (isOffside) {
        verdict = "Offside Trap";
    } else if (prog > 12.0) {
        verdict = "High-Value Run";
    } else if (prog > 0.0) {
        verdict = "Progressive Option";
    } else {
        verdict = "Decoy / Reset";
    }
    return {
        offsideX: offsideX,
        isOffside: isOffside,
        carrier: carrier,
        carrierIdx: carrierIdx,
        dist: dist,
        prog: prog,
        verdict: verdict
    };
}

function syncCanvasSize() {
    overlayCanvas.width = VID_W;
    overlayCanvas.height = VID_H;

    const rBox = document.getElementById('radar-stage-box');
    if (rBox && rBox.clientWidth > 0 && rBox.clientHeight > 0) {
        const dpr = window.devicePixelRatio || 1;
        radarCanvas.width = Math.max(300, Math.round(rBox.clientWidth * dpr));
        radarCanvas.height = Math.max(200, Math.round(rBox.clientHeight * dpr));
    } else {
        radarCanvas.width = 1050;
        radarCanvas.height = 680;
    }
}

window.addEventListener('resize', () => {
    syncCanvasSize();
    renderFrame(currentFrameIdx);
});

function renderFrame(idx) {
    const t0 = performance.now();
    const bundle = getFrameBundle(idx) || {};

    try {
        renderBroadcastAR(bundle);
    } catch (errAR) {
        console.warn("renderBroadcastAR error:", errAR);
    }

    try {
        renderRadar2D(bundle);
    } catch (errRadar) {
        console.warn("renderRadar2D error:", errRadar);
    }

    const t1 = performance.now();
    const lat = t1 - t0;
    hudLatency.innerText = lat.toFixed(2) + " ms";
    hudFrame.innerText = (idx + 1);

    if (hudFormations) {
        hudFormations.innerText = bundle && bundle.form ? "• " + bundle.form : "";
    }
    if (hudXt) {
        hudXt.innerText = bundle && bundle.xt !== undefined ? "• xT: " + bundle.xt : "";
    }

    if (bundle && bundle.vor) {
        document.getElementById('stat-territory').innerText = 
            bundle.vor.a_pct + "% - " + bundle.vor.b_pct + "%";
        document.getElementById('radar-dominance').innerText =
            "Team A: " + bundle.vor.a_pct + "% | Team B: " + bundle.vor.b_pct + "%";
    } else {
        document.getElementById('stat-territory').innerText = "--";
        document.getElementById('radar-dominance').innerText = HAS_CACHE ? "No Data for Frame" : "Analysis Not Yet Generated";
    }
    if (bundle && bundle.hull) {
        document.getElementById('stat-hull').innerText = 
            bundle.hull.area + " m²";
    } else {
        document.getElementById('stat-hull').innerText = "--";
    }
    if (bundle && (bundle.xt !== undefined || bundle.b)) {
        const xtVal = bundle.xt !== undefined ? Number(bundle.xt) : 0.052;
        const xtEl = document.getElementById('stat-xt');
        if (xtEl) {
            xtEl.innerText = xtVal.toFixed(3) + " xT";
        }
    } else {
        const xtEl = document.getElementById('stat-xt');
        if (xtEl) {
            xtEl.innerText = "--";
        }
    }
    if (bundle && bundle.c !== null && bundle.c !== undefined && bundle.c >= 0) {
        document.getElementById('stat-carrier').innerText = 
            "Player A#" + (bundle.c + 1);
    } else {
        document.getElementById('stat-carrier').innerText = "--";
    }

    const statGhost = document.getElementById('stat-ghost');
    if (statGhost) {
        if (showGhost && bundle && bundle.a && bundle.a.length > 0) {
            const gt = getGhostTelemetry(bundle);
            if (gt.isOffside) {
                statGhost.innerText = "OFFSIDE (+" + gt.prog.toFixed(1) + "m)";
                statGhost.style.color = "var(--accent-red)";
            } else {
                statGhost.innerText = (gt.prog >= 0 ? "+" : "") + gt.prog.toFixed(1) + "m (" + gt.verdict + ")";
                statGhost.style.color = "#c084fc";
            }
        } else if (showGhost) {
            statGhost.innerText = "Active (Awaiting Data)";
            statGhost.style.color = "#c084fc";
        } else {
            statGhost.innerText = "Off";
            statGhost.style.color = "var(--text-muted)";
        }
    }
}

function renderBroadcastAR(bundle) {
    overlayCtx.clearRect(0, 0, VID_W, VID_H);
    if (!bundle || !bundle.a || bundle.a.length === 0) return;

    // 1. Anti-Bleed Turf Clipping Mask (subdivided cleanly along curved panoramas)
    const pCorners = subdivideMetricPath([[0, 0], [105, 0], [105, 68], [0, 68]], 16, true);
    const imgCorners = pCorners.map(p => projectPitchToImage(p[0], p[1]));

    overlayCtx.save();
    overlayCtx.beginPath();
    let startedClip = false;
    for (let i = 0; i < imgCorners.length; i++) {
        if (imgCorners[i]) {
            if (!startedClip) {
                overlayCtx.moveTo(imgCorners[i][0], imgCorners[i][1]);
                startedClip = true;
            } else {
                overlayCtx.lineTo(imgCorners[i][0], imgCorners[i][1]);
            }
        }
    }
    if (startedClip) {
        overlayCtx.closePath();
        overlayCtx.clip();
    }

    // 2. Voronoi Dominance Partitions
    if (showVoronoi && bundle.vor) {
        overlayCtx.fillStyle = 'rgba(56, 189, 248, 0.22)';
        overlayCtx.strokeStyle = 'rgba(56, 189, 248, 0.55)';
        overlayCtx.lineWidth = 1.5;
        (bundle.vor.ca || []).forEach(rawPoly => {
            if (rawPoly.length < 3) return;
            const poly = subdivideMetricPath(rawPoly, 5, true);
            overlayCtx.beginPath();
            let first = projectPitchToImage(poly[0][0], poly[0][1]);
            if (!first) return;
            overlayCtx.moveTo(first[0], first[1]);
            for (let k = 1; k < poly.length; k++) {
                let pt = projectPitchToImage(poly[k][0], poly[k][1]);
                if (pt) overlayCtx.lineTo(pt[0], pt[1]);
            }
            overlayCtx.closePath();
            overlayCtx.fill();
            overlayCtx.stroke();
        });

        overlayCtx.fillStyle = 'rgba(244, 63, 94, 0.22)';
        overlayCtx.strokeStyle = 'rgba(244, 63, 94, 0.55)';
        (bundle.vor.cb || []).forEach(rawPoly => {
            if (rawPoly.length < 3) return;
            const poly = subdivideMetricPath(rawPoly, 5, true);
            overlayCtx.beginPath();
            let first = projectPitchToImage(poly[0][0], poly[0][1]);
            if (!first) return;
            overlayCtx.moveTo(first[0], first[1]);
            for (let k = 1; k < poly.length; k++) {
                let pt = projectPitchToImage(poly[k][0], poly[k][1]);
                if (pt) overlayCtx.lineTo(pt[0], pt[1]);
            }
            overlayCtx.closePath();
            overlayCtx.fill();
            overlayCtx.stroke();
        });
    }

    // 3. Defensive Convex Hull
    if (showHull && bundle.hull && bundle.hull.p && bundle.hull.p.length >= 3) {
        const hp = subdivideMetricPath(bundle.hull.p, 8, true);
        overlayCtx.fillStyle = 'rgba(239, 68, 68, 0.25)';
        overlayCtx.strokeStyle = 'rgba(239, 68, 68, 0.95)';
        overlayCtx.lineWidth = 3;
        overlayCtx.setLineDash([8, 6]);

        overlayCtx.beginPath();
        let first = projectPitchToImage(hp[0][0], hp[0][1]);
        if (first) {
            overlayCtx.moveTo(first[0], first[1]);
            for (let k = 1; k < hp.length; k++) {
                let pt = projectPitchToImage(hp[k][0], hp[k][1]);
                if (pt) overlayCtx.lineTo(pt[0], pt[1]);
            }
            overlayCtx.closePath();
            overlayCtx.fill();
            overlayCtx.stroke();
        }
        overlayCtx.setLineDash([]);
    }

    // 4. Passing Graph
    if (showGraph) {
        const edges = getGraphEdges(bundle);
        overlayCtx.save();
        overlayCtx.strokeStyle = 'rgba(251, 191, 36, 0.85)';
        overlayCtx.lineWidth = 2.5;
        overlayCtx.shadowColor = '#f59e0b';
        overlayCtx.shadowBlur = 8;
        overlayCtx.setLineDash([8, 5]);
        edges.forEach(e => {
            const segPts = subdivideMetricPath([e.sp, e.tp], 8, false);
            let first = projectPitchToImage(segPts[0][0], segPts[0][1]);
            if (first) {
                overlayCtx.beginPath();
                overlayCtx.moveTo(first[0], first[1]);
                for (let k = 1; k < segPts.length; k++) {
                    let pk = projectPitchToImage(segPts[k][0], segPts[k][1]);
                    if (pk) overlayCtx.lineTo(pk[0], pk[1]);
                }
                overlayCtx.stroke();
            }
        });
        overlayCtx.restore();
    }

    overlayCtx.restore();

    // 6. Player Foot Contact Rings
    (bundle.a || []).forEach((p, idx) => {
        let pt = projectPitchToImage(p[0], p[1]);
        if (!pt) return;
        overlayCtx.save();
        overlayCtx.translate(pt[0], pt[1]);
        overlayCtx.scale(1.0, 0.45);

        overlayCtx.beginPath();
        overlayCtx.arc(0, 0, 18, 0, Math.PI * 2);
        overlayCtx.fillStyle = 'rgba(56, 189, 248, 0.35)';
        overlayCtx.fill();
        overlayCtx.strokeStyle = '#38bdf8';
        overlayCtx.lineWidth = 3;
        overlayCtx.stroke();
        overlayCtx.restore();

        if (showLabels) {
            overlayCtx.fillStyle = '#ffffff';
            overlayCtx.font = 'bold 12px monospace';
            overlayCtx.textAlign = 'center';
            overlayCtx.fillText('A' + (idx + 1), pt[0], pt[1] - 12);
        }
    });

    (bundle.b || []).forEach((p, idx) => {
        let pt = projectPitchToImage(p[0], p[1]);
        if (!pt) return;
        overlayCtx.save();
        overlayCtx.translate(pt[0], pt[1]);
        overlayCtx.scale(1.0, 0.45);

        overlayCtx.beginPath();
        overlayCtx.arc(0, 0, 18, 0, Math.PI * 2);
        overlayCtx.fillStyle = 'rgba(244, 63, 94, 0.35)';
        overlayCtx.fill();
        overlayCtx.strokeStyle = '#f43f5e';
        overlayCtx.lineWidth = 3;
        overlayCtx.stroke();
        overlayCtx.restore();

        if (showLabels) {
            overlayCtx.fillStyle = '#ffffff';
            overlayCtx.font = 'bold 12px monospace';
            overlayCtx.textAlign = 'center';
            overlayCtx.fillText('B' + (idx + 1), pt[0], pt[1] - 12);
        }
    });

    if (bundle.ball) {
        let bpt = projectPitchToImage(bundle.ball[0], bundle.ball[1]);
        if (bpt) {
            overlayCtx.beginPath();
            overlayCtx.arc(bpt[0], bpt[1], 8, 0, Math.PI * 2);
            overlayCtx.fillStyle = '#facc15';
            overlayCtx.shadowColor = '#facc15';
            overlayCtx.shadowBlur = 12;
            overlayCtx.fill();
            overlayCtx.strokeStyle = '#ffffff';
            overlayCtx.lineWidth = 2;
            overlayCtx.stroke();
            overlayCtx.shadowBlur = 0;
        }
    }

    if (bundle.c !== undefined && bundle.c >= 0 && bundle.a && bundle.a[bundle.c]) {
        let cpt = projectPitchToImage(bundle.a[bundle.c][0], bundle.a[bundle.c][1]);
        if (cpt) {
            overlayCtx.beginPath();
            overlayCtx.arc(cpt[0], cpt[1] - 25, 6, 0, Math.PI * 2);
            overlayCtx.fillStyle = '#10b981';
            overlayCtx.fill();
        }
    }

    // 7. Ghost Runner Hologram (Interactive What-If Tactical Simulation)
    if (showGhost) {
        const gt = getGhostTelemetry(bundle);
        const gpt = projectPitchToImage(ghostPos[0], ghostPos[1]);
        if (gpt && gpt[0] >= -100 && gpt[0] <= VID_W + 100 && gpt[1] >= -100 && gpt[1] <= VID_H + 100) {
            const ghostColor = gt.isOffside ? '#ef4444' : '#c084fc';
            const ghostFill = gt.isOffside ? 'rgba(239, 68, 68, 0.35)' : 'rgba(168, 85, 247, 0.40)';

            // Pass corridor vector from carrier to ghost
            const cpt = projectPitchToImage(gt.carrier[0], gt.carrier[1]);
            if (cpt) {
                overlayCtx.save();
                overlayCtx.strokeStyle = ghostColor;
                overlayCtx.lineWidth = 3;
                overlayCtx.setLineDash([8, 6]);
                overlayCtx.shadowColor = ghostColor;
                overlayCtx.shadowBlur = 8;
                overlayCtx.beginPath();
                overlayCtx.moveTo(cpt[0], cpt[1]);
                overlayCtx.lineTo(gpt[0], gpt[1]);
                overlayCtx.stroke();
                overlayCtx.restore();
            }

            // Holographic base ellipse on turf
            overlayCtx.save();
            overlayCtx.translate(gpt[0], gpt[1]);
            overlayCtx.scale(1.0, 0.42);
            overlayCtx.beginPath();
            overlayCtx.arc(0, 0, 24, 0, Math.PI * 2);
            overlayCtx.fillStyle = ghostFill;
            overlayCtx.fill();
            overlayCtx.strokeStyle = ghostColor;
            overlayCtx.lineWidth = 3;
            overlayCtx.shadowColor = ghostColor;
            overlayCtx.shadowBlur = 10;
            overlayCtx.stroke();
            overlayCtx.restore();

            // Holographic vertical laser pillar
            const grad = overlayCtx.createLinearGradient(gpt[0], gpt[1], gpt[0], gpt[1] - 45);
            grad.addColorStop(0, ghostColor);
            grad.addColorStop(1, 'rgba(168, 85, 247, 0.0)');
            overlayCtx.save();
            overlayCtx.strokeStyle = grad;
            overlayCtx.lineWidth = 3;
            overlayCtx.beginPath();
            overlayCtx.moveTo(gpt[0], gpt[1]);
            overlayCtx.lineTo(gpt[0], gpt[1] - 45);
            overlayCtx.stroke();
            overlayCtx.restore();

            // Floating tactical badge
            const badgeText = gt.isOffside ? "⚠️ OFFSIDE" : ("+" + gt.prog.toFixed(1) + "m RUN");
            drawTacticalBadge(overlayCtx, gpt[0], gpt[1] - 48, badgeText, ghostColor);
        }
    }
}

function renderRadar2D(bundle) {
    const W = radarCanvas.width || 1050;
    const H = radarCanvas.height || 680;
    const dpr = window.devicePixelRatio || 1;
    const pad = Math.min(24 * dpr, Math.min(W, H) * 0.08);
    const pw = Math.max(20, W - 2 * pad);
    const ph = Math.max(20, H - 2 * pad);

    radarCtx.clearRect(0, 0, W, H);
    radarCtx.fillStyle = '#064e3b';
    radarCtx.fillRect(0, 0, W, H);

    function toRadar(x, y) {
        const px = (typeof x === 'number' && Number.isFinite(x)) ? x : 52.5;
        const py = (typeof y === 'number' && Number.isFinite(y)) ? y : 34.0;
        return [
            pad + (px / 105.0) * pw,
            pad + (py / 68.0) * ph
        ];
    }

    radarCtx.strokeStyle = 'rgba(255, 255, 255, 0.7)';
    radarCtx.lineWidth = 2 * dpr;

    radarCtx.strokeRect(pad, pad, pw, ph);

    radarCtx.beginPath();
    radarCtx.moveTo(pad + pw / 2, pad);
    radarCtx.lineTo(pad + pw / 2, pad + ph);
    radarCtx.stroke();

    const cRad = (9.15 / 105.0) * pw;
    radarCtx.beginPath();
    radarCtx.arc(pad + pw / 2, pad + ph / 2, cRad, 0, Math.PI * 2);
    radarCtx.stroke();

    const pbW = (16.5 / 105.0) * pw;
    const pbH = (40.32 / 68.0) * ph;
    const pbY = pad + (ph - pbH) / 2;
    radarCtx.strokeRect(pad, pbY, pbW, pbH);
    radarCtx.strokeRect(pad + pw - pbW, pbY, pbW, pbH);

    if (showVoronoi && bundle.vor) {
        radarCtx.save();
        radarCtx.beginPath();
        radarCtx.rect(pad, pad, pw, ph);
        radarCtx.clip();

        radarCtx.fillStyle = 'rgba(56, 189, 248, 0.28)';
        radarCtx.strokeStyle = 'rgba(56, 189, 248, 0.6)';
        radarCtx.lineWidth = 1;
        (bundle.vor.ca || []).forEach(poly => {
            if (poly.length < 3) return;
            radarCtx.beginPath();
            let p0 = toRadar(poly[0][0], poly[0][1]);
            radarCtx.moveTo(p0[0], p0[1]);
            for (let k = 1; k < poly.length; k++) {
                let pk = toRadar(poly[k][0], poly[k][1]);
                radarCtx.lineTo(pk[0], pk[1]);
            }
            radarCtx.closePath();
            radarCtx.fill();
            radarCtx.stroke();
        });

        radarCtx.fillStyle = 'rgba(244, 63, 94, 0.28)';
        radarCtx.strokeStyle = 'rgba(244, 63, 94, 0.6)';
        (bundle.vor.cb || []).forEach(poly => {
            if (poly.length < 3) return;
            radarCtx.beginPath();
            let p0 = toRadar(poly[0][0], poly[0][1]);
            radarCtx.moveTo(p0[0], p0[1]);
            for (let k = 1; k < poly.length; k++) {
                let pk = toRadar(poly[k][0], poly[k][1]);
                radarCtx.lineTo(pk[0], pk[1]);
            }
            radarCtx.closePath();
            radarCtx.fill();
            radarCtx.stroke();
        });
        radarCtx.restore();
    }

    if (showHull && bundle.hull && bundle.hull.p && bundle.hull.p.length >= 3) {
        const hp = bundle.hull.p;
        radarCtx.fillStyle = 'rgba(239, 68, 68, 0.25)';
        radarCtx.strokeStyle = 'rgba(239, 68, 68, 0.9)';
        radarCtx.lineWidth = 2 * (window.devicePixelRatio || 1);
        radarCtx.setLineDash([4, 4]);

        radarCtx.beginPath();
        let p0 = toRadar(hp[0][0], hp[0][1]);
        radarCtx.moveTo(p0[0], p0[1]);
        for (let k = 1; k < hp.length; k++) {
            let pk = toRadar(hp[k][0], hp[k][1]);
            radarCtx.lineTo(pk[0], pk[1]);
        }
        radarCtx.closePath();
        radarCtx.fill();
        radarCtx.stroke();
        radarCtx.setLineDash([]);
    }

    // Continuous Pitch Control Surface
    if (showPitchControl && bundle) {
        radarCtx.save();
        radarCtx.beginPath();
        radarCtx.rect(pad, pad, pw, ph);
        radarCtx.clip();

        (bundle.a || []).forEach(p => {
            const pt = toRadar(p[0], p[1]);
            const grad = radarCtx.createRadialGradient(pt[0], pt[1], 4, pt[0], pt[1], 48);
            grad.addColorStop(0, 'rgba(56, 189, 248, 0.38)');
            grad.addColorStop(1, 'rgba(56, 189, 248, 0.0)');
            radarCtx.fillStyle = grad;
            radarCtx.beginPath();
            radarCtx.arc(pt[0], pt[1], 48, 0, Math.PI * 2);
            radarCtx.fill();
        });

        (bundle.b || []).forEach(p => {
            const pt = toRadar(p[0], p[1]);
            const grad = radarCtx.createRadialGradient(pt[0], pt[1], 4, pt[0], pt[1], 48);
            grad.addColorStop(0, 'rgba(239, 68, 68, 0.38)');
            grad.addColorStop(1, 'rgba(239, 68, 68, 0.0)');
            radarCtx.fillStyle = grad;
            radarCtx.beginPath();
            radarCtx.arc(pt[0], pt[1], 48, 0, Math.PI * 2);
            radarCtx.fill();
        });
        radarCtx.restore();
    }

    // Expected Threat (xT) 16x12 Tactical Grid Overlay
    if (showXt) {
        radarCtx.save();
        radarCtx.beginPath();
        radarCtx.rect(pad, pad, pw, ph);
        radarCtx.clip();

        const cols = 16;
        const rows = 12;
        const cw = pw / cols;
        const ch = ph / rows;

        for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols; c++) {
                const val = (XT_GRID_16x12[r] && XT_GRID_16x12[r][c] !== undefined) ? XT_GRID_16x12[r][c] : 0.01;
                const normVal = Math.min(1.0, val / 0.38);

                let rC, gC, bC;
                if (normVal < 0.30) {
                    const t = normVal / 0.30;
                    rC = Math.round(14 + t * (245 - 14));
                    gC = Math.round(165 + t * (158 - 165));
                    bC = Math.round(233 + t * (11 - 233));
                } else {
                    const t = (normVal - 0.30) / 0.70;
                    rC = Math.round(245 + t * (239 - 245));
                    gC = Math.round(158 + t * (68 - 158));
                    bC = Math.round(11 + t * (68 - 11));
                }
                const alpha = 0.18 + normVal * 0.28;
                radarCtx.fillStyle = `rgba(${rC}, ${gC}, ${bC}, ${alpha})`;
                radarCtx.fillRect(pad + c * cw, pad + r * ch, cw, ch);

                radarCtx.strokeStyle = 'rgba(255, 255, 255, 0.07)';
                radarCtx.lineWidth = 1;
                radarCtx.strokeRect(pad + c * cw, pad + r * ch, cw, ch);
            }
        }

        // Highlight active ball carrier possession zone
        if (bundle && bundle.ball && bundle.ball.length >= 2) {
            const bx = Math.max(0, Math.min(105, bundle.ball[0]));
            const by = Math.max(0, Math.min(68, bundle.ball[1]));
            const bCol = Math.min(15, Math.floor((bx / 105.0) * cols));
            const bRow = Math.min(11, Math.floor((by / 68.0) * rows));
            radarCtx.strokeStyle = '#00f0ff';
            radarCtx.lineWidth = 2 * (window.devicePixelRatio || 1);
            radarCtx.shadowColor = '#00f0ff';
            radarCtx.shadowBlur = 6;
            radarCtx.strokeRect(pad + bCol * cw, pad + bRow * ch, cw, ch);
            radarCtx.shadowBlur = 0;
        }
        radarCtx.restore();
    }

    const pr = 7 * (window.devicePixelRatio || 1);

    (bundle.a || []).forEach((p, idx) => {
        let rpt = toRadar(p[0], p[1]);
        radarCtx.beginPath();
        radarCtx.arc(rpt[0], rpt[1], pr, 0, Math.PI * 2);
        radarCtx.fillStyle = '#38bdf8';
        radarCtx.fill();
        radarCtx.strokeStyle = '#ffffff';
        radarCtx.lineWidth = 1.5;
        radarCtx.stroke();

        if (showLabels) {
            radarCtx.fillStyle = '#ffffff';
            radarCtx.font = 'bold ' + (10 * (window.devicePixelRatio || 1)) + 'px sans-serif';
            radarCtx.textAlign = 'center';
            radarCtx.textBaseline = 'middle';
            radarCtx.fillText(String(idx + 1), rpt[0], rpt[1]);
        }
    });

    (bundle.b || []).forEach((p, idx) => {
        let rpt = toRadar(p[0], p[1]);
        radarCtx.beginPath();
        radarCtx.arc(rpt[0], rpt[1], pr, 0, Math.PI * 2);
        radarCtx.fillStyle = '#f43f5e';
        radarCtx.fill();
        radarCtx.strokeStyle = '#ffffff';
        radarCtx.lineWidth = 1.5;
        radarCtx.stroke();

        if (showLabels) {
            radarCtx.fillStyle = '#ffffff';
            radarCtx.font = 'bold ' + (10 * (window.devicePixelRatio || 1)) + 'px sans-serif';
            radarCtx.textAlign = 'center';
            radarCtx.textBaseline = 'middle';
            radarCtx.fillText(String(idx + 1), rpt[0], rpt[1]);
        }
    });

    if (bundle && bundle.ball) {
        let brpt = toRadar(bundle.ball[0], bundle.ball[1]);
        radarCtx.beginPath();
        radarCtx.arc(brpt[0], brpt[1], pr * 0.9, 0, Math.PI * 2);
        radarCtx.fillStyle = '#facc15';
        radarCtx.fill();
        radarCtx.strokeStyle = '#000000';
        radarCtx.lineWidth = 1.5;
        radarCtx.stroke();
    }

    // 7. Ghost Runner Tactical Simulation on 2D Radar
    if (showGhost) {
        const gt = getGhostTelemetry(bundle);
        const dpr = window.devicePixelRatio || 1;
        const grpt = toRadar(ghostPos[0], ghostPos[1]);
        const ghostColor = gt.isOffside ? '#ef4444' : '#c084fc';

        // Calibrated VAR Offside Line
        const offLine = toRadar(gt.offsideX, 0);
        radarCtx.save();
        radarCtx.strokeStyle = gt.isOffside ? 'rgba(239, 68, 68, 0.85)' : 'rgba(56, 189, 248, 0.75)';
        radarCtx.lineWidth = 1.5 * dpr;
        radarCtx.setLineDash([4 * dpr, 4 * dpr]);
        radarCtx.beginPath();
        radarCtx.moveTo(offLine[0], pad);
        radarCtx.lineTo(offLine[0], pad + ph);
        radarCtx.stroke();

        // Offside label
        radarCtx.fillStyle = gt.isOffside ? 'rgba(239, 68, 68, 0.9)' : 'rgba(56, 189, 248, 0.9)';
        radarCtx.font = (8 * dpr) + 'px sans-serif';
        radarCtx.textAlign = 'right';
        radarCtx.fillText('VAR Line: ' + gt.offsideX.toFixed(1) + 'm', offLine[0] - 4, pad + 12 * dpr);
        radarCtx.restore();

        // Pass corridor vector from carrier (only if real players exist)
        if (bundle && bundle.a && bundle.a.length > 0) {
            const crpt = toRadar(gt.carrier[0], gt.carrier[1]);
            radarCtx.save();
            radarCtx.strokeStyle = ghostColor;
            radarCtx.lineWidth = 2.5 * dpr;
            radarCtx.setLineDash([4 * dpr, 3 * dpr]);
            radarCtx.beginPath();
            radarCtx.moveTo(crpt[0], crpt[1]);
            radarCtx.lineTo(grpt[0], grpt[1]);
            radarCtx.stroke();
            radarCtx.restore();
        }

        // Ghost node with halo
        radarCtx.save();
        radarCtx.beginPath();
        radarCtx.arc(grpt[0], grpt[1], pr * 1.6, 0, Math.PI * 2);
        radarCtx.fillStyle = gt.isOffside ? 'rgba(239, 68, 68, 0.25)' : 'rgba(168, 85, 247, 0.35)';
        radarCtx.fill();

        radarCtx.beginPath();
        radarCtx.arc(grpt[0], grpt[1], pr * 1.1, 0, Math.PI * 2);
        radarCtx.fillStyle = ghostColor;
        radarCtx.fill();
        radarCtx.strokeStyle = '#ffffff';
        radarCtx.lineWidth = 2 * dpr;
        radarCtx.stroke();

        radarCtx.fillStyle = '#ffffff';
        radarCtx.font = 'bold ' + (9 * dpr) + 'px sans-serif';
        radarCtx.textAlign = 'center';
        radarCtx.textBaseline = 'middle';
        radarCtx.fillText('👻', grpt[0], grpt[1]);
        radarCtx.restore();
    }
}

let activeRvfcId = null;

function cancelActiveRvfc() {
    if (activeRvfcId !== null) {
        if ('cancelVideoFrameCallback' in HTMLVideoElement.prototype) {
            video.cancelVideoFrameCallback(activeRvfcId);
        } else {
            cancelAnimationFrame(activeRvfcId);
        }
        activeRvfcId = null;
    }
}

function queueNextVideoFrame() {
    cancelActiveRvfc();
    if (!video.paused) {
        if ('requestVideoFrameCallback' in HTMLVideoElement.prototype) {
            activeRvfcId = video.requestVideoFrameCallback(onVideoFrame);
        } else {
            activeRvfcId = requestAnimationFrame(() => {
                onVideoFrame(performance.now(), { mediaTime: video.currentTime });
            });
        }
    }
}

function onVideoFrame(now, metadata) {
    activeRvfcId = null;
    if (!isScrubbing && !video.paused) {
        const mediaTime = (metadata && typeof metadata.mediaTime === 'number' && Number.isFinite(metadata.mediaTime))
            ? metadata.mediaTime
            : video.currentTime;
        const targetFrame = Math.min(TOTAL_FRAMES - 1, Math.max(0, Math.floor((mediaTime + 0.001) * FPS)));
        currentFrameIdx = targetFrame;
        timeline.value = currentFrameIdx;
        curTimeEl.innerText = formatTime(mediaTime);

        try {
            renderFrame(currentFrameIdx);
        } catch (e) {
            console.error("renderFrame error:", e);
        }
    }

    if (!video.paused) {
        queueNextVideoFrame();
    }
}

function playVideo() {
    cancelActiveRvfc();
    const playPromise = video.play();
    if (playPromise !== undefined) {
        playPromise.then(() => {
            isPlaying = true;
            btnPlay.innerText = "⏸️ Pause";
            queueNextVideoFrame();
        }).catch(err => {
            console.warn("Playback requires mute policy:", err);
            video.muted = true;
            video.play().then(() => {
                isPlaying = true;
                btnPlay.innerText = "⏸️ Pause";
                queueNextVideoFrame();
            }).catch(e2 => console.error("Play failed:", e2));
        });
    }
}

function pauseVideo() {
    video.pause();
    isPlaying = false;
    btnPlay.innerText = "▶️ Play";
    cancelActiveRvfc();
}

let isScrubbing = false;
let wasPlayingBeforeScrub = false;
let scrubRafId = null;
let isExplicitStep = false;

function seekToFrame(frameIdx, forcePause = false) {
    if (forcePause) pauseVideo();
    isExplicitStep = true;
    currentFrameIdx = Math.min(TOTAL_FRAMES - 1, Math.max(0, frameIdx));
    const targetTime = currentFrameIdx / FPS;
    
    // Set video position
    if ('fastSeek' in video) {
        video.fastSeek(targetTime);
    } else {
        video.currentTime = targetTime;
    }
    timeline.value = currentFrameIdx;
    curTimeEl.innerText = formatTime(targetTime);
    renderFrame(currentFrameIdx);
}

// Global Playback & Sync Watchdogs
btnPlay.addEventListener('click', () => {
    if (video.paused) playVideo();
    else pauseVideo();
});

video.addEventListener('play', () => {
    isPlaying = true;
    btnPlay.innerText = "⏸️ Pause";
    queueNextVideoFrame();
});

video.addEventListener('playing', () => {
    isPlaying = true;
    btnPlay.innerText = "⏸️ Pause";
    queueNextVideoFrame();
});

video.addEventListener('pause', () => {
    isPlaying = false;
    btnPlay.innerText = "▶️ Play";
    cancelActiveRvfc();
});

video.addEventListener('ended', () => {
    pauseVideo();
    seekToFrame(0);
});

video.addEventListener('seeked', () => {
    if (isScrubbing) return;
    const computedIdx = Math.min(TOTAL_FRAMES - 1, Math.max(0, Math.floor((video.currentTime + 0.001) * FPS)));
    if (!isExplicitStep || Math.abs(computedIdx - currentFrameIdx) > 2) {
        currentFrameIdx = computedIdx;
    }
    isExplicitStep = false;
    timeline.value = currentFrameIdx;
    curTimeEl.innerText = formatTime(video.currentTime);
    renderFrame(currentFrameIdx);
});

video.addEventListener('timeupdate', () => {
    if (isScrubbing) return;
    if (!video.paused) {
        const computedIdx = Math.min(TOTAL_FRAMES - 1, Math.max(0, Math.floor((video.currentTime + 0.001) * FPS)));
        // If drift is detected between rVFC and currentTime, resync
        if (Math.abs(computedIdx - currentFrameIdx) >= 2) {
            currentFrameIdx = computedIdx;
            timeline.value = currentFrameIdx;
            curTimeEl.innerText = formatTime(video.currentTime);
            renderFrame(currentFrameIdx);
        }
    }
});

video.addEventListener('error', (e) => {
    console.error("Video load error:", video.error);
    const msg = video.error ? ("Code " + video.error.code) : "Load failed";
    hudFps.innerText = msg;
});

// Smooth Scrubbing Handlers
function startScrub() {
    isScrubbing = true;
    wasPlayingBeforeScrub = !video.paused;
    if (wasPlayingBeforeScrub) {
        video.pause();
    }
}

function handleScrub(val) {
    currentFrameIdx = Math.min(TOTAL_FRAMES - 1, Math.max(0, val));
    timeline.value = currentFrameIdx;
    const targetTime = currentFrameIdx / FPS;
    curTimeEl.innerText = formatTime(targetTime);
    renderFrame(currentFrameIdx);

    if (scrubRafId !== null) cancelAnimationFrame(scrubRafId);
    scrubRafId = requestAnimationFrame(() => {
        if ('fastSeek' in video) {
            video.fastSeek(targetTime);
        } else {
            video.currentTime = targetTime;
        }
        scrubRafId = null;
    });
}

function stopScrub() {
    if (!isScrubbing) return;
    isScrubbing = false;
    const targetTime = currentFrameIdx / FPS;
    video.currentTime = targetTime;
    if (wasPlayingBeforeScrub) {
        playVideo();
    }
}

timeline.addEventListener('pointerdown', startScrub);
timeline.addEventListener('mousedown', startScrub);
timeline.addEventListener('touchstart', startScrub, { passive: true });

timeline.addEventListener('input', (e) => {
    handleScrub(parseInt(e.target.value));
});

timeline.addEventListener('pointerup', stopScrub);
timeline.addEventListener('mouseup', stopScrub);
timeline.addEventListener('touchend', stopScrub);
timeline.addEventListener('change', stopScrub);

document.getElementById('btn-step-back').addEventListener('click', () => {
    seekToFrame(currentFrameIdx - 1, true);
});

document.getElementById('btn-step-fwd').addEventListener('click', () => {
    seekToFrame(currentFrameIdx + 1, true);
});

document.querySelectorAll('.speed-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
        document.querySelectorAll('.speed-btn').forEach(b => b.classList.remove('active'));
        e.target.classList.add('active');
        playbackSpeed = parseFloat(e.target.getAttribute('data-speed'));
        video.playbackRate = playbackSpeed;
        hudFps.innerText = (FPS * playbackSpeed).toFixed(2);
    });
});

const viewSide = document.getElementById('view-side');
const viewBcast = document.getElementById('view-bcast');
const viewRad = document.getElementById('view-rad');

viewSide.addEventListener('click', () => {
    viewSide.classList.add('active');
    viewBcast.classList.remove('active');
    viewRad.classList.remove('active');
    studioGrid.className = 'studio-container';
    cardBroadcast.style.display = 'flex';
    cardRadar.style.display = 'flex';
    setTimeout(() => { syncCanvasSize(); renderFrame(currentFrameIdx); }, 50);
});

viewBcast.addEventListener('click', () => {
    viewSide.classList.remove('active');
    viewBcast.classList.add('active');
    viewRad.classList.remove('active');
    studioGrid.className = 'studio-container view-broadcast';
    cardBroadcast.style.display = 'flex';
    cardRadar.style.display = 'none';
    setTimeout(() => { syncCanvasSize(); renderFrame(currentFrameIdx); }, 50);
});

viewRad.addEventListener('click', () => {
    viewSide.classList.remove('active');
    viewBcast.classList.remove('active');
    viewRad.classList.add('active');
    studioGrid.className = 'studio-container view-radar';
    cardBroadcast.style.display = 'none';
    cardRadar.style.display = 'flex';
    setTimeout(() => { syncCanvasSize(); renderFrame(currentFrameIdx); }, 50);
});

function wireToggle(id, getter, setter) {
    const el = document.getElementById(id);
    el.addEventListener('click', () => {
        const newVal = !getter();
        setter(newVal);
        if (newVal) el.classList.add('active');
        else el.classList.remove('active');
        renderFrame(currentFrameIdx);
    });
}
wireToggle('toggle-hull', () => showHull, v => showHull = v);
wireToggle('toggle-vor', () => showVoronoi, v => showVoronoi = v);
wireToggle('toggle-pitch-ctrl', () => showPitchControl, v => showPitchControl = v);
wireToggle('toggle-xt', () => showXt, v => showXt = v);
wireToggle('toggle-labels', () => showLabels, v => showLabels = v);
wireToggle('toggle-ghost', () => showGhost, v => {
    showGhost = v;
    const hint = document.getElementById('radar-ghost-hint');
    if (hint) hint.style.display = v ? 'inline' : 'none';
    const presets = document.getElementById('ghost-presets-bar');
    if (presets) presets.style.display = v ? 'inline-flex' : 'none';
    const vidBox = document.getElementById('video-stage-box');
    if (v) {
        radarCanvas.classList.add('ghost-mode');
        if (vidBox) vidBox.classList.add('ghost-mode');
    } else {
        radarCanvas.classList.remove('ghost-mode');
        if (vidBox) vidBox.classList.remove('ghost-mode');
    }
});

document.querySelectorAll('.preset-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const gx = parseFloat(btn.getAttribute('data-gx'));
        const gy = parseFloat(btn.getAttribute('data-gy'));
        ghostPos = [gx, gy];
        renderFrame(currentFrameIdx);
    });
});

function getPitchCoordsFromEvent(e) {
    const rect = radarCanvas.getBoundingClientRect();
    if (!rect || !rect.width || !rect.height) return [52.5, 34.0];
    const clientX = e.clientX - rect.left;
    const clientY = e.clientY - rect.top;
    const dpr = window.devicePixelRatio || 1;
    const canvasX = clientX * (radarCanvas.width / rect.width);
    const canvasY = clientY * (radarCanvas.height / rect.height);
    const pad = 24 * dpr;
    const pw = radarCanvas.width - 2 * pad;
    const ph = radarCanvas.height - 2 * pad;
    if (pw <= 0 || ph <= 0) return [52.5, 34.0];
    const px = ((canvasX - pad) / pw) * 105.0;
    const py = ((canvasY - pad) / ph) * 68.0;
    return [
        Math.max(0.5, Math.min(104.5, Number.isFinite(px) ? px : 52.5)),
        Math.max(0.5, Math.min(67.5, Number.isFinite(py) ? py : 34.0))
    ];
}

radarCanvas.addEventListener('mousedown', (e) => {
    if (!showGhost) return;
    const coords = getPitchCoordsFromEvent(e);
    ghostPos = coords;
    isDraggingGhost = true;
    renderFrame(currentFrameIdx);
});

window.addEventListener('mousemove', (e) => {
    if (!isDraggingGhost || !showGhost) return;
    const coords = getPitchCoordsFromEvent(e);
    ghostPos = coords;
    renderFrame(currentFrameIdx);
});

window.addEventListener('mouseup', () => {
    if (isDraggingGhost) {
        isDraggingGhost = false;
    }
});

radarCanvas.addEventListener('touchstart', (e) => {
    if (!showGhost || !e.touches || e.touches.length === 0) return;
    ghostPos = getPitchCoordsFromEvent(e.touches[0]);
    isDraggingGhost = true;
    renderFrame(currentFrameIdx);
}, { passive: true });

window.addEventListener('touchmove', (e) => {
    if (!isDraggingGhost || !showGhost || !e.touches || e.touches.length === 0) return;
    ghostPos = getPitchCoordsFromEvent(e.touches[0]);
    renderFrame(currentFrameIdx);
}, { passive: true });

window.addEventListener('touchend', () => {
    if (isDraggingGhost) isDraggingGhost = false;
});

const videoStageBox = document.getElementById('video-stage-box');
if (videoStageBox) {
    videoStageBox.addEventListener('click', (e) => {
        if (showGhost) {
            const rect = videoStageBox.getBoundingClientRect();
            const clientX = e.clientX - rect.left;
            const clientY = e.clientY - rect.top;
            const u = (clientX / rect.width) * VID_W;
            const v = (clientY / rect.height) * VID_H;
            const pitchCoords = projectImageToPitch(u, v);
            if (pitchCoords) {
                ghostPos = pitchCoords;
                renderFrame(currentFrameIdx);
            }
        } else {
            if (video.paused) playVideo();
            else pauseVideo();
        }
    });
}

window.addEventListener('keydown', (e) => {
    if (e.code === 'Space') {
        e.preventDefault();
        if (video.paused) playVideo();
        else pauseVideo();
    } else if (e.code === 'ArrowLeft') {
        e.preventDefault();
        pauseVideo();
        seekToFrame(currentFrameIdx - (e.shiftKey ? 10 : 1));
    } else if (e.code === 'ArrowRight') {
        e.preventDefault();
        pauseVideo();
        seekToFrame(currentFrameIdx + (e.shiftKey ? 10 : 1));
    }
});

// Immediate Hydration & Frame 0 Render
totalTimeEl.innerText = formatTime(TOTAL_FRAMES / FPS);
syncCanvasSize();
renderFrame(0);

video.addEventListener('loadedmetadata', () => {
    syncCanvasSize();
    renderFrame(currentFrameIdx);
});

if (window.ResizeObserver) {
    const ro = new ResizeObserver(() => {
        syncCanvasSize();
        renderFrame(currentFrameIdx);
    });
    const rBox = document.getElementById('radar-stage-box');
    if (rBox) ro.observe(rBox);
    const vBox = document.getElementById('video-stage-box');
    if (vBox) ro.observe(vBox);
}
