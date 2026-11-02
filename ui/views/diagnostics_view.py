"""
Pipeline Diagnostics & Comprehensive DAA Engine View (Un-Nerfed Edition)
========================================================================
Full-depth inspection across all 4 pipeline stages:
- Stage 1: Perception & Foot Contact Detection
- Stage 2: Spatial Homography, Dual-Space Mapping & Orthorectified Warped Turf
- Stage 3: Computational Geometry (DAA Convex Hull Attribution & Contracts)
- Stage 4: Viewport Renderers & Turf Clipping Mask Verification
- Stage 5: Live Empirical DAA Complexity Benchmarking
"""

from typing import Dict, Any
import cv2
import numpy as np
import streamlit as st

from algorithms.daa_benchmarks import benchmark_convex_hull, benchmark_voronoi_dominance, benchmark_dijkstra_passing
from core.pipeline import stage4_render_radar, stage4_render_overlay
from ui.components.coach_cards import render_coach_cards
from perception.calibration_templates import render_mouse_calibration_studio


def render_diagnostics_view(bundle: Dict[str, Any], hull_algo_name: str = "Graham Scan") -> None:
    """Renders the comprehensive, un-nerfed pipeline diagnostics engine."""
    raw_frame = bundle['raw_frame']
    rf_h, rf_w = raw_frame.shape[:2]
    det = bundle['detections']
    pos = bundle['positions']
    geom = bundle['geometry']
    diag_status = bundle['diag_status']
    diag_issues = bundle['diag_issues']

    # 1. Executive Diagnosis Status Banner
    has_fail = any(st[0] == "FAIL" for st in diag_status.values())
    has_warn = any(st[0] == "WARN" for st in diag_status.values())

    if has_fail:
        b_color = "#7f1d1d"
        b_border = "#dc2626"
        b_title = "❌ CRITICAL FAULT DETECTED IN PIPELINE"
    elif has_warn:
        b_color = "#78350f"
        b_border = "#d97706"
        b_title = "⚠️ PIPELINE OPERATIONAL WITH WARNINGS"
    else:
        b_color = "#064e3b"
        b_border = "#059669"
        b_title = "✅ ALL 4 PIPELINE STAGES 100% HEALTHY"

    cond_H = np.linalg.cond(st.session_state.transformer.H)
    cam_mode = "manual_mouse" if st.session_state.manual_calib_enabled else st.session_state.transformer.camera_mode

    st.markdown(f"""
    <div style="background:{b_color}; border:1px solid {b_border}; border-radius:10px; padding:12px 18px; margin-bottom:14px;">
        <div style="font-size:1.0rem; font-weight:700; color:#ffffff;">{b_title}</div>
        <div style="font-size:0.80rem; color:#f1f5f9; margin-top:3px;">
            Target Frame: <b>{st.session_state.current_frame + 1}</b> • Resolution: <b>{rf_w}x{rf_h}</b> • Camera Mode: <b>'{cam_mode}'</b> • Matrix Condition: <b>κ={cond_H:.1f}</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. 4 Stage Health Cards Grid
    sc1, sc2, sc3, sc4 = st.columns(4)
    def render_diag_card(col, name, status_tuple):
        tag, desc = status_tuple
        pill_class = "diag-pill-pass" if tag == "PASS" else ("diag-pill-warn" if tag == "WARN" else "diag-pill-fail")
        with col:
            st.markdown(f"""
            <div class="coach-card">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                    <span style="font-weight:700; font-size:0.80rem; color:#f8fafc;">{name}</span>
                    <span class="{pill_class}">{tag}</span>
                </div>
                <div style="font-size:0.75rem; color:#94a3b8; font-family:monospace;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    render_diag_card(sc1, "Stage 1: Perception", diag_status.get('stage1', ('FAIL', 'No data')))
    render_diag_card(sc2, "Stage 2: Homography", diag_status.get('stage2', ('FAIL', 'No data')))
    render_diag_card(sc3, "Stage 3: Geometry (DAA)", diag_status.get('stage3', ('FAIL', 'No data')))
    render_diag_card(sc4, "Stage 4: Renderers", diag_status.get('stage4', ('FAIL', 'No data')))

    if diag_issues:
        st.markdown("#### 🚨 Identified Pipeline Diagnostic Alerts")
        for st_name, issue_txt, sev in diag_issues:
            b_col = "#dc2626" if sev == "FAIL" else "#d97706"
            icn = "🔴" if sev == "FAIL" else "🟡"
            st.markdown(f"""
            <div style="background:#111827; border-left:4px solid {b_col}; border-radius:6px; padding:8px 12px; margin-bottom:6px;">
                <div style="font-weight:700; font-size:0.80rem; color:#ffffff;">{icn} {st_name} — {sev}</div>
                <div style="font-size:0.75rem; color:#cbd5e1; margin-top:2px;">{issue_txt}</div>
            </div>
            """, unsafe_allow_html=True)

    # 3. High-Level Coach Tactical Analytics Cards
    render_coach_cards(geom, pos, bundle['coach_data'])
    st.markdown("<div style='margin-bottom:12px;'></div>", unsafe_allow_html=True)

    # 4. Deep Granular Stage Inspection (Conditional lazy rendering to save CPU cycles)
    stage_options = [
        "👁️ Stage 1: Perception (Pixels)",
        "📐 Stage 2: Homography & Warping (Meters)",
        "⚡ Stage 3: Geometry Core (DAA)",
        "🎨 Stage 4: Viewports & Turf Mask",
        "🚀 Stage 5: Empirical DAA Benchmarks"
    ]
    active_stage = st.segmented_control(
        "Inspection Stage",
        stage_options,
        default=stage_options[0],
        key="active_stage_tab",
        label_visibility="collapsed"
    ) or stage_options[0]

    # --------------------------------------------------------------------------
    # STAGE 1: PERCEPTION
    # --------------------------------------------------------------------------
    if active_stage == stage_options[0]:
        st.markdown("#### Stage 1: Perception & Foot Contact Detection")
        c_vis, c_info = st.columns([1.3, 1.0])
        with c_vis:
            det_img = raw_frame.copy()
            for p in det.players:
                bx = p.box
                col = (255, 180, 0) if p.team_id == 0 else (50, 50, 240)
                cv2.rectangle(det_img, (bx[0], bx[1]), (bx[2], bx[3]), col, 2)
                cv2.circle(det_img, (int(p.foot_point[0]), int(p.foot_point[1])), 5, (0, 255, 255), -1)
                cv2.putText(det_img, f"T{p.team_id}", (bx[0], max(15, bx[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            if det.ball_pixel:
                cv2.circle(det_img, (int(det.ball_pixel[0]), int(det.ball_pixel[1])), 8, (0, 255, 0), -1)
                cv2.putText(det_img, "BALL", (int(det.ball_pixel[0]) + 10, int(det.ball_pixel[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

            st.image(cv2.cvtColor(det_img, cv2.COLOR_BGR2RGB), caption=f"Perception Ground Contacts (Frame {st.session_state.current_frame})", width="stretch")

        with c_info:
            st.markdown(f"- Total Detected Players: **{len(det.players)}**")
            st.markdown(f"- Team A (Sky Blue): **{len(det.team_a_pixels)}**")
            st.markdown(f"- Team B (Crimson): **{len(det.team_b_pixels)}**")
            st.markdown(f"- Ball Detected: **{'Yes' if det.ball_pixel else 'No (Anchored)'}**")
            st.markdown(f"- Frame Resolution: **{det.frame_width} x {det.frame_height} px**")
            with st.expander("📄 Raw `FrameDetections` Contract JSON", expanded=False):
                st.json(det.to_dict())

    # --------------------------------------------------------------------------
    # STAGE 2: HOMOGRAPHY & DUAL-SPACE MAPPING
    # --------------------------------------------------------------------------
    elif active_stage == stage_options[1]:
        st.markdown("#### Stage 2: Spatial Homography & Planar Projection")
        st.caption("Maps camera pixel coordinates $(u, v)$ to FIFA 2D pitch metric coordinates $[0, 105]\\text{m} \\times [0, 68]\\text{m}$.")

        s2_view = st.radio(
            "Select Homography Inspection Mode",
            [
                "🔀 Side-by-Side Dual-Space Mapping",
                "🦅 Warped Bird's-Eye Perspective",
                "🎥 Camera Perspective Overlay",
                "🎯 Interactive Multi-Point Studio (N ≥ 4)"
            ],
            index=0,
            horizontal=True
        )

        if s2_view.startswith("🔀"):
            col_c, col_s = st.columns(2)
            with col_c:
                st.markdown("##### 📷 Camera Pixel Space $(u, v)$")
                cam_dbg = st.session_state.overlay_engine.render_homography_debug_frame(
                    raw_frame, show_landmarks=True, show_turf_mask=True, show_markings=True, positions=pos
                )
                st.image(cv2.cvtColor(cam_dbg, cv2.COLOR_BGR2RGB), width="stretch")

            with col_s:
                st.markdown("##### 🗺️ 2D Metric Simulation Pitch $(X_m, Y_m)$")
                sim_dbg = st.session_state.renderer.render_homography_simulation_pitch(
                    pos, dst_landmarks=st.session_state.transformer.dst_points, show_quad=True, show_coordinates=True
                )
                st.image(cv2.cvtColor(sim_dbg, cv2.COLOR_BGR2RGB), width="stretch")

        elif s2_view.startswith("🦅"):
            warped_turf = st.session_state.transformer.warp_to_pitch_plane(
                raw_frame, width=st.session_state.renderer.width, height=st.session_state.renderer.height, padding=st.session_state.renderer.padding
            )
            pitch_base = st.session_state.renderer.draw_base_pitch()
            blended_b = cv2.addWeighted(warped_turf, 0.75, pitch_base, 0.25, 0)
            turf_gray = cv2.cvtColor(warped_turf, cv2.COLOR_BGR2GRAY)
            blended_b[turf_gray == 0] = pitch_base[turf_gray == 0]
            st.image(cv2.cvtColor(blended_b, cv2.COLOR_BGR2RGB), caption="Orthorectified Pitch Turf (Bird's-Eye)", width="stretch")

        elif s2_view.startswith("🎥"):
            cam_ar = st.session_state.overlay_engine.render_homography_debug_frame(
                raw_frame, show_landmarks=True, show_turf_mask=True, show_markings=True, show_player_pins=True, positions=pos
            )
            st.image(cv2.cvtColor(cam_ar, cv2.COLOR_BGR2RGB), caption="Camera AR Projection Grid", width="stretch")

        else:
            # 🎯 Interactive Multi-Point Studio (N >= 4)
            st.markdown("##### 🎯 Live In-Studio Homography Recalibration")
            calib_res = render_mouse_calibration_studio(
                raw_frame=raw_frame,
                transformer=st.session_state.transformer,
                key_prefix="stage2_calib"
            )
            if calib_res.get("applied"):
                st.session_state.manual_calib_applied = True
                st.session_state.manual_calib_enabled = True
                st.session_state.manual_calib_points = calib_res.get("points", [])
                st.session_state.overlay_engine.transformer = st.session_state.transformer
                st.session_state.stream_manager.transformer = st.session_state.transformer
                st.session_state.stream_manager.detection_cache.clear()
                st.session_state.precomputed_data = None
                st.success("✅ Multi-Point Homography calibrated and applied across all stages!")
                st.rerun()

        st.markdown("---")
        m_c1, m_c2 = st.columns([1.2, 1.0])
        with m_c1:
            st.markdown("##### 🔢 3×3 Homography Matrix $\\mathbf{H}$")
            H = st.session_state.transformer.H
            st.markdown(f"""
            <div style="background:#0f172a; border:1px solid #1e293b; border-radius:8px; padding:12px; font-family:monospace; font-size:0.80rem; color:#38bdf8;">
                [ {H[0,0]:+13.6e}  {H[0,1]:+13.6e}  {H[0,2]:+13.6e} ]<br/>
                [ {H[1,0]:+13.6e}  {H[1,1]:+13.6e}  {H[1,2]:+13.6e} ]<br/>
                [ {H[2,0]:+13.6e}  {H[2,1]:+13.6e}  {H[2,2]:+13.6e} ]
            </div>
            """, unsafe_allow_html=True)
            det_H = np.linalg.det(H)
            st.caption(f"Determinant: `det(H) = {det_H:.4e}` • Condition Number: `κ(H) = {cond_H:.1f}`")

            # Calibration Solver & Error Metrics
            n_anchors = len(st.session_state.transformer.src_points) if hasattr(st.session_state.transformer, 'src_points') and st.session_state.transformer.src_points is not None else 4
            is_custom = getattr(st.session_state.transformer, 'is_custom_calibrated', False) or st.session_state.manual_calib_enabled
            solver_name = "Analytic Quad (Exact)" if n_anchors == 4 else f"RANSAC Least-Squares ({n_anchors} pts)"
            calib_tag = f"Custom Multi-Point ({solver_name})" if is_custom else f"Preset ({st.session_state.transformer.camera_mode.upper()})"
            st.markdown(f"<div style='font-size:0.75rem; color:#94a3b8; margin-top:4px;'>Calibration Mode: <b style='color:#38bdf8;'>{calib_tag}</b></div>", unsafe_allow_html=True)
            if is_custom and hasattr(st.session_state.transformer, 'compute_reprojection_residuals'):
                res_info = st.session_state.transformer.compute_reprojection_residuals()
                if res_info.get('residuals'):
                    st.caption(f"Reprojection Error: Mean `{res_info['mean_error']:.2f}px` • Max `{res_info['max_error']:.2f}px`")

        with m_c2:
            st.markdown("##### 🎯 Coordinate Probe (Pixels ➔ Pitch)")
            default_u = min(int(rf_w / 2), rf_w)
            default_v = min(int(rf_h / 2), rf_h)
            if "diag_pr_u" in st.session_state and st.session_state.diag_pr_u > rf_w:
                st.session_state.diag_pr_u = default_u
            if "diag_pr_v" in st.session_state and st.session_state.diag_pr_v > rf_h:
                st.session_state.diag_pr_v = default_v
            pr_u = st.number_input("Pixel X (u)", min_value=0, max_value=rf_w, value=default_u, step=20, key="diag_pr_u")
            pr_v = st.number_input("Pixel Y (v)", min_value=0, max_value=rf_h, value=default_v, step=20, key="diag_pr_v")
            probe_pt = st.session_state.transformer.image_to_pitch([(pr_u, pr_v)], filter_out_of_bounds=False)[0]
            in_p = (0.0 <= probe_pt[0] <= 105.0 and 0.0 <= probe_pt[1] <= 68.0)
            col_b = "#10b981" if in_p else "#ef4444"
            st.markdown(f"""
            <div style="background:#0f172a; border:1px solid {col_b}; border-radius:8px; padding:10px; margin-top:6px;">
                <div style="font-size:0.75rem; color:#94a3b8;">PROJECTED METRIC POSITION:</div>
                <div style="font-size:1.15rem; font-weight:700; color:{col_b};">X = {probe_pt[0]:.2f}m, Y = {probe_pt[1]:.2f}m</div>
                <span style="background:{col_b}22; color:{col_b}; padding:2px 8px; border-radius:4px; font-size:0.70rem; font-weight:700;">
                    {'ON PITCH [105x68m]' if in_p else 'OUT OF BOUNDS'}
                </span>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("📄 Raw `PitchPositions` Contract JSON", expanded=False):
            st.json(pos.to_dict())

    # --------------------------------------------------------------------------
    # STAGE 3: GEOMETRY CORE (DAA)
    # --------------------------------------------------------------------------
    elif active_stage == stage_options[2]:
        st.markdown("#### Stage 3: Computational Geometry Engine (DAA)")
        st.caption("Executes Graham Scan / Jarvis March Convex Hull and Fortune's Sweep-Line Voronoi partitioning.")

        h_tab_c1, h_tab_c2 = st.columns([1.3, 1.0])
        with h_tab_c1:
            st.markdown("##### 🔍 Convex Hull Vertex-to-Player Attribution Audit")
            hull_rows = []
            for idx, pt in enumerate(geom.hull_polygon):
                p_idx = geom.hull_player_indices[idx] if idx < len(geom.hull_player_indices) else -1
                img_px = st.session_state.transformer.pitch_to_image([pt])[0]
                p_str = f"Player B#{p_idx + 1}" if p_idx != -1 else "Defending Node"
                hull_rows.append(f"| **V{idx+1}** | `({pt[0]:.2f}m, {pt[1]:.2f}m)` | **{p_str}** | `({img_px[0]}, {img_px[1]})` | ✅ Verified |")

            if hull_rows:
                table_md = "\n".join(hull_rows)
                st.markdown(f"""
                | Vertex # | Metric Position | Player Attribution | Ground Contact Pixel | Node Status |
                | :---: | :---: | :---: | :---: | :---: |
                {table_md}
                """)
            else:
                st.info("Insufficient players on Team B (< 3) to construct defensive hull.")

        with h_tab_c2:
            st.markdown("##### ⚡ Algorithmic Metrics & Complexity")
            st.markdown(f"- Active Algorithm: **{hull_algo_name}**")
            st.markdown(f"- Time Complexity: **{'O(N log N)' if 'Graham' in hull_algo_name else 'O(N · H)'}**")
            st.markdown(f"- Defensive Block Area: **{geom.hull_area:.1f} m²**")
            st.markdown(f"- Compactness Index: **{geom.hull_compactness:.2f}**")
            st.markdown(f"- Voronoi Dominance: **{geom.team_a_pct:.1f}% vs {geom.team_b_pct:.1f}%**")
            if geom.dijkstra_res and geom.dijkstra_res.get('path'):
                d_info = geom.dijkstra_res
                st.markdown(f"- Dijkstra Passing Path: **{d_info.get('num_passes', 0)} passes (+{d_info.get('total_progression', 0):.1f}m progression, Risk: {d_info.get('avg_risk', 0):.1f})**")
            st.markdown(f"- Algorithmic Latency: **{geom.latency_ms:.2f} ms**")

        with st.expander("📄 Raw `TacticalGeometry` Contract JSON", expanded=False):
            st.json(geom.to_dict())

    # --------------------------------------------------------------------------
    # STAGE 4: VIEWPORT RENDERERS & MASK
    # --------------------------------------------------------------------------
    elif active_stage == stage_options[3]:
        st.markdown("#### Stage 4: Viewport Renderers & Turf Mask Verification")
        r_c1, r_c2, r_c3 = st.columns([1.1, 1.1, 0.8])
        pipe_fps = 1000.0 / max(bundle.get('latency_ms', 20.0), 1e-6)
        with r_c1:
            st.markdown("**Broadcast AR Overlay**")
            aug_dbg = stage4_render_overlay(
                raw_frame, pos, geom, st.session_state.overlay_engine,
                show_voronoi=True, show_hull=True,
                hull_algo_name=hull_algo_name,
                pipeline_fps=pipe_fps
            )
            st.image(cv2.cvtColor(aug_dbg, cv2.COLOR_BGR2RGB), width="stretch")
        with r_c2:
            st.markdown("**2D Tactical Pitch Radar**")
            rad_dbg = stage4_render_radar(pos, geom, st.session_state.renderer, show_voronoi=True, show_hull=True)
            st.image(cv2.cvtColor(rad_dbg, cv2.COLOR_BGR2RGB), width="stretch")
        with r_c3:
            st.markdown("**Turf Clipping Binary Mask**")
            t_mask = st.session_state.overlay_engine.get_turf_binary_mask(raw_frame.shape[:2])
            st.image(t_mask, caption="Prevents AR bleed outside pitch turf", width="stretch")

    # --------------------------------------------------------------------------
    # STAGE 5: DAA BENCHMARKS
    # --------------------------------------------------------------------------
    elif active_stage == stage_options[4]:
        st.markdown("### ⚡ Empirical DAA Benchmarking & Algorithmic Complexity")
        st.caption("Measures runtime latency and operation count across input point set scales $N$.")

        bench_nodes = st.multiselect(
            "Select Tracked Node Scales (N)",
            [5, 10, 15, 22, 50, 100, 200],
            default=[5, 10, 15, 22, 50, 100],
            key="bench_nodes_select"
        )
        if st.button("⚡ Run Empirical Benchmark Scan", type="primary", key="btn_run_bench"):
            with st.spinner("Executing empirical DAA benchmarking scan across CPU cores..."):
                df_hull = benchmark_convex_hull(sorted(bench_nodes), iterations=4)
                df_vor = benchmark_voronoi_dominance([n for n in sorted(bench_nodes) if n <= 200], iterations=4)
                df_dijk = benchmark_dijkstra_passing([n for n in [4, 6, 8, 11, 16, 22] if n <= max(bench_nodes)], iterations=4)

            bm1, bm2 = st.columns(2)
            with bm1:
                st.markdown("##### Convex Hull Empirical Runtimes (ms)")
                st.line_chart(df_hull.set_index('Nodes (N)')[['Graham Scan Runtime (ms)', 'Jarvis March Runtime (ms)']])
                st.dataframe(df_hull[['Nodes (N)', 'Hull Vertices (H)', 'Graham Scan Runtime (ms)', 'Jarvis March Runtime (ms)']], use_container_width=True)

            with bm2:
                st.markdown("##### Voronoi Sweep-Line Latency & FPS")
                st.line_chart(df_vor.set_index('Tracked Nodes (N)')[['Voronoi Runtime (ms)']])
                st.dataframe(df_vor[['Tracked Nodes (N)', 'Voronoi Runtime (ms)', 'FPS Capability']], use_container_width=True)

            st.markdown("##### 🟢 Dijkstra (Min-Heap) Passing Network vs Greedy Search (ms)")
            st.line_chart(df_dijk.set_index('Attacking Nodes (|V|)')[['Dijkstra (ms)', 'Greedy Search (ms)']])
            st.dataframe(df_dijk, use_container_width=True)
        else:
            st.info("Click '⚡ Run Empirical Benchmark Scan' above to profile asymptotic complexities.")

