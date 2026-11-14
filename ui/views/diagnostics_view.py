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
import threading

from algorithms.daa_benchmarks import benchmark_convex_hull, benchmark_voronoi_dominance, benchmark_dijkstra_passing
from core.pipeline import stage4_render_radar, stage4_render_overlay
from ui.components.coach_cards import render_coach_cards
from ui.state import invalidate_custom_user_cache, get_active_source_info
from ui.components.calibration_studio import render_mouse_calibration_studio
from ui.components.pitch_mask_studio import render_pitch_mask_studio


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
        b_title = "❌ ANALYSIS CHECK NEEDED"
    elif has_warn:
        b_color = "#78350f"
        b_border = "#d97706"
        b_title = "⚠️ ANALYSIS ACTIVE WITH WARNINGS"
    else:
        b_color = "#064e3b"
        b_border = "#059669"
        b_title = "✅ MATCH ANALYSIS HEALTHY"

    try:
        cond_H = float(np.linalg.cond(st.session_state.transformer.H)) if st.session_state.transformer.H is not None else 1.0
    except Exception:
        cond_H = 1.0
    cam_mode = "Custom Pitch Alignment" if st.session_state.manual_calib_enabled else ("Wide / Sideline" if st.session_state.transformer.camera_mode == "sideline" else st.session_state.transformer.camera_mode.title())

    st.markdown(f"""
    <div style="background:{b_color}; border:1px solid {b_border}; border-radius:10px; padding:12px 18px; margin-bottom:14px;">
        <div style="font-size:1.0rem; font-weight:700; color:#ffffff;">{b_title}</div>
        <div style="font-size:0.80rem; color:#f1f5f9; margin-top:3px;">
            Frame: <b>{st.session_state.current_frame + 1}</b> • Resolution: <b>{rf_w}×{rf_h}</b> • Camera Alignment: <b>{cam_mode}</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. 4 Health Cards Grid
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
                <div style="font-size:0.75rem; color:#94a3b8;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    render_diag_card(sc1, "Player Detection", diag_status.get('stage1', ('FAIL', 'No data')))
    render_diag_card(sc2, "Pitch Alignment", diag_status.get('stage2', ('FAIL', 'No data')))
    render_diag_card(sc3, "Tactical Metrics", diag_status.get('stage3', ('FAIL', 'No data')))
    render_diag_card(sc4, "Video & 2D Sync", diag_status.get('stage4', ('FAIL', 'No data')))

    if diag_issues:
        st.markdown("#### ⚠️ Attention Required")
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

    # 4. Inspector Tabs
    stage_options = [
        "👁️ Player Detection",
        "📐 Pitch Alignment",
        "📊 Tactical Metrics",
        "📺 Split Preview",
        "⏱️ Performance"
    ]
    active_stage = st.segmented_control(
        "Inspection Stage",
        stage_options,
        default=stage_options[3],
        key="active_stage_tab",
        label_visibility="collapsed"
    ) or stage_options[3]

    # --------------------------------------------------------------------------
    # PLAYER DETECTION
    # --------------------------------------------------------------------------
    if active_stage == stage_options[0]:
        st.markdown("#### Player & Ball Detection")
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

            st.image(cv2.cvtColor(det_img, cv2.COLOR_BGR2RGB), caption=f"Detected Players & Ball (Frame {st.session_state.current_frame + 1})", width="stretch")

        with c_info:
            st.markdown(f"- Total Players Tracked: **{len(det.players)}**")
            st.markdown(f"- Team A (Blue): **{len(det.team_a_pixels)}**")
            st.markdown(f"- Team B (Red): **{len(det.team_b_pixels)}**")
            st.markdown(f"- Ball Tracked: **{'Yes' if det.ball_pixel else 'Estimated'}**")
            st.markdown(f"- Video Resolution: **{det.frame_width} × {det.frame_height} px**")

            telemetry = bundle.get('onnx_telemetry') or (
                st.session_state.detector.get_onnx_telemetry()
                if hasattr(st.session_state, 'detector') and hasattr(st.session_state.detector, 'get_onnx_telemetry')
                else {}
            )
            m_name = telemetry.get('model_name', 'yolov8n.pt')
            is_ort = telemetry.get('is_onnx', False)
            intra_th = telemetry.get('intra_op_threads', 1)
            inter_th = telemetry.get('inter_op_threads', 1)
            ex_mode = telemetry.get('execution_mode', 'ORT_SEQUENTIAL')
            opt_lvl = telemetry.get('graph_optimization', 'ORT_ENABLE_ALL')
            provider = telemetry.get('provider', 'CPUExecutionProvider')
            cfg_active = telemetry.get('session_configured', False)

            rb_data = bundle.get('ring_buffer') or (
                st.session_state.stream_manager.get_ring_buffer_telemetry()
                if hasattr(st.session_state, 'stream_manager') and hasattr(st.session_state.stream_manager, 'get_ring_buffer_telemetry')
                else {}
            )
            rb_cnt = rb_data.get('count', 0)
            rb_cap = rb_data.get('capacity', 120)
            rb_hits = rb_data.get('hits', 0)
            rb_misses = rb_data.get('misses', 0)
            rb_ratio = rb_data.get('hit_ratio_pct', 0.0)
            rb_mem = rb_data.get('memory_mb', 0.0)

            with st.expander("System & Hardware Telemetry", expanded=False):
                st.markdown("##### Detection Engine")
                st.markdown(f"- **Runtime**: `{'ONNX Runtime' if is_ort else 'PyTorch'}` (`{m_name}`)")
                st.markdown(f"- **CPU Workers**: `{intra_th}` intra-op / `{inter_th}` inter-op (`{ex_mode}`)")
                st.markdown(f"- **Optimization / Provider**: `{opt_lvl}` / `{provider}` (`{'Tuned' if (is_ort and cfg_active) else 'Standard'}`)")
                st.markdown("##### Playback Memory Buffer")
                st.markdown(f"- **Buffered Frames**: `{rb_cnt} / {rb_cap}` ({rb_mem:.1f} MB RAM)")
                st.markdown(f"- **Cache Hit Rate**: `{rb_ratio:.1f}%` (Hits: `{rb_hits}`, Misses: `{rb_misses}`)")
                st.json(det.to_dict())

    # --------------------------------------------------------------------------
    # PITCH ALIGNMENT
    # --------------------------------------------------------------------------
    elif active_stage == stage_options[1]:
        st.markdown("#### Pitch Alignment & Top-Down Mapping")
        st.caption("Checks how accurately camera positions align onto the standard 105m × 68m tactical pitch.")

        s2_view = st.radio(
            "Alignment View",
            [
                "🔀 Split Video & 2D Pitch",
                "🦅 Top-Down Field View",
                "🎥 Field Grid Overlay",
                "🎯 Edit Pitch Alignment"
            ],
            index=0,
            horizontal=True
        )

        if s2_view.startswith("🔀"):
            col_c, col_s = st.columns(2)
            with col_c:
                st.markdown("##### 📷 Camera View")
                cam_dbg = st.session_state.overlay_engine.render_homography_debug_frame(
                    raw_frame, show_landmarks=True, show_turf_mask=True, show_markings=True, show_player_pins=True, positions=pos, player_detections=det.players
                )
                st.image(cv2.cvtColor(cam_dbg, cv2.COLOR_BGR2RGB), width="stretch")

            with col_s:
                st.markdown("##### 🗺️ 2D Tactical Pitch")
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
            st.image(cv2.cvtColor(blended_b, cv2.COLOR_BGR2RGB), caption="Top-Down Projected Pitch", width="stretch")

        elif s2_view.startswith("🎥"):
            cam_ar = st.session_state.overlay_engine.render_homography_debug_frame(
                raw_frame, show_landmarks=True, show_turf_mask=True, show_markings=True, show_player_pins=True, positions=pos, player_detections=det.players
            )
            st.image(cv2.cvtColor(cam_ar, cv2.COLOR_BGR2RGB), caption="Camera Field Grid Overlay", width="stretch")

        else:
            st.markdown("##### 🎯 Recalibrate Pitch Alignment")
            active_video_path = get_active_source_info(
                st.session_state.get("active_source_key", "premier_league")
            ).get("path", "")
            calib_res = render_pitch_mask_studio(
                raw_frame=raw_frame,
                transformer=st.session_state.transformer,
                video_path=active_video_path,
                key_prefix="stage2_calib_mask"
            )
            with st.expander("Advanced: Point-by-Point Coordinate Picker", expanded=False):
                pt_res = render_mouse_calibration_studio(
                    raw_frame=raw_frame,
                    transformer=st.session_state.transformer,
                    key_prefix="stage2_calib",
                    video_path=active_video_path
                )
                if pt_res.get("just_applied"):
                    calib_res = pt_res
            if calib_res.get("just_applied"):
                st.session_state.manual_calib_applied = True
                st.session_state.manual_calib_enabled = True
                st.session_state.manual_calib_points = calib_res.get("points", [])
                st.session_state.overlay_engine.transformer = st.session_state.transformer
                st.session_state.stream_manager.transformer = st.session_state.transformer
                if hasattr(st.session_state.stream_manager, 'clear_broadcast_cache'):
                    st.session_state.stream_manager.clear_broadcast_cache()
                elif hasattr(st.session_state.stream_manager, 'detection_cache'):
                    with getattr(st.session_state.stream_manager, '_cache_lock', threading.Lock()):
                        st.session_state.stream_manager.detection_cache.clear()
                st.session_state.precomputed_data = None
                invalidate_custom_user_cache(clear_user_cache_file=False)
                st.success("✅ Pitch alignment updated across all views!")
                st.rerun()
            elif calib_res.get("just_reset"):
                st.session_state.manual_calib_applied = False
                st.session_state.manual_calib_enabled = False
                st.session_state.tv_calib_applied = False
                st.session_state.tv_calib_enabled = False
                st.session_state.manual_calib_points = []
                st.session_state.tv_calib_points = []
                st.session_state.tv_calib_mesh_handles = None
                st.session_state.overlay_engine.transformer = st.session_state.transformer
                st.session_state.stream_manager.transformer = st.session_state.transformer
                if hasattr(st.session_state.stream_manager, 'clear_all_caches'):
                    st.session_state.stream_manager.clear_all_caches()
                elif hasattr(st.session_state.stream_manager, 'detection_cache'):
                    with getattr(st.session_state.stream_manager, '_cache_lock', threading.Lock()):
                        st.session_state.stream_manager.detection_cache.clear()
                st.session_state.precomputed_data = None
                invalidate_custom_user_cache(clear_user_cache_file=False)
                st.info("Pitch alignment reset to default camera preset.")
                st.rerun()

        st.markdown("---")
        m_c1, m_c2 = st.columns([1.2, 1.0])
        with m_c1:
            st.markdown("##### 📐 Alignment Status")
            n_anchors = len(st.session_state.transformer.src_points) if hasattr(st.session_state.transformer, 'src_points') and st.session_state.transformer.src_points is not None else 4
            is_custom = getattr(st.session_state.transformer, 'is_custom_calibrated', False) or st.session_state.manual_calib_enabled
            w_mode = getattr(st.session_state.transformer, 'warp_mode', 'planar')
            if w_mode == 'universal':
                cur_lam = getattr(st.session_state.transformer, 'lens_lambda', 0.0)
                solver_name = f"Universal Camera (λ = {cur_lam:+.3f}, {n_anchors} pts)"
            elif w_mode in ('panorama_tps', 'panorama'):
                solver_name = f"Wide / Panoramic ({n_anchors} pts)"
            elif n_anchors == 4:
                solver_name = "4-Point Standard"
            else:
                solver_name = f"Multi-Point ({n_anchors} pts)"
            calib_tag = f"Custom ({solver_name})" if is_custom else f"Preset ({st.session_state.transformer.camera_mode.title()})"
            st.markdown(f"- **Active Mode**: **{calib_tag}**")
            if is_custom and hasattr(st.session_state.transformer, 'compute_reprojection_residuals'):
                res_info = st.session_state.transformer.compute_reprojection_residuals()
                if res_info.get('residuals'):
                    st.markdown(f"- **Alignment Accuracy**: Mean `{res_info['mean_error']:.1f} px` • Max `{res_info['max_error']:.1f} px`")

            with st.expander("Technical Matrix Details", expanded=False):
                H = st.session_state.transformer.H
                st.markdown(f"""
                <div style="background:#0f172a; border:1px solid #1e293b; border-radius:8px; padding:12px; font-family:monospace; font-size:0.80rem; color:#38bdf8;">
                    [ {H[0,0]:+13.6e}  {H[0,1]:+13.6e}  {H[0,2]:+13.6e} ]<br/>
                    [ {H[1,0]:+13.6e}  {H[1,1]:+13.6e}  {H[1,2]:+13.6e} ]<br/>
                    [ {H[2,0]:+13.6e}  {H[2,1]:+13.6e}  {H[2,2]:+13.6e} ]
                </div>
                """, unsafe_allow_html=True)
                try:
                    det_H = float(np.linalg.det(H)) if H is not None else 0.0
                except Exception:
                    det_H = 0.0
                st.caption(f"Determinant: `{det_H:.4e}` • Condition: `κ = {cond_H:.1f}`")
                st.json(pos.to_dict())

        with m_c2:
            st.markdown("##### 🎯 Pitch Coordinate Checker")
            default_u = min(int(rf_w / 2), rf_w)
            default_v = min(int(rf_h / 2), rf_h)
            if "diag_pr_u" in st.session_state and st.session_state.diag_pr_u > rf_w:
                st.session_state.diag_pr_u = default_u
            if "diag_pr_v" in st.session_state and st.session_state.diag_pr_v > rf_h:
                st.session_state.diag_pr_v = default_v
            pr_u = st.number_input("Video X (px)", min_value=0, max_value=rf_w, value=default_u, step=20, key="diag_pr_u")
            pr_v = st.number_input("Video Y (px)", min_value=0, max_value=rf_h, value=default_v, step=20, key="diag_pr_v")
            probe_pt = st.session_state.transformer.image_to_pitch([(pr_u, pr_v)], filter_out_of_bounds=False)[0]
            in_p = (0.0 <= probe_pt[0] <= 105.0 and 0.0 <= probe_pt[1] <= 68.0)
            col_b = "#10b981" if in_p else "#ef4444"
            st.markdown(f"""
            <div style="background:#0f172a; border:1px solid {col_b}; border-radius:8px; padding:10px; margin-top:6px;">
                <div style="font-size:0.75rem; color:#94a3b8;">PITCH POSITION:</div>
                <div style="font-size:1.15rem; font-weight:700; color:{col_b};">X = {probe_pt[0]:.2f}m, Y = {probe_pt[1]:.2f}m</div>
                <span style="background:{col_b}22; color:{col_b}; padding:2px 8px; border-radius:4px; font-size:0.70rem; font-weight:700;">
                    {'ON PITCH (105×68m)' if in_p else 'OUT OF BOUNDS'}
                </span>
            </div>
            """, unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # TACTICAL METRICS
    # --------------------------------------------------------------------------
    elif active_stage == stage_options[2]:
        st.markdown("#### Tactical Metrics & Space Analysis")
        st.caption("Detailed breakdown of defensive block compactness, space control, best passing lanes, and simulated runs.")

        s3_hull_toggle = st.session_state.get('show_hull', True)
        s3_vor_toggle = st.session_state.get('show_voronoi', True)
        s3_dijk_toggle = st.session_state.get('show_dijkstra', True)
        s3_graph_toggle = st.session_state.get('show_passing_graph', False)

        geom_canvas = st.session_state.overlay_engine.render_geometry_debug_frame(
            raw_frame,
            positions=pos,
            geometry=geom,
            show_hull=s3_hull_toggle,
            show_voronoi=s3_vor_toggle,
            show_dijkstra=s3_dijk_toggle,
            show_passing_graph=s3_graph_toggle,
            show_centroids=True
        )
        st.image(
            cv2.cvtColor(geom_canvas, cv2.COLOR_BGR2RGB),
            caption=f"Tactical Overlay Frame {st.session_state.current_frame + 1}",
            width="stretch"
        )

        st.markdown("---")
        s3_sub = st.radio(
            "Select Analysis Section",
            [
                "🛡️ Defensive Block",
                "🌐 Space Control",
                "⚡ Passing Lanes",
                "👻 Run Simulator",
                "📄 Raw Data"
            ],
            index=0,
            horizontal=True,
            key="s3_component_radio"
        )

        if s3_sub.startswith("🛡️"):
            h_tab_c1, h_tab_c2 = st.columns([1.3, 1.0])
            with h_tab_c1:
                st.markdown("##### 🔍 Defensive Perimeter Players")
                hull_rows = []
                for idx, pt in enumerate(geom.hull_polygon):
                    p_idx = geom.hull_player_indices[idx] if idx < len(geom.hull_player_indices) else -1
                    proj_pxs = st.session_state.transformer.pitch_to_image([pt])
                    img_px = proj_pxs[0] if proj_pxs else (0, 0)
                    p_str = f"Player B#{p_idx + 1}" if p_idx != -1 else "Defender"
                    hull_rows.append(f"| **#{idx+1}** | `({pt[0]:.1f}m, {pt[1]:.1f}m)` | **{p_str}** | `({img_px[0]}, {img_px[1]})` | ✅ Tracked |")

                if hull_rows:
                    table_md = "\n".join(hull_rows)
                    st.markdown(f"""
                    | Point | Pitch Position | Player | Video Pixel | Status |
                    | :---: | :---: | :---: | :---: | :---: |
                    {table_md}
                    """)
                else:
                    st.info("Not enough Team B players (< 3) to outline the defensive block.")

            with h_tab_c2:
                st.markdown("##### 🛡️ Defensive Shape Summary")
                st.markdown(f"- **Defensive Block Area**: **{geom.hull_area:.1f} m²**")
                st.markdown(f"- **Compactness Score**: **{geom.hull_compactness:.2f}**")
                st.markdown(f"- **Boundary Players**: **{len(geom.hull_polygon)}**")
                st.markdown(f"- **Overlay Visible**: **{'Yes' if s3_hull_toggle else 'Hidden'}**")

        elif s3_sub.startswith("🌐"):
            v_col1, v_col2 = st.columns([1.3, 1.0])
            with v_col1:
                st.markdown("##### 🗺️ Space Control Zones")
                st.markdown(f"""
                - **Team A Space Share**: **{geom.team_a_pct:.1f}%** ({len(geom.voronoi_cells_a)} zones)
                - **Team B Space Share**: **{geom.team_b_pct:.1f}%** ({len(geom.voronoi_cells_b)} zones)
                """)

                cell_rows = []
                for idx, c in enumerate(geom.voronoi_cells_a[:5]):
                    gen_site = c.get('generator_site') or (0.0, 0.0)
                    gx, gy = float(gen_site[0]), float(gen_site[1])
                    v_cnt = len(c.get('polygon', []))
                    cell_rows.append(f"| Team A #{idx+1} | `({gx:.1f}m, {gy:.1f}m)` | {v_cnt} | Attacking |")
                for idx, c in enumerate(geom.voronoi_cells_b[:5]):
                    gen_site = c.get('generator_site') or (0.0, 0.0)
                    gx, gy = float(gen_site[0]), float(gen_site[1])
                    v_cnt = len(c.get('polygon', []))
                    cell_rows.append(f"| Team B #{idx+1} | `({gx:.1f}m, {gy:.1f}m)` | {v_cnt} | Defending |")

                if cell_rows:
                    t_body = "\n".join(cell_rows)
                    st.markdown(f"""
                    | Player | Pitch Position (X, Y) | Zone Corners | Role |
                    | :---: | :---: | :---: | :---: |
                    {t_body}
                    """)

            with v_col2:
                st.markdown("##### 🌐 Zone Summary")
                st.markdown("- **Pitch Area**: Standard 105m × 68m")
                st.markdown("- **Total Player Zones**: **" + str(len(geom.voronoi_cells_a) + len(geom.voronoi_cells_b)) + "**")
                st.markdown(f"- **Overlay Visible**: **{'Yes' if s3_vor_toggle else 'Hidden'}**")

        elif s3_sub.startswith("⚡"):
            d_col1, d_col2 = st.columns([1.3, 1.0])
            d_res = geom.dijkstra_res or {}
            with d_col1:
                st.markdown("##### 🟢 Recommended Passing Sequence")
                d_path = d_res.get('path', [])
                d_pts = d_res.get('corridor_points') or d_res.get('path_points', [])
                if len(d_path) >= 2 and d_pts and len(d_pts) >= len(d_path):
                    pass_rows = []
                    for i in range(len(d_path) - 1):
                        u_idx = d_path[i]
                        v_idx = d_path[i + 1]
                        p1 = d_pts[i]
                        p2 = d_pts[i + 1]
                        dx = p2[0] - p1[0]
                        dy = p2[1] - p1[1]
                        dist = np.hypot(dx, dy)
                        pass_rows.append(f"| Pass {i+1} | Player A#{u_idx + 1} ➔ Player A#{v_idx + 1} | `{dist:.1f}m` | `+{dx:.1f}m` | Best Lane |")
                    
                    t_pass_body = "\n".join(pass_rows)
                    st.markdown(f"""
                    | Step | Pass | Distance | Forward Gain | Priority |
                    | :---: | :---: | :---: | :---: | :---: |
                    {t_pass_body}
                    """)
                else:
                    st.info("No forward passing option available from current ball carrier.")

            with d_col2:
                st.markdown("##### ⚡ Passing Summary")
                total_g_edges = d_res.get('total_edges', len(d_res.get('graph_edges', [])))
                st.markdown(f"- **Open Pass Options**: **{total_g_edges}**")
                st.markdown(f"- **Net Forward Gain**: **+{d_res.get('total_progression', 0.0):.1f} m**")
                st.markdown(f"- **Average Interception Risk**: **{d_res.get('avg_risk', 0.0):.2f}**")
                st.markdown(f"- **Pass Sequence Length**: **{d_res.get('num_passes', max(0, len(d_path) - 1))} passes**")
                st.markdown(f"- **Best Pass Overlay**: **{'Shown' if s3_dijk_toggle else 'Hidden'}**")
                st.markdown(f"- **All Pass Options Overlay**: **{'Shown' if s3_graph_toggle else 'Off'}**")

            g_edges = d_res.get('graph_edges', [])
            if g_edges:
                with st.expander(f"🕸️ All Passing Options ({len(g_edges)} lanes)", expanded=False):
                    opt_pairs = [[d_path[k], d_path[k+1]] for k in range(len(d_path) - 1)] if len(d_path) >= 2 else []
                    edge_rows = []
                    for e_idx, e in enumerate(g_edges[:35]):
                        is_opt = [e['source_idx'], e['target_idx']] in opt_pairs
                        status_tag = "🟢 Recommended" if is_opt else "⚪ Option"
                        edge_rows.append(f"| #{e_idx+1} | Player A#{e['source_idx']+1} ➔ Player A#{e['target_idx']+1} | `{e['distance']:.1f}m` | `{e['pressure']:.2f}` | `{e['weight']:.1f}` | {status_tag} |")

                    e_table_body = "\n".join(edge_rows)
                    st.markdown(f"""
                    | # | Pass | Length | Defender Pressure | Lane Risk | Status |
                    | :---: | :---: | :---: | :---: | :---: | :---: |
                    {e_table_body}
                    """)
                    if len(g_edges) > 35:
                        st.caption(f"Showing first 35 of {len(g_edges)} passing options.")

        elif s3_sub.startswith("👻"):
            from algorithms.ghost_runner import simulate_ghost_runner

            st.markdown("##### 👻 Run Simulator (What-If Testing)")
            st.caption("Place a simulated attacking runner on the pitch to test how it opens space and creates passing options.")

            g_col1, g_col2 = st.columns([1.2, 1.8])
            with g_col1:
                st.markdown("###### 🎯 Runner Position")
                preset_choice = st.selectbox(
                    "Run Presets",
                    [
                        "Underlapping Wing Run (75m, 12m)",
                        "Half-Space Pocket (72m, 48m)",
                        "Far-Post Blindside Run (88m, 20m)",
                        "Central Overload (68m, 34m)",
                        "Custom Position"
                    ],
                    index=0,
                    key="diag_ghost_preset"
                )
                if preset_choice.startswith("Underlapping"):
                    def_x, def_y = 75.0, 12.0
                elif preset_choice.startswith("Half-Space"):
                    def_x, def_y = 72.0, 48.0
                elif preset_choice.startswith("Far-Post"):
                    def_x, def_y = 88.0, 20.0
                elif preset_choice.startswith("Central"):
                    def_x, def_y = 68.0, 34.0
                else:
                    def_x, def_y = 70.0, 34.0

                gx = st.slider("Lengthwise X (m)", 0.0, 105.0, float(def_x), 0.5, key="diag_ghost_x")
                gy = st.slider("Sideways Y (m)", 0.0, 68.0, float(def_y), 0.5, key="diag_ghost_y")

            sim_res = simulate_ghost_runner(
                attacking_players=pos.team_a,
                defending_players=pos.team_b,
                ghost_coord=(gx, gy),
                carrier_idx=pos.carrier_idx,
                ball_coord=pos.ball
            )

            with g_col2:
                v_border = "#ef4444" if sim_res['is_offside'] else ("#10b981" if sim_res['passes_to_ghost'] else "#38bdf8")
                st.markdown(f"""
                <div style="background:#0f172a; border-left:4px solid {v_border}; border-radius:6px; padding:10px 14px; margin-bottom:12px;">
                    <div style="font-weight:700; font-size:0.85rem; color:#ffffff;">{sim_res['tactical_verdict']}</div>
                    <div style="font-size:0.75rem; color:#94a3b8; margin-top:3px;">
                        Offside Line: <b>X = {sim_res['offside_x']:.1f}m</b> • Margin: <b>{sim_res['offside_margin_m']:+.2f}m</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                mc1, mc2, mc3 = st.columns(3)
                with mc1:
                    delta_prog = sim_res['progression_delta_m']
                    st.metric("Forward Gain", f"{delta_prog:+.1f} m", delta=f"{delta_prog:+.1f} m" if delta_prog != 0 else None)
                with mc2:
                    delta_terr = sim_res['territory_delta_pct']
                    st.metric("Space Shift", f"{delta_terr:+.1f}%", delta=f"{delta_terr:+.1f}%" if delta_terr != 0 else None)
                with mc3:
                    st.metric("Space Opened", f"{sim_res['space_claimed_m2']:.0f} m²", help="Open space created by the simulated runner")

            st.markdown("###### ⚡ Pass Route Comparison")
            base_p = sim_res['baseline_dijkstra'].get('path', [])
            sim_p = sim_res['simulated_dijkstra'].get('path', [])
            b_str = " ➔ ".join([f"Player A#{idx+1}" for idx in base_p]) if base_p else "No sequence"
            s_nodes = []
            for idx in sim_p:
                if idx == sim_res['ghost_player_idx']:
                    s_nodes.append("👻 **Simulated Runner**")
                else:
                    s_nodes.append(f"Player A#{idx+1}")
            s_str = " ➔ ".join(s_nodes) if s_nodes else "No sequence"

            st.markdown(f"- **Current Route**: `{b_str}` ({sim_res['baseline_progression_m']:.1f}m)")
            st.markdown(f"- **With Simulated Run**: `{s_str}` ({sim_res['simulated_progression_m']:.1f}m)")

        else:
            st.markdown("##### 📄 Raw Frame Metrics")
            st.caption("Complete exported metrics payload for the current frame.")
            st.json(geom.to_dict())

    # --------------------------------------------------------------------------
    # SPLIT PREVIEW & FIELD MASK
    # --------------------------------------------------------------------------
    elif active_stage == stage_options[3]:
        st.markdown("#### Split Preview & Field Mask")
        st.caption("Preview synchronized video overlays, the 2D tactical pitch, and the pitch boundary mask.")

        v_layout_col, v_status_col = st.columns([1.6, 2.4])
        with v_layout_col:
            s4_layout = st.radio(
                "Preview Layout",
                [
                    "⚡ Split View",
                    "🗺️ 2D Pitch Only",
                    "📺 Video Only",
                    "🛡️ Field Mask Only"
                ],
                index=0,
                horizontal=True,
                key="s4_layout_choice",
                label_visibility="collapsed"
            )
        with v_status_col:
            show_h = st.session_state.get('show_hull', True)
            show_v = st.session_state.get('show_voronoi', True)
            show_d = st.session_state.get('show_dijkstra', True)
            show_g = st.session_state.get('show_passing_graph', False)
            show_r = st.session_state.get('show_ruler', False)

            badges = [
                f"<span style='color:{'#38bdf8' if show_h else '#475569'};'>{'🛡️ Block' if show_h else '<s>🛡️ Block</s>'}</span>",
                f"<span style='color:{'#38bdf8' if show_v else '#475569'};'>{'🌐 Space' if show_v else '<s>🌐 Space</s>'}</span>",
                f"<span style='color:{'#10b981' if show_d else '#475569'};'>{'⚡ Best Pass' if show_d else '<s>⚡ Best Pass</s>'}</span>",
                f"<span style='color:{'#a855f7' if show_g else '#475569'};'>{'🕸️ Pass Options' if show_g else '<s>🕸️ Pass Options</s>'}</span>",
                f"<span style='color:{'#f59e0b' if show_r else '#475569'};'>{'📏 Ruler' if show_r else '<s>📏 Ruler</s>'}</span>"
            ]
            st.markdown(f"""
            <div style="display:flex; justify-content:flex-end; align-items:center; gap:8px; margin-top:6px; font-size:0.75rem;">
                <span style="color:#94a3b8; font-weight:600;">Active Layers:</span>
                {' • '.join(badges)}
            </div>
            """, unsafe_allow_html=True)

        lat_val = float(bundle.get('latency_ms') or 20.0)
        pipe_fps = 1000.0 / max(lat_val, 1e-6)
        r_data = bundle.get('ruler_data') if show_r else None

        if show_r and r_data:
            breaches = r_data.get('offside_breaches', [])
            off_tag = f"🚨 {len(breaches)} Offside" if breaches else "✅ Onside"
            b_border = "#ef4444" if breaches else "#059669"
            st.markdown(f"""
            <div style="background:#0f172a; border:1px solid {b_border}; border-radius:8px; padding:10px 16px; margin-bottom:12px; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
                <div style="font-weight:700; font-size:0.85rem; color:#f59e0b;">
                    📏 Distance & Offside:
                    <span style="color:#ffffff; margin-left:6px;">{r_data.get('label_a', 'Player A')} ➔ {r_data.get('label_b', 'Player B')}</span>
                </div>
                <div style="font-size:0.80rem; color:#cbd5e1; font-family:monospace;">
                    Distance: <b style="color:#38bdf8;">{r_data.get('distance_m', 0.0):.2f}m</b> |
                    Depth Δx: <b>{r_data.get('delta_x_m', 0.0):.2f}m</b> |
                    Width Δy: <b>{r_data.get('delta_y_m', 0.0):.2f}m</b> |
                    Offside Line: <b>X = {r_data.get('offside_x_m', 0.0):.2f}m</b> |
                    <span style="color:{'#ef4444' if breaches else '#10b981'}; font-weight:700;">{off_tag}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        if s4_layout.startswith("⚡"):
            r_c1, r_c2 = st.columns(2)
            with r_c1:
                st.markdown("##### 📺 Match Video Overlay")
                aug_dbg = stage4_render_overlay(
                    raw_frame, pos, geom, st.session_state.overlay_engine,
                    show_voronoi=show_v, show_hull=show_h, show_dijkstra=show_d,
                    show_passing_graph=show_g,
                    ruler_data=r_data,
                    hull_algo_name=hull_algo_name,
                    pipeline_fps=pipe_fps
                )
                st.image(cv2.cvtColor(aug_dbg, cv2.COLOR_BGR2RGB), width="stretch")

            with r_c2:
                st.markdown("##### 🗺️ 2D Tactical Pitch")
                rad_dbg = stage4_render_radar(
                    pos, geom, st.session_state.renderer,
                    show_voronoi=show_v, show_hull=show_h, show_dijkstra=show_d,
                    show_passing_graph=show_g,
                    ruler_data=r_data
                )
                st.image(cv2.cvtColor(rad_dbg, cv2.COLOR_BGR2RGB), width="stretch")

        elif s4_layout.startswith("🗺️"):
            st.markdown("##### 🗺️ 2D Tactical Pitch")
            rad_dbg = stage4_render_radar(
                pos, geom, st.session_state.renderer,
                show_voronoi=show_v, show_hull=show_h, show_dijkstra=show_d,
                show_passing_graph=show_g,
                ruler_data=r_data
            )
            st.image(cv2.cvtColor(rad_dbg, cv2.COLOR_BGR2RGB), width="stretch")

        elif s4_layout.startswith("📺"):
            st.markdown("##### 📺 Match Video Overlay")
            aug_dbg = stage4_render_overlay(
                raw_frame, pos, geom, st.session_state.overlay_engine,
                show_voronoi=show_v, show_hull=show_h, show_dijkstra=show_d,
                show_passing_graph=show_g,
                ruler_data=r_data,
                hull_algo_name=hull_algo_name,
                pipeline_fps=pipe_fps
            )
            st.image(cv2.cvtColor(aug_dbg, cv2.COLOR_BGR2RGB), width="stretch")

        else:
            st.markdown("##### 🛡️ Field Turf Boundary Mask")
            t_mask = st.session_state.overlay_engine.get_turf_binary_mask(raw_frame.shape[:2])
            st.image(t_mask, caption="Keeps field overlays strictly inside the pitch turf", width="stretch")

    # --------------------------------------------------------------------------
    # PERFORMANCE CHECK
    # --------------------------------------------------------------------------
    elif active_stage == stage_options[4]:
        st.markdown("### ⏱️ System Performance Check")
        st.caption("Measure tactical computation speed on your hardware across different player counts.")

        bench_nodes = st.multiselect(
            "Player Count Scales",
            [5, 10, 15, 22, 50, 100, 200],
            default=[5, 10, 15, 22, 50, 100],
            key="bench_nodes_select"
        )
        if st.button("⏱️ Run Speed Test", type="primary", key="btn_run_bench"):
            active_nodes = sorted(bench_nodes) if bench_nodes else [5, 10, 15, 22, 50, 100]
            max_n = max(active_nodes)
            with st.spinner("Running performance benchmark..."):
                df_hull = benchmark_convex_hull(active_nodes, iterations=4)
                df_vor = benchmark_voronoi_dominance([n for n in active_nodes if n <= 200], iterations=4)
                df_dijk = benchmark_dijkstra_passing([n for n in [4, 6, 8, 11, 16, 22] if n <= max_n], iterations=4)

            bm1, bm2 = st.columns(2)
            with bm1:
                st.markdown("##### Defensive Block Speed (ms)")
                if not df_hull.empty and 'Nodes (N)' in df_hull.columns:
                    st.line_chart(df_hull.set_index('Nodes (N)')[['Graham Scan Runtime (ms)', 'Jarvis March Runtime (ms)']])
                    st.dataframe(df_hull[['Nodes (N)', 'Hull Vertices (H)', 'Graham Scan Runtime (ms)', 'Jarvis March Runtime (ms)']], width="stretch")

            with bm2:
                st.markdown("##### Space Control Speed & FPS")
                if not df_vor.empty and 'Tracked Nodes (N)' in df_vor.columns:
                    st.line_chart(df_vor.set_index('Tracked Nodes (N)')[['Voronoi Runtime (ms)']])
                    st.dataframe(df_vor[['Tracked Nodes (N)', 'Voronoi Runtime (ms)', 'FPS Capability']], width="stretch")

            st.markdown("##### 🟢 Passing Lane Speed (ms)")
            if not df_dijk.empty and 'Attacking Nodes (|V|)' in df_dijk.columns:
                st.line_chart(df_dijk.set_index('Attacking Nodes (|V|)')[['Dijkstra (ms)', 'Greedy Search (ms)']])
                st.dataframe(df_dijk, width="stretch")
        else:
            st.info("Click '⏱️ Run Speed Test' above to benchmark analysis performance.")

