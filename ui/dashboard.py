"""
Real-time UI Dashboard Overlay renderer for OpenCV frames.
"""

from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np


class DashboardOverlay:
    """
    Renders 6-DoF pose telemetry, 3D coordinate axes, and status overlays onto live camera frames.
    """

    def __init__(self) -> None:
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def draw_3d_axis(
        self,
        frame: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        length: float = 0.05
    ) -> np.ndarray:
        """Projects and draws 3D coordinate axes (X: Red, Y: Green, Z: Blue) onto OpenCV frame."""
        if hasattr(cv2, "drawFrameAxes"):
            return cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, length)
        elif hasattr(cv2.aruco, "drawAxis"):
            return cv2.aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvec, tvec, length)
        return frame

    def render_telemetry(
        self,
        frame: np.ndarray,
        pos_xyz: Tuple[float, float, float],
        rot_euler: Tuple[float, float, float],
        deltas: Optional[Dict[str, Tuple[float, float, float]]] = None
    ) -> np.ndarray:
        """Draws live numeric telemetry HUD onto the camera frame."""
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (10, 10), (420, 180 if deltas else 100), (0, 0, 0), -1)

        x, y, z = pos_xyz
        rx, ry, rz = rot_euler

        cv2.putText(frame, f"Pos (m):  X={x:.3f} Y={y:.3f} Z={z:.3f}", (20, 35), self.font, 0.5, (0, 255, 255), 1)
        cv2.putText(frame, f"Rot (deg): Rx={rx:.1f} Ry={ry:.1f} Rz={rz:.1f}", (20, 65), self.font, 0.5, (255, 255, 0), 1)

        if deltas:
            dx, dy, dz = deltas["translation_delta"]
            drx, dry, drz = deltas["rotation_delta"]
            cv2.putText(frame, f"dX/dY/dZ:  {dx:+.3f} {dy:+.3f} {dz:+.3f}", (20, 115), self.font, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, f"dRx/dRy/dRz: {drx:+.1f} {dry:+.1f} {drz:+.1f}", (20, 145), self.font, 0.5, (0, 255, 0), 1)

        return frame

    def render_calibration_hud(
        self,
        frame: np.ndarray,
        chessboard_detected: bool,
        num_frames_captured: int,
        rms_error: Optional[float] = None,
        status_message: str = "",
        pattern_size: Tuple[int, int] = (9, 6),
        square_size_mm: float = 25.0
    ) -> np.ndarray:
        """
        Renders live calibration HUD overlay:
        - Chessboard detection state
        - Number of captured calibration views
        - Calculated RMS reprojection error
        - Keyboard controls guide (SPACE, C, R, Q)
        """
        overlay = frame.copy()
        h, w = frame.shape[:2]

        # Top banner panel
        panel_h = 160
        cv2.rectangle(overlay, (10, 10), (520, 10 + panel_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        
        border_color = (0, 255, 0) if chessboard_detected else (0, 165, 255)
        cv2.rectangle(frame, (10, 10), (520, 10 + panel_h), border_color, 2)

        # Title & Pattern Config
        cv2.putText(frame, "STAGE 2: CAMERA CALIBRATION", (25, 35), self.font, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Grid: {pattern_size[0]}x{pattern_size[1]} inner corners | Square: {square_size_mm:.1f} mm", 
                    (25, 60), self.font, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        # Chessboard Detection Status
        if chessboard_detected:
            cv2.putText(frame, "STATUS: CHESSBOARD DETECTED [Ready for SPACE]", (25, 85), self.font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "STATUS: SEARCHING CHESSBOARD...", (25, 85), self.font, 0.5, (0, 165, 255), 1, cv2.LINE_AA)

        # Frame Count & Reprojection Error
        cv2.putText(frame, f"Captured Views: {num_frames_captured} / 5+ minimum", (25, 110), self.font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        if rms_error is not None:
            cv2.putText(frame, f"RMS Error: {rms_error:.4f} px (Calibrated)", (25, 135), self.font, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
        elif status_message:
            cv2.putText(frame, f"Info: {status_message[:45]}", (25, 135), self.font, 0.45, (255, 255, 0), 1, cv2.LINE_AA)

        # Bottom Controls Bar
        cv2.rectangle(frame, (10, h - 45), (w - 10, h - 10), (0, 0, 0), -1)
        cv2.putText(
            frame,
            "[SPACE] Capture Frame  |  [C] Calculate Calibration  |  [R] Reset Frames  |  [Q] Quit",
            (20, h - 22),
            self.font,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        return frame

    def render_verification_hud(
        self,
        frame_original: np.ndarray,
        frame_undistorted: np.ndarray,
        rms_error: Optional[float],
        resolution: Tuple[int, int]
    ) -> np.ndarray:
        """
        Renders a side-by-side verification HUD comparing original camera feed and undistorted feed.
        """
        h_orig, w_orig = frame_original.shape[:2]

        # Draw section headers on each feed image
        orig_view = frame_original.copy()
        undist_view = frame_undistorted.copy()

        cv2.rectangle(orig_view, (10, 10), (320, 45), (0, 0, 0), -1)
        cv2.putText(orig_view, "ORIGINAL FEED (Distorted)", (20, 32), self.font, 0.55, (0, 165, 255), 2, cv2.LINE_AA)

        cv2.rectangle(undist_view, (10, 10), (360, 45), (0, 0, 0), -1)
        cv2.putText(undist_view, "UNDISTORTED FEED (Lens Corrected)", (20, 32), self.font, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

        # Concatenate side-by-side
        combined = np.hstack([orig_view, undist_view])
        ch, cw = combined.shape[:2]

        # Top overlay banner across center
        panel_w = 460
        panel_h = 100
        panel_x = (cw - panel_w) // 2
        
        overlay = combined.copy()
        cv2.rectangle(overlay, (panel_x, 10), (panel_x + panel_w, 10 + panel_h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.8, combined, 0.2, 0, combined)
        cv2.rectangle(combined, (panel_x, 10), (panel_x + panel_w, 10 + panel_h), (0, 255, 0), 2)

        # Telemetry Text
        cv2.putText(combined, "CALIBRATION VERIFICATION MODE", (panel_x + 20, 35), self.font, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(combined, "Calibration: LOADED", (panel_x + 20, 60), self.font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        
        rms_str = f"{rms_error:.4f} px" if rms_error is not None else "N/A"
        cv2.putText(combined, f"RMS Error: {rms_str}", (panel_x + 220, 60), self.font, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(combined, f"Camera Resolution: {resolution[0]}x{resolution[1]}", (panel_x + 20, 85), self.font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Bottom Controls bar
        cv2.rectangle(combined, (10, ch - 40), (cw - 10, ch - 10), (0, 0, 0), -1)
        cv2.putText(combined, "[Q] / [ESC] Quit Verification Mode", (20, ch - 18), self.font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        return combined

    def render_aruco_hud(
        self,
        frame: np.ndarray,
        is_detected: bool,
        marker_ids: List[int],
        dictionary_name: str,
        marker_size_mm: float,
        is_calibrated: bool = True
    ) -> np.ndarray:
        """
        Renders live ArUco detection status overlay:
        - ARUCO: DETECTED / ARUCO: NOT DETECTED
        - ID: <id>
        - Dictionary & marker size configuration
        """
        overlay = frame.copy()
        h, w = frame.shape[:2]

        panel_w = 420
        panel_h = 110
        cv2.rectangle(overlay, (15, 15), (15 + panel_w, 15 + panel_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        border_color = (0, 255, 0) if is_detected else (0, 165, 255)
        cv2.rectangle(frame, (15, 15), (15 + panel_w, 15 + panel_h), border_color, 2)

        # ARUCO Status
        if is_detected:
            cv2.putText(frame, "ARUCO: DETECTED", (30, 45), self.font, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            ids_str = ", ".join(str(i) for i in marker_ids)
            cv2.putText(frame, f"ID: {ids_str}", (30, 75), self.font, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "ARUCO: NOT DETECTED", (30, 45), self.font, 0.6, (0, 165, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "Searching for physical ArUco target...", (30, 75), self.font, 0.48, (200, 200, 200), 1, cv2.LINE_AA)

        # Config text
        calib_str = "CALIBRATED" if is_calibrated else "UNCALIBRATED"
        cv2.putText(
            frame,
            f"Dict: {dictionary_name} | Size: {marker_size_mm:.1f}mm | {calib_str}",
            (30, 105),
            self.font,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        # Bottom Controls bar
        cv2.rectangle(frame, (10, h - 40), (w - 10, h - 10), (0, 0, 0), -1)
        cv2.putText(frame, "[Q] / [ESC] Quit ArUco Tracking Mode", (20, h - 18), self.font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        return frame

    def render_6dof_pose_hud(
        self,
        frame: np.ndarray,
        is_detected: bool,
        marker_id: Optional[int] = None,
        pose_data: Optional[Dict[str, Any]] = None,
        ref_data: Optional[Dict[str, Any]] = None,
        delta_data: Optional[Dict[str, Any]] = None,
        status_message: str = "",
        fps: float = 0.0,
        calibration_loaded: bool = True,
        rms_error: Optional[float] = None,
        active_mode: str = "TRACKING"
    ) -> np.ndarray:
        """
        Renders a clean, high-performance real-time 6-DoF telemetry dashboard overlay:
        - Top Status Bar: CAMERA STATUS (FPS/Res), ARUCO STATUS (ID), REFERENCE STATUS, CALIBRATION STATUS (RMS)
        - Column 1: LIVE POSE (X, Y, Z mm, Rx, Ry, Rz deg)
        - Column 2: REFERENCE POSE (X, Y, Z mm, Rx, Ry, Rz deg)
        - Column 3: RELATIVE CHANGE (ΔX, ΔY, ΔZ mm, ΔRx, ΔRy, ΔRz deg)
        - Stage 8 Panel: 4x4 SE(3) Homogeneous Transformation Matrix (T_ref_to_curr)
        - Controls Bar: [SPACE] Save Ref | [R] Reset Ref | [C] Calibrate | [T] Tracking | [A] Accuracy | [Q] Quit
        """
        overlay = frame.copy()
        h, w = frame.shape[:2]

        # ------------------- 1. TOP STATUS BADGES BAR (y: 10 to 65) -------------------
        card_w = (w - 50) // 4
        card_h = 55
        card_y = 10

        # Draw semi-transparent header background cards
        for i in range(4):
            cx = 10 + i * (card_w + 10)
            cv2.rectangle(overlay, (cx, card_y), (cx + card_w, card_y + card_h), (18, 18, 18), -1)
        
        # Apply alpha blending for translucent look
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

        # Card 1: CAMERA STATUS & FPS
        cx1 = 10
        cv2.rectangle(frame, (cx1, card_y), (cx1 + card_w, card_y + card_h), (0, 255, 0), 1)
        cv2.putText(frame, "CAMERA STATUS", (cx1 + 10, card_y + 20), self.font, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
        fps_str = f"FPS: {fps:.1f}" if fps > 0 else "FPS: --"
        cv2.putText(frame, f"OK ({w}x{h}) | {fps_str}", (cx1 + 10, card_y + 42), self.font, 0.44, (255, 255, 255), 1, cv2.LINE_AA)

        # Card 2: ARUCO STATUS & MARKER ID
        cx2 = 10 + (card_w + 10)
        aruco_color = (0, 255, 0) if is_detected else (0, 0, 255)
        cv2.rectangle(frame, (cx2, card_y), (cx2 + card_w, card_y + card_h), aruco_color, 1)
        status_text = "ARUCO: DETECTED" if is_detected else "ARUCO: NOT DETECTED"
        id_text = f"Marker ID: {marker_id}" if is_detected and marker_id is not None else "Marker ID: N/A"
        cv2.putText(frame, status_text, (cx2 + 10, card_y + 20), self.font, 0.45, aruco_color, 1, cv2.LINE_AA)
        cv2.putText(frame, id_text, (cx2 + 10, card_y + 42), self.font, 0.44, (0, 255, 255), 1, cv2.LINE_AA)

        # Card 3: REFERENCE STATUS
        cx3 = 10 + 2 * (card_w + 10)
        ref_has = ref_data is not None
        ref_color = (0, 255, 255) if ref_has else (160, 160, 160)
        cv2.rectangle(frame, (cx3, card_y), (cx3 + card_w, card_y + card_h), ref_color, 1)
        ref_st_str = "REFERENCE: SAVED" if ref_has else "REFERENCE: NOT SAVED"
        ref_sub_str = f"ID: {ref_data.get('marker_id')}" if ref_has else "[SPACE] to Record"
        cv2.putText(frame, ref_st_str, (cx3 + 10, card_y + 20), self.font, 0.45, ref_color, 1, cv2.LINE_AA)
        cv2.putText(frame, ref_sub_str, (cx3 + 10, card_y + 42), self.font, 0.44, (200, 200, 200), 1, cv2.LINE_AA)

        # Card 4: CALIBRATION STATUS & REPROJECTION ERROR
        cx4 = 10 + 3 * (card_w + 10)
        cal_color = (0, 255, 0) if calibration_loaded else (0, 165, 255)
        cv2.rectangle(frame, (cx4, card_y), (cx4 + card_w, card_y + card_h), cal_color, 1)
        cal_st_str = "CALIB: LOADED" if calibration_loaded else "UNCALIBRATED"
        rms_str = f"RMS: {rms_error:.4f} px" if rms_error is not None else "RMS: N/A"
        cv2.putText(frame, cal_st_str, (cx4 + 10, card_y + 20), self.font, 0.45, cal_color, 1, cv2.LINE_AA)
        cv2.putText(frame, rms_str, (cx4 + 10, card_y + 42), self.font, 0.44, (255, 255, 0), 1, cv2.LINE_AA)

        # ------------------- 2. THREE TELEMETRY PANELS (y: 75 to 345) -------------------
        pan_w = (w - 40) // 3
        pan_h = 265
        pan_y = 75

        overlay_pan = frame.copy()
        for i in range(3):
            px = 10 + i * (pan_w + 10)
            cv2.rectangle(overlay_pan, (px, pan_y), (px + pan_w, pan_y + pan_h), (12, 12, 12), -1)
        cv2.addWeighted(overlay_pan, 0.82, frame, 0.18, 0, frame)

        # ------------------- PANEL 1: LIVE POSE -------------------
        px1 = 10
        p1_border = (0, 255, 0) if is_detected else (0, 0, 255)
        cv2.rectangle(frame, (px1, pan_y), (px1 + pan_w, pan_y + pan_h), p1_border, 2)
        cv2.putText(frame, "LIVE POSE", (px1 + 15, pan_y + 28), self.font, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
        if marker_id is not None and is_detected:
            cv2.putText(frame, f"[ID: {marker_id}]", (px1 + 130, pan_y + 28), self.font, 0.48, (0, 255, 0), 1, cv2.LINE_AA)

        if is_detected and pose_data is not None:
            tx, ty, tz = pose_data["translation_mm"]
            rx, ry, rz = pose_data["rotation_deg"]

            cv2.putText(frame, "TRANSLATION (mm):", (px1 + 15, pan_y + 55), self.font, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(frame, f"X : {tx:+8.1f} mm", (px1 + 35, pan_y + 82), self.font, 0.52, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Y : {ty:+8.1f} mm", (px1 + 35, pan_y + 108), self.font, 0.52, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Z : {tz:+8.1f} mm", (px1 + 35, pan_y + 134), self.font, 0.52, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.putText(frame, "ROTATION (Euler deg):", (px1 + 15, pan_y + 168), self.font, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Rx: {rx:+6.1f} deg", (px1 + 35, pan_y + 195), self.font, 0.5, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Ry: {ry:+6.1f} deg", (px1 + 35, pan_y + 221), self.font, 0.5, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Rz: {rz:+6.1f} deg", (px1 + 35, pan_y + 247), self.font, 0.5, (255, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "⚠️ TRACKING LOST", (px1 + 25, pan_y + 110), self.font, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "Target ArUco marker is missing", (px1 + 25, pan_y + 145), self.font, 0.48, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(frame, "Point camera at physical marker...", (px1 + 25, pan_y + 175), self.font, 0.44, (160, 160, 160), 1, cv2.LINE_AA)

        # ------------------- PANEL 2: REFERENCE POSE -------------------
        px2 = 10 + (pan_w + 10)
        p2_border = (0, 255, 255) if ref_has else (100, 100, 100)
        cv2.rectangle(frame, (px2, pan_y), (px2 + pan_w, pan_y + pan_h), p2_border, 2)
        cv2.putText(frame, "REFERENCE POSE", (px2 + 15, pan_y + 28), self.font, 0.55, (0, 255, 255), 2, cv2.LINE_AA)

        if ref_has and ref_data is not None:
            r_x = ref_data["X"]
            r_y = ref_data["Y"]
            r_z = ref_data["Z"]
            r_rx = ref_data["Rx"]
            r_ry = ref_data["Ry"]
            r_rz = ref_data["Rz"]

            cv2.putText(frame, "POSITION (mm):", (px2 + 15, pan_y + 55), self.font, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(frame, f"X : {r_x:+8.1f} mm", (px2 + 35, pan_y + 82), self.font, 0.52, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Y : {r_y:+8.1f} mm", (px2 + 35, pan_y + 108), self.font, 0.52, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Z : {r_z:+8.1f} mm", (px2 + 35, pan_y + 134), self.font, 0.52, (0, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(frame, "ORIENTATION (deg):", (px2 + 15, pan_y + 168), self.font, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Rx: {r_rx:+6.1f} deg", (px2 + 35, pan_y + 195), self.font, 0.5, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Ry: {r_ry:+6.1f} deg", (px2 + 35, pan_y + 221), self.font, 0.5, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Rz: {r_rz:+6.1f} deg", (px2 + 35, pan_y + 247), self.font, 0.5, (255, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "REFERENCE NOT SAVED", (px2 + 25, pan_y + 110), self.font, 0.55, (0, 165, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "Press [SPACE] when marker visible", (px2 + 25, pan_y + 145), self.font, 0.46, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(frame, "to set baseline reference pose.", (px2 + 25, pan_y + 175), self.font, 0.44, (160, 160, 160), 1, cv2.LINE_AA)

        # ------------------- PANEL 3: RELATIVE CHANGE -------------------
        px3 = 10 + 2 * (pan_w + 10)
        p3_border = (0, 255, 0) if (is_detected and delta_data is not None) else (100, 100, 100)
        cv2.rectangle(frame, (px3, pan_y), (px3 + pan_w, pan_y + pan_h), p3_border, 2)
        cv2.putText(frame, "RELATIVE CHANGE", (px3 + 15, pan_y + 28), self.font, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

        if is_detected and delta_data is not None:
            dx = delta_data["delta_X_mm"]
            dy = delta_data["delta_Y_mm"]
            dz = delta_data["delta_Z_mm"]
            drx = delta_data["delta_Rx_deg"]
            dry = delta_data["delta_Ry_deg"]
            drz = delta_data["delta_Rz_deg"]

            cv2.putText(frame, "TRANSLATION DELTAS:", (px3 + 15, pan_y + 55), self.font, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(frame, f"dX : {dx:+8.1f} mm", (px3 + 35, pan_y + 82), self.font, 0.52, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"dY : {dy:+8.1f} mm", (px3 + 35, pan_y + 108), self.font, 0.52, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"dZ : {dz:+8.1f} mm", (px3 + 35, pan_y + 134), self.font, 0.52, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.putText(frame, "ROTATION DELTAS:", (px3 + 15, pan_y + 168), self.font, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(frame, f"dRx: {drx:+6.1f} deg", (px3 + 35, pan_y + 195), self.font, 0.5, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"dRy: {dry:+6.1f} deg", (px3 + 35, pan_y + 221), self.font, 0.5, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"dRz: {drz:+6.1f} deg", (px3 + 35, pan_y + 247), self.font, 0.5, (255, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "DELTAS UNAVAILABLE", (px3 + 25, pan_y + 110), self.font, 0.52, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "Requires active ArUco tracking", (px3 + 25, pan_y + 145), self.font, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(frame, "and saved baseline reference pose.", (px3 + 25, pan_y + 175), self.font, 0.44, (160, 160, 160), 1, cv2.LINE_AA)

        # ------------------- 3. STAGE 8 MATRIX & STATUS BANNER (y: 355 to 470) -------------------
        mat_w = pan_w * 2 + 10
        mat_h = 115
        mat_y = 355

        overlay_bottom = frame.copy()
        cv2.rectangle(overlay_bottom, (10, mat_y), (10 + mat_w, mat_y + mat_h), (10, 10, 10), -1)
        
        banner_x = 10 + mat_w + 10
        banner_w = w - banner_x - 10
        cv2.rectangle(overlay_bottom, (banner_x, mat_y), (banner_x + banner_w, mat_y + mat_h), (15, 15, 15), -1)
        
        cv2.addWeighted(overlay_bottom, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (10, mat_y), (10 + mat_w, mat_y + mat_h), (0, 255, 255), 1)
        cv2.rectangle(frame, (banner_x, mat_y), (banner_x + banner_w, mat_y + mat_h), (0, 255, 0), 1)

        # Stage 8 Relative SE(3) Matrix Display
        cv2.putText(frame, "SE(3) RELATIVE HOMOGENEOUS TRANSFORMATION (T_ref_to_curr)", 
                    (20, mat_y + 20), self.font, 0.42, (0, 255, 255), 1, cv2.LINE_AA)

        if delta_data is not None and "T_reference_to_current" in delta_data:
            T_mat = delta_data["T_reference_to_current"]
            row_ys = [mat_y + 40, mat_y + 60, mat_y + 80, mat_y + 100]
            for r in range(4):
                r_str = f"[{T_mat[r,0]:+7.3f} {T_mat[r,1]:+7.3f} {T_mat[r,2]:+7.3f} | {T_mat[r,3]:+8.2f}]"
                color = (0, 255, 0) if r < 3 else (200, 200, 200)
                cv2.putText(frame, r_str, (20, row_ys[r]), self.font, 0.42, color, 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "[ Identity / Reference matrix unavailable ]", (20, mat_y + 60), self.font, 0.45, (140, 140, 140), 1, cv2.LINE_AA)

        # Active Mode & Status Message Banner
        cv2.putText(frame, f"ACTIVE MODE: {active_mode.upper()}", (banner_x + 15, mat_y + 25), self.font, 0.52, (0, 255, 0), 2, cv2.LINE_AA)
        if status_message:
            cv2.putText(frame, f"Info: {status_message[:45]}", (banner_x + 15, mat_y + 60), self.font, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "System operational. Real camera active.", (banner_x + 15, mat_y + 60), self.font, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

        # ------------------- 4. BOTTOM CONTROLS BAR (y: h - 45 to h - 10) -------------------
        cv2.rectangle(frame, (10, h - 45), (w - 10, h - 10), (0, 0, 0), -1)
        controls_str = "[SPACE] Save Ref  |  [R] Reset Ref  |  [C] Calibrate Mode  |  [T] Tracking Mode  |  [A] Accuracy Test  |  [Q] Quit"
        cv2.putText(frame, controls_str, (20, h - 20), self.font, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        return frame

    def render_accuracy_eval_hud(
        self,
        frame: np.ndarray,
        is_detected: bool,
        expected_movement_mm: Tuple[float, float, float],
        sample_count: int,
        stats: Optional[Dict[str, Any]] = None,
        status_message: str = ""
    ) -> np.ndarray:
        """
        Renders Stage 7 physical accuracy evaluation HUD overlay.
        """
        overlay = frame.copy()
        h, w = frame.shape[:2]

        panel_w = 480
        panel_h = 240
        cv2.rectangle(overlay, (15, 15), (15 + panel_w, 15 + panel_h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

        border_color = (0, 255, 0) if is_detected else (0, 0, 255)
        cv2.rectangle(frame, (15, 15), (15 + panel_w, 15 + panel_h), border_color, 2)

        # Title Banner
        cv2.putText(frame, "STAGE 7: PHYSICAL ACCURACY EVALUATION", (30, 42), self.font, 0.52, (0, 255, 255), 2, cv2.LINE_AA)

        exp_x, exp_y, exp_z = expected_movement_mm
        cv2.putText(frame, f"Expected Movement : dX={exp_x:+.1f} dY={exp_y:+.1f} dZ={exp_z:+.1f} mm",
                    (30, 68), self.font, 0.44, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.putText(frame, f"Collected Samples : {sample_count} frames from real webcam",
                    (30, 92), self.font, 0.46, (255, 255, 255), 1, cv2.LINE_AA)

        if stats is not None and stats["num_samples"] > 0:
            mx, my, mz = stats["mean_measured_mm"]
            ax, ay, az = stats["abs_error_mm"]
            sx, sy, sz = stats["std_dev_mm"]
            euc_err = stats["euclidean_error_mean_mm"]
            pct_3d = stats["pct_error_3d"]

            cv2.putText(frame, f"Mean Measured     : dX={mx:+.1f} dY={my:+.1f} dZ={mz:+.1f} mm",
                        (30, 120), self.font, 0.45, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Absolute Error    : dX={ax:.1f} dY={ay:.1f} dZ={az:.1f} mm",
                        (30, 145), self.font, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, f"3D Mean Error     : {euc_err:.2f} mm ({pct_3d:.1f}%)",
                        (30, 170), self.font, 0.52, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Standard Dev      : sX={sx:.1f} sY={sy:.1f} sZ={sz:.1f} mm",
                        (30, 195), self.font, 0.44, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Min/Max Error     : Min={stats['euclidean_error_min_mm']:.1f} mm | Max={stats['euclidean_error_max_mm']:.1f} mm",
                        (30, 220), self.font, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "STATUS: ACCUMULATING REAL WEBCAM SAMPLES...", (30, 130), self.font, 0.48, (0, 165, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, "Move physical object by expected displacement,", (30, 160), self.font, 0.44, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(frame, "then press [SPACE] to capture evaluation samples.", (30, 185), self.font, 0.44, (200, 200, 200), 1, cv2.LINE_AA)

        # Status Message Banner
        if status_message:
            cv2.rectangle(frame, (15, 265), (15 + panel_w, 295), (0, 120, 0), -1)
            cv2.putText(frame, status_message, (25, 285), self.font, 0.46, (255, 255, 255), 1, cv2.LINE_AA)

        # Bottom Controls Bar
        cv2.rectangle(frame, (10, h - 40), (w - 10, h - 10), (0, 0, 0), -1)
        cv2.putText(
            frame,
            "[SPACE] Capture Samples  |  [S] Save CSV Report  |  [R] Reset  |  [Q] Quit",
            (20, h - 18),
            self.font,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        return frame







