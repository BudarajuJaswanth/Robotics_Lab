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
        status_message: str = ""
    ) -> np.ndarray:
        """
        Renders Stage 6 HUD telemetry with:
        - Status badges: TRACKING vs TRACKING LOST | REFERENCE SAVED vs REFERENCE NOT SAVED
        - REFERENCE panel (X, Y, Z mm, Rx, Ry, Rz deg)
        - CURRENT panel (X, Y, Z mm, Rx, Ry, Rz deg)
        - CHANGE panel (dX, dY, dZ mm)
        - ROTATION CHANGE panel (dRx, dRy, dRz deg)
        """
        overlay = frame.copy()
        h, w = frame.shape[:2]

        panel_w = 380
        panel_h = 240

        # Draw Left Panel (Current Pose)
        cv2.rectangle(overlay, (15, 55), (15 + panel_w, 55 + panel_h), (15, 15, 15), -1)
        
        # Draw Right Panel (Reference Pose & Delta Comparison)
        ref_x = w - panel_w - 15
        cv2.rectangle(overlay, (ref_x, 55), (ref_x + panel_w, 55 + panel_h), (15, 15, 15), -1)

        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

        # ------------------- TOP STATUS BADGES -------------------
        # Badge 1: Tracking Status
        if is_detected:
            cv2.rectangle(frame, (15, 12), (185, 45), (0, 160, 0), -1)
            cv2.putText(frame, "TRACKING", (35, 35), self.font, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.rectangle(frame, (15, 12), (210, 45), (0, 0, 200), -1)
            cv2.putText(frame, "TRACKING LOST", (25, 35), self.font, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        # Badge 2: Reference Status
        if ref_data is not None:
            cv2.rectangle(frame, (225, 12), (450, 45), (0, 140, 180), -1)
            cv2.putText(frame, "REFERENCE SAVED", (235, 35), self.font, 0.52, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.rectangle(frame, (225, 12), (480, 45), (0, 120, 200), -1)
            cv2.putText(frame, "REFERENCE NOT SAVED", (235, 35), self.font, 0.52, (255, 255, 255), 2, cv2.LINE_AA)

        # ------------------- LEFT PANEL: CURRENT POSE -------------------
        left_border = (0, 255, 0) if is_detected else (0, 0, 255)
        cv2.rectangle(frame, (15, 55), (15 + panel_w, 55 + panel_h), left_border, 2)

        if is_detected and pose_data is not None:
            tx, ty, tz = pose_data["translation_mm"]
            rx, ry, rz = pose_data["rotation_deg"]

            cv2.putText(frame, f"CURRENT POSE [ID: {marker_id}]", (30, 80), self.font, 0.52, (0, 255, 255), 2, cv2.LINE_AA)
            
            cv2.putText(frame, "TRANSLATION (mm):", (30, 105), self.font, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(frame, f"X: {tx:+8.1f} mm", (50, 128), self.font, 0.52, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Y: {ty:+8.1f} mm", (50, 150), self.font, 0.52, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Z: {tz:+8.1f} mm", (50, 172), self.font, 0.52, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.putText(frame, "ROTATION (Euler deg):", (30, 198), self.font, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Rx: {rx:+6.1f} deg", (50, 220), self.font, 0.5, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Ry: {ry:+6.1f} deg", (50, 242), self.font, 0.5, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Rz: {rz:+6.1f} deg", (50, 264), self.font, 0.5, (255, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "CURRENT POSE", (30, 85), self.font, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "⚠️ TRACKING LOST", (30, 130), self.font, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "Target ArUco marker is missing!", (30, 165), self.font, 0.48, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(frame, "Point camera at physical marker...", (30, 190), self.font, 0.45, (160, 160, 160), 1, cv2.LINE_AA)

        # ------------------- RIGHT PANEL: REFERENCE & DELTAS -------------------
        right_border = (0, 255, 255) if ref_data is not None else (100, 100, 100)
        cv2.rectangle(frame, (ref_x, 55), (ref_x + panel_w, 55 + panel_h), right_border, 2)

        if ref_data is not None:
            cv2.putText(frame, f"REFERENCE POSE [ID: {ref_data.get('marker_id')}]", (ref_x + 15, 80), self.font, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
            
            # Reference Position Values
            rx_val = ref_data["X"]
            ry_val = ref_data["Y"]
            rz_val = ref_data["Z"]
            cv2.putText(frame, f"REF X: {rx_val:+7.1f}  Y: {ry_val:+7.1f}  Z: {rz_val:+7.1f} mm", (ref_x + 15, 105), self.font, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

            # Delta Comparison Values (if tracking active and delta calculated)
            if is_detected and delta_data is not None:
                dx = delta_data["delta_X_mm"]
                dy = delta_data["delta_Y_mm"]
                dz = delta_data["delta_Z_mm"]
                drx = delta_data["delta_Rx_deg"]
                dry = delta_data["delta_Ry_deg"]
                drz = delta_data["delta_Rz_deg"]

                cv2.putText(frame, "CHANGE (TRANSLATION DELTA):", (ref_x + 15, 135), self.font, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(frame, f"dX: {dx:+8.1f} mm", (ref_x + 35, 158), self.font, 0.52, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f"dY: {dy:+8.1f} mm", (ref_x + 35, 180), self.font, 0.52, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f"dZ: {dz:+8.1f} mm", (ref_x + 35, 202), self.font, 0.52, (0, 255, 0), 2, cv2.LINE_AA)

                cv2.putText(frame, "ROTATION CHANGE (DELTA EULER):", (ref_x + 15, 226), self.font, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(frame, f"dRx: {drx:+6.1f} | dRy: {dry:+6.1f} | dRz: {drz:+6.1f} deg", (ref_x + 15, 250), self.font, 0.44, (255, 255, 0), 1, cv2.LINE_AA)
            else:
                cv2.putText(frame, "CHANGE (TRANSLATION DELTA):", (ref_x + 15, 140), self.font, 0.45, (160, 160, 160), 1, cv2.LINE_AA)
                cv2.putText(frame, "TRACKING LOST (Delta Unavailable)", (ref_x + 15, 175), self.font, 0.48, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "REFERENCE POSE", (ref_x + 15, 85), self.font, 0.55, (160, 160, 160), 2, cv2.LINE_AA)
            cv2.putText(frame, "REFERENCE NOT SAVED", (ref_x + 15, 130), self.font, 0.52, (0, 165, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "Press [SPACE] when marker is visible", (ref_x + 15, 165), self.font, 0.46, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(frame, "to record baseline reference pose.", (ref_x + 15, 190), self.font, 0.44, (160, 160, 160), 1, cv2.LINE_AA)

        # ------------------- STAGE 8: 4x4 HOMOGENEOUS MATRIX PANEL -------------------
        if delta_data is not None and "T_reference_to_current" in delta_data:
            T_mat = delta_data["T_reference_to_current"]
            mat_w = panel_w
            mat_h = 115
            mat_y = 55 + panel_h + 10
            
            # Left bottom overlay panel for Matrix
            cv2.rectangle(overlay, (15, mat_y), (15 + mat_w, mat_y + mat_h), (10, 10, 10), -1)
            cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
            cv2.rectangle(frame, (15, mat_y), (15 + mat_w, mat_y + mat_h), (0, 255, 255), 1)

            cv2.putText(frame, "STAGE 8: 4x4 RELATIVE TRANSFORMATION (T_ref_to_curr)", 
                        (25, mat_y + 20), self.font, 0.42, (0, 255, 255), 1, cv2.LINE_AA)

            row_ys = [mat_y + 40, mat_y + 60, mat_y + 80, mat_y + 100]
            for r in range(4):
                r_str = f"[{T_mat[r,0]:+7.3f} {T_mat[r,1]:+7.3f} {T_mat[r,2]:+7.3f} | {T_mat[r,3]:+8.2f}]"
                color = (0, 255, 0) if r < 3 else (200, 200, 200)
                cv2.putText(frame, r_str, (25, row_ys[r]), self.font, 0.42, color, 1, cv2.LINE_AA)

        # Status Message Banner (bottom right / left)
        if status_message:
            msg_bg = (0, 0, 180) if "Cannot" in status_message or "failed" in status_message else (0, 120, 0)
            cv2.rectangle(frame, (ref_x, 55 + panel_h + 10), (ref_x + panel_w, 55 + panel_h + 45), msg_bg, -1)
            cv2.putText(frame, status_message, (ref_x + 10, 55 + panel_h + 32), self.font, 0.44, (255, 255, 255), 1, cv2.LINE_AA)

        # Bottom Controls Bar
        cv2.rectangle(frame, (10, h - 40), (w - 10, h - 10), (0, 0, 0), -1)
        cv2.putText(
            frame,
            "[SPACE] Record Baseline Reference  |  [R] Reset Reference  |  [Q] Quit",
            (20, h - 18),
            self.font,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

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







