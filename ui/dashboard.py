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
        pose_data: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Renders live raw 6-DoF pose telemetry panel:
        TRANSLATION: X, Y, Z in millimeters
        ROTATION   : Rx (Pitch), Ry (Yaw), Rz (Roll) in degrees
        """
        overlay = frame.copy()
        h, w = frame.shape[:2]

        panel_w = 420
        panel_h = 240
        cv2.rectangle(overlay, (15, 15), (15 + panel_w, 15 + panel_h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        border_color = (0, 255, 0) if is_detected else (0, 165, 255)
        cv2.rectangle(frame, (15, 15), (15 + panel_w, 15 + panel_h), border_color, 2)

        if is_detected and pose_data is not None:
            tx, ty, tz = pose_data["translation_mm"]
            rx, ry, rz = pose_data["rotation_deg"]

            cv2.putText(frame, f"6-DoF POSE [ID: {marker_id}] (RAW UNFILTERED)", (30, 42), self.font, 0.52, (0, 255, 255), 2, cv2.LINE_AA)

            # Translation Header & Values
            cv2.putText(frame, "TRANSLATION (mm):", (30, 70), self.font, 0.48, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(frame, f"X: {tx:+8.1f} mm", (50, 95), self.font, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Y: {ty:+8.1f} mm", (50, 120), self.font, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Z: {tz:+8.1f} mm", (50, 145), self.font, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

            # Rotation Header & Values
            cv2.putText(frame, "ROTATION (Euler Degrees):", (30, 172), self.font, 0.48, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Rx (Pitch): {rx:+6.1f} deg", (50, 195), self.font, 0.52, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Ry (Yaw)  : {ry:+6.1f} deg", (50, 218), self.font, 0.52, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Rz (Roll) : {rz:+6.1f} deg", (50, 241), self.font, 0.52, (255, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "6-DoF POSE (RAW UNFILTERED)", (30, 45), self.font, 0.55, (0, 165, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "STATUS: ARUCO MARKER NOT DETECTED", (30, 85), self.font, 0.5, (0, 165, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, "Point laptop camera at physical ArUco target...", (30, 115), self.font, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        # Bottom bar
        cv2.rectangle(frame, (10, h - 40), (w - 10, h - 10), (0, 0, 0), -1)
        cv2.putText(frame, "[Q] / [ESC] Quit 6-DoF Pose Tracking", (20, h - 18), self.font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        return frame




