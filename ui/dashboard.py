"""
Real-time UI Dashboard Overlay renderer for OpenCV frames.
"""

from typing import Dict, Optional, Tuple
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
