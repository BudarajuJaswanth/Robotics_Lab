"""
Real-Time 6-DoF Pose Estimation module using PnP solvers on calibrated camera feeds.

Coordinate System & Rotation Conventions:
=========================================
1. Camera Coordinate System (Right-Handed):
   - Origin (0,0,0): Pinhole optical center of the laptop camera lens.
   - +X axis: Points RIGHT across the horizontal camera sensor.
   - +Y axis: Points DOWN across the vertical camera sensor.
   - +Z axis: Points FORWARD along the optical axis into the physical 3D scene (Optical Depth / Distance).

2. Marker Local Coordinate System:
   - Origin (0,0,0): Physical center of the ArUco marker square.
   - +X axis: Points along the top edge to the right.
   - +Y axis: Points along the right edge downwards.
   - +Z axis: Points OUTWARD normal from the front printed face of the marker.

3. Euler Rotation Convention:
   - Intrinsically decomposed Euler angles (Rx, Ry, Rz) using standard X-Y-Z order:
     - Rx (Pitch): Rotation around the X axis (tilting up/down) in degrees.
     - Ry (Yaw)  : Rotation around the Y axis (panning left/right) in degrees.
     - Rz (Roll) : Rotation around the Z axis (swiveling clockwise/counterclockwise) in degrees.
"""

import logging
from typing import Dict, Tuple, Optional, Any
import cv2
import numpy as np

logger = logging.getLogger("PoseEstimator")


class PoseEstimator:
    """
    Computes 6-DoF physical translation (X, Y, Z in mm) and rotation Euler angles (Rx, Ry, Rz in degrees)
    from detected 2D marker corners using Perspective-n-Point (PnP) solvers.
    """

    def __init__(self, marker_size_mm: float = 50.0) -> None:
        """
        :param marker_size_mm: Physical side length of printed ArUco marker square in millimeters.
        """
        self.marker_size_mm = marker_size_mm
        self.marker_size_m = marker_size_mm / 1000.0

        # Define 3D physical object points for square marker centered at (0, 0, 0)
        half_l = self.marker_size_m / 2.0
        self.obj_points = np.array([
            [-half_l,  half_l, 0.0],
            [ half_l,  half_l, 0.0],
            [ half_l, -half_l, 0.0],
            [-half_l, -half_l, 0.0]
        ], dtype=np.float32)

    def estimate_pose(
        self,
        corners: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Solves Perspective-n-Point (PnP) for a single physical ArUco marker corner set.
        
        Returns:
            (success_flag, pose_dict)
            pose_dict contains:
              - 'translation_mm': (X_mm, Y_mm, Z_mm)
              - 'translation_m': (X_m, Y_m, Z_m)
              - 'rotation_deg': (Rx_deg, Ry_deg, Rz_deg)
              - 'rvec': Rodrigues vector (3, 1)
              - 'tvec': Translation vector (3, 1) in meters
              - 'R_matrix': 3x3 Rotation matrix
        """
        img_points = corners.reshape((4, 2)).astype(np.float32)

        # Solve PnP using IPPE Square flag optimized for planar quad markers
        success, rvec, tvec = cv2.solvePnP(
            self.obj_points,
            img_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )

        if not success or rvec is None or tvec is None:
            return False, None

        # Extract translation in meters & convert to physical millimeters
        x_m, y_m, z_m = tvec.flatten()
        x_mm = x_m * 1000.0
        y_mm = y_m * 1000.0
        z_mm = z_m * 1000.0

        # Convert Rodrigues rotation vector to 3x3 rotation matrix R
        R, _ = cv2.Rodrigues(rvec)

        # Convert rotation matrix to Euler angles (Rx: Pitch, Ry: Yaw, Rz: Roll) in degrees
        rx_deg, ry_deg, rz_deg = self._rotation_matrix_to_euler_angles(R)

        pose_data = {
            "translation_mm": (float(x_mm), float(y_mm), float(z_mm)),
            "translation_m": (float(x_m), float(y_m), float(z_m)),
            "rotation_deg": (float(rx_deg), float(ry_deg), float(rz_deg)),
            "rvec": rvec,
            "tvec": tvec,
            "R_matrix": R,
            "euler_convention": "XYZ (Pitch, Yaw, Roll)",
            "coordinate_system": "Camera-relative (+X Right, +Y Down, +Z Optical Depth)"
        }

        return True, pose_data

    @staticmethod
    def _rotation_matrix_to_euler_angles(R: np.ndarray) -> Tuple[float, float, float]:
        """
        Decomposes 3x3 rotation matrix into intrinsic XYZ Euler angles (Pitch, Yaw, Roll) in degrees.
        """
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6

        if not singular:
            rx = np.arctan2(R[2, 1], R[2, 2])
            ry = np.arctan2(-R[2, 0], sy)
            rz = np.arctan2(R[1, 0], R[0, 0])
        else:
            rx = np.arctan2(-R[1, 2], R[1, 1])
            ry = np.arctan2(-R[2, 0], sy)
            rz = 0.0

        return np.degrees(rx), np.degrees(ry), np.degrees(rz)

