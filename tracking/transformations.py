"""
Spatial Transformation Utilities module for SE(3) matrix manipulations,
Rodrigues conversions, and Euler angle rotations (Rx, Ry, Rz).
"""

from typing import Tuple
import cv2
import numpy as np


class SpatialTransformations:
    """
    3D Rigid-body spatial transformation helper functions.
    """

    @staticmethod
    def rvec_tvec_to_matrix(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        """Converts rotation vector and translation vector to 4x4 homogeneous matrix SE(3)."""
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4, dtype=np.float64)
        T[0:3, 0:3] = R
        T[0:3, 3] = tvec.flatten()
        return T

    @staticmethod
    def matrix_to_rvec_tvec(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extracts rotation vector (rvec) and translation vector (tvec) from 4x4 SE(3) matrix."""
        R = T[0:3, 0:3]
        tvec = T[0:3, 3].reshape((3, 1))
        rvec, _ = cv2.Rodrigues(R)
        return rvec, tvec

    @staticmethod
    def rvec_to_euler_angles(rvec: np.ndarray) -> Tuple[float, float, float]:
        """
        Converts Rodrigues rotation vector to Euler angles (Rx, Ry, Rz / Pitch, Yaw, Roll) in degrees.
        """
        R, _ = cv2.Rodrigues(rvec)
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
