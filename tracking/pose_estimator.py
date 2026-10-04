"""
6-DoF Pose Estimation module using PnP solvers on calibrated camera feed.
"""

from typing import Dict, Tuple, Optional
import cv2
import numpy as np


class PoseEstimator:
    """
    Computes 3D translation vectors (tvec) and rotation vectors (rvec) from detected 2D marker corners
    and calibrated camera parameters.
    """

    def __init__(self, marker_length: float = 0.05) -> None:
        self.marker_length = marker_length  # Side length of physical ArUco marker in meters

    def estimate_pose_single_marker(
        self,
        corners: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray
    ) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Solves Perspective-n-Point (PnP) for a single physical ArUco marker corner set.
        Returns (success, rvec, tvec).
        """
        half_l = self.marker_length / 2.0
        obj_points = np.array([
            [-half_l,  half_l, 0.0],
            [ half_l,  half_l, 0.0],
            [ half_l, -half_l, 0.0],
            [-half_l, -half_l, 0.0]
        ], dtype=np.float32)

        img_points = corners.reshape((4, 2)).astype(np.float32)

        success, rvec, tvec = cv2.solvePnP(
            obj_points,
            img_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        return success, rvec, tvec
