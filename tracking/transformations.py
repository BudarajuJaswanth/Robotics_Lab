"""
Spatial Transformation Utilities module for SE(3) homogeneous matrix manipulations,
Rodrigues conversions, analytical matrix inverses, relative coordinate transformations,
and 3D point cloud mapping.

Mathematical Foundations & Conventions:
=========================================
1. Homogeneous Matrix Representation T in SE(3):
   T = [ R_{3x3}   t_{3x1} ]
       [ 0_{1x3}     1     ]
   where R in SO(3) is a 3x3 orthogonal rotation matrix (R^T = R^(-1), det(R) = +1)
   and t in R^3 is a 3x1 translation vector.

2. Analytical SE(3) Matrix Inverse:
   T^(-1) = [ R^T    -R^T * t ]
            [ 0         1     ]

3. Reference to Current Transformation (T_ref_to_curr):
   T_ref_to_curr = (T_ref)^(-1) * T_curr
   - Physical Meaning: Rigid 6-DoF spatial displacement of the current marker frame relative to baseline.
   - Point Mapping   : P_curr = T_ref_to_curr * P_ref

4. Current to Reference Transformation (T_curr_to_ref):
   T_curr_to_ref = (T_curr)^(-1) * T_ref = (T_ref_to_curr)^(-1)
   - Physical Meaning: Inverse mapping transforming coordinates in current frame back to baseline reference frame.
   - Point Mapping   : P_ref = T_curr_to_ref * P_curr
"""

import logging
from typing import Tuple, List, Optional
import cv2
import numpy as np

logger = logging.getLogger("SpatialTransformations")


class SpatialTransformations:
    """
    Complete SE(3) Lie Group rigid-body spatial transformation utility suite.
    """

    @staticmethod
    def rvec_to_rotation_matrix(rvec: np.ndarray) -> np.ndarray:
        """
        Converts 3x1 Rodrigues rotation vector to 3x3 rotation matrix R in SO(3).
        """
        R, _ = cv2.Rodrigues(rvec)
        return R

    @staticmethod
    def pose_to_homogeneous_matrix(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        """
        Combines 3x1 rotation vector (rvec) and 3x1 translation vector (tvec) into
        a 4x4 homogeneous transformation matrix T in SE(3).
        """
        R = SpatialTransformations.rvec_to_rotation_matrix(rvec)
        T = np.eye(4, dtype=np.float64)
        T[0:3, 0:3] = R
        T[0:3, 3] = tvec.flatten()
        return T

    @staticmethod
    def rvec_tvec_to_matrix(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        """Alias for pose_to_homogeneous_matrix."""
        return SpatialTransformations.pose_to_homogeneous_matrix(rvec, tvec)

    @staticmethod
    def matrix_to_rvec_tvec(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts 3x1 rotation vector (rvec) and 3x1 translation vector (tvec) from 4x4 SE(3) matrix.
        """
        R = T[0:3, 0:3]
        tvec = T[0:3, 3].reshape((3, 1))
        rvec, _ = cv2.Rodrigues(R)
        return rvec, tvec

    @staticmethod
    def inverse_transformation(T: np.ndarray) -> np.ndarray:
        """
        Computes the analytical SE(3) inverse transformation matrix:
        T^(-1) = [ R^T  -R^T * t ]
                 [  0       1    ]
        """
        R = T[0:3, 0:3]
        t = T[0:3, 3]
        R_T = R.T
        t_inv = -R_T @ t

        T_inv = np.eye(4, dtype=np.float64)
        T_inv[0:3, 0:3] = R_T
        T_inv[0:3, 3] = t_inv
        return T_inv

    @staticmethod
    def relative_transformation(T_source: np.ndarray, T_target: np.ndarray) -> np.ndarray:
        """
        Computes SE(3) relative transformation matrix: T_rel = T_source^(-1) * T_target.
        """
        T_source_inv = SpatialTransformations.inverse_transformation(T_source)
        return T_source_inv @ T_target

    @staticmethod
    def compute_ref_to_curr(T_ref: np.ndarray, T_curr: np.ndarray) -> np.ndarray:
        """
        Computes T_ref_to_curr = (T_ref)^(-1) * T_curr.
        Transforms coordinates from reference frame to current marker frame.
        """
        return SpatialTransformations.relative_transformation(T_ref, T_curr)

    @staticmethod
    def compute_curr_to_ref(T_ref: np.ndarray, T_curr: np.ndarray) -> np.ndarray:
        """
        Computes T_curr_to_ref = (T_curr)^(-1) * T_ref = (T_ref_to_curr)^(-1).
        Transforms coordinates from current marker frame back to baseline reference frame.
        """
        return SpatialTransformations.relative_transformation(T_curr, T_ref)

    @staticmethod
    def transform_3d_points(T: np.ndarray, points_3d: np.ndarray) -> np.ndarray:
        """
        Transforms Nx3 array of 3D point coordinates by 4x4 matrix T.
        Points P_transformed = (R * P^T + t)^T
        """
        pts = np.atleast_2d(points_3d)
        N = pts.shape[0]
        pts_hom = np.hstack([pts, np.ones((N, 1), dtype=pts.dtype)])  # Nx4
        pts_transformed_hom = (T @ pts_hom.T).T                        # Nx4
        return pts_transformed_hom[:, 0:3]

    @staticmethod
    def rvec_to_euler_angles(rvec: np.ndarray) -> Tuple[float, float, float]:
        """
        Converts 3x1 Rodrigues rotation vector to intrinsic XYZ Euler angles (Rx, Ry, Rz) in degrees.
        """
        R = SpatialTransformations.rvec_to_rotation_matrix(rvec)
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

