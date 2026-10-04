"""
Reference Manager module for recording baseline object pose and computing relative 6-DoF deltas.
"""

from typing import Dict, Optional, Tuple
import numpy as np
from tracking.transformations import SpatialTransformations


class ReferenceManager:
    """
    Manages baseline reference pose locking and dynamic relative spatial delta computations.
    """

    def __init__(self) -> None:
        self.ref_transform: Optional[np.ndarray] = None

    def set_reference_pose(self, rvec: np.ndarray, tvec: np.ndarray) -> None:
        """Locks current observed pose as the origin reference frame SE(3)."""
        self.ref_transform = SpatialTransformations.rvec_tvec_to_matrix(rvec, tvec)

    def clear_reference(self) -> None:
        """Clears stored baseline reference pose."""
        self.ref_transform = None

    def calculate_delta(
        self,
        curr_rvec: np.ndarray,
        curr_tvec: np.ndarray
    ) -> Optional[Dict[str, Tuple[float, float, float]]]:
        """
        Computes 6-DoF translation delta (dX, dY, dZ) and rotation delta (dRx, dRy, dRz)
        relative to stored baseline pose.
        """
        if self.ref_transform is None:
            return None

        curr_transform = SpatialTransformations.rvec_tvec_to_matrix(curr_rvec, curr_tvec)
        # T_relative = T_ref_inv * T_curr
        ref_inv = np.linalg.inv(self.ref_transform)
        rel_transform = ref_inv @ curr_transform

        rel_rvec, rel_tvec = SpatialTransformations.matrix_to_rvec_tvec(rel_transform)
        dRx, dRy, dRz = SpatialTransformations.rvec_to_euler_angles(rel_rvec)
        dX, dY, dZ = rel_tvec.flatten()

        return {
            "translation_delta": (float(dX), float(dY), float(dZ)),
            "rotation_delta": (float(dRx), float(dRy), float(dRz))
        }
