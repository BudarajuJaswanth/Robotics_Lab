"""
Reference Pose Manager module for recording real physical 6-DoF baseline poses
and persisting baseline reference frames to reference/data.json.
"""

import os
import json
import time
import logging
from typing import Dict, Optional, Tuple, Any, List
import numpy as np
from tracking.transformations import SpatialTransformations

logger = logging.getLogger("ReferenceManager")


class ReferenceManager:
    """
    Manages physical reference pose recording, persistent storage in reference/data.json,
    loading upon restart, and clearing.
    """

    def __init__(self, filepath: str = "reference/data.json") -> None:
        self.filepath = filepath
        self.ref_data: Optional[Dict[str, Any]] = None
        self.ref_transform: Optional[np.ndarray] = None

        # Automatically attempt loading reference data if file exists
        self.load_reference()

    def has_reference(self) -> bool:
        """Returns True if a valid baseline reference pose is currently loaded."""
        return self.ref_data is not None and self.ref_transform is not None

    def save_reference_pose(
        self,
        marker_id: int,
        pose_data: Dict[str, Any],
        calibration_info: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Records current real physical 6-DoF pose and saves it to reference/data.json.
        """
        if pose_data is None:
            return False, "Cannot save reference: marker pose unavailable."

        rvec = pose_data["rvec"]
        tvec = pose_data["tvec"]
        T_ref = SpatialTransformations.rvec_tvec_to_matrix(rvec, tvec)

        tx_mm, ty_mm, tz_mm = pose_data["translation_mm"]
        rx_deg, ry_deg, rz_deg = pose_data["rotation_deg"]

        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "marker_id": int(marker_id),
            "X": float(tx_mm),
            "Y": float(ty_mm),
            "Z": float(tz_mm),
            "Rx": float(rx_deg),
            "Ry": float(ry_deg),
            "Rz": float(rz_deg),
            "rvec": rvec.flatten().tolist(),
            "tvec": tvec.flatten().tolist(),
            "SE3_matrix": T_ref.tolist(),
            "camera_calibration_version": calibration_info or "Standard"
        }

        try:
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
            self.ref_data = data
            self.ref_transform = T_ref
            logger.info(f"✅ REFERENCE SAVED successfully to {self.filepath}")
            return True, "REFERENCE: SAVED"
        except Exception as e:
            logger.error(f"Failed to save reference pose to {self.filepath}: {e}")
            return False, f"Save failed: {str(e)}"

    def load_reference(self) -> bool:
        """
        Loads baseline reference pose parameters from reference/data.json if it exists.
        """

        if not os.path.exists(self.filepath):
            return False

        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.ref_data = data
            self.ref_transform = np.array(data["SE3_matrix"], dtype=np.float64)
            logger.info(f"✅ Loaded persistent reference pose from {self.filepath} (ID: {data.get('marker_id')})")
            return True
        except Exception as e:
            logger.error(f"Failed to load reference pose from {self.filepath}: {e}")
            self.ref_data = None
            self.ref_transform = None
            return False

    def clear_reference(self) -> Tuple[bool, str]:
        """Clears baseline reference pose from memory and disk."""
        self.ref_data = None
        self.ref_transform = None
        if os.path.exists(self.filepath):
            try:
                os.remove(self.filepath)
                logger.info(f"Removed reference file {self.filepath}")
            except Exception as e:
                logger.warning(f"Could not delete reference file {self.filepath}: {e}")
        return True, "REFERENCE: CLEARED"

    def calculate_delta(self, curr_pose_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Computes real physical translation deltas (dX, dY, dZ in mm) and rotation deltas
        (dRx, dRy, dRz in degrees) between current observed pose and stored baseline reference.
        """
        if not self.has_reference() or self.ref_data is None or self.ref_transform is None:
            return None

        if curr_pose_data is None:
            return None

        rvec_curr = curr_pose_data["rvec"]
        tvec_curr = curr_pose_data["tvec"]
        T_curr = SpatialTransformations.rvec_tvec_to_matrix(rvec_curr, tvec_curr)

        # SE(3) Rigid body relative transformations
        T_ref_to_curr = SpatialTransformations.compute_ref_to_curr(self.ref_transform, T_curr)
        T_curr_to_ref = SpatialTransformations.compute_curr_to_ref(self.ref_transform, T_curr)

        rel_rvec, rel_tvec = SpatialTransformations.matrix_to_rvec_tvec(T_ref_to_curr)
        dRx, dRy, dRz = SpatialTransformations.rvec_to_euler_angles(rel_rvec)

        # Coordinate difference deltas in physical millimeters
        curr_x, curr_y, curr_z = curr_pose_data["translation_mm"]
        ref_x = self.ref_data["X"]
        ref_y = self.ref_data["Y"]
        ref_z = self.ref_data["Z"]

        dX_mm = curr_x - ref_x
        dY_mm = curr_y - ref_y
        dZ_mm = curr_z - ref_z

        return {
            "delta_X_mm": float(dX_mm),
            "delta_Y_mm": float(dY_mm),
            "delta_Z_mm": float(dZ_mm),
            "delta_Rx_deg": float(dRx),
            "delta_Ry_deg": float(dRy),
            "delta_Rz_deg": float(dRz),
            "rel_matrix": T_ref_to_curr,
            "T_reference_to_current": T_ref_to_curr,
            "T_current_to_reference": T_curr_to_ref
        }


