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

