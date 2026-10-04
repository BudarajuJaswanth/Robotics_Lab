"""
ArUco marker detection, 2D corner extraction, and 3D coordinate axis drawing module.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import cv2
import numpy as np

logger = logging.getLogger("ArUcoTracker")

# Map dictionary name strings to OpenCV predefined dictionary constants
ARUCO_DICTIONARY_MAP: Dict[str, int] = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}


class ArUcoTracker:
    """
    Detects physical ArUco markers, extracts 2D corner coordinates,
    and draws 3D coordinate axes using calibrated camera matrices.
    """

    def __init__(
        self,
        dictionary_name: str = "DICT_6X6_250",
        marker_size_mm: float = 50.0
    ) -> None:
        """
        :param dictionary_name: ArUco dictionary string (e.g., 'DICT_6X6_250', 'DICT_4X4_50').
        :param marker_size_mm: Physical side length of printed ArUco marker in millimeters.
        """
        self.dictionary_name = dictionary_name.upper()
        self.marker_size_mm = marker_size_mm
        self.marker_size_m = marker_size_mm / 1000.0

        if self.dictionary_name not in ARUCO_DICTIONARY_MAP:
            logger.warning(f"Unknown dictionary '{self.dictionary_name}'. Defaulting to 'DICT_6X6_250'.")
            self.dictionary_name = "DICT_6X6_250"

        self.dictionary_id = ARUCO_DICTIONARY_MAP[self.dictionary_name]

        # Initialize dictionary & detector across OpenCV versions
        if hasattr(cv2.aruco, "getPredefinedDictionary"):
            self.dictionary = cv2.aruco.getPredefinedDictionary(self.dictionary_id)
            self.parameters = cv2.aruco.DetectorParameters()
            self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)
        else:
            self.dictionary = cv2.aruco.Dictionary_get(self.dictionary_id)
            self.parameters = cv2.aruco.DetectorParameters_create()
            self.detector = None

    def detect_markers(
        self, frame: np.ndarray
    ) -> Tuple[bool, List[int], List[np.ndarray], Dict[str, Any]]:
        """
        Detects ArUco markers in raw physical camera frame.
        Returns: (is_detected, list_of_ids, list_of_4_corner_arrays, metadata_dict).
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.detector is not None:
            corners, ids, rejected = self.detector.detectMarkers(gray)
        else:
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.dictionary, parameters=self.parameters)

        if ids is not None and len(ids) > 0:
            flat_ids = [int(i[0]) for i in ids]
            corner_list = [c for c in corners]
            meta = {
                "count": len(flat_ids),
                "ids": flat_ids,
                "status": "DETECTED"
            }
            return True, flat_ids, corner_list, meta
        else:
            return False, [], [], {"count": 0, "ids": [], "status": "NOT DETECTED"}

    def draw_corners(self, frame: np.ndarray, corners: List[np.ndarray], ids: List[int]) -> np.ndarray:
        """Draws the 4 detected marker boundary corners and green outline box."""
        if not corners or not ids:
            return frame

        np_ids = np.array([[i] for i in ids], dtype=np.int32)
        cv2.aruco.drawDetectedMarkers(frame, corners, np_ids)
        return frame

    def draw_axes_and_pose(
        self,
        frame: np.ndarray,
        corners: List[np.ndarray],
        ids: List[int],
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Solves PnP for each detected marker and draws 3D coordinate axes (X: Red, Y: Green, Z: Blue).
        Returns: (annotated_frame, list_of_poses).
        """
        poses = []
        half_l = self.marker_size_m / 2.0
        obj_points = np.array([
            [-half_l,  half_l, 0.0],
            [ half_l,  half_l, 0.0],
            [ half_l, -half_l, 0.0],
            [-half_l, -half_l, 0.0]
        ], dtype=np.float32)

        for i, corner in enumerate(corners):
            marker_id = ids[i]
            img_points = corner.reshape((4, 2)).astype(np.float32)

            success, rvec, tvec = cv2.solvePnP(
                obj_points,
                img_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )

            if success:
                # Draw 3D axes (length equal to marker size)
                axis_length = self.marker_size_m
                if hasattr(cv2, "drawFrameAxes"):
                    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, axis_length)
                elif hasattr(cv2.aruco, "drawAxis"):
                    cv2.aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvec, tvec, axis_length)

                # Draw ID text label near corner
                c0 = img_points[0]
                cv2.putText(
                    frame,
                    f"ID: {marker_id}",
                    (int(c0[0]), int(c0[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )

                poses.append({
                    "id": marker_id,
                    "rvec": rvec,
                    "tvec": tvec,
                    "corners": img_points
                })

        return frame, poses

