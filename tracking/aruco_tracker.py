"""
ArUco marker detection and tracking module.
"""

from typing import List, Tuple, Optional
import cv2
import numpy as np


class ArUcoTracker:
    """
    Detects physical ArUco markers in camera frames.
    """

    def __init__(self, dictionary_id: int = cv2.aruco.DICT_6X6_250) -> None:
        self.dictionary_id = dictionary_id
        # Compatibility handling across OpenCV versions
        if hasattr(cv2.aruco, "getPredefinedDictionary"):
            self.dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
            self.parameters = cv2.aruco.DetectorParameters()
            self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)
        else:
            self.dictionary = cv2.aruco.Dictionary_get(dictionary_id)
            self.parameters = cv2.aruco.DetectorParameters_create()
            self.detector = None

    def detect_markers(self, frame: np.ndarray) -> Tuple[List[np.ndarray], Optional[np.ndarray], List[np.ndarray]]:
        """
        Detects ArUco marker corners and IDs in raw camera frame.
        Returns (corners, ids, rejected_points).
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.detector is not None:
            corners, ids, rejected = self.detector.detectMarkers(gray)
        else:
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.dictionary, parameters=self.parameters)
        return corners, ids, rejected
