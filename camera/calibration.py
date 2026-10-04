"""
Camera Calibration module for computing intrinsic matrices and lens distortion
using physical printed chessboard patterns.
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np

logger = logging.getLogger("CameraCalibrator")


class CameraCalibrator:
    """
    Handles physical chessboard pattern detection, sub-pixel refinement,
    OpenCV intrinsic calibration calculation, and JSON persistence.
    """

    def __init__(self, pattern_size: Tuple[int, int] = (9, 6), square_size_mm: float = 25.0) -> None:
        """
        :param pattern_size: Inner corners (columns, rows) e.g. (9, 6) for 10x7 square chessboard.
        :param square_size_mm: Physical size of each square in millimeters.
        """
        self.pattern_size = pattern_size
        self.square_size_mm = square_size_mm
        # Square size in meters for physical 3D world coordinate calculations
        self.square_size_m = square_size_mm / 1000.0

        # Sub-pixel corner refinement criteria
        self.subpixel_criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001
        )

        # 3D Object points template for a single flat chessboard frame (Z = 0)
        self._single_objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        self._single_objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        self._single_objp *= self.square_size_m

        # Collected frame buffers
        self.obj_points: List[np.ndarray] = []  # 3D points in real world space
        self.img_points: List[np.ndarray] = []  # 2D points in image plane
        self.image_size: Optional[Tuple[int, int]] = None

        # Calibration outputs
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None
        self.rvecs: Optional[List[np.ndarray]] = None
        self.tvecs: Optional[List[np.ndarray]] = None
        self.rms_reprojection_error: Optional[float] = None
        self.is_calibrated: bool = False

    def detect_chessboard(self, frame: np.ndarray) -> Tuple[bool, Optional[np.ndarray], np.ndarray]:
        """
        Detects and refines chessboard corners in a raw camera frame.
        Returns: (found_flag, refined_corners_array, visualization_frame).
        """
        display_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.image_size = (frame.shape[1], frame.shape[0])

        # Detect inner corners
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray, self.pattern_size, flags)

        refined_corners = None
        if found and corners is not None:
            # Sub-pixel corner refinement
            refined_corners = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), self.subpixel_criteria
            )
            # Draw color-coded corner grid on display frame
            cv2.drawChessboardCorners(display_frame, self.pattern_size, refined_corners, found)

        return found, refined_corners, display_frame

    def add_calibration_frame(self, corners: np.ndarray) -> bool:
        """
        Stores current detected corners and corresponding 3D object points.
        """
        if corners is None:
            return False
        self.obj_points.append(self._single_objp.copy())
        self.img_points.append(corners.copy())
        logger.info(f"Captured calibration frame #{len(self.img_points)}")
        return True

    def reset_frames(self) -> None:
        """Clears all captured calibration frames and resets calibration state."""
        self.obj_points.clear()
        self.img_points.clear()
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvecs = None
        self.tvecs = None
        self.rms_reprojection_error = None
        self.is_calibrated = False
        logger.info("Reset all captured calibration frames.")

    def calibrate(self) -> Tuple[bool, str]:
        """
        Executes OpenCV camera calibration using accumulated physical chessboard frames.
        Returns: (success_flag, status_message).
        """
        num_frames = len(self.img_points)
        if num_frames < 5:
            msg = f"Insufficient views! Need at least 5 frames (Currently captured: {num_frames})."
            logger.warning(msg)
            return False, msg

        if self.image_size is None:
            return False, "Image size not set."

        logger.info(f"Calculating camera calibration matrix from {num_frames} physical views...")
        try:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                self.obj_points,
                self.img_points,
                self.image_size,
                None,
                None
            )
            self.rms_reprojection_error = float(ret)
            self.camera_matrix = mtx
            self.dist_coeffs = dist
            self.rvecs = rvecs
            self.tvecs = tvecs
            self.is_calibrated = True

            logger.info(f"✅ Calibration successful! RMS Reprojection Error: {self.rms_reprojection_error:.4f} px")
            return True, f"Calibration Successful! RMS Error: {self.rms_reprojection_error:.4f} px"
        except Exception as e:
            logger.error(f"Error during cv2.calibrateCamera: {e}")
            return False, f"Calibration Error: {str(e)}"

    def save_calibration(self, filepath: str = "calibration_data/camera_calibration.json") -> Tuple[bool, str]:
        """
        Saves calibration matrix and lens distortion parameters to JSON file.
        """
        if not self.is_calibrated or self.camera_matrix is None or self.dist_coeffs is None:
            return False, "Cannot save: Camera is not calibrated yet."

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coeffs": self.dist_coeffs.tolist(),
            "rms_reprojection_error": self.rms_reprojection_error,
            "pattern_size_inner_corners": list(self.pattern_size),
            "square_size_mm": self.square_size_mm,
            "square_size_m": self.square_size_m,
            "image_width": self.image_size[0] if self.image_size else 0,
            "image_height": self.image_size[1] if self.image_size else 0,
            "num_captured_frames": len(self.img_points),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
            logger.info(f"Saved calibration parameters to {filepath}")
            return True, f"Saved to {filepath}"
        except Exception as e:
            logger.error(f"Failed to save calibration file: {e}")
            return False, f"Save failed: {str(e)}"

    def load_calibration(self, filepath: str = "calibration_data/camera_calibration.json") -> bool:
        """
        Loads camera matrix and distortion parameters from JSON file.
        """
        if not os.path.exists(filepath):
            logger.warning(f"Calibration file not found at {filepath}")
            return False

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.camera_matrix = np.array(data["camera_matrix"], dtype=np.float64)
            self.dist_coeffs = np.array(data["dist_coeffs"], dtype=np.float64)
            self.rms_reprojection_error = data.get("rms_reprojection_error")
            self.square_size_mm = data.get("square_size_mm", self.square_size_mm)
            self.square_size_m = self.square_size_mm / 1000.0
            self.is_calibrated = True
            logger.info(f"Loaded valid camera calibration matrix from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to load calibration file {filepath}: {e}")
            return False

