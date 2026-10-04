"""
Camera Hardware Manager module. Handles real webcam stream acquisition via OpenCV.
"""

import os
import sys
import time
import logging
from typing import Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger("CameraManager")


class CameraManager:
    """
    Manages physical camera connection, real-time frame acquisition,
    actual hardware resolution retrieval, FPS calculation, and overlay rendering.
    """

    def __init__(self, camera_id: int = 0, target_width: int = 1280, target_height: int = 720) -> None:
        self.camera_id = camera_id
        self.target_width = target_width
        self.target_height = target_height
        self.cap: Optional[cv2.VideoCapture] = None
        self.actual_width: int = 0
        self.actual_height: int = 0
        
        # FPS Calculation variables
        self._prev_time: float = 0.0
        self._current_fps: float = 0.0
        self._alpha: float = 0.1  # Smoothing factor for exponential moving average

    def start(self) -> bool:
        """
        Attempts to open the physical laptop webcam hardware feed.
        Returns True if successful, False otherwise.
        """
        logger.info(f"Attempting to connect to camera hardware index {self.camera_id}...")

        # Select backend based on OS
        backends = []
        if os.name == 'nt':
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        else:
            backends = [cv2.CAP_ANY]

        # Try specified camera_id across backends
        for backend in backends:
            cap = cv2.VideoCapture(self.camera_id, backend)
            if cap.isOpened():
                self.cap = cap
                logger.info(f"Successfully opened camera index {self.camera_id} with backend {backend}.")
                break
            cap.release()

        # Fallback: Try alternative camera indices (e.g., index 1) if default index 0 failed
        if self.cap is None or not self.cap.isOpened():
            fallback_id = 1 if self.camera_id == 0 else 0
            logger.warning(f"Could not open camera index {self.camera_id}. Attempting fallback index {fallback_id}...")
            for backend in backends:
                cap = cv2.VideoCapture(fallback_id, backend)
                if cap.isOpened():
                    self.cap = cap
                    self.camera_id = fallback_id
                    logger.info(f"Successfully opened fallback camera index {fallback_id}.")
                    break
                cap.release()

        if self.cap is None or not self.cap.isOpened():
            logger.error("❌ ERROR: Unable to access built-in camera or USB webcam.")
            logger.error("Please verify that:")
            logger.error(" 1. Your laptop webcam is connected and enabled.")
            logger.error(" 2. Privacy settings grant camera permission to Python/Terminal.")
            logger.error(" 3. No other application (Zoom, Teams, Skype, Browser) is currently locking the camera.")
            return False

        # Attempt setting target resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)

        # Retrieve actual hardware parameters
        self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        hardware_fps = self.cap.get(cv2.CAP_PROP_FPS)

        logger.info(f"Hardware camera ready. Actual Resolution: {self.actual_width}x{self.actual_height}, Reported Hardware FPS: {hardware_fps}")
        self._prev_time = time.time()
        return True

    def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Reads actual live camera frame from hardware feed.
        Returns (success_flag, bgr_frame_numpy_array).
        """
        if self.cap is None or not self.cap.isOpened():
            return False, None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False, None

        # Compute real runtime measured FPS
        curr_time = time.time()
        delta = curr_time - self._prev_time
        self._prev_time = curr_time

        if delta > 0:
            instant_fps = 1.0 / delta
            if self._current_fps == 0.0:
                self._current_fps = instant_fps
            else:
                self._current_fps = self._alpha * instant_fps + (1.0 - self._alpha) * self._current_fps

        return True, frame

    def get_resolution(self) -> Tuple[int, int]:
        """Returns actual hardware camera resolution (width, height)."""
        return self.actual_width, self.actual_height

    def get_fps(self) -> float:
        """Returns real-time measured FPS."""
        return self._current_fps

    def draw_status_overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        Renders live status overlay on camera frame:
        CAMERA: CONNECTED
        Resolution: WxH
        FPS: ...
        """
        overlay = frame.copy()
        h, w = frame.shape[:2]

        # Draw semi-transparent top-left panel
        panel_w, panel_h = 320, 110
        cv2.rectangle(overlay, (15, 15), (15 + panel_w, 15 + panel_h), (20, 20, 20), -1)
        
        # Apply alpha blending for clean UI look
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (15, 15), (15 + panel_w, 15 + panel_h), (0, 255, 0), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "CAMERA: CONNECTED", (30, 45), font, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Resolution: {self.actual_width}x{self.actual_height}", (30, 75), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {self._current_fps:.1f}", (30, 105), font, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

        return frame

    def stop(self) -> None:
        """Safely releases physical webcam hardware resource."""
        if self.cap is not None:
            if self.cap.isOpened():
                self.cap.release()
            self.cap = None
            logger.info("Physical camera hardware released successfully.")

