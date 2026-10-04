"""
RealTime_6DOF_Object_Tracker - Stage 1 Entry Point

Orchestrates real laptop webcam ingestion, live feed display, FPS & resolution overlay,
and safe keypress resource cleanup.
"""

import sys
import logging
import cv2
from camera.camera_manager import CameraManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("Main")


def main() -> None:
    logger.info("Initializing AI and Robotics Lab - Real-Time 6-DoF Object Tracker [Stage 1: Webcam Feed]...")

    camera_manager = CameraManager(camera_id=0, target_width=1280, target_height=720)

    # Automatically attempt to open the physical laptop webcam hardware
    if not camera_manager.start():
        logger.critical("\n" + "=" * 65)
        logger.critical("❌ ERROR: Failed to open physical camera stream.")
        logger.critical("Please check camera connections, privacy permissions, or close")
        logger.critical("any other application using the webcam.")
        logger.critical("=" * 65 + "\n")
        sys.exit(1)

    window_name = "AI & Robotics Lab - 6DOF Tracker (Physical Camera Feed)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    logger.info("Live physical camera stream started successfully.")
    logger.info("Press 'Q' or 'ESC' in the window to safely stop camera feed and exit.")

    try:
        while True:
            success, frame = camera_manager.get_frame()
            if not success or frame is None:
                logger.warning("Frame acquisition returned empty or failed. Retrying...")
                continue

            # Render status overlay (CAMERA CONNECTED, Resolution, FPS)
            frame = camera_manager.draw_status_overlay(frame)

            # Display real camera feed in OpenCV window
            cv2.imshow(window_name, frame)

            # Check key press (Q or ESC to exit)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q') or key == 27:
                logger.info("User requested exit key ('Q' / 'ESC'). Stopping application...")
                break
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Exiting...")
    finally:
        # Safely release physical webcam hardware and destroy OpenCV windows
        camera_manager.stop()
        cv2.destroyAllWindows()
        logger.info("Webcam released and OpenCV windows closed successfully.")


if __name__ == "__main__":
    main()

