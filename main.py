"""
AI and Robotics Lab - Real-Time 6-DoF Object Tracker
Stage 2: Real Camera Calibration Entry Point

Orchestrates real laptop webcam ingestion, physical chessboard corner detection,
sub-pixel refinement, OpenCV intrinsic camera calibration calculation, and JSON persistence.
"""

import sys
import argparse
import logging
import cv2
from camera.camera_manager import CameraManager
from camera.calibration import CameraCalibrator
from ui.dashboard import DashboardOverlay

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("Main")


def parse_args():
    parser = argparse.ArgumentParser(description="AI & Robotics Lab - Real-Time 6-DoF Object Tracker (Stage 2: Calibration)")
    parser.add_argument("--cols", type=int, default=9, help="Number of inner chessboard corners along columns (default: 9)")
    parser.add_argument("--rows", type=int, default=6, help="Number of inner chessboard corners along rows (default: 6)")
    parser.add_argument("--square-size", type=float, default=25.0, help="Physical square size in millimeters (default: 25.0)")
    parser.add_argument("--camera-id", type=int, default=0, help="Webcam hardware index (default: 0)")
    parser.add_argument("--output", type=str, default="calibration_data/camera_calibration.json", help="Output calibration JSON path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("Initializing AI and Robotics Lab - Stage 2: Camera Calibration...")
    logger.info(f"Calibration Configuration: {args.cols}x{args.rows} inner corners, Square Size: {args.square_size} mm")

    camera_manager = CameraManager(camera_id=args.camera_id, target_width=1280, target_height=720)
    calibrator = CameraCalibrator(pattern_size=(args.cols, args.rows), square_size_mm=args.square_size)
    dashboard = DashboardOverlay()

    if not camera_manager.start():
        logger.critical("\n" + "=" * 65)
        logger.critical("❌ ERROR: Failed to open physical laptop camera stream.")
        logger.critical("Please verify camera connection and permissions.")
        logger.critical("=" * 65 + "\n")
        sys.exit(1)

    window_name = "AI & Robotics Lab - Stage 2: Camera Calibration"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    status_message = "Point webcam at physical printed chessboard."
    logger.info("Controls: [SPACE] Capture Frame | [C] Calibrate | [R] Reset | [Q] Quit")

    try:
        while True:
            success, raw_frame = camera_manager.get_frame()
            if not success or raw_frame is None:
                continue

            # Detect & refine chessboard corners in real-time
            found, refined_corners, display_frame = calibrator.detect_chessboard(raw_frame)

            # Render UI HUD telemetry
            display_frame = dashboard.render_calibration_hud(
                display_frame,
                chessboard_detected=found,
                num_frames_captured=len(calibrator.img_points),
                rms_error=calibrator.rms_reprojection_error,
                status_message=status_message,
                pattern_size=calibrator.pattern_size,
                square_size_mm=calibrator.square_size_mm
            )

            cv2.imshow(window_name, display_frame)

            # Process Keyboard Controls
            key = cv2.waitKey(1) & 0xFF

            # [SPACE] -> Capture calibration frame
            if key == 32:
                if found and refined_corners is not None:
                    calibrator.add_calibration_frame(refined_corners)
                    status_message = f"Captured Frame #{len(calibrator.img_points)}!"
                else:
                    status_message = "❌ Cannot capture: Chessboard corners not found!"
                    logger.warning(status_message)

            # [C] -> Calculate OpenCV camera calibration
            elif key == ord('c') or key == ord('C'):
                logger.info("Calculating camera calibration parameters...")
                calib_success, msg = calibrator.calibrate()
                status_message = msg
                if calib_success:
                    save_success, save_msg = calibrator.save_calibration(args.output)
                    status_message += f" | {save_msg}"
                    logger.info("=" * 60)
                    logger.info("Camera Intrinsic Matrix (K):")
                    logger.info(f"\n{calibrator.camera_matrix}")
                    logger.info("Lens Distortion Coefficients:")
                    logger.info(f"\n{calibrator.dist_coeffs}")
                    logger.info("=" * 60)

            # [R] -> Reset captured calibration frames
            elif key == ord('r') or key == ord('R'):
                calibrator.reset_frames()
                status_message = "Reset all captured calibration frames."

            # [Q] or [ESC] -> Quit
            elif key == ord('q') or key == ord('Q') or key == 27:
                logger.info("Exiting calibration mode...")
                break

    except KeyboardInterrupt:
        logger.info("Interrupted by user. Exiting...")
    finally:
        camera_manager.stop()
        cv2.destroyAllWindows()
        logger.info("Webcam released and OpenCV windows destroyed.")


if __name__ == "__main__":
    main()


