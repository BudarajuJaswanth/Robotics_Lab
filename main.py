"""
AI and Robotics Lab - Real-Time 6-DoF Object Tracker
Stage 5: Real Reference Pose Recording Entry Point

Orchestrates live physical webcam ingestion, ArUco marker detection, Perspective-n-Point (PnP) 
6-DoF pose estimation, baseline reference pose recording to reference/data.json via SPACE,
persistence across application restarts, and real-time HUD rendering.
"""

import sys
import os
import argparse
import logging
import cv2
import numpy as np
from camera.camera_manager import CameraManager
from camera.calibration import CameraCalibrator
from tracking.aruco_tracker import ArUcoTracker
from tracking.pose_estimator import PoseEstimator
from reference.reference_manager import ReferenceManager
from ui.dashboard import DashboardOverlay

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("Main")


def parse_args():
    parser = argparse.ArgumentParser(description="AI & Robotics Lab - Real-Time 6-DoF Object Tracker")
    parser.add_argument("--mode", type=str, choices=["track", "calibrate", "verify"], default="track",
                        help="Operation mode: 'track' for live 6-DoF pose estimation & reference recording, 'calibrate' for chessboard calibration, 'verify' for undistortion test (default: track)")
    parser.add_argument("--dict", type=str, default="DICT_6X6_250",
                        help="ArUco dictionary name (e.g. DICT_6X6_250, DICT_4X4_50, DICT_5X5_100, etc.) (default: DICT_6X6_250)")
    parser.add_argument("--marker-size", type=float, default=50.0,
                        help="Physical ArUco marker side length in millimeters (default: 50.0)")
    parser.add_argument("--cols", type=int, default=9, help="Number of inner chessboard corners along columns (default: 9)")
    parser.add_argument("--rows", type=int, default=6, help="Number of inner chessboard corners along rows (default: 6)")
    parser.add_argument("--square-size", type=float, default=25.0, help="Physical square size in millimeters (default: 25.0)")
    parser.add_argument("--camera-id", type=int, default=0, help="Webcam hardware index (default: 0)")
    parser.add_argument("--calibration-file", type=str, default="calibration_data/camera_calibration.json",
                        help="Calibration JSON file path (default: calibration_data/camera_calibration.json)")
    parser.add_argument("--reference-file", type=str, default="reference/data.json",
                        help="Reference pose JSON file path (default: reference/data.json)")
    return parser.parse_args()


def run_tracking_mode(args) -> None:
    logger.info("Initializing Stage 5: Real 6-DoF Pose Tracking & Reference Recording...")
    logger.info(f"Configuration: ArUco Dict='{args.dict}', Physical Marker Size={args.marker_size} mm")

    calibrator = CameraCalibrator()
    if not calibrator.load_calibration(args.calibration_file):
        logger.error("\n" + "=" * 70)
        logger.error(f"❌ ERROR: Calibration file '{args.calibration_file}' does not exist or is invalid!")
        logger.error("Real 6-DoF pose estimation requires camera calibration parameters.")
        logger.error("Please run the calibration stage first using:")
        logger.error("    python main.py --mode calibrate")
        logger.error("=" * 70 + "\n")
        sys.exit(1)

    logger.info(f"✅ Loaded camera calibration from {args.calibration_file}")

    aruco_tracker = ArUcoTracker(dictionary_name=args.dict, marker_size_mm=args.marker_size)
    pose_estimator = PoseEstimator(marker_size_mm=args.marker_size)
    ref_manager = ReferenceManager(filepath=args.reference_file)
    camera_manager = CameraManager(camera_id=args.camera_id, target_width=1280, target_height=720)
    dashboard = DashboardOverlay()

    if not camera_manager.start():
        logger.critical("❌ ERROR: Failed to open physical camera stream.")
        sys.exit(1)

    window_name = "AI & Robotics Lab - Real-Time 6-DoF Pose & Reference Tracker"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    status_message = ""
    if ref_manager.has_reference():
        logger.info(f"Persistent reference pose loaded from {args.reference_file}")

    logger.info("Controls: [SPACE] Save Reference Pose  |  [R] Clear Reference  |  [Q] Quit")

    try:
        while True:
            success, raw_frame = camera_manager.get_frame()
            if not success or raw_frame is None:
                continue

            # Detect ArUco marker in physical camera frame
            is_detected, marker_ids, corners, meta = aruco_tracker.detect_markers(raw_frame)

            display_frame = raw_frame.copy()
            pose_data = None
            primary_id = None

            if is_detected and len(corners) > 0:
                display_frame = aruco_tracker.draw_corners(display_frame, corners, marker_ids)
                primary_id = marker_ids[0]

                # Estimate 6-DoF Pose (PnP IPPE Square Solver)
                pose_success, pose_data = pose_estimator.estimate_pose(
                    corners[0], calibrator.camera_matrix, calibrator.dist_coeffs
                )

                if pose_success and pose_data is not None:
                    # Draw 3D Coordinate Frame Axes (X: Red, Y: Green, Z: Blue)
                    rvec = pose_data["rvec"]
                    tvec = pose_data["tvec"]
                    axis_len = pose_estimator.marker_size_m
                    if hasattr(cv2, "drawFrameAxes"):
                        cv2.drawFrameAxes(display_frame, calibrator.camera_matrix, calibrator.dist_coeffs, rvec, tvec, axis_len)
                    elif hasattr(cv2.aruco, "drawAxis"):
                        cv2.aruco.drawAxis(display_frame, calibrator.camera_matrix, calibrator.dist_coeffs, rvec, tvec, axis_len)

            # Render 6-DoF Telemetry and Reference HUD Overlay
            display_frame = dashboard.render_6dof_pose_hud(
                display_frame,
                is_detected=is_detected and (pose_data is not None),
                marker_id=primary_id,
                pose_data=pose_data,
                ref_data=ref_manager.ref_data,
                status_message=status_message
            )

            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF

            # [SPACE] -> Record/Save current real pose as reference
            if key == 32:
                if is_detected and pose_data is not None and primary_id is not None:
                    ok, msg = ref_manager.save_reference_pose(
                        primary_id, pose_data, calibration_info=args.calibration_file
                    )
                    status_message = msg
                else:
                    status_message = "Cannot save reference: marker pose unavailable."
                    logger.warning(status_message)

            # [R] -> Clear reference pose
            elif key == ord('r') or key == ord('R'):
                ok, msg = ref_manager.clear_reference()
                status_message = msg
                logger.info(msg)

            # [Q] or [ESC] -> Quit
            elif key == ord('q') or key == ord('Q') or key == 27:
                logger.info("Exiting 6-DoF tracking mode...")
                break

    finally:
        camera_manager.stop()
        cv2.destroyAllWindows()


def run_calibration_mode(args) -> None:
    logger.info("Initializing Stage 2: Camera Calibration Mode...")
    logger.info(f"Configuration: {args.cols}x{args.rows} inner corners, Square Size: {args.square_size} mm")

    camera_manager = CameraManager(camera_id=args.camera_id, target_width=1280, target_height=720)
    calibrator = CameraCalibrator(pattern_size=(args.cols, args.rows), square_size_mm=args.square_size)
    dashboard = DashboardOverlay()

    if not camera_manager.start():
        logger.critical("❌ ERROR: Failed to open physical laptop camera stream.")
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

            found, refined_corners, display_frame = calibrator.detect_chessboard(raw_frame)

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
            key = cv2.waitKey(1) & 0xFF

            if key == 32:  # SPACE
                if found and refined_corners is not None:
                    calibrator.add_calibration_frame(refined_corners)
                    status_message = f"Captured Frame #{len(calibrator.img_points)}!"
                else:
                    status_message = "❌ Cannot capture: Chessboard corners not found!"

            elif key == ord('c') or key == ord('C'):
                logger.info("Calculating camera calibration parameters...")
                calib_success, msg = calibrator.calibrate()
                status_message = msg
                if calib_success:
                    save_success, save_msg = calibrator.save_calibration(args.calibration_file)
                    status_message += f" | {save_msg}"
                    logger.info("=" * 60)
                    logger.info("Camera Intrinsic Matrix (K):")
                    logger.info(f"\n{calibrator.camera_matrix}")
                    logger.info("Lens Distortion Coefficients:")
                    logger.info(f"\n{calibrator.dist_coeffs}")
                    logger.info("=" * 60)

            elif key == ord('r') or key == ord('R'):
                calibrator.reset_frames()
                status_message = "Reset all captured calibration frames."

            elif key == ord('q') or key == ord('Q') or key == 27:
                logger.info("Exiting calibration mode...")
                break

    finally:
        camera_manager.stop()
        cv2.destroyAllWindows()


def run_verification_mode(args) -> None:
    logger.info("Initializing Camera Calibration Verification Mode...")

    calibrator = CameraCalibrator()
    if not calibrator.load_calibration(args.calibration_file):
        logger.error("\n" + "=" * 70)
        logger.error(f"❌ ERROR: Calibration file '{args.calibration_file}' does not exist or is invalid!")
        logger.error("Please run the calibration stage first using:")
        logger.error("    python main.py --mode calibrate")
        logger.error("=" * 70 + "\n")
        sys.exit(1)

    logger.info(f"✅ Calibration successfully loaded from {args.calibration_file}")
    logger.info(f"Loaded RMS Error: {calibrator.rms_reprojection_error}")

    camera_manager = CameraManager(camera_id=args.camera_id, target_width=1280, target_height=720)
    dashboard = DashboardOverlay()

    if not camera_manager.start():
        logger.critical("❌ ERROR: Failed to open physical camera stream.")
        sys.exit(1)

    w, h = camera_manager.get_resolution()

    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
        calibrator.camera_matrix, calibrator.dist_coeffs, (w, h), 1, (w, h)
    )
    mapx, mapy = cv2.initUndistortRectifyMap(
        calibrator.camera_matrix, calibrator.dist_coeffs, None, new_camera_matrix, (w, h), cv2.CV_32FC1
    )

    window_name = "AI & Robotics Lab - Calibration Verification (Original vs Undistorted)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    logger.info("Displaying live side-by-side verification stream. Press 'Q' or 'ESC' to exit.")

    try:
        while True:
            success, raw_frame = camera_manager.get_frame()
            if not success or raw_frame is None:
                continue

            undistorted_frame = cv2.remap(raw_frame, mapx, mapy, cv2.INTER_LINEAR)

            combined = dashboard.render_verification_hud(
                raw_frame,
                undistorted_frame,
                calibrator.rms_reprojection_error,
                (w, h)
            )

            cv2.imshow(window_name, combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q') or key == 27:
                logger.info("Exiting verification mode...")
                break
    finally:
        camera_manager.stop()
        cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    if args.mode == "calibrate":
        run_calibration_mode(args)
    elif args.mode == "verify":
        run_verification_mode(args)
    else:
        run_tracking_mode(args)


if __name__ == "__main__":
    main()
