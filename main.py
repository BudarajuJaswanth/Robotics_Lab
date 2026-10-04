"""
AI and Robotics Lab - Real-Time 6-DoF Object Tracker
Stage 7: Physical Accuracy Testing & Evaluation Entry Point

Modes:
  1. Tracking (--mode track): Real-time 6-DoF pose estimation, baseline reference locking,
     and delta comparison telemetry.
  2. Calibration (--mode calibrate): Physical chessboard corner detection, sub-pixel refinement,
     cv2.calibrateCamera intrinsic matrix solver, and JSON persistence.
  3. Verification (--mode verify): Real-time side-by-side comparison of raw webcam feed vs.
     undistorted camera feed using saved JSON parameters.
  4. Evaluation (--mode evaluate): Physical accuracy experiment mode comparing real webcam
     measurements against user-provided ground-truth movement (Exp X, Y, Z in mm), exporting CSV.
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
from tests.accuracy_evaluator import AccuracyEvaluator
from ui.dashboard import DashboardOverlay

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("Main")


def parse_args():
    parser = argparse.ArgumentParser(description="AI & Robotics Lab - Real-Time 6-DoF Object Tracker")
    parser.add_argument("--mode", type=str, choices=["track", "calibrate", "verify", "evaluate"], default="track",
                        help="Operation mode: 'track' for live pose & reference comparison, 'calibrate' for chessboard calibration, 'verify' for undistortion test, 'evaluate' for physical accuracy experiment (default: track)")
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
    parser.add_argument("--exp-x", type=float, default=50.0, help="Expected ground-truth physical movement along X in mm (default: 50.0)")
    parser.add_argument("--exp-y", type=float, default=0.0, help="Expected ground-truth physical movement along Y in mm (default: 0.0)")
    parser.add_argument("--exp-z", type=float, default=0.0, help="Expected ground-truth physical movement along Z in mm (default: 0.0)")
    parser.add_argument("--output-csv", type=str, default="data/accuracy_results.csv", help="Accuracy evaluation output CSV path (default: data/accuracy_results.csv)")
    return parser.parse_args()


def run_evaluation_mode(args) -> None:
    logger.info("Initializing Stage 7: Physical Accuracy Testing Mode...")
    logger.info(f"Ground-Truth Expected Movement: Exp_X={args.exp_x} mm, Exp_Y={args.exp_y} mm, Exp_Z={args.exp_z} mm")

    calibrator = CameraCalibrator()
    if not calibrator.load_calibration(args.calibration_file):
        logger.error("\n" + "=" * 70)
        logger.error(f"❌ ERROR: Calibration file '{args.calibration_file}' does not exist or is invalid!")
        logger.error("Accuracy testing requires valid camera calibration parameters.")
        logger.error("Please run the calibration stage first using:")
        logger.error("    python main.py --mode calibrate")
        logger.error("=" * 70 + "\n")
        sys.exit(1)

    aruco_tracker = ArUcoTracker(dictionary_name=args.dict, marker_size_mm=args.marker_size)
    pose_estimator = PoseEstimator(marker_size_mm=args.marker_size)
    ref_manager = ReferenceManager(filepath=args.reference_file)
    evaluator = AccuracyEvaluator(
        expected_movement_mm=(args.exp_x, args.exp_y, args.exp_z),
        output_csv=args.output_csv
    )
    camera_manager = CameraManager(camera_id=args.camera_id, target_width=1280, target_height=720)
    dashboard = DashboardOverlay()

    if not camera_manager.start():
        logger.critical("❌ ERROR: Failed to open physical camera stream.")
        sys.exit(1)

    window_name = "AI & Robotics Lab - Stage 7: Physical Accuracy Testing"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    status_message = "1. Press [SPACE] to lock reference pose. 2. Move physical object. 3. Press [SPACE] to collect samples."
    logger.info("Controls: [SPACE] Lock Ref / Collect Samples | [S] Save CSV | [R] Reset | [Q] Quit")

    try:
        while True:
            success, raw_frame = camera_manager.get_frame()
            if not success or raw_frame is None:
                continue

            is_detected, marker_ids, corners, meta = aruco_tracker.detect_markers(raw_frame)

            display_frame = raw_frame.copy()
            pose_data = None
            primary_id = None
            delta_data = None

            if is_detected and len(corners) > 0:
                display_frame = aruco_tracker.draw_corners(display_frame, corners, marker_ids)
                primary_id = marker_ids[0]

                pose_success, pose_data = pose_estimator.estimate_pose(
                    corners[0], calibrator.camera_matrix, calibrator.dist_coeffs
                )

                if pose_success and pose_data is not None:
                    rvec = pose_data["rvec"]
                    tvec = pose_data["tvec"]
                    axis_len = pose_estimator.marker_size_m
                    if hasattr(cv2, "drawFrameAxes"):
                        cv2.drawFrameAxes(display_frame, calibrator.camera_matrix, calibrator.dist_coeffs, rvec, tvec, axis_len)

                    if ref_manager.has_reference():
                        delta_data = ref_manager.calculate_delta(pose_data)

            # Render Accuracy Evaluation HUD
            stats = evaluator.compute_statistics()
            display_frame = dashboard.render_accuracy_eval_hud(
                display_frame,
                is_detected=is_detected and (pose_data is not None),
                expected_movement_mm=(args.exp_x, args.exp_y, args.exp_z),
                sample_count=len(evaluator.samples),
                stats=stats,
                status_message=status_message
            )

            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF

            # [SPACE] -> Lock baseline reference or collect real sample
            if key == 32:
                if is_detected and pose_data is not None and primary_id is not None:
                    if not ref_manager.has_reference():
                        ref_manager.save_reference_pose(primary_id, pose_data)
                        status_message = "Baseline reference LOCKED! Move object, then press [SPACE] to capture."
                    else:
                        if delta_data is not None:
                            meas_delta = (delta_data["delta_X_mm"], delta_data["delta_Y_mm"], delta_data["delta_Z_mm"])
                            evaluator.add_sample(meas_delta)
                            status_message = f"Captured Sample #{len(evaluator.samples)}!"
                else:
                    status_message = "Cannot sample: physical ArUco marker lost!"

            # [S] -> Export CSV report & print terminal summary
            elif key == ord('s') or key == ord('S'):
                if evaluator.save_csv():
                    summary_text = evaluator.format_summary_report()
                    print(summary_text)
                    status_message = f"Saved CSV: {args.output_csv}"

            # [R] -> Reset evaluation experiment
            elif key == ord('r') or key == ord('R'):
                evaluator.reset()
                ref_manager.clear_reference()
                status_message = "Reset evaluation experiment and reference."

            # [Q] or [ESC] -> Quit
            elif key == ord('q') or key == ord('Q') or key == 27:
                if len(evaluator.samples) > 0:
                    evaluator.save_csv()
                    print(evaluator.format_summary_report())
                logger.info("Exiting accuracy evaluation mode...")
                break

    finally:
        camera_manager.stop()
        cv2.destroyAllWindows()


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

            is_detected, marker_ids, corners, meta = aruco_tracker.detect_markers(raw_frame)

            display_frame = raw_frame.copy()
            pose_data = None
            primary_id = None

            if is_detected and len(corners) > 0:
                display_frame = aruco_tracker.draw_corners(display_frame, corners, marker_ids)
                primary_id = marker_ids[0]

                pose_success, pose_data = pose_estimator.estimate_pose(
                    corners[0], calibrator.camera_matrix, calibrator.dist_coeffs
                )

                if pose_success and pose_data is not None:
                    rvec = pose_data["rvec"]
                    tvec = pose_data["tvec"]
                    axis_len = pose_estimator.marker_size_m
                    if hasattr(cv2, "drawFrameAxes"):
                        cv2.drawFrameAxes(display_frame, calibrator.camera_matrix, calibrator.dist_coeffs, rvec, tvec, axis_len)
                    elif hasattr(cv2.aruco, "drawAxis"):
                        cv2.aruco.drawAxis(display_frame, calibrator.camera_matrix, calibrator.dist_coeffs, rvec, tvec, axis_len)

            delta_data = None
            if is_detected and pose_data is not None and ref_manager.has_reference():
                delta_data = ref_manager.calculate_delta(pose_data)

            display_frame = dashboard.render_6dof_pose_hud(
                display_frame,
                is_detected=is_detected and (pose_data is not None),
                marker_id=primary_id,
                pose_data=pose_data,
                ref_data=ref_manager.ref_data,
                delta_data=delta_data,
                status_message=status_message
            )

            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == 32:
                if is_detected and pose_data is not None and primary_id is not None:
                    ok, msg = ref_manager.save_reference_pose(
                        primary_id, pose_data, calibration_info=args.calibration_file
                    )
                    status_message = msg
                else:
                    status_message = "Cannot save reference: marker pose unavailable."
                    logger.warning(status_message)

            elif key == ord('r') or key == ord('R'):
                ok, msg = ref_manager.clear_reference()
                status_message = msg
                logger.info(msg)

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
    elif args.mode == "evaluate":
        run_evaluation_mode(args)
    else:
        run_tracking_mode(args)


if __name__ == "__main__":
    main()
