"""
Unit and Component Verification Test Suite for RealTime_6DOF_Object_Tracker.
"""

import os
import json
import csv
import unittest
import numpy as np

from tracking.transformations import SpatialTransformations
from reference.reference_manager import ReferenceManager
from tests.accuracy_evaluator import AccuracyEvaluator
from tracking.pose_estimator import PoseEstimator
from tracking.aruco_tracker import ArUcoTracker
from camera.calibration import CameraCalibrator


class TestSpatialTransformations(unittest.TestCase):

    def test_rvec_to_rotation_matrix(self):
        rvec = np.zeros((3, 1), dtype=np.float64)
        R = SpatialTransformations.rvec_to_rotation_matrix(rvec)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-6)

    def test_pose_to_homogeneous_matrix_and_inverse(self):
        rvec = np.array([[0.1], [0.2], [0.3]], dtype=np.float64)
        tvec = np.array([[10.0], [20.0], [30.0]], dtype=np.float64)
        
        T = SpatialTransformations.pose_to_homogeneous_matrix(rvec, tvec)
        self.assertEqual(T.shape, (4, 4))
        self.assertEqual(T[3, 3], 1.0)
        
        T_inv = SpatialTransformations.inverse_transformation(T)
        identity = T @ T_inv
        np.testing.assert_allclose(identity, np.eye(4), atol=1e-6)

    def test_relative_transformations(self):
        rvec_ref = np.zeros((3, 1), dtype=np.float64)
        tvec_ref = np.array([[100.0], [200.0], [300.0]], dtype=np.float64)
        T_ref = SpatialTransformations.pose_to_homogeneous_matrix(rvec_ref, tvec_ref)

        rvec_curr = np.zeros((3, 1), dtype=np.float64)
        tvec_curr = np.array([[150.0], [200.0], [300.0]], dtype=np.float64)
        T_curr = SpatialTransformations.pose_to_homogeneous_matrix(rvec_curr, tvec_curr)

        T_ref_to_curr = SpatialTransformations.compute_ref_to_curr(T_ref, T_curr)
        T_curr_to_ref = SpatialTransformations.compute_curr_to_ref(T_ref, T_curr)

        # dX should be +50 mm in relative translation
        np.testing.assert_allclose(T_ref_to_curr[0, 3], 50.0, atol=1e-5)
        
        # T_ref_to_curr and T_curr_to_ref should be inverse of each other
        product = T_ref_to_curr @ T_curr_to_ref
        np.testing.assert_allclose(product, np.eye(4), atol=1e-6)

    def test_transform_3d_points(self):
        T = np.eye(4, dtype=np.float64)
        T[0:3, 3] = [10.0, -5.0, 2.0]
        points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float64)
        
        transformed = SpatialTransformations.transform_3d_points(T, points)
        expected = np.array([[10.0, -5.0, 2.0], [11.0, -4.0, 3.0]], dtype=np.float64)
        np.testing.assert_allclose(transformed, expected, atol=1e-6)


class TestReferenceManager(unittest.TestCase):

    def setUp(self):
        self.test_json = "reference/test_data.json"
        if os.path.exists(self.test_json):
            os.remove(self.test_json)

    def tearDown(self):
        if os.path.exists(self.test_json):
            os.remove(self.test_json)

    def test_save_and_load_reference(self):
        ref_mgr = ReferenceManager(filepath=self.test_json)
        self.assertFalse(ref_mgr.has_reference())

        rvec = np.zeros((3, 1), dtype=np.float64)
        tvec = np.array([[0.1], [0.2], [0.5]], dtype=np.float64)
        pose_data = {
            "rvec": rvec,
            "tvec": tvec,
            "translation_mm": (100.0, 200.0, 500.0),
            "rotation_deg": (0.0, 0.0, 0.0)
        }

        success, msg = ref_mgr.save_reference_pose(marker_id=23, pose_data=pose_data)
        self.assertTrue(success)
        self.assertTrue(ref_mgr.has_reference())

        # Test loading
        ref_mgr_new = ReferenceManager(filepath=self.test_json)
        self.assertTrue(ref_mgr_new.has_reference())
        self.assertEqual(ref_mgr_new.ref_data["marker_id"], 23)
        self.assertEqual(ref_mgr_new.ref_data["X"], 100.0)

        # Test delta calculation
        curr_tvec = np.array([[0.15], [0.20], [0.50]], dtype=np.float64)
        curr_pose_data = {
            "rvec": rvec,
            "tvec": curr_tvec,
            "translation_mm": (150.0, 200.0, 500.0),
            "rotation_deg": (0.0, 0.0, 0.0)
        }
        delta = ref_mgr.calculate_delta(curr_pose_data)
        self.assertIsNotNone(delta)
        self.assertAlmostEqual(delta["delta_X_mm"], 50.0, places=4)
        self.assertIn("T_reference_to_current", delta)
        self.assertIn("T_current_to_reference", delta)


class TestAccuracyEvaluator(unittest.TestCase):

    def setUp(self):
        self.test_csv = "data/test_accuracy.csv"
        if os.path.exists(self.test_csv):
            os.remove(self.test_csv)

    def tearDown(self):
        if os.path.exists(self.test_csv):
            os.remove(self.test_csv)

    def test_accuracy_statistics_and_csv(self):
        evaluator = AccuracyEvaluator(
            expected_movement_mm=(50.0, 0.0, 0.0),
            output_csv=self.test_csv
        )
        self.assertIsNone(evaluator.compute_statistics())

        # Add mock measured webcam samples
        evaluator.add_sample((49.0, 1.0, 0.0))
        evaluator.add_sample((51.0, -1.0, 0.0))

        stats = evaluator.compute_statistics()
        self.assertIsNotNone(stats)
        self.assertEqual(stats["num_samples"], 2)
        self.assertAlmostEqual(stats["mean_measured_mm"][0], 50.0, places=4)
        self.assertAlmostEqual(stats["abs_error_mm"][0], 0.0, places=4)

        # Test CSV export
        saved = evaluator.save_csv()
        self.assertTrue(saved)
        self.assertTrue(os.path.exists(self.test_csv))


class TestPoseEstimator(unittest.TestCase):

    def test_pose_estimator_init(self):
        estimator = PoseEstimator(marker_size_mm=100.0)
        self.assertEqual(estimator.marker_size_mm, 100.0)
        self.assertEqual(estimator.marker_size_m, 0.1)
        self.assertEqual(estimator.obj_points.shape, (4, 3))


class TestArUcoTracker(unittest.TestCase):

    def test_aruco_tracker_init(self):
        tracker = ArUcoTracker(dictionary_name="DICT_6X6_250", marker_size_mm=50.0)
        self.assertEqual(tracker.dictionary_name, "DICT_6X6_250")
        self.assertEqual(tracker.marker_size_m, 0.05)


class TestCameraCalibrator(unittest.TestCase):

    def test_calibrator_init(self):
        calibrator = CameraCalibrator(pattern_size=(9, 6), square_size_mm=25.0)
        self.assertEqual(calibrator.pattern_size, (9, 6))
        self.assertEqual(calibrator.square_size_m, 0.025)
        self.assertEqual(calibrator._single_objp.shape, (54, 3))


if __name__ == "__main__":
    unittest.main()
