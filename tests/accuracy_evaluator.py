"""
Physical Accuracy Evaluator Module for 6-DoF Object Tracking.

Records real webcam pose measurements, compares against user-provided physical ground-truth movements,
computes statistical error metrics (Mean, Absolute Error, % Error, Std Dev, Min/Max Error),
and exports CSV reports to data/accuracy_results.csv.
"""

import os
import csv
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

logger = logging.getLogger("AccuracyEvaluator")


class AccuracyEvaluator:
    """
    Evaluates physical movement accuracy of the camera tracking system against known ground-truth displacement.
    """

    def __init__(
        self,
        expected_movement_mm: Tuple[float, float, float] = (50.0, 0.0, 0.0),
        output_csv: str = "data/accuracy_results.csv"
    ) -> None:
        """
        :param expected_movement_mm: Ground-truth physical movement vector (Exp_X, Exp_Y, Exp_Z) in millimeters.
        :param output_csv: Output CSV filepath for saving raw sample trials.
        """
        self.expected_dx, self.expected_dy, self.expected_dz = expected_movement_mm
        self.output_csv = output_csv
        self.samples: List[Tuple[float, float, float]] = []
        self.timestamps: List[str] = []

    def reset(self) -> None:
        """Clears collected samples."""
        self.samples.clear()
        self.timestamps.clear()

    def add_sample(self, measured_delta_mm: Tuple[float, float, float]) -> None:
        """Appends a real measured displacement sample from the webcam."""
        self.samples.append(measured_delta_mm)
        self.timestamps.append(time.strftime("%Y-%m-%d %H:%M:%S"))

    def compute_statistics(self) -> Optional[Dict[str, Any]]:
        """
        Calculates statistical metrics:
        - Mean measured movement (dX, dY, dZ)
        - Absolute error (X, Y, Z, 3D Euclidean)
        - Percentage error (%)
        - Standard deviation (Std Dev X, Y, Z)
        - Minimum & Maximum errors
        """
        if not self.samples:
            return None

        arr = np.array(self.samples)  # Shape (N, 3)
        meas_x = arr[:, 0]
        meas_y = arr[:, 1]
        meas_z = arr[:, 2]

        # Mean measured displacement
        mean_x = float(np.mean(meas_x))
        mean_y = float(np.mean(meas_y))
        mean_z = float(np.mean(meas_z))

        # Absolute errors (mean measured - expected)
        abs_err_x = abs(mean_x - self.expected_dx)
        abs_err_y = abs(mean_y - self.expected_dy)
        abs_err_z = abs(mean_z - self.expected_dz)

        # 3D Euclidean displacement error
        expected_vec = np.array([self.expected_dx, self.expected_dy, self.expected_dz])
        sample_errors = arr - expected_vec
        euclidean_errors = np.linalg.norm(sample_errors, axis=1)

        mean_euclidean_err = float(np.mean(euclidean_errors))
        min_euclidean_err = float(np.min(euclidean_errors))
        max_euclidean_err = float(np.max(euclidean_errors))

        # Percentage error calculation (avoid division by zero)
        pct_err_x = (abs_err_x / abs(self.expected_dx) * 100.0) if abs(self.expected_dx) > 1e-3 else 0.0
        pct_err_y = (abs_err_y / abs(self.expected_dy) * 100.0) if abs(self.expected_dy) > 1e-3 else 0.0
        pct_err_z = (abs_err_z / abs(self.expected_dz) * 100.0) if abs(self.expected_dz) > 1e-3 else 0.0

        # Total 3D expected magnitude
        exp_mag = float(np.linalg.norm(expected_vec))
        pct_err_3d = (mean_euclidean_err / exp_mag * 100.0) if exp_mag > 1e-3 else 0.0

        # Standard deviations
        std_x = float(np.std(meas_x))
        std_y = float(np.std(meas_y))
        std_z = float(np.std(meas_z))

        stats = {
            "num_samples": len(self.samples),
            "expected_mm": (self.expected_dx, self.expected_dy, self.expected_dz),
            "mean_measured_mm": (mean_x, mean_y, mean_z),
            "abs_error_mm": (abs_err_x, abs_err_y, abs_err_z),
            "pct_error": (pct_err_x, pct_err_y, pct_err_z),
            "std_dev_mm": (std_x, std_y, std_z),
            "euclidean_error_mean_mm": mean_euclidean_err,
            "euclidean_error_min_mm": min_euclidean_err,
            "euclidean_error_max_mm": max_euclidean_err,
            "pct_error_3d": pct_err_3d
        }
        return stats

    def save_csv(self) -> bool:
        """Exports raw frame samples and statistical summary to data/accuracy_results.csv."""
        if not self.samples:
            logger.warning("No samples collected to save.")
            return False

        os.makedirs(os.path.dirname(self.output_csv), exist_ok=True)

        try:
            with open(self.output_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Timestamp", "Sample_ID",
                    "Expected_dX_mm", "Expected_dY_mm", "Expected_dZ_mm",
                    "Measured_dX_mm", "Measured_dY_mm", "Measured_dZ_mm",
                    "Error_dX_mm", "Error_dY_mm", "Error_dZ_mm", "Euclidean_Error_mm"
                ])

                for idx, (mx, my, mz) in enumerate(self.samples):
                    ts = self.timestamps[idx]
                    err_x = mx - self.expected_dx
                    err_y = my - self.expected_dy
                    err_z = mz - self.expected_dz
                    euc_err = np.sqrt(err_x**2 + err_y**2 + err_z**2)

                    writer.writerow([
                        ts, idx + 1,
                        f"{self.expected_dx:.2f}", f"{self.expected_dy:.2f}", f"{self.expected_dz:.2f}",
                        f"{mx:.2f}", f"{my:.2f}", f"{mz:.2f}",
                        f"{err_x:.2f}", f"{err_y:.2f}", f"{err_z:.2f}", f"{euc_err:.2f}"
                    ])

            logger.info(f"✅ Accuracy evaluation results saved to {self.output_csv}")
            return True
        except Exception as e:
            logger.error(f"Failed to write CSV output {self.output_csv}: {e}")
            return False

    def format_summary_report(self) -> str:
        """Formats clean terminal statistical evaluation summary table."""
        stats = self.compute_statistics()
        if stats is None:
            return "No accuracy evaluation data available."

        exp_x, exp_y, exp_z = stats["expected_mm"]
        mx, my, mz = stats["mean_measured_mm"]
        ax, ay, az = stats["abs_error_mm"]
        px, py, pz = stats["pct_error"]
        sx, sy, sz = stats["std_dev_mm"]

        report = [
            "\n" + "=" * 70,
            "📊 PHYSICAL ACCURACY EVALUATION RESULTS REPORT",
            "=" * 70,
            f"Samples Collected: {stats['num_samples']} frames from live webcam",
            f"CSV Saved Path   : {self.output_csv}",
            "-" * 70,
            f"{'Metric':<25} | {'X Axis (mm)':<12} | {'Y Axis (mm)':<12} | {'Z Axis (mm)':<12}",
            "-" * 70,
            f"{'Expected Movement':<25} | {exp_x:+12.2f} | {exp_y:+12.2f} | {exp_z:+12.2f}",
            f"{'Mean Measured Movement':<25} | {mx:+12.2f} | {my:+12.2f} | {mz:+12.2f}",
            f"{'Absolute Error':<25} | {ax:12.2f} | {ay:12.2f} | {az:12.2f}",
            f"{'Percentage Error (%)':<25} | {px:11.2f}% | {py:11.2f}% | {pz:11.2f}%",
            f"{'Standard Deviation':<25} | {sx:12.2f} | {sy:12.2f} | {sz:12.2f}",
            "-" * 70,
            f"3D Euclidean Mean Error: {stats['euclidean_error_mean_mm']:.2f} mm ({stats['pct_error_3d']:.2f}%)",
            f"3D Euclidean Min Error : {stats['euclidean_error_min_mm']:.2f} mm",
            f"3D Euclidean Max Error : {stats['euclidean_error_max_mm']:.2f} mm",
            "=" * 70 + "\n"
        ]
        return "\n".join(report)
