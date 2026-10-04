# AI and Robotics Lab - Real-Time 6-DoF Object Tracker

A real-time physical computer vision application for 6-DoF (Degrees of Freedom) object tracking using Python, OpenCV, and NumPy, developed for the **AI and Robotics Lab**.

> **Note:** This project operates exclusively on live physical camera feeds from hardware webcams. It does not use fake frames, simulated pose metrics, pre-recorded videos, or cloud vision APIs.

---

## 📌 Project Overview


This application measures real-time 6-DoF object poses relative to a calibrated camera frame:
- **3-DoF Translation**: X, Y, Z coordinates in physical space (meters/millimeters).
- **3-DoF Rotation**: Pitch, Yaw, Roll ($R_x, R_y, R_z$) orientation angles.

### Key Capabilities
1. **Webcam Capture & Hardware Control**: Direct low-latency video feed handling via OpenCV (`cv2.VideoCapture`).
2. **Camera Calibration**: Calculation of intrinsic camera matrix and lens distortion coefficients using physical chessboard pattern capture.
3. **ArUco Marker & Board Detection**: Identification and corner extraction of physical fiducial markers.
4. **6-DoF Pose Estimation**: Solving Perspective-n-Point ($PnP$) using calibrated camera parameters.
5. **Reference Pose Recording**: Storing baseline physical coordinates for differential tracking.
6. **Dynamic Relative Tracking**: Real-time delta computation ($\Delta X, \Delta Y, \Delta Z, \Delta R_x, \Delta R_y, \Delta R_z$) from baseline pose.
7. **3D Coordinate Transformation**: rigid-body spatial transformations ($SE(3)$ matrix operations).
8. **Real-Time Visual UI**: Live dashboard overlay rendering pose matrices, coordinate axes, and status diagnostics.

---

## 📂 Project Directory Structure

```text
RealTime_6DOF_Object_Tracker/
├── main.py                    # Application entry point & orchestration
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
│
├── camera/                    # Camera feed ingestion & intrinsic calibration
│   ├── __init__.py
│   ├── camera_manager.py      # Hardware camera initialization & frame capture
│   └── calibration.py         # Chessboard calibration & parameters persistence
│
├── tracking/                  # Computer vision pose estimation pipeline
│   ├── __init__.py
│   ├── aruco_tracker.py       # ArUco marker detection & corner extraction
│   ├── pose_estimator.py      # PnP pose solver (translation & rotation vectors)
│   └── transformations.py     # Rodrigues transform & SE(3) matrix utilities
│
├── reference/                 # Baseline pose management
│   ├── __init__.py
│   └── reference_manager.py   # Reference frame locking & delta calculation
│
├── ui/                        # Real-time visualization & rendering
│   ├── __init__.py
│   └── dashboard.py           # HUD overlay, telemetry display, & 3D axis visualization
│
├── data/                      # Reference poses and session output data
│   └── .gitkeep
│
├── calibration_data/          # Intrinsic matrices & distortion parameters
│   └── .gitkeep
│
└── tests/                     # Validation & test suite
    └── .gitkeep
```

---

## 🏁 Stage 2: Physical Camera Calibration Guide

Camera calibration calculates the intrinsic parameters (focal length $f_x, f_y$, principal point $c_x, c_y$) and lens distortion coefficients ($k_1, k_2, p_1, p_2, k_3$) of your physical laptop webcam.

### 📄 1. Printing & Preparing the Physical Chessboard Target
1. **Download/Generate a standard 10x7 square chessboard pattern** (or 9x6 inner corners pattern).
2. **Print at 100% scale**: When printing, ensure scale is set to **Actual Size (100%)**. Do not select "Fit to Printable Area".
3. **Mount Rigidly**: Tape or glue the printed paper onto a flat, rigid board (such as cardboard, clipboard, or foam board). *Bended paper creates lens distortion errors!*
4. **Measure Square Size**: Use a physical ruler to measure the exact length of one black square edge in millimeters (e.g. `25.0` mm or `30.0` mm).

---

### 📷 2. Running Stage 2 Calibration

```bash
# Run calibration with default 9x6 inner corners and 25.0 mm square size
python main.py

# Or customize inner corners and square size via CLI arguments
python main.py --cols 9 --rows 6 --square-size 25.0 --output calibration_data/camera_calibration.json
```

---

### 🎮 3. Keyboard Controls During Live Calibration

| Key | Action | Description |
| :--- | :--- | :--- |
| **`SPACE`** | **Capture View** | Captures the current frame when green chessboard corners are visible. |
| **`C`** | **Calibrate & Save** | Calculates OpenCV intrinsic matrix & saves parameters to JSON. |
| **`R`** | **Reset** | Clears all captured calibration views to start over. |
| **`Q` / `ESC`** | **Quit** | Exits the calibration tool safely. |

---

### 📐 4. Physical Positioning Strategy for High Accuracy
To achieve a low **RMS Reprojection Error (< 0.5 px)**, capture at least **10–15 views** covering your camera's field of view:
1. **Distance**: Capture views close to the camera, at medium distance, and further away.
2. **Screen Coverage**: Position the target in all 4 corners (top-left, top-right, bottom-left, bottom-right) and center.
3. **Angles & Tilts**: Tilt the chessboard target physically:
   - Pitch: Tilt forward and backward (~15° to 30°).
   - Yaw: Rotate left and right (~15° to 30°).
   - Roll: Rotate clockwise and counter-clockwise in the plane.

---

### 💾 5. Calibration Persistence Output

When you press **`C`**, OpenCV calculates the intrinsic matrix and saves the resulting configuration to:
`calibration_data/camera_calibration.json`

### 🔍 6. Running Calibration Verification Mode

After generating your `calibration_data/camera_calibration.json` file, run the verification mode to inspect real-time lens distortion correction:

```bash
python main.py --mode verify
```

- Displays a live side-by-side feed:
  - **Left**: Raw physical webcam feed (`ORIGINAL FEED`)
  - **Right**: Lens corrected feed using `cv2.remap` and your saved intrinsic matrix (`UNDISTORTED FEED`)
- Shows HUD statistics: `Calibration: LOADED`, `RMS Error: ...`, and `Camera Resolution: ...`.
- Press **`Q`** or **`ESC`** to exit.


