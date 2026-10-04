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

---

## 🎯 Stage 3: Real-Time ArUco Marker Tracking Guide

Stage 3 detects physical ArUco markers, identifies marker IDs, draws 2D corners, and projects 3D coordinate frame axes ($X$: Red, $Y$: Green, $Z$: Blue) using your calibrated camera parameters.

### 🖨️ 1. Printing & Preparing the Physical ArUco Marker
1. **Generate an ArUco Marker**:
   - Select Dictionary: **`DICT_6X6_250`** (or `DICT_4X4_50`, `DICT_5X5_100`).
   - Select Marker ID (e.g. `23` or `0`).
2. **Print at 100% Scale**: Print the marker so that the outer black square side is a known physical length (e.g. `50.0` mm or `100.0` mm).
3. **Attach Flat**: Tape the marker securely onto a flat rigid physical object (box, cardboard, or wooden block). *Avoid bending or crinkling paper.*
4. **Measure Physical Size**: Measure the exact outer black square width with a ruler in millimeters (e.g. `50.0` mm).

---

### 🚀 2. Running Stage 3 Tracking

```bash
# Run real-time ArUco tracking (default dictionary: DICT_6X6_250, marker size: 50.0 mm)
python main.py --mode track

# Or customize dictionary name and physical marker size
python main.py --mode track --dict DICT_6X6_250 --marker-size 50.0
```

- **Screen Telemetry**:
  - Displays **`ARUCO: DETECTED`** and **`ID: <marker_id>`** when visible.
  - Displays **`ARUCO: NOT DETECTED`** when no marker is in camera view.
- **3D Coordinate Axes**:
  - Draws RGB 3D spatial coordinate axes directly on the marker center:
    - **Red**: X axis
    - **Green**: Y axis
    - **Blue**: Z axis (pointing outwards perpendicular to marker plane)
- Press **`Q`** or **`ESC`** to quit cleanly.

---

## 📐 Stage 4: REAL 6-DoF Pose Estimation & Coordinate Conventions

Stage 4 calculates the 3D translation ($X, Y, Z$ in millimeters) and 3D rotation ($R_x, R_y, R_z$ in degrees) of the detected physical ArUco marker relative to the laptop camera optical center.

### 🌐 1. Coordinate System Conventions

1. **Camera Optical Coordinate Frame (Right-Handed System)**:
   - **Origin $(0,0,0)$**: The optical center (pinhole focus point) of the laptop camera lens.
   - **$+X$ axis**: Points **RIGHT** across the camera's horizontal field of view.
   - **$+Y$ axis**: Points **DOWN** across the camera's vertical field of view.
   - **$+Z$ axis**: Points **FORWARD** along the optical axis into the physical 3D scene (Optical Depth / Distance from camera lens).

2. **Physical Marker Coordinate Frame**:
   - **Origin $(0,0,0)$**: Located at the exact **center** of the physical square ArUco marker.
   - **$+Z$ axis**: Points **OUTWARD NORMAL** perpendicular to the printed front face of the marker.

3. **Euler Angle Rotation Conventions (Pitch, Yaw, Roll)**:
   - **Order**: Intrinsic $X \rightarrow Y \rightarrow Z$ Euler decomposition.
   - **$R_x$ (Pitch)**: Rotation around the $X$-axis (tilting target up or down) in degrees ($^\circ$).
   - **$R_y$ (Yaw)**: Rotation around the $Y$-axis (panning target left or right) in degrees ($^\circ$).
   - **$R_z$ (Roll)**: Rotation around the $Z$-axis (swiveling target clockwise/counterclockwise) in degrees ($^\circ$).

---

### 🖥️ 2. Live Telemetry HUD Display

```text
POSE 6-DoF [ID: 23] (RAW UNFILTERED)
TRANSLATION (mm):
X:   +124.5 mm
Y:    -45.2 mm
Z:   +450.8 mm

ROTATION (Euler Degrees):
Rx (Pitch):   +12.4 deg
Ry (Yaw)  :    -5.1 deg
Rz (Roll) :  +178.2 deg
```

---

### 🔍 3. Raw Measurement Noise Evaluation
- The displayed $X, Y, Z, R_x, R_y, R_z$ values are **raw, unfiltered physical measurements** derived from Perspective-n-Point ($PnP$) estimation (`cv2.SOLVEPNP_IPPE_SQUARE`).
- Showing raw values allows direct evaluation of physical camera sensor noise and illumination stability before applying temporal smoothing filters in downstream stages.

---

## 📌 Stage 5: REAL Reference Pose Recording & Persistence

Stage 5 allows locking a baseline reference pose ($X, Y, Z, R_x, R_y, R_z$) from live webcam measurements, persisting it to `reference/data.json`, and automatically restoring it upon application restart.

### 🎮 1. Reference Pose Keyboard Controls

| Key | Action | Description |
| :--- | :--- | :--- |
| **`SPACE`** | **Record Reference** | Captures the current live real pose and saves it to `reference/data.json`. |
| **`R`** | **Clear Reference** | Resets baseline reference in memory and removes `reference/data.json`. |
| **`Q` / `ESC`** | **Quit** | Exits the tracking session safely. |

> **Note**: Pressing `SPACE` when no marker is visible will display a warning banner: `Cannot save reference: marker pose unavailable.` and will not save invalid data.

---

### 💾 2. Reference JSON Schema (`reference/data.json`)

```json
{
    "timestamp": "2026-10-04 18:45:00",
    "marker_id": 23,
    "X": 124.5,
    "Y": -45.2,
    "Z": 450.8,
    "Rx": 12.4,
    "Ry": -5.1,
    "Rz": 178.2,
    "rvec": [0.21, -0.09, 3.10],
    "tvec": [0.1245, -0.0452, 0.4508],
    "SE3_matrix": [
        [0.98, -0.15, 0.12, 0.1245],
        [0.14, 0.99, 0.05, -0.0452],
        [-0.13, -0.03, 0.99, 0.4508],
        [0.0, 0.0, 0.0, 1.0]
    ],
    "camera_calibration_version": "calibration_data/camera_calibration.json"
}
```

---

## ⚡ Stage 6: REAL-TIME Reference Comparison & Delta Telemetry

Stage 6 computes real-time 6-DoF spatial translation deltas ($\Delta X, \Delta Y, \Delta Z$ in mm) and rotation deltas ($\Delta R_x, \Delta R_y, \Delta R_z$ in degrees) relative to your locked baseline reference.

### 📊 1. Screen Status Badges & HUD Display

- **Tracking Status Badges**:
  - `TRACKING` (Green badge): Physical ArUco marker is actively tracked.
  - `TRACKING LOST` (Red badge): Marker is out of view. *(Missing detections are never treated as zero movement).*
- **Reference Status Badges**:
  - `REFERENCE SAVED` (Blue/Green badge): Persistent reference loaded from `reference/data.json`.
  - `REFERENCE NOT SAVED` (Orange badge): Baseline reference not recorded yet.

### 🖥️ 2. Real-Time Telemetry Panels

```text
CURRENT POSE [ID: 23]       REFERENCE POSE [ID: 23]
TRANSLATION (mm):            REF X: +124.5  Y: -45.2  Z: +450.8 mm
X:   +140.2 mm
Y:    -40.1 mm               CHANGE (TRANSLATION DELTA):
Z:   +465.0 mm               dX:   +15.7 mm
                             dY:    +5.1 mm
ROTATION (Euler deg):        dZ:   +14.2 mm
Rx:   +14.5 deg
Ry:    -6.4 deg               ROTATION CHANGE (DELTA EULER):
Rz:  +178.7 deg               dRx:  +2.1 | dRy: -1.3 | dRz: +0.5 deg
```

---

## 🧪 Stage 7: Physical Accuracy Testing & CSV Evaluation

Stage 7 evaluates the physical measurement accuracy of the laptop camera tracking system by comparing real webcam displacement measurements against user-defined physical ground-truth movements (e.g. `Exp_X = 50.0` mm, `Exp_Y = 0.0` mm, `Exp_Z = 0.0` mm).

### 🚀 1. Running Stage 7 Physical Accuracy Testing

```bash
# Evaluate physical 50.0 mm movement along X axis
python main.py --mode evaluate --exp-x 50.0 --exp-y 0.0 --exp-z 0.0 --output-csv data/accuracy_results.csv

# Or evaluate 100.0 mm movement along Z axis (optical depth)
python main.py --mode evaluate --exp-x 0.0 --exp-y 0.0 --exp-z 100.0
```

### 🎮 2. Interactive Experiment Workflow
1. Point your laptop camera at the physical ArUco marker.
2. Press **`SPACE`**: Locks the initial baseline reference pose.
3. Move the physical target or webcam by the exact expected ground-truth distance (e.g., `50.0` mm along X using a ruler).
4. Press **`SPACE`**: Captures real physical webcam samples into the evaluation buffer.
5. Press **`S`**: Saves raw trials and summary metrics to `data/accuracy_results.csv` and prints the statistical summary table in your terminal.
6. Press **`R`**: Resets the experiment session to test another displacement vector.
7. Press **`Q`**: Exits the experiment.

---

### 📊 3. Statistical Metrics Calculated

- **Mean Measured Movement ($\mu_{\Delta}$)**: Average displacement measured across real camera frames.
- **Absolute Error ($E_{abs}$)**: $|\mu_{\text{measured}} - \text{Expected}|$ in mm.
- **Percentage Error ($E_{\%}$)**: $\frac{|\mu_{\text{measured}} - \text{Expected}|}{\text{Expected}} \times 100\%$.
- **Standard Deviation ($\sigma_{\Delta}$)**: Measurement precision/jitter across frames.
- **3D Euclidean Error**: $\sqrt{(dX - Exp_X)^2 + (dY - Exp_Y)^2 + (dZ - Exp_Z)^2}$ (Mean, Min, Max in mm).

---

### 📁 4. CSV Schema (`data/accuracy_results.csv`)


---

## 🔄 Stage 8: Reference Coordinate Transformation & $SE(3)$ Homogeneous Matrices

Stage 8 implements 3D Lie Group $SE(3)$ homogeneous coordinate transformations to rigorously compute physical relative motion between the baseline reference frame and the current marker pose.

### 🧮 1. Mathematical Structure of $SE(3)$ Homogeneous Matrix

A 6-DoF spatial pose is represented as a $4 \times 4$ homogeneous transformation matrix $T \in SE(3)$:

$$T = \begin{bmatrix} R_{3 \times 3} & \mathbf{t}_{3 \times 1} \\ \mathbf{0}_{1 \times 3} & 1 \end{bmatrix} = \begin{bmatrix} R_{11} & R_{12} & R_{13} & t_x \\ R_{21} & R_{22} & R_{23} & t_y \\ R_{31} & R_{32} & R_{33} & t_z \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

- **$R_{3 \times 3} \in SO(3)$**: 3D orthogonal rotation matrix ($\det(R) = +1, R^T = R^{-1}$).
- **$\mathbf{t}_{3 \times 1} \in \mathbb{R}^3$**: 3D translation vector $[t_x, t_y, t_z]^T$ in physical millimeters.

---

### 📐 2. Transformation Definitions & Physical Meanings

#### A. Reference Pose Matrix ($T_{\text{ref}}$)
Represents the baseline physical coordinate frame saved in `reference/data.json` relative to the camera optical center.

#### B. Current Pose Matrix ($T_{\text{curr}}$)
Represents the real-time physical pose of the ArUco marker currently observed by the laptop webcam relative to the camera optical center.

#### C. Analytical $SE(3)$ Matrix Inverse ($T^{-1}$)
Fast, exact inverse computation leveraging rotation orthogonality ($R^{-1} = R^T$):

$$T^{-1} = \begin{bmatrix} R^T & -R^T \cdot \mathbf{t} \\ \mathbf{0}_{1 \times 3} & 1 \end{bmatrix}$$

#### D. Reference-to-Current Transformation ($T_{\text{ref\_to\_curr}}$)

$$T_{\text{ref\_to\_curr}} = (T_{\text{ref}})^{-1} \cdot T_{\text{curr}}$$

- **Physical Meaning**: Rigid 6-DoF spatial transformation describing how the marker has physically moved from its original baseline position.
- **Coordinate Point Mapping**: Transforms a 3D point $P_{\text{ref}}$ defined in the baseline reference frame into the current marker frame:

$$P_{\text{curr}} = T_{\text{ref\_to\_curr}} \cdot P_{\text{ref}}$$

#### E. Current-to-Reference Transformation ($T_{\text{curr\_to\_ref}}$)

$$T_{\text{curr\_to\_ref}} = (T_{\text{curr}})^{-1} \cdot T_{\text{ref}} = (T_{\text{ref\_to\_curr}})^{-1}$$

- **Physical Meaning**: Inverse transformation mapping points from the current marker frame back into the original baseline reference coordinate system.
- **Coordinate Point Mapping**:

$$P_{\text{ref}} = T_{\text{curr\_to\_ref}} \cdot P_{\text{curr}}$$

---

### 💻 3. Transformations Module Functions (`tracking/transformations.py`)

- `rvec_to_rotation_matrix(rvec)`: Converts 3x1 Rodrigues rotation vector to 3x3 rotation matrix $R \in SO(3)$.
- `pose_to_homogeneous_matrix(rvec, tvec)`: Combines `rvec` and `tvec` into $4 \times 4$ homogeneous matrix $T \in SE(3)$.
- `inverse_transformation(T)`: Computes analytical $SE(3)$ inverse $T^{-1} = \begin{bmatrix} R^T & -R^T t \\ 0 & 1 \end{bmatrix}$.
- `relative_transformation(T_source, T_target)`: Computes $T_{\text{rel}} = T_{\text{source}}^{-1} \cdot T_{\text{target}}$.
- `compute_ref_to_curr(T_ref, T_curr)`: Computes $T_{\text{ref\_to\_curr}} = T_{\text{ref}}^{-1} \cdot T_{\text{curr}}$.
- `compute_curr_to_ref(T_ref, T_curr)`: Computes $T_{\text{curr\_to\_ref}} = T_{\text{curr}}^{-1} \cdot T_{\text{ref}}$.
- `transform_3d_points(T, points_3d)`: Transforms $N \times 3$ array of 3D point coordinates by $4 \times 4$ matrix $T$.
- `rvec_to_euler_angles(rvec)`: Converts Rodrigues vector to intrinsic XYZ Euler angles ($R_x, R_y, R_z$) in degrees.

---

### 🖥️ 4. Live UI HUD Matrix Overlay

During tracking mode (`python main.py --mode track`), when baseline reference is recorded and tracking is active, the UI renders the live $4 \times 4$ relative homogeneous matrix panel:

```text
STAGE 8: 4x4 RELATIVE TRANSFORMATION (T_ref_to_curr)
[ +0.998 -0.045 +0.012 |  +12.40]
[ +0.044 +0.999 +0.018 |  -45.20]
[ -0.013 -0.017 +0.999 | +105.80]
[  0.000  0.000  0.000 |   1.00]
```

---

## 📜 Summary of Keyboard Shortcuts

| Key | Mode | Function |
| :--- | :--- | :--- |
| **`SPACE`** | `calibrate` | Capture chessboard calibration frame |
| **`C`** | `calibrate` | Compute intrinsic camera matrix & save JSON |
| **`R`** | `calibrate` / `track` / `evaluate` | Reset captured frames / reference pose / evaluation samples |
| **`SPACE`** | `track` | Record baseline reference pose to `reference/data.json` |
| **`SPACE`** | `evaluate` | Lock baseline reference OR capture accuracy test sample |
| **`S`** | `evaluate` | Export accuracy experiment results to `data/accuracy_results.csv` |
| **`Q` / `ESC`** | All | Quit current application mode |








