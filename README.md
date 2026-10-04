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

## 🛠️ Setup & Prerequisites

### Prerequisites
- **Python 3.9+**
- Physical built-in or USB Webcam
- Printed Chessboard target (for calibration)
- Printed ArUco marker target (e.g., `DICT_6X6_250`)

### Installation

```bash
# Clone or open project directory
cd RealTime_6DOF_Object_Tracker

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Running the Application

```bash
python main.py
```
