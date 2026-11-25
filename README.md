# Autonomous Vehicle Surround-Vision & Sensor Fusion Intelligence

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()
[![Tests: Pytest](https://img.shields.io/badge/tests-100%25%20passing-success.svg)]()

An autonomous driving perception and sensor intelligence system featuring surround multi-camera vision, Bird's-Eye-View (BEV) occupancy mapping, Extended Kalman Filter (EKF) state estimation, 3D multi-object tracking (MOT), Time-To-Collision (TTC) risk assessment, quintic jerk-minimizing trajectory planning, Stanley lateral control, and automotive CAN bus telemetry.

---

## Interactive Web Visualizer & HUD

Open `examples/web_visualizer.html` in any modern web browser to view real-time 360° sensor coverage, camera perspective cones, forward radar Doppler velocity vectors, ultrasonic proximity safety zones, dynamic traffic, and instrument telemetry.

---

## System Architecture

### 1. Surround 8-Camera Optical Suite (`src/sensors/camera_rig.py`)
- Geometric optical modeling of an 8-camera surround vision rig:
  - **Front Wide** (120° FOV, 60m range)
  - **Front Main** (50° FOV, 150m range)
  - **Front Narrow** (35° FOV, 250m range)
  - **Pillars Left & Right Forward Oblique** (90° FOV, 80m range)
  - **Fender Repeaters Left & Right Rear Oblique** (90° FOV, 100m range)
  - **Rear Backup Center** (120° FOV, 50m range)
- Intrinsic calibration matrix $\mathbf{K}$, 3D extrinsic rotation matrices $\mathbf{R}$, and pinhole perspective projection:
  $$\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \mathbf{K} \left[ \mathbf{R}^T (\mathbf{P}_{veh} - \mathbf{T}) \right]$$

### 2. Radar & 12-Channel Ultrasonic Sonar (`src/sensors/ultrasonic_radar.py`)
- **12 Bumper Ultrasonic Transducers**: 6 front, 6 rear with acoustic beam cone geometry, obstacle distance ranging (0.15m to 8.0m), and 4-tier safety classification (`SAFE`, `CAUTION`, `WARNING`, `CRITICAL_STOP`).
- **77 GHz Forward Radar**: Range $r$, azimuth $\theta$, elevation $\phi$, and radial Doppler relative velocity $v_r$:
  $$v_r = (\mathbf{v}_{target} - \mathbf{v}_{ego}) \cdot \frac{\mathbf{r}}{\|\mathbf{r}\|}$$

### 3. Bird's-Eye-View (BEV) Occupancy Network (`src/perception/bev_transformer.py`)
- Transforms multi-camera perspectives into an ego-centric spatial 2D/3D voxel occupancy probability grid ($100\text{m} \times 40\text{m}$) with $0.5\text{m}$ resolution.
- Oriented 3D bounding box projection, velocity vector grids, and collision-free corridor verification.

### 4. Extended Kalman Filter (EKF) State Estimator (`src/fusion/ekf_fusion.py`)
- 9-state non-linear kinematic state vector:
  $$\mathbf{x} = \begin{bmatrix} p_x & p_y & p_z & v_x & v_y & v_z & \psi & \dot{\psi} & a_x \end{bmatrix}^T$$
- Fuses high-rate (100 Hz) 6-DOF IMU body accelerations and gyro yaw rates with RTK-GPS (10 Hz) East-North-Up (ENU) geodetic positions and wheel odometry.
- Joseph form covariance update ensuring guaranteed positive-definite numerical stability:
  $$\mathbf{P}_{k|k} = (\mathbf{I} - \mathbf{K}_k \mathbf{H}_k) \mathbf{P}_{k|k-1} (\mathbf{I} - \mathbf{K}_k \mathbf{H}_k)^T + \mathbf{K}_k \mathbf{R}_k \mathbf{K}_k^T$$

### 5. 3D Multi-Object Tracking & Risk Assessment (`src/tracking/`)
- Multi-target tracking with Kalman filtering and gated data association.
- Real-time **Time-To-Collision (TTC)**, **Forward Collision Warning (FCW)**, and **Automatic Emergency Braking (AEB)**:
  $$\text{TTC} = \frac{d_{rel}}{v_{closing}} = \frac{x_{target} - x_{ego}}{v_{ego} - v_{target}}$$
  $$a_{req} = \frac{v_{closing}^2}{2 \cdot d_{rel}}$$

### 6. Quintic Jerk-Minimizing Trajectory Planner (`src/planning/trajectory_planner.py`)
- Optimal 5th-order polynomial trajectory generation in Frenet frames:
  $$s(t) = a_0 + a_1 t + a_2 t^2 + a_3 t^3 + a_4 t^4 + a_5 t^5$$
- Multi-horizon evaluation minimizing composite cost across jerk $\int \dddot{s}(t)^2 dt$, lane offset, speed deviation, and surround obstacle clearances.

### 7. Stanley Controller & CAN Bus Interface (`src/control/`)
- Stanley lateral steering control law:
  $$\delta(t) = \theta_e(t) + \arctan\left(\frac{k \cdot e_{fa}(t)}{v(t) + k_{soft}}\right)$$
- Automotive CAN bus encoding with SAE J1850 CRC-8 checksums for steer control, drive control, and wheel speed frames.

---

## Project Structure

```
tesla-autopilot-sensor-fusion/
├── src/
│   ├── sensors/
│   │   ├── camera_rig.py          # 8-Camera 360° Surround Vision Suite
│   │   ├── imu_gps_sensor.py      # 6-DOF IMU, RTK-GPS, WGS-84/ENU & Wheel Odometry
│   │   └── ultrasonic_radar.py    # 12-ch Ultrasonic Array & 77GHz Forward Radar
│   ├── perception/
│   │   ├── bev_transformer.py     # Bird's-Eye-View (BEV) Occupancy Network
│   │   ├── lane_detector.py       # Polynomial Lane Boundaries & Road Curvature
│   │   └── traffic_semantics.py   # Traffic Signals, Signs & Right-of-Way Logic
│   ├── fusion/
│   │   └── ekf_fusion.py          # 9-State Extended Kalman Filter Estimator
│   ├── tracking/
│   │   ├── object_tracker.py      # 3D Multi-Object Tracking (MOT) with Kalman Filter
│   │   └── risk_assessment.py     # TTC, FCW & Automatic Emergency Braking (AEB)
│   ├── planning/
│   │   └── trajectory_planner.py  # Quintic Jerk-Minimizing Trajectory Planner
│   ├── control/
│   │   ├── mpc_controller.py      # Stanley Lateral & Longitudinal PID Controllers
│   │   └── can_bus_interface.py   # Automotive CAN Bus Frame Encoders
│   └── simulation/
│       ├── simulator.py           # Closed-Loop Autonomous Driving Simulator
│       └── dashboard.py           # Terminal Cockpit HUD & ASCII Dashboard
├── tests/
│   └── test_all_modules.py        # Complete Pytest Verification Suite
├── examples/
│   ├── run_autopilot_simulation.py# Standalone Demo Script
│   └── web_visualizer.html        # Interactive 2D/3D Web Visualizer
├── main.py                        # Top-level CLI Entry Point
├── pyproject.toml                 # Package Configuration
└── requirements.txt               # Dependencies
```

---

## Quickstart Guide

### 1. Installation
```bash
git clone https://github.com/Sanskarsoni-013/tesla-autopilot-sensor-fusion.git
cd tesla-autopilot-sensor-fusion
pip install -r requirements.txt
```

### 2. Run Autonomous Simulation in Terminal
```bash
# Highway Cruise Scenario
python main.py --scenario highway_cruise --steps 80

# Obstacle Avoidance (Cut-in Vehicle)
python main.py --scenario obstacle_avoidance --steps 80

# Automatic Emergency Braking (AEB Hard Stop)
python main.py --scenario emergency_aeb_stop --steps 60
```

### 3. Launch Interactive Web Visualizer
Open `examples/web_visualizer.html` directly in your browser.

### 4. Run Pytest Test Suite
```bash
pytest tests/ -v
```

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
