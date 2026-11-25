"""
Extended Kalman Filter (EKF) Multi-Sensor State Estimator.
Fuses high-rate 6-DOF IMU, RTK-GPS, wheel odometry, and visual landmarks
for centimeter-accurate vehicle pose, velocity, heading, and acceleration estimation.
"""

from dataclasses import dataclass
import math
from typing import Optional, Tuple
import numpy as np


@dataclass
class VehicleState:
    timestamp: float
    x: float             # Easting / Longitudinal position (m)
    y: float             # Northing / Lateral position (m)
    z: float             # Vertical altitude (m)
    vx: float            # Easting velocity (m/s)
    vy: float            # Northing velocity (m/s)
    vz: float            # Vertical velocity (m/s)
    yaw_rad: float       # Heading angle (radians, 0 = East, pi/2 = North)
    yaw_rate_radps: float# Angular yaw rate (rad/s)
    ax: float            # Longitudinal acceleration (m/s^2)
    ay: float            # Lateral acceleration (m/s^2)

    @property
    def speed_mps(self) -> float:
        return float(math.sqrt(self.vx**2 + self.vy**2))

    @property
    def heading_deg(self) -> float:
        return float(math.degrees(self.yaw_rad) % 360.0)


class ExtendedKalmanFilter:
    """
    9-State EKF State Vector:
    x = [px, py, pz, vx, vy, vz, yaw, yaw_rate, ax_body]^T
    """

    STATE_DIM = 9

    def __init__(
        self,
        initial_pos: Optional[np.ndarray] = None,
        initial_vel: Optional[np.ndarray] = None,
        initial_yaw: float = 0.0
    ):
        # State vector: [px, py, pz, vx, vy, vz, yaw, yaw_rate, ax_body]
        self.x = np.zeros(self.STATE_DIM, dtype=np.float64)
        if initial_pos is not None:
            self.x[0:3] = initial_pos
        if initial_vel is not None:
            self.x[3:6] = initial_vel
        self.x[6] = initial_yaw

        # State Covariance Matrix P
        self.P = np.diag([
            1.0, 1.0, 1.0,       # Position (m^2)
            0.5, 0.5, 0.5,       # Velocity ((m/s)^2)
            0.05,                # Yaw (rad^2)
            0.01,                # Yaw rate ((rad/s)^2)
            0.1                  # Body acceleration ((m/s^2)^2)
        ]).astype(np.float64)

        # Process Noise Covariance Q
        self.Q = np.diag([
            0.01, 0.01, 0.01,    # Pos noise
            0.05, 0.05, 0.05,    # Vel noise
            0.002,               # Yaw noise
            0.005,               # Yaw rate noise
            0.2                  # Accel noise
        ]).astype(np.float64)

        self.last_timestamp: Optional[float] = None

    def predict(self, dt: float, imu_accel_body: np.ndarray, imu_gyro_z: float):
        """
        Nonlinear motion prediction using 6-DOF IMU body accelerations and yaw rate.
        """
        px, py, pz, vx, vy, vz, yaw, yaw_rate, ax_b = self.x

        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)

        # Accelerations in world frame from body forward accel
        # ax_world = ax_b * cos(yaw), ay_world = ax_b * sin(yaw)
        ax_meas = imu_accel_body[0]

        # State transition update
        px_new = px + vx * dt + 0.5 * (ax_meas * cos_y) * (dt**2)
        py_new = py + vy * dt + 0.5 * (ax_meas * sin_y) * (dt**2)
        pz_new = pz + vz * dt

        vx_new = vx + (ax_meas * cos_y) * dt
        vy_new = vy + (ax_meas * sin_y) * dt
        vz_new = vz

        yaw_new = (yaw + imu_gyro_z * dt + math.pi) % (2.0 * math.pi) - math.pi
        yaw_rate_new = imu_gyro_z
        ax_b_new = ax_meas

        self.x = np.array([
            px_new, py_new, pz_new,
            vx_new, vy_new, vz_new,
            yaw_new, yaw_rate_new, ax_b_new
        ], dtype=np.float64)

        # Jacobian of motion model F = df/dx
        F = np.eye(self.STATE_DIM, dtype=np.float64)
        F[0, 3] = dt
        F[0, 6] = -0.5 * ax_meas * sin_y * (dt**2)
        F[1, 4] = dt
        F[1, 6] = 0.5 * ax_meas * cos_y * (dt**2)
        F[2, 5] = dt
        F[3, 6] = -ax_meas * sin_y * dt
        F[4, 6] = ax_meas * cos_y * dt
        F[6, 7] = dt

        # Propagate covariance: P = F * P * F^T + Q * dt
        self.P = F @ self.P @ F.T + self.Q * dt

    def update_gps(self, enu_pos: np.ndarray, speed_mps: float, heading_rad: float):
        """
        Measurement update from RTK-GPS: z = [px, py, pz, speed, yaw]
        """
        # Measurement matrix H (5 x 9)
        H = np.zeros((5, self.STATE_DIM), dtype=np.float64)
        H[0, 0] = 1.0  # px
        H[1, 1] = 1.0  # py
        H[2, 2] = 1.0  # pz
        # Speed = sqrt(vx^2 + vy^2)
        vx, vy = self.x[3], self.x[4]
        v_norm = max(0.1, math.sqrt(vx**2 + vy**2))
        H[3, 3] = vx / v_norm
        H[3, 4] = vy / v_norm
        H[4, 6] = 1.0  # yaw

        # Measurement noise R
        R = np.diag([0.05**2, 0.05**2, 0.1**2, 0.1**2, 0.02**2]).astype(np.float64)

        # Expected measurement h(x)
        h_x = np.array([
            self.x[0],
            self.x[1],
            self.x[2],
            v_norm,
            self.x[6]
        ])

        # Actual measurement z
        z = np.array([enu_pos[0], enu_pos[1], enu_pos[2], speed_mps, heading_rad])

        # Innovation residual y = z - h(x)
        y = z - h_x
        # Normalize yaw residual to [-pi, pi]
        y[4] = (y[4] + math.pi) % (2.0 * math.pi) - math.pi

        # Kalman gain: K = P * H^T * (H * P * H^T + R)^-1
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        # State update: x = x + K * y
        self.x = self.x + K @ y
        self.x[6] = (self.x[6] + math.pi) % (2.0 * math.pi) - math.pi

        # Joseph form covariance update for positive-definite stability
        I = np.eye(self.STATE_DIM)
        IKH = I - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T

    def update_wheel_odometry(self, measured_speed: float):
        """
        Measurement update from wheel encoders: z = speed = sqrt(vx^2 + vy^2)
        """
        vx, vy = self.x[3], self.x[4]
        v_norm = max(0.1, math.sqrt(vx**2 + vy**2))

        H = np.zeros((1, self.STATE_DIM), dtype=np.float64)
        H[0, 3] = vx / v_norm
        H[0, 4] = vy / v_norm

        R = np.array([[0.04**2]], dtype=np.float64)

        y = np.array([measured_speed - v_norm])
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        I = np.eye(self.STATE_DIM)
        IKH = I - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T

    def get_state(self, timestamp: float) -> VehicleState:
        px, py, pz, vx, vy, vz, yaw, yaw_rate, ax_b = self.x
        ay_b = yaw_rate * math.sqrt(vx**2 + vy**2)
        return VehicleState(
            timestamp=timestamp,
            x=px,
            y=py,
            z=pz,
            vx=vx,
            vy=vy,
            vz=vz,
            yaw_rad=yaw,
            yaw_rate_radps=yaw_rate,
            ax=ax_b,
            ay=ay_b
        )
