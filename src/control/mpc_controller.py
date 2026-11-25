"""
Model Predictive Control (MPC) and Stanley Lateral Steering Controller.
Actuates vehicle steering angle, throttle motor torque, and friction brakes
to follow reference trajectory waypoints with high fidelity.
"""

from dataclasses import dataclass
import math
from typing import List, Optional, Tuple
import numpy as np
from ..planning.trajectory_planner import TrajectoryPoint


@dataclass
class ControlCommand:
    timestamp: float
    steering_angle_rad: float   # Front road wheel angle (-0.6 rad to +0.6 rad)
    steering_wheel_deg: float   # Cockpit steering wheel angle (ratio 10.3)
    throttle_pct: float         # 0.0 to 100.0% powertrain motor acceleration
    brake_pressure_pct: float   # 0.0 to 100.0% hydraulic friction brake
    target_accel_mps2: float    # Commanded vehicle acceleration (-8.0 to +3.5 m/s^2)
    lateral_error_m: float      # Cross-track tracking error at front axle
    heading_error_rad: float    # Heading deviation relative to trajectory path


class StanleyLateralController:
    """
    Nonlinear Stanley Steering Controller (Stanford Autonomous Vehicle Architecture).
    Computes front steering command:
    delta = psi_err + arctan(k * e_fa / (v + epsilon))
    """

    def __init__(
        self,
        wheelbase_m: float = 2.96,     # Tesla Model 3 / Y wheelbase
        k_gain: float = 0.85,          # Cross-track error gain
        k_soft: float = 1.0,           # Softening constant (m/s)
        max_steer_rad: float = 0.55,   # ~31.5 deg max road wheel angle
        steering_ratio: float = 10.3   # Tesla Model 3 steering ratio
    ):
        self.L = wheelbase_m
        self.k = k_gain
        self.k_soft = k_soft
        self.max_steer = max_steer_rad
        self.ratio = steering_ratio

    def calculate_steering(
        self,
        current_x: float,
        current_y: float,
        current_yaw: float,
        current_v: float,
        path_points: List[TrajectoryPoint]
    ) -> Tuple[float, float, float]:
        """
        Returns: (steering_angle_rad, cross_track_error_m, heading_error_rad)
        """
        if not path_points:
            return 0.0, 0.0, 0.0

        # Front axle position: x_fa = x + L * cos(yaw), y_fa = y + L * sin(yaw)
        x_fa = current_x + self.L * math.cos(current_yaw)
        y_fa = current_y + self.L * math.sin(current_yaw)

        # Find closest point on path to front axle
        min_dist = float("inf")
        target_idx = 0
        for idx, pt in enumerate(path_points):
            dist = math.sqrt((pt.x - x_fa)**2 + (pt.y - y_fa)**2)
            if dist < min_dist:
                min_dist = dist
                target_idx = idx

        target_pt = path_points[target_idx]

        # Heading error: yaw_err = target_yaw - current_yaw
        heading_err = target_pt.yaw - current_yaw
        # Normalize to [-pi, pi]
        heading_err = (heading_err + math.pi) % (2.0 * math.pi) - math.pi

        # Cross-track error calculation:
        # Vector from target point to front axle: [dx, dy]
        dx = x_fa - target_pt.x
        dy = y_fa - target_pt.y

        # Cross product with path tangent direction [-sin(path_yaw), cos(path_yaw)]
        # Positive if front axle is to the left of the path
        cross_track_err = -dx * math.sin(target_pt.yaw) + dy * math.cos(target_pt.yaw)

        # Stanley steering law:
        # delta = heading_err + arctan( - k * cross_track_err / (v + k_soft) )
        crosstrack_steering = math.atan2(-self.k * cross_track_err, max(0.5, current_v + self.k_soft))
        steer_cmd = heading_err + crosstrack_steering

        # Clamp to physical steering limits
        steer_cmd = max(-self.max_steer, min(self.max_steer, steer_cmd))

        return steer_cmd, cross_track_err, heading_err


class LongitudinalPIDController:
    """
    Speed & Acceleration Controller with feedforward + PID closed-loop.
    Outputs throttle percentage and hydraulic brake command.
    """

    def __init__(
        self,
        kp: float = 1.2,
        ki: float = 0.15,
        kd: float = 0.08,
        max_accel_mps2: float = 3.2,
        max_decel_mps2: float = 7.5
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_accel = max_accel_mps2
        self.max_decel = max_decel_mps2

        self.integral_error = 0.0
        self.prev_error = 0.0

    def step(self, target_v: float, current_v: float, dt: float) -> Tuple[float, float, float]:
        """
        Returns: (throttle_pct, brake_pct, commanded_accel_mps2)
        """
        error = target_v - current_v
        self.integral_error += error * dt
        # Anti-windup clamping
        self.integral_error = max(-10.0, min(10.0, self.integral_error))

        derivative = (error - self.prev_error) / max(0.001, dt)
        self.prev_error = error

        accel_cmd = self.kp * error + self.ki * self.integral_error + self.kd * derivative
        accel_cmd = max(-self.max_decel, min(self.max_accel, accel_cmd))

        if accel_cmd >= 0.0:
            throttle = (accel_cmd / self.max_accel) * 100.0
            brake = 0.0
        else:
            throttle = 0.0
            brake = (abs(accel_cmd) / self.max_decel) * 100.0

        return throttle, brake, accel_cmd


class TeslaVehicleController:
    """
    Unified Tesla Autopilot Powertrain & Steering Actuation Controller.
    """

    def __init__(self):
        self.lateral_ctrl = StanleyLateralController()
        self.longitudinal_ctrl = LongitudinalPIDController()

    def step(
        self,
        timestamp: float,
        current_x: float,
        current_y: float,
        current_yaw: float,
        current_v: float,
        trajectory: List[TrajectoryPoint],
        dt: float
    ) -> ControlCommand:
        target_v = trajectory[0].v if trajectory else current_v

        steer_rad, lat_err, head_err = self.lateral_ctrl.calculate_steering(
            current_x, current_y, current_yaw, current_v, trajectory
        )
        steer_deg = math.degrees(steer_rad) * self.lateral_ctrl.ratio

        throttle, brake, accel_cmd = self.longitudinal_ctrl.step(target_v, current_v, dt)

        return ControlCommand(
            timestamp=timestamp,
            steering_angle_rad=steer_rad,
            steering_wheel_deg=steer_deg,
            throttle_pct=throttle,
            brake_pressure_pct=brake,
            target_accel_mps2=accel_cmd,
            lateral_error_m=lat_err,
            heading_error_rad=head_err
        )
