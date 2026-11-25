"""
Tesla 12-Channel Ultrasonic Sensor Suite & Forward Long-Range Radar Model.
Models near-field surround obstacle proximity detection (<8m) and
high-frequency radar Doppler velocity/ranging (<250m).
"""

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Dict, List, Optional, Tuple
import numpy as np


class AlertZone(Enum):
    SAFE = "safe"              # > 2.5 m
    CAUTION = "caution"        # 1.2 m - 2.5 m
    WARNING = "warning"        # 0.5 m - 1.2 m
    CRITICAL_STOP = "critical" # < 0.5 m


class UltrasonicPosition(Enum):
    FRONT_LEFT_CORNER = "flc"
    FRONT_LEFT_MID = "flm"
    FRONT_CENTER_LEFT = "fcl"
    FRONT_CENTER_RIGHT = "fcr"
    FRONT_RIGHT_MID = "frm"
    FRONT_RIGHT_CORNER = "frc"
    REAR_LEFT_CORNER = "rlc"
    REAR_LEFT_MID = "rlm"
    REAR_CENTER_LEFT = "rcl"
    REAR_CENTER_RIGHT = "rcr"
    REAR_RIGHT_MID = "rrm"
    REAR_RIGHT_CORNER = "rrc"


@dataclass
class UltrasonicReading:
    sensor_id: UltrasonicPosition
    distance_m: float
    confidence: float
    alert_zone: AlertZone
    bearing_rad: float
    sensor_pos_m: np.ndarray


@dataclass
class RadarTarget:
    target_id: int
    range_m: float            # Distance along line-of-sight
    azimuth_rad: float        # Lateral angle relative to vehicle bore-sight
    elevation_rad: float      # Vertical angle
    doppler_radial_mps: float # Relative radial velocity (+ moving away, - approaching)
    rcs_dbsm: float           # Radar Cross Section in dBm^2 (e.g. 10=car, 0=bike, -10=ped)
    confidence: float
    cartesian_xyz_m: np.ndarray


class UltrasonicArray:
    """
    12-Channel Tesla Bumper Ultrasonic Transducer Array.
    Frequency: ~48 kHz, Range: 0.15m to 8.0m, FOV: ~70 deg per sensor.
    """

    def __init__(self):
        # Coordinates in vehicle frame [X_forward, Y_left, Z_up]
        self.sensor_coords: Dict[UltrasonicPosition, Tuple[np.ndarray, float]] = {
            # Front bumper: X ~= +2.3m
            UltrasonicPosition.FRONT_LEFT_CORNER: (np.array([2.15, 0.90, 0.45]), math.radians(75.0)),
            UltrasonicPosition.FRONT_LEFT_MID: (np.array([2.25, 0.60, 0.45]), math.radians(45.0)),
            UltrasonicPosition.FRONT_CENTER_LEFT: (np.array([2.32, 0.20, 0.45]), math.radians(15.0)),
            UltrasonicPosition.FRONT_CENTER_RIGHT: (np.array([2.32, -0.20, 0.45]), math.radians(-15.0)),
            UltrasonicPosition.FRONT_RIGHT_MID: (np.array([2.25, -0.60, 0.45]), math.radians(-45.0)),
            UltrasonicPosition.FRONT_RIGHT_CORNER: (np.array([2.15, -0.90, 0.45]), math.radians(-75.0)),

            # Rear bumper: X ~= -1.05m
            UltrasonicPosition.REAR_LEFT_CORNER: (np.array([-0.95, 0.90, 0.45]), math.radians(105.0)),
            UltrasonicPosition.REAR_LEFT_MID: (np.array([-1.02, 0.60, 0.45]), math.radians(135.0)),
            UltrasonicPosition.REAR_CENTER_LEFT: (np.array([-1.05, 0.20, 0.45]), math.radians(165.0)),
            UltrasonicPosition.REAR_CENTER_RIGHT: (np.array([-1.05, -0.20, 0.45]), math.radians(-165.0)),
            UltrasonicPosition.REAR_RIGHT_MID: (np.array([-1.02, -0.60, 0.45]), math.radians(-135.0)),
            UltrasonicPosition.REAR_RIGHT_CORNER: (np.array([-0.95, -0.90, 0.45]), math.radians(-105.0)),
        }
        self.max_range = 8.0
        self.min_range = 0.15
        self.half_fov = math.radians(45.0)

    @staticmethod
    def classify_distance(d: float) -> AlertZone:
        if d < 0.50:
            return AlertZone.CRITICAL_STOP
        elif d < 1.20:
            return AlertZone.WARNING
        elif d < 2.50:
            return AlertZone.CAUTION
        return AlertZone.SAFE

    def scan(self, obstacle_points_3d: List[np.ndarray]) -> Dict[UltrasonicPosition, UltrasonicReading]:
        """
        Calculates echo reflection for all 12 sensors against a cluster of 3D obstacle points.
        """
        readings: Dict[UltrasonicPosition, UltrasonicReading] = {}

        for sensor_id, (pos, heading) in self.sensor_coords.items():
            min_dist = self.max_range
            sensor_dir = np.array([math.cos(heading), math.sin(heading), 0.0])

            for pt in obstacle_points_3d:
                rel = pt - pos
                dist = float(np.linalg.norm(rel))
                if dist < self.min_range or dist > self.max_range:
                    continue

                # Angular check within beam cone
                angle = math.acos(np.clip(np.dot(rel[:2], sensor_dir[:2]) / max(1e-4, np.linalg.norm(rel[:2])), -1.0, 1.0))
                if angle <= self.half_fov:
                    if dist < min_dist:
                        min_dist = dist

            zone = self.classify_distance(min_dist)
            conf = max(0.0, 1.0 - (min_dist / self.max_range)) if min_dist < self.max_range else 0.0

            readings[sensor_id] = UltrasonicReading(
                sensor_id=sensor_id,
                distance_m=min_dist,
                confidence=conf,
                alert_zone=zone,
                bearing_rad=heading,
                sensor_pos_m=pos
            )

        return readings

    def get_closest_distance(self, readings: Dict[UltrasonicPosition, UltrasonicReading]) -> Tuple[float, AlertZone]:
        """
        Returns minimum detected distance across all 12 sensors and the highest alert level.
        """
        min_d = min(r.distance_m for r in readings.values())
        return min_d, self.classify_distance(min_d)


class ForwardRadarModel:
    """
    77 GHz Long-Range Automotive Radar (e.g. Continental ARS4-B / HW4 HD Radar).
    Field of view: ±45 deg (mid-range), ±10 deg (long-range up to 250m).
    Direct measurement of range, azimuth angle, and radial Doppler velocity.
    """

    def __init__(self, mount_pos_m: Optional[np.ndarray] = None):
        # Mounted at front fascia center
        self.mount_pos = mount_pos_m if mount_pos_m is not None else np.array([2.35, 0.0, 0.40])
        self.max_range = 250.0
        self.min_range = 1.0
        self.azimuth_fov_rad = math.radians(45.0)

        # Measurement uncertainties
        self.sigma_range = 0.20        # meters
        self.sigma_azimuth = 0.005     # radians (~0.3 deg)
        self.sigma_doppler = 0.10      # m/s

    def measure(
        self,
        target_positions: List[np.ndarray],
        target_velocities: List[np.ndarray],
        ego_velocity_mps: float,
        rng: Optional[np.random.RandomState] = None
    ) -> List[RadarTarget]:
        """
        Generates radar target detections with Doppler velocity and measurement noise.
        """
        if rng is None:
            rng = np.random.RandomState()

        targets: List[RadarTarget] = []
        for idx, (pos_3d, vel_3d) in enumerate(zip(target_positions, target_velocities)):
            rel_pos = pos_3d - self.mount_pos
            r_true = float(np.linalg.norm(rel_pos))

            if r_true < self.min_range or r_true > self.max_range:
                continue

            azimuth_true = math.atan2(rel_pos[1], rel_pos[0])
            if abs(azimuth_true) > self.azimuth_fov_rad:
                continue

            elevation_true = math.atan2(rel_pos[2], math.sqrt(rel_pos[0]**2 + rel_pos[1]**2))

            # Radial Doppler relative velocity: v_r = (v_target - v_ego) . u_r
            ego_vel_vec = np.array([ego_velocity_mps, 0.0, 0.0])
            rel_vel = vel_3d - ego_vel_vec
            u_r = rel_pos / r_true
            v_r_true = float(np.dot(rel_vel, u_r))

            # Add sensor noise
            r_noisy = max(0.5, r_true + rng.normal(0.0, self.sigma_range))
            azimuth_noisy = azimuth_true + rng.normal(0.0, self.sigma_azimuth)
            v_r_noisy = v_r_true + rng.normal(0.0, self.sigma_doppler)

            # Reconstruct estimated cartesian coordinate in vehicle frame
            x_est = self.mount_pos[0] + r_noisy * math.cos(azimuth_noisy) * math.cos(elevation_true)
            y_est = self.mount_pos[1] + r_noisy * math.sin(azimuth_noisy) * math.cos(elevation_true)
            z_est = self.mount_pos[2] + r_noisy * math.sin(elevation_true)

            targets.append(RadarTarget(
                target_id=idx + 1,
                range_m=r_noisy,
                azimuth_rad=azimuth_noisy,
                elevation_rad=elevation_true,
                doppler_radial_mps=v_r_noisy,
                rcs_dbsm=12.5,
                confidence=0.95,
                cartesian_xyz_m=np.array([x_est, y_est, z_est])
            ))

        return targets
