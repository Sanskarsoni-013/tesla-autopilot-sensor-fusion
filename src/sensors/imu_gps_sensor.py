"""
IMU, RTK-GPS, and Wheel Odometry Sensor Models.
Simulates realistic 6-DOF inertial measurement, satellite positioning,
WGS-84 to Local ENU coordinates, and wheel tick odometry with sensor noise.
"""

from dataclasses import dataclass, field
import math
from typing import Optional, Tuple
import numpy as np


# WGS-84 Ellipsoid constants
WGS84_A = 6378137.0          # Semi-major axis (meters)
WGS84_F = 1.0 / 298.257223563  # Flattening
WGS84_B = WGS84_A * (1.0 - WGS84_F)
WGS84_E2 = 1.0 - (WGS84_B**2 / WGS84_A**2)  # First eccentricity squared


@dataclass
class IMUReading:
    timestamp: float
    accel_mps2: np.ndarray        # [ax, ay, az] in vehicle body frame (forward, left, up)
    gyro_radps: np.ndarray        # [wx, wy, wz] (roll rate, pitch rate, yaw rate)
    temperature_c: float = 25.0
    bias_accel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bias_gyro: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class GPSReading:
    timestamp: float
    lat_deg: float
    lon_deg: float
    alt_m: float
    heading_deg: float
    speed_mps: float
    enu_position_m: np.ndarray    # [East, North, Up] in meters relative to reference origin
    hdop: float = 0.8             # Horizontal Dilution of Precision
    fix_quality: int = 4          # 4 = RTK Fixed, 1 = Standard GPS, 0 = Invalid


@dataclass
class WheelOdometryReading:
    timestamp: float
    speed_mps: float
    wheel_speeds_mps: np.ndarray  # [FL, FR, RL, RR]
    wheel_ticks: np.ndarray       # [FL, FR, RL, RR] cumulative encoder pulses
    steering_angle_rad: float     # Road wheel angle (positive = left turn)


class CoordinateTransformer:
    """
    Converts Geodetic (Lat, Lon, Alt) to Local Cartesian East-North-Up (ENU).
    """

    def __init__(self, ref_lat_deg: float, ref_lon_deg: float, ref_alt_m: float):
        self.ref_lat = math.radians(ref_lat_deg)
        self.ref_lon = math.radians(ref_lon_deg)
        self.ref_alt = ref_alt_m
        self.ref_ecef = self.geodetic_to_ecef(ref_lat_deg, ref_lon_deg, ref_alt_m)

    @staticmethod
    def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
        phi = math.radians(lat_deg)
        lam = math.radians(lon_deg)
        sin_phi = math.sin(phi)
        cos_phi = math.cos(phi)
        sin_lam = math.sin(lam)
        cos_lam = math.cos(lam)

        N = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_phi**2)

        x = (N + alt_m) * cos_phi * cos_lam
        y = (N + alt_m) * cos_phi * sin_lam
        z = (N * (1.0 - WGS84_E2) + alt_m) * sin_phi
        return np.array([x, y, z])

    def ecef_to_enu(self, ecef_point: np.ndarray) -> np.ndarray:
        d = ecef_point - self.ref_ecef
        sin_phi = math.sin(self.ref_lat)
        cos_phi = math.cos(self.ref_lat)
        sin_lam = math.sin(self.ref_lon)
        cos_lam = math.cos(self.ref_lon)

        east = -sin_lam * d[0] + cos_lam * d[1]
        north = -sin_phi * cos_lam * d[0] - sin_phi * sin_lam * d[1] + cos_phi * d[2]
        up = cos_phi * cos_lam * d[0] + cos_phi * sin_lam * d[1] + sin_phi * d[2]
        return np.array([east, north, up])

    def geodetic_to_enu(self, lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
        ecef = self.geodetic_to_ecef(lat_deg, lon_deg, alt_m)
        return self.ecef_to_enu(ecef)


class InertialNavSimulator:
    """
    Generates synthetic IMU, RTK-GPS, and Wheel Odometry measurements
    with configurable Gaussian noise and bias walk.
    """

    def __init__(
        self,
        ref_lat: float = 37.3861,
        ref_lon: float = -122.0839,
        ref_alt: float = 30.0,
        imu_accel_noise_std: float = 0.05,
        imu_gyro_noise_std: float = 0.002,
        gps_pos_noise_std: float = 0.03,
        wheel_speed_noise_std: float = 0.02,
        seed: Optional[int] = 42
    ):
        self.transformer = CoordinateTransformer(ref_lat, ref_lon, ref_alt)
        self.rng = np.random.RandomState(seed)

        self.accel_noise_std = imu_accel_noise_std
        self.gyro_noise_std = imu_gyro_noise_std
        self.gps_noise_std = gps_pos_noise_std
        self.wheel_noise_std = wheel_speed_noise_std

        # Sensor biases
        self.accel_bias = np.array([0.015, -0.010, 0.02])
        self.gyro_bias = np.array([0.0005, -0.0003, 0.001])
        self.wheel_pulses_per_meter = 48.0
        self.cum_ticks = np.zeros(4, dtype=np.int64)

    def sample_imu(
        self,
        timestamp: float,
        true_accel_body: np.ndarray,
        true_angular_velocity: np.ndarray
    ) -> IMUReading:
        # Add gravity component (9.81 m/s^2 on Z-axis in resting state)
        gravity_vector = np.array([0.0, 0.0, 9.80665])
        accel_with_gravity = true_accel_body + gravity_vector

        # Add bias + Gaussian noise
        noise_a = self.rng.normal(0.0, self.accel_noise_std, size=3)
        noise_w = self.rng.normal(0.0, self.gyro_noise_std, size=3)

        measured_accel = accel_with_gravity + self.accel_bias + noise_a
        measured_gyro = true_angular_velocity + self.gyro_bias + noise_w

        return IMUReading(
            timestamp=timestamp,
            accel_mps2=measured_accel,
            gyro_radps=measured_gyro,
            bias_accel=self.accel_bias.copy(),
            bias_gyro=self.gyro_bias.copy()
        )

    def sample_gps(
        self,
        timestamp: float,
        true_enu: np.ndarray,
        true_speed: float,
        true_heading_rad: float
    ) -> GPSReading:
        noise_pos = self.rng.normal(0.0, self.gps_noise_std, size=3)
        noisy_enu = true_enu + noise_pos

        # Approximate conversion back to lat/lon for telemetry
        # 1 deg lat ~= 111,139 m, 1 deg lon ~= 111,139 * cos(lat) m
        lat = math.degrees(self.transformer.ref_lat) + (noisy_enu[1] / 111139.0)
        lon = math.degrees(self.transformer.ref_lon) + (
            noisy_enu[0] / (111139.0 * math.cos(self.transformer.ref_lat))
        )
        alt = self.transformer.ref_alt + noisy_enu[2]

        speed_noise = self.rng.normal(0.0, 0.05)
        heading_noise = math.degrees(self.rng.normal(0.0, 0.005))

        return GPSReading(
            timestamp=timestamp,
            lat_deg=lat,
            lon_deg=lon,
            alt_m=alt,
            heading_deg=(math.degrees(true_heading_rad) + heading_noise) % 360.0,
            speed_mps=max(0.0, true_speed + speed_noise),
            enu_position_m=noisy_enu,
            hdop=0.75,
            fix_quality=4
        )

    def sample_wheel_odometry(
        self,
        timestamp: float,
        true_speed: float,
        steering_angle_rad: float,
        dt: float
    ) -> WheelOdometryReading:
        wheelbase = 2.96  # Tesla Model 3 / Y wheelbase
        track_width = 1.63

        # Differential wheel speeds via Ackerman kinematics
        if abs(steering_angle_rad) > 1e-4:
            turn_radius = wheelbase / math.tan(steering_angle_rad)
            r_fl = turn_radius - (track_width / 2.0)
            r_fr = turn_radius + (track_width / 2.0)
            r_rl = turn_radius - (track_width / 2.0)
            r_rr = turn_radius + (track_width / 2.0)
            w_speeds = true_speed * np.array([
                r_fl / turn_radius,
                r_fr / turn_radius,
                r_rl / turn_radius,
                r_rr / turn_radius
            ])
        else:
            w_speeds = np.full(4, true_speed)

        # Add slip / measurement noise
        noise = self.rng.normal(0.0, self.wheel_noise_std, size=4)
        measured_speeds = np.maximum(0.0, w_speeds + noise)

        # Increment cumulative encoder ticks
        new_ticks = (measured_speeds * dt * self.wheel_pulses_per_meter).astype(np.int64)
        self.cum_ticks += new_ticks

        avg_speed = float(np.mean(measured_speeds))

        return WheelOdometryReading(
            timestamp=timestamp,
            speed_mps=avg_speed,
            wheel_speeds_mps=measured_speeds,
            wheel_ticks=self.cum_ticks.copy(),
            steering_angle_rad=steering_angle_rad
        )
