"""
Tesla Hardware 3/4 8-Camera Surround Vision Suite.
Models 360-degree optical coverage, intrinsic/extrinsic geometry, and perspective projection.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple
import math
import numpy as np


class CameraType(Enum):
    FRONT_WIDE = "front_wide"
    FRONT_MAIN = "front_main"
    FRONT_NARROW = "front_narrow"
    B_PILLAR_LEFT = "pillar_left"
    B_PILLAR_RIGHT = "pillar_right"
    REPEATER_LEFT = "repeater_left"
    REPEATER_RIGHT = "repeater_right"
    REAR_BACKUP = "rear_backup"


@dataclass
class CameraParams:
    name: str
    fov_deg: float
    max_range_m: float
    resolution: Tuple[int, int]  # (width, height)
    position_m: np.ndarray       # [x, y, z] relative to rear axle center
    orientation_rad: np.ndarray  # [yaw, pitch, roll]

    @property
    def focal_length_px(self) -> float:
        # f = (width / 2) / tan(fov / 2)
        half_fov = math.radians(self.fov_deg / 2.0)
        return (self.resolution[0] / 2.0) / math.tan(half_fov)

    @property
    def intrinsic_matrix(self) -> np.ndarray:
        f = self.focal_length_px
        cx = self.resolution[0] / 2.0
        cy = self.resolution[1] / 2.0
        return np.array([
            [f, 0.0, cx],
            [0.0, f, cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

    @property
    def rotation_matrix(self) -> np.ndarray:
        yaw, pitch, roll = self.orientation_rad
        # Rz * Ry * Rx
        Rz = np.array([
            [math.cos(yaw), -math.sin(yaw), 0],
            [math.sin(yaw), math.cos(yaw), 0],
            [0, 0, 1]
        ])
        Ry = np.array([
            [math.cos(pitch), 0, math.sin(pitch)],
            [0, 1, 0],
            [-math.sin(pitch), 0, math.cos(pitch)]
        ])
        Rx = np.array([
            [1, 0, 0],
            [0, math.cos(roll), -math.sin(roll)],
            [0, math.sin(roll), math.cos(roll)]
        ])
        return Rz @ Ry @ Rx


class CameraRig:
    """
    Manages Tesla 8-Camera 360-degree Vision Rig.
    """

    def __init__(self):
        self.cameras: Dict[CameraType, CameraParams] = {
            CameraType.FRONT_WIDE: CameraParams(
                name="Front Wide Fisheye",
                fov_deg=120.0,
                max_range_m=60.0,
                resolution=(1280, 960),
                position_m=np.array([2.1, 0.0, 1.3]),
                orientation_rad=np.array([0.0, math.radians(3.0), 0.0])
            ),
            CameraType.FRONT_MAIN: CameraParams(
                name="Front Main Telephoto",
                fov_deg=50.0,
                max_range_m=150.0,
                resolution=(1280, 960),
                position_m=np.array([2.1, 0.0, 1.3]),
                orientation_rad=np.array([0.0, math.radians(1.0), 0.0])
            ),
            CameraType.FRONT_NARROW: CameraParams(
                name="Front Narrow Long Range",
                fov_deg=35.0,
                max_range_m=250.0,
                resolution=(1280, 960),
                position_m=np.array([2.1, 0.0, 1.3]),
                orientation_rad=np.array([0.0, 0.0, 0.0])
            ),
            CameraType.B_PILLAR_LEFT: CameraParams(
                name="B-Pillar Left Forward Oblique",
                fov_deg=90.0,
                max_range_m=80.0,
                resolution=(1280, 960),
                position_m=np.array([1.2, 0.9, 1.2]),
                orientation_rad=np.array([math.radians(65.0), 0.0, 0.0])
            ),
            CameraType.B_PILLAR_RIGHT: CameraParams(
                name="B-Pillar Right Forward Oblique",
                fov_deg=90.0,
                max_range_m=80.0,
                resolution=(1280, 960),
                position_m=np.array([1.2, -0.9, 1.2]),
                orientation_rad=np.array([math.radians(-65.0), 0.0, 0.0])
            ),
            CameraType.REPEATER_LEFT: CameraParams(
                name="Fender Repeater Left Rear Oblique",
                fov_deg=90.0,
                max_range_m=100.0,
                resolution=(1280, 960),
                position_m=np.array([1.8, 1.0, 0.8]),
                orientation_rad=np.array([math.radians(145.0), math.radians(2.0), 0.0])
            ),
            CameraType.REPEATER_RIGHT: CameraParams(
                name="Fender Repeater Right Rear Oblique",
                fov_deg=90.0,
                max_range_m=100.0,
                resolution=(1280, 960),
                position_m=np.array([1.8, -1.0, 0.8]),
                orientation_rad=np.array([math.radians(-145.0), math.radians(2.0), 0.0])
            ),
            CameraType.REAR_BACKUP: CameraParams(
                name="Rear Center Trunk View",
                fov_deg=120.0,
                max_range_m=50.0,
                resolution=(1280, 960),
                position_m=np.array([-1.0, 0.0, 0.9]),
                orientation_rad=np.array([math.radians(180.0), math.radians(-10.0), 0.0])
            )
        }

    def project_point_to_camera(self, point_vehicle_frame: np.ndarray, cam_type: CameraType) -> Tuple[bool, np.ndarray, float]:
        """
        Projects 3D vehicle frame point [X, Y, Z] to 2D image coordinates (u, v).
        Returns: (is_visible, image_coords, distance_m)
        """
        cam = self.cameras[cam_type]
        R_cam = cam.rotation_matrix
        T_cam = cam.position_m

        # Transform to camera frame: P_cam = R^T * (P_veh - T)
        P_rel = point_vehicle_frame - T_cam
        P_cam = R_cam.T @ P_rel

        # Camera frame standard: X right, Y down, Z forward
        z_forward = P_cam[0]  # Vehicle forward aligns with camera X in base rotation
        dist = np.linalg.norm(P_rel)

        # Basic depth check
        if dist > cam.max_range_m or dist < 0.5:
            return False, np.array([0.0, 0.0]), dist

        # Angle check relative to camera principal axis
        angle_deg = math.degrees(math.atan2(np.linalg.norm(P_cam[1:]), max(0.001, P_cam[0])))
        if angle_deg > (cam.fov_deg / 2.0):
            return False, np.array([0.0, 0.0]), dist

        # Perspective projection
        K = cam.intrinsic_matrix
        f = cam.focal_length_px
        cx, cy = cam.resolution[0] / 2.0, cam.resolution[1] / 2.0

        u = cx + (P_cam[1] / max(0.1, P_cam[0])) * f
        v = cy - (P_cam[2] / max(0.1, P_cam[0])) * f

        is_in_bounds = bool((0 <= u < cam.resolution[0]) and (0 <= v < cam.resolution[1]))
        return is_in_bounds, np.array([u, v]), float(dist)

    def find_observing_cameras(self, target_3d: np.ndarray) -> List[Tuple[CameraType, float]]:
        """
        Returns list of cameras that have a direct line-of-sight to the 3D obstacle.
        """
        observing = []
        for ctype in CameraType:
            visible, _, dist = self.project_point_to_camera(target_3d, ctype)
            if visible:
                observing.append((ctype, dist))
        return sorted(observing, key=lambda x: x[1])
