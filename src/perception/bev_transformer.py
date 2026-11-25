"""
Bird's-Eye-View (BEV) Transformer & Occupancy Network.
Transforms multi-camera surround optical perspective into a unified 3D/2D
ego-centric spatial voxel occupancy grid and static/dynamic obstacle map.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class BEVGridConfig:
    x_min: float = -20.0   # 20m behind rear axle
    x_max: float = 80.0    # 80m ahead
    y_min: float = -20.0   # 20m right
    y_max: float = 20.0    # 20m left
    z_min: float = -1.0    # Road level
    z_max: float = 4.0     # 4m overhead
    resolution_m: float = 0.5  # 0.5m grid cell size

    @property
    def nx(self) -> int:
        return int(round((self.x_max - self.x_min) / self.resolution_m))

    @property
    def ny(self) -> int:
        return int(round((self.y_max - self.y_min) / self.resolution_m))

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.nx, self.ny)


class BEVTransformer:
    """
    Constructs Bird's-Eye-View Occupancy Grid from surround vision,
    ultrasonics, and forward radar inputs.
    """

    def __init__(self, config: Optional[BEVGridConfig] = None):
        self.config = config if config is not None else BEVGridConfig()
        # Probability grids: 0.0 = free space, 1.0 = occupied
        self.occupancy_grid = np.zeros(self.config.shape, dtype=np.float32)
        self.velocity_x_grid = np.zeros(self.config.shape, dtype=np.float32)
        self.velocity_y_grid = np.zeros(self.config.shape, dtype=np.float32)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.config.shape

    def world_to_grid(self, x_veh: float, y_veh: float) -> Optional[Tuple[int, int]]:
        """
        Converts vehicle Cartesian coordinates (meters) to 2D BEV grid indices (ix, iy).
        """
        if (self.config.x_min <= x_veh < self.config.x_max and
                self.config.y_min <= y_veh < self.config.y_max):
            ix = int((x_veh - self.config.x_min) / self.config.resolution_m)
            iy = int((y_veh - self.config.y_min) / self.config.resolution_m)
            ix = max(0, min(self.config.nx - 1, ix))
            iy = max(0, min(self.config.ny - 1, iy))
            return (ix, iy)
        return None

    def grid_to_world(self, ix: int, iy: int) -> Tuple[float, float]:
        """
        Converts 2D BEV grid indices (ix, iy) back to vehicle Cartesian coordinates (meters).
        """
        x_veh = self.config.x_min + (ix + 0.5) * self.config.resolution_m
        y_veh = self.config.y_min + (iy + 0.5) * self.config.resolution_m
        return (x_veh, y_veh)

    def reset_grid(self, decay_factor: float = 0.85):
        """
        Decays previous occupancy grid state (temporal memory persistence).
        """
        self.occupancy_grid *= decay_factor
        self.velocity_x_grid *= decay_factor
        self.velocity_y_grid *= decay_factor

    def insert_obstacle_bbox(
        self,
        center_x: float,
        center_y: float,
        length_m: float,
        width_m: float,
        heading_rad: float,
        velocity_vx: float = 0.0,
        velocity_vy: float = 0.0,
        confidence: float = 0.95
    ):
        """
        Splats an oriented 2D bounding box (vehicle/pedestrian/cyclist) into the BEV occupancy grid.
        """
        cos_h = np.cos(heading_rad)
        sin_h = np.sin(heading_rad)
        half_l = length_m / 2.0
        half_w = width_m / 2.0

        # Scan enclosing bounding circle
        radius = np.sqrt(half_l**2 + half_w**2)
        min_x = max(self.config.x_min, center_x - radius)
        max_x = min(self.config.x_max, center_x + radius)
        min_y = max(self.config.y_min, center_y - radius)
        max_y = min(self.config.y_max, center_y + radius)

        ix_start = max(0, int((min_x - self.config.x_min) / self.config.resolution_m))
        ix_end = min(self.config.nx, int((max_x - self.config.x_min) / self.config.resolution_m) + 1)
        iy_start = max(0, int((min_y - self.config.y_min) / self.config.resolution_m))
        iy_end = min(self.config.ny, int((max_y - self.config.y_min) / self.config.resolution_m) + 1)

        for ix in range(ix_start, ix_end):
            for iy in range(iy_start, iy_end):
                gx, gy = self.grid_to_world(ix, iy)
                # Rotate point into box local frame
                dx = gx - center_x
                dy = gy - center_y
                local_x = dx * cos_h + dy * sin_h
                local_y = -dx * sin_h + dy * cos_h

                if abs(local_x) <= half_l and abs(local_y) <= half_w:
                    self.occupancy_grid[ix, iy] = max(self.occupancy_grid[ix, iy], confidence)
                    self.velocity_x_grid[ix, iy] = velocity_vx
                    self.velocity_y_grid[ix, iy] = velocity_vy

    def insert_point_cloud(self, points_3d: np.ndarray, prob_value: float = 0.8):
        """
        Maps a set of 3D points (e.g. from radar / stereo depth) directly onto the BEV grid.
        """
        for pt in points_3d:
            idx = self.world_to_grid(pt[0], pt[1])
            if idx is not None:
                ix, iy = idx
                self.occupancy_grid[ix, iy] = max(self.occupancy_grid[ix, iy], prob_value)

    def is_corridor_free(
        self,
        trajectory_waypoints: List[Tuple[float, float]],
        vehicle_safety_margin_m: float = 1.2
    ) -> bool:
        """
        Checks whether a candidate trajectory path penetrates occupied BEV grid cells.
        """
        half_margin_cells = int(np.ceil(vehicle_safety_margin_m / self.config.resolution_m))
        for wp_x, wp_y in trajectory_waypoints:
            idx = self.world_to_grid(wp_x, wp_y)
            if idx is None:
                continue
            ix, iy = idx
            for dx in range(-half_margin_cells, half_margin_cells + 1):
                for dy in range(-half_margin_cells, half_margin_cells + 1):
                    nx = ix + dx
                    ny = iy + dy
                    if 0 <= nx < self.config.nx and 0 <= ny < self.config.ny:
                        if self.occupancy_grid[nx, ny] > 0.50:
                            return False
        return True
