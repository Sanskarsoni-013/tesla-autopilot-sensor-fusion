"""
3D Multi-Object Tracking (MOT) with Kalman Filtering and Data Association.
Tracks surround vehicles, pedestrians, and cyclists across 8-camera surround
optical detections and forward radar returns.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Dict, List, Optional, Tuple
import numpy as np


class TrackStatus(Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    DELETED = "deleted"


class ObjectClass(Enum):
    CAR = "car"
    TRUCK = "truck"
    PEDESTRIAN = "pedestrian"
    CYCLIST = "cyclist"
    UNKNOWN = "unknown"


@dataclass
class Detection3D:
    position_3d: np.ndarray       # [x, y, z] in vehicle frame
    velocity_3d: Optional[np.ndarray] = None # [vx, vy, vz] (e.g. from Radar)
    dimensions: np.ndarray = field(default_factory=lambda: np.array([4.5, 1.8, 1.5])) # [length, width, height]
    heading_rad: float = 0.0
    obj_class: ObjectClass = ObjectClass.CAR
    confidence: float = 0.90
    sensor_source: str = "camera_rig" # "camera_rig", "radar", or "fused"


class ObjectTrack:
    """
    Individual 3D track filter using a linear Kalman filter.
    State vector: [x, y, vx, vy, heading, length, width]^T
    """

    def __init__(self, track_id: int, det: Detection3D, initial_time: float):
        self.track_id = track_id
        self.obj_class = det.obj_class
        self.dimensions = det.dimensions.copy()
        self.creation_time = initial_time
        self.last_update_time = initial_time

        # Initial state: [x, y, vx, vy, heading, l, w]
        vx = det.velocity_3d[0] if det.velocity_3d is not None else 0.0
        vy = det.velocity_3d[1] if det.velocity_3d is not None else 0.0

        self.state = np.array([
            det.position_3d[0],
            det.position_3d[1],
            vx,
            vy,
            det.heading_rad,
            det.dimensions[0],
            det.dimensions[1]
        ], dtype=np.float64)

        # Covariance
        self.P = np.diag([0.5, 0.5, 1.0, 1.0, 0.1, 0.2, 0.2]).astype(np.float64)
        self.Q = np.diag([0.1, 0.1, 0.5, 0.5, 0.02, 0.001, 0.001]).astype(np.float64)

        self.hits = 1
        self.misses = 0
        self.status = TrackStatus.TENTATIVE

    @property
    def position(self) -> np.ndarray:
        return np.array([self.state[0], self.state[1], 0.0])

    @property
    def velocity(self) -> np.ndarray:
        return np.array([self.state[2], self.state[3], 0.0])

    @property
    def speed_mps(self) -> float:
        return float(math.sqrt(self.state[2]**2 + self.state[3]**2))

    @property
    def heading(self) -> float:
        return float(self.state[4])

    @property
    def distance(self) -> float:
        return float(math.sqrt(self.state[0]**2 + self.state[1]**2))

    def predict(self, dt: float):
        # x' = x + vx * dt, y' = y + vy * dt
        F = np.eye(7, dtype=np.float64)
        F[0, 2] = dt
        F[1, 3] = dt

        self.state = F @ self.state
        self.P = F @ self.P @ F.T + self.Q * dt
        self.misses += 1

    def update(self, det: Detection3D, timestamp: float):
        self.last_update_time = timestamp
        self.hits += 1
        self.misses = 0

        if self.hits >= 3 and self.status == TrackStatus.TENTATIVE:
            self.status = TrackStatus.CONFIRMED

        # If radar provided doppler velocity, measure [x, y, vx, vy]
        if det.velocity_3d is not None:
            H = np.zeros((4, 7), dtype=np.float64)
            H[0, 0] = 1.0
            H[1, 1] = 1.0
            H[2, 2] = 1.0
            H[3, 3] = 1.0
            z = np.array([det.position_3d[0], det.position_3d[1], det.velocity_3d[0], det.velocity_3d[1]])
            R = np.diag([0.2**2, 0.2**2, 0.3**2, 0.3**2]).astype(np.float64)
        else:
            # Camera only: measure [x, y]
            H = np.zeros((2, 7), dtype=np.float64)
            H[0, 0] = 1.0
            H[1, 1] = 1.0
            z = np.array([det.position_3d[0], det.position_3d[1]])
            R = np.diag([0.4**2, 0.4**2]).astype(np.float64)

        y = z - H @ self.state
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.state = self.state + K @ y
        I = np.eye(7)
        self.P = (I - K @ H) @ self.P


class MultiObjectTracker:
    """
    Surround 360-degree Multi-Object Tracker with association gating.
    """

    def __init__(self, max_misses: int = 5, distance_threshold_m: float = 4.0):
        self.tracks: Dict[int, ObjectTrack] = {}
        self.next_track_id = 1
        self.max_misses = max_misses
        self.dist_threshold = distance_threshold_m

    def step(self, detections: List[Detection3D], timestamp: float, dt: float) -> List[ObjectTrack]:
        # 1. Predict all existing tracks
        for track in self.tracks.values():
            track.predict(dt)

        # 2. Compute cost / distance matrix
        active_ids = list(self.tracks.keys())
        unmatched_detections = set(range(len(detections)))
        unmatched_tracks = set(active_ids)

        if active_ids and detections:
            # Greedy nearest neighbor association with gating
            cost_matrix = np.zeros((len(active_ids), len(detections)))
            for i, tid in enumerate(active_ids):
                t_pos = self.tracks[tid].position[:2]
                for j, det in enumerate(detections):
                    d_pos = det.position_3d[:2]
                    dist = float(np.linalg.norm(t_pos - d_pos))
                    cost_matrix[i, j] = dist

            # Match greedily below distance threshold
            while True:
                min_val = np.min(cost_matrix)
                if min_val > self.dist_threshold or np.isinf(min_val):
                    break
                i, j = np.unravel_index(np.argmin(cost_matrix), cost_matrix.shape)
                tid = active_ids[i]

                # Update matched track
                self.tracks[tid].update(detections[j], timestamp)
                unmatched_tracks.discard(tid)
                unmatched_detections.discard(j)

                # Set row and column to infinity
                cost_matrix[i, :] = np.inf
                cost_matrix[:, j] = np.inf

        # 3. Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            new_track = ObjectTrack(self.next_track_id, detections[det_idx], timestamp)
            self.tracks[self.next_track_id] = new_track
            self.next_track_id += 1

        # 4. Prune dead tracks
        dead_ids = [tid for tid, trk in self.tracks.items() if trk.misses > self.max_misses]
        for did in dead_ids:
            del self.tracks[did]

        # Return confirmed active tracks (and stable tentative ones)
        return [t for t in self.tracks.values() if t.status == TrackStatus.CONFIRMED or t.hits >= 2]
