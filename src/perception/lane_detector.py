"""
Tesla Vision Lane Detection & Boundary Perception System.
Extracts 3rd-order polynomial road boundaries, calculates lane curvature,
ego lateral lane offset, and heading angle error for Autopilot steering.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Dict, List, Optional, Tuple
import numpy as np


class LaneMarkingType(Enum):
    SOLID_WHITE = "solid_white"
    DASHED_WHITE = "dashed_white"
    SOLID_YELLOW = "solid_yellow"
    DOUBLE_SOLID_YELLOW = "double_solid_yellow"
    ROAD_EDGE_CURB = "road_edge_curb"


@dataclass
class LaneBoundary:
    marking_type: LaneMarkingType
    # Polynomial coefficients: y(x) = c0 + c1*x + c2*x^2 + c3*x^3
    # where x is longitudinal forward distance, y is lateral position (+ = left, - = right)
    poly_coeffs: np.ndarray  # [c0, c1, c2, c3]
    valid_range_m: Tuple[float, float] = (0.0, 80.0)
    confidence: float = 0.95

    def eval_y(self, x: float) -> float:
        c0, c1, c2, c3 = self.poly_coeffs
        return c0 + c1 * x + c2 * (x**2) + c3 * (x**3)

    def eval_heading(self, x: float) -> float:
        # dy/dx = c1 + 2*c2*x + 3*c3*x^2
        c0, c1, c2, c3 = self.poly_coeffs
        dydx = c1 + 2.0 * c2 * x + 3.0 * c3 * (x**2)
        return math.atan(dydx)

    def eval_curvature(self, x: float) -> float:
        # kappa = |y''| / (1 + (y')^2)^(3/2)
        c0, c1, c2, c3 = self.poly_coeffs
        dydx = c1 + 2.0 * c2 * x + 3.0 * c3 * (x**2)
        d2ydx2 = 2.0 * c2 + 6.0 * c3 * x
        return abs(d2ydx2) / ((1.0 + dydx**2)**1.5)


@dataclass
class EgoLaneState:
    left_boundary: LaneBoundary
    right_boundary: LaneBoundary
    center_offset_m: float       # Ego lateral displacement from lane center (+ left, - right)
    heading_error_rad: float     # Ego yaw angle relative to road tangent (+ left)
    lane_width_m: float          # Width between left and right boundaries at x=0
    curvature_radpm: float       # Road curvature kappa at ego vehicle
    radius_of_curvature_m: float # R = 1 / kappa
    is_lane_departure: bool      # True if ego crosses safe boundary threshold


class LaneDetector:
    """
    Simulates Tesla Vision neural network lane extraction and ego tracking.
    """

    def __init__(self, standard_lane_width: float = 3.7):
        self.standard_lane_width = standard_lane_width
        self.lane_departure_threshold_m = 0.65  # Alert if > 65cm from lane center

    def estimate_ego_lane(
        self,
        left_poly: np.ndarray,
        right_poly: np.ndarray,
        left_type: LaneMarkingType = LaneMarkingType.DASHED_WHITE,
        right_type: LaneMarkingType = LaneMarkingType.SOLID_WHITE
    ) -> EgoLaneState:
        left_bound = LaneBoundary(marking_type=left_type, poly_coeffs=left_poly)
        right_bound = LaneBoundary(marking_type=right_type, poly_coeffs=right_poly)

        # Evaluate at vehicle front axle (x = 0)
        y_left_0 = left_bound.eval_y(0.0)
        y_right_0 = right_bound.eval_y(0.0)
        lane_width = y_left_0 - y_right_0

        # Ego is at (0, 0). Center of lane is (y_left + y_right)/2
        lane_center_0 = (y_left_0 + y_right_0) / 2.0
        # Lateral offset: ego y (0) - lane_center_0 = -lane_center_0
        offset_m = -lane_center_0

        # Road heading angle tangent at x=0
        heading_left = left_bound.eval_heading(0.0)
        heading_right = right_bound.eval_heading(0.0)
        road_heading = (heading_left + heading_right) / 2.0
        heading_error = -road_heading  # Ego heading is 0, so error is 0 - road_heading

        # Curvature
        curv_left = left_bound.eval_curvature(0.0)
        curv_right = right_bound.eval_curvature(0.0)
        avg_curv = max(1e-6, (curv_left + curv_right) / 2.0)
        radius = 1.0 / avg_curv

        is_departure = bool(abs(offset_m) > self.lane_departure_threshold_m)

        return EgoLaneState(
            left_boundary=left_bound,
            right_boundary=right_bound,
            center_offset_m=offset_m,
            heading_error_rad=heading_error,
            lane_width_m=lane_width,
            curvature_radpm=avg_curv,
            radius_of_curvature_m=radius,
            is_lane_departure=is_departure
        )

    def sample_lane_center_path(
        self,
        left_poly: np.ndarray,
        right_poly: np.ndarray,
        ds_m: float = 2.0,
        max_dist_m: float = 60.0
    ) -> List[Tuple[float, float, float]]:
        """
        Samples reference center line waypoints [x, y, heading] for path following.
        """
        waypoints = []
        c_left = LaneBoundary(LaneMarkingType.SOLID_WHITE, left_poly)
        c_right = LaneBoundary(LaneMarkingType.SOLID_WHITE, right_poly)

        x_coords = np.arange(0.0, max_dist_m, ds_m)
        for x in x_coords:
            y_l = c_left.eval_y(x)
            y_r = c_right.eval_y(x)
            y_center = (y_l + y_r) / 2.0
            heading = (c_left.eval_heading(x) + c_right.eval_heading(x)) / 2.0
            waypoints.append((float(x), float(y_center), float(heading)))

        return waypoints
