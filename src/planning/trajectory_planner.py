"""
Quintic Polynomial Jerk-Minimizing Trajectory Planner.
Generates optimal smooth 5th-order polynomial paths in Frenet coordinates (s, d)
with cost optimization across collision safety, lateral acceleration, and jerk.
"""

from dataclasses import dataclass
import math
from typing import List, Optional, Tuple
import numpy as np


class QuinticPolynomial:
    """
    1D Quintic Polynomial: p(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
    Solves boundary value problem given start (x0, v0, a0) and end (x1, v1, a1) over duration T.
    """

    def __init__(self, xs: float, vxs: float, axs: float, xe: float, vxe: float, axe: float, T: float):
        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([
            [T**3, T**4, T**5],
            [3.0 * T**2, 4.0 * T**3, 5.0 * T**4],
            [6.0 * T, 12.0 * T**2, 20.0 * T**3]
        ])

        b = np.array([
            xe - self.a0 - self.a1 * T - self.a2 * (T**2),
            vxe - self.a1 - 2.0 * self.a2 * T,
            axe - 2.0 * self.a2
        ])

        try:
            x = np.linalg.solve(A, b)
            self.a3 = x[0]
            self.a4 = x[1]
            self.a5 = x[2]
        except np.linalg.LinAlgError:
            self.a3 = self.a4 = self.a5 = 0.0

    def calc_point(self, t: float) -> float:
        return self.a0 + self.a1 * t + self.a2 * (t**2) + self.a3 * (t**3) + self.a4 * (t**4) + self.a5 * (t**5)

    def calc_first_derivative(self, t: float) -> float:
        return self.a1 + 2.0 * self.a2 * t + 3.0 * self.a3 * (t**2) + 4.0 * self.a4 * (t**3) + 5.0 * self.a5 * (t**4)

    def calc_second_derivative(self, t: float) -> float:
        return 2.0 * self.a2 + 6.0 * self.a3 * t + 12.0 * self.a4 * (t**2) + 20.0 * self.a5 * (t**3)

    def calc_third_derivative(self, t: float) -> float:
        return 6.0 * self.a3 + 24.0 * self.a4 * t + 60.0 * self.a5 * (t**2)


@dataclass
class TrajectoryPoint:
    t: float
    x: float
    y: float
    yaw: float
    v: float
    a: float
    jerk: float


@dataclass
class TrajectoryCandidate:
    points: List[TrajectoryPoint]
    cost: float
    valid: bool


class TrajectoryPlanner:
    """
    Frenet Optimal Trajectory Generator.
    Evaluates candidate maneuvers: lane keep, lane change left, lane change right,
    and adaptive cruise braking.
    """

    def __init__(
        self,
        max_speed_mps: float = 33.3,   # ~120 km/h (75 mph)
        max_accel_mps2: float = 3.5,
        max_decel_mps2: float = 7.0,
        max_jerk_mps3: float = 4.0,
        lane_width_m: float = 3.7
    ):
        self.max_speed = max_speed_mps
        self.max_accel = max_accel_mps2
        self.max_decel = max_decel_mps2
        self.max_jerk = max_jerk_mps3
        self.lane_w = lane_width_m

        # Weights for cost function
        self.k_jerk = 0.1
        self.k_time = 0.2
        self.k_target_speed = 1.0
        self.k_lateral_offset = 1.5
        self.k_collision = 1000.0

    def plan_trajectory(
        self,
        current_x: float,
        current_y: float,
        current_yaw: float,
        current_v: float,
        current_a: float,
        target_v: float,
        target_lane_offset: float = 0.0,
        obstacles_2d: Optional[List[Tuple[float, float, float]]] = None, # (x, y, radius)
        dt: float = 0.1
    ) -> TrajectoryCandidate:
        if obstacles_2d is None:
            obstacles_2d = []

        best_trajectory: Optional[TrajectoryCandidate] = None
        min_cost = float("inf")

        # Sample planning horizons T from 2.0s to 4.5s
        horizons = [2.5, 3.5, 4.5]
        # Sample target speeds around target_v
        target_speeds = [target_v - 2.0, target_v, target_v + 1.0]

        for T in horizons:
            for v_end in target_speeds:
                v_end = max(0.0, min(self.max_speed, v_end))

                # Longitudinal quintic
                # x(0) = current_x, vx(0) = current_v * cos(yaw), ax(0) = current_a
                # Target: xe = current_x + ((current_v + v_end)/2) * T
                xe = current_x + 0.5 * (current_v + v_end) * T
                poly_x = QuinticPolynomial(
                    current_x, current_v * math.cos(current_yaw), current_a * math.cos(current_yaw),
                    xe, v_end * math.cos(current_yaw), 0.0, T
                )

                # Lateral quintic
                # y(0) = current_y, vy(0) = current_v * sin(yaw)
                poly_y = QuinticPolynomial(
                    current_y, current_v * math.sin(current_yaw), current_a * math.sin(current_yaw),
                    target_lane_offset, 0.0, 0.0, T
                )

                # Discretize trajectory
                time_steps = np.arange(0.0, T + dt, dt)
                points: List[TrajectoryPoint] = []
                cost = 0.0
                is_valid = True

                prev_pt = None
                for t in time_steps:
                    px = poly_x.calc_point(t)
                    py = poly_y.calc_point(t)
                    vx = poly_x.calc_first_derivative(t)
                    vy = poly_y.calc_first_derivative(t)
                    ax = poly_x.calc_second_derivative(t)
                    ay = poly_y.calc_second_derivative(t)
                    jx = poly_x.calc_third_derivative(t)
                    jy = poly_y.calc_third_derivative(t)

                    v_mag = math.sqrt(vx**2 + vy**2)
                    a_mag = math.sqrt(ax**2 + ay**2)
                    j_mag = math.sqrt(jx**2 + jy**2)
                    yaw = math.atan2(vy, max(0.1, vx))

                    # Feasibility checks
                    if a_mag > self.max_accel or j_mag > self.max_jerk:
                        is_valid = False

                    # Collision checking against surround obstacles
                    for ox, oy, orad in obstacles_2d:
                        dist_sq = (px - ox)**2 + (py - oy)**2
                        safety_dist = orad + 1.2
                        if dist_sq < safety_dist**2:
                            cost += self.k_collision
                            is_valid = False

                    pt = TrajectoryPoint(t=t, x=px, y=py, yaw=yaw, v=v_mag, a=a_mag, jerk=j_mag)
                    points.append(pt)

                    # Running cost
                    cost += self.k_jerk * (j_mag**2) * dt
                    cost += self.k_lateral_offset * ((py - target_lane_offset)**2) * dt

                # Terminal cost
                cost += self.k_time * T
                cost += self.k_target_speed * ((points[-1].v - target_v)**2)

                cand = TrajectoryCandidate(points=points, cost=cost, valid=is_valid)

                if is_valid and cost < min_cost:
                    min_cost = cost
                    best_trajectory = cand

        # Fallback if all failed: generate conservative braking trajectory
        if best_trajectory is None:
            points = []
            T_fallback = 3.0
            t_steps = np.arange(0.0, T_fallback + dt, dt)
            decel = min(3.0, current_v / T_fallback)
            for t in t_steps:
                v = max(0.0, current_v - decel * t)
                dist = current_v * t - 0.5 * decel * (t**2)
                points.append(TrajectoryPoint(
                    t=t,
                    x=current_x + dist * math.cos(current_yaw),
                    y=current_y + dist * math.sin(current_yaw),
                    yaw=current_yaw,
                    v=v,
                    a=-decel,
                    jerk=0.0
                ))
            best_trajectory = TrajectoryCandidate(points=points, cost=999.0, valid=True)

        return best_trajectory
