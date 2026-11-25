"""
Traffic Semantic Intelligence: Traffic Lights, Signs, and Right-of-Way.
Extracts environmental regulation semantics and calculates stopping
feasibility and intersection safety clearance.
"""

from dataclasses import dataclass
from enum import Enum
import math
from typing import List, Optional, Tuple
import numpy as np


class TrafficLightColor(Enum):
    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"
    OFF = "off"
    UNKNOWN = "unknown"


class SignType(Enum):
    SPEED_LIMIT = "speed_limit"
    STOP_SIGN = "stop_sign"
    YIELD = "yield"
    PEDESTRIAN_CROSSING = "crosswalk"
    CONSTRUCTION = "construction"


@dataclass
class TrafficLight:
    light_id: str
    color: TrafficLightColor
    position_3d: np.ndarray  # [X, Y, Z] in vehicle frame
    time_in_state_s: float
    confidence: float = 0.98

    @property
    def distance_m(self) -> float:
        return float(np.linalg.norm(self.position_3d[:2]))


@dataclass
class TrafficSign:
    sign_id: str
    sign_type: SignType
    value: Optional[float]   # e.g., speed limit 65.0 mph or 100.0 km/h
    position_3d: np.ndarray
    confidence: float = 0.95

    @property
    def distance_m(self) -> float:
        return float(np.linalg.norm(self.position_3d[:2]))


@dataclass
class TrafficDecision:
    action: str              # "GO", "DECELERATE_TO_STOP", "STOP_IMMEDIATE", "CAUTION"
    target_stop_dist_m: Optional[float]
    speed_limit_mps: float
    reason: str


class TrafficSemanticEngine:
    """
    Processes visual traffic lights and road signs to synthesize driving rules.
    """

    def __init__(self, default_speed_limit_mps: float = 29.06): # 65 mph ~= 29.06 m/s
        self.current_speed_limit_mps = default_speed_limit_mps
        self.comfortable_decel_mps2 = 2.5   # Comfortable passenger braking
        self.max_service_decel_mps2 = 4.5   # Standard firm braking

    def evaluate_intersection(
        self,
        ego_speed_mps: float,
        traffic_lights: List[TrafficLight],
        traffic_signs: List[TrafficSign],
        distance_to_stop_line_m: float
    ) -> TrafficDecision:
        # Check signs first (Stop sign or Speed limit changes)
        for sign in traffic_signs:
            if sign.sign_type == SignType.SPEED_LIMIT and sign.value is not None:
                # Assuming input value in mph -> convert to m/s
                self.current_speed_limit_mps = sign.value * 0.44704

            if sign.sign_type == SignType.STOP_SIGN and sign.distance_m < 50.0:
                return TrafficDecision(
                    action="DECELERATE_TO_STOP",
                    target_stop_dist_m=max(1.0, sign.distance_m - 2.0),
                    speed_limit_mps=0.0,
                    reason=f"Stop sign detected {sign.distance_m:.1f}m ahead"
                )

        # Check traffic light states
        if not traffic_lights:
            return TrafficDecision(
                action="GO",
                target_stop_dist_m=None,
                speed_limit_mps=self.current_speed_limit_mps,
                reason="No active traffic constraints"
            )

        # Select closest relevant light in forward path
        forward_lights = [l for l in traffic_lights if l.position_3d[0] > 0]
        if not forward_lights:
            return TrafficDecision("GO", None, self.current_speed_limit_mps, "Path clear")

        primary_light = min(forward_lights, key=lambda l: l.position_3d[0])
        dist = primary_light.position_3d[0]

        if primary_light.color == TrafficLightColor.GREEN:
            return TrafficDecision(
                action="GO",
                target_stop_dist_m=None,
                speed_limit_mps=self.current_speed_limit_mps,
                reason="Green light signal confirmed"
            )

        elif primary_light.color == TrafficLightColor.RED:
            return TrafficDecision(
                action="DECELERATE_TO_STOP",
                target_stop_dist_m=max(0.5, dist - 3.0),
                speed_limit_mps=0.0,
                reason=f"Red light {dist:.1f}m ahead"
            )

        elif primary_light.color == TrafficLightColor.YELLOW:
            # Yellow light dilemma zone calculation
            # Required stopping distance: d_stop = v^2 / (2 * a)
            req_stop_dist = (ego_speed_mps**2) / (2.0 * self.comfortable_decel_mps2)
            if dist > req_stop_dist:
                return TrafficDecision(
                    action="DECELERATE_TO_STOP",
                    target_stop_dist_m=dist - 3.0,
                    speed_limit_mps=0.0,
                    reason=f"Yellow light - safe comfortable stopping distance {req_stop_dist:.1f}m < {dist:.1f}m"
                )
            else:
                # Too close to stop comfortably; continue through intersection
                return TrafficDecision(
                    action="GO",
                    target_stop_dist_m=None,
                    speed_limit_mps=self.current_speed_limit_mps,
                    reason=f"Yellow light dilemma zone - clearance preferred over harsh braking"
                )

        return TrafficDecision("CAUTION", None, self.current_speed_limit_mps, "Unknown light state")
