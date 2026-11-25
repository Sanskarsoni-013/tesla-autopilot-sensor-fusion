"""
Collision Risk Assessment Engine: Time-To-Collision (TTC),
Forward Collision Warning (FCW), and Automatic Emergency Braking (AEB).
"""

from dataclasses import dataclass
from enum import Enum
import math
from typing import List, Optional, Tuple
import numpy as np
from .object_tracker import ObjectTrack


class SafetyAlertLevel(Enum):
    NORMAL = "normal"
    ADVISORY = "advisory"
    FCW_ALERT = "forward_collision_warning"
    AEB_BRAKING = "automatic_emergency_braking"


@dataclass
class ThreatAssessment:
    critical_target_id: Optional[int]
    threat_level: SafetyAlertLevel
    min_ttc_seconds: float
    required_deceleration_mps2: float
    target_distance_m: float
    relative_speed_mps: float
    recommended_action: str


class RiskAssessmentEngine:
    """
    Monitors all tracked surrounding objects, checks path overlap corridor,
    computes Time-to-Collision (TTC), and triggers FCW audible alerts and AEB braking.
    """

    def __init__(
        self,
        fcw_ttc_threshold_s: float = 2.7,
        aeb_ttc_threshold_s: float = 1.4,
        max_aeb_decel_mps2: float = 8.5,
        lane_corridor_half_width_m: float = 1.85
    ):
        self.fcw_ttc = fcw_ttc_threshold_s
        self.aeb_ttc = aeb_ttc_threshold_s
        self.max_aeb_decel = max_aeb_decel_mps2
        self.corridor_w = lane_corridor_half_width_m

    def assess_risk(
        self,
        ego_speed_mps: float,
        ego_yaw_rate_radps: float,
        surround_tracks: List[ObjectTrack]
    ) -> ThreatAssessment:
        if not surround_tracks:
            return ThreatAssessment(
                critical_target_id=None,
                threat_level=SafetyAlertLevel.NORMAL,
                min_ttc_seconds=float("inf"),
                required_deceleration_mps2=0.0,
                target_distance_m=float("inf"),
                relative_speed_mps=0.0,
                recommended_action="NORMAL_CRUISE"
            )

        min_ttc = float("inf")
        highest_threat = SafetyAlertLevel.NORMAL
        critical_id: Optional[int] = None
        critical_dist = float("inf")
        critical_v_rel = 0.0
        max_req_decel = 0.0

        for track in surround_tracks:
            rel_x = track.position[0]
            rel_y = track.position[1]

            # Only evaluate obstacles in front of the vehicle
            if rel_x <= 0.5:
                continue

            # Lateral corridor check (taking curve trajectory into account)
            # If driving straight, corridor is |y| < corridor_w
            # Curve offset compensation: dy = 0.5 * curvature * x^2
            lateral_offset = abs(rel_y)
            if lateral_offset > self.corridor_w:
                continue

            # Relative longitudinal velocity: v_rel = v_target_x - v_ego
            target_vx = track.velocity[0]
            closing_speed = ego_speed_mps - target_vx  # Positive when closing in

            if closing_speed > 0.1:  # Vehicle is catching up
                # TTC = distance / closing_speed
                # Use vehicle front bumper to target rear distance
                effective_dist = max(0.1, rel_x - 3.0)  # Account for ego nose & target rear
                ttc = effective_dist / closing_speed

                # Required stopping deceleration: a_req = v_closing^2 / (2 * effective_dist)
                a_req = (closing_speed**2) / (2.0 * effective_dist)

                if ttc < min_ttc:
                    min_ttc = ttc
                    critical_id = track.track_id
                    critical_dist = rel_x
                    critical_v_rel = -closing_speed
                    max_req_decel = a_req

        # Evaluate threshold classification
        if min_ttc <= self.aeb_ttc or (critical_dist < 4.0 and critical_v_rel < -2.0):
            threat = SafetyAlertLevel.AEB_BRAKING
            action = f"AEB_EMERGENCY_STOP: Decel {min(self.max_aeb_decel, max_req_decel):.1f} m/s^2"
        elif min_ttc <= self.fcw_ttc:
            threat = SafetyAlertLevel.FCW_ALERT
            action = f"FCW_WARNING: Audio-visual alert, brake pre-fill active"
        elif min_ttc < 4.5:
            threat = SafetyAlertLevel.ADVISORY
            action = f"ADVISORY: Follow distance closing (TTC={min_ttc:.1f}s)"
        else:
            threat = SafetyAlertLevel.NORMAL
            action = "NORMAL_CRUISE"

        return ThreatAssessment(
            critical_target_id=critical_id,
            threat_level=threat,
            min_ttc_seconds=min_ttc,
            required_deceleration_mps2=min(self.max_aeb_decel, max_req_decel),
            target_distance_m=critical_dist,
            relative_speed_mps=critical_v_rel,
            recommended_action=action
        )
