"""
Terminal Telemetry HUD & ASCII Dashboard.
Renders real-time vehicle instrumentation, sensor readings, and threat matrix.
"""

from .simulator import TelemetryFrame, SafetyAlertLevel


class TerminalDashboard:
    """
    Renders high-refresh ASCII Autopilot cockpit telemetry.
    """

    @staticmethod
    def render_frame(frame: TelemetryFrame) -> str:
        speed_mph = frame.ego_state.speed_mps * 2.23694
        speed_kmh = frame.ego_state.speed_mps * 3.6

        # Threat alert banner
        if frame.threat.threat_level == SafetyAlertLevel.AEB_BRAKING:
            alert_bar = " [!!!] AUTOMATIC EMERGENCY BRAKING (AEB) ACTIVATED - FULL BRAKE [!!!] "
        elif frame.threat.threat_level == SafetyAlertLevel.FCW_ALERT:
            alert_bar = " [ ! ] FORWARD COLLISION WARNING (FCW) - BRAKE NOW [ ! ] "
        elif frame.threat.threat_level == SafetyAlertLevel.ADVISORY:
            alert_bar = " [ * ] ADVISORY: Approaching Forward Vehicle "
        else:
            alert_bar = " [ OK ] AUTOPILOT ACTIVE - PATH CLEAR "

        ttc_str = f"{frame.threat.min_ttc_seconds:.2f} s" if frame.threat.min_ttc_seconds < 99.0 else "INF"

        lines = [
            "=" * 78,
            f"  TESLA AUTOPILOT SENSOR INTELLIGENCE  |  Time: {frame.timestamp:6.2f}s  |  Step: {frame.step:4d}",
            "=" * 78,
            f"{alert_bar.center(78, '#')}",
            "-" * 78,
            f" SPEED: {speed_mph:5.1f} mph ({speed_kmh:5.1f} km/h)  |  ACCEL: {frame.control.target_accel_mps2:+5.2f} m/s²  |  TTC: {ttc_str:>8s}",
            f" STEER: {frame.control.steering_wheel_deg:+6.1f}° (Whl)  |  THROTTLE: {frame.control.throttle_pct:4.1f}%  |  BRAKE: {frame.control.brake_pressure_pct:4.1f}%",
            "-" * 78,
            f" [LANE SENSING] Offset: {frame.lane_state.center_offset_m:+5.2f} m | Curve Radius: {frame.lane_state.radius_of_curvature_m:6.0f} m | Departure: {frame.lane_state.is_lane_departure}",
            f" [RADAR/SONAR]  Radar Targets: {frame.radar_target_count:2d} | Closest Sonar: {frame.closest_ultrasonic_dist_m:4.2f} m ({frame.ultrasonic_alert.value})",
            f" [EKF FUSION]   Pos: ({frame.ego_state.x:6.1f}, {frame.ego_state.y:5.1f}) m | Yaw: {frame.ego_state.heading_deg:5.1f}° | Tracks: {len(frame.tracked_objects)}",
            "-" * 78,
            f" CAN BUS: DAS_steer 0x048 [{frame.can_frames[0].data.hex()}] | DAS_ctrl 0x116 [{frame.can_frames[1].data.hex()}]",
            "=" * 78
        ]

        return "\n".join(lines)
