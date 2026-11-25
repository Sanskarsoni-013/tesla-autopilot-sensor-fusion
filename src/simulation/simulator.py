"""
Closed-Loop Autonomous Driving Simulation Engine.
Integrates 8-Camera Surround Vision, IMU/GPS, Ultrasonic Radar, EKF Fusion,
3D MOT Tracking, Risk Assessment, Quintic Planning, and Vehicle Control.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Dict, List, Optional, Tuple
import numpy as np

from ..sensors.camera_rig import CameraRig, CameraType
from ..sensors.imu_gps_sensor import InertialNavSimulator, IMUReading, GPSReading, WheelOdometryReading
from ..sensors.ultrasonic_radar import UltrasonicArray, ForwardRadarModel, AlertZone
from ..perception.bev_transformer import BEVTransformer, BEVGridConfig
from ..perception.lane_detector import LaneDetector, LaneMarkingType, EgoLaneState
from ..perception.traffic_semantics import TrafficSemanticEngine, TrafficLight, TrafficLightColor
from ..fusion.ekf_fusion import ExtendedKalmanFilter, VehicleState
from ..tracking.object_tracker import MultiObjectTracker, Detection3D, ObjectClass, ObjectTrack
from ..tracking.risk_assessment import RiskAssessmentEngine, ThreatAssessment, SafetyAlertLevel
from ..planning.trajectory_planner import TrajectoryPlanner, TrajectoryCandidate
from ..control.mpc_controller import TeslaVehicleController, ControlCommand
from ..control.can_bus_interface import TeslaCANInterface, CANFrame


class SimulationScenario(Enum):
    HIGHWAY_CRUISE = "highway_cruise"
    OBSTACLE_AVOIDANCE = "obstacle_avoidance"
    EMERGENCY_AEB_STOP = "emergency_aeb_stop"


@dataclass
class DynamicTarget:
    target_id: int
    x: float
    y: float
    vx: float
    vy: float
    length: float = 4.6
    width: float = 1.9
    heading: float = 0.0
    obj_class: ObjectClass = ObjectClass.CAR


@dataclass
class TelemetryFrame:
    step: int
    timestamp: float
    scenario: SimulationScenario
    ego_state: VehicleState
    lane_state: EgoLaneState
    threat: ThreatAssessment
    control: ControlCommand
    tracked_objects: List[ObjectTrack]
    closest_ultrasonic_dist_m: float
    ultrasonic_alert: AlertZone
    radar_target_count: int
    can_frames: List[CANFrame]
    trajectory: TrajectoryCandidate


class AutopilotSimulator:
    """
    End-to-end Tesla Full Self-Driving (FSD) / Autopilot simulator.
    Executes all perception, fusion, tracking, planning, and control pipelines.
    """

    def __init__(
        self,
        scenario: SimulationScenario = SimulationScenario.HIGHWAY_CRUISE,
        dt: float = 0.05  # 20 Hz simulation rate
    ):
        self.scenario = scenario
        self.dt = dt
        self.step_idx = 0
        self.sim_time = 0.0

        # Ego vehicle physical ground truth
        self.ego_x = 0.0
        self.ego_y = 0.0
        self.ego_yaw = 0.0
        self.ego_v = 24.58  # ~55 mph (88 km/h)
        self.ego_a = 0.0

        # Subsystems
        self.camera_rig = CameraRig()
        self.sensor_sim = InertialNavSimulator(seed=123)
        self.ultrasonics = UltrasonicArray()
        self.radar = ForwardRadarModel()
        self.bev = BEVTransformer(BEVGridConfig())
        self.lane_detector = LaneDetector()
        self.traffic_engine = TrafficSemanticEngine()
        self.ekf = ExtendedKalmanFilter(
            initial_pos=np.array([0.0, 0.0, 0.0]),
            initial_vel=np.array([self.ego_v, 0.0, 0.0]),
            initial_yaw=0.0
        )
        self.tracker = MultiObjectTracker()
        self.risk_engine = RiskAssessmentEngine()
        self.planner = TrajectoryPlanner()
        self.controller = TeslaVehicleController()
        self.can_bus = TeslaCANInterface()

        # Target speed
        self.set_speed_mps = 29.06  # 65 mph (104 km/h)

        # Initialize surrounding dynamic targets based on scenario
        self.targets: List[DynamicTarget] = []
        self._init_scenario_targets()

    def _init_scenario_targets(self):
        if self.scenario == SimulationScenario.HIGHWAY_CRUISE:
            # Lead vehicle 45m ahead traveling at 26 m/s
            self.targets = [
                DynamicTarget(target_id=1, x=45.0, y=0.0, vx=26.0, vy=0.0),
                # Adjacent car in left lane
                DynamicTarget(target_id=2, x=20.0, y=3.7, vx=28.0, vy=0.0),
                # Slower truck in right lane
                DynamicTarget(target_id=3, x=35.0, y=-3.7, vx=22.0, vy=0.0)
            ]
        elif self.scenario == SimulationScenario.OBSTACLE_AVOIDANCE:
            # Slower moving obstacle cutting into lane
            self.targets = [
                DynamicTarget(target_id=1, x=38.0, y=2.5, vx=18.0, vy=-0.6), # Lane change into our lane
                DynamicTarget(target_id=2, x=15.0, y=-3.7, vx=24.0, vy=0.0)
            ]
        elif self.scenario == SimulationScenario.EMERGENCY_AEB_STOP:
            # Stopped vehicle directly in front at 35m
            self.targets = [
                DynamicTarget(target_id=1, x=35.0, y=0.0, vx=0.0, vy=0.0)
            ]

    def step(self) -> TelemetryFrame:
        self.step_idx += 1
        self.sim_time += self.dt

        # 1. Update dynamic surrounding obstacles
        target_positions = []
        target_velocities = []
        obstacle_points_for_ultrasonic = []
        planner_obstacles = []

        for tgt in self.targets:
            tgt.x += (tgt.vx - self.ego_v) * self.dt
            tgt.y += tgt.vy * self.dt

            pos_3d = np.array([tgt.x, tgt.y, 0.0])
            vel_3d = np.array([tgt.vx, tgt.vy, 0.0])
            target_positions.append(pos_3d)
            target_velocities.append(vel_3d)

            # Bounding box points for ultrasonics
            for dx in [-tgt.length/2, 0.0, tgt.length/2]:
                for dy in [-tgt.width/2, tgt.width/2]:
                    obstacle_points_for_ultrasonic.append(np.array([tgt.x + dx, tgt.y + dy, 0.5]))

            planner_obstacles.append((tgt.x, tgt.y, 1.8))

        # 2. Sensor Measurements
        # IMU, GPS, Wheel Odometry
        imu = self.sensor_sim.sample_imu(
            self.sim_time,
            true_accel_body=np.array([self.ego_a, 0.0, 0.0]),
            true_angular_velocity=np.array([0.0, 0.0, 0.0])
        )
        gps = self.sensor_sim.sample_gps(
            self.sim_time,
            true_enu=np.array([self.ego_x, self.ego_y, 0.0]),
            true_speed=self.ego_v,
            true_heading_rad=self.ego_yaw
        )
        wheel_odo = self.sensor_sim.sample_wheel_odometry(
            self.sim_time, self.ego_v, 0.0, self.dt
        )

        # Ultrasonics & Radar
        us_readings = self.ultrasonics.scan(obstacle_points_for_ultrasonic)
        closest_us_dist, us_alert = self.ultrasonics.get_closest_distance(us_readings)
        radar_targets = self.radar.measure(target_positions, target_velocities, self.ego_v)

        # 3. EKF Sensor Fusion Update
        self.ekf.predict(self.dt, imu.accel_mps2, imu.gyro_radps[2])
        if self.step_idx % 2 == 0:  # 10 Hz GPS update
            self.ekf.update_gps(gps.enu_position_m, gps.speed_mps, math.radians(gps.heading_deg))
        self.ekf.update_wheel_odometry(wheel_odo.speed_mps)
        ego_state = self.ekf.get_state(self.sim_time)

        # 4. Perception: Lane Detection & BEV Transformer
        # Nominal highway lane: left lane at +1.85m, right lane at -1.85m
        left_poly = np.array([1.85, 0.0, 0.0001, 0.0])
        right_poly = np.array([-1.85, 0.0, 0.0001, 0.0])
        lane_state = self.lane_detector.estimate_ego_lane(left_poly, right_poly)

        # BEV Occupancy Map update
        self.bev.reset_grid(decay_factor=0.90)
        for tgt in self.targets:
            self.bev.insert_obstacle_bbox(tgt.x, tgt.y, tgt.length, tgt.width, tgt.heading, tgt.vx, tgt.vy)

        # 5. Multi-Object Tracking & Fusion
        detections: List[Detection3D] = []
        for rt in radar_targets:
            detections.append(Detection3D(
                position_3d=rt.cartesian_xyz_m,
                velocity_3d=np.array([self.ego_v + rt.doppler_radial_mps, 0.0, 0.0]),
                sensor_source="radar",
                confidence=rt.confidence
            ))
        tracks = self.tracker.step(detections, self.sim_time, self.dt)

        # 6. Safety & Risk Assessment (TTC, FCW, AEB)
        threat = self.risk_engine.assess_risk(self.ego_v, ego_state.yaw_rate_radps, tracks)

        # 7. Trajectory Path Planning
        target_v = self.set_speed_mps
        if threat.threat_level == SafetyAlertLevel.AEB_BRAKING:
            target_v = 0.0
        elif threat.threat_level == SafetyAlertLevel.FCW_ALERT:
            target_v = max(0.0, self.ego_v - 8.0)
        elif tracks:
            # Adaptive Cruise Control follow logic
            lead_tracks = [t for t in tracks if t.position[0] > 0 and abs(t.position[1]) < 1.8]
            if lead_tracks:
                closest_lead = min(lead_tracks, key=lambda t: t.position[0])
                if closest_lead.position[0] < 40.0:
                    target_v = min(self.set_speed_mps, closest_lead.velocity[0])

        trajectory = self.planner.plan_trajectory(
            current_x=0.0,
            current_y=lane_state.center_offset_m,
            current_yaw=lane_state.heading_error_rad,
            current_v=self.ego_v,
            current_a=self.ego_a,
            target_v=target_v,
            target_lane_offset=0.0,
            obstacles_2d=planner_obstacles,
            dt=0.1
        )

        # 8. Actuation & Vehicle Control
        control = self.controller.step(
            timestamp=self.sim_time,
            current_x=0.0,
            current_y=0.0,
            current_yaw=self.ego_yaw,
            current_v=self.ego_v,
            trajectory=trajectory.points,
            dt=self.dt
        )

        # Override control with AEB maximum emergency deceleration if active
        if threat.threat_level == SafetyAlertLevel.AEB_BRAKING:
            control.target_accel_mps2 = -threat.required_deceleration_mps2
            control.brake_pressure_pct = 100.0
            control.throttle_pct = 0.0

        # Update physical ego vehicle
        self.ego_a = control.target_accel_mps2
        self.ego_v = max(0.0, self.ego_v + self.ego_a * self.dt)
        self.ego_x += self.ego_v * math.cos(self.ego_yaw) * self.dt
        self.ego_y += self.ego_v * math.sin(self.ego_yaw) * self.dt

        # 9. Tesla CAN Bus Encoding
        can_steer = self.can_bus.encode_steering_control(
            control.steering_angle_rad, control.steering_wheel_deg, self.sim_time
        )
        can_drive = self.can_bus.encode_drive_control(
            control.target_accel_mps2,
            threat.threat_level == SafetyAlertLevel.AEB_BRAKING,
            self.sim_time
        )

        return TelemetryFrame(
            step=self.step_idx,
            timestamp=self.sim_time,
            scenario=self.scenario,
            ego_state=ego_state,
            lane_state=lane_state,
            threat=threat,
            control=control,
            tracked_objects=tracks,
            closest_ultrasonic_dist_m=closest_us_dist,
            ultrasonic_alert=us_alert,
            radar_target_count=len(radar_targets),
            can_frames=[can_steer, can_drive],
            trajectory=trajectory
        )
