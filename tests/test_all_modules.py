"""
Comprehensive Unit & Integration Test Suite for Tesla Autopilot & Sensor Intelligence.
Tests Camera Rig, IMU/GPS, Radar, Ultrasonics, BEV Transformer, Lane Detector,
Traffic Semantics, EKF State Estimator, 3D MOT Tracker, Risk Engine,
Trajectory Planner, Stanley Controller, CAN Bus, and Closed-loop Simulation.
"""

import math
import numpy as np
import pytest

from src.sensors.camera_rig import CameraRig, CameraType
from src.sensors.imu_gps_sensor import (
    CoordinateTransformer,
    InertialNavSimulator,
    IMUReading,
    GPSReading,
)
from src.sensors.ultrasonic_radar import (
    UltrasonicArray,
    AlertZone,
    ForwardRadarModel,
)
from src.perception.bev_transformer import BEVTransformer, BEVGridConfig
from src.perception.lane_detector import LaneDetector, LaneMarkingType
from src.perception.traffic_semantics import (
    TrafficSemanticEngine,
    TrafficLight,
    TrafficLightColor,
    TrafficSign,
    SignType,
)
from src.fusion.ekf_fusion import ExtendedKalmanFilter, VehicleState
from src.tracking.object_tracker import (
    MultiObjectTracker,
    Detection3D,
    ObjectClass,
    TrackStatus,
)
from src.tracking.risk_assessment import (
    RiskAssessmentEngine,
    SafetyAlertLevel,
)
from src.planning.trajectory_planner import (
    QuinticPolynomial,
    TrajectoryPlanner,
)
from src.control.mpc_controller import (
    StanleyLateralController,
    LongitudinalPIDController,
    TeslaVehicleController,
)
from src.control.can_bus_interface import (
    TeslaCANInterface,
    crc8_sae_j1850,
)
from src.simulation.simulator import (
    AutopilotSimulator,
    SimulationScenario,
)


class TestCameraRig:
    def test_camera_initialization(self):
        rig = CameraRig()
        assert len(rig.cameras) == 8
        assert CameraType.FRONT_MAIN in rig.cameras
        assert CameraType.B_PILLAR_LEFT in rig.cameras
        assert CameraType.REAR_BACKUP in rig.cameras

    def test_front_projection(self):
        rig = CameraRig()
        # Point 20m in front of vehicle
        target = np.array([20.0, 0.0, 1.3])
        visible, coords, dist = rig.project_point_to_camera(target, CameraType.FRONT_MAIN)
        assert visible is True
        assert 15.0 < dist < 25.0
        assert 0 <= coords[0] <= 1280
        assert 0 <= coords[1] <= 960

    def test_camera_coverage_finding(self):
        rig = CameraRig()
        # Lead vehicle 15m ahead
        lead_pos = np.array([15.0, 0.0, 1.0])
        observing = rig.find_observing_cameras(lead_pos)
        cam_types = [c[0] for c in observing]
        assert CameraType.FRONT_WIDE in cam_types or CameraType.FRONT_MAIN in cam_types


class TestInertialGPS:
    def test_enu_coordinate_transformation(self):
        transformer = CoordinateTransformer(ref_lat_deg=37.3861, ref_lon_deg=-122.0839, ref_alt_m=30.0)
        # Reference point transformed should be at origin (0, 0, 0)
        enu = transformer.geodetic_to_enu(37.3861, -122.0839, 30.0)
        assert np.linalg.norm(enu) < 1e-3

    def test_sensor_simulation_samples(self):
        sim = InertialNavSimulator(seed=42)
        imu = sim.sample_imu(0.0, np.array([1.5, 0.0, 0.0]), np.array([0.0, 0.0, 0.05]))
        assert isinstance(imu, IMUReading)
        assert imu.accel_mps2[0] > 1.0
        assert imu.accel_mps2[2] > 9.0  # Includes gravity

        gps = sim.sample_gps(0.0, np.array([10.0, 5.0, 0.0]), 20.0, 0.5)
        assert isinstance(gps, GPSReading)
        assert abs(gps.speed_mps - 20.0) < 0.5


class TestUltrasonicRadar:
    def test_ultrasonic_alert_zones(self):
        assert UltrasonicArray.classify_distance(0.3) == AlertZone.CRITICAL_STOP
        assert UltrasonicArray.classify_distance(0.8) == AlertZone.WARNING
        assert UltrasonicArray.classify_distance(1.8) == AlertZone.CAUTION
        assert UltrasonicArray.classify_distance(5.0) == AlertZone.SAFE

    def test_ultrasonic_scanning(self):
        sonar = UltrasonicArray()
        # Close obstacle in front of bumper
        obstacles = [np.array([2.6, 0.20, 0.45])]
        readings = sonar.scan(obstacles)
        min_d, zone = sonar.get_closest_distance(readings)
        assert min_d < 1.0
        assert zone in (AlertZone.CRITICAL_STOP, AlertZone.WARNING)

    def test_forward_radar_measurement(self):
        radar = ForwardRadarModel()
        targets = radar.measure(
            target_positions=[np.array([40.0, 0.0, 0.5])],
            target_velocities=[np.array([25.0, 0.0, 0.0])],
            ego_velocity_mps=30.0
        )
        assert len(targets) == 1
        assert abs(targets[0].range_m - 37.6) < 2.0
        # Ego is closing in (30 m/s > 25 m/s), doppler velocity should be negative
        assert targets[0].doppler_radial_mps < 0.0


class TestPerception:
    def test_bev_occupancy_grid(self):
        cfg = BEVGridConfig(x_min=-10.0, x_max=50.0, y_min=-10.0, y_max=10.0, resolution_m=0.5)
        bev = BEVTransformer(cfg)
        assert bev.shape == (120, 40)

        bev.insert_obstacle_bbox(center_x=20.0, center_y=0.0, length_m=4.5, width_m=1.8, heading_rad=0.0)
        idx = bev.world_to_grid(20.0, 0.0)
        assert idx is not None
        assert bev.occupancy_grid[idx[0], idx[1]] > 0.8

        # Check corridor
        free_path = [(0.0, 5.0), (10.0, 5.0), (20.0, 5.0)]
        blocked_path = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)]
        assert bev.is_corridor_free(free_path) is True
        assert bev.is_corridor_free(blocked_path) is False

    def test_lane_detector(self):
        detector = LaneDetector()
        left_poly = np.array([1.85, 0.0, 0.0, 0.0])
        right_poly = np.array([-1.85, 0.0, 0.0, 0.0])
        lane = detector.estimate_ego_lane(left_poly, right_poly)
        assert abs(lane.lane_width_m - 3.7) < 1e-3
        assert abs(lane.center_offset_m) < 1e-3
        assert lane.is_lane_departure is False

    def test_traffic_semantics_red_light(self):
        engine = TrafficSemanticEngine()
        light = TrafficLight(
            light_id="TL1",
            color=TrafficLightColor.RED,
            position_3d=np.array([25.0, 1.0, 4.0]),
            time_in_state_s=5.0
        )
        decision = engine.evaluate_intersection(
            ego_speed_mps=15.0,
            traffic_lights=[light],
            traffic_signs=[],
            distance_to_stop_line_m=25.0
        )
        assert decision.action == "DECELERATE_TO_STOP"
        assert decision.speed_limit_mps == 0.0


class TestFusionAndTracking:
    def test_ekf_prediction_and_update(self):
        ekf = ExtendedKalmanFilter(initial_pos=np.array([0.0, 0.0, 0.0]), initial_yaw=0.0)
        # Predict forward motion at 2.0 m/s^2 accel
        ekf.predict(dt=0.1, imu_accel_body=np.array([2.0, 0.0, 0.0]), imu_gyro_z=0.0)
        state1 = ekf.get_state(0.1)
        assert state1.vx > 0.1

        # GPS update
        ekf.update_gps(np.array([0.2, 0.0, 0.0]), speed_mps=0.2, heading_rad=0.0)
        state2 = ekf.get_state(0.1)
        assert abs(state2.x - 0.2) < 0.1

    def test_multi_object_tracking_lifecycle(self):
        tracker = MultiObjectTracker()
        det = Detection3D(position_3d=np.array([30.0, 0.0, 0.0]), velocity_3d=np.array([20.0, 0.0, 0.0]))

        # Step 1: Tentative
        tracks = tracker.step([det], timestamp=0.1, dt=0.1)
        assert len(tracker.tracks) == 1
        track = list(tracker.tracks.values())[0]
        assert track.status == TrackStatus.TENTATIVE

        # Step 2 & 3: Confirmation
        det2 = Detection3D(position_3d=np.array([32.0, 0.0, 0.0]), velocity_3d=np.array([20.0, 0.0, 0.0]))
        tracker.step([det2], timestamp=0.2, dt=0.1)
        confirmed_tracks = tracker.step([det2], timestamp=0.3, dt=0.1)
        assert track.status == TrackStatus.CONFIRMED

    def test_risk_assessment_ttc_and_aeb(self):
        engine = RiskAssessmentEngine()
        tracker = MultiObjectTracker()
        # Fast closing target (closing at 20 m/s from 25m distance -> TTC = ~1.25s)
        det = Detection3D(position_3d=np.array([25.0, 0.0, 0.0]), velocity_3d=np.array([5.0, 0.0, 0.0]))
        tracker.step([det], 0.1, 0.1)
        tracker.step([det], 0.2, 0.1)
        tracks = tracker.step([det], 0.3, 0.1)

        threat = engine.assess_risk(ego_speed_mps=25.0, ego_yaw_rate_radps=0.0, surround_tracks=tracks)
        assert threat.threat_level == SafetyAlertLevel.AEB_BRAKING
        assert threat.required_deceleration_mps2 > 0.0


class TestPlanningAndControl:
    def test_quintic_polynomial_boundary_conditions(self):
        p = QuinticPolynomial(xs=0.0, vxs=10.0, axs=0.0, xe=50.0, vxe=15.0, axe=0.0, T=3.0)
        assert abs(p.calc_point(0.0) - 0.0) < 1e-4
        assert abs(p.calc_first_derivative(0.0) - 10.0) < 1e-4
        assert abs(p.calc_point(3.0) - 50.0) < 1e-4
        assert abs(p.calc_first_derivative(3.0) - 15.0) < 1e-4

    def test_trajectory_planner(self):
        planner = TrajectoryPlanner()
        cand = planner.plan_trajectory(
            current_x=0.0, current_y=0.0, current_yaw=0.0,
            current_v=20.0, current_a=0.0, target_v=25.0, target_lane_offset=0.0
        )
        assert cand.valid is True
        assert len(cand.points) > 10
        assert cand.points[-1].v > 18.0

    def test_stanley_controller(self):
        ctrl = StanleyLateralController()
        planner = TrajectoryPlanner()
        cand = planner.plan_trajectory(0.0, 0.0, 0.0, 20.0, 0.0, 20.0)
        # Ego displaced 0.5m to the right -> controller should steer left (positive delta)
        steer, err, _ = ctrl.calculate_steering(0.0, -0.5, 0.0, 20.0, cand.points)
        assert steer > 0.0

    def test_tesla_can_crc_and_encoding(self):
        data = bytes([0x01, 0x02, 0x03, 0x04])
        crc = crc8_sae_j1850(data)
        assert isinstance(crc, int)
        assert 0 <= crc <= 255

        can = TeslaCANInterface()
        steer_frame = can.encode_steering_control(0.05, 5.0, 0.1)
        assert steer_frame.arbitration_id == 0x048
        assert len(steer_frame.data) == 8

        drive_frame = can.encode_drive_control(1.5, False, 0.1)
        assert drive_frame.arbitration_id == 0x116
        assert len(drive_frame.data) == 8


class TestEndToEndSimulation:
    def test_all_scenarios_execute(self):
        for sc in SimulationScenario:
            sim = AutopilotSimulator(scenario=sc, dt=0.05)
            frame = sim.step()
            assert frame.step == 1
            assert frame.ego_state.speed_mps >= 0.0
            assert len(frame.can_frames) == 2
            assert frame.closest_ultrasonic_dist_m > 0.0
