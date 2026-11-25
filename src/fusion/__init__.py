"""
Fusion package: Extended Kalman Filter multi-sensor state estimation.
"""

from .ekf_fusion import VehicleState, ExtendedKalmanFilter

__all__ = ["VehicleState", "ExtendedKalmanFilter"]
