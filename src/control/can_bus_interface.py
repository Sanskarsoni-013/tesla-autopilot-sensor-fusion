"""
Tesla CAN Bus Interface: Emulates Controller Area Network (CAN) Frames.
Encodes Drive Control (0x116), Steering Control (0x048), Speed Status (0x118),
and Autopilot System States (0x399) with CRC-8 and message sequence counters.
"""

from dataclasses import dataclass
import struct
from typing import Dict, Optional, Tuple
import numpy as np


def crc8_sae_j1850(data: bytes) -> int:
    """
    Standard SAE J1850 CRC-8 checksum calculation used on automotive CAN buses.
    Polynomial: x^8 + x^4 + x^3 + x^2 + 1 (0x1D), Initial: 0xFF, Final XOR: 0xFF.
    """
    crc = 0xFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ 0x1D) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc ^ 0xFF


@dataclass
class CANFrame:
    arbitration_id: int   # 11-bit standard CAN ID
    data: bytes           # 8-byte payload
    timestamp: float


class TeslaCANInterface:
    """
    Tesla Autopilot HW3/HW4 CAN Gateway Transceiver.
    """

    ID_STEERING_CONTROL = 0x048  # DAS_steeringControl
    ID_DRIVE_CONTROL = 0x116     # DAS_control (accel / brake)
    ID_SPEED_STATUS = 0x118      # ESP_wheelSpeeds
    ID_AUTOPILOT_STATUS = 0x399  # DAS_status

    def __init__(self):
        self.msg_counters: Dict[int, int] = {
            self.ID_STEERING_CONTROL: 0,
            self.ID_DRIVE_CONTROL: 0,
            self.ID_SPEED_STATUS: 0,
            self.ID_AUTOPILOT_STATUS: 0
        }

    def _next_counter(self, can_id: int) -> int:
        c = self.msg_counters.get(can_id, 0)
        self.msg_counters[can_id] = (c + 1) % 16
        return c

    def encode_steering_control(
        self,
        steering_angle_rad: float,
        steering_wheel_deg: float,
        timestamp: float
    ) -> CANFrame:
        """
        Encodes DAS_steeringControl (0x048):
        - Steering angle request in units of 0.1 deg
        - Max torque request
        - Message rolling counter (4-bit)
        - CRC-8 checksum
        """
        counter = self._next_counter(self.ID_STEERING_CONTROL)
        # Scaled integer representation: angle * 10
        raw_angle = int(np.clip(steering_wheel_deg * 10.0, -32768, 32767))
        torque_request = 100  # Default nominal assist torque (0.01 Nm)

        payload_without_crc = bytearray(7)
        # Bytes 0-1: Commanded angle (int16 little endian)
        struct.pack_into("<h", payload_without_crc, 0, raw_angle)
        # Byte 2: Torque request
        payload_without_crc[2] = torque_request & 0xFF
        # Byte 3: Enable flag and counter (bit 0-3 counter, bit 7 active)
        payload_without_crc[3] = (counter & 0x0F) | 0x80
        # Byte 4-6: Reserved / state flags
        payload_without_crc[4] = 0x01
        payload_without_crc[5] = 0x00
        payload_without_crc[6] = 0x00

        crc = crc8_sae_j1850(bytes(payload_without_crc))
        full_payload = bytes(payload_without_crc) + bytes([crc])

        return CANFrame(self.ID_STEERING_CONTROL, full_payload, timestamp)

    def encode_drive_control(
        self,
        accel_mps2: float,
        aeb_emergency: bool,
        timestamp: float
    ) -> CANFrame:
        """
        Encodes DAS_control (0x116):
        - Commanded acceleration in units of 0.05 m/s^2 (offset: +10.0 m/s^2)
        - AEB trigger bit
        - Rolling counter and CRC-8
        """
        counter = self._next_counter(self.ID_DRIVE_CONTROL)
        # Accel scaled: raw = (accel + 10.0) / 0.05
        raw_accel = int(np.clip((accel_mps2 + 10.0) / 0.05, 0, 400))

        payload_without_crc = bytearray(7)
        struct.pack_into("<H", payload_without_crc, 0, raw_accel)
        payload_without_crc[2] = 0x01 if aeb_emergency else 0x00
        payload_without_crc[3] = (counter & 0x0F) | 0x40
        payload_without_crc[4] = 0x00
        payload_without_crc[5] = 0x00
        payload_without_crc[6] = 0x00

        crc = crc8_sae_j1850(bytes(payload_without_crc))
        full_payload = bytes(payload_without_crc) + bytes([crc])

        return CANFrame(self.ID_DRIVE_CONTROL, full_payload, timestamp)

    def decode_speed_status(self, frame: CANFrame) -> float:
        """
        Decodes ESP_wheelSpeeds (0x118) into vehicle speed in m/s.
        """
        if len(frame.data) < 4:
            return 0.0
        # Average front left and front right wheel speed (scaled 0.05 km/h)
        raw_speed_fl = struct.unpack_from("<H", frame.data, 0)[0]
        speed_kmh = raw_speed_fl * 0.05
        return speed_kmh / 3.6
