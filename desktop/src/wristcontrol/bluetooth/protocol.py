"""Protocol definitions for watch communication."""

import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional


class GestureType(IntEnum):
    """Types of gestures recognized by the watch."""

    TAP = 0x01
    DOUBLE_TAP = 0x02
    HOLD = 0x03
    HOLD_RELEASE = 0x04
    PALM_UP = 0x05
    PALM_DOWN = 0x06
    WRIST_FLICK_LEFT = 0x07
    WRIST_FLICK_RIGHT = 0x08
    WRIST_FLICK_UP = 0x09
    WRIST_FLICK_DOWN = 0x0A


@dataclass
class SensorData:
    """Parsed sensor data from the watch.

    All values are in SI units:
    - Acceleration in m/s^2
    - Angular velocity in rad/s
    - Orientation as quaternion (w, x, y, z)
    """

    timestamp_ms: int
    accel_x: float
    accel_y: float
    accel_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float
    orientation_w: float
    orientation_x: float
    orientation_y: float
    orientation_z: float


@dataclass
class GestureEvent:
    """Parsed gesture event from the watch."""

    timestamp_ms: int
    gesture_type: GestureType
    confidence: float  # 0.0 to 1.0


class WatchProtocol:
    """Protocol for parsing data from the watch."""

    # Sensor data packet format:
    # - timestamp (4 bytes, uint32, ms)
    # - accel_x, accel_y, accel_z (3 * 4 bytes, float)
    # - gyro_x, gyro_y, gyro_z (3 * 4 bytes, float)
    # - orientation_w, orientation_x, orientation_y, orientation_z (4 * 4 bytes, float)
    # Total: 44 bytes
    SENSOR_FORMAT = "<I10f"
    SENSOR_SIZE = struct.calcsize(SENSOR_FORMAT)

    # Gesture event packet format:
    # - timestamp (4 bytes, uint32, ms)
    # - gesture_type (1 byte, uint8)
    # - confidence (4 bytes, float)
    # Total: 9 bytes
    GESTURE_FORMAT = "<IBf"
    GESTURE_SIZE = struct.calcsize(GESTURE_FORMAT)

    @classmethod
    def parse_sensor_data(cls, data: bytes) -> Optional[SensorData]:
        """Parse raw sensor data bytes into SensorData.

        Args:
            data: Raw bytes from BLE notification.

        Returns:
            Parsed SensorData, or None if parsing failed.
        """
        if len(data) < cls.SENSOR_SIZE:
            return None

        try:
            unpacked = struct.unpack(cls.SENSOR_FORMAT, data[: cls.SENSOR_SIZE])
            return SensorData(
                timestamp_ms=unpacked[0],
                accel_x=unpacked[1],
                accel_y=unpacked[2],
                accel_z=unpacked[3],
                gyro_x=unpacked[4],
                gyro_y=unpacked[5],
                gyro_z=unpacked[6],
                orientation_w=unpacked[7],
                orientation_x=unpacked[8],
                orientation_y=unpacked[9],
                orientation_z=unpacked[10],
            )
        except struct.error:
            return None

    @classmethod
    def parse_gesture_event(cls, data: bytes) -> Optional[GestureEvent]:
        """Parse raw gesture event bytes into GestureEvent.

        Args:
            data: Raw bytes from BLE notification.

        Returns:
            Parsed GestureEvent, or None if parsing failed.
        """
        if len(data) < cls.GESTURE_SIZE:
            return None

        try:
            unpacked = struct.unpack(cls.GESTURE_FORMAT, data[: cls.GESTURE_SIZE])
            return GestureEvent(
                timestamp_ms=unpacked[0],
                gesture_type=GestureType(unpacked[1]),
                confidence=unpacked[2],
            )
        except (struct.error, ValueError):
            return None
