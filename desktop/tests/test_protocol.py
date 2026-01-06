"""Tests for watch protocol parsing."""

import struct

from wristcontrol.bluetooth.protocol import (
    GestureEvent,
    GestureType,
    SensorData,
    WatchProtocol,
)


def test_parse_sensor_data() -> None:
    """Test parsing sensor data from bytes."""
    # Create test data
    data = struct.pack(
        WatchProtocol.SENSOR_FORMAT,
        1000,  # timestamp_ms
        1.0, 2.0, 9.8,  # accel x, y, z
        0.1, 0.2, 0.3,  # gyro x, y, z
        1.0, 0.0, 0.0, 0.0,  # orientation quaternion
    )

    result = WatchProtocol.parse_sensor_data(data)

    assert result is not None
    assert result.timestamp_ms == 1000
    assert result.accel_x == 1.0
    assert result.accel_y == 2.0
    assert abs(result.accel_z - 9.8) < 0.01
    assert result.gyro_x == 0.1
    assert result.orientation_w == 1.0


def test_parse_sensor_data_invalid() -> None:
    """Test parsing invalid sensor data returns None."""
    result = WatchProtocol.parse_sensor_data(b"short")
    assert result is None


def test_parse_gesture_event() -> None:
    """Test parsing gesture event from bytes."""
    data = struct.pack(
        WatchProtocol.GESTURE_FORMAT,
        2000,  # timestamp_ms
        GestureType.TAP,  # gesture_type
        0.95,  # confidence
    )

    result = WatchProtocol.parse_gesture_event(data)

    assert result is not None
    assert result.timestamp_ms == 2000
    assert result.gesture_type == GestureType.TAP
    assert abs(result.confidence - 0.95) < 0.01


def test_parse_gesture_event_invalid() -> None:
    """Test parsing invalid gesture event returns None."""
    result = WatchProtocol.parse_gesture_event(b"x")
    assert result is None
