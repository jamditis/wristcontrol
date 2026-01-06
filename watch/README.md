# WristControl Watch Application

Samsung Galaxy Watch application for the WristControl system. Captures sensor data and detects gestures, streaming them to the desktop companion app via Bluetooth Low Energy.

## Platform

This app targets **Wear OS** (newer Samsung Galaxy Watch models). Development requires:

- Android Studio
- Wear OS SDK
- Kotlin/Java

## Planned Features

- **Sensor Streaming**: Accelerometer and gyroscope data at 50-100Hz
- **On-device Gesture Detection**: Tap, double-tap, hold, palm orientation
- **Audio Streaming**: Microphone capture for voice commands (Phase 4)
- **BLE Services**: Custom GATT services for data streaming
- **Visual Feedback**: Connection status and activation state indicators

## BLE Service UUIDs

| Service | UUID | Description |
|---------|------|-------------|
| Main Service | `0000fff0-0000-1000-8000-00805f9b34fb` | Main WristControl service |
| Sensor Characteristic | `0000fff1-0000-1000-8000-00805f9b34fb` | IMU sensor data stream |
| Gesture Characteristic | `0000fff2-0000-1000-8000-00805f9b34fb` | Gesture event notifications |
| Audio Characteristic | `0000fff3-0000-1000-8000-00805f9b34fb` | Audio data stream |

## Development Phases

1. **Phase 3**: Basic watch app with gesture detection and sensor streaming
2. **Phase 4**: Audio streaming for voice commands
3. **Phase 5**: Refined motion tracking for cursor control

## Reference

- [DoublePoint TouchSDK](https://github.com/doublepointlab/touch-sdk-py) - Reference implementation for gesture detection
- [Wear OS Developer Guide](https://developer.android.com/training/wearables)

## Status

**Not yet implemented** - This is a placeholder for future development.
