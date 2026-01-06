# WristControl Implementation Plan
## Voice and Gesture Computer Control System

**Version:** 1.0
**Date:** January 2026
**Project:** WristControl - Samsung Galaxy Watch Companion System

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Phase 1: Foundation](#3-phase-1-foundation)
4. [Phase 2: Watch Application](#4-phase-2-watch-application)
5. [Phase 3: Desktop Application](#5-phase-3-desktop-application)
6. [Phase 4: Voice Integration](#6-phase-4-voice-integration)
7. [Phase 5: Cursor Control](#7-phase-5-cursor-control)
8. [Phase 6: Integration & Polish](#8-phase-6-integration--polish)
9. [Technical Specifications](#9-technical-specifications)
10. [Risk Mitigation](#10-risk-mitigation)

---

## 1. Executive Summary

### Project Goals
Build a system that enables hands-free computer control using a Samsung Galaxy Watch, translating wrist movements into cursor control and voice commands into system actions.

### Key Performance Targets
| Metric | Target | Critical Path |
|--------|--------|---------------|
| Cursor Latency | <50ms | BLE polling + Madgwick filter |
| Voice Command Latency | <500ms | faster-whisper tiny model |
| Gesture Accuracy | >95% | Algorithm + ML hybrid |
| Battery Life | 4-6 hours active | Adaptive sampling + batching |
| Cross-Platform | Windows, macOS, Linux | Python + pynput |

### Technology Stack Summary

**Watch App (WearOS/Kotlin):**
- SensorManager API (50-100Hz IMU)
- AudioRecord for microphone
- Nordic BLE Library for GATT server
- Algorithm-based gesture detection

**Desktop App (Python):**
- bleak for BLE connectivity
- pynput for input injection
- faster-whisper for local STT
- PyQt6 for settings UI
- pystray for system tray

---

## 2. Architecture Overview

### System Diagram
```
┌─────────────────────────────────────────────────────────────────────┐
│                    SAMSUNG GALAXY WATCH                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │
│  │ Accelerometer│  │  Gyroscope  │  │ Microphone  │                 │
│  │   100Hz     │  │   100Hz     │  │   16kHz     │                 │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                 │
│         │                │                │                         │
│         └────────────────┼────────────────┘                         │
│                          ▼                                          │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │                  Gesture Detection Engine                      │ │
│  │  • Finger pinch (accelerometer spike)                         │ │
│  │  • Wrist flick (gyroscope rotation)                           │ │
│  │  • Palm orientation (accelerometer gravity)                   │ │
│  └────────────────────────┬──────────────────────────────────────┘ │
│                           ▼                                         │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │              BLE GATT Server (Nordic Library)                  │ │
│  │  • Sensor Service (batched IMU data)                          │ │
│  │  • Gesture Service (detected events)                          │ │
│  │  • Audio Service (Opus compressed)                            │ │
│  └───────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                               │
                          BLE 5.0
                     7.5ms connection interval
                        247+ MTU
                               │
┌─────────────────────────────────────────────────────────────────────┐
│                      DESKTOP COMPANION APP                           │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │                 BLE Manager (bleak)                            │ │
│  │  • Auto-reconnect with exponential backoff                    │ │
│  │  • Polling for low-latency (<25ms)                            │ │
│  │  • MTU negotiation                                            │ │
│  └────────────────────────┬──────────────────────────────────────┘ │
│                           │                                         │
│           ┌───────────────┼───────────────┐                        │
│           ▼               ▼               ▼                        │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐               │
│  │ Motion       │ │ Gesture      │ │ Voice        │               │
│  │ Processor    │ │ Handler      │ │ Processor    │               │
│  ├──────────────┤ ├──────────────┤ ├──────────────┤               │
│  │ Madgwick     │ │ Tap → Click  │ │ Opus decode  │               │
│  │ Filter       │ │ Double → Dbl │ │ VAD (Silero) │               │
│  │ One Euro     │ │ Hold → Drag  │ │ STT (Whisper)│               │
│  │ Filter       │ │ Palm → Mode  │ │ Intent Parse │               │
│  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘               │
│         │                │                │                        │
│         └────────────────┼────────────────┘                        │
│                          ▼                                         │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │               Input Controller (pynput)                        │ │
│  │  • Cursor movement (3-7ms latency)                            │ │
│  │  • Mouse clicks and scroll                                    │ │
│  │  • Keyboard input                                             │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │                      UI Layer                                  │ │
│  │  • System tray (pystray)                                      │ │
│  │  • Settings window (PyQt6)                                    │ │
│  │  • Connection status                                          │ │
│  └───────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Flow Summary
1. **Watch**: Sensors → Batch (10 samples) → BLE notify
2. **Desktop**: BLE poll → Unpack → Filter → Cursor move
3. **Voice**: Microphone → Opus → BLE → Decode → VAD → STT → Execute

---

## 3. Phase 1: Foundation

### 3.1 Project Setup

**Duration:** 3-5 days

#### Watch Project Setup
```
wristcontrol/
├── watch-app/
│   ├── app/
│   │   ├── src/main/
│   │   │   ├── kotlin/com/wristcontrol/
│   │   │   │   ├── MainActivity.kt
│   │   │   │   ├── sensors/
│   │   │   │   │   ├── SensorManager.kt
│   │   │   │   │   └── GestureDetector.kt
│   │   │   │   ├── ble/
│   │   │   │   │   ├── GattServer.kt
│   │   │   │   │   └── BleService.kt
│   │   │   │   ├── audio/
│   │   │   │   │   └── AudioCapture.kt
│   │   │   │   └── service/
│   │   │   │       └── StreamingService.kt
│   │   │   ├── res/
│   │   │   └── AndroidManifest.xml
│   │   └── build.gradle
│   └── settings.gradle
```

#### Desktop Project Setup
```
wristcontrol/
├── desktop-app/
│   ├── src/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── config/
│   │   │   ├── __init__.py
│   │   │   └── settings.py
│   │   ├── ble/
│   │   │   ├── __init__.py
│   │   │   ├── connection.py
│   │   │   └── gatt_client.py
│   │   ├── motion/
│   │   │   ├── __init__.py
│   │   │   ├── sensor_fusion.py
│   │   │   ├── cursor_control.py
│   │   │   └── filters.py
│   │   ├── gesture/
│   │   │   ├── __init__.py
│   │   │   └── handler.py
│   │   ├── voice/
│   │   │   ├── __init__.py
│   │   │   ├── audio_receiver.py
│   │   │   ├── stt_engine.py
│   │   │   └── command_parser.py
│   │   ├── input/
│   │   │   ├── __init__.py
│   │   │   └── controller.py
│   │   └── ui/
│   │       ├── __init__.py
│   │       ├── tray.py
│   │       └── settings_window.py
│   ├── tests/
│   ├── requirements.txt
│   └── setup.py
```

#### Key Dependencies

**Watch (build.gradle):**
```gradle
dependencies {
    // WearOS
    implementation 'androidx.wear:wear:1.3.0'
    implementation 'androidx.wear.compose:compose-material:1.3.0'

    // BLE
    implementation 'no.nordicsemi.android:ble:2.11.0'
    implementation 'no.nordicsemi.android:ble-ktx:2.11.0'

    // Background
    implementation 'androidx.wear:wear-ongoing:1.1.0'

    // Coroutines
    implementation 'org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3'
}
```

**Desktop (requirements.txt):**
```
bleak>=2.1.1
bleak-retry-connector>=3.0.0
pynput>=1.7.6
numpy>=1.24.0
scipy>=1.11.0
faster-whisper>=0.10.0
silero-vad>=4.0.0
opuslib>=3.0.1
PyQt6>=6.6.0
pystray>=0.19.5
dataclasses-json>=0.6.3
```

### 3.2 Development Environment

**Watch Development:**
1. Android Studio Otter 2 (2025.2.2+)
2. WearOS 5 SDK (API 35)
3. Samsung Galaxy Watch for testing (emulator insufficient for BLE/sensors)

**Desktop Development:**
1. Python 3.10+
2. VS Code or PyCharm
3. Virtual environment (venv or conda)

### 3.3 BLE Protocol Specification

**Custom GATT Services:**

| Service | UUID | Description |
|---------|------|-------------|
| Sensor Service | `00001234-0000-1000-8000-00805f9b34fb` | IMU data streaming |
| Gesture Service | `00001235-0000-1000-8000-00805f9b34fb` | Gesture events |
| Audio Service | `00001236-0000-1000-8000-00805f9b34fb` | Voice data |
| Command Service | `00001237-0000-1000-8000-00805f9b34fb` | Control commands |

**Sensor Characteristic Format (28 bytes per sample):**
```
Bytes 0-3:   Accel X (float32, little-endian)
Bytes 4-7:   Accel Y (float32)
Bytes 8-11:  Accel Z (float32)
Bytes 12-15: Gyro X (float32, rad/s)
Bytes 16-19: Gyro Y (float32)
Bytes 20-23: Gyro Z (float32)
Bytes 24-27: Timestamp (uint32, milliseconds)
```

**Batched Packet (10 samples = 280 bytes):**
- Fits within 512-byte MTU
- Sent every ~100ms (10 samples at 100Hz)
- Reduces BLE overhead by 10x

---

## 4. Phase 2: Watch Application

### 4.1 Sensor Data Collection

**Duration:** 5-7 days

**Implementation:**

```kotlin
// SensorStreamingManager.kt
class SensorStreamingManager(context: Context) : SensorEventListener {

    private val sensorManager: SensorManager
    private val accelerometer: Sensor?
    private val gyroscope: Sensor?

    private val sampleBuffer = mutableListOf<SensorSample>()
    private val batchSize = 10

    private var onBatchReady: ((List<SensorSample>) -> Unit)? = null

    fun startStreaming(callback: (List<SensorSample>) -> Unit) {
        onBatchReady = callback

        sensorManager.registerListener(
            this,
            accelerometer,
            SensorManager.SENSOR_DELAY_FASTEST  // 50-100Hz
        )
        sensorManager.registerListener(
            this,
            gyroscope,
            SensorManager.SENSOR_DELAY_FASTEST
        )
    }

    override fun onSensorChanged(event: SensorEvent) {
        // Combine accel + gyro into single sample
        // Buffer until batch is ready
        // Call onBatchReady when 10 samples collected
    }
}
```

**Key Implementation Points:**
- Use `SENSOR_DELAY_FASTEST` for maximum sample rate
- Combine accelerometer and gyroscope data by timestamp
- Buffer 10 samples before sending via BLE
- Include millisecond timestamp in each sample

### 4.2 On-Device Gesture Detection

**Duration:** 5-7 days

**Gesture Types:**
1. **Finger Pinch/Tap** - Accelerometer magnitude spike detection
2. **Double Tap** - Two taps within 500ms window
3. **Hold** - Sustained pinch for drag operations
4. **Wrist Flick** - Gyroscope rotation spike
5. **Palm Orientation** - Gravity direction for mode switching

**Implementation:**

```kotlin
// GestureDetector.kt
class GestureDetector {

    private val tapThreshold = 2.5f  // g acceleration
    private val flickThreshold = 3.0f  // rad/s
    private val holdDuration = 500L  // ms

    fun processAccelerometer(x: Float, y: Float, z: Float, timestamp: Long): GestureEvent? {
        val magnitude = sqrt(x*x + y*y + z*z)

        // Spike detection for taps
        if (magnitude > tapThreshold + 1.0f) {  // +1g baseline
            return detectTapOrDoubleTap(timestamp)
        }

        // Palm orientation
        return detectPalmOrientation(x, y, z)
    }

    fun processGyroscope(x: Float, y: Float, z: Float, timestamp: Long): GestureEvent? {
        // Flick detection
        if (abs(y) > flickThreshold) {
            return if (y > 0) GestureEvent.FLICK_RIGHT else GestureEvent.FLICK_LEFT
        }
        return null
    }
}
```

### 4.3 BLE GATT Server

**Duration:** 5-7 days

**Implementation using Nordic BLE Library:**

```kotlin
// WristControlGattServer.kt
class WristControlGattServer(context: Context) : BleManager(context) {

    private lateinit var sensorCharacteristic: BluetoothGattCharacteristic
    private lateinit var gestureCharacteristic: BluetoothGattCharacteristic
    private lateinit var audioCharacteristic: BluetoothGattCharacteristic

    override fun initialize() {
        // Setup characteristics with notify support
        setWriteCallback(commandCharacteristic) { device, data ->
            handleCommand(data.value)
        }
    }

    fun sendSensorBatch(batch: List<SensorSample>) {
        val data = packSensorBatch(batch)
        sensorCharacteristic.value = data
        notifyCharacteristicChanged(sensorCharacteristic)
    }

    fun sendGestureEvent(event: GestureEvent) {
        val data = byteArrayOf(event.ordinal.toByte())
        gestureCharacteristic.value = data
        notifyCharacteristicChanged(gestureCharacteristic)
    }

    private fun packSensorBatch(batch: List<SensorSample>): ByteArray {
        val buffer = ByteBuffer.allocate(batch.size * 28)
            .order(ByteOrder.LITTLE_ENDIAN)

        batch.forEach { sample ->
            buffer.putFloat(sample.accelX)
            buffer.putFloat(sample.accelY)
            buffer.putFloat(sample.accelZ)
            buffer.putFloat(sample.gyroX)
            buffer.putFloat(sample.gyroY)
            buffer.putFloat(sample.gyroZ)
            buffer.putInt(sample.timestamp.toInt())
        }

        return buffer.array()
    }
}
```

### 4.4 Audio Capture and Streaming

**Duration:** 3-5 days

**Implementation:**

```kotlin
// AudioCaptureManager.kt
class AudioCaptureManager {

    private val sampleRate = 16000
    private val channelConfig = AudioFormat.CHANNEL_IN_MONO
    private val audioFormat = AudioFormat.ENCODING_PCM_16BIT

    private var audioRecord: AudioRecord? = null
    private var opusEncoder: OpusEncoder? = null

    fun startCapture(onAudioData: (ByteArray) -> Unit) {
        audioRecord = AudioRecord(
            MediaRecorder.AudioSource.VOICE_RECOGNITION,
            sampleRate,
            channelConfig,
            audioFormat,
            bufferSize
        )

        // Opus compression: 16kbps for voice
        opusEncoder = OpusEncoder(sampleRate, 1, OpusEncoder.OPUS_APPLICATION_VOIP)
        opusEncoder?.bitrate = 16000

        CoroutineScope(Dispatchers.IO).launch {
            val pcmBuffer = ShortArray(160)  // 10ms at 16kHz
            val opusBuffer = ByteArray(1024)

            while (isActive) {
                audioRecord?.read(pcmBuffer, 0, pcmBuffer.size)

                val encoded = opusEncoder?.encode(pcmBuffer, opusBuffer)
                if (encoded > 0) {
                    onAudioData(opusBuffer.copyOf(encoded))
                }
            }
        }
    }
}
```

### 4.5 Foreground Service

**Duration:** 2-3 days

**Implementation:**

```kotlin
// StreamingForegroundService.kt
class StreamingForegroundService : Service() {

    private lateinit var sensorManager: SensorStreamingManager
    private lateinit var gestureDetector: GestureDetector
    private lateinit var gattServer: WristControlGattServer

    override fun onCreate() {
        super.onCreate()

        // Create notification with Ongoing Activity
        val notification = createOngoingNotification()
        startForeground(NOTIFICATION_ID, notification)

        // Initialize components
        sensorManager = SensorStreamingManager(this)
        gestureDetector = GestureDetector()
        gattServer = WristControlGattServer(this)
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        // Start all systems
        gattServer.start()
        sensorManager.startStreaming { batch ->
            gattServer.sendSensorBatch(batch)

            // Check for gestures in each sample
            batch.forEach { sample ->
                gestureDetector.process(sample)?.let { gesture ->
                    gattServer.sendGestureEvent(gesture)
                }
            }
        }

        return START_STICKY
    }
}
```

---

## 5. Phase 3: Desktop Application

### 5.1 BLE Connection Manager

**Duration:** 5-7 days

**Implementation:**

```python
# ble/connection.py
import asyncio
from bleak import BleakClient, BleakScanner
from typing import Optional, Callable

class WatchConnectionManager:
    """Manages BLE connection with auto-reconnect."""

    def __init__(self, device_name: str = "Galaxy Watch"):
        self.device_name = device_name
        self.device_address: Optional[str] = None
        self.client: Optional[BleakClient] = None
        self.is_connected = False

        # Callbacks
        self.on_connected: Optional[Callable] = None
        self.on_disconnected: Optional[Callable] = None
        self.on_sensor_data: Optional[Callable] = None
        self.on_gesture: Optional[Callable] = None
        self.on_audio: Optional[Callable] = None

    async def find_device(self) -> bool:
        """Scan for watch device."""
        devices = await BleakScanner.discover(timeout=10.0)

        for device in devices:
            if self.device_name in (device.name or ""):
                self.device_address = device.address
                return True
        return False

    async def connect(self) -> bool:
        """Connect to watch with retry logic."""
        if not self.device_address:
            if not await self.find_device():
                return False

        self.client = BleakClient(
            self.device_address,
            disconnected_callback=self._on_disconnect
        )

        await self.client.connect()

        # Negotiate MTU
        mtu = await self.client.mtu_size
        print(f"Negotiated MTU: {mtu}")

        # Subscribe to notifications
        await self._subscribe_characteristics()

        self.is_connected = True
        if self.on_connected:
            self.on_connected()

        return True

    async def maintain_connection(self):
        """Main loop with auto-reconnect."""
        reconnect_delay = 2.0

        while True:
            try:
                if not self.is_connected:
                    await self.connect()
                    reconnect_delay = 2.0  # Reset on success

                # Use polling for low latency sensor data
                await self._poll_sensor_data()

            except Exception as e:
                print(f"Connection error: {e}")
                self.is_connected = False

                if self.on_disconnected:
                    self.on_disconnected()

                # Exponential backoff
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, 60.0)

    async def _poll_sensor_data(self):
        """Poll sensor characteristic for low latency."""
        SENSOR_CHAR_UUID = "00001234-0001-1000-8000-00805f9b34fb"

        while self.is_connected:
            try:
                data = await self.client.read_gatt_char(SENSOR_CHAR_UUID)

                if self.on_sensor_data:
                    samples = self._unpack_sensor_data(data)
                    for sample in samples:
                        self.on_sensor_data(sample)

                await asyncio.sleep(0)  # Yield to event loop

            except Exception:
                break
```

### 5.2 Sensor Data Processing

**Duration:** 5-7 days

**Madgwick Filter Implementation:**

```python
# motion/sensor_fusion.py
import numpy as np
from dataclasses import dataclass

@dataclass
class Quaternion:
    w: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

class MadgwickFilter:
    """
    Madgwick AHRS filter for sensor fusion.
    Converts accelerometer + gyroscope data to orientation.
    """

    def __init__(self, sample_freq: float = 100.0, beta: float = 0.1):
        self.sample_freq = sample_freq
        self.beta = beta  # Filter gain (0.01-0.5)
        self.q = Quaternion()

    def update(self, gyro: np.ndarray, accel: np.ndarray) -> Quaternion:
        """Update filter with new sensor data."""
        q = self.q
        gx, gy, gz = gyro
        ax, ay, az = accel

        # Normalize accelerometer
        norm = np.sqrt(ax*ax + ay*ay + az*az)
        if norm == 0:
            return q
        ax, ay, az = ax/norm, ay/norm, az/norm

        # Gradient descent algorithm
        _2q0 = 2.0 * q.w
        _2q1 = 2.0 * q.x
        _2q2 = 2.0 * q.y
        _2q3 = 2.0 * q.z
        _4q0 = 4.0 * q.w
        _4q1 = 4.0 * q.x
        _4q2 = 4.0 * q.y
        _8q1 = 8.0 * q.x
        _8q2 = 8.0 * q.y
        q0q0 = q.w * q.w
        q1q1 = q.x * q.x
        q2q2 = q.y * q.y
        q3q3 = q.z * q.z

        # Gradient
        s0 = _4q0 * q2q2 + _2q2 * ax + _4q0 * q1q1 - _2q1 * ay
        s1 = _4q1 * q3q3 - _2q3 * ax + 4.0 * q0q0 * q.x - _2q0 * ay - _4q1 + _8q1 * q1q1 + _8q1 * q2q2 + _4q1 * az
        s2 = 4.0 * q0q0 * q.y + _2q0 * ax + _4q2 * q3q3 - _2q3 * ay - _4q2 + _8q2 * q1q1 + _8q2 * q2q2 + _4q2 * az
        s3 = 4.0 * q1q1 * q.z - _2q1 * ax + 4.0 * q2q2 * q.z - _2q2 * ay

        # Normalize gradient
        norm = np.sqrt(s0*s0 + s1*s1 + s2*s2 + s3*s3)
        if norm > 0:
            s0, s1, s2, s3 = s0/norm, s1/norm, s2/norm, s3/norm

        # Apply feedback
        qDot1 = 0.5 * (-q.x * gx - q.y * gy - q.z * gz) - self.beta * s0
        qDot2 = 0.5 * (q.w * gx + q.y * gz - q.z * gy) - self.beta * s1
        qDot3 = 0.5 * (q.w * gy - q.x * gz + q.z * gx) - self.beta * s2
        qDot4 = 0.5 * (q.w * gz + q.x * gy - q.y * gx) - self.beta * s3

        # Integrate
        dt = 1.0 / self.sample_freq
        q.w += qDot1 * dt
        q.x += qDot2 * dt
        q.y += qDot3 * dt
        q.z += qDot4 * dt

        # Normalize quaternion
        norm = np.sqrt(q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z)
        q.w, q.x, q.y, q.z = q.w/norm, q.x/norm, q.y/norm, q.z/norm

        self.q = q
        return q

    def get_euler_angles(self) -> tuple:
        """Convert quaternion to Euler angles (roll, pitch, yaw)."""
        q = self.q

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (q.w * q.x + q.y * q.z)
        cosr_cosp = 1 - 2 * (q.x * q.x + q.y * q.y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (q.w * q.y - q.z * q.x)
        pitch = np.arcsin(np.clip(sinp, -1, 1))

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)
```

**One Euro Filter for Smoothing:**

```python
# motion/filters.py
import math
import time

class OneEuroFilter:
    """
    Adaptive low-pass filter for cursor smoothing.
    Reduces jitter while maintaining responsiveness.
    """

    def __init__(self, freq: float = 100.0, mincutoff: float = 1.0,
                 beta: float = 0.007, dcutoff: float = 1.0):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff

        self.x_prev = None
        self.dx_prev = 0.0

    def _smoothing_factor(self, cutoff: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        te = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)

    def filter(self, x: float) -> float:
        if self.x_prev is None:
            self.x_prev = x
            return x

        # Calculate derivative
        dx = (x - self.x_prev) * self.freq

        # Smooth derivative
        alpha_d = self._smoothing_factor(self.dcutoff)
        dx_smoothed = alpha_d * dx + (1 - alpha_d) * self.dx_prev

        # Adaptive cutoff
        cutoff = self.mincutoff + self.beta * abs(dx_smoothed)

        # Smooth value
        alpha = self._smoothing_factor(cutoff)
        x_filtered = alpha * x + (1 - alpha) * self.x_prev

        self.x_prev = x_filtered
        self.dx_prev = dx_smoothed

        return x_filtered

class TwoDimensionalOneEuroFilter:
    """One Euro Filter for 2D cursor position."""

    def __init__(self, freq: float = 100.0, mincutoff: float = 1.0,
                 beta: float = 0.007, dcutoff: float = 1.0):
        self.filter_x = OneEuroFilter(freq, mincutoff, beta, dcutoff)
        self.filter_y = OneEuroFilter(freq, mincutoff, beta, dcutoff)

    def filter(self, x: float, y: float) -> tuple:
        return self.filter_x.filter(x), self.filter_y.filter(y)
```

### 5.3 Cursor Control System

**Duration:** 5-7 days

**Implementation:**

```python
# motion/cursor_control.py
import numpy as np
from pynput.mouse import Controller, Button
from dataclasses import dataclass
from typing import Optional

from .sensor_fusion import MadgwickFilter
from .filters import TwoDimensionalOneEuroFilter

@dataclass
class CursorConfig:
    sensitivity: float = 2.0
    dead_zone: float = 2.0  # degrees
    max_speed: float = 1200  # pixels/second
    smoothing_beta: float = 0.007

class CursorController:
    """
    Converts orientation to cursor movement.
    Uses relative positioning (tilt-to-move).
    """

    def __init__(self, config: CursorConfig = None):
        self.config = config or CursorConfig()
        self.mouse = Controller()

        # Sensor fusion
        self.madgwick = MadgwickFilter(sample_freq=100.0, beta=0.1)

        # Smoothing filter
        self.smoother = TwoDimensionalOneEuroFilter(
            freq=100.0,
            mincutoff=1.2,
            beta=self.config.smoothing_beta
        )

        # State
        self.enabled = True
        self.calibration_offset = (0.0, 0.0)
        self.last_update = 0

    def process_sensor_data(self, accel: np.ndarray, gyro: np.ndarray,
                           timestamp: float):
        """Process IMU data and update cursor position."""
        if not self.enabled:
            return

        # Update orientation
        self.madgwick.update(gyro, accel)
        roll, pitch, yaw = self.madgwick.get_euler_angles()

        # Apply calibration offset
        roll -= self.calibration_offset[0]
        pitch -= self.calibration_offset[1]

        # Apply dead zone
        roll = self._apply_dead_zone(roll)
        pitch = self._apply_dead_zone(pitch)

        # Convert to velocity
        dt = timestamp - self.last_update if self.last_update > 0 else 0.01
        self.last_update = timestamp

        velocity_x = roll * self.config.sensitivity
        velocity_y = -pitch * self.config.sensitivity  # Invert pitch

        # Apply speed limit
        speed = np.sqrt(velocity_x**2 + velocity_y**2)
        if speed > self.config.max_speed:
            scale = self.config.max_speed / speed
            velocity_x *= scale
            velocity_y *= scale

        # Smooth
        velocity_x, velocity_y = self.smoother.filter(velocity_x, velocity_y)

        # Calculate movement
        dx = int(velocity_x * dt)
        dy = int(velocity_y * dt)

        # Apply movement
        if abs(dx) > 0 or abs(dy) > 0:
            current_x, current_y = self.mouse.position
            self.mouse.position = (current_x + dx, current_y + dy)

    def _apply_dead_zone(self, value: float) -> float:
        """Apply radial dead zone."""
        if abs(value) < self.config.dead_zone:
            return 0.0

        sign = 1 if value > 0 else -1
        return sign * (abs(value) - self.config.dead_zone)

    def calibrate(self):
        """Set current orientation as neutral position."""
        roll, pitch, _ = self.madgwick.get_euler_angles()
        self.calibration_offset = (roll, pitch)
        print(f"Calibrated: roll={roll:.1f}°, pitch={pitch:.1f}°")

    def click(self, button: Button = Button.left, count: int = 1):
        """Perform mouse click."""
        self.mouse.click(button, count)

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False
```

### 5.4 Gesture Handler

**Duration:** 3-5 days

**Implementation:**

```python
# gesture/handler.py
from enum import IntEnum
from pynput.mouse import Button, Controller as MouseController
from pynput.keyboard import Key, Controller as KeyboardController

class GestureType(IntEnum):
    TAP = 0
    DOUBLE_TAP = 1
    HOLD_START = 2
    HOLD_END = 3
    FLICK_LEFT = 4
    FLICK_RIGHT = 5
    PALM_UP = 6
    PALM_DOWN = 7

class GestureHandler:
    """Translates gesture events to system actions."""

    def __init__(self, cursor_controller):
        self.cursor = cursor_controller
        self.mouse = MouseController()
        self.keyboard = KeyboardController()

        self.is_dragging = False
        self.is_cursor_enabled = True

        # Configurable gesture mappings
        self.mappings = {
            GestureType.TAP: self._handle_tap,
            GestureType.DOUBLE_TAP: self._handle_double_tap,
            GestureType.HOLD_START: self._handle_hold_start,
            GestureType.HOLD_END: self._handle_hold_end,
            GestureType.FLICK_LEFT: self._handle_flick_left,
            GestureType.FLICK_RIGHT: self._handle_flick_right,
            GestureType.PALM_UP: self._handle_palm_up,
            GestureType.PALM_DOWN: self._handle_palm_down,
        }

    def handle_gesture(self, gesture_type: int):
        """Process incoming gesture event."""
        try:
            gesture = GestureType(gesture_type)
            handler = self.mappings.get(gesture)
            if handler:
                handler()
        except ValueError:
            print(f"Unknown gesture type: {gesture_type}")

    def _handle_tap(self):
        """Left click."""
        self.mouse.click(Button.left, 1)

    def _handle_double_tap(self):
        """Double click."""
        self.mouse.click(Button.left, 2)

    def _handle_hold_start(self):
        """Begin drag operation."""
        self.mouse.press(Button.left)
        self.is_dragging = True

    def _handle_hold_end(self):
        """End drag operation."""
        self.mouse.release(Button.left)
        self.is_dragging = False

    def _handle_flick_left(self):
        """Browser back or scroll left."""
        self.mouse.scroll(-3, 0)

    def _handle_flick_right(self):
        """Browser forward or scroll right."""
        self.mouse.scroll(3, 0)

    def _handle_palm_up(self):
        """Disable cursor control (rest position)."""
        self.cursor.disable()
        self.is_cursor_enabled = False

    def _handle_palm_down(self):
        """Enable cursor control (active position)."""
        self.cursor.enable()
        self.is_cursor_enabled = True
```

---

## 6. Phase 4: Voice Integration

### 6.1 Audio Processing Pipeline

**Duration:** 5-7 days

**Implementation:**

```python
# voice/audio_receiver.py
import opuslib
import numpy as np
from collections import deque
from typing import Callable

class AudioReceiver:
    """Receives and decodes Opus audio from watch."""

    def __init__(self):
        self.decoder = opuslib.Decoder(fs=16000, channels=1)
        self.audio_buffer = deque(maxlen=160000)  # 10 seconds
        self.on_speech_complete: Callable = None

        # VAD state
        self.is_speech = False
        self.silence_chunks = 0
        self.speech_buffer = []

    def process_packet(self, opus_data: bytes):
        """Decode Opus packet and detect speech."""
        try:
            # Decode to PCM
            pcm_data = self.decoder.decode(opus_data, frame_size=160)
            audio_np = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0

            # Voice activity detection
            if self._is_speech(audio_np):
                if not self.is_speech:
                    self.is_speech = True
                    print("Speech started")

                self.speech_buffer.extend(audio_np)
                self.silence_chunks = 0
            else:
                if self.is_speech:
                    self.silence_chunks += 1

                    # End of speech after 500ms silence
                    if self.silence_chunks > 15:  # 15 * 32ms
                        print("Speech ended")

                        if self.on_speech_complete and len(self.speech_buffer) > 1600:
                            audio = np.array(self.speech_buffer)
                            self.on_speech_complete(audio)

                        self.speech_buffer = []
                        self.is_speech = False

        except Exception as e:
            print(f"Audio decode error: {e}")

    def _is_speech(self, audio: np.ndarray) -> bool:
        """Simple energy-based VAD (replace with Silero for production)."""
        energy = np.sqrt(np.mean(audio ** 2))
        return energy > 0.01  # Threshold
```

### 6.2 Speech-to-Text Engine

**Duration:** 5-7 days

**Implementation with faster-whisper:**

```python
# voice/stt_engine.py
from faster_whisper import WhisperModel
import numpy as np
from typing import Optional

class STTEngine:
    """Local speech-to-text using faster-whisper."""

    def __init__(self, model_size: str = "tiny"):
        print(f"Loading Whisper {model_size} model...")

        self.model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8"  # Quantized for speed
        )

        print("Model loaded")

    def transcribe(self, audio: np.ndarray) -> Optional[str]:
        """Transcribe audio to text."""
        if len(audio) < 1600:  # < 100ms
            return None

        try:
            # Whisper expects float32 audio
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            segments, info = self.model.transcribe(
                audio,
                beam_size=1,  # Fast decoding
                language="en",
                vad_filter=True
            )

            text = " ".join(segment.text for segment in segments)
            return text.strip()

        except Exception as e:
            print(f"Transcription error: {e}")
            return None
```

### 6.3 Command Parser

**Duration:** 3-5 days

**Implementation:**

```python
# voice/command_parser.py
import re
from typing import Dict, Any, Optional

class CommandParser:
    """Rule-based voice command parser."""

    def __init__(self):
        self.patterns = {
            'click': [
                r'\b(click|tap|press)\b',
                r'\bleft click\b',
            ],
            'double_click': [
                r'\bdouble click\b',
                r'\bclick twice\b',
            ],
            'right_click': [
                r'\bright click\b',
                r'\bcontext menu\b',
            ],
            'scroll_up': [
                r'\bscroll up\b',
                r'\bscroll up (\d+)',
            ],
            'scroll_down': [
                r'\bscroll down\b',
                r'\bscroll down (\d+)',
            ],
            'type': [
                r'\btype (.+)',
                r'\benter (.+)',
            ],
            'press_key': [
                r'\bpress (enter|escape|tab|space|backspace)\b',
                r'\bpress (control|alt|shift) ([a-z])\b',
            ],
            'calibrate': [
                r'\bcalibrate\b',
                r'\breset position\b',
            ],
        }

    def parse(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse voice command text."""
        text = text.lower().strip()

        for command, patterns in self.patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    return self._extract_command(command, match)

        return None

    def _extract_command(self, command: str, match) -> Dict[str, Any]:
        result = {'command': command}

        if command in ('scroll_up', 'scroll_down'):
            result['count'] = int(match.group(1)) if match.lastindex else 1

        elif command == 'type':
            result['text'] = match.group(1)

        elif command == 'press_key':
            if match.lastindex == 1:
                result['key'] = match.group(1)
            else:
                result['modifier'] = match.group(1)
                result['key'] = match.group(2)

        return result
```

### 6.4 Voice Command Executor

**Duration:** 2-3 days

**Implementation:**

```python
# voice/executor.py
from pynput.mouse import Button, Controller as MouseController
from pynput.keyboard import Key, Controller as KeyboardController

class VoiceCommandExecutor:
    """Executes parsed voice commands."""

    def __init__(self, cursor_controller):
        self.cursor = cursor_controller
        self.mouse = MouseController()
        self.keyboard = KeyboardController()

        self.key_map = {
            'enter': Key.enter,
            'escape': Key.esc,
            'tab': Key.tab,
            'space': Key.space,
            'backspace': Key.backspace,
        }

        self.modifier_map = {
            'control': Key.ctrl,
            'alt': Key.alt,
            'shift': Key.shift,
        }

    def execute(self, command: dict):
        """Execute a parsed command."""
        cmd_type = command['command']

        handlers = {
            'click': self._click,
            'double_click': self._double_click,
            'right_click': self._right_click,
            'scroll_up': lambda c: self._scroll(c, up=True),
            'scroll_down': lambda c: self._scroll(c, up=False),
            'type': self._type_text,
            'press_key': self._press_key,
            'calibrate': lambda c: self.cursor.calibrate(),
        }

        handler = handlers.get(cmd_type)
        if handler:
            handler(command)
            print(f"Executed: {command}")

    def _click(self, cmd):
        self.mouse.click(Button.left, 1)

    def _double_click(self, cmd):
        self.mouse.click(Button.left, 2)

    def _right_click(self, cmd):
        self.mouse.click(Button.right, 1)

    def _scroll(self, cmd, up: bool):
        count = cmd.get('count', 1)
        amount = 3 if up else -3
        for _ in range(count):
            self.mouse.scroll(0, amount)

    def _type_text(self, cmd):
        text = cmd.get('text', '')
        self.keyboard.type(text)

    def _press_key(self, cmd):
        key_name = cmd.get('key')
        modifier = cmd.get('modifier')

        key = self.key_map.get(key_name, key_name)

        if modifier:
            mod_key = self.modifier_map[modifier]
            with self.keyboard.pressed(mod_key):
                self.keyboard.press(key)
                self.keyboard.release(key)
        else:
            self.keyboard.press(key)
            self.keyboard.release(key)
```

---

## 7. Phase 5: Cursor Control

### 7.1 UX Pipeline Integration

**Duration:** 3-5 days

The complete cursor processing pipeline:

```python
# motion/pipeline.py
import numpy as np
from dataclasses import dataclass

@dataclass
class PipelineConfig:
    # Sensor fusion
    madgwick_beta: float = 0.1

    # Dead zone
    dead_zone_radius: float = 2.0  # degrees

    # Sensitivity
    sensitivity: float = 2.0
    sensitivity_curve_power: float = 2.0

    # Smoothing
    smoothing_mincutoff: float = 1.2
    smoothing_beta: float = 0.007

    # Speed limits
    max_velocity: float = 1200  # pixels/sec
    max_acceleration: float = 4000  # pixels/sec²

class CursorPipeline:
    """Complete motion-to-cursor processing pipeline."""

    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()

        # Components
        self.madgwick = MadgwickFilter(beta=self.config.madgwick_beta)
        self.smoother = TwoDimensionalOneEuroFilter(
            mincutoff=self.config.smoothing_mincutoff,
            beta=self.config.smoothing_beta
        )

        # State
        self.calibration = (0.0, 0.0)
        self.last_velocity = np.array([0.0, 0.0])

    def process(self, accel: np.ndarray, gyro: np.ndarray,
                dt: float) -> tuple:
        """
        Full processing pipeline.
        Returns (dx, dy) cursor movement in pixels.
        """
        # 1. Sensor fusion
        self.madgwick.update(gyro, accel)
        roll, pitch, _ = self.madgwick.get_euler_angles()

        # 2. Calibration offset
        roll -= self.calibration[0]
        pitch -= self.calibration[1]

        # 3. Dead zone
        roll = self._apply_dead_zone(roll)
        pitch = self._apply_dead_zone(pitch)

        # 4. Sensitivity curve
        velocity_x = self._apply_sensitivity_curve(roll)
        velocity_y = self._apply_sensitivity_curve(-pitch)

        # 5. Smoothing
        velocity_x, velocity_y = self.smoother.filter(velocity_x, velocity_y)

        # 6. Acceleration limiting
        velocity = np.array([velocity_x, velocity_y])
        velocity = self._apply_acceleration_limit(velocity, dt)

        # 7. Speed limiting
        speed = np.linalg.norm(velocity)
        if speed > self.config.max_velocity:
            velocity = velocity * (self.config.max_velocity / speed)

        # 8. Calculate movement
        dx = int(velocity[0] * dt)
        dy = int(velocity[1] * dt)

        return dx, dy

    def _apply_dead_zone(self, value: float) -> float:
        """Radial dead zone."""
        if abs(value) < self.config.dead_zone_radius:
            return 0.0
        sign = 1 if value > 0 else -1
        return sign * (abs(value) - self.config.dead_zone_radius)

    def _apply_sensitivity_curve(self, value: float) -> float:
        """Power curve for non-linear sensitivity."""
        sign = 1 if value > 0 else -1
        normalized = min(1.0, abs(value) / 90.0)  # Normalize to 0-1
        curved = normalized ** self.config.sensitivity_curve_power
        return sign * curved * 90.0 * self.config.sensitivity

    def _apply_acceleration_limit(self, target_velocity: np.ndarray,
                                   dt: float) -> np.ndarray:
        """Limit rate of velocity change."""
        delta = target_velocity - self.last_velocity
        delta_mag = np.linalg.norm(delta)

        max_delta = self.config.max_acceleration * dt

        if delta_mag > max_delta:
            delta = delta * (max_delta / delta_mag)

        self.last_velocity = self.last_velocity + delta
        return self.last_velocity

    def calibrate(self):
        """Set current orientation as neutral."""
        roll, pitch, _ = self.madgwick.get_euler_angles()
        self.calibration = (roll, pitch)
```

### 7.2 User Profiles

**Duration:** 2-3 days

```python
# config/profiles.py
from dataclasses import dataclass

@dataclass
class UserProfile:
    name: str
    sensitivity: float
    dead_zone: float
    smoothing: float
    enable_target_assist: bool

PROFILES = {
    'default': UserProfile(
        name='Default',
        sensitivity=2.0,
        dead_zone=2.0,
        smoothing=0.007,
        enable_target_assist=False
    ),
    'gaming': UserProfile(
        name='Gaming',
        sensitivity=3.0,
        dead_zone=1.5,
        smoothing=0.015,
        enable_target_assist=False
    ),
    'productivity': UserProfile(
        name='Productivity',
        sensitivity=2.0,
        dead_zone=2.0,
        smoothing=0.007,
        enable_target_assist=True
    ),
    'accessibility': UserProfile(
        name='Accessibility',
        sensitivity=1.5,
        dead_zone=3.0,
        smoothing=0.004,
        enable_target_assist=True
    ),
}
```

---

## 8. Phase 6: Integration & Polish

### 8.1 System Tray Application

**Duration:** 3-5 days

```python
# ui/tray.py
import pystray
from PIL import Image
import threading

class SystemTrayApp:
    """System tray application for WristControl."""

    def __init__(self, app_controller):
        self.controller = app_controller
        self.icon = None

    def create_menu(self):
        """Create context menu."""
        return pystray.Menu(
            pystray.MenuItem(
                "Status: Connected" if self.controller.is_connected else "Status: Disconnected",
                None,
                enabled=False
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Calibrate", self._calibrate),
            pystray.MenuItem("Settings...", self._open_settings),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                "Enable Cursor",
                self._toggle_cursor,
                checked=lambda item: self.controller.cursor_enabled
            ),
            pystray.MenuItem(
                "Enable Voice",
                self._toggle_voice,
                checked=lambda item: self.controller.voice_enabled
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", self._quit),
        )

    def run(self):
        """Run system tray."""
        image = Image.open("assets/icon.png")

        self.icon = pystray.Icon(
            "WristControl",
            image,
            "WristControl",
            self.create_menu()
        )

        self.icon.run()
```

### 8.2 Settings Window

**Duration:** 5-7 days

```python
# ui/settings_window.py
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QComboBox, QCheckBox, QPushButton,
    QTabWidget, QGroupBox
)
from PyQt6.QtCore import Qt

class SettingsWindow(QMainWindow):
    """Settings configuration window."""

    def __init__(self, config_manager):
        super().__init__()
        self.config = config_manager
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle("WristControl Settings")
        self.setMinimumSize(400, 500)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Tabs
        tabs = QTabWidget()
        tabs.addTab(self._create_cursor_tab(), "Cursor")
        tabs.addTab(self._create_voice_tab(), "Voice")
        tabs.addTab(self._create_gestures_tab(), "Gestures")
        tabs.addTab(self._create_connection_tab(), "Connection")
        layout.addWidget(tabs)

        # Buttons
        buttons = QHBoxLayout()
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self._save_settings)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.close)
        buttons.addStretch()
        buttons.addWidget(save_btn)
        buttons.addWidget(cancel_btn)
        layout.addLayout(buttons)

    def _create_cursor_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Sensitivity
        sens_group = QGroupBox("Sensitivity")
        sens_layout = QVBoxLayout(sens_group)

        self.sensitivity_slider = QSlider(Qt.Orientation.Horizontal)
        self.sensitivity_slider.setRange(1, 50)
        self.sensitivity_slider.setValue(int(self.config.get('cursor.sensitivity') * 10))

        sens_layout.addWidget(QLabel("Sensitivity"))
        sens_layout.addWidget(self.sensitivity_slider)
        layout.addWidget(sens_group)

        # Dead zone
        dz_group = QGroupBox("Dead Zone")
        dz_layout = QVBoxLayout(dz_group)

        self.dead_zone_slider = QSlider(Qt.Orientation.Horizontal)
        self.dead_zone_slider.setRange(0, 50)
        self.dead_zone_slider.setValue(int(self.config.get('cursor.dead_zone') * 10))

        dz_layout.addWidget(QLabel("Dead Zone (degrees)"))
        dz_layout.addWidget(self.dead_zone_slider)
        layout.addWidget(dz_group)

        # Profile selector
        profile_group = QGroupBox("Profile")
        profile_layout = QVBoxLayout(profile_group)

        self.profile_combo = QComboBox()
        self.profile_combo.addItems(['Default', 'Gaming', 'Productivity', 'Accessibility'])
        profile_layout.addWidget(self.profile_combo)
        layout.addWidget(profile_group)

        layout.addStretch()
        return widget
```

### 8.3 Configuration Management

**Duration:** 2-3 days

```python
# config/settings.py
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from dataclasses_json import dataclass_json

@dataclass_json
@dataclass
class AppConfig:
    # Connection
    watch_name: str = "Galaxy Watch"
    auto_connect: bool = True

    # Cursor
    cursor_sensitivity: float = 2.0
    cursor_dead_zone: float = 2.0
    cursor_smoothing: float = 0.007
    cursor_enabled: bool = True

    # Voice
    voice_enabled: bool = True
    stt_engine: str = "local"  # local, deepgram
    stt_model: str = "tiny"  # tiny, base, small

    # Gestures
    tap_action: str = "left_click"
    double_tap_action: str = "double_click"
    hold_action: str = "drag"

    # Privacy
    store_history: bool = False
    telemetry: bool = False

class ConfigManager:
    """Manages application configuration."""

    def __init__(self, config_file: str = "~/.wristcontrol/config.json"):
        self.config_path = Path(config_file).expanduser()
        self.config = self.load()

    def load(self) -> AppConfig:
        if self.config_path.exists():
            with open(self.config_path) as f:
                data = json.load(f)
                return AppConfig.from_dict(data)
        return AppConfig()

    def save(self):
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2)

    def get(self, key: str, default=None):
        parts = key.split('.')
        value = self.config
        for part in parts:
            value = getattr(value, part, default)
            if value is default:
                break
        return value

    def set(self, key: str, value):
        parts = key.split('.')
        obj = self.config
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)
        self.save()
```

---

## 9. Technical Specifications

### 9.1 Latency Budget

| Component | Target | Notes |
|-----------|--------|-------|
| IMU sampling | 10ms | 100Hz on watch |
| Watch processing | 5ms | Gesture detection + batching |
| BLE transmission | 15ms | 7.5ms connection interval |
| Desktop BLE poll | 5ms | asyncio.sleep(0) |
| Sensor fusion | 2ms | Madgwick filter |
| Smoothing filter | 1ms | One Euro filter |
| Input injection | 5ms | pynput |
| **Total Cursor** | **43ms** | Within 50ms target |

### 9.2 Voice Command Latency

| Component | Target | Notes |
|-----------|--------|-------|
| Audio capture | 10ms | 10ms buffer |
| Opus encoding | 20ms | On watch |
| BLE transmission | 30ms | ~20 packets for 1 second audio |
| Opus decoding | 10ms | On desktop |
| VAD processing | 10ms | Silero VAD |
| STT processing | 200ms | faster-whisper tiny |
| Command parsing | 5ms | Regex matching |
| Execution | 10ms | pynput |
| **Total Voice** | **295ms** | Within 500ms target |

### 9.3 Battery Consumption Estimates

| Mode | Power Draw | Battery Life (400mAh) |
|------|------------|----------------------|
| Active control (100Hz) | ~40mA | 10 hours |
| Idle monitoring (25Hz) | ~15mA | 26 hours |
| Voice active | ~55mA | 7 hours |
| Combined (cursor + voice) | ~60mA | 6.6 hours |

---

## 10. Risk Mitigation

### 10.1 Technical Risks

| Risk | Mitigation |
|------|------------|
| BLE latency too high | Use polling instead of notifications (proven 20-25ms) |
| Battery drain too fast | Adaptive sampling, batching, gesture-based sleep |
| Cursor jitter | One Euro filter with tuned parameters |
| Voice recognition accuracy | Local Whisper with cloud fallback option |
| Cross-platform issues | pynput abstraction, platform-specific testing |

### 10.2 Development Risks

| Risk | Mitigation |
|------|------------|
| Watch hardware variance | Test on multiple Galaxy Watch models |
| OS permission changes | Monitor Android/WearOS updates |
| Library deprecation | Prefer well-maintained libraries with active communities |

### 10.3 Contingency Plans

1. **If BLE polling doesn't achieve <50ms:**
   - Implement hybrid notification + polling
   - Consider TCP/WiFi for lower latency

2. **If local Whisper is too slow:**
   - Default to Deepgram cloud API
   - Implement voice command caching

3. **If gesture detection accuracy is poor:**
   - Add ML-based gesture recognition
   - Implement user calibration wizard

---

## Implementation Schedule Summary

| Phase | Duration | Milestones |
|-------|----------|------------|
| Phase 1: Foundation | 3-5 days | Project structure, BLE protocol spec |
| Phase 2: Watch App | 15-20 days | Sensors, gestures, BLE server |
| Phase 3: Desktop App | 15-20 days | BLE client, motion processing, UI |
| Phase 4: Voice | 15-20 days | Audio pipeline, STT, commands |
| Phase 5: Cursor Control | 10-15 days | Full pipeline, profiles |
| Phase 6: Integration | 10-15 days | Polish, settings, packaging |
| **Total** | **68-95 days** | |

---

*Document generated from research conducted January 2026*
