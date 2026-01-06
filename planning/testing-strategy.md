# WristControl Testing Strategy
## Voice and Gesture Computer Control System

**Version:** 1.0
**Date:** January 2026
**Project:** WristControl - Samsung Galaxy Watch Companion System

---

## Table of Contents

1. [Testing Overview](#1-testing-overview)
2. [Unit Testing](#2-unit-testing)
3. [Integration Testing](#3-integration-testing)
4. [System Testing](#4-system-testing)
5. [Performance Testing](#5-performance-testing)
6. [User Acceptance Testing](#6-user-acceptance-testing)
7. [Automated Testing](#7-automated-testing)
8. [Testing Tools & Infrastructure](#8-testing-tools--infrastructure)
9. [Test Data & Fixtures](#9-test-data--fixtures)
10. [Quality Metrics](#10-quality-metrics)

---

## 1. Testing Overview

### 1.1 Testing Philosophy

WristControl requires comprehensive testing across two platforms (WearOS and Desktop) with hardware-dependent components (sensors, BLE, microphone). Our testing strategy emphasizes:

1. **Simulation-First**: Unit tests with mock sensor data
2. **Progressive Integration**: Build confidence from components to system
3. **Hardware-in-Loop**: Critical path testing on real devices
4. **Performance Budgets**: Latency and battery targets as test criteria

### 1.2 Testing Pyramid

```
                    ┌─────────────┐
                    │   Manual    │  5%
                    │   E2E       │
                    ├─────────────┤
                    │  System     │  10%
                    │  Tests      │
                ┌───┴─────────────┴───┐
                │   Integration       │  25%
                │   Tests             │
            ┌───┴─────────────────────┴───┐
            │       Unit Tests            │  60%
            │       (Mocked Hardware)     │
            └─────────────────────────────┘
```

### 1.3 Test Categories by Component

| Component | Unit | Integration | System | Hardware |
|-----------|------|-------------|--------|----------|
| Sensor Fusion (Madgwick) | Yes | Yes | - | Yes |
| Gesture Detection | Yes | Yes | Yes | Yes |
| BLE Protocol | Yes | Yes | Yes | Yes |
| Voice Pipeline | Yes | Yes | Yes | Yes |
| Cursor Control | Yes | Yes | Yes | Yes |
| UI Components | Yes | - | Yes | - |

---

## 2. Unit Testing

### 2.1 Watch App (Kotlin)

**Framework:** JUnit 5 + MockK

#### Gesture Detection Tests

```kotlin
// test/kotlin/com/wristcontrol/gesture/GestureDetectorTest.kt
import io.mockk.*
import org.junit.jupiter.api.*
import kotlin.test.assertEquals
import kotlin.test.assertNull

class GestureDetectorTest {

    private lateinit var detector: GestureDetector

    @BeforeEach
    fun setup() {
        detector = GestureDetector()
    }

    @Test
    fun `tap detected on accelerometer spike`() {
        // Baseline readings (1g gravity)
        repeat(10) {
            detector.processAccelerometer(0f, 0f, 9.8f, it * 10_000_000L)
        }

        // Spike (finger pinch)
        val result = detector.processAccelerometer(
            x = 0f,
            y = 0f,
            z = 14.0f,  // ~2.4g total magnitude
            timestamp = 100_000_000L
        )

        assertEquals(GestureEvent.TAP, result)
    }

    @Test
    fun `double tap detected within 500ms window`() {
        // First tap
        detector.processAccelerometer(0f, 0f, 14.0f, 0L)

        // Baseline
        repeat(5) {
            detector.processAccelerometer(0f, 0f, 9.8f, (it + 1) * 50_000_000L)
        }

        // Second tap within 500ms
        val result = detector.processAccelerometer(0f, 0f, 14.0f, 300_000_000L)

        assertEquals(GestureEvent.DOUBLE_TAP, result)
    }

    @Test
    fun `no double tap when taps are over 500ms apart`() {
        // First tap
        detector.processAccelerometer(0f, 0f, 14.0f, 0L)

        // Baseline for 600ms
        repeat(12) {
            detector.processAccelerometer(0f, 0f, 9.8f, (it + 1) * 50_000_000L)
        }

        // Second tap after 600ms - should be new single tap
        val result = detector.processAccelerometer(0f, 0f, 14.0f, 600_000_000L)

        assertEquals(GestureEvent.TAP, result)  // Not DOUBLE_TAP
    }

    @Test
    fun `wrist flick right detected`() {
        val result = detector.processGyroscope(
            x = 0f,
            y = 4.0f,  // > 3.0 threshold
            z = 0f,
            timestamp = 0L
        )

        assertEquals(GestureEvent.FLICK_RIGHT, result)
    }

    @Test
    fun `wrist flick left detected`() {
        val result = detector.processGyroscope(
            x = 0f,
            y = -4.0f,  // < -3.0 threshold
            z = 0f,
            timestamp = 0L
        )

        assertEquals(GestureEvent.FLICK_LEFT, result)
    }

    @Test
    fun `palm up orientation detected`() {
        val result = detector.detectOrientation(0f, 0f, -9.8f)

        assertEquals(PalmOrientation.UP, result)
    }

    @Test
    fun `palm down orientation detected`() {
        val result = detector.detectOrientation(0f, 0f, 9.8f)

        assertEquals(PalmOrientation.DOWN, result)
    }

    @Test
    fun `no gesture on normal movement`() {
        val result = detector.processAccelerometer(
            x = 1.0f,
            y = 1.0f,
            z = 9.8f,
            timestamp = 0L
        )

        assertNull(result)
    }
}
```

#### Sensor Data Packing Tests

```kotlin
// test/kotlin/com/wristcontrol/ble/DataPackingTest.kt
class DataPackingTest {

    @Test
    fun `sensor batch packs correctly to 280 bytes`() {
        val samples = (0 until 10).map { i ->
            SensorSample(
                accelX = 1.0f,
                accelY = 2.0f,
                accelZ = 9.8f,
                gyroX = 0.1f,
                gyroY = 0.2f,
                gyroZ = 0.0f,
                timestamp = i * 10L
            )
        }

        val packed = GattServer.packSensorBatch(samples)

        assertEquals(280, packed.size)
    }

    @Test
    fun `unpacked values match packed values`() {
        val original = SensorSample(
            accelX = 1.23f,
            accelY = -4.56f,
            accelZ = 9.81f,
            gyroX = 0.12f,
            gyroY = -0.34f,
            gyroZ = 0.56f,
            timestamp = 12345L
        )

        val packed = GattServer.packSensorBatch(listOf(original))
        val unpacked = GattServer.unpackSensorSample(packed, 0)

        assertEquals(original.accelX, unpacked.accelX, 0.001f)
        assertEquals(original.accelY, unpacked.accelY, 0.001f)
        assertEquals(original.accelZ, unpacked.accelZ, 0.001f)
        assertEquals(original.gyroX, unpacked.gyroX, 0.001f)
        assertEquals(original.gyroY, unpacked.gyroY, 0.001f)
        assertEquals(original.gyroZ, unpacked.gyroZ, 0.001f)
        assertEquals(original.timestamp, unpacked.timestamp)
    }
}
```

### 2.2 Desktop App (Python)

**Framework:** pytest + pytest-asyncio

#### Madgwick Filter Tests

```python
# tests/motion/test_sensor_fusion.py
import pytest
import numpy as np
from src.motion.sensor_fusion import MadgwickFilter, Quaternion

class TestMadgwickFilter:

    @pytest.fixture
    def filter(self):
        return MadgwickFilter(sample_freq=100.0, beta=0.1)

    def test_initial_quaternion_is_identity(self, filter):
        q = filter.q
        assert q.w == pytest.approx(1.0)
        assert q.x == pytest.approx(0.0)
        assert q.y == pytest.approx(0.0)
        assert q.z == pytest.approx(0.0)

    def test_quaternion_stays_normalized(self, filter):
        """Quaternion should remain normalized after updates."""
        gyro = np.array([0.1, 0.2, 0.3])
        accel = np.array([0.0, 0.0, 9.8])

        for _ in range(100):
            filter.update(gyro, accel)

        q = filter.q
        magnitude = np.sqrt(q.w**2 + q.x**2 + q.y**2 + q.z**2)
        assert magnitude == pytest.approx(1.0, abs=0.001)

    def test_gravity_alignment(self, filter):
        """With no rotation, accelerometer should align to gravity."""
        gyro = np.array([0.0, 0.0, 0.0])
        accel = np.array([0.0, 0.0, -9.8])  # Gravity pointing down

        # Converge
        for _ in range(1000):
            filter.update(gyro, accel)

        roll, pitch, yaw = filter.get_euler_angles()

        # Should be close to zero (no tilt)
        assert roll == pytest.approx(0.0, abs=2.0)
        assert pitch == pytest.approx(0.0, abs=2.0)

    def test_90_degree_roll(self, filter):
        """Tilting 90 degrees should be detected."""
        gyro = np.array([0.0, 0.0, 0.0])
        # Gravity pointing in X direction = 90 degree roll
        accel = np.array([9.8, 0.0, 0.0])

        for _ in range(1000):
            filter.update(gyro, accel)

        roll, pitch, yaw = filter.get_euler_angles()
        assert abs(roll) == pytest.approx(90.0, abs=5.0)

    def test_gyro_integration(self, filter):
        """Gyroscope rotation should affect orientation."""
        # 1 rad/s rotation around Z axis for 1 second at 100Hz
        gyro = np.array([0.0, 0.0, 1.0])
        accel = np.array([0.0, 0.0, 9.8])

        for _ in range(100):
            filter.update(gyro, accel)

        _, _, yaw = filter.get_euler_angles()

        # ~57 degrees in 1 second at 1 rad/s
        assert abs(yaw) > 30  # Should show significant rotation
```

#### One Euro Filter Tests

```python
# tests/motion/test_filters.py
import pytest
from src.motion.filters import OneEuroFilter, TwoDimensionalOneEuroFilter

class TestOneEuroFilter:

    @pytest.fixture
    def filter(self):
        return OneEuroFilter(freq=100.0, mincutoff=1.0, beta=0.007)

    def test_first_value_passes_through(self, filter):
        result = filter.filter(10.0)
        assert result == 10.0

    def test_smoothes_jitter(self, filter):
        """Small oscillations should be smoothed out."""
        values = [10.0, 10.1, 9.9, 10.05, 9.95]
        results = [filter.filter(v) for v in values]

        # Variance should decrease
        input_var = np.var(values)
        output_var = np.var(results)

        assert output_var < input_var

    def test_responds_to_fast_movement(self, filter):
        """Large changes should pass through quickly."""
        # Start at 0
        filter.filter(0.0)

        # Jump to 100 and stay there
        results = []
        for _ in range(10):
            results.append(filter.filter(100.0))

        # Should reach close to 100 quickly
        assert results[-1] == pytest.approx(100.0, abs=5.0)

    def test_adaptive_cutoff(self):
        """Higher beta should make filter more responsive."""
        slow_filter = OneEuroFilter(freq=100.0, mincutoff=1.0, beta=0.001)
        fast_filter = OneEuroFilter(freq=100.0, mincutoff=1.0, beta=0.1)

        values = [0.0] + [100.0] * 10

        slow_results = [slow_filter.filter(v) for v in values]
        fast_results = [fast_filter.filter(v) for v in values]

        # Fast filter should reach 100 more quickly
        assert fast_results[5] > slow_results[5]

class TestTwoDimensionalOneEuroFilter:

    def test_filters_both_dimensions(self):
        filter = TwoDimensionalOneEuroFilter()

        x, y = filter.filter(10.0, 20.0)
        assert x == 10.0
        assert y == 20.0

        x, y = filter.filter(10.1, 20.1)
        # Should be smoothed
        assert 10.0 <= x <= 10.1
        assert 20.0 <= y <= 20.1
```

#### Voice Command Parser Tests

```python
# tests/voice/test_command_parser.py
import pytest
from src.voice.command_parser import CommandParser

class TestCommandParser:

    @pytest.fixture
    def parser(self):
        return CommandParser()

    # Click commands
    def test_parses_click(self, parser):
        result = parser.parse("click")
        assert result['command'] == 'click'

    def test_parses_tap(self, parser):
        result = parser.parse("tap")
        assert result['command'] == 'click'

    def test_parses_double_click(self, parser):
        result = parser.parse("double click")
        assert result['command'] == 'double_click'

    def test_parses_right_click(self, parser):
        result = parser.parse("right click")
        assert result['command'] == 'right_click'

    # Scroll commands
    def test_parses_scroll_down(self, parser):
        result = parser.parse("scroll down")
        assert result['command'] == 'scroll_down'
        assert result.get('count', 1) == 1

    def test_parses_scroll_with_count(self, parser):
        result = parser.parse("scroll down 5")
        assert result['command'] == 'scroll_down'
        assert result['count'] == 5

    def test_parses_scroll_up(self, parser):
        result = parser.parse("scroll up")
        assert result['command'] == 'scroll_up'

    # Type commands
    def test_parses_type_text(self, parser):
        result = parser.parse("type hello world")
        assert result['command'] == 'type'
        assert result['text'] == 'hello world'

    def test_parses_enter_text(self, parser):
        result = parser.parse("enter my email")
        assert result['command'] == 'type'
        assert result['text'] == 'my email'

    # Key commands
    def test_parses_press_enter(self, parser):
        result = parser.parse("press enter")
        assert result['command'] == 'press_key'
        assert result['key'] == 'enter'

    def test_parses_press_escape(self, parser):
        result = parser.parse("press escape")
        assert result['command'] == 'press_key'
        assert result['key'] == 'escape'

    def test_parses_control_key_combo(self, parser):
        result = parser.parse("press control c")
        assert result['command'] == 'press_key'
        assert result['modifier'] == 'control'
        assert result['key'] == 'c'

    # Edge cases
    def test_case_insensitive(self, parser):
        result = parser.parse("CLICK")
        assert result['command'] == 'click'

    def test_extra_whitespace(self, parser):
        result = parser.parse("  double   click  ")
        assert result['command'] == 'double_click'

    def test_unknown_command_returns_none(self, parser):
        result = parser.parse("do something weird")
        assert result is None

    def test_partial_match_works(self, parser):
        result = parser.parse("please click here")
        assert result['command'] == 'click'
```

#### Cursor Control Tests

```python
# tests/motion/test_cursor_control.py
import pytest
from unittest.mock import Mock, patch
import numpy as np
from src.motion.cursor_control import CursorController, CursorConfig

class TestCursorController:

    @pytest.fixture
    def mock_mouse(self):
        with patch('src.motion.cursor_control.Controller') as mock:
            mouse = Mock()
            mouse.position = (500, 500)
            mock.return_value = mouse
            yield mouse

    @pytest.fixture
    def controller(self, mock_mouse):
        return CursorController(CursorConfig())

    def test_neutral_position_no_movement(self, controller, mock_mouse):
        """Neutral orientation should not move cursor."""
        accel = np.array([0.0, 0.0, -9.8])  # Level
        gyro = np.array([0.0, 0.0, 0.0])

        # Let filter converge
        for i in range(100):
            controller.process_sensor_data(accel, gyro, i * 0.01)

        # After convergence, should be minimal movement
        mock_mouse.position = (500, 500)

        controller.process_sensor_data(accel, gyro, 1.01)

        # Check final position hasn't moved significantly
        final_x, final_y = mock_mouse.position
        assert abs(final_x - 500) < 10
        assert abs(final_y - 500) < 10

    def test_tilt_right_moves_cursor_right(self, controller, mock_mouse):
        """Tilting right should move cursor right."""
        # Tilt 20 degrees right
        tilt_angle = 20 * np.pi / 180
        accel = np.array([
            9.8 * np.sin(tilt_angle),
            0.0,
            -9.8 * np.cos(tilt_angle)
        ])
        gyro = np.array([0.0, 0.0, 0.0])

        initial_x = 500
        mock_mouse.position = (initial_x, 500)

        # Process samples
        for i in range(100):
            controller.process_sensor_data(accel, gyro, i * 0.01)

        final_x, _ = mock_mouse.position
        assert final_x > initial_x

    def test_dead_zone_prevents_small_movements(self, controller, mock_mouse):
        """Small tilts within dead zone should not move cursor."""
        controller.config.dead_zone = 5.0  # 5 degrees

        # Tilt 2 degrees (within dead zone)
        tilt_angle = 2 * np.pi / 180
        accel = np.array([
            9.8 * np.sin(tilt_angle),
            0.0,
            -9.8 * np.cos(tilt_angle)
        ])
        gyro = np.array([0.0, 0.0, 0.0])

        mock_mouse.position = (500, 500)

        # Converge filter
        for i in range(200):
            controller.process_sensor_data(accel, gyro, i * 0.01)

        final_x, final_y = mock_mouse.position
        # Should be minimal movement
        assert abs(final_x - 500) < 20
        assert abs(final_y - 500) < 20

    def test_calibration_sets_neutral(self, controller, mock_mouse):
        """Calibration should set current position as neutral."""
        # Tilt to 20 degrees
        tilt_angle = 20 * np.pi / 180
        accel = np.array([
            9.8 * np.sin(tilt_angle),
            0.0,
            -9.8 * np.cos(tilt_angle)
        ])
        gyro = np.array([0.0, 0.0, 0.0])

        # Converge filter
        for i in range(200):
            controller.process_sensor_data(accel, gyro, i * 0.01)

        # Calibrate
        controller.calibrate()

        # Now same tilt should be neutral
        mock_mouse.position = (500, 500)

        for i in range(200, 400):
            controller.process_sensor_data(accel, gyro, i * 0.01)

        final_x, final_y = mock_mouse.position
        assert abs(final_x - 500) < 20
        assert abs(final_y - 500) < 20

    def test_disable_prevents_movement(self, controller, mock_mouse):
        """Disabled controller should not move cursor."""
        controller.disable()

        accel = np.array([9.8, 0.0, 0.0])  # 90 degree tilt
        gyro = np.array([0.0, 0.0, 0.0])

        mock_mouse.position = (500, 500)

        for i in range(100):
            controller.process_sensor_data(accel, gyro, i * 0.01)

        assert mock_mouse.position == (500, 500)
```

---

## 3. Integration Testing

### 3.1 BLE Communication Tests

```python
# tests/integration/test_ble_integration.py
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from src.ble.connection import WatchConnectionManager

class TestBLEIntegration:

    @pytest.fixture
    def connection_manager(self):
        return WatchConnectionManager(device_name="Test Watch")

    @pytest.mark.asyncio
    async def test_connection_retry_on_failure(self, connection_manager):
        """Should retry connection with exponential backoff."""
        connect_attempts = []

        async def mock_connect():
            connect_attempts.append(len(connect_attempts))
            if len(connect_attempts) < 3:
                raise Exception("Connection failed")
            return True

        with patch.object(connection_manager, 'connect', side_effect=mock_connect):
            with patch.object(asyncio, 'sleep', new_callable=AsyncMock):
                # Start connection loop with timeout
                try:
                    await asyncio.wait_for(
                        connection_manager.maintain_connection(),
                        timeout=5.0
                    )
                except asyncio.TimeoutError:
                    pass

        assert len(connect_attempts) >= 3

    @pytest.mark.asyncio
    async def test_sensor_data_callback(self, connection_manager):
        """Sensor data should be unpacked and callback invoked."""
        received_samples = []

        def on_sensor(sample):
            received_samples.append(sample)

        connection_manager.on_sensor_data = on_sensor

        # Simulate receiving packed data
        packed_data = bytes(28)  # 1 sample
        samples = connection_manager._unpack_sensor_data(packed_data)

        for sample in samples:
            on_sensor(sample)

        assert len(received_samples) == 1

    @pytest.mark.asyncio
    async def test_gesture_callback(self, connection_manager):
        """Gesture events should trigger callbacks."""
        received_gestures = []

        connection_manager.on_gesture = lambda g: received_gestures.append(g)

        # Simulate gesture event
        connection_manager._handle_gesture(bytes([0]))  # TAP

        assert len(received_gestures) == 1
        assert received_gestures[0] == 0  # TAP enum value
```

### 3.2 Voice Pipeline Integration

```python
# tests/integration/test_voice_pipeline.py
import pytest
import numpy as np
from src.voice.audio_receiver import AudioReceiver
from src.voice.stt_engine import STTEngine
from src.voice.command_parser import CommandParser

class TestVoicePipeline:

    @pytest.fixture
    def audio_receiver(self):
        return AudioReceiver()

    @pytest.fixture
    def stt_engine(self):
        # Use tiny model for tests
        return STTEngine(model_size="tiny")

    @pytest.fixture
    def parser(self):
        return CommandParser()

    def test_full_pipeline_click_command(self, audio_receiver, stt_engine, parser):
        """Test full pipeline from audio to command."""
        commands_received = []

        # Load test audio file (pre-recorded "click")
        test_audio = np.load("tests/fixtures/audio_click.npy")

        # STT
        text = stt_engine.transcribe(test_audio)
        assert text is not None

        # Parse
        command = parser.parse(text)
        if command:
            commands_received.append(command)

        # Should detect click command
        assert any(c['command'] == 'click' for c in commands_received)

    def test_vad_segments_speech(self, audio_receiver):
        """VAD should correctly segment speech from silence."""
        # Silent audio
        silence = np.zeros(1600, dtype=np.float32)  # 100ms
        assert not audio_receiver._is_speech(silence)

        # Speech-like audio (random with energy)
        speech = np.random.randn(1600).astype(np.float32) * 0.1
        # This depends on VAD implementation
        # May need pre-recorded test audio

    def test_speech_end_triggers_callback(self, audio_receiver):
        """End of speech should trigger transcription callback."""
        callback_invoked = []

        audio_receiver.on_speech_complete = lambda audio: callback_invoked.append(audio)

        # Simulate speech followed by silence
        speech = np.random.randn(16000).astype(np.float32) * 0.1  # 1 second

        # Process speech (would need to simulate packet-by-packet)
        # For integration test, call internal method
        audio_receiver.speech_buffer = speech.tolist()
        audio_receiver.is_speech = True
        audio_receiver.silence_chunks = 20  # Trigger end

        audio_receiver._check_speech_end()

        assert len(callback_invoked) == 1
```

### 3.3 Cursor + Gesture Integration

```python
# tests/integration/test_cursor_gesture_integration.py
import pytest
from unittest.mock import Mock, patch
import numpy as np
from src.motion.cursor_control import CursorController
from src.gesture.handler import GestureHandler, GestureType

class TestCursorGestureIntegration:

    @pytest.fixture
    def mock_mouse(self):
        with patch('pynput.mouse.Controller') as mock:
            mouse = Mock()
            mouse.position = (500, 500)
            mock.return_value = mouse
            yield mouse

    @pytest.fixture
    def cursor(self, mock_mouse):
        return CursorController()

    @pytest.fixture
    def gesture_handler(self, cursor, mock_mouse):
        return GestureHandler(cursor)

    def test_tap_clicks_at_current_position(self, gesture_handler, mock_mouse):
        """Tap gesture should click at cursor position."""
        mock_mouse.position = (200, 300)

        gesture_handler.handle_gesture(GestureType.TAP)

        mock_mouse.click.assert_called_once()

    def test_palm_up_disables_cursor(self, gesture_handler, cursor):
        """Palm up gesture should disable cursor control."""
        assert cursor.enabled

        gesture_handler.handle_gesture(GestureType.PALM_UP)

        assert not cursor.enabled

    def test_palm_down_enables_cursor(self, gesture_handler, cursor):
        """Palm down gesture should enable cursor control."""
        cursor.disable()

        gesture_handler.handle_gesture(GestureType.PALM_DOWN)

        assert cursor.enabled

    def test_hold_initiates_drag(self, gesture_handler, mock_mouse):
        """Hold gesture should start drag operation."""
        gesture_handler.handle_gesture(GestureType.HOLD_START)

        assert gesture_handler.is_dragging
        mock_mouse.press.assert_called_once()

        gesture_handler.handle_gesture(GestureType.HOLD_END)

        assert not gesture_handler.is_dragging
        mock_mouse.release.assert_called_once()
```

---

## 4. System Testing

### 4.1 End-to-End Test Scenarios

#### Scenario 1: Basic Cursor Control

```markdown
**Test ID:** E2E-001
**Title:** Basic Cursor Control with Watch
**Preconditions:**
- Watch app installed and running
- Desktop app running and connected
- User wearing watch in natural position

**Steps:**
1. Calibrate by holding neutral position and triggering calibrate
2. Tilt wrist right
3. Verify cursor moves right on screen
4. Tilt wrist left
5. Verify cursor moves left
6. Tilt wrist forward (away from body)
7. Verify cursor moves up
8. Tilt wrist backward (toward body)
9. Verify cursor moves down

**Expected Results:**
- Cursor moves smoothly in direction of tilt
- No jitter at rest position
- Response time < 100ms perceived
- Movement feels natural and proportional

**Pass Criteria:**
- All directional movements correct
- Latency feels responsive
- No observable jitter
```

#### Scenario 2: Gesture Clicking

```markdown
**Test ID:** E2E-002
**Title:** Gesture-Based Mouse Clicks
**Preconditions:**
- System connected and calibrated
- Test clickable elements on screen

**Steps:**
1. Position cursor over clickable button
2. Perform finger pinch gesture (tap)
3. Verify single click registered
4. Position cursor over icon
5. Perform quick double pinch (double tap)
6. Verify double click registered
7. Position cursor over context menu target
8. Perform hold gesture
9. Verify right click menu appears

**Expected Results:**
- Single tap = single left click
- Double tap within 500ms = double click
- Hold > 500ms = drag operation or right click

**Pass Criteria:**
- > 95% gesture recognition accuracy
- No false positives in 30-second rest period
```

#### Scenario 3: Voice Commands

```markdown
**Test ID:** E2E-003
**Title:** Voice Command Execution
**Preconditions:**
- System connected
- Quiet environment
- Microphone permissions granted

**Steps:**
1. Say "click"
2. Verify click at current cursor position
3. Say "scroll down"
4. Verify page scrolls down
5. Say "scroll up 3"
6. Verify page scrolls up 3 times
7. Say "type hello world"
8. Verify text is typed
9. Say "press enter"
10. Verify enter key pressed

**Expected Results:**
- Commands recognized within 500ms
- Correct action executed
- No erroneous actions from ambient noise

**Pass Criteria:**
- > 90% command recognition accuracy
- < 500ms end-to-end latency
- No false activations in 1-minute silence test
```

### 4.2 System Test Matrix

| Test Area | Windows | macOS | Linux | Priority |
|-----------|---------|-------|-------|----------|
| BLE Connection | P1 | P1 | P1 | Critical |
| Cursor Movement | P1 | P1 | P1 | Critical |
| Click Gestures | P1 | P1 | P1 | Critical |
| Voice Commands | P1 | P1 | P1 | Critical |
| System Tray | P1 | P1 | P1 | High |
| Settings UI | P2 | P2 | P2 | Medium |
| Auto-reconnect | P1 | P1 | P1 | High |
| Battery Reporting | P2 | P2 | P2 | Medium |

---

## 5. Performance Testing

### 5.1 Latency Benchmarks

```python
# tests/performance/test_latency.py
import pytest
import time
import statistics
from src.motion.pipeline import CursorPipeline
from src.voice.stt_engine import STTEngine

class TestLatencyBenchmarks:

    def test_cursor_pipeline_latency(self):
        """Cursor processing should complete in < 10ms."""
        pipeline = CursorPipeline()
        accel = np.array([0.0, 0.0, 9.8])
        gyro = np.array([0.0, 0.0, 0.0])

        latencies = []

        for i in range(1000):
            start = time.perf_counter()
            pipeline.process(accel, gyro, dt=0.01)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)

        avg_latency = statistics.mean(latencies)
        p99_latency = statistics.quantiles(latencies, n=100)[-1]

        assert avg_latency < 5.0, f"Average latency {avg_latency}ms exceeds 5ms"
        assert p99_latency < 10.0, f"P99 latency {p99_latency}ms exceeds 10ms"

        print(f"Cursor pipeline: avg={avg_latency:.2f}ms, p99={p99_latency:.2f}ms")

    def test_stt_latency(self):
        """STT processing should complete in < 300ms for 1s audio."""
        engine = STTEngine(model_size="tiny")

        # Generate 1 second of test audio
        audio = np.random.randn(16000).astype(np.float32) * 0.01

        latencies = []

        for _ in range(10):
            start = time.perf_counter()
            engine.transcribe(audio)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)

        avg_latency = statistics.mean(latencies)

        assert avg_latency < 300.0, f"STT latency {avg_latency}ms exceeds 300ms"

        print(f"STT latency (1s audio): avg={avg_latency:.2f}ms")

    def test_gesture_detection_latency(self):
        """Gesture detection should complete in < 1ms."""
        from src.gesture.detector import GestureDetector
        detector = GestureDetector()

        latencies = []

        for i in range(10000):
            accel = np.random.randn(3) * 0.1 + np.array([0, 0, 9.8])

            start = time.perf_counter()
            detector.process_accelerometer(*accel, timestamp=i * 10_000_000)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)

        avg_latency = statistics.mean(latencies)

        assert avg_latency < 0.5, f"Gesture latency {avg_latency}ms exceeds 0.5ms"

        print(f"Gesture detection: avg={avg_latency*1000:.2f}us")
```

### 5.2 Memory & CPU Benchmarks

```python
# tests/performance/test_resource_usage.py
import pytest
import tracemalloc
import psutil
import os
import time

class TestResourceUsage:

    def test_memory_stability(self):
        """Memory should not grow continuously during operation."""
        from src.motion.pipeline import CursorPipeline

        pipeline = CursorPipeline()
        accel = np.array([0.0, 0.0, 9.8])
        gyro = np.array([0.0, 0.0, 0.0])

        # Warmup
        for i in range(1000):
            pipeline.process(accel, gyro, dt=0.01)

        tracemalloc.start()
        initial_memory = tracemalloc.get_traced_memory()[0]

        # Run for simulated 10 minutes at 100Hz
        for i in range(60000):
            pipeline.process(accel, gyro, dt=0.01)

        final_memory = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()

        memory_growth = (final_memory - initial_memory) / 1024 / 1024  # MB

        assert memory_growth < 10.0, f"Memory grew by {memory_growth}MB"

        print(f"Memory growth over 10min simulation: {memory_growth:.2f}MB")

    def test_cpu_usage(self):
        """CPU usage should remain reasonable during operation."""
        from src.main import WristControlApp

        app = WristControlApp()
        process = psutil.Process(os.getpid())

        # Measure baseline
        baseline_cpu = process.cpu_percent(interval=1.0)

        # Simulate operation
        app.start_simulation_mode()

        time.sleep(5.0)
        active_cpu = process.cpu_percent(interval=1.0)

        app.stop()

        assert active_cpu < 25.0, f"CPU usage {active_cpu}% exceeds 25%"

        print(f"CPU usage: baseline={baseline_cpu}%, active={active_cpu}%")
```

### 5.3 Battery Drain Testing (Watch)

```kotlin
// test/kotlin/com/wristcontrol/performance/BatteryTest.kt
/**
 * Manual battery drain test procedure:
 *
 * 1. Charge watch to 100%
 * 2. Disable all other apps and notifications
 * 3. Start WristControl in active mode (100Hz sensors + BLE)
 * 4. Let run for 1 hour
 * 5. Record battery percentage
 * 6. Calculate drain rate
 *
 * Target: < 15% per hour in active mode
 *        < 5% per hour in idle monitoring mode
 */

class BatteryDrainTest {

    @Test
    @Ignore("Manual test - requires real device")
    fun measureBatteryDrain() {
        val batteryManager = context.getSystemService(BATTERY_SERVICE) as BatteryManager
        val initialLevel = batteryManager.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY)

        // Start sensor streaming
        sensorManager.startStreaming()
        gattServer.startAdvertising()

        // Wait 1 hour
        Thread.sleep(3600_000L)

        val finalLevel = batteryManager.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY)
        val drainPercent = initialLevel - finalLevel

        println("Battery drain: ${drainPercent}% per hour")

        assertTrue("Battery drain $drainPercent% exceeds 15% target") {
            drainPercent <= 15
        }
    }
}
```

---

## 6. User Acceptance Testing

### 6.1 UAT Test Cases

#### UAT-001: First-Time Setup

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Install desktop app | App installs without errors |
| 2 | Install watch app | App appears on watch |
| 3 | Grant permissions on watch | Permissions accepted |
| 4 | Open desktop app | App appears in system tray |
| 5 | Enable Bluetooth on desktop | BLE adapter detected |
| 6 | Wait for connection | Watch auto-discovered and connected within 30s |
| 7 | Follow calibration prompt | Calibration completes successfully |
| 8 | Test cursor movement | Cursor moves with wrist tilt |

#### UAT-002: Daily Usage Workflow

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Turn on computer | Desktop app starts automatically |
| 2 | Wake watch by raising wrist | Watch reconnects automatically |
| 3 | Calibrate (optional) | Neutral position set |
| 4 | Browse web using cursor | Smooth cursor control |
| 5 | Click links with tap gesture | Links clicked reliably |
| 6 | Scroll with voice command | Page scrolls as requested |
| 7 | Type with voice | Text typed correctly |
| 8 | Pause by resting hand | Cursor stops moving |

#### UAT-003: Accessibility User Test

| Participant | Condition | Task | Success Criteria |
|-------------|-----------|------|------------------|
| User A | Limited hand mobility | Navigate web page | Complete basic browsing |
| User B | Repetitive strain injury | Check email | Complete without pain |
| User C | Post-surgery arm | Watch video | Adjust volume, pause |

### 6.2 Usability Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Task completion rate | > 95% | % of tasks completed successfully |
| Time-on-task | < 2x mouse | Time vs. traditional mouse |
| Error rate | < 5% | Unintended actions / total actions |
| User satisfaction | > 4.0/5.0 | Post-session survey |
| Learning curve | < 5 min | Time to basic proficiency |

### 6.3 UAT Feedback Form

```markdown
## WristControl UAT Feedback Form

**Date:** _______________
**Participant ID:** _______________

### Task Completion (1-5 scale)
- [ ] Cursor control felt natural
- [ ] Gestures were recognized reliably
- [ ] Voice commands worked correctly
- [ ] Settings were easy to adjust

### Comfort (1-5 scale)
- [ ] Watch was comfortable to wear
- [ ] Arm fatigue was acceptable
- [ ] Movement range was appropriate

### Issues Encountered
_Describe any problems:_

### Suggestions for Improvement
_What would make this better:_

### Overall Satisfaction: ___/5
```

---

## 7. Automated Testing

### 7.1 CI/CD Pipeline

```yaml
# .github/workflows/test.yml
name: WristControl Tests

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  desktop-tests:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.10, 3.11, 3.12]

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt

      - name: Run unit tests
        run: |
          pytest tests/unit -v --cov=src --cov-report=xml

      - name: Run integration tests
        run: |
          pytest tests/integration -v

      - name: Upload coverage
        uses: codecov/codecov-action@v3

  watch-tests:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Set up JDK
        uses: actions/setup-java@v4
        with:
          java-version: '17'
          distribution: 'temurin'

      - name: Build and test
        working-directory: ./watch-app
        run: ./gradlew test

  lint:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Python lint
        run: |
          pip install ruff mypy
          ruff check src/
          mypy src/

      - name: Kotlin lint
        working-directory: ./watch-app
        run: ./gradlew ktlintCheck

  performance:
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'

    steps:
      - uses: actions/checkout@v4

      - name: Run performance tests
        run: |
          pip install -r requirements.txt
          pytest tests/performance -v --benchmark-json=benchmark.json

      - name: Store benchmark result
        uses: benchmark-action/github-action-benchmark@v1
        with:
          tool: 'pytest'
          output-file-path: benchmark.json
```

### 7.2 Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-json

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.9
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]

  - repo: local
    hooks:
      - id: pytest-unit
        name: pytest unit tests
        entry: pytest tests/unit -q
        language: system
        pass_filenames: false
        always_run: true
```

### 7.3 Test Coverage Requirements

| Component | Minimum Coverage | Critical Paths |
|-----------|-----------------|----------------|
| Sensor Fusion | 90% | Filter algorithms |
| Gesture Detection | 95% | All gesture types |
| Command Parser | 100% | All command patterns |
| BLE Protocol | 85% | Data packing/unpacking |
| Input Controller | 80% | Click, type, scroll |
| Configuration | 90% | Load/save/validate |

---

## 8. Testing Tools & Infrastructure

### 8.1 Required Tools

| Tool | Purpose | Version |
|------|---------|---------|
| pytest | Python unit testing | 7.4+ |
| pytest-asyncio | Async test support | 0.23+ |
| pytest-cov | Coverage reporting | 4.1+ |
| JUnit 5 | Kotlin testing | 5.10+ |
| MockK | Kotlin mocking | 1.13+ |
| Android Emulator | Watch simulation | 33.1+ |
| BlueZ | Linux BLE testing | 5.66+ |

### 8.2 Test Fixtures

```python
# conftest.py
import pytest
import numpy as np

@pytest.fixture
def sample_imu_data():
    """Generate realistic IMU data samples."""
    return {
        'accel_neutral': np.array([0.0, 0.0, -9.8]),
        'accel_tilt_right': np.array([4.9, 0.0, -8.5]),
        'accel_tilt_forward': np.array([0.0, 4.9, -8.5]),
        'gyro_still': np.array([0.0, 0.0, 0.0]),
        'gyro_rotating': np.array([0.0, 1.0, 0.0]),
    }

@pytest.fixture
def sample_audio_commands():
    """Pre-recorded audio samples for voice testing."""
    return {
        'click': np.load('tests/fixtures/audio_click.npy'),
        'scroll_down': np.load('tests/fixtures/audio_scroll_down.npy'),
        'type_hello': np.load('tests/fixtures/audio_type_hello.npy'),
    }

@pytest.fixture
def mock_ble_client():
    """Mock BLE client for testing without hardware."""
    from unittest.mock import AsyncMock
    client = AsyncMock()
    client.is_connected = True
    client.mtu_size = 247
    return client
```

### 8.3 Hardware Test Lab Setup

```markdown
## Test Lab Requirements

### Devices
- Samsung Galaxy Watch4 (WearOS 3)
- Samsung Galaxy Watch5 (WearOS 4)
- Samsung Galaxy Watch6 (WearOS 5)

### Desktop Systems
- Windows 11 PC with Bluetooth 5.0
- macOS Ventura MacBook with M1/M2
- Ubuntu 22.04 with Intel AX200 BLE

### Network
- Isolated WiFi network
- RF-shielded test chamber (optional)

### Monitoring
- Bluetooth sniffer (Ubertooth One)
- Power meter for battery testing
- High-speed camera for latency verification
```

---

## 9. Test Data & Fixtures

### 9.1 Recorded Sensor Data

```python
# tests/fixtures/generate_fixtures.py
"""
Script to generate test fixtures from real device recordings.
Run on device with actual sensors to capture realistic data.
"""

import numpy as np
import json

def record_gesture(gesture_name: str, duration_seconds: float = 2.0):
    """Record gesture for test fixture."""
    samples = []
    # ... recording logic ...
    np.save(f'fixtures/gesture_{gesture_name}.npy', np.array(samples))

def record_voice_command(command_name: str):
    """Record voice command for test fixture."""
    # ... audio recording logic ...
    np.save(f'fixtures/audio_{command_name}.npy', audio_data)

# Generate standard test fixtures
GESTURES = ['tap', 'double_tap', 'hold', 'flick_left', 'flick_right']
COMMANDS = ['click', 'scroll_down', 'scroll_up', 'type_hello', 'press_enter']
```

### 9.2 Synthetic Test Data

```python
# tests/fixtures/synthetic.py

def generate_sine_wave_motion(frequency: float = 0.5, amplitude: float = 30.0,
                              duration: float = 5.0, sample_rate: float = 100.0):
    """Generate synthetic oscillating motion data."""
    samples = int(duration * sample_rate)
    t = np.linspace(0, duration, samples)

    roll = amplitude * np.sin(2 * np.pi * frequency * t)
    pitch = amplitude * np.sin(2 * np.pi * frequency * t + np.pi/4)

    # Convert to accelerometer readings
    accel_x = 9.8 * np.sin(np.radians(roll))
    accel_y = 9.8 * np.sin(np.radians(pitch))
    accel_z = -9.8 * np.cos(np.radians(roll)) * np.cos(np.radians(pitch))

    return np.column_stack([accel_x, accel_y, accel_z])

def generate_tap_spike(spike_magnitude: float = 15.0, baseline: float = 9.8):
    """Generate synthetic tap gesture data."""
    samples = 50  # 500ms at 100Hz

    accel = np.zeros((samples, 3))
    accel[:, 2] = -baseline  # Gravity

    # Add spike at center
    accel[20:25, 2] += -spike_magnitude  # Downward spike

    return accel
```

---

## 10. Quality Metrics

### 10.1 Test Quality Dashboard

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Unit Test Coverage | > 80% | - | - |
| Integration Test Coverage | > 60% | - | - |
| Test Pass Rate | 100% | - | - |
| Flaky Test Rate | < 1% | - | - |
| Test Execution Time | < 5 min | - | - |

### 10.2 Bug Tracking

| Severity | Response Time | Fix Time | Examples |
|----------|---------------|----------|----------|
| Critical | < 4 hours | < 24 hours | Crash, data loss, security |
| High | < 24 hours | < 1 week | Feature broken, major UX issue |
| Medium | < 1 week | < 2 weeks | Minor feature issue |
| Low | < 2 weeks | As time permits | Cosmetic, edge cases |

### 10.3 Release Criteria

```markdown
## Release Checklist

### Code Quality
- [ ] All unit tests passing
- [ ] All integration tests passing
- [ ] Code coverage >= 80%
- [ ] No critical or high bugs open
- [ ] Security scan passed

### Performance
- [ ] Cursor latency < 50ms (measured)
- [ ] Voice latency < 500ms (measured)
- [ ] Memory stable over 1 hour
- [ ] CPU usage < 15% idle

### Compatibility
- [ ] Tested on Windows 10/11
- [ ] Tested on macOS Ventura+
- [ ] Tested on Ubuntu 22.04+
- [ ] Tested on Galaxy Watch4/5/6

### Documentation
- [ ] README updated
- [ ] CHANGELOG updated
- [ ] User guide reviewed
```

---

*Testing strategy document for WristControl project - January 2026*
