# Reference Implementations for WristControl Desktop App

## Quick Start Examples

### 1. Minimal Working Example (5 minutes)

```python
"""
minimal_demo.py - Simplest possible cursor control demo
Tests input injection without any sensor data
"""
from pynput.mouse import Controller
import time
import math

mouse = Controller()

print("Moving cursor in a circle...")
print("Press Ctrl+C to stop")

try:
    radius = 200
    center_x, center_y = 500, 500
    angle = 0

    while True:
        # Calculate position
        x = center_x + int(radius * math.cos(angle))
        y = center_y + int(radius * math.sin(angle))

        # Move mouse
        mouse.position = (x, y)

        # Increment angle
        angle += 0.1

        # Control speed (100 Hz)
        time.sleep(0.01)

except KeyboardInterrupt:
    print("\nDemo stopped")
```

---

### 2. Simulated Sensor Data Demo

```python
"""
simulated_sensor_demo.py - Test cursor control with simulated gyroscope data
Simulates what the watch would send
"""
from pynput.mouse import Controller
import time
import math
import random

class SimulatedWatch:
    """Simulates sensor data from a smartwatch"""

    def __init__(self):
        self.time = 0

    def get_sensor_data(self):
        """Simulate gyroscope data (degrees/second)"""
        # Simulate slow circular movement
        gyro_x = 10 * math.sin(self.time * 0.5)
        gyro_y = 10 * math.cos(self.time * 0.5)
        gyro_z = 0

        # Add some noise
        gyro_x += random.gauss(0, 0.5)
        gyro_y += random.gauss(0, 0.5)

        self.time += 0.02  # 50 Hz

        return {
            'gyro_x': gyro_x,
            'gyro_y': gyro_y,
            'gyro_z': gyro_z,
            'timestamp': time.time()
        }

class SimpleCursorController:
    """Convert gyroscope data to cursor movement"""

    def __init__(self):
        self.mouse = Controller()
        self.sensitivity = 2.0

    def update(self, sensor_data):
        """Update cursor position based on sensor data"""
        # Convert gyroscope angular velocity to cursor velocity
        # gyro_x = rotation around X axis (pitch) -> affects Y cursor movement
        # gyro_y = rotation around Y axis (yaw) -> affects X cursor movement

        dx = sensor_data['gyro_y'] * self.sensitivity
        dy = -sensor_data['gyro_x'] * self.sensitivity  # Inverted for natural feel

        # Move cursor
        self.mouse.move(int(dx), int(dy))

# Main loop
def main():
    watch = SimulatedWatch()
    controller = SimpleCursorController()

    print("Simulated sensor demo running...")
    print("Cursor will move in a slow circle")
    print("Press Ctrl+C to stop")

    try:
        while True:
            # Get sensor data (50 Hz)
            data = watch.get_sensor_data()

            # Update cursor
            controller.update(data)

            # Maintain 50 Hz update rate
            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\nDemo stopped")

if __name__ == '__main__':
    main()
```

---

### 3. Gesture Click Demo

```python
"""
gesture_demo.py - Demonstrate gesture-based clicking
Simulates finger tap detection from accelerometer spikes
"""
from pynput.mouse import Controller, Button
import time
import random

class GestureDetector:
    """Detect gestures from accelerometer data"""

    def __init__(self):
        self.tap_threshold = 2.0  # G-force threshold for tap
        self.last_tap_time = 0
        self.double_tap_window = 0.3  # seconds

    def process_accel_data(self, accel_magnitude):
        """Process accelerometer magnitude to detect taps"""
        current_time = time.time()
        gesture = None

        # Detect tap (spike in acceleration)
        if accel_magnitude > self.tap_threshold:
            time_since_last_tap = current_time - self.last_tap_time

            if time_since_last_tap < self.double_tap_window:
                gesture = 'double_tap'
            else:
                gesture = 'tap'

            self.last_tap_time = current_time

        return gesture

class InteractiveCursorDemo:
    """Combine cursor movement with gesture clicks"""

    def __init__(self):
        self.mouse = Controller()
        self.gesture_detector = GestureDetector()

    def handle_gesture(self, gesture):
        """Execute action for detected gesture"""
        if gesture == 'tap':
            print("  -> Left click")
            self.mouse.click(Button.left, 1)
        elif gesture == 'double_tap':
            print("  -> Double click")
            self.mouse.click(Button.left, 2)

def simulate_tap_sequence():
    """Simulate a sequence of taps"""
    demo = InteractiveCursorDemo()

    print("Gesture demo - Simulating taps...")
    print("Watch the mouse cursor click!")

    # Simulate some taps
    tap_times = [1.0, 3.0, 3.2, 5.0, 7.0]  # Single, double, single, single

    start_time = time.time()
    last_accel = 1.0  # Normal gravity

    while time.time() - start_time < 8.0:
        elapsed = time.time() - start_time

        # Simulate tap at specific times
        if any(abs(elapsed - t) < 0.05 for t in tap_times):
            accel = 3.5  # Spike
            print(f"[{elapsed:.2f}s] TAP detected")
        else:
            accel = 1.0 + random.gauss(0, 0.1)  # Normal + noise

        # Process gesture
        gesture = demo.gesture_detector.process_accel_data(accel)
        if gesture:
            demo.handle_gesture(gesture)

        time.sleep(0.02)  # 50 Hz

    print("Demo complete")

if __name__ == '__main__':
    simulate_tap_sequence()
```

---

### 4. System Tray Minimal Example

```python
"""
tray_minimal.py - Minimal system tray application
Foundation for WristControl tray app
"""
import pystray
from pystray import MenuItem as item
from PIL import Image, ImageDraw
import threading
import time

class MinimalTrayApp:
    def __init__(self):
        self.running = False
        self.icon = None

    def create_icon(self):
        """Create a simple colored circle icon"""
        # Create 64x64 image
        img = Image.new('RGB', (64, 64), color='white')
        draw = ImageDraw.Draw(img)

        # Draw colored circle
        color = 'green' if self.running else 'red'
        draw.ellipse([8, 8, 56, 56], fill=color, outline='black')

        return img

    def toggle(self, icon, item):
        """Toggle running state"""
        self.running = not self.running
        print(f"State: {'Running' if self.running else 'Stopped'}")

        # Update icon
        icon.icon = self.create_icon()

        # Show notification
        icon.notify(
            f"WristControl {'enabled' if self.running else 'disabled'}",
            "WristControl"
        )

    def quit(self, icon, item):
        """Quit application"""
        print("Quitting...")
        icon.stop()

    def background_task(self):
        """Simulate background processing"""
        while True:
            if self.running:
                print(f"Processing... (running={self.running})")
            time.sleep(2)

    def run(self):
        """Run the tray application"""
        # Start background thread
        bg_thread = threading.Thread(target=self.background_task, daemon=True)
        bg_thread.start()

        # Create menu
        menu = pystray.Menu(
            item(
                'Toggle',
                self.toggle,
                checked=lambda item: self.running
            ),
            pystray.Menu.SEPARATOR,
            item('Quit', self.quit)
        )

        # Create and run icon
        self.icon = pystray.Icon(
            'wristcontrol',
            self.create_icon(),
            'WristControl',
            menu
        )

        print("Tray app started - check system tray")
        self.icon.run()

if __name__ == '__main__':
    app = MinimalTrayApp()
    app.run()
```

---

### 5. Complete Prototype (Production-Ready Structure)

```python
"""
wristcontrol_prototype.py - Complete working prototype
Demonstrates all core functionality in one file
"""
import time
import threading
import queue
import math
from dataclasses import dataclass
from typing import Optional, Callable
from pynput.mouse import Controller as MouseController, Button
from pynput.keyboard import Controller as KeyboardController

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class SensorData:
    """Sensor data from watch"""
    timestamp: float
    accel_x: float
    accel_y: float
    accel_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float

    @property
    def accel_magnitude(self):
        """Total acceleration magnitude"""
        return math.sqrt(self.accel_x**2 + self.accel_y**2 + self.accel_z**2)

@dataclass
class Config:
    """Application configuration"""
    sensitivity: float = 2.0
    dead_zone: float = 0.5
    smoothing: float = 0.3
    tap_threshold: float = 2.0

# ============================================================================
# INPUT CONTROL
# ============================================================================

class InputController:
    """Handles OS input injection"""

    def __init__(self):
        self.mouse = MouseController()
        self.keyboard = KeyboardController()

    def move_cursor(self, dx: int, dy: int):
        """Move cursor relative to current position"""
        self.mouse.move(dx, dy)

    def click(self, button='left', count=1):
        """Perform mouse click"""
        btn = Button.left if button == 'left' else Button.right
        self.mouse.click(btn, count)

    def scroll(self, dy: int):
        """Scroll vertically"""
        self.mouse.scroll(0, dy)

# ============================================================================
# SENSOR PROCESSING
# ============================================================================

class MotionProcessor:
    """Process sensor data into cursor movements"""

    def __init__(self, config: Config):
        self.config = config
        self.smoothed_dx = 0.0
        self.smoothed_dy = 0.0

    def process(self, sensor_data: SensorData) -> tuple:
        """Convert sensor data to cursor delta"""
        # Use gyroscope for cursor control
        raw_dx = sensor_data.gyro_y * self.config.sensitivity
        raw_dy = -sensor_data.gyro_x * self.config.sensitivity

        # Apply dead zone
        if abs(raw_dx) < self.config.dead_zone:
            raw_dx = 0
        if abs(raw_dy) < self.config.dead_zone:
            raw_dy = 0

        # Apply exponential smoothing
        alpha = self.config.smoothing
        self.smoothed_dx = alpha * self.smoothed_dx + (1 - alpha) * raw_dx
        self.smoothed_dy = alpha * self.smoothed_dy + (1 - alpha) * raw_dy

        return int(self.smoothed_dx), int(self.smoothed_dy)

class GestureProcessor:
    """Detect gestures from sensor data"""

    def __init__(self, config: Config):
        self.config = config
        self.last_tap_time = 0
        self.double_tap_window = 0.3

    def process(self, sensor_data: SensorData) -> Optional[str]:
        """Detect gestures from sensor data"""
        # Detect taps from acceleration spikes
        if sensor_data.accel_magnitude > self.config.tap_threshold:
            current_time = time.time()
            time_since_last = current_time - self.last_tap_time

            if time_since_last < self.double_tap_window:
                gesture = 'double_tap'
            else:
                gesture = 'tap'

            self.last_tap_time = current_time
            return gesture

        return None

# ============================================================================
# SIMULATED DATA SOURCE
# ============================================================================

class SimulatedWatch:
    """Simulates sensor data from a watch"""

    def __init__(self):
        self.time = 0
        self.tap_schedule = [2.0, 4.0, 4.2, 6.0]  # Times to simulate taps

    def read_sensors(self) -> SensorData:
        """Generate simulated sensor data"""
        # Smooth circular motion
        gyro_x = 8 * math.sin(self.time * 0.3)
        gyro_y = 8 * math.cos(self.time * 0.3)

        # Check if we should simulate a tap
        accel_magnitude = 1.0  # Normal gravity
        for tap_time in self.tap_schedule:
            if abs(self.time - tap_time) < 0.05:
                accel_magnitude = 3.0  # Tap!

        # Accelerometer (simplified)
        accel_x = 0
        accel_y = 0
        accel_z = accel_magnitude

        self.time += 0.02  # 50 Hz

        return SensorData(
            timestamp=time.time(),
            accel_x=accel_x,
            accel_y=accel_y,
            accel_z=accel_z,
            gyro_x=gyro_x,
            gyro_y=gyro_y,
            gyro_z=0
        )

# ============================================================================
# MAIN APPLICATION
# ============================================================================

class WristControlPrototype:
    """Main application"""

    def __init__(self):
        self.config = Config()
        self.input_controller = InputController()
        self.motion_processor = MotionProcessor(self.config)
        self.gesture_processor = GestureProcessor(self.config)

        # Data source (replace with BLE in production)
        self.watch = SimulatedWatch()

        # State
        self.enabled = False
        self.running = False

        # Statistics
        self.frame_count = 0
        self.last_stats_time = time.time()

    def process_frame(self):
        """Process one frame of sensor data"""
        # Read sensors
        sensor_data = self.watch.read_sensors()

        if self.enabled:
            # Process motion
            dx, dy = self.motion_processor.process(sensor_data)
            if dx != 0 or dy != 0:
                self.input_controller.move_cursor(dx, dy)

            # Process gestures
            gesture = self.gesture_processor.process(sensor_data)
            if gesture:
                self.handle_gesture(gesture)

        # Statistics
        self.frame_count += 1
        if time.time() - self.last_stats_time >= 1.0:
            print(f"FPS: {self.frame_count}, Enabled: {self.enabled}")
            self.frame_count = 0
            self.last_stats_time = time.time()

    def handle_gesture(self, gesture: str):
        """Handle detected gesture"""
        print(f"  Gesture: {gesture}")

        if gesture == 'tap':
            self.input_controller.click('left', 1)
        elif gesture == 'double_tap':
            self.input_controller.click('left', 2)

    def run(self):
        """Run main loop"""
        self.running = True
        self.enabled = True

        print("WristControl Prototype Running")
        print("Cursor will move in a circle and click periodically")
        print("Press Ctrl+C to stop")
        print()

        try:
            while self.running:
                self.process_frame()
                time.sleep(0.02)  # 50 Hz

        except KeyboardInterrupt:
            print("\nStopping...")
            self.running = False

# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    app = WristControlPrototype()
    app.run()

if __name__ == '__main__':
    main()
```

---

### 6. Performance Test Suite

```python
"""
performance_test.py - Measure input injection latency
Critical for meeting <50ms latency requirement
"""
import time
import statistics
from pynput.mouse import Controller

def test_input_latency(iterations=1000):
    """Measure cursor movement latency"""
    mouse = Controller()
    latencies = []

    print(f"Testing input latency ({iterations} iterations)...")

    for i in range(iterations):
        start = time.perf_counter()

        # Perform input operation
        mouse.move(1, 0)

        end = time.perf_counter()
        latency_ms = (end - start) * 1000
        latencies.append(latency_ms)

        # Small delay between tests
        time.sleep(0.001)

    # Statistics
    print("\nResults:")
    print(f"  Mean latency: {statistics.mean(latencies):.3f} ms")
    print(f"  Median latency: {statistics.median(latencies):.3f} ms")
    print(f"  Min latency: {min(latencies):.3f} ms")
    print(f"  Max latency: {max(latencies):.3f} ms")
    print(f"  Std deviation: {statistics.stdev(latencies):.3f} ms")

    # Check if meets requirement
    avg_latency = statistics.mean(latencies)
    if avg_latency < 50:
        print(f"\n✓ PASS: Average latency {avg_latency:.2f}ms < 50ms target")
    else:
        print(f"\n✗ FAIL: Average latency {avg_latency:.2f}ms > 50ms target")

    return latencies

def test_throughput(duration=10):
    """Test maximum update rate"""
    mouse = Controller()
    count = 0
    start = time.time()

    print(f"\nTesting throughput ({duration} seconds)...")

    while time.time() - start < duration:
        mouse.move(1, 0)
        count += 1

    elapsed = time.time() - start
    fps = count / elapsed

    print(f"\nResults:")
    print(f"  Updates: {count}")
    print(f"  Duration: {elapsed:.2f} s")
    print(f"  Throughput: {fps:.1f} updates/sec")

    if fps >= 50:
        print(f"✓ PASS: Throughput {fps:.1f} >= 50 Hz target")
    else:
        print(f"✗ FAIL: Throughput {fps:.1f} < 50 Hz target")

    return fps

if __name__ == '__main__':
    # Run tests
    test_input_latency(1000)
    test_throughput(10)
```

---

### 7. Platform-Specific Permission Checker

```python
"""
permission_check.py - Check and request platform permissions
Run this before main application
"""
import platform
import sys
import os

def check_windows():
    """Check Windows permissions"""
    print("Platform: Windows")
    print("  No special permissions required")
    print("  ✓ Ready to run")
    return True

def check_macos():
    """Check macOS Accessibility permissions"""
    print("Platform: macOS")

    try:
        import Quartz

        if Quartz.AXIsProcessTrusted():
            print("  ✓ Accessibility permissions granted")
            return True
        else:
            print("  ✗ Accessibility permissions required")
            print("\nTo grant permissions:")
            print("  1. Open System Preferences")
            print("  2. Go to Security & Privacy > Privacy > Accessibility")
            print("  3. Add Terminal (or your Python app)")
            print("  4. Check the box to enable")

            # Try to prompt
            options = {Quartz.kAXTrustedCheckOptionPrompt: True}
            Quartz.AXIsProcessTrustedWithOptions(options)

            return False

    except ImportError:
        print("  ✗ pyobjc not installed")
        print("  Install with: pip install pyobjc-framework-Quartz")
        return False

def check_linux():
    """Check Linux permissions"""
    print("Platform: Linux")

    # Check display server
    session_type = os.environ.get('XDG_SESSION_TYPE', 'unknown')
    print(f"  Display server: {session_type}")

    if session_type == 'x11':
        if os.environ.get('DISPLAY'):
            print("  ✓ X11 display accessible")
            return True
        else:
            print("  ✗ DISPLAY not set")
            return False

    elif session_type == 'wayland':
        print("  Note: Wayland detected")
        print("  Some features may be limited")

        # Check uinput access
        if os.path.exists('/dev/uinput'):
            if os.access('/dev/uinput', os.W_OK):
                print("  ✓ uinput accessible")
                return True
            else:
                print("  ✗ uinput not writable")
                print("\nTo fix:")
                print("  sudo usermod -a -G input $USER")
                print("  Then log out and log back in")
                return False
        else:
            print("  ✗ uinput not available")
            print("  sudo modprobe uinput")
            return False

    return False

def main():
    """Check permissions for current platform"""
    print("WristControl Permission Checker")
    print("=" * 50)
    print()

    system = platform.system()

    if system == 'Windows':
        result = check_windows()
    elif system == 'Darwin':
        result = check_macos()
    elif system == 'Linux':
        result = check_linux()
    else:
        print(f"Unknown platform: {system}")
        result = False

    print()
    if result:
        print("✓ All checks passed - ready to run WristControl")
        sys.exit(0)
    else:
        print("✗ Permission checks failed - please fix issues above")
        sys.exit(1)

if __name__ == '__main__':
    main()
```

---

## Usage Instructions

### Quick Test Sequence

1. **Test basic input injection:**
   ```bash
   python minimal_demo.py
   ```

2. **Test with simulated sensors:**
   ```bash
   python simulated_sensor_demo.py
   ```

3. **Test gestures:**
   ```bash
   python gesture_demo.py
   ```

4. **Test system tray:**
   ```bash
   python tray_minimal.py
   ```

5. **Run complete prototype:**
   ```bash
   python wristcontrol_prototype.py
   ```

6. **Check performance:**
   ```bash
   python performance_test.py
   ```

7. **Verify permissions:**
   ```bash
   python permission_check.py
   ```

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install pynput pillow pystray

# Platform-specific
# macOS:
pip install pyobjc-framework-Quartz

# Linux:
pip install python-xlib  # For X11

# For PyQt UI (optional):
pip install PyQt6
```

### Expected Results

- **Minimal demo**: Cursor moves in a circle
- **Sensor demo**: Cursor moves in smooth circular pattern
- **Gesture demo**: Mouse clicks automatically at timed intervals
- **Tray demo**: Icon appears in system tray with toggle option
- **Prototype**: Complete demo with cursor movement and clicking
- **Performance test**: Should show <10ms average latency on modern systems

### Troubleshooting

**macOS - "Not authorized to send input":**
- Run `permission_check.py`
- Grant Accessibility permissions in System Preferences
- Restart terminal/IDE

**Linux - "Permission denied on /dev/uinput":**
- Add user to input group: `sudo usermod -a -G input $USER`
- Log out and log back in
- Or run with sudo (not recommended for production)

**Windows - High latency:**
- Close resource-intensive applications
- Run `enable_high_precision_timer()` from Windows examples
- Check Windows Defender isn't scanning Python

