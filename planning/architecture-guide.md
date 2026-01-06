# WristControl Desktop Application Architecture Guide

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Samsung Galaxy Watch                          │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐      │
│  │  Sensors     │  │  Microphone  │  │  Gesture        │      │
│  │  (IMU)       │  │              │  │  Detection      │      │
│  └──────┬───────┘  └──────┬───────┘  └────────┬────────┘      │
│         │                  │                   │                │
│         └──────────────────┴───────────────────┘                │
│                            │                                    │
│                   ┌────────▼────────┐                          │
│                   │  BLE Service    │                          │
│                   │  (GATT Server)  │                          │
│                   └────────┬────────┘                          │
└────────────────────────────┼─────────────────────────────────┘
                             │ Bluetooth LE
                             │
┌────────────────────────────▼─────────────────────────────────┐
│              Desktop Companion Application                    │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              BLE Connection Layer                     │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐     │   │
│  │  │  Scanner   │  │  Client    │  │ Reconnect  │     │   │
│  │  │            │  │  (bleak)   │  │  Logic     │     │   │
│  │  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘     │   │
│  └────────┼───────────────┼───────────────┼────────────┘   │
│           │               │               │                 │
│  ┌────────▼───────────────▼───────────────▼────────────┐   │
│  │         Data Processing Pipeline                     │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌───────────┐ │   │
│  │  │   Sensor     │  │   Gesture    │  │   Voice   │ │   │
│  │  │  Processing  │  │  Processing  │  │Processing │ │   │
│  │  │              │  │              │  │   (STT)   │ │   │
│  │  └──────┬───────┘  └──────┬───────┘  └─────┬─────┘ │   │
│  └─────────┼──────────────────┼────────────────┼───────┘   │
│            │                  │                │            │
│  ┌─────────▼──────────────────▼────────────────▼───────┐   │
│  │           Command Execution Layer                   │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────┐ │   │
│  │  │   Cursor     │  │    Click     │  │Keyboard  │ │   │
│  │  │   Control    │  │   Executor   │  │  Input   │ │   │
│  │  │   (pynput)   │  │   (pynput)   │  │ (pynput) │ │   │
│  │  └──────┬───────┘  └──────┬───────┘  └────┬─────┘ │   │
│  └─────────┼──────────────────┼───────────────┼───────┘   │
│            │                  │               │            │
│  ┌─────────▼──────────────────▼───────────────▼───────┐   │
│  │          Configuration & State Management          │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐         │   │
│  │  │ Config   │  │ Profiles │  │  State   │         │   │
│  │  │ Manager  │  │ Manager  │  │ Machine  │         │   │
│  │  └──────────┘  └──────────┘  └──────────┘         │   │
│  └──────────────────────────────────────────────────────┘   │
│            │                                                │
│  ┌─────────▼────────────────────────────────────────────┐  │
│  │              User Interface Layer                     │  │
│  │  ┌────────────────┐       ┌──────────────────┐      │  │
│  │  │  System Tray   │       │ Settings Window  │      │  │
│  │  │   (pystray)    │◄─────►│   (PySide6)      │      │  │
│  │  └────────────────┘       └──────────────────┘      │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │  Operating System    │
                  │  Input Subsystem     │
                  └──────────────────────┘
```

---

## Component Architecture

### 1. Multi-threaded Event Processing

```
Main Thread                UI Thread              Sensor Thread         Gesture Thread
    │                          │                       │                     │
    │  Start App               │                       │                     │
    ├──────────────────────────►                      │                     │
    │                          │                       │                     │
    │  Create UI               │                       │                     │
    ├──────────────────────────►                      │                     │
    │                          │                       │                     │
    │  Start Sensor Processing │                       │                     │
    ├──────────────────────────┼──────────────────────►                     │
    │                          │                       │                     │
    │  Start Gesture Processing│                       │                     │
    ├──────────────────────────┼───────────────────────┼────────────────────►
    │                          │                       │                     │
    │                          │    Sensor Data        │                     │
    │                          │   ◄───────────────────┤                     │
    │                          │                       │                     │
    │                          │   Process → Move      │                     │
    │                          │        Cursor         │                     │
    │                          │                       │                     │
    │                          │                       │   Gesture Event     │
    │                          │   ◄───────────────────┼─────────────────────┤
    │                          │                       │                     │
    │                          │   Execute Click       │                     │
    │                          │                       │                     │
    │  User Changes Settings   │                       │                     │
    │◄─────────────────────────┤                       │                     │
    │                          │                       │                     │
    │  Update Config           │                       │                     │
    ├──────────────────────────┼──────────────────────►                     │
    │                          │                       │   Apply New Config  │
    │                          │                       ├─────────────────────►
    │                          │                       │                     │
```

### Thread Responsibilities

**Main Thread:**
- Application lifecycle management
- Configuration loading/saving
- BLE connection management
- Thread coordination

**UI Thread:**
- System tray icon updates
- Settings window rendering
- User input handling
- Status notifications

**Sensor Processing Thread (High Priority):**
- Receive sensor data from BLE
- Apply motion algorithm
- Inject cursor movements
- Target: <5ms latency per frame

**Gesture Processing Thread:**
- Receive gesture events
- Execute click/scroll actions
- Apply gesture mappings
- Lower priority than sensor thread

---

## Data Flow Diagrams

### Cursor Movement Data Flow

```
┌─────────────┐
│Watch Sensors│
│  50-100 Hz  │
└──────┬──────┘
       │ IMU Data (accel, gyro, mag)
       ▼
┌─────────────┐
│ BLE Stream  │
│   ~50ms     │
└──────┬──────┘
       │ Raw sensor packets
       ▼
┌─────────────┐
│Parse & Queue│
│   <1ms      │
└──────┬──────┘
       │ SensorData object
       ▼
┌─────────────┐
│  Sensor     │
│  Fusion     │◄────── Config (sensitivity, etc.)
│   ~2ms      │
└──────┬──────┘
       │ Fused orientation
       ▼
┌─────────────┐
│Motion       │
│Algorithm    │◄────── Dead zone, smoothing
│   ~2ms      │
└──────┬──────┘
       │ dx, dy (cursor delta)
       ▼
┌─────────────┐
│ Smoothing   │
│ Filter      │
│   ~1ms      │
└──────┬──────┘
       │ Smoothed dx, dy
       ▼
┌─────────────┐
│  pynput     │
│ move(dx,dy) │
│   ~5ms      │
└──────┬──────┘
       │ OS API calls
       ▼
┌─────────────┐
│   Cursor    │
│  Position   │
└─────────────┘

Total Latency: ~65ms (well within budget)
```

### Gesture Click Data Flow

```
┌──────────────┐
│ Finger Pinch │
│  (Physical)  │
└──────┬───────┘
       │ Accelerometer spike
       ▼
┌──────────────┐
│  On-Watch    │
│  Detection   │
│    ~10ms     │
└──────┬───────┘
       │ Gesture event
       ▼
┌──────────────┐
│ BLE Transmit │
│    ~20ms     │
└──────┬───────┘
       │ Event packet
       ▼
┌──────────────┐
│ Gesture Queue│
│    <1ms      │
└──────┬───────┘
       │ GestureEvent object
       ▼
┌──────────────┐
│  Gesture     │
│  Processor   │◄────── Gesture mappings
│    ~2ms      │
└──────┬───────┘
       │ Click command
       ▼
┌──────────────┐
│   pynput     │
│  click()     │
│    ~5ms      │
└──────┬───────┘
       │ OS API call
       ▼
┌──────────────┐
│ Mouse Click  │
└──────────────┘

Total Latency: ~38ms (excellent)
```

### Configuration Flow

```
User Adjusts Slider
       │
       ▼
┌──────────────┐
│ Qt Signal    │
│ Emitted      │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Config       │
│ Updated      │
└──────┬───────┘
       │
       ├──────────────┬──────────────┐
       ▼              ▼              ▼
┌─────────┐   ┌──────────┐   ┌──────────┐
│ Save to │   │  Update  │   │  Apply   │
│  JSON   │   │ Live UI  │   │To Engine │
└─────────┘   └──────────┘   └──────────┘
```

---

## State Machine

### Application States

```
┌──────────────┐
│ INITIALIZED  │
└──────┬───────┘
       │ start()
       ▼
┌──────────────┐
│  SCANNING    │◄──────────┐
└──────┬───────┘           │ connection lost
       │ device found      │
       ▼                   │
┌──────────────┐           │
│ CONNECTING   │           │
└──────┬───────┘           │
       │ connected         │
       ▼                   │
┌──────────────┐           │
│  CONNECTED   │───────────┘
└──────┬───────┘
       │ enable()
       ▼
┌──────────────┐
│   ACTIVE     │◄──────┐
└──────┬───────┘       │ enable()
       │ disable()     │
       ▼               │
┌──────────────┐       │
│    PAUSED    │───────┘
└──────┬───────┘
       │ disconnect()
       ▼
┌──────────────┐
│DISCONNECTED  │
└──────────────┘
```

### State Transitions

```python
from enum import Enum

class AppState(Enum):
    INITIALIZED = "initialized"
    SCANNING = "scanning"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ACTIVE = "active"
    PAUSED = "paused"
    DISCONNECTED = "disconnected"

class StateMachine:
    def __init__(self):
        self.state = AppState.INITIALIZED
        self.transitions = {
            AppState.INITIALIZED: [AppState.SCANNING],
            AppState.SCANNING: [AppState.CONNECTING, AppState.DISCONNECTED],
            AppState.CONNECTING: [AppState.CONNECTED, AppState.SCANNING],
            AppState.CONNECTED: [AppState.ACTIVE, AppState.DISCONNECTED],
            AppState.ACTIVE: [AppState.PAUSED, AppState.DISCONNECTED],
            AppState.PAUSED: [AppState.ACTIVE, AppState.DISCONNECTED],
            AppState.DISCONNECTED: [AppState.SCANNING],
        }

    def transition(self, new_state: AppState):
        """Transition to new state if valid"""
        if new_state in self.transitions[self.state]:
            print(f"State: {self.state.value} → {new_state.value}")
            self.state = new_state
            self.on_state_change(new_state)
        else:
            raise ValueError(f"Invalid transition: {self.state} → {new_state}")

    def on_state_change(self, new_state: AppState):
        """Handle state change side effects"""
        handlers = {
            AppState.SCANNING: self.start_scanning,
            AppState.CONNECTING: self.start_connecting,
            AppState.CONNECTED: self.on_connected,
            AppState.ACTIVE: self.start_input_injection,
            AppState.PAUSED: self.pause_input_injection,
            AppState.DISCONNECTED: self.cleanup,
        }
        handler = handlers.get(new_state)
        if handler:
            handler()
```

---

## File Structure

### Recommended Project Organization

```
wristcontrol/
│
├── wristcontrol/              # Main package
│   ├── __init__.py
│   ├── __main__.py           # Entry point
│   │
│   ├── core/                 # Core functionality
│   │   ├── __init__.py
│   │   ├── application.py    # Main app class
│   │   ├── state_machine.py  # State management
│   │   └── config.py         # Configuration
│   │
│   ├── input/                # Input injection
│   │   ├── __init__.py
│   │   ├── controller.py     # Abstract controller
│   │   ├── pynput_impl.py    # pynput implementation
│   │   └── platform_native.py # Platform-specific optimizations
│   │
│   ├── processing/           # Data processing
│   │   ├── __init__.py
│   │   ├── motion.py         # Motion algorithm
│   │   ├── gestures.py       # Gesture detection
│   │   ├── filters.py        # Kalman, smoothing, etc.
│   │   └── voice.py          # Voice processing
│   │
│   ├── connection/           # BLE communication
│   │   ├── __init__.py
│   │   ├── ble_manager.py    # BLE connection
│   │   ├── scanner.py        # Device discovery
│   │   └── protocol.py       # Data protocol
│   │
│   ├── ui/                   # User interface
│   │   ├── __init__.py
│   │   ├── tray.py           # System tray
│   │   ├── settings.py       # Settings window
│   │   └── resources/        # Icons, images
│   │       ├── icon.png
│   │       └── icon_connected.png
│   │
│   └── utils/                # Utilities
│       ├── __init__.py
│       ├── platform.py       # Platform detection
│       ├── permissions.py    # Permission checks
│       └── logging.py        # Logging setup
│
├── tests/                    # Unit tests
│   ├── __init__.py
│   ├── test_motion.py
│   ├── test_gestures.py
│   ├── test_config.py
│   └── test_integration.py
│
├── scripts/                  # Development scripts
│   ├── simulate_watch.py    # Sensor simulator
│   ├── benchmark.py         # Performance tests
│   └── build.py             # Build script
│
├── docs/                     # Documentation
│   ├── api.md
│   ├── architecture.md
│   └── user_guide.md
│
├── requirements.txt          # Dependencies
├── setup.py                  # Package setup
├── README.md
├── LICENSE
└── .gitignore
```

---

## Quick Start Implementation Plan

### Week 1: Foundation

**Day 1-2: Input Injection**
```python
# File: wristcontrol/input/controller.py
from pynput.mouse import Controller

class InputController:
    def __init__(self):
        self.mouse = Controller()

    def move_cursor(self, dx, dy):
        self.mouse.move(int(dx), int(dy))

    def click(self, button='left'):
        # Implementation...
```

**Day 3-4: Configuration**
```python
# File: wristcontrol/core/config.py
from dataclasses import dataclass
import json

@dataclass
class Config:
    sensitivity: float = 50.0
    # ...

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(asdict(self), f)
```

**Day 5-7: Basic UI**
```python
# File: wristcontrol/ui/tray.py
import pystray

class TrayApp:
    def __init__(self):
        # Implementation...
```

### Week 2: Processing

**Day 1-3: Motion Algorithm**
```python
# File: wristcontrol/processing/motion.py
class MotionProcessor:
    def process(self, sensor_data):
        # Convert gyro to cursor delta
        # Apply smoothing
        # Return dx, dy
```

**Day 4-5: Gesture Processing**
```python
# File: wristcontrol/processing/gestures.py
class GestureProcessor:
    def detect(self, sensor_data):
        # Detect taps from accel spikes
        # Return gesture type
```

**Day 6-7: Integration**
```python
# File: wristcontrol/core/application.py
class WristControl:
    def __init__(self):
        self.input = InputController()
        self.motion = MotionProcessor()
        self.gestures = GestureProcessor()

    def run(self):
        # Main loop
```

### Week 3: BLE Integration

**Day 1-3: BLE Connection**
```python
# File: wristcontrol/connection/ble_manager.py
import asyncio
from bleak import BleakClient

class BLEManager:
    async def connect(self, address):
        # Connect to watch
        # Subscribe to characteristics
```

**Day 4-5: Data Protocol**
```python
# File: wristcontrol/connection/protocol.py
import struct

def parse_sensor_packet(data):
    # Unpack binary data
    # Return SensorData object
```

**Day 6-7: Testing**
- Test with simulated data
- Test BLE connection stability
- Measure latency

### Week 4: Polish

**Day 1-2: Settings UI**
- Complete PyQt settings window
- Wire up to config system

**Day 3-4: State Management**
- Implement state machine
- Add auto-reconnect logic

**Day 5-6: Packaging**
- Create PyInstaller spec
- Build executables for all platforms
- Test on clean systems

**Day 7: Documentation**
- Write user guide
- Document API
- Create demo videos

---

## Performance Optimization Checklist

### Low-Latency Optimizations

- [ ] **Use queue.Queue for thread communication** (not locks)
- [ ] **Process sensor data in dedicated high-priority thread**
- [ ] **Minimize allocations in hot path** (reuse objects)
- [ ] **Use numpy for vector operations** (if processing batches)
- [ ] **Profile with cProfile** to find bottlenecks
- [ ] **Consider C extension** for critical paths (only if needed)
- [ ] **Batch BLE notifications** (reduce context switches)
- [ ] **Use asyncio for BLE** (non-blocking I/O)

### Memory Optimizations

- [ ] **Circular buffers** for smoothing (fixed size)
- [ ] **Lazy-load settings UI** (only when opened)
- [ ] **Release resources** when not in use
- [ ] **Monitor memory** with tracemalloc

### CPU Optimizations

- [ ] **Sleep when idle** (don't spin)
- [ ] **Throttle UI updates** (not every frame)
- [ ] **Use generator expressions** over list comprehensions
- [ ] **Cache computed values** (e.g., config-derived params)

---

## Security Considerations

### Data Privacy

1. **No cloud by default:** Process everything locally
2. **Optional cloud:** User must opt-in for cloud STT
3. **No telemetry:** Don't send usage data unless explicit consent
4. **Encrypted BLE:** Use LE Secure Connections

### Input Injection Safety

1. **Require user activation:** Don't inject input without permission
2. **Escape mechanism:** Palm-up gesture to disable
3. **Visual feedback:** Show when input is active
4. **Sandboxing:** Don't inject into password fields (if detectable)

### Configuration Security

1. **Validate all inputs:** Check ranges, types
2. **Safe defaults:** Conservative settings
3. **Permission checks:** Verify before enabling
4. **Auto-disable on errors:** Fail safe

---

## Testing Strategy

### Unit Tests

```python
# tests/test_motion.py
def test_motion_processor():
    processor = MotionProcessor(Config())
    sensor_data = SensorData(
        timestamp=0,
        gyro_x=10.0,
        gyro_y=5.0,
        # ...
    )
    dx, dy = processor.process(sensor_data)
    assert abs(dx - 10.0) < 1.0  # Approximate match
```

### Integration Tests

```python
# tests/test_integration.py
def test_end_to_end():
    app = WristControl()
    # Inject simulated sensor data
    # Verify cursor moved
```

### Performance Tests

```python
# tests/test_performance.py
def test_latency():
    # Measure input injection latency
    latencies = []
    for _ in range(1000):
        start = time.time()
        controller.move_cursor(1, 1)
        latencies.append(time.time() - start)
    assert mean(latencies) < 0.05  # <50ms
```

---

## Deployment Checklist

### Pre-Release

- [ ] All tests passing
- [ ] Performance benchmarks met
- [ ] Tested on all target platforms
- [ ] Permission checks working
- [ ] Auto-reconnect tested
- [ ] Error handling comprehensive
- [ ] Logging implemented
- [ ] User documentation complete

### Build

- [ ] Version number updated
- [ ] Icons included
- [ ] PyInstaller spec configured
- [ ] Builds created for Windows/Mac/Linux
- [ ] Executables tested on clean machines
- [ ] Code signing (macOS, Windows)

### Release

- [ ] Release notes written
- [ ] GitHub release created
- [ ] Binaries uploaded
- [ ] Documentation published
- [ ] Demo video created

---

## Next Steps

1. **Set up development environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Run reference implementations:**
   ```bash
   python planning/reference-implementations.md # Extract examples
   python minimal_demo.py
   ```

3. **Start implementing core:**
   - Create file structure
   - Implement InputController
   - Test cursor movement

4. **Iterate:**
   - Add motion processing
   - Integrate BLE (with simulator first)
   - Build UI
   - Package and test

This architecture provides a solid foundation for building a production-quality WristControl desktop application with low latency, cross-platform support, and professional user experience.
