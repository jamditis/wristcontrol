# Python BLE Libraries Research Report
## Building a Desktop Companion App for Smartwatch Data Streaming

*Research compiled: January 2026*

---

## Executive Summary

This report provides comprehensive research on Python Bluetooth Low Energy (BLE) libraries for building a desktop companion app that receives high-frequency sensor data (50-100Hz) and audio streams from a Samsung Galaxy Watch. The primary focus is on achieving **sub-50ms latency** for sensor-to-cursor movement while maintaining stable connections across Windows, macOS, and Linux.

**Key Findings:**
- **Bleak** is the recommended Python BLE library (cross-platform, asyncio-based)
- **Polling beats notifications** for ultra-low latency (20-25ms vs 180-200ms)
- **Connection interval of 7.5ms** is critical for sub-50ms end-to-end latency
- **MTU negotiation to 247+ bytes** increases throughput by 15%
- **Audio over BLE** requires compression (Opus codec) and push-to-talk for battery life
- **DoublePoint TouchSDK** provides proven architecture for watch-to-desktop communication

---

## 1. Bleak Library: Deep Dive

### 1.1 Overview

**Bleak** (Bluetooth Low Energy platform Agnostic Klient) is the industry-standard Python library for BLE communication. It provides a cross-platform, asyncio-based API for GATT client operations.

- **Latest Version:** 2.1.1 (as of January 2026)
- **License:** MIT
- **Repository:** https://github.com/hbldh/bleak
- **Documentation:** https://bleak.readthedocs.io/

**Why Bleak:**
- Cross-platform abstraction (Windows/macOS/Linux/Android)
- Built on asyncio for non-blocking I/O
- Active maintenance and community support
- Platform-specific backends handle OS differences automatically
- Support for BLE 5.0+ features

### 1.2 Platform-Specific Backends

Bleak uses different backends for each OS:

| Platform | Backend | Requirements |
|----------|---------|--------------|
| Windows | WinRT (Windows Runtime) | Windows 10 version 16299+ |
| macOS | CoreBluetooth | macOS 10.11+ |
| Linux | BlueZ via DBus | BlueZ >= 5.43 |
| Android | Python4Android (community) | Community-supported |

### 1.3 Installation

```bash
# Install Bleak
pip install bleak

# For automatic retry logic
pip install bleak-retry-connector
```

### 1.4 Basic Connection Pattern

```python
import asyncio
from bleak import BleakClient, BleakScanner

# Device MAC address or UUID
WATCH_ADDRESS = "AA:BB:CC:DD:EE:FF"

async def connect_to_watch():
    """Basic connection using async context manager (recommended)."""
    async with BleakClient(WATCH_ADDRESS) as client:
        print(f"Connected: {client.is_connected}")

        # Discover services
        services = client.services
        for service in services:
            print(f"Service: {service.uuid}")
            for char in service.characteristics:
                print(f"  Characteristic: {char.uuid}")
                print(f"    Properties: {char.properties}")

        # Keep connection alive
        await asyncio.sleep(10.0)

# Run the async function
asyncio.run(connect_to_watch())
```

### 1.5 Discovering Devices

```python
async def scan_for_watch():
    """Scan for BLE devices and find your watch."""
    print("Scanning for BLE devices...")

    devices = await BleakScanner.discover(timeout=10.0)

    for device in devices:
        print(f"Device: {device.name} ({device.address})")
        print(f"  RSSI: {device.rssi}")
        print(f"  Metadata: {device.metadata}")

    return devices

# Find specific device by name
async def find_watch_by_name(watch_name="Galaxy Watch"):
    """Find device by name pattern."""
    device = await BleakScanner.find_device_by_name(watch_name, timeout=10.0)
    if device:
        print(f"Found watch: {device.name} at {device.address}")
        return device
    else:
        print("Watch not found")
        return None
```

### 1.6 Reading Characteristics

```python
async def read_battery_level(client):
    """Read battery level characteristic."""
    # Standard Battery Service UUID
    BATTERY_LEVEL_UUID = "00002a19-0000-1000-8000-00805f9b34fb"

    try:
        value = await client.read_gatt_char(BATTERY_LEVEL_UUID)
        battery_level = int(value[0])
        print(f"Battery level: {battery_level}%")
        return battery_level
    except Exception as e:
        print(f"Error reading battery: {e}")
        return None
```

### 1.7 Writing Characteristics

```python
async def write_configuration(client, characteristic_uuid, data):
    """Write configuration data to a characteristic."""
    try:
        # data should be bytes or bytearray
        await client.write_gatt_char(characteristic_uuid, data, response=True)
        print(f"Successfully wrote {len(data)} bytes")
    except Exception as e:
        print(f"Error writing characteristic: {e}")
```

### 1.8 Subscribing to Notifications (Standard Method)

```python
async def subscribe_to_imu_data(client):
    """Subscribe to IMU sensor notifications."""
    # Your custom IMU characteristic UUID
    IMU_CHAR_UUID = "12345678-1234-5678-1234-56789abcdef0"

    def notification_handler(sender, data: bytearray):
        """Callback for incoming notifications."""
        # Parse sensor data (example: 6 floats for accel + gyro)
        import struct
        if len(data) == 24:  # 6 floats * 4 bytes
            values = struct.unpack('<6f', data)
            accel_x, accel_y, accel_z = values[0:3]
            gyro_x, gyro_y, gyro_z = values[3:6]
            print(f"Accel: ({accel_x:.2f}, {accel_y:.2f}, {accel_z:.2f})")
            print(f"Gyro: ({gyro_x:.2f}, {gyro_y:.2f}, {gyro_z:.2f})")

    # Start notifications
    await client.start_notify(IMU_CHAR_UUID, notification_handler)
    print("Subscribed to IMU notifications")

    # Keep receiving for 30 seconds
    await asyncio.sleep(30.0)

    # Stop notifications
    await client.stop_notify(IMU_CHAR_UUID)
    print("Unsubscribed from notifications")
```

### 1.9 Advanced: Notification with Queue Pattern

For decoupling notification handling from processing:

```python
import asyncio
from bleak import BleakClient

async def notification_with_queue_pattern():
    """Advanced pattern: Use asyncio.Queue to decouple notification from processing."""

    IMU_CHAR_UUID = "12345678-1234-5678-1234-56789abcdef0"
    WATCH_ADDRESS = "AA:BB:CC:DD:EE:FF"

    # Queue for passing data from notification callback to processor
    data_queue = asyncio.Queue()

    def notification_handler(sender, data: bytearray):
        """Callback puts data in queue (runs in event loop)."""
        # Use put_nowait since we're already in the event loop
        data_queue.put_nowait((asyncio.get_event_loop().time(), data))

    async def process_sensor_data():
        """Consumer coroutine that processes queued data."""
        while True:
            timestamp, data = await data_queue.get()

            # Process the sensor data
            import struct
            if len(data) == 24:
                values = struct.unpack('<6f', data)
                # Do something with the data...
                latency = asyncio.get_event_loop().time() - timestamp
                print(f"Processing latency: {latency*1000:.2f}ms")

            # Signal task is done
            data_queue.task_done()

    async with BleakClient(WATCH_ADDRESS) as client:
        # Start the processor
        processor_task = asyncio.create_task(process_sensor_data())

        # Start notifications
        await client.start_notify(IMU_CHAR_UUID, notification_handler)

        # Run for 60 seconds
        await asyncio.sleep(60.0)

        # Cleanup
        await client.stop_notify(IMU_CHAR_UUID)
        processor_task.cancel()
        try:
            await processor_task
        except asyncio.CancelledError:
            pass
```

### 1.10 MTU Negotiation

```python
async def negotiate_mtu(client, desired_mtu=512):
    """
    Negotiate MTU size for higher throughput.

    Note: MTU negotiation is often handled automatically by Bleak,
    but you can check the negotiated MTU.
    """
    # On some platforms, you can access MTU via client.mtu_size
    try:
        mtu = client.mtu_size
        print(f"Negotiated MTU: {mtu} bytes")
        # Effective payload = MTU - 3 (ATT overhead)
        payload_size = mtu - 3
        print(f"Max payload per notification: {payload_size} bytes")
        return mtu
    except AttributeError:
        print("MTU size not available on this platform")
        return None
```

---

## 2. WebBLE vs Native BLE

### 2.1 Comparison Matrix

| Feature | WebBLE (Web Bluetooth API) | Native BLE (Bleak) |
|---------|---------------------------|-------------------|
| **Deployment** | No installation, runs in browser | Requires app installation |
| **Platform Support** | Chrome/Edge/Opera (limited Safari) | Windows, macOS, Linux, Android |
| **Performance** | Variable, browser-dependent | Optimized, direct OS APIs |
| **Latency** | Higher (browser overhead) | Lower (direct access) |
| **Background Operation** | Limited/impossible | Full support |
| **Device Access** | BLE only | Full hardware access |
| **Development** | JavaScript | Python, full ecosystem |
| **Security** | Sandboxed, HTTPS required | OS-level permissions |
| **MTU Control** | Limited | Full control |
| **Best For** | Quick prototypes, web apps, demos | Production apps, low latency |

### 2.2 When to Use WebBLE

**Ideal Use Cases:**
- Quick prototyping and demos
- Fleet management tools (update many devices)
- Web-based dashboards for monitoring
- Cross-platform without installation
- Firmware update utilities

**Limitations:**
- Not supported in WebView (Android/iOS)
- Requires HTTPS (except localhost)
- Background operation very limited
- Higher latency than native
- Battery life concerns for continuous streaming

### 2.3 When to Use Native BLE (Bleak)

**Ideal Use Cases:**
- **High-frequency sensor streaming (50-100Hz)** ✓ Your use case
- **Sub-50ms latency requirements** ✓ Your use case
- **Audio streaming** ✓ Your use case
- Background operation (system tray app)
- Complex processing pipelines
- Offline operation
- Battery-sensitive applications

**Recommendation for Your Project:**

Given your requirements:
- 50-100Hz sensor data streaming
- Sub-50ms sensor-to-cursor latency
- Audio streaming from watch microphone
- System tray background service

**Use Bleak (native Python BLE)** for the desktop companion app. WebBLE is not suitable for these latency and throughput requirements.

---

## 3. High-Frequency Data Streaming (50-100Hz)

### 3.1 Challenges

**Throughput Bottlenecks:**
- Default BLE connection interval: 30-50ms
- Default MTU: 23 bytes (only 20 bytes payload)
- Notification queuing delays
- Python processing overhead

**Real-World Performance:**
- At 100Hz sampling on watch, receive rate may drop to 20-30Hz over BLE
- Adding multiple sensors (accel + gyro) further reduces effective rate
- Notification floods can overwhelm queues (>256 pending)

### 3.2 Solutions

#### 3.2.1 Optimize Connection Parameters

**Key Parameters:**
```python
# Connection interval: 7.5ms minimum (vs 30-50ms default)
# Slave latency: 0 (no skipped events)
# Supervision timeout: 5000ms
# Packets per interval: Maximize based on MTU

# Note: These are configured on the peripheral (watch) side
# Desktop (central) can request these parameters
```

**Connection Parameter Impact:**
```
Throughput = (8 * BytesPerPacket * PacketsPerInterval * 1000) / ConnectionInterval

Example with optimized parameters:
- Connection interval: 7.5ms
- MTU: 247 bytes (244 payload)
- Packets per interval: 4

Throughput = (8 * 244 * 4 * 1000) / 7.5
         = 1,042,133 bps
         = ~127 KB/s
```

#### 3.2.2 Increase MTU Size

```python
async def optimize_for_high_throughput(client):
    """
    Request maximum MTU and verify negotiation.

    Benefits:
    - 15% throughput increase
    - Reduced packet overhead
    - Lower power consumption
    """
    # MTU negotiation happens automatically during connection
    # Check the result
    mtu = getattr(client, 'mtu_size', 23)
    print(f"MTU: {mtu} bytes")

    if mtu < 247:
        print(f"Warning: MTU {mtu} is below optimal (247+)")
        print("Throughput will be reduced.")

    # Calculate optimal packet size
    max_payload = mtu - 3  # ATT overhead
    print(f"Max notification payload: {max_payload} bytes")

    return max_payload
```

**MTU Recommendations:**
- Request **247+ bytes** to minimize overhead
- Maximum per BLE spec: **512 bytes** (most stacks support this)
- iOS devices: Typically **185 bytes**
- Android devices: Typically **247-517 bytes**

#### 3.2.3 Batching Sensor Samples

Instead of sending each sensor sample individually, batch multiple samples:

```python
# Watch-side pseudocode (send 10 samples per notification)
SAMPLES_PER_PACKET = 10
sample_buffer = []

def on_sensor_reading(accel, gyro):
    sample_buffer.append((accel, gyro))

    if len(sample_buffer) >= SAMPLES_PER_PACKET:
        # Pack 10 samples into one notification
        # Each sample: 6 floats (3 accel + 3 gyro) = 24 bytes
        # 10 samples = 240 bytes (fits in 247 MTU)
        data = pack_samples(sample_buffer)
        ble_notify(IMU_CHARACTERISTIC, data)
        sample_buffer.clear()
```

**Desktop-side unpacking:**

```python
import struct

def unpack_batched_samples(data: bytearray):
    """Unpack batched sensor samples."""
    SAMPLE_SIZE = 24  # 6 floats * 4 bytes
    num_samples = len(data) // SAMPLE_SIZE

    samples = []
    for i in range(num_samples):
        offset = i * SAMPLE_SIZE
        sample_data = data[offset:offset + SAMPLE_SIZE]
        values = struct.unpack('<6f', sample_data)
        samples.append({
            'accel': values[0:3],
            'gyro': values[3:6]
        })

    return samples

# In notification handler
def notification_handler(sender, data: bytearray):
    samples = unpack_batched_samples(data)
    for sample in samples:
        process_sensor_sample(sample)
```

#### 3.2.4 Use Notifications (Not Indications)

```python
# GOOD: Notifications (no acknowledgment, higher throughput)
await client.start_notify(CHAR_UUID, handler)

# BAD: Indications (require ACK, lower throughput)
# Indications cut throughput in half due to round-trip ACKs
```

#### 3.2.5 Clock Synchronization

For multi-sensor streaming, handle clock drift:

```python
import numpy as np
from scipy import interpolate

class SensorTimestampSync:
    """Synchronize timestamps across multiple sensors with clock drift."""

    def __init__(self, target_rate_hz=100):
        self.target_rate_hz = target_rate_hz
        self.target_dt = 1.0 / target_rate_hz
        self.samples = []

    def add_sample(self, timestamp, value):
        """Add a sample with its device timestamp."""
        self.samples.append((timestamp, value))

    def interpolate_to_fixed_rate(self):
        """Interpolate samples to fixed rate using pchip (shape-preserving)."""
        if len(self.samples) < 2:
            return []

        # Extract timestamps and values
        timestamps = np.array([s[0] for s in self.samples])
        values = np.array([s[1] for s in self.samples])

        # Create target timestamps at fixed rate
        t_start = timestamps[0]
        t_end = timestamps[-1]
        target_timestamps = np.arange(t_start, t_end, self.target_dt)

        # Shape-preserving piecewise cubic interpolation
        interp_func = interpolate.PchipInterpolator(timestamps, values)
        interpolated_values = interp_func(target_timestamps)

        return list(zip(target_timestamps, interpolated_values))
```

### 3.3 Realistic Throughput Expectations

**BLE 4.2 (with DLE):**
- Theoretical max: ~780 Kbps (97.6 KB/s)
- Realistic: ~500 Kbps (62.5 KB/s)

**BLE 5.0:**
- Theoretical max: ~1.25 Mbps (156 KB/s)
- Realistic: ~800 Kbps (100 KB/s)

**For 100Hz IMU streaming:**
- 6 floats per sample (accel + gyro): 24 bytes
- 100 samples/sec: 2.4 KB/s
- **Easily achievable** even with BLE 4.2

**For 100Hz with 10 sensors:**
- 24 KB/s required
- **Achievable** with optimized parameters

---

## 4. Connection Management & Auto-Reconnect

### 4.1 The Disconnect Challenge

BLE connections can drop due to:
- Radio interference
- Device moving out of range
- Watch entering power saving mode
- OS Bluetooth stack resets
- Application crashes

**Critical for your use case:** Users expect seamless reconnection without losing cursor control for more than a moment.

### 4.2 Disconnect Detection

```python
import asyncio
from bleak import BleakClient

class WatchConnection:
    """Manages watch connection with disconnect detection."""

    def __init__(self, address):
        self.address = address
        self.client = None
        self.disconnect_event = asyncio.Event()
        self.is_connected = False

    def disconnected_callback(self, client):
        """Called when device disconnects."""
        print(f"Device {self.address} disconnected!")
        self.is_connected = False
        self.disconnect_event.set()

    async def connect(self):
        """Connect to watch with disconnect callback."""
        self.client = BleakClient(
            self.address,
            disconnected_callback=self.disconnected_callback
        )
        await self.client.connect()
        self.is_connected = True
        self.disconnect_event.clear()
        print(f"Connected to {self.address}")
        return self.client
```

### 4.3 Auto-Reconnect Pattern (Robust)

```python
import asyncio
from bleak import BleakClient, BleakScanner

class AutoReconnectWatchClient:
    """BLE client with automatic reconnection."""

    def __init__(self, address, reconnect_delay=2.0, max_attempts=None):
        self.address = address
        self.reconnect_delay = reconnect_delay
        self.max_attempts = max_attempts  # None = infinite
        self.client = None
        self.disconnect_event = asyncio.Event()
        self.should_stop = False

        # Callbacks
        self.on_connected = None
        self.on_disconnected = None
        self.notification_handlers = {}

    def disconnected_callback(self, client):
        """Internal disconnect handler."""
        print(f"Disconnected from {self.address}")
        if self.on_disconnected:
            self.on_disconnected()
        self.disconnect_event.set()

    async def _find_device(self):
        """Find device by address."""
        print(f"Searching for device {self.address}...")
        device = await BleakScanner.find_device_by_address(
            self.address,
            timeout=10.0
        )
        return device

    async def _connect_once(self):
        """Single connection attempt."""
        try:
            # Find device first (recommended for reliability)
            device = await self._find_device()
            if not device:
                raise Exception(f"Device {self.address} not found")

            # Connect using BleakDevice object
            self.client = BleakClient(
                device,
                disconnected_callback=self.disconnected_callback
            )
            await self.client.connect()

            # Re-subscribe to characteristics
            for char_uuid, handler in self.notification_handlers.items():
                await self.client.start_notify(char_uuid, handler)

            self.disconnect_event.clear()

            if self.on_connected:
                self.on_connected(self.client)

            print(f"Connected to {self.address}")
            return True

        except Exception as e:
            print(f"Connection attempt failed: {e}")
            return False

    async def maintain_connection(self):
        """
        Main loop that maintains connection with auto-reconnect.

        This runs indefinitely until stopped.
        """
        attempt = 0

        while not self.should_stop:
            # Connect
            connected = await self._connect_once()

            if connected:
                attempt = 0  # Reset attempt counter on success

                # Wait for disconnect event
                await self.disconnect_event.wait()

                if self.should_stop:
                    break

                print("Connection lost, will attempt to reconnect...")
            else:
                attempt += 1

                if self.max_attempts and attempt >= self.max_attempts:
                    print(f"Max reconnection attempts ({self.max_attempts}) reached")
                    break

                print(f"Reconnection attempt {attempt} failed")

            # Exponential backoff with cap
            delay = min(self.reconnect_delay * (2 ** min(attempt, 5)), 60.0)
            print(f"Waiting {delay:.1f}s before next attempt...")
            await asyncio.sleep(delay)

    def add_notification_handler(self, char_uuid, handler):
        """Register a notification handler that persists across reconnects."""
        self.notification_handlers[char_uuid] = handler

    async def stop(self):
        """Stop the connection manager."""
        self.should_stop = True
        self.disconnect_event.set()

        if self.client and self.client.is_connected:
            await self.client.disconnect()


# Usage example
async def main():
    WATCH_ADDRESS = "AA:BB:CC:DD:EE:FF"
    IMU_CHAR_UUID = "12345678-1234-5678-1234-56789abcdef0"

    client_manager = AutoReconnectWatchClient(WATCH_ADDRESS)

    # Set up callbacks
    def on_connected(client):
        print("Connected callback - ready to use watch!")

    def on_disconnected():
        print("Disconnected callback - cursor control paused")

    def imu_notification_handler(sender, data):
        # Process sensor data
        print(f"Received {len(data)} bytes")

    client_manager.on_connected = on_connected
    client_manager.on_disconnected = on_disconnected
    client_manager.add_notification_handler(IMU_CHAR_UUID, imu_notification_handler)

    # Start the connection manager (runs forever)
    await client_manager.maintain_connection()

asyncio.run(main())
```

### 4.4 Using bleak-retry-connector

For simpler cases, use the `bleak-retry-connector` package:

```python
from bleak_retry_connector import establish_connection, BleakClientWithServiceCache

async def connect_with_retry(address):
    """Connect with automatic retry logic."""

    # establish_connection handles:
    # - Automatic retries with exponential backoff
    # - Service caching for faster reconnection
    # - Robust error handling

    client = await establish_connection(
        BleakClientWithServiceCache,
        address,
        name="Galaxy Watch",
        max_attempts=3
    )

    return client
```

### 4.5 Connection Stability Best Practices

```python
# 1. Always use BleakDevice objects when connecting to multiple devices
devices = await BleakScanner.discover()
for device in devices:
    if device.name == "Galaxy Watch":
        client = BleakClient(device)  # Not just the address
        await client.connect()

# 2. Use async context manager for cleanup
async with BleakClient(device) as client:
    # Connection is automatically cleaned up
    pass

# 3. Set appropriate timeouts
client = BleakClient(device, timeout=30.0)  # Default changed to 30s in 2.1.1

# 4. Handle graceful shutdown
import signal

async def shutdown(signal, loop, client):
    print(f"Received exit signal {signal.name}")
    if client.is_connected:
        await client.disconnect()
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    [task.cancel() for task in tasks]
    await asyncio.gather(*tasks, return_exceptions=True)
    loop.stop()

# Register signal handlers
loop = asyncio.get_event_loop()
for sig in (signal.SIGTERM, signal.SIGINT):
    loop.add_signal_handler(
        sig,
        lambda s=sig: asyncio.create_task(shutdown(s, loop, client))
    )
```

---

## 5. Cross-Platform Compatibility

### 5.1 Platform-Specific Backends

| Platform | Backend | Status | Notes |
|----------|---------|--------|-------|
| Windows | WinRT | ✓ Fully supported | Requires Win10 16299+ |
| macOS | CoreBluetooth | ✓ Fully supported | macOS 10.11+ |
| Linux | BlueZ (DBus) | ✓ Fully supported | BlueZ 5.43+ |
| Android | Python4Android | ⚠ Community | Not tested by maintainers |

### 5.2 Linux-Specific Considerations

**BlueZ Requirements:**
```bash
# Check BlueZ version
bluetoothctl --version  # Need 5.43+

# Common issues on Linux
sudo systemctl status bluetooth  # Ensure service is running

# Grant permissions for Bluetooth access
sudo setcap cap_net_raw,cap_net_admin+eip $(readlink -f $(which python3))

# Or add user to bluetooth group
sudo usermod -a -G bluetooth $USER
```

**Linux Auto-Reconnect Issue:**

When stopping code abruptly (e.g., IDE stop button), BlueZ may keep the device connected:

```bash
# Restart Bluetooth service to clean up
sudo systemctl restart bluetooth
```

**Solution in code:**

```python
import atexit

async def cleanup_bluetooth():
    """Ensure proper disconnect on Linux."""
    if client and client.is_connected:
        await client.disconnect()
        await asyncio.sleep(0.5)  # Give BlueZ time to process

atexit.register(lambda: asyncio.run(cleanup_bluetooth()))
```

### 5.3 Windows-Specific Considerations

**Windows Quirk:**
- Windows may auto-reconnect when peripheral disconnects
- This can interfere with your reconnection logic

**Solution:**

```python
import platform

if platform.system() == "Windows":
    # Give Windows time to settle before reconnecting
    await asyncio.sleep(1.0)
```

### 5.4 macOS-Specific Considerations

**CoreBluetooth Permissions:**
- macOS 10.15+ requires Bluetooth permission in System Preferences
- App must request permission on first use

**Connection Limits:**
- macOS typically handles up to 8-10 concurrent BLE connections well

### 5.5 Cross-Platform MTU Differences

```python
async def get_platform_mtu(client):
    """Get platform-specific MTU information."""
    import platform

    system = platform.system()
    mtu = getattr(client, 'mtu_size', 23)

    print(f"Platform: {system}")
    print(f"MTU: {mtu} bytes")

    # Expected MTU by platform
    expected = {
        'Windows': '23-512 (negotiated)',
        'Darwin': '185 (iOS) or 512 (macOS)',  # Darwin = macOS
        'Linux': '23-512 (negotiated)'
    }

    print(f"Expected for {system}: {expected.get(system, 'Unknown')}")

    return mtu
```

### 5.6 Unified Installation Script

```python
# setup.py
from setuptools import setup, find_packages
import platform

install_requires = [
    'bleak>=2.1.1',
    'bleak-retry-connector>=3.0.0',
]

# Platform-specific dependencies
if platform.system() == 'Linux':
    install_requires.append('dbus-python>=1.2.18')
elif platform.system() == 'Windows':
    install_requires.append('winrt>=1.0.21033.1')

setup(
    name='wristcontrol-desktop',
    version='0.1.0',
    packages=find_packages(),
    install_requires=install_requires,
    python_requires='>=3.8',
)
```

---

## 6. Latency Optimization: Achieving Sub-50ms

### 6.1 Latency Budget Breakdown

To achieve **sub-50ms sensor-to-cursor** latency, understand where time is spent:

```
Total Latency = Sensor_Sampling + Watch_Processing + BLE_TX +
                BLE_RX + Desktop_Processing + OS_Injection

Target breakdown:
- Sensor sampling:      ~10ms (100Hz = 10ms intervals)
- Watch processing:     ~2-5ms (gesture detection, packing)
- BLE transmission:     ~7.5-20ms (connection interval dependent)
- BLE reception:        ~1-5ms (Python callback)
- Desktop processing:   ~2-5ms (unpacking, cursor calc)
- OS injection:         ~1-2ms (pynput/pyautogui)
---------------------------------------------------------------
Total:                  ~23.5-47ms ✓ Achievable
```

### 6.2 Critical Optimization: Polling vs Notifications

**Shocking Discovery:**

Using `start_notify()` on Windows 10 results in **180-200ms latency**. Switching to polling with `asyncio.sleep(0)` reduces this to **20-25ms**.

**Standard Notification Approach (HIGH LATENCY):**

```python
# Approach 1: Notifications (180-200ms latency on Windows)
async def high_latency_approach(client):
    def handler(sender, data):
        timestamp = asyncio.get_event_loop().time()
        # By the time this runs, 180-200ms have elapsed!
        print(f"Data: {data}")

    await client.start_notify(IMU_CHAR_UUID, handler)
    await asyncio.sleep(60)
```

**Optimized Polling Approach (LOW LATENCY):**

```python
# Approach 2: Polling (20-25ms latency)
async def low_latency_polling(client):
    """Poll characteristic in tight loop for minimal latency."""

    IMU_CHAR_UUID = "12345678-1234-5678-1234-56789abcdef0"
    last_value = None

    while True:
        try:
            # Read characteristic
            value = await client.read_gatt_char(IMU_CHAR_UUID)

            # Only process if value changed
            if value != last_value:
                process_sensor_data(value)
                last_value = value

            # Critical: asyncio.sleep(0) yields to event loop
            # This prevents blocking while maintaining low latency
            await asyncio.sleep(0)

        except Exception as e:
            print(f"Error reading: {e}")
            await asyncio.sleep(0.01)  # Brief pause on error
```

**Hybrid Approach (Best of Both):**

For even better performance, use notifications to wake up polling:

```python
async def hybrid_approach(client):
    """Use notifications as trigger, then poll for latest value."""

    IMU_CHAR_UUID = "12345678-1234-5678-1234-56789abcdef0"
    new_data_event = asyncio.Event()

    def notification_handler(sender, data):
        # Just signal that new data is available
        new_data_event.set()

    async def polling_loop():
        while True:
            # Wait for notification signal
            await new_data_event.wait()
            new_data_event.clear()

            # Immediately poll for latest value
            value = await client.read_gatt_char(IMU_CHAR_UUID)
            process_sensor_data(value)

    # Start both
    await client.start_notify(IMU_CHAR_UUID, notification_handler)
    await polling_loop()
```

### 6.3 Connection Interval Optimization

**The Single Most Important Parameter:**

```python
# Connection interval directly impacts latency floor

Connection Interval → Minimum Latency
- 7.5ms  → ~15ms total (best case)
- 15ms   → ~22ms total
- 30ms   → ~38ms total
- 50ms   → ~58ms total (exceeds target!)
```

**Watch-Side Configuration (WearOS/Tizen):**

```kotlin
// Kotlin (WearOS) - Request minimum connection interval
val connectionPriority = BluetoothGatt.CONNECTION_PRIORITY_HIGH
gatt.requestConnectionPriority(connectionPriority)

// This requests:
// - Interval: 11.25 - 15ms (Android default for HIGH priority)
// - Latency: 0
// - Timeout: 20s
```

**For even lower (7.5ms):**

```kotlin
// Advanced: Use L2CAP connection parameter update
// Note: Requires specific OS support
val params = BluetoothGattServerCallback.ConnectionParams(
    interval = 6,      // 7.5ms (6 * 1.25ms)
    latency = 0,       // No slave latency
    timeout = 500      // 5000ms supervision timeout
)
```

**Desktop-Side (Python/Bleak):**

Bleak doesn't directly control connection parameters (that's peripheral's job), but you can verify:

```python
async def check_connection_params(client):
    """
    Check effective connection parameters.
    Note: Not all platforms expose this info.
    """
    # This is platform-dependent and may not be available
    # On BlueZ (Linux), you can query via DBus

    print("Connection parameters:")
    print(f"  MTU: {getattr(client, 'mtu_size', 'N/A')}")
    # Connection interval typically not exposed in Bleak
    # Monitor latency empirically instead
```

### 6.4 Minimize Python Processing Overhead

**Use Efficient Data Structures:**

```python
import struct
from dataclasses import dataclass
from typing import Tuple

@dataclass
class SensorSample:
    """Efficient sensor sample representation."""
    accel: Tuple[float, float, float]
    gyro: Tuple[float, float, float]
    timestamp: float

# Fast unpacking with struct
def unpack_sample_fast(data: bytearray) -> SensorSample:
    """Unpack sensor sample with minimal overhead."""
    # Unpack all at once
    values = struct.unpack('<6fL', data)  # 6 floats + 1 unsigned long

    return SensorSample(
        accel=(values[0], values[1], values[2]),
        gyro=(values[3], values[4], values[5]),
        timestamp=values[6] / 1000.0  # Convert ms to seconds
    )
```

**Avoid Blocking Operations:**

```python
# BAD: Blocking I/O in notification handler
def bad_handler(sender, data):
    result = requests.post('http://api.example.com', json={'data': data})  # BLOCKS!

# GOOD: Queue for async processing
def good_handler(sender, data):
    data_queue.put_nowait(data)  # Non-blocking

async def async_processor():
    while True:
        data = await data_queue.get()
        async with aiohttp.ClientSession() as session:
            await session.post('http://api.example.com', json={'data': data})
```

### 6.5 OS Injection Optimization

**Library Comparison:**

| Library | Latency | Cross-platform | Notes |
|---------|---------|----------------|-------|
| pynput | ~1-2ms | ✓ | Recommended |
| pyautogui | ~5-10ms | ✓ | Slower |
| ctypes (direct) | <1ms | ✗ | Platform-specific, complex |

**Optimized Cursor Control:**

```python
from pynput.mouse import Controller, Button
import time

class LowLatencyCursor:
    """Optimized cursor control with minimal latency."""

    def __init__(self):
        self.mouse = Controller()
        self._last_position = self.mouse.position
        self._move_threshold = 1  # pixels

    def move_relative(self, dx: int, dy: int):
        """Move cursor relative to current position."""
        if abs(dx) < self._move_threshold and abs(dy) < self._move_threshold:
            return  # Skip sub-pixel movements

        current_x, current_y = self.mouse.position
        self.mouse.position = (current_x + dx, current_y + dy)

    def move_absolute(self, x: int, y: int):
        """Move cursor to absolute position."""
        self.mouse.position = (x, y)

    def click(self, button=Button.left):
        """Perform click."""
        self.mouse.click(button)

# Usage
cursor = LowLatencyCursor()

def process_sensor_to_cursor(accel, gyro):
    """Convert sensor data to cursor movement with minimal latency."""
    # Simple algorithm: use gyro for cursor velocity
    dx = int(gyro[1] * 50)  # Scale factor
    dy = int(-gyro[0] * 50)

    cursor.move_relative(dx, dy)
```

### 6.6 End-to-End Latency Measurement

```python
import time
import asyncio
from collections import deque

class LatencyMonitor:
    """Monitor end-to-end latency."""

    def __init__(self, window_size=100):
        self.latencies = deque(maxlen=window_size)
        self.start_times = {}

    def mark_start(self, sample_id):
        """Mark when sensor reading occurred."""
        self.start_times[sample_id] = time.perf_counter()

    def mark_end(self, sample_id):
        """Mark when cursor movement completed."""
        if sample_id in self.start_times:
            latency = (time.perf_counter() - self.start_times[sample_id]) * 1000
            self.latencies.append(latency)
            del self.start_times[sample_id]
            return latency
        return None

    def get_stats(self):
        """Get latency statistics."""
        if not self.latencies:
            return None

        import statistics
        return {
            'mean': statistics.mean(self.latencies),
            'median': statistics.median(self.latencies),
            'min': min(self.latencies),
            'max': max(self.latencies),
            'p95': sorted(self.latencies)[int(len(self.latencies) * 0.95)],
            'p99': sorted(self.latencies)[int(len(self.latencies) * 0.99)]
        }

# Usage
monitor = LatencyMonitor()

# On watch: Include timestamp in packet
# timestamp = current_time_ms()
# send_packet(sensor_data, timestamp)

# On desktop:
def notification_handler(sender, data):
    timestamp = struct.unpack('<L', data[-4:])[0]  # Last 4 bytes
    monitor.mark_start(timestamp)

    # Process and move cursor
    process_and_move_cursor(data[:-4])

    latency = monitor.mark_end(timestamp)
    if latency:
        print(f"Latency: {latency:.2f}ms")

# Periodically log stats
async def log_latency_stats():
    while True:
        await asyncio.sleep(10)
        stats = monitor.get_stats()
        if stats:
            print(f"Latency stats: {stats}")
```

---

## 7. Audio over BLE: Challenges & Solutions

### 7.1 The Challenge

Audio streaming over BLE is **significantly more challenging** than sensor data:

**Requirements:**
- Sample rate: 16kHz minimum for acceptable voice quality
- Bit depth: 16-bit
- Channels: 1 (mono)
- **Raw bitrate: 256 Kbps** (16000 * 16 * 1)

**BLE Reality:**
- Maximum realistic throughput: ~800 Kbps (BLE 5.0)
- With overhead: ~500 Kbps usable
- **Raw audio won't fit!**

**Solution: Compression is mandatory**

### 7.2 Codec Selection

| Codec | Bitrate | Latency | Quality | Complexity |
|-------|---------|---------|---------|------------|
| Opus | 6-64 Kbps | Very Low (~20ms) | Excellent | Medium |
| ADPCM | 32 Kbps | Very Low | Good | Low |
| SBC | 328 Kbps | Medium | Good | Medium |
| LC3 | 16-320 Kbps | Low | Excellent | High |

**Recommendation: Opus**

- Best quality-to-bitrate ratio
- Very low latency (~20ms encoding)
- Optimized for speech (6-24 Kbps) and music (up to 64 Kbps)
- Widely supported libraries

### 7.3 Watch-Side: Audio Capture & Encoding (Pseudocode)

```kotlin
// Kotlin (WearOS) - Audio capture with Opus encoding

import android.media.AudioRecord
import com.score.rahasak.utils.OpusEncoder

class AudioStreamer(private val bleGatt: BluetoothGatt) {
    private val sampleRate = 16000  // 16kHz
    private val channels = AudioFormat.CHANNEL_IN_MONO
    private val audioFormat = AudioFormat.ENCODING_PCM_16BIT
    private val bufferSize = AudioRecord.getMinBufferSize(sampleRate, channels, audioFormat)

    private lateinit var audioRecord: AudioRecord
    private lateinit var opusEncoder: OpusEncoder

    fun startStreaming() {
        // Initialize Opus encoder
        opusEncoder = OpusEncoder().apply {
            init(sampleRate, 1, OpusEncoder.OPUS_APPLICATION_VOIP)
            setBitrate(16000)  // 16 Kbps for voice
        }

        // Initialize audio recorder
        audioRecord = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            sampleRate,
            channels,
            audioFormat,
            bufferSize
        )

        audioRecord.startRecording()

        // Streaming loop
        thread {
            val pcmBuffer = ShortArray(160)  // 10ms at 16kHz
            val opusBuffer = ByteArray(1024)

            while (isStreaming) {
                // Capture PCM audio
                val read = audioRecord.read(pcmBuffer, 0, pcmBuffer.size)

                if (read > 0) {
                    // Encode to Opus
                    val encoded = opusEncoder.encode(pcmBuffer, read, opusBuffer)

                    if (encoded > 0) {
                        // Send via BLE notification
                        val packet = opusBuffer.copyOf(encoded)
                        sendAudioPacket(packet)
                    }
                }
            }
        }
    }

    private fun sendAudioPacket(data: ByteArray) {
        val audioCharacteristic = gatt.getService(AUDIO_SERVICE_UUID)
            .getCharacteristic(AUDIO_CHAR_UUID)

        audioCharacteristic.value = data
        gatt.writeCharacteristic(audioCharacteristic)
        // Or use notifications if configured
    }
}
```

### 7.4 Desktop-Side: Audio Reception & Decoding

```python
import asyncio
import opuslib
from opuslib import Decoder as OpusDecoder
import sounddevice as sd
import numpy as np
from collections import deque

class AudioReceiver:
    """Receive and decode Opus audio from BLE watch."""

    def __init__(self):
        # Opus decoder
        self.decoder = OpusDecoder(
            fs=16000,      # 16kHz sample rate
            channels=1     # Mono
        )

        # Audio playback stream
        self.stream = None
        self.audio_queue = deque(maxlen=100)

        # For STT
        self.stt_buffer = bytearray()

    async def start(self, client):
        """Start receiving audio."""
        AUDIO_CHAR_UUID = "12345678-1234-5678-1234-567890abcdef"

        # Start notifications
        await client.start_notify(AUDIO_CHAR_UUID, self.audio_notification_handler)

        # Start playback stream (optional - for monitoring)
        self.stream = sd.OutputStream(
            samplerate=16000,
            channels=1,
            dtype=np.int16,
            callback=self.audio_callback
        )
        self.stream.start()

    def audio_notification_handler(self, sender, opus_data: bytearray):
        """Handle incoming Opus audio packets."""
        try:
            # Decode Opus to PCM
            pcm_data = self.decoder.decode(
                bytes(opus_data),
                frame_size=160,  # 10ms at 16kHz
                decode_fec=False
            )

            # Add to playback queue
            pcm_array = np.frombuffer(pcm_data, dtype=np.int16)
            self.audio_queue.append(pcm_array)

            # Also buffer for STT
            self.stt_buffer.extend(pcm_data)

            # If buffer is large enough, send to STT
            if len(self.stt_buffer) >= 32000:  # 1 second at 16kHz
                asyncio.create_task(self.process_stt())

        except Exception as e:
            print(f"Audio decode error: {e}")

    def audio_callback(self, outdata, frames, time_info, status):
        """Callback for audio playback stream."""
        if self.audio_queue:
            data = self.audio_queue.popleft()
            if len(data) < frames:
                # Pad with zeros
                data = np.pad(data, (0, frames - len(data)))
            outdata[:] = data[:frames].reshape(-1, 1)
        else:
            # No audio available, output silence
            outdata.fill(0)

    async def process_stt(self):
        """Send buffered audio to speech-to-text service."""
        audio_data = bytes(self.stt_buffer)
        self.stt_buffer.clear()

        # Send to STT service (example with OpenAI Whisper API)
        # result = await whisper_api.transcribe(audio_data)
        # print(f"Transcription: {result}")
        pass

    async def stop(self, client):
        """Stop receiving audio."""
        AUDIO_CHAR_UUID = "12345678-1234-5678-1234-567890abcdef"
        await client.stop_notify(AUDIO_CHAR_UUID)

        if self.stream:
            self.stream.stop()
            self.stream.close()
```

**Install Opus Python library:**

```bash
pip install opuslib
pip install sounddevice numpy
```

### 7.5 Latency Considerations for Audio

**Audio Latency Budget:**

```
Total Audio Latency = Capture + Encoding + BLE_TX + BLE_RX + Decoding + STT

For <200ms target (acceptable for voice commands):
- Audio capture:        ~10ms (buffer 10ms chunks)
- Opus encoding:        ~20ms
- BLE transmission:     ~15-30ms (depends on connection interval)
- BLE reception:        ~5ms
- Opus decoding:        ~10ms
- Buffering for STT:    ~100ms (accumulate 1 second for Whisper)
- STT processing:       ~200-500ms (cloud) or 1-3s (local Whisper)
---------------------------------------------------------------------
Total:                  ~360-678ms
```

**Optimization: Streaming STT**

Instead of buffering 1 second, use streaming STT:

```python
import asyncio
import aiohttp

class StreamingSTT:
    """Streaming speech-to-text for lower latency."""

    def __init__(self, api_endpoint):
        self.api_endpoint = api_endpoint
        self.session = None
        self.ws = None

    async def start_stream(self):
        """Start streaming STT session."""
        self.session = aiohttp.ClientSession()
        self.ws = await self.session.ws_connect(self.api_endpoint)

        # Start receiver task
        asyncio.create_task(self.receive_transcriptions())

    async def send_audio_chunk(self, audio_data: bytes):
        """Send audio chunk to streaming STT."""
        if self.ws:
            await self.ws.send_bytes(audio_data)

    async def receive_transcriptions(self):
        """Receive transcription results."""
        async for msg in self.ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                result = msg.json()
                print(f"Transcription: {result['text']}")
                # Process voice command
                self.handle_voice_command(result['text'])

    def handle_voice_command(self, text):
        """Handle recognized voice command."""
        text_lower = text.lower()

        if 'click' in text_lower:
            # Trigger click
            pass
        elif 'scroll up' in text_lower:
            # Scroll
            pass
        # ... more commands
```

### 7.6 Battery Optimization: Push-to-Talk

Continuous audio streaming kills battery. Use push-to-talk:

```python
class PushToTalkAudio:
    """Audio streaming with push-to-talk activation."""

    def __init__(self, client):
        self.client = client
        self.is_streaming = False
        self.audio_receiver = AudioReceiver()

    async def start_recording(self):
        """Start audio streaming when button pressed."""
        if not self.is_streaming:
            # Send command to watch to start audio capture
            await self.send_watch_command('START_AUDIO')

            # Start receiving
            await self.audio_receiver.start(self.client)
            self.is_streaming = True
            print("Recording started")

    async def stop_recording(self):
        """Stop audio streaming when button released."""
        if self.is_streaming:
            # Send command to watch to stop audio capture
            await self.send_watch_command('STOP_AUDIO')

            # Stop receiving
            await self.audio_receiver.stop(self.client)
            self.is_streaming = False
            print("Recording stopped")

    async def send_watch_command(self, command):
        """Send command to watch."""
        COMMAND_CHAR_UUID = "12345678-1234-5678-1234-567890abcde0"
        command_byte = {'START_AUDIO': 0x01, 'STOP_AUDIO': 0x02}[command]
        await self.client.write_gatt_char(
            COMMAND_CHAR_UUID,
            bytes([command_byte])
        )
```

**Keyboard shortcut for activation:**

```python
from pynput import keyboard

class VoiceCommandActivator:
    """Activate voice recording with keyboard shortcut."""

    def __init__(self, push_to_talk: PushToTalkAudio):
        self.ptt = push_to_talk
        self.listener = None

    def start_listening(self):
        """Listen for hotkey press."""
        def on_press(key):
            if key == keyboard.Key.space:  # Or any other key
                asyncio.create_task(self.ptt.start_recording())

        def on_release(key):
            if key == keyboard.Key.space:
                asyncio.create_task(self.ptt.stop_recording())

        self.listener = keyboard.Listener(
            on_press=on_press,
            on_release=on_release
        )
        self.listener.start()
```

### 7.7 Alternative: Voice Activity Detection (VAD)

Instead of push-to-talk, use VAD to only stream when speech detected:

```python
import webrtcvad

class VADAudioStreamer:
    """Audio streaming with Voice Activity Detection."""

    def __init__(self):
        self.vad = webrtcvad.Vad(mode=3)  # Aggressive mode
        self.is_speech_active = False

    def process_audio_frame(self, pcm_frame: bytes, sample_rate=16000):
        """
        Process audio frame through VAD.

        Args:
            pcm_frame: PCM audio data (must be 10, 20, or 30ms)
            sample_rate: 8000, 16000, 32000, or 48000 Hz

        Returns:
            True if speech detected, False otherwise
        """
        # Frame must be 10, 20, or 30ms
        # For 16kHz: 160, 320, or 480 samples (320, 640, 960 bytes)

        is_speech = self.vad.is_speech(pcm_frame, sample_rate)

        if is_speech and not self.is_speech_active:
            print("Speech started")
            self.is_speech_active = True
            # Start streaming to STT
        elif not is_speech and self.is_speech_active:
            print("Speech ended")
            self.is_speech_active = False
            # Stop streaming, process final result

        return is_speech
```

---

## 8. Complete Architecture Example

Here's a complete desktop companion app architecture:

```python
"""
WristControl Desktop Companion App
Complete architecture example with BLE sensor + audio streaming
"""

import asyncio
import struct
from dataclasses import dataclass
from typing import Optional, Callable
from collections import deque

from bleak import BleakClient, BleakScanner
from pynput.mouse import Controller, Button
import opuslib
import sounddevice as sd
import numpy as np


# ============================================================================
# Configuration
# ============================================================================

WATCH_NAME = "Galaxy Watch"
WATCH_ADDRESS = "AA:BB:CC:DD:EE:FF"  # Set your watch's address

# GATT UUIDs (define these based on your watch app)
IMU_SERVICE_UUID = "12345678-0000-1000-8000-00805f9b34fb"
IMU_CHAR_UUID = "12345678-1234-5678-1234-56789abcdef0"
AUDIO_CHAR_UUID = "12345678-1234-5678-1234-567890abcdef"
GESTURE_CHAR_UUID = "12345678-1234-5678-1234-56789abcde1"
COMMAND_CHAR_UUID = "12345678-1234-5678-1234-56789abcde0"


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class IMUSample:
    """IMU sensor sample."""
    accel_x: float
    accel_y: float
    accel_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float
    timestamp_ms: int


@dataclass
class GestureEvent:
    """Gesture event from watch."""
    gesture_type: str  # 'tap', 'double_tap', 'hold', 'palm_up'
    timestamp_ms: int


# ============================================================================
# Cursor Control
# ============================================================================

class CursorController:
    """Low-latency cursor control from sensor data."""

    def __init__(self, sensitivity=50.0):
        self.mouse = Controller()
        self.sensitivity = sensitivity
        self.enabled = True

    def update_from_gyro(self, gyro_x: float, gyro_y: float):
        """Update cursor position from gyroscope data."""
        if not self.enabled:
            return

        # Convert gyro to cursor delta
        dx = int(gyro_y * self.sensitivity)
        dy = int(-gyro_x * self.sensitivity)

        # Apply movement
        if abs(dx) > 0 or abs(dy) > 0:
            current_x, current_y = self.mouse.position
            self.mouse.position = (current_x + dx, current_y + dy)

    def click(self, button=Button.left):
        """Perform mouse click."""
        self.mouse.click(button)

    def enable(self):
        """Enable cursor control."""
        self.enabled = True
        print("Cursor control enabled")

    def disable(self):
        """Disable cursor control."""
        self.enabled = False
        print("Cursor control disabled")


# ============================================================================
# Audio Receiver
# ============================================================================

class AudioReceiver:
    """Receive and decode audio from watch."""

    def __init__(self):
        self.decoder = opuslib.Decoder(fs=16000, channels=1)
        self.audio_queue = deque(maxlen=100)
        self.stt_callback: Optional[Callable] = None
        self.stt_buffer = bytearray()

    def process_audio_packet(self, opus_data: bytes):
        """Decode and process Opus audio packet."""
        try:
            # Decode Opus to PCM
            pcm_data = self.decoder.decode(opus_data, frame_size=160)

            # Buffer for STT
            self.stt_buffer.extend(pcm_data)

            # Process when we have 1 second
            if len(self.stt_buffer) >= 32000:
                if self.stt_callback:
                    self.stt_callback(bytes(self.stt_buffer))
                self.stt_buffer.clear()

        except Exception as e:
            print(f"Audio decode error: {e}")


# ============================================================================
# Watch Connection Manager
# ============================================================================

class WatchConnectionManager:
    """Manages BLE connection to watch with auto-reconnect."""

    def __init__(self, address: str):
        self.address = address
        self.client: Optional[BleakClient] = None
        self.disconnect_event = asyncio.Event()
        self.should_stop = False

        # Components
        self.cursor_controller = CursorController()
        self.audio_receiver = AudioReceiver()

        # Callbacks
        self.on_connected: Optional[Callable] = None
        self.on_disconnected: Optional[Callable] = None

    def disconnected_callback(self, client):
        """Handle disconnection."""
        print("Watch disconnected")
        if self.on_disconnected:
            self.on_disconnected()
        self.disconnect_event.set()

    async def _find_and_connect(self) -> bool:
        """Find device and connect."""
        try:
            print(f"Searching for {self.address}...")
            device = await BleakScanner.find_device_by_address(
                self.address,
                timeout=10.0
            )

            if not device:
                print("Watch not found")
                return False

            print(f"Found {device.name}, connecting...")
            self.client = BleakClient(
                device,
                disconnected_callback=self.disconnected_callback
            )
            await self.client.connect()

            # Subscribe to characteristics
            await self._subscribe_to_characteristics()

            self.disconnect_event.clear()

            if self.on_connected:
                self.on_connected()

            print("Connected successfully!")
            return True

        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    async def _subscribe_to_characteristics(self):
        """Subscribe to watch notifications."""

        # IMU data
        await self.client.start_notify(
            IMU_CHAR_UUID,
            self._imu_notification_handler
        )

        # Gesture events
        await self.client.start_notify(
            GESTURE_CHAR_UUID,
            self._gesture_notification_handler
        )

        # Audio data
        await self.client.start_notify(
            AUDIO_CHAR_UUID,
            self._audio_notification_handler
        )

        print("Subscribed to all characteristics")

    def _imu_notification_handler(self, sender, data: bytearray):
        """Process IMU sensor data."""
        if len(data) == 28:  # 6 floats + 1 uint32
            values = struct.unpack('<6fL', data)
            sample = IMUSample(
                accel_x=values[0],
                accel_y=values[1],
                accel_z=values[2],
                gyro_x=values[3],
                gyro_y=values[4],
                gyro_z=values[5],
                timestamp_ms=values[6]
            )

            # Update cursor from gyro
            self.cursor_controller.update_from_gyro(
                sample.gyro_x,
                sample.gyro_y
            )

    def _gesture_notification_handler(self, sender, data: bytearray):
        """Process gesture events."""
        if len(data) >= 5:  # 1 byte type + 4 byte timestamp
            gesture_type_id = data[0]
            timestamp = struct.unpack('<L', data[1:5])[0]

            gesture_map = {
                0x01: 'tap',
                0x02: 'double_tap',
                0x03: 'hold',
                0x04: 'palm_up'
            }

            gesture = gesture_map.get(gesture_type_id, 'unknown')

            # Handle gesture
            if gesture == 'tap':
                self.cursor_controller.click(Button.left)
            elif gesture == 'double_tap':
                self.cursor_controller.click(Button.left)
                self.cursor_controller.click(Button.left)
            elif gesture == 'hold':
                self.cursor_controller.click(Button.right)
            elif gesture == 'palm_up':
                self.cursor_controller.disable()
            else:
                self.cursor_controller.enable()

    def _audio_notification_handler(self, sender, data: bytearray):
        """Process audio data."""
        self.audio_receiver.process_audio_packet(bytes(data))

    async def send_command(self, command: str):
        """Send command to watch."""
        if not self.client or not self.client.is_connected:
            print("Not connected")
            return

        command_map = {
            'START_AUDIO': 0x01,
            'STOP_AUDIO': 0x02,
        }

        command_byte = command_map.get(command)
        if command_byte:
            await self.client.write_gatt_char(
                COMMAND_CHAR_UUID,
                bytes([command_byte])
            )

    async def maintain_connection(self):
        """Main loop with auto-reconnect."""
        attempt = 0

        while not self.should_stop:
            connected = await self._find_and_connect()

            if connected:
                attempt = 0

                # Wait for disconnect
                await self.disconnect_event.wait()

                if self.should_stop:
                    break

                print("Attempting to reconnect...")
            else:
                attempt += 1

            # Exponential backoff
            delay = min(2.0 * (2 ** min(attempt, 5)), 60.0)
            await asyncio.sleep(delay)

    async def stop(self):
        """Stop connection manager."""
        self.should_stop = True
        self.disconnect_event.set()

        if self.client and self.client.is_connected:
            await self.client.disconnect()


# ============================================================================
# Main Application
# ============================================================================

async def main():
    """Main application entry point."""

    print("WristControl Desktop Companion")
    print("=" * 50)

    # Create connection manager
    manager = WatchConnectionManager(WATCH_ADDRESS)

    # Set up callbacks
    def on_connected():
        print("\n✓ Watch connected - cursor control active")

    def on_disconnected():
        print("\n✗ Watch disconnected - cursor control paused")

    manager.on_connected = on_connected
    manager.on_disconnected = on_disconnected

    # Start connection manager
    try:
        await manager.maintain_connection()
    except KeyboardInterrupt:
        print("\nShutting down...")
        await manager.stop()


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 9. Key Recommendations Summary

### For Your Project (WristControl)

**1. Use Bleak for BLE**
- Cross-platform support (Windows/macOS/Linux)
- Mature, well-maintained
- Asyncio-based for non-blocking I/O

**2. Connection Management**
- Implement auto-reconnect with exponential backoff
- Use disconnect callbacks for awareness
- Consider `bleak-retry-connector` for simplicity

**3. High-Frequency Sensor Streaming (50-100Hz)**
- Request 7.5ms connection interval from watch
- Negotiate MTU to 247+ bytes
- Batch 10 samples per notification (240 bytes fits in MTU)
- Use notifications (not indications)
- Expected throughput: ~2.4 KB/s easily achievable

**4. Achieve Sub-50ms Latency**
- Use polling with `asyncio.sleep(0)` instead of notifications (20-25ms vs 180-200ms)
- Or hybrid: notification triggers polling
- Minimize processing in handlers
- Use pynput for cursor control (lowest latency)
- Monitor latency continuously

**5. Audio Streaming**
- **Use Opus codec** at 16 Kbps for voice
- Implement push-to-talk to save battery
- Or use VAD (Voice Activity Detection) for automatic activation
- Buffer 10ms chunks for encoding
- Use streaming STT for lower latency (Deepgram, Google)
- Expected total latency: ~360-500ms (acceptable for voice commands)

**6. Architecture**
- Separate concerns: Connection, Sensor Processing, Audio, Cursor Control
- Use asyncio.Queue for decoupling
- Implement latency monitoring
- Handle reconnection gracefully (don't lose state)

**7. DoublePoint TouchSDK**
- Study their Python SDK as reference: github.com/doublepointlab/touch-sdk-py
- They've solved many of the problems you'll face
- Consider building on their architecture
- WowMouse has proven this approach works at scale

---

## 10. Further Resources

### Libraries
- **Bleak:** https://github.com/hbldh/bleak
- **bleak-retry-connector:** https://pypi.org/project/bleak-retry-connector/
- **opuslib:** https://github.com/OnBeep/opuslib
- **pynput:** https://github.com/moses-palmer/pynput
- **sounddevice:** https://python-sounddevice.readthedocs.io/

### Documentation
- **BLE Spec:** https://www.bluetooth.com/specifications/specs/
- **Bleak Docs:** https://bleak.readthedocs.io/
- **DoublePoint Docs:** https://docs.doublepoint.com/
- **BLE Throughput Guide:** https://interrupt.memfault.com/blog/ble-throughput-primer
- **Connection Parameters:** https://www.btframework.com/connparams.htm

### Research Papers
- "Analysis of Latency Performance of Bluetooth Low Energy (BLE) Networks" - https://pmc.ncbi.nlm.nih.gov/articles/PMC4327007/

---

## Sources

This research report compiled information from the following sources:

**Bleak Library:**
- [GitHub - hbldh/bleak](https://github.com/hbldh/bleak)
- [bleak · PyPI](https://pypi.org/project/bleak/)
- [bleak — bleak 2.1.1 documentation](https://bleak.readthedocs.io/)
- [BleakClient class — bleak 2.1.1 documentation](https://bleak.readthedocs.io/en/latest/api/client.html)
- [Minimal Python script to list & read BLE device characteristics using Python (Bleak) | TechOverflow](https://techoverflow.net/2025/08/04/minimal-python-script-to-list-read-ble-device-characteristics-using-python-bleak/)

**Connection Management:**
- [Automatic reconnect after loss of connection · hbldh/bleak · Discussion #1158](https://github.com/hbldh/bleak/discussions/1158)
- [bleak-retry-connector · PyPI](https://pypi.org/project/bleak-retry-connector/)
- [Troubleshooting — bleak 2.1.0 documentation](https://bleak.readthedocs.io/en/latest/troubleshooting.html)

**High-Frequency Data Streaming:**
- [Bluetooth LE GATT: Python Bleak IoT Device Control 2025](https://johal.in/bluetooth-le-gatt-python-bleak-iot-device-control-2025/)
- [Simple Stream With BLE — SensiML Documentation](https://sensiml.com/documentation/simple-streaming-specification/simple-ble-streaming.html)
- [Sensor sampling rate is set at 100Hz but the receive rate is only about 20 or 30Hz — MbientLab](https://mbientlab.com/community/discussion/3567/sensor-sampling-rate-is-set-at-100hz-but-the-receive-rate-is-only-about-20-or-30hz)

**WebBLE vs Native:**
- [Web BLE Implementation: Streamlining Web Bluetooth API Connect](https://stormotion.io/blog/web-ble-implementation/)
- [Native vs. Cross-Platform Bluetooth Low Energy Mobile App Development | Novel Bits](https://novelbits.io/native-vs-cross-platform-bluetooth-low-energy-mobile-app-platforms/)
- [PWAs and Bluetooth Low Energy (BLE): A Comprehensive Guide | by Sparkleo | Medium](https://medium.com/@sparkleo/pwas-and-bluetooth-low-energy-ble-a-comprehensive-guide-9bddaa0a8d51)

**Latency Optimization:**
- [Is there a way to reduce the latency in start_notify()? · Issue #801 · hbldh/bleak](https://github.com/hbldh/bleak/issues/801)
- [Analysis of Latency Performance of Bluetooth Low Energy (BLE) Networks - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4327007/)
- [BLE Performance Optimization - circuitlabs.net](https://circuitlabs.net/ble-performance-optimization/)
- [What are the Bluetooth LE connection parameters?](https://www.btframework.com/connparams.htm)

**MTU & Throughput:**
- [A Practical Guide to BLE Throughput | Interrupt](https://interrupt.memfault.com/blog/ble-throughput-primer)
- [Maximizing BLE Throughput Part 2: Use Larger ATT MTU – Punch Through](https://punchthrough.com/maximizing-ble-throughput-part-2-use-larger-att-mtu/)
- [Maximizing BLE Throughput Part 4: Everything You Need To Know – Punch Through](https://punchthrough.com/ble-throughput-part-4/)
- [Bluetooth 5 speed: How to achieve maximum throughput for your BLE application](https://novelbits.io/bluetooth-5-speed-maximum-throughput/)

**Audio Streaming:**
- [Low-Latency Audio Processing with Python and MeetStream API](https://blog.meetstream.ai/low-latency-audio-processing-with-python-and-meetstream-api/)
- [python-rtmixer: Reliable low-latency audio playback and recording with Python](https://github.com/spatialaudio/python-rtmixer)

**Cross-Platform:**
- [Bleak | mbedded.ninja](https://blog.mbedded.ninja/programming/languages/python/bleak/)
- [Bleak: A Cross-Platform Asynchronous BLE GATT Client Library - Boardor](https://boardor.com/blog/bleak-a-cross-platform-asynchronous-ble-gatt-client-library)

**DoublePoint TouchSDK:**
- [TouchSDK | Gesture Recognition for Smartwatches](https://docs.doublepoint.com/docs/touch-sdk/)
- [WowMouse - Next Generation Touch Interface](https://docs.doublepoint.com/)
- [Doublepoint · GitHub](https://github.com/doublepointlab)

---

*End of Report*
