# Cross-Platform Desktop Application Research
## OS Input Injection for WristControl Project

*Research conducted: January 2026*

---

## 1. Input Injection Libraries

### 1.1 pynput (Recommended Primary Choice)

**Overview:** Cross-platform library for controlling and monitoring input devices (mouse and keyboard).

**Strengths:**
- Pure Python with minimal dependencies
- Works on Windows, macOS, and Linux
- Both input injection AND monitoring capabilities
- Active maintenance and good documentation
- Low-level enough for precise control, high-level enough for ease of use

**Installation:**
```bash
pip install pynput
```

**Mouse Control Examples:**

```python
from pynput.mouse import Controller, Button
import time

# Initialize mouse controller
mouse = Controller()

# Get current position
current_pos = mouse.position
print(f"Current position: {current_pos}")

# Move mouse (absolute positioning)
mouse.position = (100, 200)

# Move mouse (relative positioning)
mouse.move(10, -10)  # dx, dy

# Click operations
mouse.click(Button.left, 1)  # Left click
mouse.click(Button.right, 1)  # Right click
mouse.click(Button.left, 2)   # Double click

# Press and release (for drag operations)
mouse.press(Button.left)
mouse.move(100, 100)
mouse.release(Button.left)

# Scroll operations
mouse.scroll(0, 2)  # Horizontal, Vertical (units vary by platform)
```

**Keyboard Control Examples:**

```python
from pynput.keyboard import Controller, Key

keyboard = Controller()

# Type text
keyboard.type('Hello from wristcontrol!')

# Press special keys
keyboard.press(Key.ctrl)
keyboard.press('c')
keyboard.release('c')
keyboard.release(Key.ctrl)

# Hotkey combinations
with keyboard.pressed(Key.ctrl):
    keyboard.press('v')
    keyboard.release('v')

# Type with delays (more natural)
def type_naturally(text, delay=0.05):
    for char in text:
        keyboard.type(char)
        time.sleep(delay)
```

**Event Monitoring (useful for debugging):**

```python
from pynput import mouse, keyboard

def on_mouse_move(x, y):
    print(f"Mouse moved to ({x}, {y})")

def on_click(x, y, button, pressed):
    if pressed:
        print(f"Mouse clicked at ({x}, {y}) with {button}")

def on_key_press(key):
    print(f"Key {key} pressed")

# Mouse listener
mouse_listener = mouse.Listener(
    on_move=on_mouse_move,
    on_click=on_click
)

# Keyboard listener
keyboard_listener = keyboard.Listener(
    on_press=on_key_press
)

mouse_listener.start()
keyboard_listener.start()
```

**Platform-Specific Notes:**
- **Windows:** Uses `SendInput` API (reliable, low-latency)
- **macOS:** Requires Accessibility permissions (prompt user on first run)
- **Linux:** Works with both X11 and Wayland (X11 more reliable)

---

### 1.2 PyAutoGUI (Alternative)

**Overview:** Higher-level automation library with image recognition.

**Pros:**
- Simpler API
- Built-in image recognition (locate elements on screen)
- Failsafe features (move mouse to corner to abort)

**Cons:**
- Higher latency than pynput
- Heavier dependencies
- Less suitable for real-time control

**Example:**
```python
import pyautogui

# Screen information
screen_width, screen_height = pyautogui.size()

# Mouse control (smoother movement)
pyautogui.moveTo(100, 200, duration=0.2)  # Animated movement
pyautogui.moveRel(10, 10, duration=0.1)   # Relative

# Click with positions
pyautogui.click(100, 200)
pyautogui.doubleClick()
pyautogui.rightClick()

# Drag operations
pyautogui.dragTo(300, 400, duration=0.5)

# Keyboard
pyautogui.write('Hello World', interval=0.05)
pyautogui.press('enter')
pyautogui.hotkey('ctrl', 'c')

# Failsafe
pyautogui.FAILSAFE = True  # Move mouse to corner to raise exception
```

**Recommendation:** Use PyAutoGUI for screen automation scripts, but use pynput for real-time cursor control.

---

### 1.3 Platform-Specific APIs

#### Windows: pywin32 and ctypes

**pywin32 Example:**
```python
import win32api
import win32con

# Low-level mouse input
def move_mouse_win32(x, y):
    win32api.SetCursorPos((x, y))

def click_win32():
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

# Get cursor position
x, y = win32api.GetCursorPos()
```

**ctypes (no dependencies) Example:**
```python
import ctypes

# Windows API structures
class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))
    ]

class INPUT(ctypes.Structure):
    class _INPUT(ctypes.Union):
        _fields_ = [("mi", MOUSEINPUT)]
    _fields_ = [
        ("type", ctypes.c_ulong),
        ("value", _INPUT)
    ]

# Move mouse with SendInput (most reliable)
def send_mouse_input(x, y):
    # Convert to absolute coordinates (0-65535 range)
    screen_width = ctypes.windll.user32.GetSystemMetrics(0)
    screen_height = ctypes.windll.user32.GetSystemMetrics(1)

    abs_x = int(x * 65535 / screen_width)
    abs_y = int(y * 65535 / screen_height)

    extra = ctypes.c_ulong(0)
    ii_ = INPUT()
    ii_.type = 0  # INPUT_MOUSE
    ii_.value.mi.dx = abs_x
    ii_.value.mi.dy = abs_y
    ii_.value.mi.mouseData = 0
    ii_.value.mi.dwFlags = 0x0001 | 0x8000  # MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE
    ii_.value.mi.time = 0
    ii_.value.mi.dwExtraInfo = ctypes.pointer(extra)

    ctypes.windll.user32.SendInput(1, ctypes.byref(ii_), ctypes.sizeof(ii_))
```

#### macOS: pyobjc

**Requires Accessibility Permissions:**
```python
import Quartz

def check_accessibility_permissions():
    """Check if app has accessibility permissions on macOS"""
    return Quartz.AXIsProcessTrusted()

def request_accessibility_permissions():
    """Request accessibility permissions with prompt"""
    options = {Quartz.kAXTrustedCheckOptionPrompt: True}
    return Quartz.AXIsProcessTrustedWithOptions(options)

# Mouse control using Quartz
def move_mouse_macos(x, y):
    # Create mouse moved event
    event = Quartz.CGEventCreateMouseEvent(
        None,
        Quartz.kCGEventMouseMoved,
        (x, y),
        0
    )
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)

def click_macos(x, y, button='left'):
    if button == 'left':
        down_event = Quartz.kCGEventLeftMouseDown
        up_event = Quartz.kCGEventLeftMouseUp
    else:
        down_event = Quartz.kCGEventRightMouseDown
        up_event = Quartz.kCGEventRightMouseUp

    # Mouse down
    event = Quartz.CGEventCreateMouseEvent(None, down_event, (x, y), 0)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)

    # Mouse up
    event = Quartz.CGEventCreateMouseEvent(None, up_event, (x, y), 0)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)

# Example: Request permissions on startup
if not check_accessibility_permissions():
    print("Requesting accessibility permissions...")
    request_accessibility_permissions()
    print("Please grant permissions in System Preferences > Security & Privacy > Privacy > Accessibility")
```

#### Linux: python-xlib and uinput

**X11 with python-xlib:**
```python
from Xlib import X, display
from Xlib.ext.xtest import fake_input

# Initialize display
d = display.Display()
screen = d.screen()
root = screen.root

# Move mouse (X11)
def move_mouse_x11(x, y):
    root.warp_pointer(x, y)
    d.sync()

# Click
def click_x11(button=1):  # 1=left, 2=middle, 3=right
    fake_input(d, X.ButtonPress, button)
    d.sync()
    fake_input(d, X.ButtonRelease, button)
    d.sync()

# Get current pointer position
def get_pointer_position():
    pointer = root.query_pointer()
    return pointer.root_x, pointer.root_y
```

**uinput (works on Wayland):**
```python
import uinput

# Create virtual mouse device
device = uinput.Device([
    uinput.BTN_LEFT,
    uinput.BTN_RIGHT,
    uinput.REL_X,
    uinput.REL_Y,
    uinput.REL_WHEEL
])

# Move mouse (relative)
device.emit(uinput.REL_X, 10)
device.emit(uinput.REL_Y, 10)

# Click
device.emit(uinput.BTN_LEFT, 1)  # Press
device.emit(uinput.BTN_LEFT, 0)  # Release

# Scroll
device.emit(uinput.REL_WHEEL, 1)  # Scroll up
```

**Wayland Detection:**
```python
import os

def is_wayland():
    return os.environ.get('XDG_SESSION_TYPE') == 'wayland'

def is_x11():
    return os.environ.get('XDG_SESSION_TYPE') == 'x11'
```

---

### 1.4 Recommended Input Injection Architecture

```python
"""
Cross-platform input injection abstraction layer
"""
import platform
from abc import ABC, abstractmethod
from typing import Tuple

class InputController(ABC):
    """Abstract base class for platform-specific input controllers"""

    @abstractmethod
    def move_mouse(self, x: int, y: int):
        """Move mouse to absolute position"""
        pass

    @abstractmethod
    def move_mouse_relative(self, dx: int, dy: int):
        """Move mouse relative to current position"""
        pass

    @abstractmethod
    def click(self, button: str = 'left', count: int = 1):
        """Perform mouse click"""
        pass

    @abstractmethod
    def scroll(self, dx: int, dy: int):
        """Scroll horizontally and vertically"""
        pass

    @abstractmethod
    def type_text(self, text: str):
        """Type text"""
        pass

    @abstractmethod
    def press_key(self, key: str):
        """Press a special key"""
        pass

    @abstractmethod
    def get_position(self) -> Tuple[int, int]:
        """Get current mouse position"""
        pass


class PynputController(InputController):
    """Cross-platform controller using pynput (recommended)"""

    def __init__(self):
        from pynput.mouse import Controller as MouseController, Button
        from pynput.keyboard import Controller as KeyboardController, Key

        self.mouse = MouseController()
        self.keyboard = KeyboardController()
        self.Button = Button
        self.Key = Key

    def move_mouse(self, x: int, y: int):
        self.mouse.position = (x, y)

    def move_mouse_relative(self, dx: int, dy: int):
        self.mouse.move(dx, dy)

    def click(self, button: str = 'left', count: int = 1):
        btn = self.Button.left if button == 'left' else self.Button.right
        self.mouse.click(btn, count)

    def scroll(self, dx: int, dy: int):
        self.mouse.scroll(dx, dy)

    def type_text(self, text: str):
        self.keyboard.type(text)

    def press_key(self, key: str):
        # Map common keys
        key_map = {
            'enter': self.Key.enter,
            'tab': self.Key.tab,
            'escape': self.Key.esc,
            'backspace': self.Key.backspace,
            'delete': self.Key.delete,
        }
        k = key_map.get(key.lower(), key)
        self.keyboard.press(k)
        self.keyboard.release(k)

    def get_position(self) -> Tuple[int, int]:
        return self.mouse.position


class WindowsNativeController(InputController):
    """Windows-specific optimized controller using ctypes"""

    def __init__(self):
        import ctypes
        self.user32 = ctypes.windll.user32

    # Implementation using ctypes as shown above
    # ... (omitted for brevity, use ctypes examples above)


class MacOSNativeController(InputController):
    """macOS-specific controller using Quartz"""

    def __init__(self):
        import Quartz
        self.Quartz = Quartz

        # Check permissions on init
        if not self.Quartz.AXIsProcessTrusted():
            options = {self.Quartz.kAXTrustedCheckOptionPrompt: True}
            self.Quartz.AXIsProcessTrustedWithOptions(options)

    # Implementation using Quartz as shown above
    # ... (omitted for brevity)


def create_input_controller(use_native: bool = False) -> InputController:
    """Factory function to create appropriate input controller"""

    if use_native:
        system = platform.system()
        if system == 'Windows':
            return WindowsNativeController()
        elif system == 'Darwin':
            return MacOSNativeController()
        # Fall through to pynput for Linux or if native not available

    return PynputController()


# Usage
if __name__ == '__main__':
    controller = create_input_controller()

    # Move mouse
    controller.move_mouse(100, 100)

    # Click
    controller.click('left')

    # Type
    controller.type_text('Hello from WristControl!')
```

---

## 2. Desktop App Frameworks

### 2.1 Python with System Tray (pystray)

**Best for:** Background service with minimal UI

**Installation:**
```bash
pip install pystray pillow
```

**Basic System Tray Example:**

```python
import pystray
from pystray import MenuItem as item
from PIL import Image, ImageDraw
import threading

class WristControlTrayApp:
    def __init__(self):
        self.icon = None
        self.enabled = False

    def create_icon_image(self):
        """Create a simple icon (replace with actual icon file)"""
        width = 64
        height = 64
        color1 = "blue"
        color2 = "white"

        image = Image.new('RGB', (width, height), color1)
        dc = ImageDraw.Draw(image)
        dc.rectangle(
            [(width // 4, height // 4), (width * 3 // 4, height * 3 // 4)],
            fill=color2
        )
        return image

    def toggle_control(self, icon, item):
        """Toggle wrist control on/off"""
        self.enabled = not self.enabled
        status = "Enabled" if self.enabled else "Disabled"
        print(f"WristControl {status}")
        # Update icon to show status
        icon.notify(f"WristControl {status}")

    def open_settings(self, icon, item):
        """Open settings window"""
        print("Opening settings...")
        # Launch settings UI (Qt window, web UI, etc.)

    def quit_app(self, icon, item):
        """Quit the application"""
        print("Quitting WristControl...")
        icon.stop()

    def create_menu(self):
        """Create system tray menu"""
        return pystray.Menu(
            item('Toggle Control', self.toggle_control, default=True, checked=lambda item: self.enabled),
            item('Settings', self.open_settings),
            pystray.Menu.SEPARATOR,
            item('Quit', self.quit_app)
        )

    def run(self):
        """Run the system tray application"""
        image = self.create_icon_image()
        self.icon = pystray.Icon(
            "wristcontrol",
            image,
            "WristControl",
            self.create_menu()
        )

        # Start background services (BLE, sensor processing) in separate thread
        service_thread = threading.Thread(target=self.start_background_services, daemon=True)
        service_thread.start()

        # Run icon (blocks)
        self.icon.run()

    def start_background_services(self):
        """Start BLE listener, sensor processing, etc."""
        print("Starting background services...")
        # Initialize BLE connection
        # Start sensor data processing loop
        # etc.


if __name__ == '__main__':
    app = WristControlTrayApp()
    app.run()
```

**Advanced: Dynamic Icon Updates:**

```python
def update_icon_for_status(self, status: str):
    """Update tray icon based on connection status"""
    if status == 'connected':
        # Green icon
        image = self.create_colored_icon('green')
    elif status == 'disconnected':
        # Red icon
        image = self.create_colored_icon('red')
    else:
        # Gray icon
        image = self.create_colored_icon('gray')

    self.icon.icon = image
```

---

### 2.2 Qt/PyQt for Configuration UI

**Best for:** Native-looking cross-platform UI

**Installation:**
```bash
pip install PyQt6
# or
pip install PySide6  # Official Qt for Python
```

**Settings Window Example:**

```python
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QPushButton, QComboBox, QCheckBox, QTabWidget,
    QGroupBox, QSpinBox
)
from PyQt6.QtCore import Qt, QTimer
import sys

class SettingsWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('WristControl Settings')
        self.setGeometry(100, 100, 600, 400)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        # Tab widget
        tabs = QTabWidget()
        layout.addWidget(tabs)

        # Add tabs
        tabs.addTab(self.create_sensitivity_tab(), "Sensitivity")
        tabs.addTab(self.create_gestures_tab(), "Gestures")
        tabs.addTab(self.create_connection_tab(), "Connection")
        tabs.addTab(self.create_profiles_tab(), "Profiles")

        # Status bar
        self.statusBar().showMessage('Ready')

    def create_sensitivity_tab(self):
        """Create sensitivity settings tab"""
        widget = QWidget()
        layout = QVBoxLayout()

        # Mouse sensitivity
        mouse_group = QGroupBox("Mouse Sensitivity")
        mouse_layout = QVBoxLayout()

        # X sensitivity
        x_layout = QHBoxLayout()
        x_layout.addWidget(QLabel("X Sensitivity:"))
        x_slider = QSlider(Qt.Orientation.Horizontal)
        x_slider.setMinimum(1)
        x_slider.setMaximum(100)
        x_slider.setValue(50)
        x_slider.valueChanged.connect(lambda v: self.on_sensitivity_change('x', v))
        x_layout.addWidget(x_slider)
        x_value = QLabel("50")
        x_slider.valueChanged.connect(lambda v: x_value.setText(str(v)))
        x_layout.addWidget(x_value)
        mouse_layout.addLayout(x_layout)

        # Y sensitivity
        y_layout = QHBoxLayout()
        y_layout.addWidget(QLabel("Y Sensitivity:"))
        y_slider = QSlider(Qt.Orientation.Horizontal)
        y_slider.setMinimum(1)
        y_slider.setMaximum(100)
        y_slider.setValue(50)
        y_slider.valueChanged.connect(lambda v: self.on_sensitivity_change('y', v))
        y_layout.addWidget(y_slider)
        y_value = QLabel("50")
        y_slider.valueChanged.connect(lambda v: y_value.setText(str(v)))
        y_layout.addWidget(y_value)
        mouse_layout.addLayout(y_layout)

        # Dead zone
        deadzone_layout = QHBoxLayout()
        deadzone_layout.addWidget(QLabel("Dead Zone:"))
        deadzone_slider = QSlider(Qt.Orientation.Horizontal)
        deadzone_slider.setMinimum(0)
        deadzone_slider.setMaximum(50)
        deadzone_slider.setValue(5)
        deadzone_layout.addWidget(deadzone_slider)
        deadzone_value = QLabel("5")
        deadzone_slider.valueChanged.connect(lambda v: deadzone_value.setText(str(v)))
        deadzone_layout.addWidget(deadzone_value)
        mouse_layout.addLayout(deadzone_layout)

        # Smoothing
        smoothing_layout = QHBoxLayout()
        smoothing_layout.addWidget(QLabel("Smoothing:"))
        smoothing_slider = QSlider(Qt.Orientation.Horizontal)
        smoothing_slider.setMinimum(0)
        smoothing_slider.setMaximum(100)
        smoothing_slider.setValue(30)
        smoothing_layout.addWidget(smoothing_slider)
        smoothing_value = QLabel("30")
        smoothing_slider.valueChanged.connect(lambda v: smoothing_value.setText(str(v)))
        smoothing_layout.addWidget(smoothing_value)
        mouse_layout.addLayout(smoothing_layout)

        mouse_group.setLayout(mouse_layout)
        layout.addWidget(mouse_group)

        # Acceleration curve
        accel_group = QGroupBox("Acceleration Curve")
        accel_layout = QVBoxLayout()
        accel_combo = QComboBox()
        accel_combo.addItems(["Linear", "Exponential", "Custom"])
        accel_layout.addWidget(accel_combo)
        accel_group.setLayout(accel_layout)
        layout.addWidget(accel_group)

        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def create_gestures_tab(self):
        """Create gesture customization tab"""
        widget = QWidget()
        layout = QVBoxLayout()

        # Gesture mappings
        gestures = [
            ("Tap", ["Left Click", "Right Click", "Middle Click"]),
            ("Double Tap", ["Double Click", "Open", "Custom"]),
            ("Hold", ["Right Click", "Drag", "Menu"]),
            ("Wrist Flick Up", ["Scroll Up", "Page Up", "Volume Up"]),
            ("Wrist Flick Down", ["Scroll Down", "Page Down", "Volume Down"]),
        ]

        for gesture_name, actions in gestures:
            h_layout = QHBoxLayout()
            h_layout.addWidget(QLabel(f"{gesture_name}:"))
            combo = QComboBox()
            combo.addItems(actions)
            h_layout.addWidget(combo)
            layout.addLayout(h_layout)

        # Enable/disable gestures
        layout.addWidget(QLabel("\nEnabled Gestures:"))
        for gesture in ["Finger Tap", "Wrist Rotation", "Arm Movement", "Palm Up/Down"]:
            checkbox = QCheckBox(gesture)
            checkbox.setChecked(True)
            layout.addWidget(checkbox)

        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def create_connection_tab(self):
        """Create connection settings tab"""
        widget = QWidget()
        layout = QVBoxLayout()

        # Device selection
        device_group = QGroupBox("Watch Connection")
        device_layout = QVBoxLayout()

        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel("Device:"))
        device_combo = QComboBox()
        device_combo.addItems(["Scanning...", "Samsung Galaxy Watch 5", "Samsung Galaxy Watch 6"])
        h_layout.addWidget(device_combo)
        scan_btn = QPushButton("Scan")
        scan_btn.clicked.connect(self.scan_devices)
        h_layout.addWidget(scan_btn)
        device_layout.addLayout(h_layout)

        connect_btn = QPushButton("Connect")
        connect_btn.clicked.connect(self.connect_device)
        device_layout.addWidget(connect_btn)

        # Connection status
        self.connection_status = QLabel("Status: Disconnected")
        device_layout.addWidget(self.connection_status)

        device_group.setLayout(device_layout)
        layout.addWidget(device_group)

        # Connection options
        options_group = QGroupBox("Connection Options")
        options_layout = QVBoxLayout()

        auto_connect = QCheckBox("Auto-connect on startup")
        auto_connect.setChecked(True)
        options_layout.addWidget(auto_connect)

        auto_reconnect = QCheckBox("Auto-reconnect if disconnected")
        auto_reconnect.setChecked(True)
        options_layout.addWidget(auto_reconnect)

        options_group.setLayout(options_layout)
        layout.addWidget(options_group)

        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def create_profiles_tab(self):
        """Create profile management tab"""
        widget = QWidget()
        layout = QVBoxLayout()

        layout.addWidget(QLabel("Application-Specific Profiles:"))

        # Profile list
        profiles = ["Default", "Web Browser", "Code Editor", "Design Tools"]
        for profile in profiles:
            h_layout = QHBoxLayout()
            h_layout.addWidget(QLabel(profile))
            edit_btn = QPushButton("Edit")
            h_layout.addWidget(edit_btn)
            delete_btn = QPushButton("Delete")
            h_layout.addWidget(delete_btn)
            layout.addLayout(h_layout)

        # Add profile button
        add_btn = QPushButton("Add New Profile")
        layout.addWidget(add_btn)

        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def on_sensitivity_change(self, axis, value):
        """Handle sensitivity slider changes"""
        print(f"Sensitivity {axis}: {value}")
        # Update actual sensitivity in controller

    def scan_devices(self):
        """Scan for BLE devices"""
        self.connection_status.setText("Status: Scanning...")
        print("Scanning for devices...")
        # Implement BLE scanning

    def connect_device(self):
        """Connect to selected device"""
        self.connection_status.setText("Status: Connecting...")
        print("Connecting to device...")
        # Implement BLE connection


def launch_settings_ui():
    """Launch settings UI (can be called from tray app)"""
    app = QApplication(sys.argv)
    window = SettingsWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    launch_settings_ui()
```

---

### 2.3 Integrated Architecture: Tray + Qt Settings

```python
"""
Complete desktop app combining system tray and Qt settings UI
"""
import sys
import threading
from PyQt6.QtWidgets import QApplication
from pystray import Icon, Menu, MenuItem as item
from PIL import Image

class WristControlApp:
    def __init__(self):
        self.qt_app = None
        self.settings_window = None
        self.tray_icon = None
        self.enabled = False

        # Initialize Qt application
        self.qt_app = QApplication(sys.argv)

    def create_tray_icon(self):
        """Create system tray icon"""
        image = self.create_icon_image()

        menu = Menu(
            item('Enable/Disable', self.toggle_control, checked=lambda item: self.enabled),
            item('Settings', self.show_settings),
            Menu.SEPARATOR,
            item('Quit', self.quit_app)
        )

        self.tray_icon = Icon("wristcontrol", image, "WristControl", menu)

    def create_icon_image(self):
        # Create icon (64x64)
        image = Image.new('RGB', (64, 64), color='blue')
        return image

    def toggle_control(self):
        self.enabled = not self.enabled
        status = "Enabled" if self.enabled else "Disabled"
        if self.tray_icon:
            self.tray_icon.notify(f"WristControl {status}")

    def show_settings(self):
        """Show settings window"""
        if self.settings_window is None:
            from settings_window import SettingsWindow  # Import your settings window
            self.settings_window = SettingsWindow()

        self.settings_window.show()
        self.settings_window.raise_()
        self.settings_window.activateWindow()

    def quit_app(self):
        """Quit the application"""
        if self.tray_icon:
            self.tray_icon.stop()
        if self.qt_app:
            self.qt_app.quit()

    def run(self):
        """Run the application"""
        # Start tray icon in separate thread
        tray_thread = threading.Thread(target=self._run_tray, daemon=True)
        tray_thread.start()

        # Start Qt event loop (main thread)
        sys.exit(self.qt_app.exec())

    def _run_tray(self):
        """Run tray icon (in separate thread)"""
        self.create_tray_icon()
        self.tray_icon.run()


if __name__ == '__main__':
    app = WristControlApp()
    app.run()
```

---

### 2.4 Alternative: Web-Based UI (Flask + WebSockets)

**For lightweight browser-based configuration:**

```python
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import webview  # PyWebView for native window

app = Flask(__name__)
app.config['SECRET_KEY'] = 'wristcontrol-secret'
socketio = SocketIO(app)

@app.route('/')
def index():
    return render_template('settings.html')

@socketio.on('update_sensitivity')
def handle_sensitivity(data):
    print(f"Sensitivity updated: {data}")
    # Update actual controller
    emit('response', {'status': 'ok'})

@socketio.on('connect_device')
def handle_connect():
    print("Connecting to device...")
    emit('connection_status', {'status': 'connecting'})
    # Perform connection
    emit('connection_status', {'status': 'connected'})

def run_ui():
    # Option 1: Browser-based
    socketio.run(app, host='127.0.0.1', port=5000)

    # Option 2: Native window with PyWebView
    # webview.create_window('WristControl Settings', 'http://127.0.0.1:5000')
    # webview.start()
```

**HTML/JavaScript Frontend (settings.html):**
```html
<!DOCTYPE html>
<html>
<head>
    <title>WristControl Settings</title>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
</head>
<body>
    <h1>WristControl Settings</h1>

    <div>
        <label>X Sensitivity: <span id="x-value">50</span></label>
        <input type="range" id="x-sensitivity" min="1" max="100" value="50">
    </div>

    <div>
        <label>Y Sensitivity: <span id="y-value">50</span></label>
        <input type="range" id="y-sensitivity" min="1" max="100" value="50">
    </div>

    <button id="connect-btn">Connect Device</button>
    <div id="status">Status: Disconnected</div>

    <script>
        const socket = io();

        document.getElementById('x-sensitivity').addEventListener('change', (e) => {
            const value = e.target.value;
            document.getElementById('x-value').textContent = value;
            socket.emit('update_sensitivity', { axis: 'x', value: value });
        });

        document.getElementById('connect-btn').addEventListener('click', () => {
            socket.emit('connect_device');
        });

        socket.on('connection_status', (data) => {
            document.getElementById('status').textContent = `Status: ${data.status}`;
        });
    </script>
</body>
</html>
```

---

## 3. Cross-Platform Considerations

### 3.1 Platform Detection and Adaptation

```python
import platform
import os

class PlatformInfo:
    @staticmethod
    def get_system():
        """Get operating system"""
        return platform.system()  # 'Windows', 'Darwin' (macOS), 'Linux'

    @staticmethod
    def is_windows():
        return platform.system() == 'Windows'

    @staticmethod
    def is_macos():
        return platform.system() == 'Darwin'

    @staticmethod
    def is_linux():
        return platform.system() == 'Linux'

    @staticmethod
    def get_linux_display_server():
        """Detect X11 vs Wayland on Linux"""
        if not PlatformInfo.is_linux():
            return None

        session_type = os.environ.get('XDG_SESSION_TYPE', '').lower()
        if session_type:
            return session_type  # 'wayland' or 'x11'

        # Fallback: check for Wayland display
        if os.environ.get('WAYLAND_DISPLAY'):
            return 'wayland'
        if os.environ.get('DISPLAY'):
            return 'x11'

        return 'unknown'

    @staticmethod
    def requires_admin():
        """Check if platform requires admin for input injection"""
        if PlatformInfo.is_linux():
            # uinput requires root or uinput group membership
            return not os.access('/dev/uinput', os.W_OK)
        return False

    @staticmethod
    def check_permissions():
        """Check if app has necessary permissions"""
        if PlatformInfo.is_macos():
            import Quartz
            return Quartz.AXIsProcessTrusted()
        elif PlatformInfo.is_linux():
            # Check for X11 access or uinput access
            display_server = PlatformInfo.get_linux_display_server()
            if display_server == 'x11':
                return os.environ.get('DISPLAY') is not None
            elif display_server == 'wayland':
                # Check uinput access
                return os.access('/dev/uinput', os.W_OK)

        return True  # Windows doesn't need special permissions

    @staticmethod
    def request_permissions():
        """Request necessary permissions"""
        if PlatformInfo.is_macos():
            import Quartz
            options = {Quartz.kAXTrustedCheckOptionPrompt: True}
            Quartz.AXIsProcessTrustedWithOptions(options)
            return "Please grant Accessibility permissions in System Preferences"

        elif PlatformInfo.is_linux():
            if PlatformInfo.requires_admin():
                return "Please add your user to 'input' group: sudo usermod -a -G input $USER"

        return "No additional permissions required"
```

### 3.2 Windows-Specific Considerations

```python
"""Windows-specific optimizations and considerations"""
import ctypes
import sys

def is_admin():
    """Check if running with administrator privileges"""
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

def request_admin():
    """Request UAC elevation (restart app as admin)"""
    if not is_admin():
        # Re-run the program with admin rights
        ctypes.windll.shell32.ShellExecuteW(
            None, "runas", sys.executable, " ".join(sys.argv), None, 1
        )
        sys.exit()

# High-precision timer (important for low latency)
def enable_high_precision_timer():
    """Request 1ms timer resolution on Windows"""
    try:
        import ctypes.wintypes
        winmm = ctypes.WinDLL('winmm')
        winmm.timeBeginPeriod(1)  # 1ms resolution
        return True
    except:
        return False

# Prevent sleep during operation
def prevent_sleep():
    """Prevent Windows from sleeping during active control"""
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    ES_DISPLAY_REQUIRED = 0x00000002

    ctypes.windll.kernel32.SetThreadExecutionState(
        ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
    )

def allow_sleep():
    """Allow Windows to sleep again"""
    ES_CONTINUOUS = 0x80000000
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
```

### 3.3 macOS-Specific Considerations

```python
"""macOS-specific considerations"""
import subprocess
import os

def check_macos_accessibility():
    """Check and request accessibility permissions"""
    import Quartz

    if not Quartz.AXIsProcessTrusted():
        print("Accessibility permissions required!")
        print("Opening System Preferences...")

        # Open System Preferences to Privacy & Security
        subprocess.run([
            'open',
            'x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility'
        ])

        # Prompt for permissions
        options = {Quartz.kAXTrustedCheckOptionPrompt: True}
        Quartz.AXIsProcessTrustedWithOptions(options)

        return False

    return True

def get_macos_version():
    """Get macOS version"""
    import platform
    return platform.mac_ver()[0]

def is_macos_ventura_or_later():
    """Check if running on macOS 13+ (Ventura or later)"""
    version = get_macos_version()
    major = int(version.split('.')[0])
    return major >= 13

# App bundle considerations
def is_bundled_app():
    """Check if running as .app bundle"""
    return getattr(sys, 'frozen', False)

def get_bundle_path():
    """Get path to .app bundle"""
    if is_bundled_app():
        return os.path.dirname(sys.executable)
    return None
```

### 3.4 Linux-Specific Considerations

```python
"""Linux-specific considerations (X11 vs Wayland)"""
import os
import subprocess

def setup_uinput_permissions():
    """Setup uinput permissions (requires manual intervention)"""
    print("To use WristControl on Linux, you need uinput access.")
    print("\nOption 1: Add user to input group (recommended)")
    print("  sudo usermod -a -G input $USER")
    print("  Then log out and log back in")
    print("\nOption 2: Load uinput module")
    print("  sudo modprobe uinput")
    print("  sudo chmod +0666 /dev/uinput")

def check_x11_available():
    """Check if X11 is available"""
    return os.environ.get('DISPLAY') is not None

def check_wayland_available():
    """Check if Wayland is available"""
    return os.environ.get('WAYLAND_DISPLAY') is not None

def get_desktop_environment():
    """Detect desktop environment"""
    desktop = os.environ.get('XDG_CURRENT_DESKTOP', '').lower()
    if 'gnome' in desktop:
        return 'gnome'
    elif 'kde' in desktop or 'plasma' in desktop:
        return 'kde'
    elif 'xfce' in desktop:
        return 'xfce'
    else:
        return desktop

# X11-specific input method
class X11InputMethod:
    def __init__(self):
        from Xlib import display
        self.display = display.Display()
        self.screen = self.display.screen()
        self.root = self.screen.root

    def get_active_window(self):
        """Get currently active window (for per-app profiles)"""
        window_id = self.root.get_full_property(
            self.display.intern_atom('_NET_ACTIVE_WINDOW'),
            0
        ).value[0]
        return self.display.create_resource_object('window', window_id)

    def get_window_name(self, window):
        """Get window name/title"""
        try:
            return window.get_wm_name()
        except:
            return None

# Wayland-specific considerations
def wayland_warning():
    """Warn about Wayland limitations"""
    print("WARNING: Running on Wayland")
    print("Some features may have limitations:")
    print("  - Getting cursor position may not work")
    print("  - Per-window detection may not work")
    print("Consider switching to X11 session for full functionality")
```

---

## 4. Configuration Management

### 4.1 Configuration File Structure

```python
"""
Configuration management with JSON persistence
"""
import json
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List

@dataclass
class SensitivityConfig:
    x_sensitivity: float = 50.0
    y_sensitivity: float = 50.0
    dead_zone: float = 5.0
    smoothing: float = 30.0
    acceleration_curve: str = "linear"

@dataclass
class GestureMapping:
    tap: str = "left_click"
    double_tap: str = "double_click"
    hold: str = "right_click"
    wrist_flick_up: str = "scroll_up"
    wrist_flick_down: str = "scroll_down"
    palm_up: str = "disable"

@dataclass
class ConnectionConfig:
    device_address: str = ""
    auto_connect: bool = True
    auto_reconnect: bool = True
    connection_timeout: int = 10

@dataclass
class VoiceConfig:
    enabled: bool = True
    activation_mode: str = "push_to_talk"  # or "always_on"
    wake_word: str = "computer"
    stt_provider: str = "local"  # or "cloud"
    cloud_api_key: str = ""

@dataclass
class Profile:
    name: str
    sensitivity: SensitivityConfig
    gestures: GestureMapping

    # Application-specific triggers
    app_names: List[str] = None

    def __post_init__(self):
        if self.app_names is None:
            self.app_names = []

class ConfigManager:
    def __init__(self, config_dir: str = None):
        if config_dir is None:
            # Use platform-appropriate config directory
            if os.name == 'nt':  # Windows
                config_dir = os.path.join(os.environ['APPDATA'], 'WristControl')
            else:  # macOS, Linux
                config_dir = os.path.join(Path.home(), '.config', 'wristcontrol')

        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.config_file = self.config_dir / 'config.json'

        # Default configuration
        self.sensitivity = SensitivityConfig()
        self.gestures = GestureMapping()
        self.connection = ConnectionConfig()
        self.voice = VoiceConfig()
        self.profiles: Dict[str, Profile] = {}
        self.active_profile = "default"

        # Load existing config
        self.load()

    def load(self):
        """Load configuration from file"""
        if not self.config_file.exists():
            self.save()  # Create default config
            return

        try:
            with open(self.config_file, 'r') as f:
                data = json.load(f)

            # Load sections
            if 'sensitivity' in data:
                self.sensitivity = SensitivityConfig(**data['sensitivity'])
            if 'gestures' in data:
                self.gestures = GestureMapping(**data['gestures'])
            if 'connection' in data:
                self.connection = ConnectionConfig(**data['connection'])
            if 'voice' in data:
                self.voice = VoiceConfig(**data['voice'])
            if 'profiles' in data:
                for name, profile_data in data['profiles'].items():
                    self.profiles[name] = Profile(
                        name=name,
                        sensitivity=SensitivityConfig(**profile_data['sensitivity']),
                        gestures=GestureMapping(**profile_data['gestures']),
                        app_names=profile_data.get('app_names', [])
                    )
            if 'active_profile' in data:
                self.active_profile = data['active_profile']

            print(f"Configuration loaded from {self.config_file}")

        except Exception as e:
            print(f"Error loading config: {e}")
            print("Using default configuration")

    def save(self):
        """Save configuration to file"""
        data = {
            'sensitivity': asdict(self.sensitivity),
            'gestures': asdict(self.gestures),
            'connection': asdict(self.connection),
            'voice': asdict(self.voice),
            'profiles': {
                name: {
                    'sensitivity': asdict(profile.sensitivity),
                    'gestures': asdict(profile.gestures),
                    'app_names': profile.app_names
                }
                for name, profile in self.profiles.items()
            },
            'active_profile': self.active_profile
        }

        with open(self.config_file, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Configuration saved to {self.config_file}")

    def create_profile(self, name: str, app_names: List[str] = None):
        """Create a new profile"""
        profile = Profile(
            name=name,
            sensitivity=SensitivityConfig(),  # Use current or default
            gestures=GestureMapping(),
            app_names=app_names or []
        )
        self.profiles[name] = profile
        self.save()
        return profile

    def switch_profile(self, name: str):
        """Switch to a different profile"""
        if name in self.profiles:
            self.active_profile = name
            profile = self.profiles[name]
            # Apply profile settings
            self.sensitivity = profile.sensitivity
            self.gestures = profile.gestures
            self.save()
            return True
        return False

    def get_profile_for_app(self, app_name: str):
        """Get appropriate profile for application"""
        for name, profile in self.profiles.items():
            if app_name.lower() in [app.lower() for app in profile.app_names]:
                return profile
        return None


# Usage example
if __name__ == '__main__':
    config = ConfigManager()

    # Modify settings
    config.sensitivity.x_sensitivity = 75.0
    config.gestures.tap = "custom_action"
    config.save()

    # Create application-specific profile
    browser_profile = config.create_profile(
        "browser",
        app_names=["chrome", "firefox", "safari", "edge"]
    )
    browser_profile.sensitivity.x_sensitivity = 60.0
    config.save()
```

---

## 5. Performance Optimization

### 5.1 Low-Latency Architecture

```python
"""
High-performance event-driven architecture for low-latency input injection
"""
import threading
import queue
import time
from dataclasses import dataclass
from typing import Callable, Optional
import collections

@dataclass
class SensorData:
    timestamp: float
    accel_x: float
    accel_y: float
    accel_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float

@dataclass
class GestureEvent:
    timestamp: float
    gesture_type: str  # 'tap', 'double_tap', 'hold', etc.
    data: dict

class HighPerformanceController:
    def __init__(self, input_controller):
        self.input_controller = input_controller

        # High-priority queues for real-time processing
        self.sensor_queue = queue.Queue(maxsize=100)
        self.gesture_queue = queue.Queue(maxsize=50)

        # Circular buffer for smoothing (last N samples)
        self.smoothing_buffer_size = 5
        self.position_buffer_x = collections.deque(maxlen=self.smoothing_buffer_size)
        self.position_buffer_y = collections.deque(maxlen=self.smoothing_buffer_size)

        # Processing threads
        self.running = False
        self.sensor_thread: Optional[threading.Thread] = None
        self.gesture_thread: Optional[threading.Thread] = None

        # Performance metrics
        self.latency_samples = collections.deque(maxlen=1000)
        self.fps_counter = 0
        self.last_fps_time = time.time()

    def start(self):
        """Start processing threads"""
        self.running = True

        # High-priority sensor processing thread
        self.sensor_thread = threading.Thread(
            target=self._process_sensors,
            daemon=True,
            name="SensorProcessor"
        )
        self.sensor_thread.start()

        # Gesture event thread
        self.gesture_thread = threading.Thread(
            target=self._process_gestures,
            daemon=True,
            name="GestureProcessor"
        )
        self.gesture_thread.start()

        print("High-performance controller started")

    def stop(self):
        """Stop processing threads"""
        self.running = False
        if self.sensor_thread:
            self.sensor_thread.join(timeout=1.0)
        if self.gesture_thread:
            self.gesture_thread.join(timeout=1.0)

    def on_sensor_data(self, data: SensorData):
        """Called when new sensor data arrives (from BLE)"""
        try:
            self.sensor_queue.put_nowait(data)
        except queue.Full:
            # Drop oldest sample if queue full (prefer latest data)
            try:
                self.sensor_queue.get_nowait()
                self.sensor_queue.put_nowait(data)
            except:
                pass

    def on_gesture_event(self, event: GestureEvent):
        """Called when gesture detected"""
        try:
            self.gesture_queue.put_nowait(event)
        except queue.Full:
            print("Warning: Gesture queue full, dropping event")

    def _process_sensors(self):
        """Process sensor data and update cursor (high-priority loop)"""
        while self.running:
            try:
                # Get sensor data with minimal blocking
                data = self.sensor_queue.get(timeout=0.001)

                start_time = time.time()

                # Convert sensor data to cursor movement
                dx, dy = self._sensor_to_cursor_delta(data)

                # Apply smoothing
                dx_smooth, dy_smooth = self._apply_smoothing(dx, dy)

                # Move cursor
                self.input_controller.move_mouse_relative(
                    int(dx_smooth),
                    int(dy_smooth)
                )

                # Track latency
                latency = (time.time() - start_time) * 1000  # ms
                self.latency_samples.append(latency)

                # FPS counter
                self.fps_counter += 1
                if time.time() - self.last_fps_time >= 1.0:
                    print(f"Sensor processing: {self.fps_counter} FPS, "
                          f"Avg latency: {sum(self.latency_samples)/len(self.latency_samples):.2f}ms")
                    self.fps_counter = 0
                    self.last_fps_time = time.time()

            except queue.Empty:
                # No data, continue
                time.sleep(0.0001)  # Minimal sleep to prevent CPU spinning
            except Exception as e:
                print(f"Error in sensor processing: {e}")

    def _process_gestures(self):
        """Process gesture events (lower priority)"""
        while self.running:
            try:
                event = self.gesture_queue.get(timeout=0.1)
                self._handle_gesture(event)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in gesture processing: {e}")

    def _sensor_to_cursor_delta(self, data: SensorData) -> tuple:
        """Convert sensor data to cursor movement delta"""
        # Simple example: use gyroscope for cursor control
        # In reality, you'd use more sophisticated algorithm

        sensitivity = 2.0
        dx = data.gyro_y * sensitivity
        dy = -data.gyro_x * sensitivity  # Inverted for natural movement

        return dx, dy

    def _apply_smoothing(self, dx: float, dy: float) -> tuple:
        """Apply exponential moving average smoothing"""
        self.position_buffer_x.append(dx)
        self.position_buffer_y.append(dy)

        # Average over buffer
        dx_smooth = sum(self.position_buffer_x) / len(self.position_buffer_x)
        dy_smooth = sum(self.position_buffer_y) / len(self.position_buffer_y)

        return dx_smooth, dy_smooth

    def _handle_gesture(self, event: GestureEvent):
        """Handle gesture event"""
        gesture_type = event.gesture_type

        if gesture_type == 'tap':
            self.input_controller.click('left', 1)
        elif gesture_type == 'double_tap':
            self.input_controller.click('left', 2)
        elif gesture_type == 'hold':
            self.input_controller.click('right', 1)
        elif gesture_type == 'wrist_flick_up':
            self.input_controller.scroll(0, 3)
        elif gesture_type == 'wrist_flick_down':
            self.input_controller.scroll(0, -3)

        print(f"Gesture: {gesture_type}")


# Advanced: Motion Algorithm with Kalman Filter
import numpy as np

class KalmanFilter:
    """1D Kalman filter for smoothing cursor movement"""

    def __init__(self, process_variance=1e-5, measurement_variance=1e-1):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimate = 0.0
        self.estimate_error = 1.0

    def update(self, measurement):
        """Update filter with new measurement"""
        # Prediction
        prediction = self.estimate
        prediction_error = self.estimate_error + self.process_variance

        # Update
        kalman_gain = prediction_error / (prediction_error + self.measurement_variance)
        self.estimate = prediction + kalman_gain * (measurement - prediction)
        self.estimate_error = (1 - kalman_gain) * prediction_error

        return self.estimate

class AdvancedMotionAlgorithm:
    """Advanced motion-to-cursor algorithm with filtering"""

    def __init__(self):
        # Separate Kalman filters for X and Y
        self.kalman_x = KalmanFilter()
        self.kalman_y = KalmanFilter()

        # Complementary filter for sensor fusion
        self.alpha = 0.98  # Gyro trust factor

        # State
        self.angle_x = 0.0
        self.angle_y = 0.0
        self.last_time = time.time()

    def process(self, sensor_data: SensorData) -> tuple:
        """Process sensor data and return cursor delta"""
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time

        # Integrate gyroscope (high-frequency, drifts over time)
        gyro_angle_x = self.angle_x + sensor_data.gyro_x * dt
        gyro_angle_y = self.angle_y + sensor_data.gyro_y * dt

        # Get angle from accelerometer (low-frequency, no drift)
        accel_angle_x = np.arctan2(sensor_data.accel_y, sensor_data.accel_z)
        accel_angle_y = np.arctan2(sensor_data.accel_x, sensor_data.accel_z)

        # Complementary filter: combine gyro and accel
        self.angle_x = self.alpha * gyro_angle_x + (1 - self.alpha) * accel_angle_x
        self.angle_y = self.alpha * gyro_angle_y + (1 - self.alpha) * accel_angle_y

        # Convert angles to cursor velocity
        velocity_x = self.angle_y * 100  # Scale factor
        velocity_y = -self.angle_x * 100

        # Apply Kalman filtering
        smooth_x = self.kalman_x.update(velocity_x)
        smooth_y = self.kalman_y.update(velocity_y)

        return smooth_x, smooth_y
```

### 5.2 Multi-threading Best Practices

```python
"""
Thread management for sensor processing
"""
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

class ThreadManager:
    """Manage threads for different components"""

    def __init__(self):
        # Thread pool for background tasks
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="worker")

        # Critical threads (pinned to cores if possible)
        self.sensor_thread = None
        self.ble_thread = None
        self.ui_thread = None

    def set_thread_priority(self, thread, priority='high'):
        """Set thread priority (platform-specific)"""
        import platform

        if platform.system() == 'Windows':
            import win32api
            import win32process
            import win32con

            if priority == 'high':
                win32process.SetThreadPriority(
                    win32api.GetCurrentThread(),
                    win32con.THREAD_PRIORITY_HIGHEST
                )
        elif platform.system() == 'Linux':
            import os
            # Set nice value (requires privileges)
            try:
                os.nice(-10)  # Higher priority
            except PermissionError:
                pass

    def start_sensor_processing(self, callback):
        """Start high-priority sensor processing thread"""
        def sensor_loop():
            self.set_thread_priority(threading.current_thread(), 'high')
            while True:
                callback()

        self.sensor_thread = threading.Thread(
            target=sensor_loop,
            daemon=True,
            name="SensorProcessing"
        )
        self.sensor_thread.start()
```

---

## 6. Complete Application Architecture

```python
"""
Complete WristControl Desktop Application Architecture
Combines all components into cohesive system
"""

from dataclasses import dataclass
from typing import Optional
import threading
import time

# Imports from previous sections
from input_controller import create_input_controller
from config_manager import ConfigManager
from high_performance import HighPerformanceController
# from ble_connection import BLEManager  # To be implemented
# from voice_processor import VoiceProcessor  # To be implemented

class WristControlApplication:
    """Main application class"""

    def __init__(self):
        # Core components
        self.config = ConfigManager()
        self.input_controller = create_input_controller()
        self.performance_controller = HighPerformanceController(self.input_controller)

        # Connection components
        self.ble_manager = None  # BLEManager()
        self.voice_processor = None  # VoiceProcessor()

        # UI components
        self.tray_app = None
        self.settings_window = None

        # State
        self.enabled = False
        self.connected = False

        # Threads
        self.main_thread = None
        self.running = False

    def initialize(self):
        """Initialize all components"""
        print("Initializing WristControl...")

        # Check platform permissions
        from platform_info import PlatformInfo
        if not PlatformInfo.check_permissions():
            print(PlatformInfo.request_permissions())
            return False

        # Initialize BLE
        # self.ble_manager.initialize()
        # self.ble_manager.on_sensor_data = self.performance_controller.on_sensor_data
        # self.ble_manager.on_gesture_event = self.performance_controller.on_gesture_event

        # Initialize voice
        # self.voice_processor.initialize()

        # Start performance controller
        self.performance_controller.start()

        print("WristControl initialized")
        return True

    def start(self):
        """Start the application"""
        if not self.initialize():
            return

        self.running = True

        # Start main processing loop
        self.main_thread = threading.Thread(target=self._main_loop, daemon=True)
        self.main_thread.start()

        # Start UI (blocks until quit)
        self._start_ui()

    def stop(self):
        """Stop the application"""
        print("Stopping WristControl...")
        self.running = False
        self.performance_controller.stop()
        # self.ble_manager.disconnect()
        # self.voice_processor.stop()

    def _main_loop(self):
        """Main processing loop"""
        while self.running:
            try:
                # Check connection status
                # if self.ble_manager.is_connected():
                #     if not self.connected:
                #         self.on_connected()
                # else:
                #     if self.connected:
                #         self.on_disconnected()

                # Process any pending tasks
                time.sleep(0.1)

            except Exception as e:
                print(f"Error in main loop: {e}")

    def _start_ui(self):
        """Start user interface"""
        # Start system tray
        from tray_app import WristControlTrayApp
        self.tray_app = WristControlTrayApp()
        self.tray_app.on_toggle = self.toggle_control
        self.tray_app.on_settings = self.show_settings
        self.tray_app.on_quit = self.stop
        self.tray_app.run()  # Blocks

    def toggle_control(self):
        """Toggle input control on/off"""
        self.enabled = not self.enabled
        print(f"Control {'enabled' if self.enabled else 'disabled'}")

    def show_settings(self):
        """Show settings window"""
        if self.settings_window is None:
            from settings_window import SettingsWindow
            self.settings_window = SettingsWindow()
            self.settings_window.on_config_change = self.on_config_change

        self.settings_window.show()

    def on_config_change(self, config):
        """Handle configuration changes"""
        self.config = config
        self.config.save()
        # Apply changes to controllers

    def on_connected(self):
        """Handle device connection"""
        self.connected = True
        print("Device connected")
        if self.tray_app:
            self.tray_app.update_status("connected")

    def on_disconnected(self):
        """Handle device disconnection"""
        self.connected = False
        print("Device disconnected")
        if self.tray_app:
            self.tray_app.update_status("disconnected")


# Entry point
def main():
    app = WristControlApplication()
    app.start()

if __name__ == '__main__':
    main()
```

---

## 7. Recommended Technology Stack Summary

### Core Stack (Recommended)
- **Language**: Python 3.9+
- **Input Injection**: pynput (primary), platform-specific APIs (optimization)
- **Desktop Framework**: pystray (system tray) + PyQt6 (settings UI)
- **BLE Communication**: bleak (Python BLE library)
- **Configuration**: JSON files with dataclasses
- **Performance**: Multi-threading with queue-based architecture

### Platform-Specific Dependencies
- **Windows**: pywin32 (optional, for optimizations)
- **macOS**: pyobjc (for Accessibility APIs)
- **Linux**: python-xlib (X11) or uinput (Wayland)

### Additional Libraries
- **Sensor Processing**: numpy, scipy (for Kalman filter, signal processing)
- **Voice**: OpenAI Whisper (local), or cloud APIs
- **Logging**: Python logging module
- **Testing**: pytest

### Development Tools
- **Packaging**: PyInstaller (create standalone executables)
- **GUI Designer**: Qt Designer (for PyQt layouts)
- **Profiling**: cProfile, line_profiler (performance optimization)

---

## 8. Next Steps for Implementation

1. **Start with Input Injection**: Implement and test cursor control with pynput
2. **Build Configuration System**: Set up ConfigManager with JSON persistence
3. **Create Basic UI**: System tray + simple settings window
4. **Simulate Sensor Input**: Test motion algorithms with simulated data
5. **Optimize Performance**: Implement multi-threading and measure latency
6. **Platform Testing**: Test on Windows, macOS, and Linux
7. **Add BLE Integration**: Connect to watch (separate phase)
8. **Voice Integration**: Add STT pipeline (separate phase)

This architecture provides a solid foundation for the WristControl desktop companion application with low-latency input injection, cross-platform support, and room for future enhancements.
