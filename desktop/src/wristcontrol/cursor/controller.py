"""Cursor control and OS input injection."""

import logging
from typing import Optional

from pynput.mouse import Button, Controller as MouseController
from pynput.keyboard import Key, Controller as KeyboardController

from wristcontrol.bluetooth.protocol import SensorData

logger = logging.getLogger(__name__)


class CursorController:
    """Controls cursor movement and input injection.

    Translates sensor data from the watch into cursor movements
    and handles click/keyboard actions.
    """

    def __init__(
        self,
        sensitivity: float = 1.0,
        dead_zone: float = 0.1,
        acceleration: float = 1.5,
        smoothing: float = 0.3,
        mode: str = "relative",
    ) -> None:
        """Initialize the cursor controller.

        Args:
            sensitivity: Movement sensitivity multiplier.
            dead_zone: Minimum tilt angle to register movement (radians).
            acceleration: Acceleration curve exponent.
            smoothing: Smoothing factor (0-1, higher = more smoothing).
            mode: Control mode ("relative", "absolute", or "hybrid").
        """
        self.sensitivity = sensitivity
        self.dead_zone = dead_zone
        self.acceleration = acceleration
        self.smoothing = smoothing
        self.mode = mode

        self._mouse = MouseController()
        self._keyboard = KeyboardController()
        self._enabled = False

        # State for smoothing
        self._last_dx: float = 0.0
        self._last_dy: float = 0.0

        # Reference orientation for relative mode
        self._ref_orientation: Optional[tuple[float, float, float, float]] = None

    def enable(self) -> None:
        """Enable cursor control."""
        self._enabled = True
        self._ref_orientation = None
        logger.info("Cursor control enabled")

    def disable(self) -> None:
        """Disable cursor control."""
        self._enabled = False
        logger.info("Cursor control disabled")

    def calibrate(self, sensor_data: SensorData) -> None:
        """Set the current orientation as the reference (neutral) position.

        Args:
            sensor_data: Current sensor reading to use as reference.
        """
        self._ref_orientation = (
            sensor_data.orientation_w,
            sensor_data.orientation_x,
            sensor_data.orientation_y,
            sensor_data.orientation_z,
        )
        logger.info("Cursor calibrated to current position")

    def process_sensor_data(self, sensor_data: SensorData) -> None:
        """Process sensor data and update cursor position.

        Args:
            sensor_data: Sensor data from the watch.
        """
        if not self._enabled:
            return

        if self.mode == "relative":
            self._process_relative(sensor_data)
        elif self.mode == "absolute":
            self._process_absolute(sensor_data)
        else:
            # Hybrid mode - to be implemented
            self._process_relative(sensor_data)

    def _process_relative(self, sensor_data: SensorData) -> None:
        """Process sensor data in relative (tilt-to-move) mode.

        Uses gyroscope data for smooth, relative cursor movement.
        """
        # Use gyroscope for relative movement (rad/s -> pixels)
        # Y rotation = horizontal cursor movement
        # X rotation = vertical cursor movement
        raw_dx = -sensor_data.gyro_y * self.sensitivity * 10
        raw_dy = sensor_data.gyro_x * self.sensitivity * 10

        # Apply dead zone
        if abs(raw_dx) < self.dead_zone:
            raw_dx = 0
        if abs(raw_dy) < self.dead_zone:
            raw_dy = 0

        # Apply acceleration curve
        if raw_dx != 0:
            raw_dx = (abs(raw_dx) ** self.acceleration) * (1 if raw_dx > 0 else -1)
        if raw_dy != 0:
            raw_dy = (abs(raw_dy) ** self.acceleration) * (1 if raw_dy > 0 else -1)

        # Apply smoothing (exponential moving average)
        dx = self._last_dx * self.smoothing + raw_dx * (1 - self.smoothing)
        dy = self._last_dy * self.smoothing + raw_dy * (1 - self.smoothing)

        self._last_dx = dx
        self._last_dy = dy

        # Move cursor
        if abs(dx) > 0.1 or abs(dy) > 0.1:
            self._mouse.move(int(dx), int(dy))

    def _process_absolute(self, sensor_data: SensorData) -> None:
        """Process sensor data in absolute (point-to-position) mode.

        Maps orientation to screen position.
        """
        # TODO: Implement absolute positioning using orientation quaternion
        # This requires screen dimension awareness and more complex mapping
        pass

    # Mouse actions

    def left_click(self) -> None:
        """Perform a left mouse click."""
        self._mouse.click(Button.left)
        logger.debug("Left click")

    def right_click(self) -> None:
        """Perform a right mouse click."""
        self._mouse.click(Button.right)
        logger.debug("Right click")

    def double_click(self) -> None:
        """Perform a double left click."""
        self._mouse.click(Button.left, 2)
        logger.debug("Double click")

    def mouse_down(self) -> None:
        """Press and hold left mouse button."""
        self._mouse.press(Button.left)
        logger.debug("Mouse down")

    def mouse_up(self) -> None:
        """Release left mouse button."""
        self._mouse.release(Button.left)
        logger.debug("Mouse up")

    def scroll_up(self, amount: int = 3) -> None:
        """Scroll up.

        Args:
            amount: Number of scroll units.
        """
        self._mouse.scroll(0, amount)
        logger.debug(f"Scroll up {amount}")

    def scroll_down(self, amount: int = 3) -> None:
        """Scroll down.

        Args:
            amount: Number of scroll units.
        """
        self._mouse.scroll(0, -amount)
        logger.debug(f"Scroll down {amount}")

    def scroll_left(self, amount: int = 3) -> None:
        """Scroll left.

        Args:
            amount: Number of scroll units.
        """
        self._mouse.scroll(-amount, 0)
        logger.debug(f"Scroll left {amount}")

    def scroll_right(self, amount: int = 3) -> None:
        """Scroll right.

        Args:
            amount: Number of scroll units.
        """
        self._mouse.scroll(amount, 0)
        logger.debug(f"Scroll right {amount}")

    # Keyboard actions

    def type_text(self, text: str) -> None:
        """Type text using the keyboard.

        Args:
            text: Text to type.
        """
        self._keyboard.type(text)
        logger.debug(f"Typed: {text[:20]}...")

    def press_key(self, key: Key) -> None:
        """Press a special key.

        Args:
            key: The key to press.
        """
        self._keyboard.press(key)
        self._keyboard.release(key)

    def key_escape(self) -> None:
        """Press Escape key."""
        self.press_key(Key.esc)

    def key_enter(self) -> None:
        """Press Enter key."""
        self.press_key(Key.enter)

    def key_tab(self) -> None:
        """Press Tab key."""
        self.press_key(Key.tab)

    def key_backspace(self) -> None:
        """Press Backspace key."""
        self.press_key(Key.backspace)

    def key_delete(self) -> None:
        """Press Delete key."""
        self.press_key(Key.delete)

    # Keyboard shortcuts

    def copy(self) -> None:
        """Press Ctrl+C (copy)."""
        with self._keyboard.pressed(Key.ctrl):
            self._keyboard.press("c")
            self._keyboard.release("c")
        logger.debug("Copy")

    def paste(self) -> None:
        """Press Ctrl+V (paste)."""
        with self._keyboard.pressed(Key.ctrl):
            self._keyboard.press("v")
            self._keyboard.release("v")
        logger.debug("Paste")

    def cut(self) -> None:
        """Press Ctrl+X (cut)."""
        with self._keyboard.pressed(Key.ctrl):
            self._keyboard.press("x")
            self._keyboard.release("x")
        logger.debug("Cut")

    def undo(self) -> None:
        """Press Ctrl+Z (undo)."""
        with self._keyboard.pressed(Key.ctrl):
            self._keyboard.press("z")
            self._keyboard.release("z")
        logger.debug("Undo")

    def redo(self) -> None:
        """Press Ctrl+Y (redo)."""
        with self._keyboard.pressed(Key.ctrl):
            self._keyboard.press("y")
            self._keyboard.release("y")
        logger.debug("Redo")

    def select_all(self) -> None:
        """Press Ctrl+A (select all)."""
        with self._keyboard.pressed(Key.ctrl):
            self._keyboard.press("a")
            self._keyboard.release("a")
        logger.debug("Select all")
