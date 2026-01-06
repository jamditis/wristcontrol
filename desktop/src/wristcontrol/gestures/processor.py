"""Gesture processing and action mapping."""

import logging
from typing import Callable, Optional

from wristcontrol.bluetooth.protocol import GestureEvent, GestureType

logger = logging.getLogger(__name__)


class GestureProcessor:
    """Processes gesture events and triggers corresponding actions."""

    def __init__(self) -> None:
        """Initialize the gesture processor."""
        self._action_map: dict[GestureType, Callable[[], None]] = {}
        self._enabled = True

    def register_action(
        self, gesture_type: GestureType, action: Callable[[], None]
    ) -> None:
        """Register an action to be triggered by a gesture.

        Args:
            gesture_type: The gesture that triggers the action.
            action: The function to call when the gesture is detected.
        """
        self._action_map[gesture_type] = action
        logger.debug(f"Registered action for {gesture_type.name}")

    def process_event(self, event: GestureEvent) -> None:
        """Process a gesture event and trigger the corresponding action.

        Args:
            event: The gesture event to process.
        """
        if not self._enabled:
            return

        logger.debug(
            f"Gesture: {event.gesture_type.name} "
            f"(confidence: {event.confidence:.2f})"
        )

        action = self._action_map.get(event.gesture_type)
        if action:
            try:
                action()
            except Exception as e:
                logger.error(f"Error executing action for {event.gesture_type.name}: {e}")
        else:
            logger.debug(f"No action registered for {event.gesture_type.name}")

    def enable(self) -> None:
        """Enable gesture processing."""
        self._enabled = True
        logger.info("Gesture processing enabled")

    def disable(self) -> None:
        """Disable gesture processing."""
        self._enabled = False
        logger.info("Gesture processing disabled")
