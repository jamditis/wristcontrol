"""Main application class for WristControl."""

import asyncio
import logging
from typing import Optional

from wristcontrol.core.config import Config

logger = logging.getLogger(__name__)


class WristControlApp:
    """Main application class that orchestrates all components."""

    def __init__(self, config: Optional[Config] = None) -> None:
        """Initialize the WristControl application.

        Args:
            config: Application configuration. Uses defaults if not provided.
        """
        self.config = config or Config()
        self._running = False
        self._bluetooth_manager: Optional[object] = None
        self._gesture_processor: Optional[object] = None
        self._voice_processor: Optional[object] = None
        self._cursor_controller: Optional[object] = None

    async def run(self) -> None:
        """Run the main application loop."""
        logger.info("Initializing WristControl components...")
        self._running = True

        # TODO: Initialize components
        # - BluetoothManager for watch connection
        # - GestureProcessor for gesture recognition
        # - VoiceProcessor for speech-to-text
        # - CursorController for mouse/keyboard input

        logger.info("WristControl ready. Waiting for watch connection...")

        while self._running:
            # Main event loop - will be replaced with actual event handling
            await asyncio.sleep(1)

    def stop(self) -> None:
        """Stop the application gracefully."""
        logger.info("Stopping WristControl...")
        self._running = False

        # TODO: Cleanup components
        # - Disconnect from watch
        # - Stop voice processing
        # - Release input devices
