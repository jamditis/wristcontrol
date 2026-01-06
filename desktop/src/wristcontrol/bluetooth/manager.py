"""Bluetooth Low Energy connection manager."""

import asyncio
import logging
from typing import Callable, Optional

from bleak import BleakClient, BleakScanner
from bleak.backends.device import BLEDevice

logger = logging.getLogger(__name__)


class BluetoothManager:
    """Manages BLE connection to the smartwatch."""

    # UUIDs for WristControl BLE services (to be defined in watch app)
    SERVICE_UUID = "0000fff0-0000-1000-8000-00805f9b34fb"
    SENSOR_CHAR_UUID = "0000fff1-0000-1000-8000-00805f9b34fb"
    GESTURE_CHAR_UUID = "0000fff2-0000-1000-8000-00805f9b34fb"
    AUDIO_CHAR_UUID = "0000fff3-0000-1000-8000-00805f9b34fb"

    def __init__(
        self,
        device_name: Optional[str] = None,
        device_address: Optional[str] = None,
        auto_reconnect: bool = True,
    ) -> None:
        """Initialize the Bluetooth manager.

        Args:
            device_name: Name of the watch to connect to.
            device_address: MAC address of the watch to connect to.
            auto_reconnect: Whether to automatically reconnect on disconnect.
        """
        self.device_name = device_name
        self.device_address = device_address
        self.auto_reconnect = auto_reconnect

        self._client: Optional[BleakClient] = None
        self._connected = False
        self._sensor_callback: Optional[Callable[[bytes], None]] = None
        self._gesture_callback: Optional[Callable[[bytes], None]] = None
        self._audio_callback: Optional[Callable[[bytes], None]] = None

    @property
    def is_connected(self) -> bool:
        """Check if currently connected to a watch."""
        return self._connected and self._client is not None

    async def scan_for_devices(self, timeout: float = 10.0) -> list[BLEDevice]:
        """Scan for available BLE devices.

        Args:
            timeout: Scan timeout in seconds.

        Returns:
            List of discovered BLE devices.
        """
        logger.info(f"Scanning for BLE devices (timeout: {timeout}s)...")
        devices = await BleakScanner.discover(timeout=timeout)
        logger.info(f"Found {len(devices)} devices")
        return devices

    async def connect(self, device: Optional[BLEDevice] = None) -> bool:
        """Connect to a watch.

        Args:
            device: Specific device to connect to. If None, uses configured address/name.

        Returns:
            True if connection successful, False otherwise.
        """
        if device is None and self.device_address:
            logger.info(f"Connecting to device at {self.device_address}...")
            self._client = BleakClient(self.device_address)
        elif device is not None:
            logger.info(f"Connecting to {device.name} ({device.address})...")
            self._client = BleakClient(device.address)
        else:
            logger.error("No device specified for connection")
            return False

        try:
            await self._client.connect()
            self._connected = True
            logger.info("Connected successfully")

            # Set up disconnect callback
            self._client.set_disconnected_callback(self._on_disconnect)

            return True
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self._connected = False
            return False

    async def disconnect(self) -> None:
        """Disconnect from the watch."""
        if self._client and self._connected:
            await self._client.disconnect()
            self._connected = False
            logger.info("Disconnected from watch")

    def _on_disconnect(self, client: BleakClient) -> None:
        """Handle disconnection event."""
        logger.warning("Watch disconnected")
        self._connected = False

        if self.auto_reconnect:
            logger.info("Auto-reconnect enabled, will attempt to reconnect...")
            # Reconnection will be handled by the main app loop

    def set_sensor_callback(self, callback: Callable[[bytes], None]) -> None:
        """Set callback for sensor data notifications.

        Args:
            callback: Function to call with raw sensor data bytes.
        """
        self._sensor_callback = callback

    def set_gesture_callback(self, callback: Callable[[bytes], None]) -> None:
        """Set callback for gesture event notifications.

        Args:
            callback: Function to call with raw gesture event bytes.
        """
        self._gesture_callback = callback

    def set_audio_callback(self, callback: Callable[[bytes], None]) -> None:
        """Set callback for audio data notifications.

        Args:
            callback: Function to call with raw audio data bytes.
        """
        self._audio_callback = callback

    async def start_notifications(self) -> None:
        """Start receiving notifications from the watch."""
        if not self._client or not self._connected:
            logger.error("Cannot start notifications: not connected")
            return

        # Subscribe to sensor data
        if self._sensor_callback:
            await self._client.start_notify(
                self.SENSOR_CHAR_UUID, self._handle_sensor_notification
            )
            logger.info("Subscribed to sensor data")

        # Subscribe to gesture events
        if self._gesture_callback:
            await self._client.start_notify(
                self.GESTURE_CHAR_UUID, self._handle_gesture_notification
            )
            logger.info("Subscribed to gesture events")

        # Subscribe to audio data
        if self._audio_callback:
            await self._client.start_notify(
                self.AUDIO_CHAR_UUID, self._handle_audio_notification
            )
            logger.info("Subscribed to audio data")

    def _handle_sensor_notification(
        self, sender: int, data: bytearray
    ) -> None:
        """Handle incoming sensor data notification."""
        if self._sensor_callback:
            self._sensor_callback(bytes(data))

    def _handle_gesture_notification(
        self, sender: int, data: bytearray
    ) -> None:
        """Handle incoming gesture event notification."""
        if self._gesture_callback:
            self._gesture_callback(bytes(data))

    def _handle_audio_notification(
        self, sender: int, data: bytearray
    ) -> None:
        """Handle incoming audio data notification."""
        if self._audio_callback:
            self._audio_callback(bytes(data))
