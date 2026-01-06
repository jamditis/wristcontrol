"""Bluetooth Low Energy communication with watch."""

from wristcontrol.bluetooth.manager import BluetoothManager
from wristcontrol.bluetooth.protocol import WatchProtocol

__all__ = ["BluetoothManager", "WatchProtocol"]
