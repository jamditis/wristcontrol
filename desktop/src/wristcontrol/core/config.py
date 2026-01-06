"""Configuration management for WristControl."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class CursorConfig:
    """Cursor control configuration."""

    sensitivity: float = 1.0
    dead_zone: float = 0.1
    acceleration: float = 1.5
    smoothing: float = 0.3
    mode: str = "relative"  # "relative", "absolute", or "hybrid"


@dataclass
class GestureConfig:
    """Gesture recognition configuration."""

    tap_threshold: float = 0.5
    double_tap_window_ms: int = 300
    hold_duration_ms: int = 500
    enabled_gestures: list[str] = field(
        default_factory=lambda: ["tap", "double_tap", "hold", "palm_up"]
    )


@dataclass
class VoiceConfig:
    """Voice recognition configuration."""

    enabled: bool = True
    backend: str = "whisper_local"  # "whisper_local", "whisper_api", "google"
    activation_mode: str = "push_to_talk"  # "push_to_talk", "wake_word", "always_on"
    wake_word: str = "computer"
    language: str = "en"


@dataclass
class BluetoothConfig:
    """Bluetooth connection configuration."""

    device_name: Optional[str] = None
    device_address: Optional[str] = None
    auto_reconnect: bool = True
    reconnect_interval_s: int = 5


@dataclass
class Config:
    """Main application configuration."""

    cursor: CursorConfig = field(default_factory=CursorConfig)
    gesture: GestureConfig = field(default_factory=GestureConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    bluetooth: BluetoothConfig = field(default_factory=BluetoothConfig)

    @classmethod
    def load(cls, path: Path) -> "Config":
        """Load configuration from a JSON file.

        Args:
            path: Path to the configuration file.

        Returns:
            Loaded configuration, or defaults if file doesn't exist.
        """
        if not path.exists():
            logger.info(f"Config file not found at {path}, using defaults")
            return cls()

        try:
            with open(path) as f:
                data = json.load(f)

            return cls(
                cursor=CursorConfig(**data.get("cursor", {})),
                gesture=GestureConfig(**data.get("gesture", {})),
                voice=VoiceConfig(**data.get("voice", {})),
                bluetooth=BluetoothConfig(**data.get("bluetooth", {})),
            )
        except Exception as e:
            logger.error(f"Failed to load config from {path}: {e}")
            return cls()

    def save(self, path: Path) -> None:
        """Save configuration to a JSON file.

        Args:
            path: Path to save the configuration file.
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "cursor": {
                "sensitivity": self.cursor.sensitivity,
                "dead_zone": self.cursor.dead_zone,
                "acceleration": self.cursor.acceleration,
                "smoothing": self.cursor.smoothing,
                "mode": self.cursor.mode,
            },
            "gesture": {
                "tap_threshold": self.gesture.tap_threshold,
                "double_tap_window_ms": self.gesture.double_tap_window_ms,
                "hold_duration_ms": self.gesture.hold_duration_ms,
                "enabled_gestures": self.gesture.enabled_gestures,
            },
            "voice": {
                "enabled": self.voice.enabled,
                "backend": self.voice.backend,
                "activation_mode": self.voice.activation_mode,
                "wake_word": self.voice.wake_word,
                "language": self.voice.language,
            },
            "bluetooth": {
                "device_name": self.bluetooth.device_name,
                "device_address": self.bluetooth.device_address,
                "auto_reconnect": self.bluetooth.auto_reconnect,
                "reconnect_interval_s": self.bluetooth.reconnect_interval_s,
            },
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Configuration saved to {path}")
