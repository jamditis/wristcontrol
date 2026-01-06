"""Tests for configuration management."""

import json
from pathlib import Path
import tempfile

from wristcontrol.core.config import Config, CursorConfig, GestureConfig


def test_default_config() -> None:
    """Test that default config has expected values."""
    config = Config()

    assert config.cursor.sensitivity == 1.0
    assert config.cursor.mode == "relative"
    assert config.gesture.tap_threshold == 0.5
    assert config.voice.enabled is True
    assert config.bluetooth.auto_reconnect is True


def test_config_save_and_load() -> None:
    """Test saving and loading configuration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.json"

        # Create config with custom values
        config = Config()
        config.cursor.sensitivity = 2.0
        config.voice.backend = "whisper_api"

        # Save
        config.save(config_path)

        # Load
        loaded = Config.load(config_path)

        assert loaded.cursor.sensitivity == 2.0
        assert loaded.voice.backend == "whisper_api"


def test_config_load_missing_file() -> None:
    """Test loading from non-existent file returns defaults."""
    config = Config.load(Path("/nonexistent/config.json"))

    assert config.cursor.sensitivity == 1.0
    assert config.voice.enabled is True
