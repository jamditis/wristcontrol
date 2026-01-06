# WristControl Desktop Companion

Desktop companion application for the WristControl system. Receives sensor data and gesture events from a Samsung Galaxy Watch via Bluetooth Low Energy, and translates them into cursor movements and keyboard/mouse actions.

## Features

- **Gesture Recognition**: Finger pinch (tap, double-tap, hold), palm orientation detection
- **Voice Commands**: Speech-to-text for mouse actions and text input
- **Cursor Control**: Motion-based cursor movement using accelerometer/gyroscope data
- **Configurable**: Sensitivity, dead zones, acceleration curves, and gesture mappings

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# For cloud STT support
pip install -e ".[cloud]"
```

## Usage

```bash
# Run the application
wristcontrol

# Or run directly
python -m wristcontrol.main
```

## Configuration

Configuration is stored in `~/.config/wristcontrol/config.json`. You can modify settings through the configuration UI or by editing the JSON file directly.

### Cursor Settings

- `sensitivity`: Movement sensitivity multiplier (default: 1.0)
- `dead_zone`: Minimum tilt to register movement (default: 0.1)
- `acceleration`: Acceleration curve exponent (default: 1.5)
- `smoothing`: Movement smoothing factor (default: 0.3)
- `mode`: Control mode - "relative", "absolute", or "hybrid"

### Voice Settings

- `enabled`: Enable/disable voice processing
- `backend`: STT backend - "whisper_local", "whisper_api", or "google"
- `activation_mode`: "push_to_talk", "wake_word", or "always_on"
- `wake_word`: Wake word for voice activation (default: "computer")

## Development

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=wristcontrol

# Type checking
mypy src/wristcontrol

# Linting
ruff check src/wristcontrol
```

## Architecture

```
wristcontrol/
├── core/           # Application core and configuration
├── bluetooth/      # BLE connection and protocol handling
├── gestures/       # Gesture event processing
├── voice/          # Speech-to-text processing
├── cursor/         # Cursor control and input injection
└── ui/             # User interface components
```

## License

MIT License
