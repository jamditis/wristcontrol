# WristControl

Voice and gesture computer control using Samsung Galaxy Watch.

## Overview

WristControl transforms a Samsung Galaxy Watch into an input device for hands-free computer control through gestures and voice commands. Control cursor, click, scroll, and type without touching keyboard or mouse.

## Key features

- Voice integration with local speech-to-text option
- Motion-based cursor control
- Privacy focused (local processing)
- Open source and extensible

## Directory structure

```
wristcontrol/
├── watch/            # Galaxy Watch app (Tizen/Wear OS)
├── desktop/          # Desktop companion app
├── web/              # Web interface
├── planning/         # Design documents
├── research/         # Research notes
├── assets/           # Images and logos
└── requirements.txt  # Python dependencies
```

## Tech stack

- **Watch**: Tizen/Wear OS SDK
- **Desktop**: Python
- **Communication**: WebSocket/Bluetooth

---

## Multi-machine workflow

This repo is developed across multiple machines. GitHub is the source of truth.

**Before switching machines:**
```bash
git add . && git commit -m "WIP" && git push
```

**After switching machines:**
```bash
git pull
pip install -r requirements.txt  # For desktop app
```
