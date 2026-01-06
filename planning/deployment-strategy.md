# WristControl Deployment Strategy
## Voice and Gesture Computer Control System

**Version:** 1.0
**Date:** January 2026
**Project:** WristControl - Samsung Galaxy Watch Companion System

---

## Table of Contents

1. [Deployment Overview](#1-deployment-overview)
2. [Desktop Application Distribution](#2-desktop-application-distribution)
3. [Watch Application Distribution](#3-watch-application-distribution)
4. [Build & Release Pipeline](#4-build--release-pipeline)
5. [Version Management](#5-version-management)
6. [Update Mechanism](#6-update-mechanism)
7. [Installation Guide](#7-installation-guide)
8. [Configuration & Onboarding](#8-configuration--onboarding)
9. [Monitoring & Analytics](#9-monitoring--analytics)
10. [Rollback & Recovery](#10-rollback--recovery)

---

## 1. Deployment Overview

### 1.1 Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     DISTRIBUTION CHANNELS                        │
├─────────────────┬───────────────────┬───────────────────────────┤
│   Desktop App   │    Watch App      │      Documentation        │
├─────────────────┼───────────────────┼───────────────────────────┤
│ GitHub Releases │ Google Play Store │ GitHub Pages / Docs       │
│ Direct Download │ Samsung Galaxy    │ In-app Help               │
│ Package Managers│ Store             │                           │
│ (brew, scoop)   │                   │                           │
└─────────────────┴───────────────────┴───────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        BUILD PIPELINE                            │
├─────────────────────────────────────────────────────────────────┤
│  GitHub Actions → Build → Test → Sign → Package → Release       │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Platform Support Matrix

| Platform | Version | Architecture | Status |
|----------|---------|--------------|--------|
| Windows | 10, 11 | x64, ARM64 | Primary |
| macOS | 12+ (Monterey) | Intel, Apple Silicon | Primary |
| Linux | Ubuntu 22.04+ | x64 | Primary |
| Linux | Fedora 38+ | x64 | Secondary |
| WearOS | 3.0+ (API 30+) | ARM | Required |

### 1.3 Deployment Goals

| Goal | Target | Metric |
|------|--------|--------|
| Easy Installation | < 5 minutes | Time from download to working |
| Zero Configuration | Works out of box | % users with default config |
| Automatic Updates | Background updates | % users on latest version |
| Minimal Footprint | < 200MB total | Disk space |
| Fast Startup | < 3 seconds | Time to connected state |

---

## 2. Desktop Application Distribution

### 2.1 Packaging Strategy

#### Windows

**Primary: Installer (MSI/NSIS)**
```yaml
# Using cx_Freeze or PyInstaller + NSIS

build-windows:
  - Bundle Python runtime (embedded)
  - Include all dependencies
  - Create single-file installer
  - Sign with code signing certificate
  - Output: WristControl-Setup-{version}.exe
```

**Secondary: Portable ZIP**
```yaml
# For users who prefer no installation
portable-windows:
  - Bundle as self-contained directory
  - Include run.bat launcher
  - Output: WristControl-{version}-win64-portable.zip
```

**Package Manager: Scoop**
```json
// scoop/wristcontrol.json
{
    "version": "1.0.0",
    "description": "Voice and gesture computer control",
    "homepage": "https://github.com/wristcontrol/wristcontrol",
    "license": "MIT",
    "url": "https://github.com/wristcontrol/releases/download/v1.0.0/WristControl-1.0.0-win64-portable.zip",
    "hash": "sha256:...",
    "bin": "WristControl.exe",
    "shortcuts": [
        ["WristControl.exe", "WristControl"]
    ]
}
```

#### macOS

**Primary: DMG with App Bundle**
```yaml
build-macos:
  - Create .app bundle with py2app
  - Universal binary (Intel + Apple Silicon)
  - Code sign with Developer ID
  - Notarize with Apple
  - Package in DMG
  - Output: WristControl-{version}-macos-universal.dmg
```

**Package Manager: Homebrew**
```ruby
# Formula/wristcontrol.rb
class Wristcontrol < Formula
  desc "Voice and gesture computer control using smartwatch"
  homepage "https://github.com/wristcontrol/wristcontrol"
  url "https://github.com/wristcontrol/releases/download/v1.0.0/WristControl-1.0.0-macos-universal.tar.gz"
  sha256 "..."
  version "1.0.0"

  depends_on "python@3.11"

  def install
    bin.install "WristControl"
  end

  service do
    run [opt_bin/"WristControl", "--background"]
    keep_alive true
  end
end
```

#### Linux

**Primary: AppImage**
```yaml
build-linux-appimage:
  - Bundle with linuxdeploy
  - Include all libraries
  - Create self-contained AppImage
  - Output: WristControl-{version}-x86_64.AppImage
```

**Secondary: DEB/RPM**
```yaml
build-linux-deb:
  - Create Debian package
  - Include systemd service file
  - Desktop entry file
  - Output: wristcontrol_{version}_amd64.deb

build-linux-rpm:
  - Create RPM package
  - Output: wristcontrol-{version}.x86_64.rpm
```

**Package Managers:**
```yaml
# Snap (Ubuntu)
snap-package:
  - Build with snapcraft
  - Publish to Snap Store
  - Auto-updates enabled

# Flatpak
flatpak-package:
  - Build with flatpak-builder
  - Publish to Flathub
```

### 2.2 Code Signing

#### Windows Code Signing
```yaml
# GitHub Actions workflow
windows-sign:
  steps:
    - name: Sign executable
      env:
        CERTIFICATE_BASE64: ${{ secrets.WINDOWS_CERTIFICATE }}
        CERTIFICATE_PASSWORD: ${{ secrets.WINDOWS_CERT_PASSWORD }}
      run: |
        echo $CERTIFICATE_BASE64 | base64 -d > certificate.pfx
        signtool sign /f certificate.pfx /p $CERTIFICATE_PASSWORD /tr http://timestamp.digicert.com /td sha256 /fd sha256 WristControl.exe
```

#### macOS Code Signing & Notarization
```yaml
macos-sign:
  steps:
    - name: Import certificate
      env:
        CERTIFICATE_BASE64: ${{ secrets.APPLE_CERTIFICATE }}
        CERTIFICATE_PASSWORD: ${{ secrets.APPLE_CERT_PASSWORD }}
      run: |
        security create-keychain -p "" build.keychain
        echo $CERTIFICATE_BASE64 | base64 -d > certificate.p12
        security import certificate.p12 -k build.keychain -P $CERTIFICATE_PASSWORD -T /usr/bin/codesign

    - name: Sign app
      run: |
        codesign --force --deep --sign "Developer ID Application: Your Name" WristControl.app

    - name: Notarize
      run: |
        xcrun notarytool submit WristControl.dmg --apple-id $APPLE_ID --password $APPLE_APP_PASSWORD --team-id $TEAM_ID --wait
```

### 2.3 Release Artifacts

| Platform | Artifact | Size (est.) | Signed |
|----------|----------|-------------|--------|
| Windows | .exe installer | ~80MB | Yes |
| Windows | .zip portable | ~75MB | Yes |
| macOS | .dmg | ~90MB | Yes (notarized) |
| Linux | .AppImage | ~85MB | GPG |
| Linux | .deb | ~70MB | GPG |
| Linux | .rpm | ~70MB | GPG |

---

## 3. Watch Application Distribution

### 3.1 Google Play Store

**Store Listing:**
```yaml
app-listing:
  title: "WristControl - Cursor & Voice"
  short_description: "Control your computer with wrist gestures and voice"
  full_description: |
    WristControl turns your Galaxy Watch into a hands-free computer controller.

    Features:
    • Cursor control via wrist movements
    • Click and scroll with finger gestures
    • Voice commands for typing and shortcuts
    • Works with Windows, macOS, and Linux

    Requirements:
    • Samsung Galaxy Watch (WearOS 3+)
    • Bluetooth 5.0 compatible computer
    • WristControl desktop companion app

  category: "TOOLS"
  content_rating: "Everyone"
  privacy_policy_url: "https://wristcontrol.app/privacy"
```

**Release Tracks:**
```yaml
play-store-tracks:
  internal:
    description: "Development testing"
    users: "Team only"
    rollout: 100%

  alpha:
    description: "Early access testing"
    users: "Opt-in testers"
    rollout: 100%

  beta:
    description: "Public beta"
    users: "Open beta enrollment"
    rollout: 100%

  production:
    description: "General availability"
    users: "All users"
    rollout: "Staged (20% → 50% → 100%)"
```

### 3.2 Samsung Galaxy Store

**Additional Distribution:**
```yaml
galaxy-store:
  benefits:
    - Featured placement for Samsung devices
    - Galaxy Watch-specific promotion
    - Pre-installed possibility (partnership)

  requirements:
    - Samsung seller account
    - Content review
    - Galaxy Watch optimization
```

### 3.3 Build Configuration

```gradle
// watch-app/app/build.gradle
android {
    defaultConfig {
        applicationId "com.wristcontrol.watch"
        minSdk 30
        targetSdk 35
        versionCode getVersionCode()
        versionName getVersionName()
    }

    signingConfigs {
        release {
            storeFile file(System.getenv("KEYSTORE_PATH") ?: "release.keystore")
            storePassword System.getenv("KEYSTORE_PASSWORD")
            keyAlias System.getenv("KEY_ALIAS")
            keyPassword System.getenv("KEY_PASSWORD")
        }
    }

    buildTypes {
        release {
            minifyEnabled true
            shrinkResources true
            proguardFiles getDefaultProguardFile('proguard-android-optimize.txt'), 'proguard-rules.pro'
            signingConfig signingConfigs.release
        }
    }
}

def getVersionCode() {
    def versionMajor = 1
    def versionMinor = 0
    def versionPatch = 0
    return versionMajor * 10000 + versionMinor * 100 + versionPatch
}

def getVersionName() {
    return "1.0.0"
}
```

---

## 4. Build & Release Pipeline

### 4.1 GitHub Actions Workflow

```yaml
# .github/workflows/release.yml
name: Release Build

on:
  push:
    tags:
      - 'v*'

env:
  PYTHON_VERSION: '3.11'

jobs:
  # Build desktop apps
  build-desktop:
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        include:
          - os: ubuntu-latest
            artifact: linux
          - os: windows-latest
            artifact: windows
          - os: macos-latest
            artifact: macos

    runs-on: ${{ matrix.os }}

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pyinstaller

      - name: Build executable
        run: |
          pyinstaller --onefile --windowed --name WristControl src/main.py

      - name: Sign (Windows)
        if: matrix.os == 'windows-latest'
        run: |
          # Windows signing steps
          echo "Signing Windows executable"

      - name: Sign & Notarize (macOS)
        if: matrix.os == 'macos-latest'
        run: |
          # macOS signing steps
          echo "Signing and notarizing macOS app"

      - name: Package
        run: |
          # Platform-specific packaging

      - name: Upload artifact
        uses: actions/upload-artifact@v4
        with:
          name: desktop-${{ matrix.artifact }}
          path: dist/*

  # Build watch app
  build-watch:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Set up JDK
        uses: actions/setup-java@v4
        with:
          java-version: '17'
          distribution: 'temurin'

      - name: Build release APK
        working-directory: watch-app
        run: ./gradlew assembleRelease

      - name: Build release AAB
        working-directory: watch-app
        run: ./gradlew bundleRelease

      - name: Upload artifact
        uses: actions/upload-artifact@v4
        with:
          name: watch-app
          path: |
            watch-app/app/build/outputs/apk/release/*.apk
            watch-app/app/build/outputs/bundle/release/*.aab

  # Create release
  create-release:
    needs: [build-desktop, build-watch]
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Download all artifacts
        uses: actions/download-artifact@v4

      - name: Generate changelog
        run: |
          # Generate changelog from commits
          git log --pretty=format:"- %s" $(git describe --tags --abbrev=0 HEAD^)..HEAD > CHANGELOG.md

      - name: Create GitHub Release
        uses: softprops/action-gh-release@v1
        with:
          files: |
            desktop-windows/*
            desktop-macos/*
            desktop-linux/*
            watch-app/*.apk
          body_path: CHANGELOG.md
          draft: true

  # Publish to stores
  publish-play-store:
    needs: build-watch
    runs-on: ubuntu-latest
    if: startsWith(github.ref, 'refs/tags/v')

    steps:
      - uses: actions/checkout@v4

      - name: Download watch artifact
        uses: actions/download-artifact@v4
        with:
          name: watch-app

      - name: Publish to Play Store
        uses: r0adkll/upload-google-play@v1
        with:
          serviceAccountJsonPlainText: ${{ secrets.PLAY_STORE_SERVICE_ACCOUNT }}
          packageName: com.wristcontrol.watch
          releaseFiles: '*.aab'
          track: internal
          status: completed
```

### 4.2 Release Checklist

```markdown
## Release Checklist v{version}

### Pre-Release
- [ ] All tests passing on CI
- [ ] Code reviewed and approved
- [ ] CHANGELOG.md updated
- [ ] Version numbers bumped
- [ ] Documentation updated
- [ ] Security scan completed

### Build
- [ ] Windows installer built and signed
- [ ] macOS DMG built, signed, and notarized
- [ ] Linux AppImage built
- [ ] Watch APK/AAB built and signed

### Testing
- [ ] Smoke test on Windows
- [ ] Smoke test on macOS
- [ ] Smoke test on Linux
- [ ] Watch app tested on real device
- [ ] BLE connection tested
- [ ] Voice commands tested

### Release
- [ ] GitHub Release created (draft)
- [ ] Release notes written
- [ ] Artifacts uploaded
- [ ] Release published
- [ ] Play Store updated (staged rollout)

### Post-Release
- [ ] Monitor crash reports
- [ ] Monitor user feedback
- [ ] Update website
- [ ] Announce on social media
```

---

## 5. Version Management

### 5.1 Versioning Scheme

**Semantic Versioning (SemVer):**
```
MAJOR.MINOR.PATCH

Examples:
  1.0.0 - Initial release
  1.0.1 - Bug fix
  1.1.0 - New feature (backwards compatible)
  2.0.0 - Breaking change
```

**Version Synchronization:**
| Component | Version Source | Example |
|-----------|---------------|---------|
| Desktop App | `pyproject.toml` | 1.0.0 |
| Watch App | `build.gradle` | 1.0.0 |
| Protocol | Embedded constant | 1 |
| API | Embedded constant | 1 |

### 5.2 Compatibility Matrix

```yaml
compatibility:
  desktop_v1.0:
    watch_app: ["1.0.x", "1.1.x"]
    protocol: [1]

  desktop_v1.1:
    watch_app: ["1.0.x", "1.1.x", "1.2.x"]
    protocol: [1]

  desktop_v2.0:
    watch_app: ["2.0.x"]  # Breaking change
    protocol: [2]
```

### 5.3 Version Detection

```python
# src/version.py
import re

__version__ = "1.0.0"

# Protocol version for BLE communication
PROTOCOL_VERSION = 1

# Minimum compatible watch app version
MIN_WATCH_VERSION = "1.0.0"

def is_compatible(watch_version: str) -> bool:
    """Check if watch app version is compatible."""
    watch_parts = [int(x) for x in watch_version.split('.')]
    min_parts = [int(x) for x in MIN_WATCH_VERSION.split('.')]

    # Major version must match
    if watch_parts[0] != min_parts[0]:
        return False

    # Minor version must be >= minimum
    return tuple(watch_parts) >= tuple(min_parts)
```

---

## 6. Update Mechanism

### 6.1 Desktop Auto-Update

**Windows (using Squirrel):**
```python
# src/updater/windows.py
import subprocess
import tempfile
import requests

UPDATE_URL = "https://api.github.com/repos/wristcontrol/wristcontrol/releases/latest"

class WindowsUpdater:
    def check_for_updates(self) -> dict:
        """Check GitHub for newer version."""
        response = requests.get(UPDATE_URL)
        release = response.json()

        latest_version = release['tag_name'].lstrip('v')
        current_version = __version__

        if self._version_greater(latest_version, current_version):
            # Find Windows installer asset
            for asset in release['assets']:
                if asset['name'].endswith('.exe') and 'Setup' in asset['name']:
                    return {
                        'available': True,
                        'version': latest_version,
                        'download_url': asset['browser_download_url'],
                        'release_notes': release['body']
                    }

        return {'available': False}

    def download_and_install(self, download_url: str):
        """Download and run installer."""
        with tempfile.NamedTemporaryFile(suffix='.exe', delete=False) as f:
            response = requests.get(download_url, stream=True)
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
            installer_path = f.name

        # Run installer silently
        subprocess.Popen([installer_path, '/S'])
```

**macOS (using Sparkle):**
```xml
<!-- appcast.xml -->
<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0" xmlns:sparkle="http://www.andymatuschak.org/xml-namespaces/sparkle">
  <channel>
    <title>WristControl Updates</title>
    <item>
      <title>Version 1.0.1</title>
      <sparkle:version>1.0.1</sparkle:version>
      <sparkle:shortVersionString>1.0.1</sparkle:shortVersionString>
      <description>Bug fixes and improvements</description>
      <pubDate>Mon, 06 Jan 2026 10:00:00 +0000</pubDate>
      <enclosure url="https://wristcontrol.app/releases/WristControl-1.0.1.dmg"
                 sparkle:edSignature="..."
                 length="94371840"
                 type="application/octet-stream"/>
    </item>
  </channel>
</rss>
```

**Linux (AppImage Update):**
```python
# src/updater/linux.py
import subprocess
import os

class LinuxUpdater:
    def update_appimage(self, current_path: str):
        """Use appimageupdatetool for delta updates."""
        update_tool = "/usr/bin/appimageupdatetool"

        if os.path.exists(update_tool):
            result = subprocess.run([update_tool, current_path], capture_output=True)
            return result.returncode == 0

        return False
```

### 6.2 Watch App Updates

**Automatic via Play Store:**
- Users receive automatic updates
- Staged rollout for risk mitigation

**Manual APK Sideload:**
```bash
# For beta testing
adb install -r WristControl-1.0.1.apk
```

### 6.3 Update Notification UI

```python
# src/ui/update_dialog.py
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QTextEdit

class UpdateDialog(QDialog):
    def __init__(self, update_info: dict, parent=None):
        super().__init__(parent)
        self.update_info = update_info
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle("Update Available")
        layout = QVBoxLayout(self)

        # Version info
        layout.addWidget(QLabel(f"New version available: {self.update_info['version']}"))
        layout.addWidget(QLabel(f"Current version: {__version__}"))

        # Release notes
        notes = QTextEdit()
        notes.setReadOnly(True)
        notes.setMarkdown(self.update_info['release_notes'])
        layout.addWidget(notes)

        # Buttons
        update_btn = QPushButton("Update Now")
        update_btn.clicked.connect(self.accept)
        layout.addWidget(update_btn)

        later_btn = QPushButton("Remind Me Later")
        later_btn.clicked.connect(self.reject)
        layout.addWidget(later_btn)
```

---

## 7. Installation Guide

### 7.1 System Requirements

```markdown
## System Requirements

### Desktop Computer
- **Operating System:**
  - Windows 10/11 (64-bit)
  - macOS 12 Monterey or later
  - Ubuntu 22.04 LTS or later

- **Hardware:**
  - Bluetooth 5.0 adapter (built-in or USB)
  - 4GB RAM minimum
  - 200MB free disk space

- **Permissions:**
  - Bluetooth access
  - Accessibility/Input permission (for cursor control)
  - Microphone access (optional, for voice commands)

### Samsung Galaxy Watch
- Galaxy Watch4 or later
- WearOS 3.0 or later
- Bluetooth enabled
```

### 7.2 Installation Steps

**Windows:**
```markdown
## Windows Installation

1. Download `WristControl-Setup-1.0.0.exe` from the releases page
2. Run the installer
3. If Windows Defender SmartScreen appears, click "More info" → "Run anyway"
4. Follow the installation wizard
5. WristControl will start automatically after installation

### First Run
1. Enable Bluetooth on your computer
2. Ensure your Galaxy Watch is nearby and not connected to other devices
3. WristControl will appear in the system tray
4. Right-click the tray icon → "Connect Watch"
5. Select your watch from the list
6. Follow the pairing prompt on your watch
```

**macOS:**
```markdown
## macOS Installation

1. Download `WristControl-1.0.0.dmg` from the releases page
2. Open the DMG file
3. Drag WristControl to the Applications folder
4. On first launch, you may need to:
   - Right-click → Open (to bypass Gatekeeper)
   - Grant Accessibility permissions in System Settings → Privacy & Security

### Permissions Required
- **Bluetooth:** To connect to your watch
- **Accessibility:** To control cursor and keyboard
- **Microphone:** For voice commands (optional)
```

**Linux:**
```markdown
## Linux Installation

### AppImage (Recommended)
1. Download `WristControl-1.0.0-x86_64.AppImage`
2. Make it executable:
   ```bash
   chmod +x WristControl-1.0.0-x86_64.AppImage
   ```
3. Run the AppImage:
   ```bash
   ./WristControl-1.0.0-x86_64.AppImage
   ```

### Debian/Ubuntu
1. Download `wristcontrol_1.0.0_amd64.deb`
2. Install:
   ```bash
   sudo dpkg -i wristcontrol_1.0.0_amd64.deb
   sudo apt-get install -f  # Install dependencies if needed
   ```

### Bluetooth Setup (Linux)
Ensure your user is in the `bluetooth` group:
```bash
sudo usermod -aG bluetooth $USER
```
Log out and back in for changes to take effect.
```

### 7.3 Watch App Installation

```markdown
## Watch App Installation

### From Google Play Store (Recommended)
1. On your Galaxy Watch, open the Play Store
2. Search for "WristControl"
3. Tap Install
4. Grant requested permissions when prompted

### From Galaxy Store
1. On your Galaxy Watch, open Galaxy Store
2. Search for "WristControl"
3. Tap Install

### Manual Installation (for testers)
1. Enable Developer Options on your watch
2. Enable ADB debugging
3. Connect to your watch:
   ```bash
   adb connect <watch-ip>:5555
   ```
4. Install the APK:
   ```bash
   adb install WristControl-1.0.0.apk
   ```
```

---

## 8. Configuration & Onboarding

### 8.1 First-Run Experience

```python
# src/onboarding/wizard.py
from PyQt6.QtWidgets import QWizard, QWizardPage

class OnboardingWizard(QWizard):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("WristControl Setup")

        self.addPage(WelcomePage())
        self.addPage(BluetoothPage())
        self.addPage(WatchPairingPage())
        self.addPage(PermissionsPage())
        self.addPage(CalibrationPage())
        self.addPage(TutorialPage())
        self.addPage(CompletePage())

class WelcomePage(QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle("Welcome to WristControl")
        self.setSubTitle("Let's set up your hands-free computer control")

class BluetoothPage(QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle("Bluetooth")
        self.setSubTitle("Checking Bluetooth status...")

    def initializePage(self):
        # Check Bluetooth is enabled
        if not self.is_bluetooth_enabled():
            self.show_enable_bluetooth_instructions()

class WatchPairingPage(QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle("Connect Your Watch")
        self.setSubTitle("Searching for Galaxy Watch...")

    def initializePage(self):
        # Start BLE scan
        self.start_device_scan()

class CalibrationPage(QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle("Calibration")
        self.setSubTitle("Hold your arm in a comfortable neutral position")

    def initializePage(self):
        # Guide user through calibration
        self.start_calibration_timer()
```

### 8.2 Default Configuration

```json
// config/default_config.json
{
  "version": 1,
  "connection": {
    "auto_connect": true,
    "reconnect_attempts": 5,
    "reconnect_delay_ms": 2000
  },
  "cursor": {
    "enabled": true,
    "sensitivity": 2.0,
    "dead_zone": 2.0,
    "smoothing": 0.007,
    "max_speed": 1200,
    "profile": "default"
  },
  "gestures": {
    "tap": "left_click",
    "double_tap": "double_click",
    "hold": "drag",
    "flick_left": "scroll_left",
    "flick_right": "scroll_right",
    "palm_up": "disable_cursor",
    "palm_down": "enable_cursor"
  },
  "voice": {
    "enabled": true,
    "engine": "local",
    "model": "tiny",
    "activation": "push_to_talk"
  },
  "ui": {
    "show_connection_notification": true,
    "minimize_to_tray": true,
    "start_minimized": false,
    "start_with_system": true
  },
  "privacy": {
    "telemetry": false,
    "crash_reports": true
  }
}
```

### 8.3 Configuration Migration

```python
# src/config/migration.py

def migrate_config(config: dict, from_version: int, to_version: int) -> dict:
    """Migrate configuration between versions."""

    migrations = {
        (1, 2): migrate_v1_to_v2,
        (2, 3): migrate_v2_to_v3,
    }

    current = config.copy()
    current_version = from_version

    while current_version < to_version:
        next_version = current_version + 1
        migration = migrations.get((current_version, next_version))

        if migration:
            current = migration(current)
            current['version'] = next_version

        current_version = next_version

    return current

def migrate_v1_to_v2(config: dict) -> dict:
    """Migration from v1 to v2: Added voice activation modes."""
    config['voice']['activation'] = 'push_to_talk'  # New field
    return config
```

---

## 9. Monitoring & Analytics

### 9.1 Crash Reporting

```python
# src/monitoring/crash_reporter.py
import sentry_sdk
from sentry_sdk.integrations.threading import ThreadingIntegration

def init_crash_reporting(enabled: bool = True):
    """Initialize Sentry crash reporting."""
    if not enabled:
        return

    sentry_sdk.init(
        dsn="https://your-sentry-dsn",
        release=f"wristcontrol@{__version__}",
        environment="production",
        integrations=[ThreadingIntegration()],
        traces_sample_rate=0.1,

        # Privacy: Don't send personal data
        before_send=scrub_personal_data,
    )

def scrub_personal_data(event, hint):
    """Remove personal data from crash reports."""
    # Remove user-specific paths
    if 'exception' in event:
        for exception in event['exception']['values']:
            if 'stacktrace' in exception:
                for frame in exception['stacktrace']['frames']:
                    frame['abs_path'] = anonymize_path(frame.get('abs_path', ''))

    return event
```

### 9.2 Usage Analytics (Opt-in)

```python
# src/monitoring/analytics.py
import uuid
from typing import Optional

class Analytics:
    """Privacy-respecting usage analytics (opt-in only)."""

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.session_id = str(uuid.uuid4()) if enabled else None

    def track_event(self, event: str, properties: dict = None):
        """Track anonymous event."""
        if not self.enabled:
            return

        payload = {
            'event': event,
            'session': self.session_id,
            'version': __version__,
            'platform': platform.system(),
            'properties': properties or {}
        }

        # Send to analytics endpoint
        # No personal data, no IP tracking

    # Pre-defined events
    def app_started(self):
        self.track_event('app_started')

    def watch_connected(self, watch_model: str):
        self.track_event('watch_connected', {'model': watch_model})

    def gesture_used(self, gesture_type: str):
        self.track_event('gesture_used', {'type': gesture_type})

    def voice_command(self, command_type: str, success: bool):
        self.track_event('voice_command', {
            'type': command_type,
            'success': success
        })
```

### 9.3 Health Monitoring

```python
# src/monitoring/health.py
import logging
from dataclasses import dataclass
from datetime import datetime

@dataclass
class HealthMetrics:
    connection_uptime: float  # seconds
    cursor_latency_avg: float  # ms
    voice_latency_avg: float  # ms
    gesture_accuracy: float  # percentage
    errors_last_hour: int

class HealthMonitor:
    """Monitor application health metrics."""

    def __init__(self):
        self.metrics = HealthMetrics(
            connection_uptime=0,
            cursor_latency_avg=0,
            voice_latency_avg=0,
            gesture_accuracy=100,
            errors_last_hour=0
        )
        self.latency_samples = []

    def record_latency(self, latency_ms: float, type: str):
        """Record latency sample."""
        self.latency_samples.append((datetime.now(), type, latency_ms))

        # Keep last 1000 samples
        if len(self.latency_samples) > 1000:
            self.latency_samples = self.latency_samples[-1000:]

        # Update averages
        cursor_samples = [s[2] for s in self.latency_samples if s[1] == 'cursor']
        voice_samples = [s[2] for s in self.latency_samples if s[1] == 'voice']

        if cursor_samples:
            self.metrics.cursor_latency_avg = sum(cursor_samples) / len(cursor_samples)
        if voice_samples:
            self.metrics.voice_latency_avg = sum(voice_samples) / len(voice_samples)

    def get_status(self) -> dict:
        """Get health status summary."""
        return {
            'healthy': self.is_healthy(),
            'metrics': self.metrics,
            'warnings': self.get_warnings()
        }

    def is_healthy(self) -> bool:
        """Check if all metrics are within acceptable ranges."""
        return (
            self.metrics.cursor_latency_avg < 100 and
            self.metrics.voice_latency_avg < 600 and
            self.metrics.errors_last_hour < 10
        )
```

---

## 10. Rollback & Recovery

### 10.1 Version Rollback

**Desktop (Windows):**
```powershell
# Uninstall current version
winget uninstall WristControl

# Install specific version
winget install WristControl --version 1.0.0
```

**Desktop (macOS):**
```bash
# Move current version to trash
mv /Applications/WristControl.app ~/.Trash/

# Download and install previous version
curl -LO https://github.com/wristcontrol/releases/download/v1.0.0/WristControl-1.0.0.dmg
hdiutil attach WristControl-1.0.0.dmg
cp -R /Volumes/WristControl/WristControl.app /Applications/
```

**Watch App:**
```bash
# Sideload previous APK version
adb install -r WristControl-1.0.0.apk
```

### 10.2 Configuration Recovery

```python
# src/config/recovery.py
import shutil
from pathlib import Path

class ConfigRecovery:
    """Configuration backup and recovery."""

    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        self.backup_dir = config_dir / 'backups'

    def create_backup(self):
        """Create timestamped backup of configuration."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = self.backup_dir / f'config_{timestamp}.json'

        self.backup_dir.mkdir(exist_ok=True)
        shutil.copy(self.config_dir / 'config.json', backup_path)

        # Keep only last 10 backups
        backups = sorted(self.backup_dir.glob('config_*.json'))
        for old_backup in backups[:-10]:
            old_backup.unlink()

        return backup_path

    def restore_backup(self, backup_name: str):
        """Restore configuration from backup."""
        backup_path = self.backup_dir / backup_name
        if backup_path.exists():
            shutil.copy(backup_path, self.config_dir / 'config.json')
            return True
        return False

    def restore_defaults(self):
        """Restore factory default configuration."""
        default_config = Path(__file__).parent / 'default_config.json'
        shutil.copy(default_config, self.config_dir / 'config.json')
```

### 10.3 Emergency Recovery Mode

```python
# src/recovery/safe_mode.py

def check_safe_mode():
    """Check if app should start in safe mode."""
    crash_file = Path('~/.wristcontrol/.crash_flag').expanduser()

    if crash_file.exists():
        # App crashed on last run
        crash_count = int(crash_file.read_text())

        if crash_count >= 3:
            # Too many crashes, start in safe mode
            return True

    return False

def start_safe_mode():
    """Start application with minimal features."""
    print("Starting in Safe Mode...")
    print("- Cursor control: Disabled")
    print("- Voice commands: Disabled")
    print("- Only showing settings UI")

    # Load default config
    config = load_default_config()

    # Disable all features
    config['cursor']['enabled'] = False
    config['voice']['enabled'] = False

    # Start settings UI only
    show_settings_window(safe_mode=True)
```

### 10.4 Staged Rollout Strategy

```yaml
# Play Store staged rollout
rollout_stages:
  day_0:
    percentage: 5%
    duration: 24 hours
    halt_criteria:
      - crash_rate > 2%
      - negative_reviews > 10

  day_1:
    percentage: 20%
    duration: 48 hours
    halt_criteria:
      - crash_rate > 1%
      - rating_drop > 0.3

  day_3:
    percentage: 50%
    duration: 48 hours
    halt_criteria:
      - crash_rate > 0.5%
      - critical_bugs > 0

  day_5:
    percentage: 100%
    monitoring_period: 7 days
```

---

## Appendix A: Release Notes Template

```markdown
# WristControl v1.0.0 Release Notes

## What's New
- Initial release of WristControl
- Cursor control via wrist movements
- Gesture-based clicking (tap, double-tap, hold)
- Voice commands for text input and shortcuts
- Cross-platform support (Windows, macOS, Linux)
- Samsung Galaxy Watch4/5/6 support

## System Requirements
- Windows 10/11, macOS 12+, or Ubuntu 22.04+
- Bluetooth 5.0
- Galaxy Watch with WearOS 3.0+

## Known Issues
- [#123] Bluetooth reconnection may take up to 30 seconds
- [#145] Voice commands may not work in noisy environments

## Installation
See [Installation Guide](https://docs.wristcontrol.app/install)

## Feedback
Report issues at https://github.com/wristcontrol/issues
```

---

*Deployment strategy document for WristControl project - January 2026*
