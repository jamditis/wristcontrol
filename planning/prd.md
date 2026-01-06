# **Product requirements document: Voice and gesture computer control system**

## **Project overview**

A wearable-based input system that allows computer control through smartwatch motion sensors and voice commands, enabling hands-free or reduced-manual-input interaction with desktop environments.

## **Core user need**

Users need a way to control their computer without traditional mouse and keyboard input, using natural movements and voice commands captured by a Samsung Galaxy Watch.

## **Competitive landscape and key learnings**

### **DoublePoint WowMouse: Market leader analysis**

**What they've accomplished:**

* 140,000+ downloads since CES 2024 (as of March 2025\)  
* 4.6/5 rating on Google Play  
* Support for both WearOS and Apple Watch platforms  
* Commercial partnerships with Bosch Sensortec and Ambiq for chipset integration  
* Featured by CNBC, CNET, The Verge, and Fast Company Innovation Award

**Core technical approach:**

* Finger-pinch detection as primary gesture (tap, double-tap, hold)  
* IMU-based motion tracking for cursor control  
* 97% accuracy stationary, 95% walking, 94% running  
* Zero calibration required for basic functionality  
* WebBLE connectivity for low-latency streaming  
* On-device gesture detection algorithms

**Key gestures they support:**

* Tap (index finger \+ thumb pinch)  
* Double-tap  
* Hold/long press  
* Palm-up-tap (for back/home actions)  
* Flick to scroll (in Pro version)  
* Touch tracking on watch screen for supplementary input

**Architectural insights from their SDK:**

* Uses Bluetooth Low Energy (WebBLE) for device communication  
* Streams sensor data: acceleration, gravity, angular velocity, orientation, magnetic field  
* Processes gestures on-watch, sends events to companion  
* Open-source SDKs available: JavaScript, Python, Unity, Lens Studio  
* Events exposed: `on_tap`, `on_touch_down`, `on_touch_up`, `on_touch_move`, `on_sensors`

**Business model:**

* Free tier with basic functionality  
* Pro tier at $4.99 (WearOS) with customization and media controls  
* Specialized apps (Presenter for $3.99)  
* Developer SDK and evaluation kits for B2B

**Key differentiators for our project:**

1. **Voice integration:** WowMouse does not include voice-to-text or voice commands  
2. **Motion-based cursor control:** WowMouse focuses on gestures for clicks, not continuous cursor movement from hand motion  
3. **Gesture vocabulary:** We can extend beyond finger pinches to include wrist rotation, arm movements, etc.

## **System components**

### **1\. Watch application (input capture)**

**Platform:** Samsung Galaxy Watch (Wear OS or Tizen, depending on model)

**Core functionality:**

* Capture accelerometer and gyroscope data at high frequency for smooth cursor control  
* Implement finger-pinch detection (learning from DoublePoint's approach)  
* Record microphone input for voice commands  
* Stream both data types to desktop companion app via Bluetooth or WiFi  
* Provide visual/haptic feedback for connection status and activation state

**Technical requirements:**

* Sensor sampling rate: Target 50-100Hz for motion data to ensure smooth cursor movement  
* Audio streaming: Support continuous microphone capture with acceptable latency (\<200ms ideal)  
* Connection protocols: Prioritize Bluetooth Low Energy (WebBLE) based on DoublePoint success, with WiFi as fallback  
* Battery considerations: Implement power-efficient streaming modes (consider Ambiq's low-power approach)  
* User controls: Simple interface for enabling/disabling input modes  
* On-device processing: Detect basic gestures (tap, double-tap) on watch to reduce latency

**Development considerations:**

* Newer Samsung watches use Wear OS (Kotlin/Java development)  
* Older models use Tizen (C or web technologies)  
* May require different codebases for different watch generations  
* Consider starting with DoublePoint's TouchSDK as reference architecture

**Key improvement over WowMouse:**

* Add voice capture and streaming capability  
* Implement continuous motion tracking for cursor control (not just discrete gestures)

### **2\. Desktop companion application (processing and control)**

**Platform:** Windows, macOS, and Linux support preferred

**Core functionality:**

* Receive sensor data stream from watch via Bluetooth LE  
* Translate motion data into cursor movements using configurable algorithms  
* Receive audio stream and process through speech-to-text  
* Process gesture events from watch (tap, double-tap, hold)  
* Inject mouse movements and keyboard inputs into operating system  
* Provide configuration interface for sensitivity, dead zones, and gesture mappings

**Technical requirements:**

* Low-latency data processing (\<50ms from sensor to cursor movement)  
* Support for multiple motion-to-cursor algorithms:  
  * Relative positioning (tilt-to-move)  
  * Absolute positioning (point-at-screen)  
  * Hybrid modes with configurable behavior  
* Click/interaction triggers:  
  * Finger pinch gestures (tap, double-tap, hold) from watch  
  * Physical button presses on watch  
  * Additional gesture recognition (wrist flick, arm gestures)  
  * Voice commands ("click", "double-click", "right-click")  
* Drift correction for accumulated accelerometer errors  
* Activation/deactivation mechanisms to prevent unintended input  
* Environment-adaptive thresholds (indoor vs outdoor, stationary vs moving)  
* Configuration UI for:  
  * Sensitivity adjustment  
  * Dead zone configuration  
  * Gesture customization  
  * Voice command mapping  
  * Connection management  
  * Per-application profiles

**Suggested tech stack:**

* Python with libraries like `pynput` for OS input injection  
* WebBLE or `bleak` (Python Bluetooth library) for device communication  
* Real-time smoothing and filtering algorithms for motion data  
* Consider DoublePoint's TouchSDK Python implementation as reference

**Architecture learnings from DoublePoint:**

* Process gestures on-watch when possible for lower latency  
* Use event-driven architecture for gesture recognition  
* Separate sensor streaming from gesture detection  
* Implement connection resilience and auto-reconnect

### **3\. Speech-to-text pipeline**

**Options for implementation:**

**Option A: Local processing**

* Use OpenAI Whisper running locally  
* Pros: Privacy, no network latency, no usage costs  
* Cons: Requires decent CPU/GPU, longer initial setup  
* Best for: Privacy-conscious users, offline usage

**Option B: Cloud APIs**

* Services: OpenAI Whisper API, Google Speech-to-Text, Deepgram, or similar  
* Pros: Easier implementation, better accuracy, lower hardware requirements  
* Cons: Network latency, ongoing costs, privacy considerations  
* Best for: Quick implementation, users with reliable internet

**Requirements:**

* Support continuous listening mode with activation triggers  
* Handle natural language commands for:  
  * Mouse actions (click, double-click, right-click, scroll)  
  * Text input  
  * System commands  
  * Custom shortcuts  
* Configurable voice command vocabulary  
* Feedback when commands are recognized  
* Noise filtering to work in typical office/home environments

**Key differentiator:**

* This is our primary value-add over WowMouse  
* Voice commands can supplement gestures for complex actions  
* Text entry via voice enables full keyboard replacement

### **4\. Motion-to-cursor algorithm**

**Core challenges:**

* Translating 3D motion into 2D cursor movement  
* Preventing drift from accelerometer integration errors  
* Creating natural-feeling control that doesn't cause fatigue  
* Distinguishing intentional from incidental hand movements

**Algorithm requirements:**

**Positioning modes:**

* **Relative mode:** Hand tilt controls cursor velocity (like analog stick)  
* **Absolute mode:** Hand position maps to screen position (like pointing)  
* **Hybrid mode:** Combine approaches based on context

**Calibration features:**

* Dead zones to ignore small unintentional movements  
* Sensitivity curves (linear, exponential, custom)  
* User-specific calibration routines  
* Per-application sensitivity profiles  
* **Learning from WowMouse:** Zero-calibration baseline with optional fine-tuning

**Drift mitigation:**

* Periodic recalibration (automatic or on-demand)  
* Fusion with gyroscope data for better accuracy  
* Optional absolute position reset via gesture or button  
* **DoublePoint approach:** Orientation quaternions for drift-free tracking

**Activation control:**

* Explicit enable/disable (button hold, gesture, voice)  
* Auto-disable after inactivity  
* Visual/haptic feedback for mode changes  
* **WowMouse insight:** Palm-up gesture as natural deactivation trigger

**User experience tuning:**

* Smoothing and filtering to reduce jitter  
* Acceleration curves for both precision and speed  
* Edge snapping and target assistance options  
* Fatigue-reducing defaults  
* **Environment adaptation:** Different thresholds for indoor/outdoor, stationary/walking (learned from WowMouse's 97%/95%/94% accuracy across conditions)

**Performance targets (inspired by WowMouse):**

* 95%+ gesture recognition accuracy in typical use  
* Maintain \>90% accuracy while walking  
* Function acceptably while user is in motion

### **5\. Gesture recognition system**

**Learning from DoublePoint's approach:**

**Primary gestures (finger-pinch based):**

* **Tap:** Quick index finger \+ thumb pinch  
  * Use: Left click, confirm, select  
  * Implementation: Detect contact through accelerometer spike patterns  
* **Double-tap:** Two rapid taps  
  * Use: Double-click, open items  
* **Hold:** Sustained pinch  
  * Use: Right-click, drag, context menus  
* **Palm-up-tap:** Tap with palm facing up  
  * Use: Back/escape, cancel, system commands

**Extended gestures (our additions):**

* **Wrist flick:** Quick rotation  
  * Use: Scroll, switch tabs  
* **Arm gestures:** Larger movements  
  * Use: Window management, workspace switching  
* **Voice-triggered gestures:** Combine voice \+ motion  
  * Use: Complex operations requiring precision

**Technical implementation:**

* Process basic gestures on-watch using IMU data patterns  
* Send gesture events to desktop rather than raw sensor data when possible  
* Use accelerometer magnitude spikes to detect finger contact  
* Combine with gyroscope for orientation context  
* Optional: Use magnetometer for absolute orientation

**Advantages of on-watch processing:**

* Lower latency (no sensor data transmission lag)  
* Reduced battery usage (less BLE traffic)  
* Works even with intermittent connection  
* Privacy (gesture patterns not transmitted)

## **Interaction design**

### **Click mechanisms**

Multiple options to support different user preferences:

* **Primary:** Finger pinch (tap) gesture detected on-watch  
* **Secondary:** Double-tap for double-click  
* **Tertiary:** Hold for right-click/context menu  
* **Alternative:** Voice command ("click", "right-click")  
* **Fallback:** Physical button press on watch

### **Scroll mechanisms**

* **Wrist flick** gesture (quick rotation)  
* **Voice commands** ("scroll up", "scroll down", "scroll to bottom")  
* **Button \+ motion** combination  
* **Dedicated scroll mode** activated by voice or gesture

### **Text input flow**

1. User activates voice input (button, gesture, or wake word)  
2. Watch provides haptic feedback confirming listening state  
3. User speaks text or commands  
4. Desktop app processes speech and:  
   * Inserts text into active application, or  
   * Executes recognized commands  
5. Visual/audio feedback confirms action

### **Activation/deactivation patterns**

**Learning from WowMouse palm-up gesture:**

* **Palm facing up:** Cursor control disabled (natural resting position)  
* **Palm facing forward/down:** Cursor control enabled (natural pointing)  
* **Voice command:** "Computer, wake up" / "Computer, sleep"  
* **Button hold:** Toggle active state with haptic confirmation

## **Development phases**

### **Phase 0: Research and prototyping (1 week)**

**Goal:** Understand DoublePoint's approach and evaluate feasibility

**Tasks:**

* Install and test WowMouse on target Samsung watch  
* Experiment with DoublePoint TouchSDK (Python version)  
* Analyze sensor data patterns for gesture detection  
* Test voice capture quality from watch microphone  
* Evaluate BLE latency and connection stability

**Success criteria:**

* Can connect to watch via TouchSDK  
* Can observe sensor data streams  
* Can detect basic finger-pinch patterns  
* Voice quality acceptable for STT

### **Phase 1: Desktop foundation (2-4 weeks)**

**Goal:** Working cursor control from simulated sensor data

**Tasks:**

* Build desktop companion app skeleton using Python  
* Implement WebBLE/Bluetooth LE connection handler  
* Create basic motion-to-cursor algorithm  
* Add gesture event handling (tap, double-tap, hold)  
* Test on target operating systems  
* Build configuration UI for sensitivity and dead zones

**Success criteria:**

* Smooth cursor movement from simulated sensor input  
* Sub-50ms latency from input to cursor movement  
* Reliable gesture recognition from event stream  
* Configurable sensitivity feels responsive

**Reference implementation:**

* Use DoublePoint TouchSDK Python as starting point  
* Extend their `Watch` class with cursor control logic

### **Phase 2: Voice integration (1-2 weeks)**

**Goal:** Working voice-to-text using computer microphone

**Tasks:**

* Integrate chosen STT service (start with cloud API for speed)  
* Implement command parsing for mouse actions  
* Add text insertion to active applications  
* Build command vocabulary and customization UI  
* Test wake-word activation

**Success criteria:**

* Reliable voice command recognition  
* Text insertion works across applications  
* Commands execute with \<500ms latency  
* Wake-word detection works consistently

### **Phase 3: Watch app with gesture detection (3-4 weeks)**

**Goal:** Custom watch app that detects gestures and streams data

**Tasks:**

* Build minimal watch app for Wear OS  
* Implement on-device gesture detection (tap, double-tap, hold)  
* Add accelerometer/gyroscope data capture for motion tracking  
* Create BLE service for streaming gestures and sensor data  
* Implement palm-orientation detection for activation control  
* Add connection status feedback on watch  
* Build simple calibration routine

**Success criteria:**

* Reliable finger-pinch detection (\>95% accuracy stationary)  
* Stable gesture event stream over BLE  
* Motion data latency \<100ms  
* Palm-up detection works reliably  
* Battery life acceptable (\>4 hours of active use)

**Technical approach:**

* Study DoublePoint's gesture detection patterns  
* Use accelerometer magnitude spikes for tap detection  
* Combine with gyroscope for gesture context  
* Implement simple state machine for gesture recognition

### **Phase 4: Watch audio streaming (2-3 weeks)**

**Goal:** Voice input from watch microphone

**Tasks:**

* Add microphone capture to watch app  
* Implement audio streaming over existing BLE connection  
* Integrate with STT pipeline in desktop app  
* Optimize for audio quality and latency  
* Add voice activation controls on watch  
* Implement noise cancellation/filtering

**Success criteria:**

* Clear audio capture in typical environments  
* Audio latency \<200ms  
* Voice commands recognized as reliably as desktop mic  
* Minimal impact on battery life

**Technical considerations:**

* Audio streaming over BLE is bandwidth-intensive  
* May need compression (Opus codec)  
* Consider push-to-talk to conserve battery  
* Test in noisy environments

### **Phase 5: Motion-based cursor control (2-3 weeks)**

**Goal:** Smooth cursor movement from hand motion

**Tasks:**

* Implement relative positioning mode (tilt-to-move)  
* Add absolute positioning mode (point-to-position)  
* Build drift correction algorithms  
* Implement smoothing and filtering  
* Add acceleration curves  
* Create dead zones for unintentional movement  
* Build environment-adaptive thresholds

**Success criteria:**

* Smooth cursor movement without jitter  
* Minimal drift over 5-minute sessions  
* Natural-feeling control that doesn't cause fatigue  
* Works acceptably while walking (\>90% usable)

**Technical approach:**

* Use orientation quaternions to avoid gimbal lock  
* Implement complementary filter for sensor fusion  
* Add Kalman filtering for smoothing  
* Test across different postures and movements

### **Phase 6: UX refinement (3-4 weeks)**

**Goal:** Polished, usable interaction model

**Tasks:**

* Implement additional gestures (wrist flick, arm movements)  
* Add multi-modal interactions (voice \+ gesture)  
* Create per-application sensitivity profiles  
* Build comprehensive calibration routines  
* Conduct user testing (self-testing initially)  
* Iterate on algorithm parameters based on usage  
* Add context-aware sensitivity profiles  
* Polish visual feedback and status indicators  
* Implement indoor/outdoor mode switching

**Success criteria:**

* Comfortable for \>30 minutes of use  
* \<5 unintended inputs per hour of use  
* User can complete typical desktop tasks at 70%+ normal speed  
* Minimal learning curve for basic functions  
* Graceful handling of edge cases

### **Phase 7: Polish and reliability (2-3 weeks)**

**Goal:** Production-ready system

**Tasks:**

* Error handling and recovery  
* Connection resilience improvements  
* Battery optimization  
* Documentation and setup guide  
* Optional: Add local Whisper support  
* Optional: Cross-platform testing and fixes  
* Add telemetry for usage patterns (privacy-preserving)  
* Build onboarding tutorial

## **Technical architecture**

### **Data flow**

Watch sensors → Gesture detection (on-watch) → Gesture events → BLE → Desktop app → OS input injection  
Watch sensors → Motion data stream → BLE → Desktop app → Cursor algorithm → OS cursor control  
Watch mic → Audio stream → BLE → Desktop app → STT service → Command execution

### **Component communication**

* **Protocol:** Bluetooth Low Energy (WebBLE/GATT) following DoublePoint's approach  
* **Data format:**  
  * Gesture events: Structured messages (tap, double-tap, hold, etc.)  
  * Sensor data: Binary packed structs for efficiency  
  * Audio: Opus compressed format  
* **Services:**  
  * Gesture service: Characteristic for gesture events  
  * Sensor service: Characteristic for IMU data stream  
  * Audio service: Characteristic for microphone stream

### **Security considerations**

* Optional authentication between watch and desktop  
* Encrypted BLE communication (LE Secure Connections)  
* No storage of voice data unless explicitly configured  
* Local processing option for privacy-sensitive users  
* User consent for all data collection

## **Success metrics**

### **Performance targets**

* Sensor-to-cursor latency: \<100ms (50ms ideal)  
* Gesture recognition latency: \<30ms (on-watch processing)  
* Voice-to-text latency: \<500ms  
* Connection stability: \<1 disconnection per hour  
* Battery life: \>4 hours of active use (6+ hours target)  
* Accuracy:  
  * Gesture recognition: \>95% stationary, \>90% walking  
  * Voice commands: \>90% in typical environments  
  * Cursor control: \<5% unintended actions

### **User experience targets**

* Setup time: \<10 minutes for first-time users (target: \<5 minutes)  
* Learning curve: Basic proficiency in \<30 minutes  
* Task completion: 70%+ of normal speed for common tasks (target: 85%)  
* Comfort: Usable for \>30 minutes without fatigue  
* Reduced frustration compared to traditional troubleshooting

## **Future enhancements (out of scope for initial release)**

* Multi-watch support for different input modes (one for gestures, one for haptic feedback)  
* Machine learning for personalized gesture recognition  
* Eye tracking integration for cursor positioning (like DoublePoint's AR implementation)  
* Haptic feedback patterns for different events  
* Plugin system for application-specific commands  
* Shared configuration profiles across devices  
* Remote desktop control capabilities  
* Integration with smart home systems  
* Accessibility features for users with limited mobility  
* Developer SDK for custom gesture vocabularies

## **Risk assessment**

### **High-risk areas**

* **Motion algorithm UX:** Hardest part to get right, requires extensive iteration  
* **Audio streaming latency:** May be limited by hardware/protocol, could require optimization  
* **Battery drain:** High-frequency sensor use may impact watch battery significantly  
* **Platform fragmentation:** Different watch models/OS versions may require separate implementations  
* **Gesture detection accuracy:** Finger-pinch patterns may vary significantly between users

### **Mitigation strategies**

* Start with simpler relative positioning before absolute  
* Implement multiple latency optimization passes  
* Add power-saving modes and usage warnings  
* Target specific watch models initially, expand support gradually  
* **Learning from WowMouse:** Use proven gesture patterns (finger-pinch) rather than inventing new ones  
* Build on DoublePoint's TouchSDK architecture rather than starting from scratch  
* Use their open-source implementations as reference

### **Medium-risk areas**

* **Competition from WowMouse:** They're well-established and improving rapidly  
* **User adoption:** Learning curve may be barrier to adoption  
* **Voice privacy concerns:** Users may be hesitant about microphone access

### **Low-risk areas**

* **Technical feasibility:** DoublePoint has proven the core gesture recognition works  
* **Platform support:** WearOS is well-documented and stable  
* **STT integration:** Well-solved problem with multiple vendors

## **Competitive positioning**

### **Our advantages over WowMouse:**

1. **Voice integration:** Full voice command and dictation support  
2. **Continuous cursor control:** Motion-based cursor movement, not just discrete gestures  
3. **Open architecture:** Potential for customization and extension  
4. **Privacy focus:** Local processing options

### **Areas where WowMouse is ahead:**

1. **Mature product:** 140,000+ downloads, proven reliability  
2. **Chipset partnerships:** Integration with Bosch and Ambiq  
3. **Multi-platform:** Apple Watch and WearOS support  
4. **AR/VR integration:** Works with Meta Quest, Magic Leap  
5. **No calibration:** Zero-setup gesture recognition

### **Differentiation strategy:**

* Position as "voice \+ gesture" solution rather than pure gesture control  
* Target users who need text input (writers, programmers, accessibility users)  
* Emphasize privacy with local processing options  
* Focus on cursor control use cases (design, gaming, navigation)  
* Consider open-source release to build community

## **Open questions**

1. Should we build on DoublePoint's TouchSDK or create independent implementation?  
   * **Recommendation:** Use TouchSDK as foundation, extend with voice and motion tracking  
2. Should the system support both Wear OS and Tizen, or focus on one platform initially?  
   * **Recommendation:** Start with WearOS (larger market share, better documentation)  
3. What's the primary use case: accessibility, productivity, or specialized environments?  
   * **Recommendation:** Start with productivity, expand to accessibility  
4. Should voice commands require activation (push-to-talk) or support always-listening mode?  
   * **Recommendation:** Start with push-to-talk for privacy and battery, add always-on as option  
5. What's the acceptable price point for any cloud services required?  
   * **Recommendation:** Offer free tier with local Whisper, paid tier with cloud STT  
6. Should the desktop app be a background service or windowed application?  
   * **Recommendation:** Background service with system tray icon  
7. How do we differentiate from WowMouse Pro ($4.99) in the market?  
   * **Recommendation:** Voice as primary differentiator, target different user segments

## **Resources and references**

### **Existing tools to evaluate**

* **WowMouse:** Commercial product by DoublePoint (study competitor)  
* **DoublePoint TouchSDK:** Open-source SDK for gesture detection  
  * Python: https://github.com/doublepointlab/touch-sdk-py  
  * JavaScript: https://github.com/doublepointlab/touch-sdk-js  
  * Unity: https://github.com/doublepointlab/touch-sdk-unity  
* **Tasker \+ AutoWear:** Android automation tools with sensor bridging  
* **KDE Connect / GSConnect:** Phone-to-computer input bridging

### **Key technologies**

* Wear OS SDK documentation  
* Tizen SDK documentation (if supporting older watches)  
* OpenAI Whisper (local STT)  
* Cloud STT APIs (Google, Deepgram, etc.)  
* OS input injection libraries (pynput for Python)  
* Bluetooth Low Energy / WebBLE specifications  
* Audio codecs for streaming (Opus)

### **Academic and industry research**

* DoublePoint's published accuracy metrics (97%/95%/94% across conditions)  
* IMU-based gesture recognition research  
* Sensor fusion algorithms for wearables  
* Voice activity detection in noisy environments

## **Time and effort estimates**

### **With existing mobile development experience**

* Proof of concept: 2-3 weekends (reduced due to TouchSDK availability)  
* Usable first version: 1-2 months of side project time  
* Polished product: 3-4 months

### **Learning as you go**

* Proof of concept: 3-5 weekends (TouchSDK reduces learning curve)  
* Usable first version: 2-3 months  
* Polished product: 4-5 months

### **Vibe coding suitability**

* **High suitability:**  
  * Desktop companion app  
  * Configuration UI  
  * Voice pipeline integration  
  * Motion algorithms  
  * TouchSDK integration and extension  
* **Medium suitability:**  
  * BLE protocol implementation  
  * Audio streaming  
  * Gesture detection algorithms  
* **Lower suitability:**  
  * Watch app (emulator setup, platform-specific APIs)  
  * On-device processing optimization  
  * Battery optimization

### **Impact of leveraging DoublePoint TouchSDK**

* **Time savings:** 2-4 weeks on gesture detection implementation  
* **Reduced risk:** Proven architecture for BLE communication and gesture events  
* **Learning curve:** Excellent reference implementation for best practices  
* **Trade-off:** Some architectural decisions constrained by their design

## **Business model considerations**

### **Pricing strategy**

* **Free tier:**  
  * Basic gesture support (tap, double-tap)  
  * Limited voice commands (10 most common)  
  * Local Whisper processing only  
* **Pro tier ($9.99 one-time or $2.99/month):**  
  * Full gesture vocabulary  
  * Unlimited voice commands  
  * Cloud STT with higher accuracy  
  * Per-application profiles  
  * Priority support  
* **Developer edition ($29.99):**  
  * API access for custom integrations  
  * Raw sensor data access  
  * Custom gesture training  
  * Commercial use license

### **Competitive pricing analysis**

* WowMouse: Free with $4.99 Pro upgrade  
* WowMouse Presenter: $3.99  
* Traditional clickers/remotes: $30-50  
* Dragon NaturallySpeaking: $150-300  
* **Our positioning:** $9.99 one-time for full features (between WowMouse and Dragon)

## **Implementation priorities**

### **Must-have for MVP**

1. Finger-pinch gestures (tap, double-tap, hold)  
2. Basic voice commands (click, type, scroll)  
3. Reliable BLE connection  
4. Windows support  
5. Simple setup (\<10 minutes)

### **Should-have for MVP**

1. Motion-based cursor control  
2. macOS support  
3. Per-application profiles  
4. Local Whisper option  
5. Configuration UI

### **Nice-to-have (post-MVP)**

1. Linux support  
2. Advanced gesture vocabulary  
3. Multi-modal interactions  
4. Smart home integration  
5. Developer API

---

*Document version: 2.0*  
 *Last updated: January 2026*  
 *Incorporates learnings from DoublePoint WowMouse competitive analysis*

