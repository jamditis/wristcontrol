# WearOS Development Guide for Gesture and Voice Control System
## Samsung Galaxy Watch - Technical Research

*Last Updated: January 2026*

---

## 1. Sensor APIs: High-Frequency IMU Data Access

### Overview

Samsung Galaxy Watch (WearOS 3+) provides access to motion sensors through two primary approaches:
- **Standard Android Sensor API** - For raw, high-frequency data (50-100Hz)
- **Samsung Health Sensor SDK** - Battery-optimized continuous tracking (25Hz)

### Available Hardware Sensors

Galaxy Watch4 and later models include:
- Accelerometer (3-axis)
- Gyroscope (3-axis)
- Magnetometer (3-axis)
- Pressure sensor
- Light sensor
- Heart rate sensor (PPG)

**Note:** All other sensors are composites derived from these hardware sensors.

### Method 1: Standard Android Sensor API (Recommended for 50-100Hz)

#### Setup

Add permission to AndroidManifest.xml:
```xml
<uses-feature android:name="android.hardware.sensor.accelerometer" />
<uses-feature android:name="android.hardware.sensor.gyroscope" />
<uses-permission android:name="android.permission.BODY_SENSORS" />
```

#### Kotlin Implementation

```kotlin
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity

class SensorStreamingActivity : ComponentActivity(), SensorEventListener {

    private lateinit var sensorManager: SensorManager
    private var accelerometer: Sensor? = null
    private var gyroscope: Sensor? = null
    private var magnetometer: Sensor? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        sensorManager = getSystemService(SENSOR_SERVICE) as SensorManager

        // Get sensor instances
        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
        magnetometer = sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD)

        // Check capabilities
        accelerometer?.let { sensor ->
            val minDelayMicros = sensor.minDelay
            val maxFrequencyHz = if (minDelayMicros > 0) {
                1_000_000.0 / minDelayMicros
            } else 0.0
            Log.d("SENSOR", "Accelerometer max frequency: $maxFrequencyHz Hz")
            Log.d("SENSOR", "Min delay: $minDelayMicros μs")
        }
    }

    override fun onResume() {
        super.onResume()

        // Register for fastest sampling rate
        // Expected: 50-100 Hz on most Galaxy Watch models
        accelerometer?.let {
            sensorManager.registerListener(
                this,
                it,
                SensorManager.SENSOR_DELAY_FASTEST
            )
        }

        gyroscope?.let {
            sensorManager.registerListener(
                this,
                it,
                SensorManager.SENSOR_DELAY_FASTEST
            )
        }

        magnetometer?.let {
            sensorManager.registerListener(
                this,
                it,
                SensorManager.SENSOR_DELAY_GAME // Lower frequency acceptable for orientation
            )
        }
    }

    override fun onPause() {
        super.onPause()
        // CRITICAL: Always unregister to prevent battery drain
        sensorManager.unregisterListener(this)
    }

    override fun onSensorChanged(event: SensorEvent?) {
        event?.let {
            when (it.sensor.type) {
                Sensor.TYPE_ACCELEROMETER -> {
                    val x = it.values[0]
                    val y = it.values[1]
                    val z = it.values[2]
                    val timestamp = it.timestamp // nanoseconds

                    // Stream to BLE or process locally
                    processAccelerometerData(x, y, z, timestamp)
                }
                Sensor.TYPE_GYROSCOPE -> {
                    val x = it.values[0] // rad/s
                    val y = it.values[1]
                    val z = it.values[2]
                    processGyroscopeData(x, y, z, it.timestamp)
                }
                Sensor.TYPE_MAGNETIC_FIELD -> {
                    val x = it.values[0] // μT (microteslas)
                    val y = it.values[1]
                    val z = it.values[2]
                    processMagnetometerData(x, y, z, it.timestamp)
                }
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        Log.d("SENSOR", "Accuracy changed: ${sensor?.name} -> $accuracy")
    }

    private fun processAccelerometerData(x: Float, y: Float, z: Float, timestamp: Long) {
        // Implement gesture detection or BLE streaming
    }

    private fun processGyroscopeData(x: Float, y: Float, z: Float, timestamp: Long) {
        // Process rotation data
    }

    private fun processMagnetometerData(x: Float, y: Float, z: Float, timestamp: Long) {
        // Process orientation data
    }
}
```

#### Sensor Delay Constants

| Constant | Typical Frequency | Use Case |
|----------|------------------|----------|
| `SENSOR_DELAY_FASTEST` | 50-100 Hz | Gaming, gesture recognition |
| `SENSOR_DELAY_GAME` | ~50 Hz | Gaming |
| `SENSOR_DELAY_UI` | ~16 Hz (60 FPS) | UI updates |
| `SENSOR_DELAY_NORMAL` | ~5 Hz | Basic monitoring |

**Important:** In Android 13+, sensor sampling can be restricted to RATE_NORMAL (50 Hz) unless using SensorDirectChannel.

### Method 2: Samsung Health Sensor SDK (Battery-Optimized)

#### Advantages
- 25Hz continuous tracking
- **Minimal battery consumption** - gathers data in application processor without waking CPU
- Batch mode when screen is off
- Designed for all-day tracking

#### Setup

Add to build.gradle:
```gradle
dependencies {
    implementation 'com.samsung.android:health-sensor-control:1.x.x'
}
```

#### Kotlin Implementation

```kotlin
import com.samsung.android.service.health.tracking.HealthTracker
import com.samsung.android.service.health.tracking.HealthTrackerException
import com.samsung.android.service.health.tracking.HealthTrackingService
import com.samsung.android.service.health.tracking.data.DataPoint
import com.samsung.android.service.health.tracking.data.HealthTrackerType
import com.samsung.android.service.health.tracking.data.ValueKey

class AccelerometerTrackerActivity : ComponentActivity() {

    private var healthTrackingService: HealthTrackingService? = null
    private var accelerometerTracker: HealthTracker? = null

    private val connectionListener = object : HealthTrackingService.ConnectionListener {
        override fun onConnectionSuccess() {
            Log.d("HEALTH", "Connected to Health Tracking Service")

            // Check if accelerometer tracking is available
            val availableTrackers = healthTrackingService?.trackingCapability?.supportHealthTrackerTypes

            if (availableTrackers?.contains(HealthTrackerType.ACCELEROMETER_CONTINUOUS) == true) {
                initAccelerometerTracker()
            } else {
                Log.e("HEALTH", "Accelerometer tracking not supported")
            }
        }

        override fun onConnectionEnded() {
            Log.d("HEALTH", "Disconnected from Health Tracking Service")
        }

        override fun onConnectionFailed(error: HealthTrackerException) {
            Log.e("HEALTH", "Connection failed: ${error.message}")
        }
    }

    private val accelerometerListener = object : HealthTracker.TrackerEventListener {
        override fun onDataReceived(dataPoints: List<DataPoint>) {
            // Batch of accelerometer readings (25Hz)
            for (dataPoint in dataPoints) {
                val x = dataPoint.getValue(ValueKey.AccelerometerSet.X)
                val y = dataPoint.getValue(ValueKey.AccelerometerSet.Y)
                val z = dataPoint.getValue(ValueKey.AccelerometerSet.Z)
                val timestamp = dataPoint.timestamp

                // Process data
                processAccelerometerReading(x, y, z, timestamp)
            }
        }

        override fun onError(error: HealthTracker.TrackerError) {
            Log.e("HEALTH", "Tracker error: $error")
        }

        override fun onFlushCompleted() {
            Log.d("HEALTH", "Flush completed - batched data sent")
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Connect to Health Tracking Service
        HealthTrackingService.ConnectionListener.connectionListener = connectionListener
        HealthTrackingService.init(applicationContext)
    }

    private fun initAccelerometerTracker() {
        healthTrackingService?.let { service ->
            accelerometerTracker = service.getHealthTracker(
                HealthTrackerType.ACCELEROMETER_CONTINUOUS
            )
            accelerometerTracker?.setEventListener(accelerometerListener)
        }
    }

    override fun onDestroy() {
        super.onDestroy()

        // CRITICAL: Unset listeners and disconnect to prevent battery drain
        accelerometerTracker?.unsetEventListener()
        healthTrackingService?.disconnectService()
    }

    private fun processAccelerometerReading(x: Float, y: Float, z: Float, timestamp: Long) {
        // Your processing logic
    }

    // Use this to force batched data to be sent immediately
    private fun flushBatchedData() {
        accelerometerTracker?.flush()
    }
}
```

#### Battery Efficiency Features
- **Batching:** When screen is off, data is batched and sent periodically
- **flush():** Force immediate delivery of batched data
- **Auto-sleep:** SDK sleeps when watch is not worn (if off-body sensor available)

### Sensor Data Specifications (Samsung Galaxy Watch)

#### Accelerometer
- **Range:** ±2g, ±4g, ±8g, or ±16g (device-dependent)
- **Units:** m/s²
- **Sampling Rate:** Up to 100 Hz (hardware), 25 Hz (Health SDK)

#### Gyroscope
- **Range:** ±125, ±250, ±500, ±1000, ±2000 dps (device-dependent)
- **Units:** rad/s
- **Sampling Rate:** Up to 100 Hz

#### Magnetometer
- **Range:** ±400 to ±1600 μT
- **Units:** μT (microteslas)

**Important:** Sensor ranges may vary between Galaxy Watch models. Always check capabilities at runtime.

### Recommended Approach for Your Use Case

**For Gesture Control System:**
1. Use **Standard Android Sensor API** with `SENSOR_DELAY_FASTEST` for high-frequency (50-100Hz) real-time gesture detection
2. Combine accelerometer + gyroscope for robust motion tracking
3. Use magnetometer for absolute orientation (if needed)
4. Implement **foreground service** to maintain sensor access when screen is off

**For Battery-Conscious All-Day Monitoring:**
1. Use **Samsung Health Sensor SDK** for 25Hz continuous tracking
2. Leverage batching mode for background operation

---

## 2. Audio APIs: Microphone Input and Streaming

### Overview

On WearOS (Android Wear), **AudioRecord** is the primary (and often only) reliable API for microphone access. MediaRecorder has been reported to not work reliably on Wear OS devices.

### Setup

Add permissions to AndroidManifest.xml:
```xml
<uses-permission android:name="android.permission.RECORD_AUDIO" />
<uses-permission android:name="android.permission.MODIFY_AUDIO_SETTINGS" />
```

Request runtime permission (Android 6.0+):
```kotlin
import android.Manifest
import android.content.pm.PackageManager
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat

fun checkAudioPermission() {
    if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
        != PackageManager.PERMISSION_GRANTED) {
        ActivityCompat.requestPermissions(
            this,
            arrayOf(Manifest.permission.RECORD_AUDIO),
            REQUEST_RECORD_AUDIO_PERMISSION
        )
    }
}
```

### AudioRecord Implementation (Kotlin)

```kotlin
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import kotlinx.coroutines.*
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

class AudioCaptureManager {

    companion object {
        private const val SAMPLE_RATE = 16000 // 16 kHz (good for voice)
        private const val CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO
        private const val AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT
        private const val BUFFER_SIZE_MULTIPLIER = 2
    }

    private var audioRecord: AudioRecord? = null
    private var isRecording = false
    private var recordingJob: Job? = null

    private val bufferSize: Int by lazy {
        AudioRecord.getMinBufferSize(
            SAMPLE_RATE,
            CHANNEL_CONFIG,
            AUDIO_FORMAT
        ) * BUFFER_SIZE_MULTIPLIER
    }

    fun startRecording(onAudioData: (ByteArray, Int) -> Unit) {
        if (isRecording) return

        audioRecord = AudioRecord(
            MediaRecorder.AudioSource.VOICE_RECOGNITION, // Optimized for voice
            SAMPLE_RATE,
            CHANNEL_CONFIG,
            AUDIO_FORMAT,
            bufferSize
        )

        if (audioRecord?.state != AudioRecord.STATE_INITIALIZED) {
            Log.e("AUDIO", "AudioRecord initialization failed")
            return
        }

        audioRecord?.startRecording()
        isRecording = true

        // Start recording on background thread
        recordingJob = CoroutineScope(Dispatchers.IO).launch {
            val audioBuffer = ByteArray(bufferSize)

            while (isRecording && isActive) {
                val bytesRead = audioRecord?.read(audioBuffer, 0, audioBuffer.size) ?: 0

                if (bytesRead > 0) {
                    // Send raw PCM data to callback
                    onAudioData(audioBuffer, bytesRead)
                }
            }
        }

        Log.d("AUDIO", "Recording started - Sample Rate: $SAMPLE_RATE Hz")
    }

    fun stopRecording() {
        isRecording = false
        recordingJob?.cancel()

        audioRecord?.apply {
            stop()
            release()
        }
        audioRecord = null

        Log.d("AUDIO", "Recording stopped")
    }

    fun isRecording(): Boolean = isRecording
}

// Usage example
class VoiceStreamingActivity : ComponentActivity() {
    private val audioCapture = AudioCaptureManager()

    private fun startVoiceCapture() {
        audioCapture.startRecording { audioData, bytesRead ->
            // Option 1: Stream via BLE to desktop
            streamAudioViaBLE(audioData, bytesRead)

            // Option 2: Save to WAV file
            // appendToWavFile(audioData, bytesRead)

            // Option 3: Local speech recognition
            // processWithOnDeviceSTT(audioData, bytesRead)
        }
    }

    private fun stopVoiceCapture() {
        audioCapture.stopRecording()
    }

    private fun streamAudioViaBLE(data: ByteArray, size: Int) {
        // Send via BLE GATT characteristic (see Section 3)
        // May need compression (Opus codec) to fit BLE bandwidth
    }
}
```

### WAV File Export (for testing/debugging)

```kotlin
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder

class WavFileWriter(private val filePath: String, private val sampleRate: Int = 16000) {

    private val outputStream = RandomAccessFile(filePath, "rw")
    private var dataSize = 0

    init {
        // Write WAV header (will be updated with correct size later)
        writeWavHeader()
    }

    fun writeAudioData(pcmData: ByteArray, size: Int) {
        outputStream.write(pcmData, 0, size)
        dataSize += size
    }

    fun close() {
        // Update header with actual data size
        updateWavHeader()
        outputStream.close()
    }

    private fun writeWavHeader() {
        outputStream.apply {
            // RIFF header
            writeBytes("RIFF")
            writeInt(Integer.reverseBytes(36 + dataSize)) // File size - 8
            writeBytes("WAVE")

            // fmt chunk
            writeBytes("fmt ")
            writeInt(Integer.reverseBytes(16)) // Chunk size
            writeShort(java.lang.Short.reverseBytes(1.toShort()).toInt()) // Audio format (1 = PCM)
            writeShort(java.lang.Short.reverseBytes(1.toShort()).toInt()) // Num channels (1 = mono)
            writeInt(Integer.reverseBytes(sampleRate)) // Sample rate
            writeInt(Integer.reverseBytes(sampleRate * 2)) // Byte rate
            writeShort(java.lang.Short.reverseBytes(2.toShort()).toInt()) // Block align
            writeShort(java.lang.Short.reverseBytes(16.toShort()).toInt()) // Bits per sample

            // data chunk
            writeBytes("data")
            writeInt(Integer.reverseBytes(dataSize))
        }
    }

    private fun updateWavHeader() {
        outputStream.seek(4)
        outputStream.writeInt(Integer.reverseBytes(36 + dataSize))
        outputStream.seek(40)
        outputStream.writeInt(Integer.reverseBytes(dataSize))
    }
}
```

### Audio Configuration Recommendations

#### For Voice Commands
```kotlin
MediaRecorder.AudioSource.VOICE_RECOGNITION // Optimized for speech
Sample Rate: 16000 Hz // Standard for speech recognition
Channels: MONO
Encoding: PCM_16BIT
```

#### For High-Quality Audio
```kotlin
MediaRecorder.AudioSource.MIC // Raw microphone input
Sample Rate: 44100 Hz // CD quality
Channels: STEREO
Encoding: PCM_16BIT
```

### Audio Compression for BLE Streaming

Raw 16kHz mono PCM at 16-bit = **32 kB/s** (256 kbps)
BLE typical throughput = **5-20 kB/s** (40-160 kbps)

**Solution: Use Opus codec for compression**

Add dependency:
```gradle
implementation 'com.github.Zelgius:opus-android:0.1.2'
```

Opus can compress voice to 8-32 kbps, making it suitable for BLE streaming.

---

## 3. Bluetooth LE APIs: GATT Service Implementation

### Overview

To stream sensor and audio data from your WearOS app to a desktop companion, you'll create a **BLE GATT Server** on the watch that advertises custom services.

### Recommended Libraries

#### 1. Nordic Semiconductor BLE Library (Recommended)
```gradle
dependencies {
    implementation 'no.nordicsemi.android:ble:2.11.0'
    implementation 'no.nordicsemi.android:ble-ktx:2.11.0'
}
```

**Advantages:**
- Mature, well-tested library
- GATT server support from v2.2+
- Kotlin extensions
- Foreground service integration

#### 2. BleGattCoroutines (Kotlin-first)
```gradle
dependencies {
    implementation 'com.github.Beepiz.BleGattCoroutines:bleGattCoroutines-coroutines:0.5.0'
}
```

**Advantages:**
- Coroutine-based (sequential async code)
- Tested on WearOS
- Simpler API

### GATT Architecture for Your Use Case

```
WristControl GATT Server
├── Sensor Data Service (UUID: custom)
│   ├── Accelerometer Characteristic (READ, NOTIFY)
│   ├── Gyroscope Characteristic (READ, NOTIFY)
│   └── Magnetometer Characteristic (READ, NOTIFY)
├── Gesture Events Service (UUID: custom)
│   └── Gesture Event Characteristic (READ, NOTIFY)
└── Audio Streaming Service (UUID: custom)
    └── Audio Data Characteristic (READ, NOTIFY)
```

### BLE GATT Server Implementation (Kotlin)

```kotlin
import android.bluetooth.*
import android.bluetooth.le.AdvertiseCallback
import android.bluetooth.le.AdvertiseData
import android.bluetooth.le.AdvertiseSettings
import android.bluetooth.le.BluetoothLeAdvertiser
import android.content.Context
import android.os.ParcelUuid
import android.util.Log
import java.util.*

class WristControlGattServer(private val context: Context) {

    companion object {
        // Custom UUIDs for your service
        val SERVICE_UUID: UUID = UUID.fromString("00001234-0000-1000-8000-00805f9b34fb")
        val SENSOR_CHAR_UUID: UUID = UUID.fromString("00001235-0000-1000-8000-00805f9b34fb")
        val GESTURE_CHAR_UUID: UUID = UUID.fromString("00001236-0000-1000-8000-00805f9b34fb")
        val AUDIO_CHAR_UUID: UUID = UUID.fromString("00001237-0000-1000-8000-00805f9b34fb")
    }

    private val bluetoothManager: BluetoothManager by lazy {
        context.getSystemService(Context.BLUETOOTH_SERVICE) as BluetoothManager
    }

    private val bluetoothAdapter: BluetoothAdapter? by lazy {
        bluetoothManager.adapter
    }

    private var gattServer: BluetoothGattServer? = null
    private var bluetoothLeAdvertiser: BluetoothLeAdvertiser? = null

    private val connectedDevices = mutableSetOf<BluetoothDevice>()

    // Characteristics
    private lateinit var sensorCharacteristic: BluetoothGattCharacteristic
    private lateinit var gestureCharacteristic: BluetoothGattCharacteristic
    private lateinit var audioCharacteristic: BluetoothGattCharacteristic

    private val gattServerCallback = object : BluetoothGattServerCallback() {

        override fun onConnectionStateChange(device: BluetoothDevice, status: Int, newState: Int) {
            when (newState) {
                BluetoothProfile.STATE_CONNECTED -> {
                    Log.d("BLE", "Device connected: ${device.address}")
                    connectedDevices.add(device)
                }
                BluetoothProfile.STATE_DISCONNECTED -> {
                    Log.d("BLE", "Device disconnected: ${device.address}")
                    connectedDevices.remove(device)
                }
            }
        }

        override fun onCharacteristicReadRequest(
            device: BluetoothDevice,
            requestId: Int,
            offset: Int,
            characteristic: BluetoothGattCharacteristic
        ) {
            when (characteristic.uuid) {
                SENSOR_CHAR_UUID -> {
                    // Return current sensor data
                    gattServer?.sendResponse(
                        device,
                        requestId,
                        BluetoothGatt.GATT_SUCCESS,
                        0,
                        getCurrentSensorData()
                    )
                }
                else -> {
                    gattServer?.sendResponse(
                        device,
                        requestId,
                        BluetoothGatt.GATT_FAILURE,
                        0,
                        null
                    )
                }
            }
        }

        override fun onDescriptorWriteRequest(
            device: BluetoothDevice,
            requestId: Int,
            descriptor: BluetoothGattDescriptor,
            preparedWrite: Boolean,
            responseNeeded: Boolean,
            offset: Int,
            value: ByteArray
        ) {
            // Client is enabling/disabling notifications
            if (descriptor.uuid == UUID.fromString("00002902-0000-1000-8000-00805f9b34fb")) {
                if (Arrays.equals(BluetoothGattDescriptor.ENABLE_NOTIFICATION_VALUE, value)) {
                    Log.d("BLE", "Notifications enabled for ${descriptor.characteristic.uuid}")
                } else if (Arrays.equals(BluetoothGattDescriptor.DISABLE_NOTIFICATION_VALUE, value)) {
                    Log.d("BLE", "Notifications disabled for ${descriptor.characteristic.uuid}")
                }

                if (responseNeeded) {
                    gattServer?.sendResponse(
                        device,
                        requestId,
                        BluetoothGatt.GATT_SUCCESS,
                        0,
                        value
                    )
                }
            }
        }
    }

    fun startServer() {
        bluetoothLeAdvertiser = bluetoothAdapter?.bluetoothLeAdvertiser

        if (bluetoothLeAdvertiser == null) {
            Log.e("BLE", "BLE advertising not supported")
            return
        }

        // Create GATT server
        gattServer = bluetoothManager.openGattServer(context, gattServerCallback)

        // Create service
        val service = BluetoothGattService(
            SERVICE_UUID,
            BluetoothGattService.SERVICE_TYPE_PRIMARY
        )

        // Create characteristics
        sensorCharacteristic = BluetoothGattCharacteristic(
            SENSOR_CHAR_UUID,
            BluetoothGattCharacteristic.PROPERTY_READ or BluetoothGattCharacteristic.PROPERTY_NOTIFY,
            BluetoothGattCharacteristic.PERMISSION_READ
        )

        gestureCharacteristic = BluetoothGattCharacteristic(
            GESTURE_CHAR_UUID,
            BluetoothGattCharacteristic.PROPERTY_NOTIFY,
            0
        )

        audioCharacteristic = BluetoothGattCharacteristic(
            AUDIO_CHAR_UUID,
            BluetoothGattCharacteristic.PROPERTY_NOTIFY,
            0
        )

        // Add CCC descriptor for notifications
        val descriptor = BluetoothGattDescriptor(
            UUID.fromString("00002902-0000-1000-8000-00805f9b34fb"),
            BluetoothGattDescriptor.PERMISSION_READ or BluetoothGattDescriptor.PERMISSION_WRITE
        )

        sensorCharacteristic.addDescriptor(descriptor.clone() as BluetoothGattDescriptor)
        gestureCharacteristic.addDescriptor(descriptor.clone() as BluetoothGattDescriptor)
        audioCharacteristic.addDescriptor(descriptor.clone() as BluetoothGattDescriptor)

        // Add characteristics to service
        service.addCharacteristic(sensorCharacteristic)
        service.addCharacteristic(gestureCharacteristic)
        service.addCharacteristic(audioCharacteristic)

        // Add service to server
        gattServer?.addService(service)

        // Start advertising
        startAdvertising()
    }

    private fun startAdvertising() {
        val settings = AdvertiseSettings.Builder()
            .setAdvertiseMode(AdvertiseSettings.ADVERTISE_MODE_BALANCED)
            .setConnectable(true)
            .setTimeout(0)
            .setTxPowerLevel(AdvertiseSettings.ADVERTISE_TX_POWER_MEDIUM)
            .build()

        val data = AdvertiseData.Builder()
            .setIncludeDeviceName(true)
            .addServiceUuid(ParcelUuid(SERVICE_UUID))
            .build()

        val advertiseCallback = object : AdvertiseCallback() {
            override fun onStartSuccess(settingsInEffect: AdvertiseSettings) {
                Log.d("BLE", "BLE advertising started successfully")
            }

            override fun onStartFailure(errorCode: Int) {
                Log.e("BLE", "BLE advertising failed with code: $errorCode")
            }
        }

        bluetoothLeAdvertiser?.startAdvertising(settings, data, advertiseCallback)
    }

    fun sendSensorData(accelerometerData: FloatArray, gyroscopeData: FloatArray, timestamp: Long) {
        if (connectedDevices.isEmpty()) return

        // Pack data efficiently (24 bytes total)
        val data = ByteArray(28)
        val buffer = java.nio.ByteBuffer.wrap(data).order(java.nio.ByteOrder.LITTLE_ENDIAN)

        // Accelerometer (12 bytes)
        buffer.putFloat(accelerometerData[0])
        buffer.putFloat(accelerometerData[1])
        buffer.putFloat(accelerometerData[2])

        // Gyroscope (12 bytes)
        buffer.putFloat(gyroscopeData[0])
        buffer.putFloat(gyroscopeData[1])
        buffer.putFloat(gyroscopeData[2])

        // Timestamp (4 bytes - milliseconds since start)
        buffer.putInt((timestamp / 1_000_000).toInt())

        sensorCharacteristic.value = data

        // Notify all connected devices
        connectedDevices.forEach { device ->
            gattServer?.notifyCharacteristicChanged(device, sensorCharacteristic, false)
        }
    }

    fun sendGestureEvent(gestureType: Int, confidence: Float) {
        if (connectedDevices.isEmpty()) return

        val data = ByteArray(5)
        data[0] = gestureType.toByte()
        java.nio.ByteBuffer.wrap(data, 1, 4).putFloat(confidence)

        gestureCharacteristic.value = data

        connectedDevices.forEach { device ->
            gattServer?.notifyCharacteristicChanged(device, gestureCharacteristic, false)
        }
    }

    fun sendAudioData(audioData: ByteArray) {
        if (connectedDevices.isEmpty()) return

        // BLE characteristic max size is typically 512 bytes
        // May need to chunk larger audio buffers
        val chunkSize = 512

        for (offset in audioData.indices step chunkSize) {
            val chunk = audioData.copyOfRange(
                offset,
                minOf(offset + chunkSize, audioData.size)
            )

            audioCharacteristic.value = chunk

            connectedDevices.forEach { device ->
                gattServer?.notifyCharacteristicChanged(device, audioCharacteristic, false)
            }
        }
    }

    fun stopServer() {
        bluetoothLeAdvertiser?.stopAdvertising(object : AdvertiseCallback() {})
        gattServer?.close()
        gattServer = null
        connectedDevices.clear()
    }

    private fun getCurrentSensorData(): ByteArray {
        // Return latest sensor reading
        return ByteArray(28) // Placeholder
    }
}
```

### BLE Throughput Optimization

**BLE Connection Parameters:**
- MTU (Maximum Transmission Unit): 23-517 bytes (default 23)
- Connection Interval: 7.5ms - 4s (affects latency and throughput)
- Data Rate: ~5-20 kB/s typical

**Best Practices:**
1. **Request larger MTU**: `gatt.requestMtu(517)` on client side
2. **Pack data efficiently**: Use binary formats, not JSON
3. **Batch sensor data**: Send 10-20 samples per notification instead of individual readings
4. **Compress audio**: Use Opus codec to reduce bandwidth
5. **Prioritize gestures**: Send gesture events immediately, batch sensor data

### Data Packing Example

```kotlin
// Instead of sending each sensor reading separately:
// BAD: 7 notifications/second × 28 bytes = 196 bytes/s per sensor stream

// Pack 10 samples together:
// GOOD: 1 notification with 10 samples = 280 bytes, sent every 100ms
fun packSensorBatch(samples: List<SensorSample>): ByteArray {
    val buffer = java.nio.ByteBuffer.allocate(samples.size * 28)
        .order(java.nio.ByteOrder.LITTLE_ENDIAN)

    samples.forEach { sample ->
        buffer.putFloat(sample.accelX)
        buffer.putFloat(sample.accelY)
        buffer.putFloat(sample.accelZ)
        buffer.putFloat(sample.gyroX)
        buffer.putFloat(sample.gyroY)
        buffer.putFloat(sample.gyroZ)
        buffer.putInt((sample.timestamp / 1_000_000).toInt())
    }

    return buffer.array()
}
```

---

## 4. Development Environment Setup

### Prerequisites

**System Requirements:**
- OS: Windows, macOS, or Linux (64-bit)
- RAM: 16 GB recommended (8 GB minimum)
- Disk: 8 GB for Android Studio
- Screen: 1280×800 minimum resolution

### Android Studio Installation

1. Download **Android Studio Otter 2** (2025.2.2+) from https://developer.android.com/studio
2. Install with default settings
3. Launch Android Studio Setup Wizard
4. Select "Standard" installation to include:
   - Android SDK
   - Android SDK Platform-Tools
   - Android Emulator
   - Performance (Intel/AMD HAXM)

### SDK Configuration for WearOS

1. Open **SDK Manager** (Tools > SDK Manager)
2. In **SDK Platforms** tab, install:
   - Android 15.0 (API 35) - Latest WearOS 5
   - Android 14.0 (API 34) - WearOS 4
   - Android 13.0 (API 33) - Minimum for Health Services
3. In **SDK Tools** tab, install:
   - Android SDK Build-Tools (latest)
   - Android Emulator
   - Android SDK Platform-Tools
   - Google Play services

### Creating a WearOS Project

**File > New > Project > Wear OS**

Select template:
- **Empty Activity** - Minimal setup
- **Blank Activity (Compose)** - Jetpack Compose (recommended)

Configuration:
```
Name: WristControl
Package: com.yourname.wristcontrol
Language: Kotlin
Minimum SDK: API 30 (Android 11) or higher
```

### Project build.gradle Configuration

```gradle
// Top-level build.gradle
plugins {
    id 'com.android.application' version '8.2.0' apply false
    id 'org.jetbrains.kotlin.android' version '1.9.20' apply false
}
```

### App build.gradle (Wear Module)

```gradle
plugins {
    id 'com.android.application'
    id 'org.jetbrains.kotlin.android'
}

android {
    namespace 'com.yourname.wristcontrol'
    compileSdk 35

    defaultConfig {
        applicationId "com.yourname.wristcontrol"
        minSdk 30
        targetSdk 35
        versionCode 1
        versionName "1.0"
    }

    buildTypes {
        release {
            minifyEnabled false
            proguardFiles getDefaultProguardFile('proguard-android-optimize.txt'), 'proguard-rules.pro'
        }
    }

    compileOptions {
        sourceCompatibility JavaVersion.VERSION_17
        targetCompatibility JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = '17'
    }

    buildFeatures {
        compose true
    }

    composeOptions {
        kotlinCompilerExtensionVersion '1.5.4'
    }
}

dependencies {
    // Wear OS
    implementation 'androidx.wear:wear:1.3.0'
    implementation 'com.google.android.support:wearable:2.9.0'

    // Jetpack Compose for Wear
    implementation platform('androidx.compose:compose-bom:2024.02.00')
    implementation 'androidx.compose.ui:ui'
    implementation 'androidx.compose.material:material-icons-extended'
    implementation 'androidx.wear.compose:compose-material:1.3.0'
    implementation 'androidx.wear.compose:compose-foundation:1.3.0'

    // Health Services (for optimized sensors)
    implementation 'androidx.health:health-services-client:1.1.0-alpha03'

    // Ongoing Activity API
    implementation 'androidx.wear:wear-ongoing:1.1.0'
    implementation 'androidx.core:core:1.12.0'

    // BLE
    implementation 'no.nordicsemi.android:ble:2.7.0'
    implementation 'no.nordicsemi.android:ble-ktx:2.7.0'

    // Coroutines
    implementation 'org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3'

    // Samsung Health Sensor SDK (if using)
    // Download from https://developer.samsung.com/health/sensor
    // implementation files('libs/samsung-health-sensor-sdk.aar')
}
```

### Setting Up WearOS Emulator

**Tools > Device Manager > Create Device**

1. **Category:** Wear OS
2. **Hardware Profile:** Wear OS Small Round (or Large Round)
3. **System Image:**
   - **Wear OS 5** (API 35, Android 15) - Latest
   - **Wear OS 4** (API 33, Android 13) - Stable
4. **AVD Name:** WearOS_5_Emulator
5. Click **Finish**

**Emulator Features:**
- Test data automatically shown (WearOS 4+)
- Different screen sizes and shapes
- Limited root access on WearOS 6+ (signed builds)

**Launch Emulator:**
1. Click ▶️ button in Device Manager
2. Wait for boot (~30-60 seconds)
3. Emulator shows watch face when ready

### Configuring Real Device Testing

#### Enable Developer Options on Galaxy Watch

1. Go to **Settings > About watch > Software**
2. Tap **Software version** 7 times
3. Return to Settings, find **Developer options**
4. Enable **ADB debugging**
5. Enable **Debug over Wi-Fi** (recommended for WearOS)

#### Connect via Wi-Fi (Recommended)

On Watch:
1. **Developer options > Debug over Wi-Fi**
2. Note IP address shown (e.g., 192.168.1.100:5555)

On Computer:
```bash
# Connect to watch
adb connect 192.168.1.100:5555

# Verify connection
adb devices

# Should show:
# 192.168.1.100:5555   device
```

#### Connect via Bluetooth (Alternative)

```bash
# Enable Bluetooth debugging in Android Studio
# Settings > Build, Execution, Deployment > Debugger

# Watch must be paired with phone running Wear OS companion app
# Phone must be connected via USB
```

### Emulator vs Real Device: Decision Matrix

| Feature | Emulator | Real Device |
|---------|----------|-------------|
| **Sensor Testing** | Simulated data | Real sensor behavior |
| **Battery Testing** | Not accurate | Accurate real-world usage |
| **BLE Testing** | Limited/difficult | Full BLE stack |
| **Performance** | Slower | Actual hardware performance |
| **Haptics** | Not available | Real haptic feedback |
| **Setup Time** | 5 minutes | 10+ minutes (pairing, ADB) |
| **Cost** | Free | $200-400 for device |
| **Debugging** | Easy | Requires ADB setup |

**Recommendation for Your Project:**

1. **Early Development:** Use emulator for UI and basic logic
2. **Sensor Development:** MUST use real device (emulator sensors are simulated)
3. **BLE Testing:** MUST use real device (emulator BLE is unreliable)
4. **Battery Optimization:** MUST use real device
5. **Final Testing:** Always test on real Galaxy Watch before release

### Running Your App

**On Emulator:**
1. Select emulator from device dropdown
2. Click Run ▶️ button
3. App installs and launches automatically

**On Real Device:**
```bash
# Build and install APK
./gradlew installDebug

# Launch activity
adb shell am start -n com.yourname.wristcontrol/.MainActivity

# View logs
adb logcat | grep WristControl
```

### Debugging Tips

**Logcat Filtering:**
```bash
# Filter by tag
adb logcat -s SENSOR

# Filter by package
adb logcat | grep "com.yourname.wristcontrol"

# Clear and follow
adb logcat -c && adb logcat
```

**Common Issues:**

1. **"Waiting for target device to come online"**
   - Solution: Reconnect ADB (`adb disconnect && adb connect <ip>`)

2. **"Installation failed with message INSTALL_FAILED_UPDATE_INCOMPATIBLE"**
   - Solution: Uninstall old version (`adb uninstall com.yourname.wristcontrol`)

3. **Sensors not working in emulator**
   - Solution: Use real device (emulator sensors are limited)

---

## 5. Battery Optimization Best Practices

### Critical Battery Guidelines

**Baseline:** WearOS watches typically have 300-450 mAh batteries. Continuous sensor streaming at 100Hz can drain battery in 2-3 hours without optimization.

**Goal:** Achieve 4+ hours of active use (6+ hours target).

### Strategy 1: Use Health Services API

**Instead of:**
```kotlin
// Direct SensorManager - High battery drain
sensorManager.registerListener(
    this,
    accelerometer,
    SensorManager.SENSOR_DELAY_FASTEST
)
```

**Use Health Services (WearOS 3+):**
```kotlin
// Health Services - 20% better battery efficiency
implementation 'androidx.health:health-services-client:1.1.0-alpha03'
```

**Why it's better:**
- Intelligent batching
- Automatic sleep when inactive
- Optimized for W5+ chipsets
- System-level power management

### Strategy 2: Foreground Service with Ongoing Activity

**Always run sensor collection as foreground service** to prevent Android from killing your process.

```kotlin
import android.app.*
import android.content.Context
import android.content.Intent
import android.os.IBinder
import androidx.core.app.NotificationCompat
import androidx.wear.ongoing.OngoingActivity

class SensorStreamingService : Service() {

    companion object {
        private const val NOTIFICATION_ID = 1
        private const val CHANNEL_ID = "wristcontrol_streaming"
    }

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
        startForeground(NOTIFICATION_ID, createOngoingNotification())
    }

    private fun createNotificationChannel() {
        val channel = NotificationChannel(
            CHANNEL_ID,
            "Sensor Streaming",
            NotificationManager.IMPORTANCE_LOW
        ).apply {
            description = "Active sensor streaming to desktop"
        }

        val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        notificationManager.createNotificationChannel(channel)
    }

    private fun createOngoingNotification(): Notification {
        val intent = Intent(this, MainActivity::class.java)
        val pendingIntent = PendingIntent.getActivity(
            this,
            0,
            intent,
            PendingIntent.FLAG_IMMUTABLE
        )

        val notificationBuilder = NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_gesture)
            .setContentTitle("WristControl Active")
            .setContentText("Streaming gestures to desktop")
            .setContentIntent(pendingIntent)
            .setOngoing(true)
            .setCategory(NotificationCompat.CATEGORY_WORKOUT)
            .setVisibility(NotificationCompat.VISIBILITY_PUBLIC)

        // Create OngoingActivity for watch face integration
        val ongoingActivity = OngoingActivity.Builder(
            applicationContext,
            NOTIFICATION_ID,
            notificationBuilder
        )
            .setStaticIcon(R.drawable.ic_gesture)
            .setTouchIntent(pendingIntent)
            .setTitle("WristControl")
            .setStatus("Active")
            .build()

        ongoingActivity.apply(applicationContext)

        return notificationBuilder.build()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        // Start sensor streaming here
        return START_STICKY // Restart if killed
    }

    override fun onDestroy() {
        super.onDestroy()
        // Stop sensors and BLE
    }
}
```

**Start service:**
```kotlin
val serviceIntent = Intent(context, SensorStreamingService::class.java)
ContextCompat.startForegroundService(context, serviceIntent)
```

### Strategy 3: Avoid Wakelocks

**DON'T:**
```kotlin
// This prevents device from sleeping - HUGE battery drain
val wakeLock = powerManager.newWakeLock(
    PowerManager.PARTIAL_WAKE_LOCK,
    "WristControl::SensorLock"
)
wakeLock.acquire()
```

**DO:**
```kotlin
// Use processor time only during callbacks
override fun onSensorChanged(event: SensorEvent?) {
    // Process quickly and return
    // System holds wakelock automatically during this callback
    processData(event)
}
```

**Exception:** WorkManager and JobScheduler automatically hold wakelocks - this is OK.

### Strategy 4: Dynamic Sampling Rate

Adjust sensor frequency based on activity:

```kotlin
class AdaptiveSensorManager(private val sensorManager: SensorManager) {

    private var currentMode = SamplingMode.ACTIVE

    enum class SamplingMode(val delay: Int) {
        ACTIVE(SensorManager.SENSOR_DELAY_FASTEST),      // ~100Hz - user actively controlling
        MODERATE(SensorManager.SENSOR_DELAY_GAME),       // ~50Hz - passive monitoring
        IDLE(SensorManager.SENSOR_DELAY_NORMAL)          // ~5Hz - background only
    }

    fun setMode(mode: SamplingMode) {
        if (mode == currentMode) return

        currentMode = mode

        // Unregister and re-register with new delay
        sensorManager.unregisterListener(sensorListener)
        accelerometer?.let {
            sensorManager.registerListener(
                sensorListener,
                it,
                mode.delay
            )
        }
    }

    // Auto-detect idle state
    private fun checkIdleState() {
        if (timeSinceLastGesture > 30_000) { // 30 seconds
            setMode(SamplingMode.IDLE)
        }
    }
}
```

### Strategy 5: Data Batching

**Send 10-20 sensor samples together** instead of streaming each individually:

```kotlin
class BatchingSensorListener : SensorEventListener {

    private val sensorBatch = mutableListOf<SensorReading>()
    private val batchSize = 10

    override fun onSensorChanged(event: SensorEvent?) {
        event?.let {
            sensorBatch.add(SensorReading(
                it.values[0], it.values[1], it.values[2], it.timestamp
            ))

            if (sensorBatch.size >= batchSize) {
                // Send batch via BLE
                bleServer.sendSensorBatch(sensorBatch)
                sensorBatch.clear()
            }
        }
    }
}
```

**Reduces:**
- BLE transmission overhead
- CPU wake-ups
- Radio power consumption

### Strategy 6: Stop Unnecessary Sensors

```kotlin
// Only enable sensors when needed
class SmartSensorController {

    fun onUserActivatesGestureMode() {
        // Enable accelerometer + gyroscope
        enableMotionSensors()
    }

    fun onUserActivatesVoiceMode() {
        // Disable motion sensors, enable microphone
        disableMotionSensors()
        enableMicrophone()
    }

    fun onUserPauseSession() {
        // Disable everything
        disableAllSensors()
    }

    fun onWatchScreenOff() {
        // Reduce to minimal monitoring or stop
        if (userPreference.allowBackgroundTracking) {
            setMode(SamplingMode.IDLE)
        } else {
            disableAllSensors()
        }
    }
}
```

### Strategy 7: BLE Connection Optimization

```kotlin
// Request optimal connection parameters
fun optimizeBleConnection(gatt: BluetoothGatt) {
    // Balance between latency and power
    gatt.requestConnectionPriority(BluetoothGatt.CONNECTION_PRIORITY_BALANCED)

    // For low-latency gesture control:
    // gatt.requestConnectionPriority(BluetoothGatt.CONNECTION_PRIORITY_HIGH)
    // WARNING: Higher power consumption
}
```

### Strategy 8: Off-body Detection

**Stop sensors when watch is not being worn:**

```kotlin
class OffBodyDetector(context: Context) {

    private val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val offBodySensor = sensorManager.getDefaultSensor(Sensor.TYPE_LOW_LATENCY_OFFBODY_DETECT)

    private val offBodyListener = object : SensorEventListener {
        override fun onSensorChanged(event: SensorEvent?) {
            event?.let {
                val isWorn = it.values[0] == 0f // 0 = on body, 1 = off body

                if (isWorn) {
                    // Resume sensor streaming
                    enableSensors()
                } else {
                    // Stop sensors to save battery
                    disableSensors()
                }
            }
        }

        override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
    }

    fun startMonitoring() {
        offBodySensor?.let {
            sensorManager.registerListener(
                offBodyListener,
                it,
                SensorManager.SENSOR_DELAY_NORMAL
            )
        }
    }
}
```

### Strategy 9: Screen State Awareness

```kotlin
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter

class ScreenStateMonitor(private val context: Context) {

    private val screenReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            when (intent?.action) {
                Intent.ACTION_SCREEN_ON -> {
                    // User looking at watch - enable full rate
                    sensorController.setMode(SamplingMode.ACTIVE)
                }
                Intent.ACTION_SCREEN_OFF -> {
                    // Screen off - reduce to minimal or stop
                    if (ongoingActivity) {
                        sensorController.setMode(SamplingMode.IDLE)
                    } else {
                        sensorController.stop()
                    }
                }
            }
        }
    }

    fun register() {
        val filter = IntentFilter().apply {
            addAction(Intent.ACTION_SCREEN_ON)
            addAction(Intent.ACTION_SCREEN_OFF)
        }
        context.registerReceiver(screenReceiver, filter)
    }
}
```

### Battery Consumption Estimates

| Component | Power Draw | Duration on 400mAh |
|-----------|------------|-------------------|
| Accelerometer @ 100Hz | ~5 mA | 80 hours |
| Gyroscope @ 100Hz | ~5 mA | 80 hours |
| Both @ 100Hz | ~10 mA | 40 hours |
| Microphone recording | ~15 mA | 26 hours |
| BLE active streaming | ~10-15 mA | 26-40 hours |
| Screen on | ~80-120 mA | 3-5 hours |
| **Combined worst case** | ~150 mA | **2.6 hours** |
| **Optimized (batching, moderate rate)** | ~40 mA | **10 hours** |

### Recommended Power Profile

```kotlin
data class PowerProfile(
    val name: String,
    val sensorRate: Int,
    val batchSize: Int,
    val enabledWhenScreenOff: Boolean
)

val profiles = mapOf(
    "active_control" to PowerProfile(
        name = "Active Control",
        sensorRate = SensorManager.SENSOR_DELAY_FASTEST,
        batchSize = 5,
        enabledWhenScreenOff = false
    ),
    "background_monitoring" to PowerProfile(
        name = "Background",
        sensorRate = SensorManager.SENSOR_DELAY_GAME,
        batchSize = 20,
        enabledWhenScreenOff = true
    ),
    "power_saver" to PowerProfile(
        name = "Power Saver",
        sensorRate = SensorManager.SENSOR_DELAY_NORMAL,
        batchSize = 50,
        enabledWhenScreenOff = false
    )
)
```

**Expected Battery Life:**
- **Active Control:** 4-6 hours continuous use
- **Background Monitoring:** 8-12 hours
- **Power Saver:** 16-24 hours

---

## 6. On-Device Gesture Detection

### Overview

Processing gestures on-watch reduces:
- **Latency** (no need to stream raw data and wait for desktop processing)
- **Battery usage** (less BLE traffic)
- **Privacy** (gesture patterns stay on device)

### Approach 1: Algorithm-Based Detection (Lightweight)

Perfect for simple gestures like finger pinches (as used by DoublePoint WowMouse).

#### Finger Pinch Detection Algorithm

```kotlin
class PinchGestureDetector {

    private val accelMagnitudeHistory = ArrayDeque<Float>(10)
    private val spikeThreshold = 2.5f // Adjust based on testing
    private val spikeWindow = 100_000_000L // 100ms in nanoseconds
    private var lastSpikeTime = 0L
    private var lastGestureTime = 0L

    fun processAccelerometer(x: Float, y: Float, z: Float, timestamp: Long): GestureEvent? {
        // Calculate magnitude of acceleration
        val magnitude = kotlin.math.sqrt(x*x + y*y + z*z)

        // Track magnitude history
        accelMagnitudeHistory.addLast(magnitude)
        if (accelMagnitudeHistory.size > 10) {
            accelMagnitudeHistory.removeFirst()
        }

        // Calculate baseline (average of recent history)
        val baseline = accelMagnitudeHistory.average().toFloat()

        // Detect spike (finger pinch creates sudden acceleration spike)
        val spike = magnitude - baseline

        if (spike > spikeThreshold) {
            // Potential tap detected
            val timeSinceLastSpike = timestamp - lastSpikeTime
            lastSpikeTime = timestamp

            // Check if this is a double-tap
            if (timeSinceLastSpike in 100_000_000..500_000_000) { // 100-500ms
                return GestureEvent.DOUBLE_TAP
            }

            // Check if enough time has passed since last gesture
            if (timestamp - lastGestureTime > 300_000_000) { // 300ms debounce
                lastGestureTime = timestamp
                return GestureEvent.TAP
            }
        }

        return null
    }

    enum class GestureEvent {
        TAP,
        DOUBLE_TAP,
        HOLD
    }
}
```

#### Wrist Rotation (Flick) Detection

```kotlin
class WristFlickDetector {

    private val rotationHistory = ArrayDeque<Float>(20)
    private val flickThreshold = 3.0f // rad/s

    fun processGyroscope(x: Float, y: Float, z: Float, timestamp: Long): FlickDirection? {
        // Check rotation rate on each axis
        val maxRotation = maxOf(kotlin.math.abs(x), kotlin.math.abs(y), kotlin.math.abs(z))

        if (maxRotation > flickThreshold) {
            return when {
                kotlin.math.abs(x) > flickThreshold -> {
                    if (x > 0) FlickDirection.PITCH_UP else FlickDirection.PITCH_DOWN
                }
                kotlin.math.abs(y) > flickThreshold -> {
                    if (y > 0) FlickDirection.ROLL_RIGHT else FlickDirection.ROLL_LEFT
                }
                kotlin.math.abs(z) > flickThreshold -> {
                    if (z > 0) FlickDirection.YAW_RIGHT else FlickDirection.YAW_LEFT
                }
                else -> null
            }
        }

        return null
    }

    enum class FlickDirection {
        PITCH_UP, PITCH_DOWN,
        ROLL_LEFT, ROLL_RIGHT,
        YAW_LEFT, YAW_RIGHT
    }
}
```

#### Palm Orientation Detection

```kotlin
class PalmOrientationDetector {

    fun detectOrientation(accelX: Float, accelY: Float, accelZ: Float): PalmOrientation {
        // When palm is facing up, gravity points in negative Z direction
        // When palm is facing down, gravity points in positive Z direction

        return when {
            accelZ < -7.0 -> PalmOrientation.UP      // Palm up (resting)
            accelZ > 7.0 -> PalmOrientation.DOWN     // Palm down
            accelX > 7.0 -> PalmOrientation.RIGHT    // Watch on right side
            accelX < -7.0 -> PalmOrientation.LEFT    // Watch on left side
            accelY > 7.0 -> PalmOrientation.FORWARD  // Palm forward (natural gesture position)
            accelY < -7.0 -> PalmOrientation.BACK    // Palm back
            else -> PalmOrientation.NEUTRAL
        }
    }

    enum class PalmOrientation {
        UP, DOWN, LEFT, RIGHT, FORWARD, BACK, NEUTRAL
    }
}
```

### Approach 2: Machine Learning with TensorFlow Lite

For complex gesture recognition (custom gestures, personalization).

#### Setup TensorFlow Lite

Add to build.gradle:
```gradle
dependencies {
    implementation 'org.tensorflow:tensorflow-lite:2.14.0'
    implementation 'org.tensorflow:tensorflow-lite-support:0.4.4'
    implementation 'org.tensorflow:tensorflow-lite-gpu:2.14.0' // Optional GPU acceleration
}
```

#### Model Architecture

**Input:** 6 features (accel X/Y/Z, gyro X/Y/Z) × 50 timesteps = 300 features
**Output:** Probability distribution over gesture classes

```python
# Training script (run on desktop/cloud)
import tensorflow as tf
from tensorflow import keras

# Define model
model = keras.Sequential([
    # Input: (batch, 50 timesteps, 6 features)
    keras.layers.LSTM(64, return_sequences=True, input_shape=(50, 6)),
    keras.layers.Dropout(0.2),
    keras.layers.LSTM(32),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(num_gestures, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val))

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save model
with open('gesture_model.tflite', 'wb') as f:
    f.write(tflite_model)
```

#### TFLite Inference on WearOS

```kotlin
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

class TFLiteGestureRecognizer(modelPath: String) {

    private val interpreter: Interpreter
    private val inputSize = 50 * 6 * 4 // 50 timesteps × 6 features × 4 bytes (float)
    private val numClasses = 10

    private val sensorWindow = ArrayDeque<FloatArray>(50)

    init {
        val fileChannel = FileInputStream(modelPath).channel
        val modelBuffer = fileChannel.map(
            FileChannel.MapMode.READ_ONLY,
            0,
            fileChannel.size()
        )

        interpreter = Interpreter(modelBuffer)
    }

    fun processSensorData(accelX: Float, accelY: Float, accelZ: Float,
                          gyroX: Float, gyroY: Float, gyroZ: Float): GesturePrediction? {
        // Add to sliding window
        sensorWindow.addLast(floatArrayOf(accelX, accelY, accelZ, gyroX, gyroY, gyroZ))

        if (sensorWindow.size > 50) {
            sensorWindow.removeFirst()
        }

        // Only predict when window is full
        if (sensorWindow.size < 50) return null

        // Prepare input tensor
        val inputBuffer = ByteBuffer.allocateDirect(inputSize)
            .order(ByteOrder.nativeOrder())

        for (sample in sensorWindow) {
            for (value in sample) {
                inputBuffer.putFloat(value)
            }
        }

        // Prepare output tensor
        val outputBuffer = Array(1) { FloatArray(numClasses) }

        // Run inference
        interpreter.run(inputBuffer, outputBuffer)

        // Get prediction
        val probabilities = outputBuffer[0]
        val maxIndex = probabilities.indices.maxByOrNull { probabilities[it] } ?: 0
        val confidence = probabilities[maxIndex]

        return if (confidence > 0.7f) { // Confidence threshold
            GesturePrediction(
                gestureClass = maxIndex,
                confidence = confidence
            )
        } else {
            null
        }
    }

    fun close() {
        interpreter.close()
    }

    data class GesturePrediction(
        val gestureClass: Int,
        val confidence: Float
    )
}
```

#### Data Collection for Training

```kotlin
class GestureDataCollector(private val outputFile: File) {

    private val writer = outputFile.bufferedWriter()
    private var isRecording = false
    private val currentGestureLabel = ""

    fun startRecording(gestureLabel: String) {
        isRecording = true
        currentGestureLabel = gestureLabel
        writer.write("# Gesture: $gestureLabel\n")
    }

    fun recordSample(accelX: Float, accelY: Float, accelZ: Float,
                     gyroX: Float, gyroY: Float, gyroZ: Float,
                     timestamp: Long) {
        if (!isRecording) return

        writer.write("$timestamp,$accelX,$accelY,$accelZ,$gyroX,$gyroY,$gyroZ\n")
    }

    fun stopRecording() {
        isRecording = false
        writer.write("\n")
        writer.flush()
    }

    fun close() {
        writer.close()
    }
}

// Usage: Collect data for each gesture type
// 1. Tap
// 2. Double tap
// 3. Wrist flick left
// 4. Wrist flick right
// etc.
```

### Approach 3: Hybrid (Recommended)

Combine both approaches for best results:

```kotlin
class HybridGestureDetector {

    private val pinchDetector = PinchGestureDetector()
    private val flickDetector = WristFlickDetector()
    private val tfliteDetector = TFLiteGestureRecognizer("gesture_model.tflite")

    fun processIMUData(
        accelX: Float, accelY: Float, accelZ: Float,
        gyroX: Float, gyroY: Float, gyroZ: Float,
        timestamp: Long
    ): DetectedGesture? {
        // Fast algorithm-based detection first (low latency)
        pinchDetector.processAccelerometer(accelX, accelY, accelZ, timestamp)?.let {
            return DetectedGesture(it.name, 1.0f, source = "algorithm")
        }

        flickDetector.processGyroscope(gyroX, gyroY, gyroZ, timestamp)?.let {
            return DetectedGesture(it.name, 1.0f, source = "algorithm")
        }

        // ML-based detection for complex gestures
        tfliteDetector.processSensorData(accelX, accelY, accelZ, gyroX, gyroY, gyroZ)?.let {
            return DetectedGesture(
                gestureType = "custom_${it.gestureClass}",
                confidence = it.confidence,
                source = "ml"
            )
        }

        return null
    }

    data class DetectedGesture(
        val gestureType: String,
        val confidence: Float,
        val source: String
    )
}
```

**Benefits:**
- Simple gestures detected instantly (algorithm)
- Complex gestures recognized accurately (ML)
- Fallback if ML model not available

### Performance Considerations

**Algorithm-based detection:**
- Latency: <5ms
- CPU: Negligible
- Battery: Minimal impact

**TensorFlow Lite inference:**
- Latency: 10-30ms (CPU), 5-10ms (GPU)
- CPU: Moderate (~5-10% continuous)
- Battery: ~2-5 mA additional

**Recommendation:** Use algorithm-based for primary gestures, ML for advanced features only.

---

## Recommended Libraries Summary

### Essential
- **androidx.wear:wear** - Core WearOS components
- **androidx.wear.compose:compose-material** - UI framework
- **no.nordicsemi.android:ble-ktx** - BLE GATT server
- **androidx.wear:wear-ongoing** - Foreground service integration

### Sensor Access
- **Standard SensorManager** - High-frequency (50-100Hz) IMU data
- **com.samsung.android:health-sensor-control** - Battery-optimized (25Hz)

### Audio
- **AudioRecord** - Microphone capture (built-in)

### Machine Learning (Optional)
- **org.tensorflow:tensorflow-lite** - On-device gesture recognition

### Recommended Development Stack
```
WearOS App (Kotlin)
├── UI: Jetpack Compose for Wear
├── Sensors: SensorManager (fast) or Health Services (efficient)
├── Audio: AudioRecord
├── BLE: Nordic BLE Library
├── Gestures: Algorithm + TFLite (hybrid)
└── Background: Foreground Service + Ongoing Activity
```

---

## Next Steps

1. **Set up development environment** (Android Studio + WearOS emulator)
2. **Implement sensor streaming** (start with accelerometer only)
3. **Build BLE GATT server** (test with desktop client)
4. **Add algorithm-based gesture detection** (finger pinch)
5. **Implement audio recording** (test quality)
6. **Optimize battery usage** (foreground service, batching)
7. **Test on real Galaxy Watch** (essential for sensors/BLE)

---

## Sources & Documentation

### Samsung Developer Resources
- [Understanding Galaxy Watch Accelerometer Data](https://developer.samsung.com/sdp/blog/en/2025/04/10/understanding-and-converting-galaxy-watch-accelerometer-data)
- [Understanding Sensor Ranges for Galaxy Watch](https://developer.samsung.com/sdp/blog/en/2024/09/10/understanding-sensor-ranges-for-galaxy-watch)
- [Samsung Health Sensor SDK](https://developer.samsung.com/health/sensor/overview.html)

### Android Developer Documentation
- [Android Sensor APIs](https://developer.android.com/guide/topics/sensors/sensors_overview)
- [Bluetooth Low Energy Overview](https://developer.android.com/guide/topics/connectivity/bluetooth/ble-overview)
- [AudioRecord API Reference](https://developer.android.com/reference/android/media/AudioRecord)
- [WearOS Development](https://developer.android.com/wear)
- [Ongoing Activities](https://developer.android.com/training/wearables/notifications/ongoing-activity)
- [Conserve Power on WearOS](https://developer.android.com/training/wearables/apps/power)

### GitHub Examples
- [WearOS Sensors Library](https://github.com/GeoTecINIT/WearOSSensors)
- [Nordic BLE Library](https://github.com/NordicSemiconductor/Android-BLE-Library)
- [Kotlin BluetoothLeGatt Example](https://github.com/objectsyndicate/Kotlin-BluetoothLeGatt)
- [Android Wear Motion Sensors](https://github.com/estherjk/AndroidWearMotionSensors)
- [TensorFlow Lite Gesture Recognition](https://github.com/DevHeadsCommunity/training-gesture-with-tflite)

### Technical Articles
- [Deep Dive into BLE on Android](https://qubika.com/blog/what-is-bluetooth-low-energy-ble/)
- [Motion Gesture Detection Using TensorFlow](https://lembergsolutions.com/blog/motion-gesture-detection-using-tensorflow-android)
- [WearOS Development Guide 2025](https://topflightapps.com/ideas/how-to-develop-a-wearable-app-for-wear-os-and-watchos/)

---

*This guide compiled from latest WearOS development practices as of January 2026.*
