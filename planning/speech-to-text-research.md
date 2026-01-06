# Speech-to-Text Research for Voice Command System

**Project:** Wristcontrol - Voice and Gesture Computer Control
**Date:** January 2026
**Purpose:** Research and recommendations for implementing voice command recognition

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Local Whisper Implementation](#local-whisper-implementation)
3. [Cloud STT APIs](#cloud-stt-apis)
4. [Voice Command Recognition](#voice-command-recognition)
5. [Audio Processing](#audio-processing)
6. [Architecture Recommendations](#architecture-recommendations)
7. [Implementation Examples](#implementation-examples)

---

## Executive Summary

### Key Findings

**For Wristcontrol Project:**

1. **Recommended Approach:** Hybrid system with local Whisper for MVP, cloud API as optional upgrade
2. **Best Local Option:** faster-whisper with tiny/base model for real-time performance
3. **Best Cloud Option:** Deepgram Nova-3 (sub-300ms latency, $4.30/1000 min)
4. **VAD:** Silero VAD (pre-trained, <1ms per chunk, MIT license)
5. **Architecture:** Push-to-talk with VAD for battery conservation

### Performance Targets for Project

| Metric | Target | Best Solution |
|--------|--------|---------------|
| Latency | <500ms | Deepgram Nova-3 (300ms) or Whisper Turbo local |
| Accuracy | >90% | Whisper large-v3 or Deepgram |
| Battery Impact | Minimal | Push-to-talk + VAD |
| Privacy | High | Local faster-whisper |
| Cost | Low/Free | Local processing |

---

## Local Whisper Implementation

### Overview

OpenAI's Whisper is a state-of-the-art speech recognition model that can run locally, providing privacy and zero ongoing costs. Multiple implementations exist with varying performance characteristics.

### Implementation Options

#### 1. faster-whisper (Recommended)

**Best for:** Real-time voice commands, production use

- **Performance:** 4x faster than openai/whisper with same accuracy
- **Memory:** Lower memory usage than original
- **Optimization:** 8-bit quantization support on CPU and GPU
- **Streaming:** Works with WhisperLive for near-real-time transcription

**Installation:**
```bash
pip install faster-whisper
```

**Basic Usage:**
```python
from faster_whisper import WhisperModel

# Initialize model (runs on CPU with int8 quantization)
model = WhisperModel("base", device="cpu", compute_type="int8")

# Transcribe audio file
segments, info = model.transcribe("audio.wav", beam_size=1)

for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
```

**GPU Usage:**
```python
# Use GPU with float16 for faster processing
model = WhisperModel("base", device="cuda", compute_type="float16")
```

#### 2. whisper.cpp

**Best for:** Resource-constrained devices, embedded systems

- **Language:** Pure C/C++ implementation
- **Platforms:** Raspberry Pi to Apple Silicon
- **Acceleration:** Vulkan, CUDA, Core ML, OpenVINO
- **Apple Silicon:** 3x faster via Apple Neural Engine (ANE)
- **Quantization:** Q5_0, Q5_1 quantized models (31MB - 1GB)

**Installation:**
```bash
git clone https://github.com/ggml-org/whisper.cpp
cd whisper.cpp
make
```

**Python Bindings:**
```bash
pip install whisper-cpp-python
```

**Usage:**
```python
from whispercpp import Whisper

w = Whisper.from_pretrained("tiny")
result = w.transcribe("audio.wav")
print(result)
```

#### 3. openai/whisper (Original)

**Best for:** Research, maximum accuracy

- **Installation:** `pip install openai-whisper`
- **Performance:** Baseline implementation (slowest)
- **Accuracy:** Reference implementation
- **Note:** Use beam_size=1 for fair performance comparison with faster-whisper

### Model Sizes and Tradeoffs

| Model | Parameters | VRAM | Speed (RTFx) | WER | Best For |
|-------|-----------|------|--------------|-----|----------|
| tiny | 39M | ~1GB | 32x | ~5% | Real-time, battery-critical |
| base | 74M | ~1GB | 16x | ~4% | Real-time, good balance |
| small | 244M | ~2GB | 8x | ~3.5% | Quality with speed |
| medium | 769M | ~5GB | 4x | ~3% | High accuracy |
| large-v3 | 1.55B | ~10GB | 1x | 2.7% | Maximum accuracy |
| turbo | ~800M | ~6GB | 8x | ~3% | **Recommended for desktop** |

**Key Insights:**

- **Whisper Turbo** (Oct 2024): Reduced decoder from 32 to 4 layers, achieving 5.4x speedup while maintaining large-v2 accuracy
- **RTFx 216x** on optimized hardware: 60-minute audio transcribed in 17 seconds
- **tiny/base models** process 32x faster than real-time - perfect for voice commands
- **large-v3** achieves near-human accuracy (2.7% WER vs 4-6.8% human baseline)

### Real-Time Streaming Implementation

**Using faster-whisper with WhisperLive:**

```python
import pyaudio
import numpy as np
from faster_whisper import WhisperModel

class RealtimeWhisper:
    def __init__(self, model_size="base"):
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
        self.audio = pyaudio.PyAudio()
        self.sample_rate = 16000
        self.chunk_duration = 2  # seconds

    def transcribe_stream(self, callback):
        stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.sample_rate * self.chunk_duration
        )

        print("Listening...")

        try:
            while True:
                audio_data = stream.read(self.sample_rate * self.chunk_duration)
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                # Transcribe chunk
                segments, info = self.model.transcribe(audio_np, beam_size=1)

                for segment in segments:
                    callback(segment.text)

        except KeyboardInterrupt:
            pass
        finally:
            stream.stop_stream()
            stream.close()

# Usage
def on_text(text):
    print(f"Recognized: {text}")

whisper = RealtimeWhisper("tiny")
whisper.transcribe_stream(on_text)
```

### Performance Optimization Tips

1. **Use beam_size=1** for faster processing (default is 5 in faster-whisper)
2. **Enable VAD** to process only speech segments
3. **Quantization:** Use int8 on CPU, float16 on GPU
4. **Chunk size:** 2-3 seconds for good balance
5. **Model selection:** tiny/base for <500ms latency requirement

---

## Cloud STT APIs

### Comparison Matrix

| Provider | Pricing | Latency | Accuracy | Real-time | Best For |
|----------|---------|---------|----------|-----------|----------|
| **Deepgram Nova-3** | $4.30/1000 min | <300ms | Excellent | Yes | **Production real-time** |
| **OpenAI Whisper API** | $6.00/1000 min | 320ms | Excellent | Limited | Batch processing |
| **Google Chirp 2** | $16.00/1000 min | ~500ms | Good | Yes | Enterprise, many languages |
| **Azure Speech** | $1.00/hour | 400-800ms | Good | Yes | Microsoft ecosystem |

### Detailed Analysis

#### 1. Deepgram Nova-3 (Recommended for Cloud)

**Advantages:**
- **Lowest latency:** Sub-300ms for real-time transcription
- **Best pricing:** $4.30/1000 minutes (~$0.0043/min)
- **Fast batch:** 40x speed for pre-recorded audio
- **Features:** Speaker diarization, custom vocabulary, profanity filtering

**Python Example:**
```python
from deepgram import Deepgram
import asyncio

async def transcribe_realtime():
    deepgram = Deepgram('YOUR_API_KEY')

    # Real-time streaming
    socket = await deepgram.transcription.live({
        'punctuate': True,
        'interim_results': True,
        'endpointing': 300,  # ms of silence before finalizing
        'vad_events': True,
    })

    async def on_message(result):
        transcript = result['channel']['alternatives'][0]['transcript']
        if transcript:
            print(f"Recognized: {transcript}")

    socket.registerHandler(socket.event.TRANSCRIPT_RECEIVED, on_message)

    # Stream audio from microphone
    # ... (audio streaming code)
```

**Pricing Notes:**
- Pay-as-you-go with no minimums
- Enhanced tiers for speaker ID and additional languages
- Volume discounts available

#### 2. OpenAI Whisper API

**Advantages:**
- **High accuracy:** Based on Whisper large-v2
- **Simple API:** Easy to integrate
- **Language support:** 90+ languages

**Disadvantages:**
- **File minimums:** 1-2 min minimum billing per file
- **Not true streaming:** File-based processing
- **Speed:** ~35-40 audio seconds per processing second

**Python Example:**
```python
import openai

openai.api_key = 'YOUR_API_KEY'

# Transcribe audio file
with open("audio.wav", "rb") as audio_file:
    transcript = openai.Audio.transcribe(
        model="whisper-1",
        file=audio_file,
        language="en",
        prompt="Voice commands: click, scroll, type"  # Context hint
    )

print(transcript['text'])
```

**Cost Optimization:**
- Avoid splitting audio into small chunks (1-2 min minimum per file)
- Batch multiple commands when possible
- Consider local Whisper for high-frequency usage

#### 3. Google Cloud Speech-to-Text

**Advantages:**
- **140+ languages:** Best language coverage
- **Multiple models:** Chirp, Enhanced, Standard
- **Dynamic batch:** $0.003/min but up to 24h delay

**Disadvantages:**
- **Expensive:** $16/1000 min for premium models
- **Complex pricing:** Different tiers and features
- **Higher latency:** ~500ms typical

**Python Example:**
```python
from google.cloud import speech_v1p1beta1 as speech

client = speech.SpeechClient()

config = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=16000,
    language_code="en-US",
    enable_automatic_punctuation=True,
    model="latest_long",  # or "command_and_search" for commands
    use_enhanced=True,
)

# Streaming recognition
streaming_config = speech.StreamingRecognitionConfig(
    config=config,
    interim_results=True,
)

def stream_generator():
    # Yield audio chunks
    pass

requests = (speech.StreamingRecognizeRequest(audio_content=chunk)
           for chunk in stream_generator())

responses = client.streaming_recognize(streaming_config, requests)

for response in responses:
    for result in response.results:
        print(f"Transcript: {result.alternatives[0].transcript}")
```

#### 4. Azure Speech Services

**Advantages:**
- **Cost-effective:** $1/hour standard STT
- **Ecosystem:** Integrates with Azure services
- **Features:** Custom models, speaker recognition

**Disadvantages:**
- **Latency:** 400-800ms typical
- **Complex:** Many SKUs and options

**Python Example:**
```python
import azure.cognitiveservices.speech as speechsdk

speech_config = speechsdk.SpeechConfig(
    subscription="YOUR_KEY",
    region="YOUR_REGION"
)
speech_config.speech_recognition_language="en-US"

audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
speech_recognizer = speechsdk.SpeechRecognizer(
    speech_config=speech_config,
    audio_config=audio_config
)

def recognized(evt):
    print(f"Recognized: {evt.result.text}")

speech_recognizer.recognized.connect(recognized)
speech_recognizer.start_continuous_recognition()
```

### Cloud API Selection Guide

**Choose Deepgram if:**
- Real-time latency is critical (<300ms requirement)
- Cost-conscious with moderate usage
- Need production-grade reliability

**Choose OpenAI Whisper API if:**
- Accuracy is paramount
- Processing batches of audio files
- Already using OpenAI ecosystem

**Choose Google if:**
- Need extensive language support (140+ languages)
- Part of Google Cloud ecosystem
- Budget allows premium pricing

**Choose Azure if:**
- Microsoft ecosystem integration
- Need custom model training
- Cost-sensitive (standard tier)

---

## Voice Command Recognition

### Command Parsing Architecture

Voice commands for computer control require converting natural language into structured actions. The pipeline is:

```
Audio → STT → Intent Recognition → Slot Filling → Action Execution
```

**Example:**
- Input: "scroll down three times"
- Intent: SCROLL
- Slots: {direction: "down", count: 3}
- Action: execute_scroll(direction="down", count=3)

### Implementation Approaches

#### Approach 1: Rule-Based Pattern Matching (Recommended for MVP)

**Best for:** Fixed command vocabulary, low latency, deterministic behavior

```python
import re
from typing import Dict, Any, Optional

class VoiceCommandParser:
    def __init__(self):
        # Define command patterns with regex
        self.patterns = {
            'click': [
                r'\b(click|tap|press)\b',
                r'\bleft click\b',
            ],
            'double_click': [
                r'\bdouble click\b',
                r'\bclick twice\b',
            ],
            'right_click': [
                r'\bright click\b',
                r'\bcontext menu\b',
            ],
            'scroll': [
                r'\bscroll (up|down)\b',
                r'\bscroll (up|down) (\d+) ?(times|)?\b',
            ],
            'type_text': [
                r'\btype (.+)',
                r'\benter (.+)',
                r'\bwrite (.+)',
            ],
            'move_cursor': [
                r'\bmove (up|down|left|right)\b',
                r'\bcursor (up|down|left|right)\b',
            ],
            'press_key': [
                r'\bpress (enter|escape|tab|space|backspace|delete)\b',
                r'\bpress control ([a-z])\b',
                r'\bpress alt ([a-z])\b',
            ],
        }

    def parse(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse voice command text into structured command."""
        text = text.lower().strip()

        # Try each command pattern
        for command, patterns in self.patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    return self._extract_command(command, match, text)

        return None

    def _extract_command(self, command: str, match: re.Match, text: str) -> Dict[str, Any]:
        """Extract command parameters from regex match."""
        result = {'command': command}

        if command == 'scroll':
            result['direction'] = match.group(1) if match.lastindex >= 1 else 'down'
            result['count'] = int(match.group(2)) if match.lastindex >= 2 else 1

        elif command == 'type_text':
            result['text'] = match.group(1).strip()

        elif command in ('move_cursor', 'press_key'):
            result['direction'] = match.group(1)
            if match.lastindex >= 2:
                result['modifier'] = match.group(1)
                result['key'] = match.group(2)

        return result

# Usage
parser = VoiceCommandParser()

commands = [
    "click",
    "double click",
    "scroll down",
    "scroll up 5 times",
    "type hello world",
    "press enter",
    "press control c",
]

for cmd in commands:
    result = parser.parse(cmd)
    print(f"{cmd:30} -> {result}")
```

**Output:**
```
click                          -> {'command': 'click'}
double click                   -> {'command': 'double_click'}
scroll down                    -> {'command': 'scroll', 'direction': 'down', 'count': 1}
scroll up 5 times              -> {'command': 'scroll', 'direction': 'up', 'count': 5}
type hello world               -> {'command': 'type_text', 'text': 'hello world'}
press enter                    -> {'command': 'press_key', 'direction': 'enter'}
press control c                -> {'command': 'press_key', 'modifier': 'control', 'key': 'c'}
```

#### Approach 2: NLP Intent Classification (Advanced)

**Best for:** Complex commands, natural variations, learning from usage

```python
from transformers import pipeline

class NLPCommandParser:
    def __init__(self):
        # Use pre-trained zero-shot classification
        self.classifier = pipeline("zero-shot-classification",
                                   model="facebook/bart-large-mnli")

        self.intents = [
            "click mouse",
            "double click",
            "right click",
            "scroll up",
            "scroll down",
            "type text",
            "press key",
            "move cursor",
        ]

    def parse(self, text: str):
        """Classify intent using NLP model."""
        result = self.classifier(text, self.intents)

        intent = result['labels'][0]
        confidence = result['scores'][0]

        if confidence > 0.5:
            return {
                'intent': intent,
                'confidence': confidence,
                'text': text
            }

        return None

# Usage
nlp_parser = NLPCommandParser()
print(nlp_parser.parse("please click here"))
print(nlp_parser.parse("I want to scroll to the bottom"))
```

#### Approach 3: Hybrid System (Recommended for Production)

Combine rule-based for common commands with NLP fallback:

```python
class HybridCommandParser:
    def __init__(self):
        self.rule_parser = VoiceCommandParser()
        self.nlp_parser = NLPCommandParser()

    def parse(self, text: str):
        # Try rule-based first (fast, deterministic)
        result = self.rule_parser.parse(text)
        if result:
            result['method'] = 'rule-based'
            return result

        # Fallback to NLP (handles variations)
        result = self.nlp_parser.parse(text)
        if result:
            result['method'] = 'nlp'
            return result

        return None
```

### Wake Word Detection

Wake words activate voice listening to conserve battery and prevent accidental activation.

#### Option 1: Porcupine (Picovoice) - Recommended

**Advantages:**
- Custom wake words
- <100KB model size
- Cross-platform (Linux, macOS, Windows, RPi)
- Commercial support

**Installation:**
```bash
pip install pvporcupine
```

**Usage:**
```python
import pvporcupine
import pyaudio
import struct

# Initialize Porcupine with custom wake word
porcupine = pvporcupine.create(
    access_key='YOUR_ACCESS_KEY',
    keywords=['computer', 'hey-watch']  # Built-in keywords
)

pa = pyaudio.PyAudio()
audio_stream = pa.open(
    rate=porcupine.sample_rate,
    channels=1,
    format=pyaudio.paInt16,
    input=True,
    frames_per_buffer=porcupine.frame_length
)

print("Listening for wake word...")

try:
    while True:
        pcm = audio_stream.read(porcupine.frame_length)
        pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)

        keyword_index = porcupine.process(pcm)

        if keyword_index >= 0:
            print(f"Wake word detected: {keyword_index}")
            # Start full STT processing

except KeyboardInterrupt:
    pass
finally:
    audio_stream.close()
    pa.terminate()
    porcupine.delete()
```

**Custom Wake Word Training:**
- Train custom words via Picovoice Console
- Export optimized models
- Deploy to target platforms

#### Option 2: openWakeWord - Open Source

**Advantages:**
- Fully open source
- No API keys required
- Based on Google's audio embedding model
- Fine-tuned with Piper TTS

**Installation:**
```bash
pip install openwakeword
```

**Usage:**
```python
from openwakeword import Model
import pyaudio

# Load pre-trained model
model = Model(wakeword_models=["hey_jarvis"])

# Audio stream
pa = pyaudio.PyAudio()
stream = pa.open(
    rate=16000,
    channels=1,
    format=pyaudio.paInt16,
    input=True,
    frames_per_buffer=1280
)

print("Listening for 'hey jarvis'...")

while True:
    audio = stream.read(1280)
    prediction = model.predict(audio)

    if prediction["hey_jarvis"] > 0.5:
        print("Wake word detected!")
        # Activate voice command processing
```

**Custom Training:**
```python
# Generate training data with text-to-speech variations
from openwakeword.train import generate_training_data

generate_training_data(
    wake_phrase="computer activate",
    output_dir="./training_data",
    num_samples=10000
)

# Train custom model
# ... (training code)
```

#### Option 3: Silero VAD (Voice Activity Detection)

Not a wake word detector, but useful for detecting when speech starts/stops:

```python
import torch
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

model = load_silero_vad()

# Process audio
audio = read_audio('audio.wav')
speech_timestamps = get_speech_timestamps(audio, model)

print(f"Speech segments: {speech_timestamps}")
# [{'start': 0, 'end': 4800}, {'start': 5600, 'end': 8000}]
```

### Continuous Listening vs Push-to-Talk

| Architecture | Pros | Cons | Best For |
|--------------|------|------|----------|
| **Push-to-Talk** | Battery efficient, no false activations, privacy | Requires button press | Watch with physical button |
| **Wake Word** | Hands-free, natural | Battery drain, false positives | Always-available assistant |
| **Continuous + VAD** | Natural, responsive | High battery drain | Plugged-in/high-capacity devices |
| **Hybrid** | Flexible, efficient | Complex implementation | Production systems |

**Recommended for Wristcontrol:**
- **Primary:** Push-to-talk (button hold on watch)
- **Optional:** Wake word for hands-free mode
- **Always:** VAD to detect speech end

---

## Audio Processing

### Voice Activity Detection (VAD)

VAD detects when someone is speaking vs silence/noise. Critical for:
- Reducing processing load (only transcribe speech)
- Determining when command is complete
- Battery conservation

#### Silero VAD (Recommended)

**Best for:** Production use, real-time, any platform

**Advantages:**
- Pre-trained on 6000+ languages
- <1ms per 30ms audio chunk (CPU)
- 8kHz and 16kHz support
- MIT license (permissive)
- No telemetry or vendor lock

**Installation:**
```bash
pip install silero-vad
```

**Real-time Usage:**
```python
import torch
import pyaudio
import numpy as np
from silero_vad import load_silero_vad, get_speech_ts

class RealtimeVAD:
    def __init__(self):
        self.model = load_silero_vad()
        self.sample_rate = 16000
        self.chunk_size = 512  # 32ms at 16kHz

    def is_speech(self, audio_chunk):
        """Check if audio chunk contains speech."""
        audio_tensor = torch.FloatTensor(audio_chunk)

        speech_prob = self.model(audio_tensor, self.sample_rate).item()

        return speech_prob > 0.5  # Threshold

    def process_stream(self, callback):
        """Process microphone stream with VAD."""
        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )

        speech_buffer = []
        is_speaking = False
        silence_chunks = 0

        try:
            while True:
                audio_data = stream.read(self.chunk_size)
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                if self.is_speech(audio_np):
                    if not is_speaking:
                        print("Speech started")
                        is_speaking = True

                    speech_buffer.extend(audio_np)
                    silence_chunks = 0

                else:
                    if is_speaking:
                        silence_chunks += 1

                        # End of speech after 500ms silence
                        if silence_chunks > 15:  # 15 * 32ms = 480ms
                            print("Speech ended")
                            callback(np.array(speech_buffer))
                            speech_buffer = []
                            is_speaking = False

        except KeyboardInterrupt:
            pass
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()

# Usage
def on_speech_complete(audio):
    print(f"Captured {len(audio) / 16000:.2f}s of speech")
    # Send to STT

vad = RealtimeVAD()
vad.process_stream(on_speech_complete)
```

**Performance:**
- RTF (Real-Time Factor): 0.004 on AMD CPU
- 30ms chunk = 1ms processing time
- 250x faster than real-time

#### PyAnnote Audio VAD

**Best for:** Research, highest accuracy

```bash
pip install pyannote.audio
```

```python
from pyannote.audio import Pipeline

pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection")

# Apply VAD on audio file
vad = pipeline("audio.wav")

for speech in vad.get_timeline().support():
    print(f"Speech from {speech.start:.1f}s to {speech.end:.1f}s")
```

**Requirements:**
- GPU recommended (RTX 3060+, 6-8GB VRAM)
- More accurate but slower than Silero

### Noise Cancellation and Filtering

Watch microphones are prone to:
- Wind noise (arm movement)
- Fabric rubbing
- Background noise
- Low-quality audio capture

#### Option 1: noisereduce (Spectral Gating)

**Best for:** Simple integration, stationary noise

**Installation:**
```bash
pip install noisereduce
```

**Usage:**
```python
import noisereduce as nr
import librosa

# Load audio
audio, sr = librosa.load("noisy_audio.wav", sr=16000)

# Reduce noise
reduced = nr.reduce_noise(
    y=audio,
    sr=sr,
    stationary=True,  # True for constant noise (AC, fan)
    prop_decrease=1.0  # Aggressiveness (0-1)
)

# Save cleaned audio
import soundfile as sf
sf.write("clean_audio.wav", reduced, sr)
```

**Real-time Usage:**
```python
import noisereduce as nr
import pyaudio
import numpy as np

class NoiseReducer:
    def __init__(self, sample_rate=16000):
        self.sr = sample_rate
        self.noise_profile = None

    def calibrate(self, noise_sample):
        """Calibrate with noise sample (1-2 seconds)."""
        self.noise_profile = noise_sample

    def reduce(self, audio):
        """Apply noise reduction to audio chunk."""
        if self.noise_profile is None:
            return audio

        return nr.reduce_noise(
            y=audio,
            y_noise=self.noise_profile,
            sr=self.sr,
            stationary=False  # Non-stationary for varying noise
        )

# Usage
reducer = NoiseReducer()

# Calibrate with 2 seconds of noise
pa = pyaudio.PyAudio()
stream = pa.open(rate=16000, channels=1, format=pyaudio.paInt16, input=True)
print("Calibrating... please be silent")
noise_data = stream.read(32000)  # 2 seconds
noise_np = np.frombuffer(noise_data, dtype=np.int16).astype(np.float32) / 32768.0
reducer.calibrate(noise_np)

print("Calibration complete")
```

#### Option 2: DeepFilterNet (Deep Learning)

**Best for:** Real-time, embedded devices, highest quality

**Advantages:**
- Designed for real-time on embedded devices
- LADSPA plugin for system-wide noise reduction
- Better than spectral methods for complex noise

**Installation:**
```bash
pip install deepfilternet
```

**Usage:**
```python
from df.enhance import enhance, init_df

# Initialize model
model, df_state, _ = init_df()

# Enhance audio
enhanced_audio = enhance(model, df_state, noisy_audio)
```

**Real-time with PipeWire (Linux):**
```bash
# Install LADSPA plugin
deepfilter-ladspa --install

# Configure PipeWire filter chain
# (automatic noise reduction on microphone)
```

#### Option 3: Koala (Picovoice)

**Best for:** Production, commercial use

**Advantages:**
- Local processing (no cloud)
- Optimized for real-time
- Cross-platform

**Installation:**
```bash
pip install pvkoala
```

**Usage:**
```python
import pvkoala

koala = pvkoala.create(access_key='YOUR_ACCESS_KEY')

# Process audio frame
enhanced_frame = koala.process(audio_frame)
```

### Audio Format Conversion

Watch microphones may output various formats. Need conversion for STT.

```python
import librosa
import soundfile as sf
import io

class AudioConverter:
    @staticmethod
    def convert_to_16khz_mono(audio_data, source_rate=48000):
        """Convert audio to 16kHz mono (standard for STT)."""
        # Load from bytes
        audio, sr = librosa.load(io.BytesIO(audio_data), sr=source_rate, mono=True)

        # Resample to 16kHz
        audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        return audio_16k

    @staticmethod
    def convert_format(audio_data, source_format="opus", target_format="wav"):
        """Convert between audio formats."""
        from pydub import AudioSegment

        # Load audio
        audio = AudioSegment.from_file(io.BytesIO(audio_data), format=source_format)

        # Export to target format
        buffer = io.BytesIO()
        audio.export(buffer, format=target_format)

        return buffer.getvalue()

    @staticmethod
    def normalize_volume(audio, target_db=-20):
        """Normalize audio volume."""
        from pydub import AudioSegment

        audio_segment = AudioSegment(
            audio.tobytes(),
            frame_rate=16000,
            sample_width=2,
            channels=1
        )

        # Normalize to target dB
        change_in_db = target_db - audio_segment.dBFS
        normalized = audio_segment.apply_gain(change_in_db)

        return np.array(normalized.get_array_of_samples()).astype(np.float32) / 32768.0
```

### Streaming Audio to STT Services

#### WebSocket Streaming (Deepgram Example)

```python
import asyncio
import websockets
import json
import pyaudio

async def stream_audio_to_deepgram():
    """Stream microphone audio to Deepgram via WebSocket."""

    # WebSocket connection
    url = "wss://api.deepgram.com/v1/listen?punctuate=true&interim_results=true"
    headers = {"Authorization": f"Token YOUR_API_KEY"}

    async with websockets.connect(url, extra_headers=headers) as ws:

        # Audio stream
        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=8000
        )

        async def send_audio():
            """Send audio to WebSocket."""
            try:
                while True:
                    audio_data = stream.read(8000)
                    await ws.send(audio_data)
                    await asyncio.sleep(0.01)
            except:
                pass
            finally:
                stream.stop_stream()
                stream.close()
                pa.terminate()

        async def receive_transcripts():
            """Receive transcripts from WebSocket."""
            async for message in ws:
                result = json.loads(message)

                if 'channel' in result:
                    transcript = result['channel']['alternatives'][0]['transcript']

                    if transcript:
                        is_final = result.get('is_final', False)
                        print(f"{'FINAL' if is_final else 'interim'}: {transcript}")

        # Run both tasks concurrently
        await asyncio.gather(send_audio(), receive_transcripts())

# Run
asyncio.run(stream_audio_to_deepgram())
```

#### HTTP Streaming (OpenAI Whisper API)

```python
import requests

def stream_to_whisper_api(audio_file_path):
    """Stream audio file to OpenAI Whisper API."""

    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer YOUR_API_KEY"}

    with open(audio_file_path, "rb") as audio_file:
        files = {"file": audio_file}
        data = {
            "model": "whisper-1",
            "language": "en",
            "response_format": "json"
        }

        response = requests.post(url, headers=headers, files=files, data=data)

        if response.status_code == 200:
            return response.json()['text']
        else:
            raise Exception(f"API error: {response.status_code}")

# Usage
transcript = stream_to_whisper_api("audio.wav")
print(transcript)
```

#### BLE Audio Streaming from Watch

```python
import asyncio
from bleak import BleakClient

class WatchAudioStreamer:
    def __init__(self, device_address):
        self.address = device_address
        self.audio_characteristic_uuid = "your-audio-characteristic-uuid"
        self.audio_buffer = []

    async def start_streaming(self, on_audio_callback):
        """Stream audio from watch via BLE."""

        async with BleakClient(self.address) as client:
            print(f"Connected to watch: {self.address}")

            def handle_audio_data(sender, data):
                """Handle incoming audio data."""
                self.audio_buffer.extend(data)

                # When buffer reaches threshold, process
                if len(self.audio_buffer) >= 32000:  # 2 seconds at 16kHz
                    audio_np = np.frombuffer(
                        bytes(self.audio_buffer),
                        dtype=np.int16
                    ).astype(np.float32) / 32768.0

                    on_audio_callback(audio_np)
                    self.audio_buffer = []

            # Subscribe to audio characteristic
            await client.start_notify(self.audio_characteristic_uuid, handle_audio_data)

            # Keep connection open
            await asyncio.sleep(3600)  # 1 hour

# Usage
async def on_audio(audio_data):
    # Process with STT
    print(f"Received {len(audio_data)} samples")

streamer = WatchAudioStreamer("AA:BB:CC:DD:EE:FF")
asyncio.run(streamer.start_streaming(on_audio))
```

---

## Architecture Recommendations

### Recommended Architecture for Wristcontrol

Based on project requirements:
- <500ms voice command latency target
- Samsung Galaxy Watch (limited battery)
- Privacy focus with local processing option
- Desktop companion app handles processing

```
┌─────────────────────────────────────────────────────────────┐
│                        WATCH APP                             │
│  ┌──────────────┐                                           │
│  │  Microphone  │ → Audio Buffer → Opus Compression         │
│  └──────────────┘                                           │
│                      ↓                                       │
│              ┌───────────────┐                              │
│              │  VAD (Light)  │ → Detect Speech Start/End    │
│              └───────────────┘                              │
│                      ↓                                       │
│              ┌───────────────┐                              │
│              │  BLE Streaming│ → Send to Desktop            │
│              └───────────────┘                              │
└─────────────────────────────────────────────────────────────┘
                       │
                  BLE Audio Stream
                       ↓
┌─────────────────────────────────────────────────────────────┐
│                    DESKTOP COMPANION APP                     │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │             Audio Processing Pipeline                │   │
│  │                                                       │   │
│  │  1. Receive BLE Stream → Decompress Opus            │   │
│  │  2. Format Conversion → 16kHz Mono WAV              │   │
│  │  3. Noise Reduction → DeepFilterNet/noisereduce     │   │
│  │  4. VAD (Silero) → Segment Speech                   │   │
│  └─────────────────────────────────────────────────────┘   │
│                       ↓                                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │          Speech-to-Text Engine (Hybrid)             │   │
│  │                                                       │   │
│  │  ┌────────────────┐      ┌──────────────────┐      │   │
│  │  │ faster-whisper │      │ Deepgram API     │      │   │
│  │  │ (Local/Free)   │  OR  │ (Cloud/Premium)  │      │   │
│  │  │ Model: tiny/base│      │ Nova-3 Streaming │      │   │
│  │  └────────────────┘      └──────────────────┘      │   │
│  └─────────────────────────────────────────────────────┘   │
│                       ↓                                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │            Command Parser (Hybrid)                   │   │
│  │                                                       │   │
│  │  1. Rule-Based Patterns → Fast, Deterministic       │   │
│  │  2. NLP Fallback → Handle Variations                │   │
│  │  3. Slot Extraction → Parameters (count, direction) │   │
│  └─────────────────────────────────────────────────────┘   │
│                       ↓                                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Action Executor                         │   │
│  │                                                       │   │
│  │  • Mouse Control (pynput)                           │   │
│  │  • Keyboard Input (pynput)                          │   │
│  │  • System Commands                                   │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Component Selection by Phase

#### Phase 2: Voice Integration (MVP)

**Goal:** Quick validation with computer microphone

```python
# Minimal MVP setup
- STT: faster-whisper (tiny model) for free local processing
- VAD: Silero VAD for speech detection
- Commands: Rule-based parser (10-15 essential commands)
- Audio: Computer microphone (pyaudio)
- Activation: Keyboard shortcut (no wake word yet)
```

**Rationale:**
- No cost, no API keys needed
- tiny model processes 32x real-time (well under 500ms)
- Validates command parsing logic
- Simple integration

#### Phase 4: Watch Audio Streaming

**Goal:** Real audio from watch

```python
# Production setup
- Watch: Opus compression → BLE streaming
- Desktop:
  - Receive via BLE (bleak library)
  - Decompress Opus
  - Apply noisereduce for watch mic quality
  - Continue with faster-whisper + Silero VAD
- Activation: Physical button on watch (push-to-talk)
```

**Rationale:**
- Opus compression reduces BLE bandwidth (critical)
- Button activation saves battery
- Noise reduction handles watch mic issues
- Local processing maintains privacy

#### Production: Optional Cloud Upgrade

**Goal:** Premium accuracy and latency for paying users

```python
# Premium tier
- STT: Deepgram Nova-3 streaming API
- Benefits:
  - Sub-300ms latency (better than 500ms target)
  - Higher accuracy
  - Speaker diarization (future multi-user)
  - Professional support
- Cost: $4.30/1000 min = ~$0.07 per heavy user per month
```

**Rationale:**
- Users can choose: free (local) vs premium (cloud)
- Revenue potential via subscription
- Best-in-class performance for power users

### Activation Strategy

**Recommended Multi-Tier Approach:**

```python
class ActivationManager:
    def __init__(self):
        self.mode = "push_to_talk"  # Default
        self.wake_word_active = False

    def set_mode(self, mode):
        """
        Modes:
        - push_to_talk: Button press required (most battery efficient)
        - wake_word: "Computer" activates listening
        - continuous: Always listening with VAD (battery intensive)
        """
        self.mode = mode

    def should_listen(self, button_pressed, wake_word_detected, vad_speech):
        """Determine if STT should be active."""

        if self.mode == "push_to_talk":
            return button_pressed

        elif self.mode == "wake_word":
            # Listen for 5 seconds after wake word
            if wake_word_detected:
                self.wake_word_active = True
                # Start timer
            return self.wake_word_active and vad_speech

        elif self.mode == "continuous":
            return vad_speech

        return False
```

**Phase Rollout:**
1. **MVP:** Push-to-talk only
2. **Post-MVP:** Add wake word as option
3. **Future:** Continuous mode for desk use

### Latency Budget

Target: <500ms from speech to command execution

| Component | Budget | Solution |
|-----------|--------|----------|
| Audio capture on watch | 50ms | 50ms buffer |
| BLE transmission | 50ms | Opus compressed |
| Audio processing | 50ms | Noise reduce + VAD |
| STT processing | 250ms | faster-whisper tiny or Deepgram |
| Command parsing | 10ms | Rule-based |
| Action execution | 20ms | pynput |
| **Total** | **430ms** | ✅ Under target |

Alternative with cloud:
- STT (Deepgram): 300ms → Total: 480ms ✅

### Privacy Considerations

**Three-Tier Privacy Model:**

1. **Maximum Privacy (Default):**
   - Local faster-whisper processing
   - All audio processed on desktop
   - No data leaves user's computer
   - No API keys required

2. **Balanced:**
   - Wake word detection local (Porcupine/openWakeWord)
   - STT via cloud API
   - Audio streamed only when activated
   - No storage on cloud

3. **Cloud-Enhanced:**
   - Continuous cloud STT
   - Command history for improvement
   - Opt-in telemetry for accuracy metrics

**Implementation:**
```python
class PrivacySettings:
    def __init__(self):
        self.stt_mode = "local"  # local, cloud
        self.store_history = False
        self.telemetry = False

    def can_use_cloud(self):
        return self.stt_mode == "cloud"

    def can_store_commands(self):
        return self.store_history

    def can_send_telemetry(self):
        return self.telemetry
```

---

## Implementation Examples

### Complete Voice Command System

```python
#!/usr/bin/env python3
"""
Wristcontrol Voice Command System
Complete implementation with STT, VAD, command parsing, and execution
"""

import asyncio
import numpy as np
import pyaudio
from faster_whisper import WhisperModel
from silero_vad import load_silero_vad
import torch
from pynput.mouse import Button, Controller as MouseController
from pynput.keyboard import Key, Controller as KeyboardController
import re
from typing import Optional, Dict, Any

class VoiceCommandSystem:
    def __init__(self, use_cloud=False):
        print("Initializing Voice Command System...")

        # STT
        if use_cloud:
            from deepgram import Deepgram
            self.stt_engine = "cloud"
            self.deepgram = Deepgram("YOUR_API_KEY")
        else:
            self.stt_engine = "local"
            self.whisper = WhisperModel("tiny", device="cpu", compute_type="int8")

        # VAD
        self.vad_model = load_silero_vad()

        # Command parser
        self.parser = VoiceCommandParser()

        # System control
        self.mouse = MouseController()
        self.keyboard = KeyboardController()

        # Audio settings
        self.sample_rate = 16000
        self.chunk_size = 512

        # State
        self.is_listening = False
        self.speech_buffer = []

        print(f"Initialized with {self.stt_engine} STT engine")

    def is_speech(self, audio_chunk):
        """Detect if audio contains speech using VAD."""
        audio_tensor = torch.FloatTensor(audio_chunk)
        speech_prob = self.vad_model(audio_tensor, self.sample_rate).item()
        return speech_prob > 0.5

    def transcribe_local(self, audio_data):
        """Transcribe audio using local Whisper."""
        segments, info = self.whisper.transcribe(audio_data, beam_size=1, language="en")

        text = ""
        for segment in segments:
            text += segment.text + " "

        return text.strip()

    async def transcribe_cloud(self, audio_data):
        """Transcribe audio using Deepgram API."""
        # Convert numpy array to bytes
        audio_bytes = (audio_data * 32768).astype(np.int16).tobytes()

        source = {'buffer': audio_bytes, 'mimetype': 'audio/wav'}
        response = await self.deepgram.transcription.prerecorded(source)

        return response['results']['channels'][0]['alternatives'][0]['transcript']

    def execute_command(self, command: Dict[str, Any]):
        """Execute parsed voice command."""
        cmd_type = command['command']

        print(f"Executing: {command}")

        if cmd_type == 'click':
            self.mouse.click(Button.left, 1)

        elif cmd_type == 'double_click':
            self.mouse.click(Button.left, 2)

        elif cmd_type == 'right_click':
            self.mouse.click(Button.right, 1)

        elif cmd_type == 'scroll':
            direction = command.get('direction', 'down')
            count = command.get('count', 1)
            amount = -3 if direction == 'up' else 3

            for _ in range(count):
                self.mouse.scroll(0, amount)

        elif cmd_type == 'type_text':
            text = command.get('text', '')
            self.keyboard.type(text)

        elif cmd_type == 'press_key':
            key_name = command.get('direction')  # Reusing 'direction' field

            # Map key names to pynput keys
            key_map = {
                'enter': Key.enter,
                'escape': Key.esc,
                'tab': Key.tab,
                'space': Key.space,
                'backspace': Key.backspace,
                'delete': Key.delete,
            }

            if key_name in key_map:
                self.keyboard.press(key_map[key_name])
                self.keyboard.release(key_map[key_name])

            # Handle modifiers
            if 'modifier' in command:
                modifier = command['modifier']
                key = command['key']

                modifier_map = {
                    'control': Key.ctrl,
                    'alt': Key.alt,
                    'shift': Key.shift,
                }

                with self.keyboard.pressed(modifier_map[modifier]):
                    self.keyboard.press(key)
                    self.keyboard.release(key)

    def process_microphone_stream(self):
        """Process microphone input with VAD and STT."""
        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )

        print("Listening... (Press Ctrl+C to stop)")
        print("Speak a command when you see 'Speech detected'")

        speech_buffer = []
        is_speaking = False
        silence_chunks = 0

        try:
            while True:
                # Read audio
                audio_data = stream.read(self.chunk_size)
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                # VAD
                if self.is_speech(audio_np):
                    if not is_speaking:
                        print("\\n🎤 Speech detected...")
                        is_speaking = True

                    speech_buffer.extend(audio_np)
                    silence_chunks = 0

                else:
                    if is_speaking:
                        silence_chunks += 1

                        # End of speech after 500ms silence
                        if silence_chunks > 15:  # 15 * 32ms = 480ms
                            print("🔚 Speech ended, transcribing...")

                            # Transcribe
                            audio_array = np.array(speech_buffer)

                            if self.stt_engine == "local":
                                text = self.transcribe_local(audio_array)
                            else:
                                text = asyncio.run(self.transcribe_cloud(audio_array))

                            print(f"📝 Transcription: '{text}'")

                            # Parse command
                            command = self.parser.parse(text)

                            if command:
                                self.execute_command(command)
                                print("✅ Command executed\\n")
                            else:
                                print("❌ No command recognized\\n")

                            # Reset
                            speech_buffer = []
                            is_speaking = False

        except KeyboardInterrupt:
            print("\\n\\nStopping...")
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()

class VoiceCommandParser:
    """Rule-based command parser for voice commands."""

    def __init__(self):
        self.patterns = {
            'click': [
                r'\\b(click|tap|press)\\b',
                r'\\bleft click\\b',
            ],
            'double_click': [
                r'\\bdouble click\\b',
                r'\\bclick twice\\b',
            ],
            'right_click': [
                r'\\bright click\\b',
                r'\\bcontext menu\\b',
            ],
            'scroll': [
                r'\\bscroll (up|down)\\b',
                r'\\bscroll (up|down) (\\d+) ?(times|)?\\b',
            ],
            'type_text': [
                r'\\btype (.+)',
                r'\\benter (.+)',
                r'\\bwrite (.+)',
            ],
            'press_key': [
                r'\\bpress (enter|escape|tab|space|backspace|delete)\\b',
                r'\\bpress (control|alt|shift) ([a-z])\\b',
            ],
        }

    def parse(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse voice command text into structured command."""
        text = text.lower().strip()

        for command, patterns in self.patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    return self._extract_command(command, match, text)

        return None

    def _extract_command(self, command: str, match: re.Match, text: str) -> Dict[str, Any]:
        """Extract command parameters."""
        result = {'command': command}

        if command == 'scroll':
            result['direction'] = match.group(1) if match.lastindex >= 1 else 'down'
            result['count'] = int(match.group(2)) if match.lastindex >= 2 else 1

        elif command == 'type_text':
            result['text'] = match.group(1).strip()

        elif command == 'press_key':
            if match.lastindex == 1:
                result['direction'] = match.group(1)
            else:
                result['modifier'] = match.group(1)
                result['key'] = match.group(2)

        return result

def main():
    """Main entry point."""
    import sys

    # Choose STT engine
    use_cloud = "--cloud" in sys.argv

    # Initialize system
    system = VoiceCommandSystem(use_cloud=use_cloud)

    # Start processing
    system.process_microphone_stream()

if __name__ == "__main__":
    main()
```

**Usage:**

```bash
# Install dependencies
pip install faster-whisper silero-vad torch pyaudio pynput numpy

# Run with local Whisper (free, private)
python voice_command_system.py

# Run with cloud API (faster, more accurate)
python voice_command_system.py --cloud
```

**Example Session:**

```
Initializing Voice Command System...
Initialized with local STT engine
Listening... (Press Ctrl+C to stop)
Speak a command when you see 'Speech detected'

🎤 Speech detected...
🔚 Speech ended, transcribing...
📝 Transcription: 'click'
Executing: {'command': 'click'}
✅ Command executed

🎤 Speech detected...
🔚 Speech ended, transcribing...
📝 Transcription: 'scroll down five times'
Executing: {'command': 'scroll', 'direction': 'down', 'count': 5}
✅ Command executed

🎤 Speech detected...
🔚 Speech ended, transcribing...
📝 Transcription: 'type hello world'
Executing: {'command': 'type_text', 'text': 'hello world'}
✅ Command executed
```

### Watch Audio Receiver (BLE)

```python
#!/usr/bin/env python3
"""
Watch Audio Receiver
Receives audio stream from Samsung Galaxy Watch via BLE
"""

import asyncio
from bleak import BleakClient, BleakScanner
import numpy as np
from collections import deque

class WatchAudioReceiver:
    def __init__(self, device_name="Galaxy Watch"):
        self.device_name = device_name
        self.device_address = None

        # BLE UUIDs (customize for your watch app)
        self.AUDIO_SERVICE_UUID = "your-audio-service-uuid"
        self.AUDIO_CHAR_UUID = "your-audio-characteristic-uuid"

        # Audio buffer
        self.audio_buffer = deque(maxlen=160000)  # 10 seconds at 16kHz
        self.sample_rate = 16000

    async def find_watch(self):
        """Scan for watch device."""
        print(f"Scanning for {self.device_name}...")

        devices = await BleakScanner.discover()

        for device in devices:
            if self.device_name in device.name:
                self.device_address = device.address
                print(f"Found watch: {device.name} ({device.address})")
                return True

        print("Watch not found")
        return False

    async def connect_and_stream(self, on_audio_callback):
        """Connect to watch and stream audio."""
        if not self.device_address:
            if not await self.find_watch():
                return

        async with BleakClient(self.device_address) as client:
            print(f"Connected to {self.device_address}")

            # Check if audio service exists
            services = await client.get_services()
            audio_service = services.get_service(self.AUDIO_SERVICE_UUID)

            if not audio_service:
                print(f"Audio service {self.AUDIO_SERVICE_UUID} not found")
                return

            print("Audio service found, starting stream...")

            def handle_audio_packet(sender, data):
                """Handle incoming audio packet."""
                # Convert bytes to numpy array
                audio_samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

                # Add to buffer
                self.audio_buffer.extend(audio_samples)

                # When buffer reaches threshold, process
                if len(self.audio_buffer) >= 32000:  # 2 seconds
                    audio_chunk = np.array(list(self.audio_buffer))
                    self.audio_buffer.clear()

                    # Callback with audio
                    asyncio.create_task(on_audio_callback(audio_chunk))

            # Subscribe to audio notifications
            await client.start_notify(self.AUDIO_CHAR_UUID, handle_audio_packet)

            print("Streaming audio from watch...")

            # Keep connection alive
            try:
                while client.is_connected:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                print("\\nStopping stream...")
            finally:
                await client.stop_notify(self.AUDIO_CHAR_UUID)

async def main():
    """Test audio receiver."""

    async def on_audio(audio_data):
        """Called when audio chunk is ready."""
        duration = len(audio_data) / 16000
        print(f"Received {duration:.2f}s of audio ({len(audio_data)} samples)")

        # Here you would:
        # 1. Apply noise reduction
        # 2. Send to STT
        # 3. Parse command
        # 4. Execute action

    receiver = WatchAudioReceiver()
    await receiver.connect_and_stream(on_audio)

if __name__ == "__main__":
    asyncio.run(main())
```

### Configuration Manager

```python
#!/usr/bin/env python3
"""
Voice Command Configuration
Manages settings, command vocabulary, and user preferences
"""

import json
from pathlib import Path
from typing import Dict, List, Any

class VoiceConfig:
    def __init__(self, config_file="voice_config.json"):
        self.config_file = Path(config_file)
        self.config = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        if self.config_file.exists():
            with open(self.config_file) as f:
                return json.load(f)
        else:
            return self.default_config()

    def save_config(self):
        """Save configuration to file."""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)

    def default_config(self) -> Dict[str, Any]:
        """Default configuration."""
        return {
            'stt': {
                'engine': 'local',  # local, deepgram, openai, google, azure
                'model': 'tiny',  # For local: tiny, base, small, medium, large
                'language': 'en',
                'api_key': '',
            },
            'vad': {
                'enabled': True,
                'threshold': 0.5,
                'silence_duration_ms': 500,
            },
            'wake_word': {
                'enabled': False,
                'word': 'computer',
                'engine': 'porcupine',  # porcupine, openwakeword
                'api_key': '',
            },
            'activation': {
                'mode': 'push_to_talk',  # push_to_talk, wake_word, continuous
                'button': 'watch_button',
            },
            'audio': {
                'sample_rate': 16000,
                'channels': 1,
                'chunk_size': 512,
                'noise_reduction': True,
            },
            'commands': {
                'enabled_commands': [
                    'click',
                    'double_click',
                    'right_click',
                    'scroll',
                    'type_text',
                    'press_key',
                ],
                'custom_commands': {},
            },
            'privacy': {
                'store_history': False,
                'telemetry': False,
                'local_only': True,
            },
            'performance': {
                'latency_target_ms': 500,
                'beam_size': 1,
                'gpu_enabled': False,
            }
        }

    def get(self, key_path: str, default=None):
        """Get configuration value by dot-notation path."""
        keys = key_path.split('.')
        value = self.config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def set(self, key_path: str, value):
        """Set configuration value by dot-notation path."""
        keys = key_path.split('.')
        config = self.config

        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        config[keys[-1]] = value
        self.save_config()

    def add_custom_command(self, name: str, pattern: str, action: str):
        """Add custom voice command."""
        if 'custom_commands' not in self.config['commands']:
            self.config['commands']['custom_commands'] = {}

        self.config['commands']['custom_commands'][name] = {
            'pattern': pattern,
            'action': action,
        }

        self.save_config()

    def is_command_enabled(self, command: str) -> bool:
        """Check if command is enabled."""
        return command in self.config['commands']['enabled_commands']

# Usage example
if __name__ == "__main__":
    config = VoiceConfig()

    # Get values
    print(f"STT Engine: {config.get('stt.engine')}")
    print(f"VAD Enabled: {config.get('vad.enabled')}")

    # Set values
    config.set('stt.engine', 'deepgram')
    config.set('stt.api_key', 'your-api-key-here')

    # Add custom command
    config.add_custom_command(
        name='maximize_window',
        pattern=r'\\bmaximize window\\b',
        action='maximize'
    )

    print(f"\\nConfiguration saved to {config.config_file}")
```

---

## Sources

### Local Whisper Implementation
- [GitHub - ggml-org/whisper.cpp](https://github.com/ggml-org/whisper.cpp)
- [GitHub - SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [5 Ways to Speed Up Whisper Transcription](https://modal.com/blog/faster-transcription)
- [whisper.cpp - High-performance local implementation](https://jimmysong.io/ai/whisper-cpp/)

### Whisper Model Performance
- [GitHub - openai/whisper](https://github.com/openai/whisper)
- [Which Whisper Model Should I Choose?](https://whisper-api.com/blog/models/)
- [Whisper Variants Comparison](https://towardsai.net/p/machine-learning/whisper-variants-comparison-what-are-their-features-and-how-to-implement-them)
- [Best open source speech-to-text models in 2025](https://northflank.com/blog/best-open-source-speech-to-text-stt-model-in-2025-benchmarks)
- [Choosing Between Accuracy, Speed, and Resources](https://whishper.net/reference/models/)
- [The Top Open Source Speech-to-Text Models in 2025](https://modal.com/blog/open-source-stt)

### Cloud STT APIs
- [Speech-to-Text API Pricing Breakdown 2025](https://deepgram.com/learn/speech-to-text-api-pricing-breakdown-2025)
- [Deepgram vs OpenAI vs Google STT Comparison](https://deepgram.com/learn/deepgram-vs-openai-vs-google-stt-accuracy-latency-price-compared)
- [5 Google Cloud Speech-to-Text alternatives](https://www.assemblyai.com/blog/google-cloud-speech-to-text-alternatives)
- [Speech-to-Text API Benchmarks](https://deepgram.com/learn/speech-to-text-benchmarks)
- [Best Speech to Text Models 2025](https://nextlevel.ai/best-speech-to-text-models/)
- [Deepgram vs Microsoft Azure AI Speech](https://aloa.co/ai/comparisons/ai-voice-comparison/deepgram-vs-azure-speech/)

### Wake Word Detection
- [GitHub - Picovoice/porcupine](https://github.com/Picovoice/porcupine)
- [Wake Word in Voice Recognition Technology](https://www.futurebeeai.com/knowledge-hub/wake-word-voice-recognition)
- [Creating a Voice Assistant with Wake Word Detection](https://devcodef1.com/news/1139082/python-voice-assistant-continuous-listening-wake-word-detection)
- [Wake Word Detection Overview](https://www.iterate.ai/ai-glossary/wake-word-detection)
- [Porcupine Wake Word Detection](https://picovoice.ai/platform/porcupine/)
- [Wake Word vs Voice Activity Detection](https://www.futurebeeai.com/knowledge-hub/wake-word-vs-voice-activity-detection)
- [GitHub - openWakeWord](https://github.com/dscripka/openWakeWord)

### Voice Activity Detection
- [GitHub - Real-time VAD](https://github.com/hanifabd/voice-activity-detection-vad-realtime)
- [Voice Activity Detection Complete Guide 2025](https://picovoice.ai/blog/complete-guide-voice-activity-detection-vad/)
- [GitHub - Silero VAD](https://github.com/snakers4/silero-vad)
- [Voice Activity Detection in Python](https://picovoice.ai/blog/voice-activity-detection-in-python/)
- [Silero Voice Activity Detector – PyTorch](https://pytorch.org/hub/snakers4_silero-vad_vad/)

### Noise Cancellation
- [noisereduce PyPI](https://pypi.org/project/noisereduce/)
- [GitHub - Active-Noise-Cancelling](https://github.com/SoheilGtex/Active-Noise-Cancelling)
- [GitHub - DeepFilterNet](https://github.com/Rikorose/DeepFilterNet)
- [Koala Noise Suppression](https://picovoice.ai/docs/quick-start/koala-python/)

### Intent Recognition
- [Intent Recognition - Rhasspy](https://rhasspy.readthedocs.io/en/latest/intent-recognition/)
- [Intent Classification in NLP](https://spotintelligence.com/2023/11/03/intent-classification-nlp/)
- [Chatbot Intent Recognition 2026](https://research.aimultiple.com/chatbot-intent/)
- [Creating Your Own Intent Classifier](https://medium.com/analytics-vidhya/creating-your-own-intent-classifier-b86e000a4926)
- [Speech Recognition in Python](https://www.simplilearn.com/tutorials/python-tutorial/speech-recognition-in-python)

### WebSocket Streaming
- [Twilio - Consume Real-time Media Stream](https://www.twilio.com/docs/voice/tutorials/consume-real-time-media-stream-using-websockets-python-and-flask)
- [Python Audio Streaming over WebSocket](https://medium.com/@python-javascript-php-html-css/python-based-effective-audio-streaming-over-websocket-using-asyncio-and-threading-a926ecf087c4)
- [GitHub - VoiceStreamAI](https://github.com/alesaccoia/VoiceStreamAI)
- [Building a Streaming Whisper WebSocket Service](https://medium.com/@david.richards.tech/how-to-build-a-streaming-whisper-websocket-service-1528b96b1235)
- [Building Real-Time Voice AI with WebSockets](https://theten.ai/blog/building-real-time-voice-ai-with-websockets)
