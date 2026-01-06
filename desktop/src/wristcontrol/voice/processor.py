"""Voice processing and speech-to-text."""

import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class VoiceProcessor:
    """Processes voice input and executes commands."""

    # Built-in voice commands for mouse/keyboard actions
    DEFAULT_COMMANDS = {
        "click": "left_click",
        "left click": "left_click",
        "right click": "right_click",
        "double click": "double_click",
        "scroll up": "scroll_up",
        "scroll down": "scroll_down",
        "scroll left": "scroll_left",
        "scroll right": "scroll_right",
        "escape": "key_escape",
        "enter": "key_enter",
        "tab": "key_tab",
        "backspace": "key_backspace",
        "delete": "key_delete",
        "copy": "copy",
        "paste": "paste",
        "cut": "cut",
        "undo": "undo",
        "redo": "redo",
        "select all": "select_all",
    }

    def __init__(
        self,
        backend: str = "whisper_local",
        language: str = "en",
    ) -> None:
        """Initialize the voice processor.

        Args:
            backend: STT backend to use ("whisper_local", "whisper_api", "google").
            language: Language code for recognition.
        """
        self.backend = backend
        self.language = language
        self._enabled = False
        self._listening = False
        self._model = None

        self._command_handlers: dict[str, Callable[[], None]] = {}
        self._text_handler: Optional[Callable[[str], None]] = None

    def initialize(self) -> bool:
        """Initialize the STT backend.

        Returns:
            True if initialization successful, False otherwise.
        """
        if self.backend == "whisper_local":
            return self._init_whisper_local()
        elif self.backend == "whisper_api":
            return self._init_whisper_api()
        elif self.backend == "google":
            return self._init_google()
        else:
            logger.error(f"Unknown backend: {self.backend}")
            return False

    def _init_whisper_local(self) -> bool:
        """Initialize local Whisper model."""
        try:
            import whisper

            logger.info("Loading Whisper model (this may take a moment)...")
            self._model = whisper.load_model("base")
            logger.info("Whisper model loaded")
            return True
        except ImportError:
            logger.error("Whisper not installed. Install with: pip install openai-whisper")
            return False
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            return False

    def _init_whisper_api(self) -> bool:
        """Initialize OpenAI Whisper API client."""
        try:
            import openai

            # Client will use OPENAI_API_KEY environment variable
            logger.info("Whisper API client initialized")
            return True
        except ImportError:
            logger.error("OpenAI not installed. Install with: pip install openai")
            return False

    def _init_google(self) -> bool:
        """Initialize Google Cloud Speech-to-Text client."""
        try:
            from google.cloud import speech

            logger.info("Google Speech client initialized")
            return True
        except ImportError:
            logger.error(
                "Google Cloud Speech not installed. "
                "Install with: pip install google-cloud-speech"
            )
            return False

    def register_command_handler(
        self, command: str, handler: Callable[[], None]
    ) -> None:
        """Register a handler for a voice command.

        Args:
            command: The command action name (e.g., "left_click").
            handler: The function to call when the command is recognized.
        """
        self._command_handlers[command] = handler

    def set_text_handler(self, handler: Callable[[str], None]) -> None:
        """Set handler for text that isn't a recognized command.

        Args:
            handler: Function to call with the recognized text.
        """
        self._text_handler = handler

    def process_audio(self, audio_data: bytes) -> Optional[str]:
        """Process audio data and return recognized text.

        Args:
            audio_data: Raw audio bytes.

        Returns:
            Recognized text, or None if recognition failed.
        """
        if not self._enabled:
            return None

        # TODO: Implement actual STT processing
        # This will depend on the backend being used
        return None

    def handle_text(self, text: str) -> None:
        """Handle recognized text, executing commands or typing.

        Args:
            text: The recognized text to process.
        """
        text_lower = text.lower().strip()

        # Check for built-in commands
        if text_lower in self.DEFAULT_COMMANDS:
            action = self.DEFAULT_COMMANDS[text_lower]
            handler = self._command_handlers.get(action)
            if handler:
                logger.info(f"Executing command: {action}")
                handler()
            else:
                logger.warning(f"No handler for command: {action}")
        elif text_lower.startswith("type "):
            # Handle "type <text>" command
            text_to_type = text[5:]  # Remove "type " prefix
            if self._text_handler:
                self._text_handler(text_to_type)
        elif self._text_handler:
            # Pass through as text to type
            self._text_handler(text)

    def start_listening(self) -> None:
        """Start listening for voice input."""
        self._listening = True
        logger.info("Voice listening started")

    def stop_listening(self) -> None:
        """Stop listening for voice input."""
        self._listening = False
        logger.info("Voice listening stopped")

    def enable(self) -> None:
        """Enable voice processing."""
        self._enabled = True
        logger.info("Voice processing enabled")

    def disable(self) -> None:
        """Disable voice processing."""
        self._enabled = False
        self._listening = False
        logger.info("Voice processing disabled")
