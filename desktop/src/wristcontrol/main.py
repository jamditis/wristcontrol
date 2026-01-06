"""Main entry point for WristControl desktop companion app."""

import asyncio
import logging
import signal
import sys

from wristcontrol.core.app import WristControlApp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Main entry point for the WristControl application."""
    logger.info("Starting WristControl...")

    app = WristControlApp()

    # Handle graceful shutdown
    def signal_handler(sig: int, frame: object) -> None:
        logger.info("Shutdown signal received")
        app.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        asyncio.run(app.run())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        app.stop()
        logger.info("WristControl stopped")


if __name__ == "__main__":
    main()
