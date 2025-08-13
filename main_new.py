#!/usr/bin/env python3
"""
Simplified Clash Royale Card Detection Tool
Clean, straightforward implementation focusing on core card detection functionality
"""

import cv2
import numpy as np
import logging
from detector import TroopDetector
from video_handler import VideoHandler

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main execution function"""
    # Initialize components
    detector = TroopDetector()
    video_handler = VideoHandler()

    # Process video
    try:
        video_handler.process_video(detector)
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    except Exception as e:
        logger.error(f"Error during processing: {e}")
    finally:
        video_handler.cleanup()


if __name__ == "__main__":
    main()
