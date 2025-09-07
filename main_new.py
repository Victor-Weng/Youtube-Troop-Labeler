#!/usr/bin/env python3
"""
Simplified Clash Royale Card Detection Tool
Clean, straightforward implementation focusing on core card detection functionality
"""

import cv2
import numpy as np
import logging
import json
import os
import config_new as config
from detector import TroopDetector
from video_handler import VideoHandler

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main execution function"""
    # Initialize components
    # Load checkpoint if present to resume video index
    checkpoint_path = os.path.join(config.DATASET_OUTPUT_DIR, 'checkpoint.json')
    start_video_index = 0
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                cp = json.load(f)
            start_video_index = int(cp.get('video_index', 0))
        except Exception:
            start_video_index = 0

    detector = TroopDetector()
    video_handler = VideoHandler()

    # Provide initial video context to dataset saver (restart current video fresh)
    if hasattr(detector, 'dataset_saver') and detector.dataset_saver:
        urls = config.YOUTUBE_URLS
        if start_video_index < len(urls):
            detector.dataset_saver.set_video_context(start_video_index, urls[start_video_index])
        else:
            detector.dataset_saver.set_video_context(len(urls)-1 if urls else 0, urls[-1] if urls else None)

    # Process video
    try:
        # Process only the selected starting URL (current implementation processes first URL only).
        # If multiple URLs desired sequentially, extend VideoHandler to iterate; for now, we resume at index.
        if config.YOUTUBE_URLS:
            # Temporarily replace first URL with the resume target so VideoHandler picks it.
            # (VideoHandler currently uses YOUTUBE_URLS[0])
            if start_video_index < len(config.YOUTUBE_URLS):
                # Rotate list so desired index appears first
                target = config.YOUTUBE_URLS[start_video_index]
                remaining = [u for i,u in enumerate(config.YOUTUBE_URLS) if i != start_video_index]
                config.YOUTUBE_URLS[:] = [target] + remaining
            video_handler.process_video(detector)
            # After completion, advance video index and checkpoint
            if hasattr(detector, 'dataset_saver') and detector.dataset_saver:
                next_index = min(start_video_index + 1, len(config.YOUTUBE_URLS))
                detector.dataset_saver.set_video_context(next_index, config.YOUTUBE_URLS[next_index] if next_index < len(config.YOUTUBE_URLS) else None)
        else:
            video_handler.process_video(detector)
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    except Exception as e:
        logger.error(f"Error during processing: {e}")
    finally:
        video_handler.cleanup()


if __name__ == "__main__":
    main()
