import os
import json
import logging

import config_new as config
from detector import TroopDetector
from video_handler import VideoHandler

logger = logging.getLogger(__name__)


def main():
    """Main execution function with sequential multi-video processing and resume support."""
    checkpoint_path = os.path.join(
        config.DATASET_OUTPUT_DIR, 'checkpoint.json')
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
    urls = config.YOUTUBE_URLS
    total = len(urls)
    if total == 0:
        logger.error("No URLs to process.")
        return

    # Provide initial context
    if hasattr(detector, 'dataset_saver') and detector.dataset_saver:
        detector.dataset_saver.set_video_context(
            start_video_index, urls[start_video_index])

    try:
        for idx in range(start_video_index, total):
            current_url = urls[idx]
            logger.info(f"=== Starting video {idx+1}/{total} ===")
            if hasattr(detector, 'dataset_saver') and detector.dataset_saver:
                detector.dataset_saver.set_video_context(idx, current_url)
            video_handler.process_video(detector, youtube_url=current_url)
            # Save checkpoint after each completed video
            try:
                os.makedirs(config.DATASET_OUTPUT_DIR, exist_ok=True)
                with open(checkpoint_path, 'w', encoding='utf-8') as f:
                    json.dump({'video_index': idx+1}, f)
            except Exception:
                pass
            # Progress log
            interval = getattr(config, 'PROGRESS_VIDEO_INTERVAL', 0)
            if interval and interval > 0:
                if (idx + 1) % interval == 0 or (idx + 1) == total:
                    logger.info(
                        f"Completed videos: {idx + 1}/{total} ({(idx + 1)/total*100:.1f}%)")
        logger.info("All videos processed.")
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    except Exception as e:
        logger.error(f"Error during processing loop: {e}")
    finally:
        video_handler.cleanup()


if __name__ == "__main__":
    main()
