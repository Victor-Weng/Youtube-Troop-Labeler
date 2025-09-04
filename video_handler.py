"""
Simple video handler for YouTube and local video processing
"""

import cv2
import yt_dlp
import os
import logging
from typing import Optional
import config_new as config
from troop_tracker import TroopTracker


class VideoHandler:
    """Handles video file processing and frame extraction"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.cap = None
        self.frame_count = 0
        self.troop_tracker = TroopTracker()

    def get_youtube_stream_url(self, youtube_url: str) -> Optional[str]:
        """Get direct stream URL from YouTube"""

        try:
            ydl_opts = {
                'format': 'bestvideo[height>=720]',
                'quiet': True,
                'no_warnings': True,
                'cookiesfrombrowser': ('firefox',),
                # 'cookies': os.path.join(os.path.dirname(__file__), 'cookies.txt'), # cookies to access
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
                return info['url']

        except Exception as e:
            self.logger.error(f"Failed to resolve YouTube URL: {e}")
            return None

    def process_video(self, detector):
        """Process video file and run detection on each frame"""
        try:
            # Determine video source
            if config.TEST_VIDEO_PATH:
                video_source = config.TEST_VIDEO_PATH
                self.logger.info(f"Processing local video: {video_source}")
                self.cap = cv2.VideoCapture(video_source)
            elif config.YOUTUBE_URLS:
                youtube_url = config.YOUTUBE_URLS[0]  # Use first URL
                self.logger.info(f"Processing YouTube video: {youtube_url}")
                stream_url = self.get_youtube_stream_url(youtube_url)
                if not stream_url:
                    raise ValueError(
                        f"Failed to get stream URL for: {youtube_url}")
                self.cap = cv2.VideoCapture(stream_url)
            else:
                raise ValueError(
                    "No video source specified in config (TEST_VIDEO_PATH or YOUTUBE_URLS)")

            if not self.cap.isOpened():
                raise ValueError(f"Could not open video source")

            # Get video info
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)

            self.logger.info(
                f"Video info - Total frames: {total_frames}, FPS: {fps}")

            # Seek to start time if specified
            start_time = getattr(config, 'START_TIME_SECONDS', 0.0)
            if start_time > 0:
                self.logger.info(f"Seeking to {start_time} seconds...")
                self.cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)

            # Create display window
            cv2.namedWindow('Card Detection', cv2.WINDOW_NORMAL)
            # Portrait orientation
            cv2.resizeWindow('Card Detection', 720, 1280)

            # Process each frame
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                self.frame_count += 1

                # Skip frames based on config
                if hasattr(config, 'FRAME_SKIP') and self.frame_count % config.FRAME_SKIP != 0:
                    continue

                # Standardize frame to portrait 720x1280 (same as config coordinates)
                height, width = frame.shape[:2]
                if width != 720 or height != 1280:
                    frame = cv2.resize(frame, (720, 1280))

                # Apply additional resize factor if configured
                if hasattr(config, 'RESIZE_FACTOR') and config.RESIZE_FACTOR != 1.0:
                    height, width = frame.shape[:2]
                    new_width = int(width * config.RESIZE_FACTOR)
                    new_height = int(height * config.RESIZE_FACTOR)
                    frame = cv2.resize(frame, (new_width, new_height))

                try:
                    # Run detection on frame - this handles ALL tracking updates AND drawing internally
                    detected_objects, debug_frame, placement_events = detector.process_frame(
                        frame, self.frame_count
                    )

                    # Display frame (debug_frame already has detection + tracking boxes drawn)
                    cv2.imshow('Card Detection', debug_frame)

                    # Add delay for testing (adjust the value as needed)
                    import time
                    time.sleep(config.DELAY)  # delay to limit call rates

                    # Check for key presses
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESC key - quit
                        self.logger.info(
                            "ESC pressed - stopping video processing")
                        break
                    elif key == 32:  # SPACE key - pause
                        self.logger.info(
                            "SPACE pressed - paused (press any key to continue)")
                        cv2.waitKey(0)

                    # Log progress every 100 frames
                    if self.frame_count % 100 == 0:
                        if total_frames > 0:
                            progress = (self.frame_count / total_frames) * 100
                            self.logger.info(
                                f"Processed frame {self.frame_count}/{total_frames} ({progress:.1f}%)")
                        else:
                            self.logger.info(
                                f"Processed frame {self.frame_count}")

                except Exception as e:
                    self.logger.error(
                        f"Error processing frame {self.frame_count}: {e}")
                    continue

            cv2.destroyAllWindows()
            self.logger.info(
                f"Video processing complete. Processed {self.frame_count} frames.")

        except Exception as e:
            self.logger.error(f"Error in video processing: {e}")
            raise

    def cleanup(self):
        """Clean up video capture resources"""
        if self.cap:
            self.cap.release()
            self.logger.info("Video capture resources cleaned up")
