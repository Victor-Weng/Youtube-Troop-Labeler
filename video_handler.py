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
        # Early stop state
        self.active_streak_start_frame = None
        self.early_stop_triggered = False
        self.awaiting_new_game = False
        # Pre-compute early stop frame threshold if enabled
        self.stop_active_early = getattr(config, 'STOP_ACTIVE_EARLY', False)
        self.stop_active_early_time = getattr(config, 'STOP_ACTIVE_EARLY_TIME', 0.0)
        self.stop_active_frame_threshold = None  # will fill in after fps known
        self.fast_skip_end_frame = None  # end frame (absolute frame index) after instant fast skip
        self.fast_skip_done = False  # whether we've already performed the instant fast skip

    def _reset_for_new_game(self, detector):
        """Clear per-game state so new game can be captured fresh."""
        # Reset tracker & any dataset saving internal buffers
        try:
            self.troop_tracker = TroopTracker()
        except Exception:
            pass
        # If detector has its own per-game reset, call it
        if hasattr(detector, 'reset_for_new_game') and callable(getattr(detector, 'reset_for_new_game')):
            try:
                detector.reset_for_new_game()
            except Exception:
                pass
        # Reset streak/flags
        self.active_streak_start_frame = None
        self.early_stop_triggered = False
        self.awaiting_new_game = False

    def _draw_active_region(self, frame, is_game_active):
        """Draw region visualization on the frame"""
        x, y, w, h = config.ACTIVE_REGION

        # Choose color based on game state
        if is_game_active:
            color = (0, 255, 0)  # Green for active
            text = "ACTIVE"
        else:
            color = (0, 0, 255)  # Red for inactive
            text = "INACTIVE"

        # Draw timer region rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Add label above the rectangle
        cv2.putText(frame, text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw crosshair in center of region for precise targeting
        center_x, center_y = x + w//2, y + h//2
        cv2.line(frame, (center_x - 5, center_y),
                 (center_x + 5, center_y), color, 1)
        cv2.line(frame, (center_x, center_y - 5),
                 (center_x, center_y + 5), color, 1)

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

    def _fast_forward_frames(self, frames_to_skip:int):
        """Grab/drop the specified number of frames quickly without full decode.
        We still increment frame_count respecting FRAME_SKIP logic by counting every grabbed frame.
        """
        if frames_to_skip <= 0 or not self.cap:
            return
        grabbed = 0
        while grabbed < frames_to_skip:
            ret = self.cap.grab()
            if not ret:
                break
            grabbed += 1
            self.frame_count += 1
        self.logger.info(f"Fast-forward skipped {grabbed} frames instantly")

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

            # Compute early stop frame threshold AFTER fps is known
            if self.stop_active_early and fps and fps > 0:
                self.stop_active_frame_threshold = int(self.stop_active_early_time * fps)
                remainder_secs = getattr(config, 'STOP_ACTIVE_EARLY_SKIP_SECONDS', 0.0)
                self.fast_skip_end_frame = self.stop_active_frame_threshold + int(remainder_secs * fps)
                self.logger.info(
                    f"Early stop thresholds: trigger={self.stop_active_frame_threshold} (at {self.stop_active_early_time}s) fast_skip_end={self.fast_skip_end_frame} (remainder {remainder_secs}s)")

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
                    # Always poll for game active state
                    detector.detect_game_active(frame)

                    # Manage active streak timing
                    if detector.is_game:
                        if self.active_streak_start_frame is None:
                            self.active_streak_start_frame = self.frame_count
                        # Check early stop condition
                        if (self.stop_active_early and not self.early_stop_triggered and
                                self.stop_active_frame_threshold is not None and
                                (self.frame_count - self.active_streak_start_frame) >= self.stop_active_frame_threshold):
                            self.early_stop_triggered = True
                            self.awaiting_new_game = True
                            # Compute how many frames to skip instantly to reach fast_skip_end_frame
                            if self.fast_skip_end_frame is not None:
                                frames_to_skip = self.fast_skip_end_frame - self.frame_count
                                if frames_to_skip > 0:
                                    self.logger.info(
                                        f"Early stop triggered at frame {self.frame_count} (active streak frames: {self.frame_count - self.active_streak_start_frame}); instant skipping {frames_to_skip} frames to {self.fast_skip_end_frame}")
                                    self._fast_forward_frames(frames_to_skip)
                                    self.fast_skip_done = True
                                else:
                                    self.logger.info(
                                        f"Early stop triggered at frame {self.frame_count}; no fast skip needed (frames_to_skip={frames_to_skip})")
                            else:
                                self.logger.info(
                                    f"Early stop triggered at frame {self.frame_count} but fast_skip_end_frame unset; no fast skip performed")
                    else:
                        # Game inactive; if we were awaiting a new game after early stop, clear state so next active streak is processed
                        if self.awaiting_new_game and self.early_stop_triggered:
                            # Next time we see detector.is_game True, start new game
                            pass
                        # Reset streak if not awaiting new game
                        if not self.awaiting_new_game:
                            self.active_streak_start_frame = None

                    # After instant skip, we are directly in polling phase (fast skip done once)

                    process_full = detector.is_game and not (self.early_stop_triggered and self.awaiting_new_game)

                    if process_full and detector.is_game:
                        # Normal processing
                        detected_objects, debug_frame, placement_events = detector.process_frame(
                            frame, self.frame_count
                        )
                    else:
                        # Either inactive or skipping due to early stop
                        debug_frame = frame.copy()
                        if detector.is_game and self.early_stop_triggered and self.awaiting_new_game:
                            phase_text = "EARLY STOP POLL"
                            cv2.putText(debug_frame, phase_text, (10, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        elif not detector.is_game:
                            cv2.putText(debug_frame, "Game Inactive", (10, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        detected_objects, placement_events = [], []

                    # Transition from skip state to new game start
                    if (self.early_stop_triggered and self.awaiting_new_game and not detector.is_game):
                        # Inactive after early stop; prepare for new game
                        self.logger.info("Inactive after early stop; waiting for next activation to reset game state.")
                    if (self.early_stop_triggered and self.awaiting_new_game and detector.is_game and not process_full):
                        # Still skipping inside the same active streak
                        pass
                    if (self.early_stop_triggered and self.awaiting_new_game and not detector.is_game):
                        # Already handled above; when it becomes active again we reset
                        pass
                    if (self.early_stop_triggered and self.awaiting_new_game and not detector.is_game):
                        pass
                    # When new activation after early stop + inactive period occurs -> reset
                    if (self.early_stop_triggered and self.awaiting_new_game and detector.is_game and self.active_streak_start_frame is None):
                        self._reset_for_new_game(detector)
                        continue
                    if (self.early_stop_triggered and self.awaiting_new_game and detector.is_game and
                            (self.frame_count - self.active_streak_start_frame) <= 1):
                        self.logger.info("New active streak detected after early stop; resetting state.")
                        self._reset_for_new_game(detector)
                        continue

                    # Add timer region visualization to debug_frame
                    self._draw_active_region(debug_frame, detector.is_game)

                    # Display frame (debug_frame already has detection + tracking boxes drawn)
                    cv2.imshow('Card Detection', debug_frame)

                    # Add delay (reduced if we're skipping after early stop)
                    import time
                    if (self.early_stop_triggered and self.awaiting_new_game and detector.is_game):
                        poll_delay = getattr(config, 'STOP_ACTIVE_EARLY_DELAY', getattr(config, 'DELAY', 0.0))
                        time.sleep(poll_delay)
                    else:
                        time.sleep(config.DELAY)

                    # Check for key presses
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESC key - quit
                        self.logger.info("ESC pressed - stopping video processing")
                        # Attempt graceful close of detector dataset saver if present
                        try:
                            if hasattr(detector, 'close'):
                                detector.close()
                        except Exception:
                            pass
                        break
                    elif key == 32:  # SPACE key - pause
                        self.logger.info(
                            "SPACE pressed - paused (press any key to continue)")
                        cv2.waitKey(0)

                    # Log progress every 100 frames
                    if self.frame_count % 100 == 0:
                        if total_frames > 0:
                            progress = (self.frame_count /
                                        total_frames) * 100
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
            # Ensure detector closed at end (if not already)
            try:
                if hasattr(detector, 'close'):
                    detector.close()
            except Exception:
                pass
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
