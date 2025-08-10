"""
Stream handler for YouTube video processing with automatic retry logic
"""

import cv2
import yt_dlp
import time
import logging
import numpy as np
from typing import Optional, Tuple, Dict, Any
import config

class StreamHandler:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.current_stream_url = None
        self.current_video_id = None
        self.retry_count = 0
        self.consecutive_failed_frames = 0
        self.last_successful_frame_time = 0
        
    def resolve_youtube_url(self, youtube_url: str) -> Optional[str]:
        """
        Resolve YouTube URL to direct stream URL using yt-dlp
        """
        try:
            ydl_opts = {
                'format': 'bestvideo[height>=720]',  # Limit to 720p for performance
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
                stream_url = info['url']
                self.current_video_id = info.get('id', 'unknown')
                self.current_stream_url = stream_url
                
                self.logger.info(f"Successfully resolved stream URL for video {self.current_video_id}")
                return stream_url
                
        except Exception as e:
            self.logger.error(f"Failed to resolve YouTube URL {youtube_url}: {str(e)}")
            return None
    
    def create_video_capture(self, stream_url: str) -> Optional[cv2.VideoCapture]:
        """
        Create VideoCapture object from stream URL
        """
        try:
            cap = cv2.VideoCapture(stream_url)
            
            # Set buffer size to minimize latency
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Set timeout for frame reading (only if supported)
            try:
                cap.set(cv2.CAP_PROP_TIMEOUT, config.stream_timeout_seconds * 1000)
            except AttributeError:
                # CAP_PROP_TIMEOUT not supported in this OpenCV version
                pass
            
            if not cap.isOpened():
                self.logger.error("Failed to open video capture")
                return None
                
            self.logger.info("Successfully created video capture")
            return cap
            
        except Exception as e:
            self.logger.error(f"Failed to create video capture: {str(e)}")
            return None
    
    def read_frame(self, cap: cv2.VideoCapture) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read frame from video capture with error handling
        """
        try:
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1280, 720))
            
            if ret and frame is not None:
                self.consecutive_failed_frames = 0
                self.last_successful_frame_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # seconds
                return True, frame
            else:
                self.consecutive_failed_frames += 1
                self.logger.warning(f"Failed to read frame. Consecutive failures: {self.consecutive_failed_frames}")
                return False, None
                
        except Exception as e:
            self.consecutive_failed_frames += 1
            self.logger.error(f"Exception while reading frame: {str(e)}")
            return False, None
    
    def should_retry_stream(self) -> bool:
        """
        Determine if stream should be retried based on failure count
        """
        return (self.consecutive_failed_frames >= config.max_consecutive_failed_frames_before_retry and
                self.retry_count < config.stream_retry_limit)
    
    def attempt_stream_recovery(self, youtube_url: str) -> Optional[cv2.VideoCapture]:
        """
        Attempt to recover stream by re-resolving URL and creating new capture
        """
        self.retry_count += 1
        self.logger.info(f"Attempting stream recovery (attempt {self.retry_count}/{config.stream_retry_limit})")
        
        # Re-resolve stream URL
        new_stream_url = self.resolve_youtube_url(youtube_url)
        if not new_stream_url:
            self.logger.error("Failed to re-resolve stream URL during recovery")
            return None
        
        # Create new video capture
        new_cap = self.create_video_capture(new_stream_url)
        if not new_cap:
            self.logger.error("Failed to create new video capture during recovery")
            return None
        
        # Reset failure counters
        self.consecutive_failed_frames = 0
        self.logger.info("Stream recovery successful")
        return new_cap
    
    def process_video_stream(self, youtube_url: str, frame_callback) -> bool:
        """
        Main method to process video stream with automatic retry logic
        """
        self.retry_count = 0
        self.consecutive_failed_frames = 0
        
        # Initial stream resolution
        stream_url = self.resolve_youtube_url(youtube_url)
        if not stream_url:
            self.logger.error(f"Failed to resolve stream URL for {youtube_url}")
            return False
        
        # Create initial video capture
        cap = self.create_video_capture(stream_url)
        if not cap:
            self.logger.error(f"Failed to create video capture for {youtube_url}")
            return False
        
        frame_count = 0
        start_time = time.time()
        
        # Create window for video display (only if GUI is supported)
        gui_supported = True
        try:
            cv2.namedWindow('Troop Detection', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Troop Detection', 1280, 720)
            gui_supported = True
            self.logger.info("GUI mode enabled - video will be displayed")
        except cv2.error:
            self.logger.warning("GUI not supported - running in headless mode")
            gui_supported = False
        
        try:
            # Add before loop
            last_captured_time = None
            target_interval = 1.0 / config.frame_rate  # seconds per frame

            while True:
                if self.should_retry_stream():
                    cap.release()
                    cap = self.attempt_stream_recovery(youtube_url)
                    if not cap:
                        break
                    continue

                ret, frame = self.read_frame(cap)
                if not ret:
                    time.sleep(0.1)
                    continue

                video_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

                # Only process frames at the desired interval
                if last_captured_time is None or (video_timestamp - last_captured_time) >= target_interval:
                    frame_count += 1
                    last_captured_time = video_timestamp

                    stop_processing = frame_callback(frame, frame_count, video_timestamp)

                    if frame is not None and gui_supported:
                        display_frame = frame.copy()
                        cv2.putText(display_frame, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(display_frame, f"Time: {video_timestamp:.1f}s", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.imshow('Troop Detection', display_frame)

                        key = cv2.waitKey(1) & 0xFF
                        if key == 27:
                            break
                        elif key == 32:
                            cv2.waitKey(0)

                    if stop_processing:
                        break

                    
        except KeyboardInterrupt:
            self.logger.info("Processing interrupted by user")
        except Exception as e:
            self.logger.error(f"Unexpected error during stream processing: {str(e)}")
        finally:
            cap.release()
            if gui_supported:
                cv2.destroyAllWindows()
            self.logger.info(f"Stream processing completed. Processed {frame_count} frames")
        
        return True
    
    def get_stream_info(self, youtube_url: str) -> Optional[Dict[str, Any]]:
        """
        Get information about the YouTube video without downloading
        """
        try:
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
                return {
                    'title': info.get('title', 'Unknown'),
                    'duration': info.get('duration', 0),
                    'view_count': info.get('view_count', 0),
                    'upload_date': info.get('upload_date', 'Unknown'),
                    'id': info.get('id', 'Unknown')
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get stream info: {str(e)}")
            return None 