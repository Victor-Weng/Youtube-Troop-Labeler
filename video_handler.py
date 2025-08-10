"""
Simple video handler for YouTube and local video processing
"""

import cv2
import yt_dlp
import logging
from typing import Optional, Callable
import config_new as config

class VideoHandler:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def get_youtube_stream_url(self, youtube_url: str) -> Optional[str]:
        """Get direct stream URL from YouTube"""
        try:
            ydl_opts = {
                'format': 'bestvideo[height>=720]',
                'quiet': True,
                'no_warnings': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
                return info['url']
                
        except Exception as e:
            self.logger.error(f"Failed to resolve YouTube URL: {e}")
            return None
    
    def process_video(self, video_source: str, frame_callback: Callable, start_time: float = None) -> bool:
        """Process video frames with callback function"""
        try:
            # Use config start time if not specified
            if start_time is None:
                start_time = getattr(config, 'START_TIME_SECONDS', 0.0)
            
            # Determine if it's a YouTube URL or local file
            if video_source.startswith('http'):
                stream_url = self.get_youtube_stream_url(video_source)
                if not stream_url:
                    return False
                cap = cv2.VideoCapture(stream_url)
            else:
                cap = cv2.VideoCapture(video_source)
            
            if not cap.isOpened():
                self.logger.error(f"Failed to open video: {video_source}")
                return False
            
            # Seek to start time if specified
            if start_time > 0:
                print(f"Seeking to {start_time} seconds...")
                cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)  # Convert to milliseconds
            
            # Create display window
            cv2.namedWindow('Troop Detection', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Troop Detection', 720, 1280)  # Portrait orientation to match config UI
            
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Skip frames based on config
                if frame_count % config.FRAME_SKIP != 0:
                    continue
                
                # Standardize frame to portrait 720x1280 (same as config UI)
                height, width = frame.shape[:2]
                if width != 720 or height != 1280:
                    frame = cv2.resize(frame, (720, 1280))
                
                # Apply additional resize factor if configured
                if config.RESIZE_FACTOR != 1.0:
                    height, width = frame.shape[:2]
                    new_width = int(width * config.RESIZE_FACTOR)
                    new_height = int(height * config.RESIZE_FACTOR)
                    frame = cv2.resize(frame, (new_width, new_height))
                
                # Process frame
                should_stop = frame_callback(frame, frame_count)
                
                # Display frame
                cv2.imshow('Troop Detection', frame)
                
                # Check for key presses
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or should_stop:  # ESC key or callback requests stop
                    break
                elif key == 32:  # SPACE key - pause
                    cv2.waitKey(0)
            
            cap.release()
            cv2.destroyAllWindows()
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing video: {e}")
            return False
