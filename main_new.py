#!/usr/bin/env python3
"""
Simplified Clash Royale Troop Detection Tool
Clean, straightforward implementation focusing on core functionality
"""

import logging
import sys
import os
from typing import List

# Import simplified modules
import config_new as config
from video_handler import VideoHandler
from detector import TroopDetector
from output import OutputWriter

class TroopDetectionTool:
    def __init__(self):
        """Initialize the detection tool"""
        self.setup_logging()
        
        # Initialize components
        self.video_handler = VideoHandler()
        self.detector = TroopDetector()
        self.output_writer = OutputWriter()
        
        self.processed_frames = 0
        self.logger.info("Troop Detection Tool initialized")
    
    def setup_logging(self):
        """Setup simple logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('detection.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def process_frame(self, frame, frame_number: int) -> bool:
        """Process a single frame"""
        try:
            # Detect and track objects
            tracked_objects, debug_frame, placement_events = self.detector.process_frame(frame, frame_number)
            
            # Save frame if objects detected
            if tracked_objects:
                self.output_writer.save_frame_with_objects(debug_frame, tracked_objects)
            
            # Log placement events
            if placement_events:
                for event in placement_events:
                    self.logger.info(f"Card placed: {event['card_name']} ({event['type']})")
            
            # Update the frame for display
            frame[:] = debug_frame
            
            self.processed_frames += 1
            
            # Log progress
            if self.processed_frames % 50 == 0:
                stats = self.output_writer.get_stats()
                self.logger.info(f"Processed {self.processed_frames} frames. "
                               f"Saved: {stats['frames_saved']}, Objects: {stats['objects_detected']}")
            
            return False  # Continue processing
            
        except Exception as e:
            self.logger.error(f"Error processing frame {frame_number}: {e}")
            return False
    
    def run(self, start_time: float = None):
        """Main execution"""
        try:
            self.logger.info("Starting Troop Detection Tool")
            
            # Use command line start time or config default
            if start_time is not None:
                self.logger.info(f"Starting analysis at {start_time} seconds")
            else:
                start_time = getattr(config, 'START_TIME_SECONDS', 0.0)
                if start_time > 0:
                    self.logger.info(f"Starting analysis at {start_time} seconds (from config)")
            
            # Determine video sources
            if config.TEST_VIDEO_PATH:
                video_sources = [config.TEST_VIDEO_PATH]
                self.logger.info(f"Processing local video: {config.TEST_VIDEO_PATH}")
            else:
                video_sources = config.YOUTUBE_URLS
                self.logger.info(f"Processing {len(video_sources)} YouTube videos")
            
            if not video_sources:
                self.logger.error("No video sources configured")
                return
            
            # Process each video
            for i, video_source in enumerate(video_sources):
                self.logger.info(f"Processing video {i+1}/{len(video_sources)}: {video_source}")
                
                success = self.video_handler.process_video(video_source, self.process_frame, start_time)
                
                if not success:
                    self.logger.error(f"Failed to process video: {video_source}")
                    continue
            
            # Save final summary
            self.output_writer.save_summary()
            
            final_stats = self.output_writer.get_stats()
            self.logger.info("="*50)
            self.logger.info("PROCESSING COMPLETED")
            self.logger.info("="*50)
            self.logger.info(f"Total frames processed: {self.processed_frames}")
            self.logger.info(f"Frames with objects saved: {final_stats['frames_saved']}")
            self.logger.info(f"Total objects detected: {final_stats['objects_detected']}")
            self.logger.info(f"Output directory: {config.OUTPUT_DIR}")
            
        except KeyboardInterrupt:
            self.logger.info("Processing interrupted by user")
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            raise

def main():
    """Entry point"""
    try:
        # Check for command line start time argument
        start_time = None
        if len(sys.argv) > 1:
            try:
                start_time = float(sys.argv[1])
                print(f"Using command line start time: {start_time} seconds")
            except ValueError:
                print(f"Invalid start time '{sys.argv[1]}', using config default")
        
        tool = TroopDetectionTool()
        tool.run(start_time)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
