#!/usr/bin/env python3
"""
Main entry point for Clash Royale Troop Annotation Tool
"""

import logging
import time
import sys
import os
import cv2
from typing import List, Dict, Any

# Import our modules
import config
from stream_handler import StreamHandler
from arena_color import ArenaColorDetector
from card_tracker import CardTracker
from pixel_diff import PixelDifferenceDetector
from troop_detector import TroopDetector
from troop_tracker import TroopTracker
from output_writer import OutputWriter

class TroopAnnotationTool:
    def __init__(self):
        """Initialize the troop annotation tool"""
        self.setup_logging()
        
        # Initialize components
        self.stream_handler = StreamHandler()
        self.arena_detector = ArenaColorDetector()
        self.card_tracker = CardTracker()
        self.pixel_detector = PixelDifferenceDetector(self.arena_detector)
        self.troop_detector = TroopDetector(self.card_tracker, self.pixel_detector, self.arena_detector)
        self.troop_tracker = TroopTracker()
        self.output_writer = OutputWriter()
        
        # State tracking
        self.current_video_index = 0
        self.total_videos_processed = 0
        self.total_frames_processed = 0
        self.total_troops_detected = 0
        
        self.logger.info("Troop Annotation Tool initialized successfully")
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, config.log_level),
            format=log_format,
            handlers=[
                logging.FileHandler(config.log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Logging setup completed")
    
    def process_frame(self, frame, frame_number: int, timestamp: float) -> bool:
        """
        Process a single frame through the pipeline
        
        Returns:
            bool: True if processing should stop, False to continue
        """
        try:
            # Skip frames based on frame rate configuration
            if frame_number % config.frame_skip != 0:
                return False
            
            # Resize frame if needed
            if config.resize_factor != 1.0:
                height, width = frame.shape[:2]
                new_width = int(width * config.resize_factor)
                new_height = int(height * config.resize_factor)
                frame = cv2.resize(frame, (new_width, new_height))
            
            # Get arena color baseline on first frame
            if frame_number == 1:
                arena_color = self.arena_detector.sample_arena_color(frame)
                if arena_color is not None:
                    self.pixel_detector.set_arena_color(arena_color)
                    self.logger.info(f"Detected arena color: {arena_color}")
                else:
                    self.logger.warning("Failed to detect arena color")
            
            # Track cards in hands
            ally_cards, enemy_cards = self.card_tracker.update_hand_states(frame)
            
            # Also detect cards in hand using the dedicated method
            detected_hand_cards = self.card_tracker.detect_hand_cards(frame)
            if detected_hand_cards:
                self.logger.info(f"Detected cards in hand: {detected_hand_cards}")
            
            # Check for card disappearances (placements)
            placement_events = self.card_tracker.detect_placements(frame_number)
            
            # Detect pixel differences for troop placement
            if placement_events:
                for event in placement_events:
                    self.logger.info(f"Card placement detected: {event}")
                    
                    # Look for pixel differences in the next few frames
                    # This will be handled by the pixel difference detector
                    self.pixel_detector.set_pending_placement(event)
            
            # Detect troops using pixel differences
            detected_troops = self.pixel_detector.detect_troops(frame)
            
            # Add new troops to tracking
            for troop in detected_troops:
                troop['placement_frame'] = frame_number
                self.troop_tracker.add_troop_for_tracking(troop)
            
            # Update troop tracking
            active_troops = self.troop_tracker.update_tracking(frame, frame_number)
            
            # Save frame if there are active troops
            if active_troops:
                video_id = self.stream_handler.current_video_id or "unknown"
                self.output_writer.save_frame_with_labels(
                    frame, frame_number, active_troops, video_id
                )
                self.total_troops_detected += len(active_troops)
                
                self.logger.debug(f"Frame {frame_number}: {len(active_troops)} active troops")
            
            self.total_frames_processed += 1
            
            # Log progress every 100 frames
            if frame_number % 100 == 0:
                self.logger.info(f"Processed {frame_number} frames, {len(active_troops)} active troops")
            
            # Add debugging information to frame
            self._add_debug_overlay(frame, frame_number, detected_troops, active_troops, 
                                  ally_cards, enemy_cards, placement_events, detected_hand_cards)
            
            return False  # Continue processing
            
        except Exception as e:
            self.logger.error(f"Error processing frame {frame_number}: {str(e)}")
            return False  # Continue processing despite errors
    
    def _add_debug_overlay(self, frame, frame_number, detected_troops, active_troops, 
                          ally_cards, enemy_cards, placement_events, detected_hand_cards):
        """Add debugging information overlay to the frame"""
        try:
            # Draw arena sampling region
            if hasattr(self.arena_detector, 'sample_coords'):
                x, y, w, h = self.arena_detector.sample_coords
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Arena Sample", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw ally hand regions
            for i, (x, y, w, h) in enumerate(config.ally_hand_coords):
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, f"Ally {i+1}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                # Add detected card name if available
                if detected_hand_cards and i < len(detected_hand_cards):
                    card_name = detected_hand_cards[i]
                    if card_name != "Unknown":
                        cv2.putText(frame, card_name, (x, y+h+20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            
            # Draw enemy hand regions
            for i, (x, y, w, h) in enumerate(config.enemy_hand_coords):
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, f"Enemy {i+1}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Draw detected troops
            for troop in detected_troops:
                x, y, w, h = troop['bbox']
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(frame, f"Troop {troop['confidence']:.2f}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # Draw active tracked troops
            for troop in active_troops:
                x, y, w, h = troop['bbox']
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
                cv2.putText(frame, f"Tracked {troop.get('id', '?')}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            # Add statistics overlay
            cv2.putText(frame, f"Frame: {frame_number}", (10, frame.shape[0] - 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Detected: {len(detected_troops)}", (10, frame.shape[0] - 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Tracked: {len(active_troops)}", (10, frame.shape[0] - 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Placements: {len(placement_events)}", (10, frame.shape[0] - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Hand Cards: {len([c for c in detected_hand_cards if c != 'Unknown'])}", (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
        except Exception as e:
            self.logger.error(f"Error adding debug overlay: {str(e)}")
    
    def process_video(self, youtube_url: str) -> bool:
        """Process a single YouTube video"""
        try:
            self.logger.info(f"Starting processing of video: {youtube_url}")
            
            # Get video info
            video_info = self.stream_handler.get_stream_info(youtube_url)
            if video_info:
                self.logger.info(f"Video: {video_info['title']} (Duration: {video_info['duration']}s)")
            
            # Process the video stream
            success = self.stream_handler.process_video_stream(youtube_url, self.process_frame)
            
            if success:
                self.logger.info(f"Successfully processed video: {youtube_url}")
                self.total_videos_processed += 1
                return True
            else:
                self.logger.error(f"Failed to process video: {youtube_url}")
                self.logger.error("This could be due to:")
                self.logger.error("- Network connectivity issues")
                self.logger.error("- YouTube video restrictions")
                self.logger.error("- OpenCV compatibility issues")
                return False
            
        except Exception as e:
            self.logger.error(f"Unexpected error processing video {youtube_url}: {str(e)}")
            self.logger.error("Exception details:", exc_info=True)
            return False
    
    def run(self):
        """Main execution loop"""
        try:
            self.logger.info("Starting Troop Annotation Tool")
            
            if config.test_mode:
                self.logger.info("Running in TEST MODE with local video file")
                self.logger.info(f"Test video: {config.test_video_path}")
                
                # Check if test video exists
                if not os.path.exists(config.test_video_path):
                    self.logger.error(f"Test video file not found: {config.test_video_path}")
                    self.logger.error("Please update config.test_video_path with a valid video file")
                    return
                
                # Process test video
                success = self.process_test_video(config.test_video_path)
                
            else:
                self.logger.info(f"Processing {len(config.youtube_urls)} YouTube videos")
                
                start_time = time.time()
                
                # Process each video in the list
                videos_processed = 0
                for i, youtube_url in enumerate(config.youtube_urls):
                    self.current_video_index = i
                    self.logger.info(f"Processing video {i+1}/{len(config.youtube_urls)}")
                    
                    # Process the video
                    success = self.process_video(youtube_url)
                    
                    if not success:
                        self.logger.warning(f"Skipping to next video due to processing failure")
                        continue
                    
                    videos_processed += 1
                    # Small delay between videos
                    time.sleep(1)
                
                # Only finalize if at least one video was processed
                if videos_processed > 0:
                    self.finalize_processing(start_time)
                else:
                    self.logger.error("No videos were successfully processed. Cannot finalize output.")
                    self.logger.error("Please check your YouTube URLs and internet connection.")
            
        except KeyboardInterrupt:
            self.logger.info("Processing interrupted by user")
        except Exception as e:
            self.logger.error(f"Unexpected error in main execution: {str(e)}")
        finally:
            self.cleanup()
    
    def finalize_processing(self, start_time: float):
        """Finalize the processing and save results"""
        try:
            processing_time = time.time() - start_time
            
            # Check if we actually processed anything
            if self.total_frames_processed == 0:
                self.logger.warning("No frames were processed. Skipping dataset creation.")
                return
            
            # Save final dataset info
            video_id = "batch_processing"
            self.output_writer.save_dataset_info(
                video_id, 
                self.total_frames_processed,
                self.total_troops_detected,
                processing_time
            )
            
            # Save classes file
            self.output_writer.save_classes_file()
            
            # Validate dataset
            validation = self.output_writer.validate_dataset()
            if validation['valid']:
                self.logger.info("Dataset validation passed")
            else:
                self.logger.warning(f"Dataset validation issues: {validation['errors']}")
            
            # Print summary
            self.logger.info("=" * 50)
            self.logger.info("PROCESSING COMPLETED")
            self.logger.info("=" * 50)
            self.logger.info(f"Videos processed: {self.total_videos_processed}")
            self.logger.info(f"Frames processed: {self.total_frames_processed}")
            self.logger.info(f"Troops detected: {self.total_troops_detected}")
            self.logger.info(f"Processing time: {processing_time:.2f} seconds")
            self.logger.info(f"Output directory: {config.output_dir}")
            
        except Exception as e:
            self.logger.error(f"Error during finalization: {str(e)}")
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            # Only cleanup if we actually processed something
            if self.total_frames_processed > 0:
                # Clean up empty files
                self.output_writer.cleanup_empty_files()
                
                # Get final statistics
                stats = self.output_writer.get_output_statistics()
                self.logger.info(f"Final output statistics: {stats}")
            else:
                self.logger.info("No processing was done. Skipping cleanup operations.")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")

def main():
    """Main entry point"""
    try:
        # Check if output directory exists, create if not
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Create and run the tool
        tool = TroopAnnotationTool()
        tool.run()
        
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 