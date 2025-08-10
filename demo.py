#!/usr/bin/env python3
"""
Demo script for testing the troop detection system with local video files
"""

import cv2
import logging
import sys
import os
from typing import List, Dict, Any

# Import our modules
import config
from arena_color import ArenaColorDetector
from card_tracker import CardTracker
from pixel_diff import PixelDifferenceDetector
from troop_detector import TroopDetector
from troop_tracker import TroopTracker

class DemoMode:
    def __init__(self):
        """Initialize the demo mode"""
        self.setup_logging()
        
        # Initialize components
        self.arena_detector = ArenaColorDetector()
        self.card_tracker = CardTracker()
        self.pixel_detector = PixelDifferenceDetector(self.arena_detector)
        self.troop_detector = TroopDetector(self.card_tracker, self.pixel_detector, self.arena_detector)
        self.troop_tracker = TroopTracker()
        
        # Demo state
        self.frame_count = 0
        self.total_troops_detected = 0
        
        self.logger.info("Demo mode initialized successfully")
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def process_frame(self, frame, frame_number: int) -> bool:
        """Process a single frame"""
        try:
            # Skip frames for faster processing
            if frame_number % 30 != 0:  # Process every 30th frame
                return False
            
            # Resize frame for processing
            height, width = frame.shape[:2]
            if width > 1280:
                scale = 1280 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
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
            ally_cards = self.card_tracker.detect_ally_cards(frame)
            enemy_cards = self.card_tracker.detect_enemy_cards(frame)
            
            # Check for card disappearances (placements)
            placement_events = self.card_tracker.detect_placements(ally_cards, enemy_cards)
            
            # Detect troops using pixel differences
            detected_troops = self.pixel_detector.detect_troops(frame)
            
            # Add new troops to tracking
            for troop in detected_troops:
                troop['placement_frame'] = frame_number
                self.troop_tracker.add_troop_for_tracking(troop)
            
            # Update troop tracking
            active_troops = self.troop_tracker.update_tracking(frame, frame_number)
            
            # Update statistics
            self.total_troops_detected += len(detected_troops)
            
            # Add debugging information to frame
            self._add_debug_overlay(frame, frame_number, detected_troops, active_troops, 
                                  ally_cards, enemy_cards, placement_events)
            
            # Display the frame
            cv2.imshow('Demo - Troop Detection', frame)
            
            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                return True  # Stop processing
            elif key == 32:  # SPACE key
                self.logger.info("Paused. Press any key to continue...")
                cv2.waitKey(0)
            
            return False  # Continue processing
            
        except Exception as e:
            self.logger.error(f"Error processing frame {frame_number}: {str(e)}")
            return False
    
    def _add_debug_overlay(self, frame, frame_number, detected_troops, active_troops, 
                          ally_cards, enemy_cards, placement_events):
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
            cv2.putText(frame, f"Frame: {frame_number}", (10, frame.shape[0] - 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Detected: {len(detected_troops)}", (10, frame.shape[0] - 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Tracked: {len(active_troops)}", (10, frame.shape[0] - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Placements: {len(placement_events)}", (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
        except Exception as e:
            self.logger.error(f"Error adding debug overlay: {str(e)}")
    
    def run_demo(self, video_path: str):
        """Run demo with a local video file"""
        try:
            if not os.path.exists(video_path):
                self.logger.error(f"Video file not found: {video_path}")
                return False
            
            self.logger.info(f"Starting demo with video: {video_path}")
            
            # Open video capture
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.logger.error("Failed to open video file")
                return False
            
            # Create window
            cv2.namedWindow('Demo - Troop Detection', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Demo - Troop Detection', 1280, 720)
            
            frame_count = 0
            
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_count += 1
                    
                    # Process frame
                    if self.process_frame(frame, frame_count):
                        break  # Stop processing
                    
            except KeyboardInterrupt:
                self.logger.info("Demo interrupted by user")
            finally:
                cap.release()
                cv2.destroyAllWindows()
                
                self.logger.info("=" * 50)
                self.logger.info("DEMO COMPLETED")
                self.logger.info("=" * 50)
                self.logger.info(f"Frames processed: {frame_count}")
                self.logger.info(f"Total troops detected: {self.total_troops_detected}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error running demo: {str(e)}")
            return False

def main():
    """Main entry point for demo"""
    if len(sys.argv) != 2:
        print("Usage: python demo.py <video_file_path>")
        print("Example: python demo.py test_video.mp4")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    try:
        demo = DemoMode()
        demo.run_demo(video_path)
        
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 