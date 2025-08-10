"""
Troop detection module that combines card disappearance and pixel difference analysis
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
import config
from card_tracker import CardTracker
from pixel_diff import PixelDifferenceDetector
from arena_color import ArenaColorDetector

class TroopDetector:
    def __init__(self, card_tracker: CardTracker, pixel_detector: PixelDifferenceDetector, 
                 arena_detector: ArenaColorDetector):
        self.logger = logging.getLogger(__name__)
        self.card_tracker = card_tracker
        self.pixel_detector = pixel_detector
        self.arena_detector = arena_detector
        
        # Detection state
        self.detected_troops = []  # List of active troop detections
        self.troop_id_counter = 0
        self.min_placement_confidence = 0.6
        
        # Timing parameters
        self.placement_timeout_frames = 15  # Frames to wait for placement after card disappears
        self.max_troop_lifetime = 300  # Maximum frames to track a troop
        
    def process_frame(self, frame: np.ndarray, frame_number: int) -> List[Dict]:
        """
        Main method to process a frame and detect troop placements
        """
        # Update hand states and detect card disappearances
        ally_cards, enemy_cards = self.card_tracker.update_hand_states(frame)
        
        # Get recently disappeared cards
        disappeared_ally, disappeared_enemy = self.card_tracker.get_recent_disappearances()
        
        # Detect pixel differences
        binary_diff = self.pixel_detector.detect_changes(frame)
        
        # Detect potential placements
        potential_placements = self.pixel_detector.detect_troop_placements(frame, binary_diff)
        
        # Filter placements by arena color changes
        filtered_placements = self.pixel_detector.filter_placements_by_arena_color(
            potential_placements, frame)
        
        # Match placements with disappeared cards
        matched_placements = self._match_placements_with_cards(
            filtered_placements, disappeared_ally, disappeared_enemy, frame_number)
        
        # Update existing troop detections
        self._update_troop_detections(frame, frame_number)
        
        # Create new troop detections from matched placements
        new_troops = self._create_troop_detections(matched_placements, frame_number)
        
        # Add new troops to the list
        self.detected_troops.extend(new_troops)
        
        # Clean up old troops
        self._cleanup_old_troops(frame_number)
        
        return self.detected_troops
    
    def _match_placements_with_cards(self, placements: List[Dict], 
                                   disappeared_ally: Dict, 
                                   disappeared_enemy: Dict, 
                                   frame_number: int) -> List[Dict]:
        """
        Match detected placements with recently disappeared cards
        """
        matched_placements = []
        
        for placement in placements:
            # Find the best matching disappeared card
            best_match = self._find_best_card_match(placement, disappeared_ally, disappeared_enemy)
            
            if best_match:
                placement['matched_card'] = best_match
                placement['placement_frame'] = frame_number
                matched_placements.append(placement)
        
        return matched_placements
    
    def _find_best_card_match(self, placement: Dict, 
                             disappeared_ally: Dict, 
                             disappeared_enemy: Dict) -> Optional[Dict]:
        """
        Find the best matching disappeared card for a placement
        """
        best_match = None
        best_score = 0
        
        # Check ally disappearances
        for card_id, disappear_frame in disappeared_ally.items():
            score = self._calculate_card_placement_score(placement, card_id, 'ally')
            if score > best_score:
                best_score = score
                best_match = {
                    'card_id': card_id,
                    'side': 'ally',
                    'disappear_frame': disappear_frame,
                    'score': score
                }
        
        # Check enemy disappearances
        for card_id, disappear_frame in disappeared_enemy.items():
            score = self._calculate_card_placement_score(placement, card_id, 'enemy')
            if score > best_score:
                best_score = score
                best_match = {
                    'card_id': card_id,
                    'side': 'enemy',
                    'disappear_frame': disappear_frame,
                    'score': score
                }
        
        # Only return match if score is above threshold
        if best_score > self.min_placement_confidence:
            return best_match
        
        return None
    
    def _calculate_card_placement_score(self, placement: Dict, card_id: str, side: str) -> float:
        """
        Calculate a score for how well a placement matches a disappeared card
        """
        score = 0.0
        
        # Base score from placement confidence
        score += placement['confidence'] * 0.4
        
        # Position-based scoring (ally cards usually placed in lower half, enemy in upper)
        center_x, center_y = placement['center']
        frame_height = 720  # Assuming standard resolution
        
        if side == 'ally':
            # Ally cards typically placed in lower half
            if center_y > frame_height // 2:
                score += 0.3
            else:
                score += 0.1
        else:  # enemy
            # Enemy cards typically placed in upper half
            if center_y < frame_height // 2:
                score += 0.3
            else:
                score += 0.1
        
        # Size-based scoring (different cards have different sizes)
        area = placement['area']
        if 100 < area < 5000:  # Typical troop size range
            score += 0.2
        elif 5000 < area < 15000:  # Larger troops
            score += 0.15
        else:
            score += 0.05
        
        # Timing-based scoring (recent disappearances get higher scores)
        # This would require frame timing information
        
        return min(score, 1.0)
    
    def _create_troop_detections(self, matched_placements: List[Dict], 
                                frame_number: int) -> List[Dict]:
        """
        Create new troop detection objects from matched placements
        """
        new_troops = []
        
        for placement in matched_placements:
            if placement['confidence'] >= self.min_placement_confidence:
                troop = {
                    'id': self.troop_id_counter,
                    'bbox': placement['bbox'],
                    'center': placement['center'],
                    'area': placement['area'],
                    'confidence': placement['confidence'],
                    'card_info': placement['matched_card'],
                    'placement_frame': placement['placement_frame'],
                    'current_frame': frame_number,
                    'status': 'active',
                    'tracking_history': [{
                        'frame': frame_number,
                        'bbox': placement['bbox'],
                        'center': placement['center']
                    }]
                }
                
                new_troops.append(troop)
                self.troop_id_counter += 1
                
                self.logger.info(f"New troop detected: {troop['card_info']['card_id']} "
                               f"at {troop['center']} with confidence {troop['confidence']:.2f}")
        
        return new_troops
    
    def _update_troop_detections(self, frame: np.ndarray, frame_number: int):
        """
        Update existing troop detections
        """
        for troop in self.detected_troops:
            if troop['status'] == 'active':
                # Update frame count
                troop['current_frame'] = frame_number
                
                # Check if troop should be marked as inactive
                frames_since_placement = frame_number - troop['placement_frame']
                if frames_since_placement > self.max_troop_lifetime:
                    troop['status'] = 'inactive'
                    self.logger.info(f"Troop {troop['id']} marked as inactive")
    
    def _cleanup_old_troops(self, current_frame: int):
        """
        Remove old troop detections
        """
        active_troops = []
        
        for troop in self.detected_troops:
            # Keep troops that are still active or recently placed
            if (troop['status'] == 'active' or 
                current_frame - troop['placement_frame'] <= self.max_troop_lifetime):
                active_troops.append(troop)
        
        self.detected_troops = active_troops
    
    def get_active_troops(self) -> List[Dict]:
        """
        Get list of currently active troops
        """
        return [troop for troop in self.detected_troops if troop['status'] == 'active']
    
    def get_troop_by_id(self, troop_id: int) -> Optional[Dict]:
        """
        Get a specific troop by ID
        """
        for troop in self.detected_troops:
            if troop['id'] == troop_id:
                return troop
        return None
    
    def mark_troop_inactive(self, troop_id: int):
        """
        Mark a troop as inactive
        """
        troop = self.get_troop_by_id(troop_id)
        if troop:
            troop['status'] = 'inactive'
            self.logger.info(f"Troop {troop_id} marked as inactive")
    
    def get_detection_statistics(self) -> Dict:
        """
        Get statistics about troop detections
        """
        active_count = len(self.get_active_troops())
        total_count = len(self.detected_troops)
        
        # Count by side
        ally_count = sum(1 for troop in self.detected_troops 
                        if troop['card_info']['side'] == 'ally')
        enemy_count = sum(1 for troop in self.detected_troops 
                         if troop['card_info']['side'] == 'enemy')
        
        return {
            'active_troops': active_count,
            'total_troops_detected': total_count,
            'ally_troops': ally_count,
            'enemy_troops': enemy_count,
            'next_troop_id': self.troop_id_counter
        }
    
    def visualize_detections(self, frame: np.ndarray) -> np.ndarray:
        """
        Create visualization of detected troops
        """
        vis_frame = frame.copy()
        
        # Draw all detected troops
        for troop in self.detected_troops:
            x, y, w, h = troop['bbox']
            center_x, center_y = troop['center']
            confidence = troop['confidence']
            side = troop['card_info']['side']
            card_id = troop['card_info']['card_id']
            
            # Color based on side and status
            if troop['status'] == 'active':
                if side == 'ally':
                    color = (0, 255, 0)  # Green for active ally
                else:
                    color = (0, 0, 255)  # Red for active enemy
            else:
                color = (128, 128, 128)  # Gray for inactive
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw troop ID and card info
            label = f"ID:{troop['id']} {card_id}"
            cv2.putText(vis_frame, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw confidence score
            cv2.putText(vis_frame, f"{confidence:.2f}", (x, y+h+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Draw center point
            cv2.circle(vis_frame, (center_x, center_y), 3, color, -1)
        
        # Add statistics
        stats = self.get_detection_statistics()
        y_offset = 30
        cv2.putText(vis_frame, f"Active troops: {stats['active_troops']}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
        cv2.putText(vis_frame, f"Total detected: {stats['total_troops_detected']}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return vis_frame
    
    def reset(self):
        """
        Reset the detector state
        """
        self.detected_troops = []
        self.troop_id_counter = 0
        self.pixel_detector.reset()
        self.logger.info("Troop detector reset") 