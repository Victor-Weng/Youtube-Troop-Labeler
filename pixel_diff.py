"""
Pixel difference detection for troop placement identification
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional, Dict
import config
from arena_color import ArenaColorDetector

class PixelDifferenceDetector:
    def __init__(self, arena_detector: ArenaColorDetector):
        self.logger = logging.getLogger(__name__)
        self.arena_detector = arena_detector
        self.previous_frame = None
        self.difference_threshold = config.pixel_diff_threshold
        self.min_cluster_size = config.min_cluster_size
        self.max_cluster_size = config.max_cluster_size
        
        # Morphological kernel for noise reduction
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
    def detect_changes(self, current_frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect pixel changes between current and previous frame
        """
        if self.previous_frame is None:
            self.previous_frame = current_frame.copy()
            return None
        
        # Convert frames to grayscale for difference calculation
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        previous_gray = cv2.cvtColor(self.previous_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate absolute difference
        diff = cv2.absdiff(current_gray, previous_gray)
        
        # Apply threshold to get binary difference mask
        _, binary_diff = cv2.threshold(diff, self.difference_threshold, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to reduce noise
        binary_diff = cv2.morphologyEx(binary_diff, cv2.MORPH_OPEN, self.kernel)
        binary_diff = cv2.morphologyEx(binary_diff, cv2.MORPH_CLOSE, self.kernel)
        
        # Update previous frame
        self.previous_frame = current_frame.copy()
        
        return binary_diff
    
    def detect_troop_placements(self, current_frame: np.ndarray, 
                               binary_diff: np.ndarray) -> List[Dict]:
        """
        Detect potential troop placements based on pixel differences and arena color
        """
        if binary_diff is None:
            return []
        
        # Get arena color mask
        arena_mask = self.arena_detector.get_arena_mask(current_frame)
        
        # Find contours in the difference mask
        contours, _ = cv2.findContours(binary_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        potential_placements = []
        
        for contour in contours:
            # Calculate contour area
            area = cv2.contourArea(contour)
            
            # Filter by size
            if area < self.min_cluster_size or area > self.max_cluster_size:
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check if the change represents a troop placement (arena color to non-arena)
            if self._is_troop_placement(current_frame, x, y, w, h, arena_mask):
                placement = {
                    'bbox': (x, y, w, h),
                    'center': (x + w // 2, y + h // 2),
                    'area': area,
                    'confidence': self._calculate_placement_confidence(area, w, h),
                    'contour': contour
                }
                potential_placements.append(placement)
        
        # Sort by confidence
        potential_placements.sort(key=lambda x: x['confidence'], reverse=True)
        
        return potential_placements
    
    def _is_troop_placement(self, frame: np.ndarray, x: int, y: int, w: int, h: int, 
                           arena_mask: np.ndarray) -> bool:
        """
        Determine if a detected change represents a troop placement
        """
        # Extract the region of interest
        roi = frame[y:y+h, x:x+w]
        roi_mask = arena_mask[y:y+h, x:x+w]
        
        # Calculate the percentage of arena-colored pixels in the ROI
        arena_pixels = np.sum(roi_mask)
        total_pixels = w * h
        arena_percentage = arena_pixels / total_pixels
        
        # If most pixels are arena-colored, this is likely a troop placement
        # (troop appears on arena background)
        return arena_percentage > 0.7
    
    def _calculate_placement_confidence(self, area: float, width: int, height: int) -> float:
        """
        Calculate confidence score for a potential placement
        """
        # Base confidence on area (larger changes are more likely to be troops)
        area_confidence = min(area / self.max_cluster_size, 1.0)
        
        # Aspect ratio confidence (troops are usually roughly square or rectangular)
        aspect_ratio = max(width, height) / max(min(width, height), 1)
        aspect_confidence = 1.0 / (1.0 + abs(aspect_ratio - 1.5))  # Prefer aspect ratio around 1.5
        
        # Combine confidences
        confidence = (area_confidence * 0.7 + aspect_confidence * 0.3)
        
        return confidence
    
    def filter_placements_by_arena_color(self, placements: List[Dict], 
                                       current_frame: np.ndarray) -> List[Dict]:
        """
        Filter placements to only include those that represent arena color changes
        """
        filtered_placements = []
        
        for placement in placements:
            x, y, w, h = placement['bbox']
            
            # Check if this region shows a change from arena color
            if self._has_arena_color_change(current_frame, x, y, w, h):
                filtered_placements.append(placement)
        
        return filtered_placements
    
    def _has_arena_color_change(self, frame: np.ndarray, x: int, y: int, w: int, h: int) -> bool:
        """
        Check if a region shows a change from arena color to non-arena color
        """
        if self.previous_frame is None:
            return False
        
        # Extract regions from both frames
        current_roi = frame[y:y+h, x:x+w]
        previous_roi = self.previous_frame[y:y+h, x:x+w]
        
        # Get arena color
        arena_color = self.arena_detector.get_arena_color()
        if arena_color is None:
            return False
        
        # Check for pixels that changed from arena color to something else
        arena_color_changes = 0
        total_pixels = 0
        
        for i in range(h):
            for j in range(w):
                current_pixel = current_roi[i, j]
                previous_pixel = previous_roi[i, j]
                
                # Check if previous pixel was arena color and current is not
                if self.arena_detector.is_arena_color(previous_pixel):
                    if not self.arena_detector.is_arena_color(current_pixel):
                        arena_color_changes += 1
                    total_pixels += 1
        
        # If significant portion shows arena color changes, consider it a placement
        if total_pixels > 0:
            change_ratio = arena_color_changes / total_pixels
            return change_ratio > 0.3  # At least 30% should show arena color changes
        
        return False
    
    def get_enhanced_difference_mask(self, current_frame: np.ndarray) -> np.ndarray:
        """
        Get an enhanced difference mask with additional processing
        """
        # Basic difference detection
        basic_diff = self.detect_changes(current_frame)
        if basic_diff is None:
            return np.zeros(current_frame.shape[:2], dtype=np.uint8)
        
        # Apply additional filtering
        enhanced_diff = basic_diff.copy()
        
        # Remove small noise
        enhanced_diff = cv2.medianBlur(enhanced_diff, 3)
        
        # Apply edge detection to focus on troop boundaries
        edges = cv2.Canny(enhanced_diff, 50, 150)
        
        # Combine difference and edges
        combined = cv2.bitwise_or(enhanced_diff, edges)
        
        return combined
    
    def visualize_differences(self, current_frame: np.ndarray, 
                            placements: List[Dict], 
                            binary_diff: np.ndarray) -> np.ndarray:
        """
        Create visualization of detected differences and placements
        """
        vis_frame = current_frame.copy()
        
        # Draw difference mask overlay
        if binary_diff is not None:
            # Create colored overlay
            diff_overlay = np.zeros_like(current_frame)
            diff_overlay[binary_diff > 0] = [0, 255, 255]  # Yellow for differences
            
            # Blend with original frame
            alpha = 0.3
            vis_frame = cv2.addWeighted(vis_frame, 1 - alpha, diff_overlay, alpha, 0)
        
        # Draw detected placements
        for placement in placements:
            x, y, w, h = placement['bbox']
            confidence = placement['confidence']
            
            # Color based on confidence
            if confidence > 0.8:
                color = (0, 255, 0)  # Green for high confidence
            elif confidence > 0.5:
                color = (0, 255, 255)  # Yellow for medium confidence
            else:
                color = (0, 0, 255)  # Red for low confidence
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw confidence score
            cv2.putText(vis_frame, f"{confidence:.2f}", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw center point
            center_x, center_y = placement['center']
            cv2.circle(vis_frame, (center_x, center_y), 3, color, -1)
        
        # Add statistics
        cv2.putText(vis_frame, f"Detected placements: {len(placements)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return vis_frame
    
    def set_arena_color(self, arena_color):
        """
        Set the arena color for detection
        """
        if hasattr(self.arena_detector, 'set_arena_color'):
            self.arena_detector.set_arena_color(arena_color)
        self.logger.info(f"Arena color set: {arena_color}")
    
    def set_pending_placement(self, placement_event):
        """
        Set a pending placement event to track
        """
        # Store placement event for future reference
        if not hasattr(self, 'pending_placements'):
            self.pending_placements = []
        self.pending_placements.append(placement_event)
        self.logger.debug(f"Pending placement set: {placement_event}")
    
    def detect_troops(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect troops in the current frame using pixel differences
        """
        # Detect changes
        binary_diff = self.detect_changes(frame)
        if binary_diff is None:
            return []
        
        # Detect troop placements
        placements = self.detect_troop_placements(frame, binary_diff)
        
        # Filter by arena color changes
        filtered_placements = self.filter_placements_by_arena_color(placements, frame)
        
        # Convert placements to troop format
        troops = []
        for i, placement in enumerate(filtered_placements):
            troop = {
                'id': i,  # Simple ID for tracking
                'bbox': placement['bbox'],
                'center': placement['center'],
                'confidence': placement['confidence'],
                'type': 'unknown',  # Could be enhanced with card recognition
                'timestamp': None,  # Could be enhanced with timing
                'placement_frame': 0  # Will be set by caller
            }
            troops.append(troop)
        
        return troops
    
    def reset(self):
        """
        Reset the detector state
        """
        self.previous_frame = None
        if hasattr(self, 'pending_placements'):
            self.pending_placements = []
        self.logger.info("Pixel difference detector reset") 