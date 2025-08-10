"""
Arena color detection and management for Clash Royale troop annotation
"""

import cv2
import numpy as np
import logging
from typing import Tuple, Optional, List
import config

class ArenaColorDetector:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.arena_color = None
        self.arena_color_tolerance = 20  # Color tolerance for filtering
        self.sample_coords = config.arena_color_sample_coords
        self.color_history = []  # Store color samples for stability
        self.max_history_size = 10
        
    def sample_arena_color(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Sample the average color from the arena area
        """
        try:
            x, y, w, h = self.sample_coords
            
            # Ensure coordinates are within frame bounds
            height, width = frame.shape[:2]
            x = max(0, min(x, width - w))
            y = max(0, min(y, height - h))
            w = min(w, width - x)
            h = min(h, height - y)
            
            # Extract the arena region
            arena_region = frame[y:y+h, x:x+w]
            
            if arena_region.size == 0:
                self.logger.warning("Arena region is empty, cannot sample color")
                return None
            
            # Calculate average color (BGR format)
            avg_color = np.mean(arena_region, axis=(0, 1))
            
            # Store in history
            self.color_history.append(avg_color)
            if len(self.color_history) > self.max_history_size:
                self.color_history.pop(0)
            
            # Calculate stable average from history
            stable_color = np.mean(self.color_history, axis=0)
            
            self.arena_color = stable_color.astype(np.uint8)
            
            self.logger.info(f"Sampled arena color: BGR({self.arena_color[0]}, {self.arena_color[1]}, {self.arena_color[2]})")
            return self.arena_color
            
        except Exception as e:
            self.logger.error(f"Error sampling arena color: {str(e)}")
            return None
    
    def get_arena_color(self) -> Optional[np.ndarray]:
        """
        Get the current arena color
        """
        return self.arena_color
    
    def is_arena_color(self, pixel_color: np.ndarray, tolerance: Optional[int] = None) -> bool:
        """
        Check if a pixel color matches the arena color within tolerance
        """
        if self.arena_color is None:
            return False
        
        if tolerance is None:
            tolerance = self.arena_color_tolerance
        
        # Calculate color distance
        color_diff = np.abs(pixel_color.astype(int) - self.arena_color.astype(int))
        max_diff = np.max(color_diff)
        
        return max_diff <= tolerance
    
    def is_arena_color_change(self, old_color: np.ndarray, new_color: np.ndarray) -> bool:
        """
        Check if color change represents arena color change
        """
        if self.arena_color is None:
            return False
        
        # Check if old color was arena color and new color is not
        old_is_arena = self.is_arena_color(old_color)
        new_is_arena = self.is_arena_color(new_color)
        
        return old_is_arena and not new_is_arena
    
    def get_arena_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Create a mask for arena-colored pixels
        """
        if self.arena_color is None:
            return np.ones(frame.shape[:2], dtype=bool)
        
        # Create mask where pixels are close to arena color using vectorized operations
        # Calculate color distance for all pixels at once
        frame_float = frame.astype(np.float32)
        arena_color_float = self.arena_color.astype(np.float32)
        
        # Calculate absolute difference for each color channel
        diff = np.abs(frame_float - arena_color_float)
        max_diff = np.max(diff, axis=2)
        
        # Create mask where max difference is within tolerance
        mask = max_diff <= self.arena_color_tolerance
        
        return mask
    
    def update_arena_color(self, frame: np.ndarray, force_update: bool = False) -> bool:
        """
        Update arena color if needed or forced
        """
        if force_update or self.arena_color is None:
            return self.sample_arena_color(frame) is not None
        
        # Check if current arena color is still valid
        if self.arena_color is not None:
            x, y, w, h = self.sample_coords
            height, width = frame.shape[:2]
            
            # Ensure coordinates are within bounds
            x = max(0, min(x, width - w))
            y = max(0, min(y, height - h))
            w = min(w, width - x)
            h = min(h, height - y)
            
            if w > 0 and h > 0:
                arena_region = frame[y:y+h, x:x+w]
                current_avg = np.mean(arena_region, axis=(0, 1))
                
                # If color has changed significantly, update
                color_diff = np.abs(current_avg - self.arena_color)
                if np.max(color_diff) > self.arena_color_tolerance * 2:
                    self.logger.info("Arena color has changed significantly, updating...")
                    return self.sample_arena_color(frame) is not None
        
        return True
    
    def get_color_distance(self, color1: np.ndarray, color2: np.ndarray) -> float:
        """
        Calculate Euclidean distance between two colors
        """
        return np.linalg.norm(color1.astype(float) - color2.astype(float))
    
    def get_dominant_colors(self, frame: np.ndarray, num_colors: int = 5) -> List[Tuple[np.ndarray, int]]:
        """
        Get dominant colors in the frame (useful for debugging)
        """
        try:
            # Reshape frame to 2D array of pixels
            pixels = frame.reshape(-1, 3)
            
            # Use k-means to find dominant colors
            from sklearn.cluster import KMeans
            
            kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            # Get colors and counts
            colors = kmeans.cluster_centers_.astype(np.uint8)
            labels = kmeans.labels_
            counts = np.bincount(labels)
            
            # Sort by count (most frequent first)
            sorted_indices = np.argsort(counts)[::-1]
            dominant_colors = [(colors[i], counts[i]) for i in sorted_indices]
            
            return dominant_colors
            
        except ImportError:
            self.logger.warning("sklearn not available, using simple color counting")
            # Fallback to simple color counting
            pixels = frame.reshape(-1, 3)
            unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
            
            # Sort by count
            sorted_indices = np.argsort(counts)[::-1]
            dominant_colors = [(unique_colors[i], counts[i]) for i in sorted_indices[:num_colors]]
            
            return dominant_colors
        except Exception as e:
            self.logger.error(f"Error getting dominant colors: {str(e)}")
            return []
    
    def visualize_arena_region(self, frame: np.ndarray) -> np.ndarray:
        """
        Create a visualization showing the arena sampling region
        """
        vis_frame = frame.copy()
        
        if self.arena_color is not None:
            x, y, w, h = self.sample_coords
            
            # Draw rectangle around sampling region
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw color sample
            color_sample = np.full((50, 50, 3), self.arena_color, dtype=np.uint8)
            
            # Place color sample in top-left corner
            vis_frame[10:60, 10:60] = color_sample
            cv2.rectangle(vis_frame, (10, 10), (60, 60), (255, 255, 255), 2)
            
            # Add text label
            cv2.putText(vis_frame, f"BGR({self.arena_color[0]},{self.arena_color[1]},{self.arena_color[2]})", 
                       (70, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return vis_frame 