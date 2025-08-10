"""
Simple output writer for YOLO dataset generation
"""

import cv2
import os
import logging
import numpy as np
from typing import List, Dict
import config_new as config

class OutputWriter:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.frame_count = 0
        self.object_count = 0
        
        # Create output directories only if saving is enabled
        if config.SAVE_IMAGES:
            os.makedirs(config.OUTPUT_DIR, exist_ok=True)
            os.makedirs(config.IMAGES_DIR, exist_ok=True)
            os.makedirs(config.LABELS_DIR, exist_ok=True)
            self.logger.info("Output directories created for image saving")
        else:
            self.logger.info("Image saving disabled - skipping directory creation")
        
        # Create classes file only if saving is enabled
        if config.SAVE_IMAGES:
            self._create_classes_file()
    
    def _create_classes_file(self):
        """Create classes.txt file for YOLO"""
        classes_path = os.path.join(config.OUTPUT_DIR, 'classes.txt')
        with open(classes_path, 'w') as f:
            f.write('troop\n')  # Single class for all troops
        self.logger.info("Created classes.txt file")
    
    def save_frame_with_objects(self, frame: np.ndarray, objects: List[Dict], video_id: str = "video"):
        """Save frame and labels if objects are present"""
        if not objects:
            return
        
        self.frame_count += 1
        
        # Only save files if image saving is enabled
        if config.SAVE_IMAGES:
            # Generate filename
            filename = f"{video_id}_frame_{self.frame_count:06d}"
            image_path = os.path.join(config.IMAGES_DIR, f"{filename}.jpg")
            label_path = os.path.join(config.LABELS_DIR, f"{filename}.txt")
            
            # Save image
            cv2.imwrite(image_path, frame)
            
            # Save YOLO format labels
            height, width = frame.shape[:2]
            with open(label_path, 'w') as f:
                for obj in objects:
                    x, y, w, h = obj['bbox']
                    
                    # Convert to YOLO format (normalized center coordinates)
                    center_x = (x + w / 2) / width
                    center_y = (y + h / 2) / height
                    norm_width = w / width
                    norm_height = h / height
                    
                    # YOLO format: class_id center_x center_y width height
                    f.write(f"0 {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}\n")
        
        self.object_count += len(objects)
        
        if self.frame_count % 10 == 0:
            save_status = "saved" if config.SAVE_IMAGES else "processed (not saved)"
            self.logger.info(f"{save_status.capitalize()} {self.frame_count} frames with {self.object_count} total objects (this frame: {len(objects)} objects)")
    
    def save_summary(self):
        """Save processing summary"""
        if config.SAVE_IMAGES:
            summary_path = os.path.join(config.OUTPUT_DIR, 'summary.txt')
            with open(summary_path, 'w') as f:
                f.write(f"Processing Summary\n")
                f.write(f"=================\n")
                f.write(f"Total frames saved: {self.frame_count}\n")
                f.write(f"Total objects detected: {self.object_count}\n")
                f.write(f"Average objects per frame: {self.object_count / max(self.frame_count, 1):.2f}\n")
        
        action = "saved" if config.SAVE_IMAGES else "processed"
        self.logger.info(f"Processing complete: {self.frame_count} frames {action}, {self.object_count} objects detected")
    
    def get_stats(self) -> Dict:
        """Get current statistics"""
        return {
            'frames_saved': self.frame_count,
            'objects_detected': self.object_count,
            'avg_objects_per_frame': self.object_count / max(self.frame_count, 1)
        }
