"""
Output writer for YOLOv5 dataset generation
"""

import cv2
import os
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional
import config
from datetime import datetime

class OutputWriter:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.output_dir = config.output_dir
        self.images_dir = config.images_dir
        self.labels_dir = config.labels_dir
        self.classes_file = config.classes_file
        
        # Create output directories
        self._create_output_directories()
        
        # Class management
        self.class_mapping = {}  # class_name -> class_id
        self.class_counter = 0
        self.used_classes = set()
        
        # Frame management
        self.frame_counter = 0
        self.saved_frames = set()  # Track which frames have been saved
        
        # Initialize class mapping
        self._initialize_class_mapping()
    
    def _create_output_directories(self):
        """
        Create the necessary output directories
        """
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            os.makedirs(self.images_dir, exist_ok=True)
            os.makedirs(self.labels_dir, exist_ok=True)
            
            self.logger.info(f"Created output directories: {self.output_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to create output directories: {str(e)}")
            raise
    
    def _initialize_class_mapping(self):
        """
        Initialize the class mapping for YOLO format
        """
        # Start with common Clash Royale troops
        base_classes = [
            "ally_goblin", "ally_knight", "ally_archer", "ally_giant", "ally_wizard",
            "ally_prince", "ally_dragon", "ally_pekka", "ally_minion", "ally_hog_rider",
            "ally_valkyrie", "ally_musketeer", "ally_fireball", "ally_arrows", "ally_lightning",
            "enemy_goblin", "enemy_knight", "enemy_archer", "enemy_giant", "enemy_wizard",
            "enemy_prince", "enemy_dragon", "enemy_pekka", "enemy_minion", "enemy_hog_rider",
            "enemy_valkyrie", "enemy_musketeer", "enemy_fireball", "enemy_arrows", "enemy_lightning"
        ]
        
        for class_name in base_classes:
            self._add_class(class_name)
    
    def _add_class(self, class_name: str) -> int:
        """
        Add a new class to the mapping
        """
        if class_name not in self.class_mapping:
            self.class_mapping[class_name] = self.class_counter
            self.class_counter += 1
            self.used_classes.add(class_name)
            self.logger.debug(f"Added class: {class_name} -> {self.class_mapping[class_name]}")
        
        return self.class_mapping[class_name]
    
    def get_class_id(self, class_name: str) -> int:
        """
        Get the class ID for a given class name
        """
        if class_name not in self.class_mapping:
            return self._add_class(class_name)
        return self.class_mapping[class_name]
    
    def save_frame_with_labels(self, frame: np.ndarray, frame_number: int, 
                             troops: List[Dict], video_id: str = "unknown") -> bool:
        """
        Save a frame and its corresponding YOLO labels
        """
        try:
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            frame_filename = f"{video_id}_frame_{frame_number:06d}_{timestamp}.jpg"
            label_filename = f"{video_id}_frame_{frame_number:06d}_{timestamp}.txt"
            
            frame_path = os.path.join(self.images_dir, frame_filename)
            label_path = os.path.join(self.labels_dir, label_filename)
            
            # Save frame
            success = cv2.imwrite(frame_path, frame)
            if not success:
                self.logger.error(f"Failed to save frame: {frame_path}")
                return False
            
            # Generate and save YOLO labels
            yolo_labels = self._generate_yolo_labels(frame, troops)
            if yolo_labels:
                with open(label_path, 'w') as f:
                    for label in yolo_labels:
                        f.write(f"{label}\n")
                
                self.logger.debug(f"Saved frame and labels: {frame_filename}")
                self.frame_counter += 1
                self.saved_frames.add(frame_number)
                return True
            else:
                # No labels to save, remove the frame
                os.remove(frame_path)
                self.logger.debug(f"No labels for frame {frame_number}, removed frame")
                return False
                
        except Exception as e:
            self.logger.error(f"Error saving frame {frame_number}: {str(e)}")
            return False
    
    def _generate_yolo_labels(self, frame: np.ndarray, troops: List[Dict]) -> List[str]:
        """
        Generate YOLO format labels for detected troops
        """
        labels = []
        frame_height, frame_width = frame.shape[:2]
        
        for troop in troops:
            if troop['status'] != 'active':
                continue
            
            # Get class name from card info
            card_id = troop['card_info']['card_id']
            side = troop['card_info']['side']
            
            # Extract card name (remove ally_/enemy_ prefix and position suffix)
            card_parts = card_id.split('_')
            if len(card_parts) >= 2:
                card_name = card_parts[1]  # Get the actual card name
                class_name = f"{side}_{card_name}"
            else:
                class_name = f"{side}_{card_id}"
            
            # Get class ID
            class_id = self.get_class_id(class_name)
            
            # Get bounding box coordinates
            x, y, w, h = troop['bbox']
            
            # Convert to YOLO format (normalized center coordinates and dimensions)
            center_x = (x + w / 2) / frame_width
            center_y = (y + h / 2) / frame_height
            width = w / frame_width
            height = h / frame_height
            
            # Ensure coordinates are within [0, 1] range
            center_x = max(0.0, min(1.0, center_x))
            center_y = max(0.0, min(1.0, center_y))
            width = max(0.0, min(1.0, width))
            height = max(0.0, min(1.0, height))
            
            # Create YOLO label line: class_id center_x center_y width height
            label_line = f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"
            labels.append(label_line)
        
        return labels
    
    def save_classes_file(self):
        """
        Save the classes.txt file with all used classes
        """
        try:
            # Sort classes by ID for consistent ordering
            sorted_classes = sorted(self.class_mapping.items(), key=lambda x: x[1])
            
            with open(self.classes_file, 'w') as f:
                for class_name, class_id in sorted_classes:
                    f.write(f"{class_name}\n")
            
            self.logger.info(f"Saved classes file with {len(sorted_classes)} classes: {self.classes_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save classes file: {str(e)}")
    
    def save_dataset_info(self, video_id: str, total_frames: int, 
                         total_troops: int, processing_time: float):
        """
        Save dataset information and statistics
        """
        try:
            info_file = os.path.join(self.output_dir, "dataset_info.txt")
            
            with open(info_file, 'w') as f:
                f.write("Clash Royale Troop Annotation Dataset\n")
                f.write("=" * 40 + "\n\n")
                f.write(f"Video ID: {video_id}\n")
                f.write(f"Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Frames Processed: {total_frames}\n")
                f.write(f"Frames with Labels: {len(self.saved_frames)}\n")
                f.write(f"Total Troops Detected: {total_troops}\n")
                f.write(f"Processing Time: {processing_time:.2f} seconds\n")
                f.write(f"Classes Used: {len(self.used_classes)}\n\n")
                
                f.write("Class Mapping:\n")
                sorted_classes = sorted(self.class_mapping.items(), key=lambda x: x[1])
                for class_name, class_id in sorted_classes:
                    f.write(f"  {class_id}: {class_name}\n")
                
                f.write(f"\nOutput Directory: {self.output_dir}\n")
                f.write(f"Images Directory: {self.images_dir}\n")
                f.write(f"Labels Directory: {self.labels_dir}\n")
                f.write(f"Classes File: {self.classes_file}\n")
            
            self.logger.info(f"Saved dataset info: {info_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save dataset info: {str(e)}")
    
    def get_output_statistics(self) -> Dict:
        """
        Get statistics about the output
        """
        return {
            'total_frames_saved': self.frame_counter,
            'total_classes': len(self.class_mapping),
            'used_classes': len(self.used_classes),
            'output_directory': self.output_dir,
            'images_directory': self.images_dir,
            'labels_directory': self.labels_dir
        }
    
    def cleanup_empty_files(self):
        """
        Remove any empty label files
        """
        try:
            removed_count = 0
            
            for filename in os.listdir(self.labels_dir):
                if filename.endswith('.txt'):
                    file_path = os.path.join(self.labels_dir, filename)
                    
                    # Check if file is empty
                    if os.path.getsize(file_path) == 0:
                        os.remove(file_path)
                        removed_count += 1
                        
                        # Also remove corresponding image if it exists
                        image_filename = filename.replace('.txt', '.jpg')
                        image_path = os.path.join(self.images_dir, image_filename)
                        if os.path.exists(image_path):
                            os.remove(image_path)
            
            if removed_count > 0:
                self.logger.info(f"Cleaned up {removed_count} empty label files")
                
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
    
    def validate_dataset(self) -> Dict:
        """
        Validate the generated dataset
        """
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Check if directories exist
            if not os.path.exists(self.images_dir):
                validation_results['valid'] = False
                validation_results['errors'].append(f"Images directory does not exist: {self.images_dir}")
            
            if not os.path.exists(self.labels_dir):
                validation_results['valid'] = False
                validation_results['errors'].append(f"Labels directory does not exist: {self.labels_dir}")
            
            # Count files
            image_files = [f for f in os.listdir(self.images_dir) if f.endswith(('.jpg', '.png'))]
            label_files = [f for f in os.listdir(self.labels_dir) if f.endswith('.txt')]
            
            if len(image_files) == 0:
                validation_results['warnings'].append("No image files found")
            
            if len(label_files) == 0:
                validation_results['warnings'].append("No label files found")
            
            # Check for matching files
            image_bases = {os.path.splitext(f)[0] for f in image_files}
            label_bases = {os.path.splitext(f)[0] for f in label_files}
            
            missing_labels = image_bases - label_bases
            missing_images = label_bases - image_bases
            
            if missing_labels:
                validation_results['warnings'].append(f"Images without labels: {len(missing_labels)}")
            
            if missing_images:
                validation_results['warnings'].append(f"Labels without images: {len(missing_images)}")
            
            # Validate YOLO format
            for label_file in label_files[:10]:  # Check first 10 files
                label_path = os.path.join(self.labels_dir, label_file)
                try:
                    with open(label_path, 'r') as f:
                        for line_num, line in enumerate(f, 1):
                            line = line.strip()
                            if line:
                                parts = line.split()
                                if len(parts) != 5:
                                    validation_results['errors'].append(
                                        f"Invalid YOLO format in {label_file}:{line_num}: {line}")
                                    validation_results['valid'] = False
                                else:
                                    # Check if values are valid floats
                                    try:
                                        values = [float(part) for part in parts]
                                        if not all(0 <= val <= 1 for val in values[1:]):
                                            validation_results['warnings'].append(
                                                f"Coordinate out of range in {label_file}:{line_num}: {line}")
                                    except ValueError:
                                        validation_results['errors'].append(
                                            f"Non-numeric values in {label_file}:{line_num}: {line}")
                                        validation_results['valid'] = False
                except Exception as e:
                    validation_results['errors'].append(f"Error reading {label_file}: {str(e)}")
                    validation_results['valid'] = False
            
        except Exception as e:
            validation_results['valid'] = False
            validation_results['errors'].append(f"Validation error: {str(e)}")
        
        return validation_results
    
    def reset(self):
        """
        Reset the output writer state
        """
        self.frame_counter = 0
        self.saved_frames.clear()
        self.logger.info("Output writer reset") 