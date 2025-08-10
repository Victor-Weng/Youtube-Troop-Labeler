"""
Troop tracking module using OpenCV tracking algorithms
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
import config

class TroopTracker:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.trackers = {}  # troop_id -> tracker object
        self.tracking_history = {}  # troop_id -> list of tracking points
        self.max_history_length = 100
        self.tracker_type = config.tracker_type
        self.tracking_confidence_threshold = config.tracking_confidence_threshold
        
        # Initialize tracker factory
        self.tracker_factory = self._create_tracker_factory()
        
    def _create_tracker_factory(self):
        """
        Create a factory function for the specified tracker type
        """
        try:
            if self.tracker_type == 'CSRT':
                return cv2.TrackerCSRT_create
            elif self.tracker_type == 'KCF':
                return cv2.TrackerKCF_create
            elif self.tracker_type == 'MOSSE':
                return cv2.TrackerMOSSE_create
            else:
                self.logger.warning(f"Unknown tracker type: {self.tracker_type}, trying available trackers")
        except AttributeError:
            pass
        
        # Try available trackers in order of preference
        try:
            return cv2.TrackerDaSiamRPN_create
        except AttributeError:
            try:
                return cv2.TrackerGOTURN_create
            except AttributeError:
                try:
                    return cv2.TrackerMIL_create
                except AttributeError:
                    try:
                        return cv2.TrackerNano_create
                    except AttributeError:
                        try:
                            return cv2.TrackerVit_create
                        except AttributeError:
                            self.logger.error("No available trackers found")
                            return None
    
    def add_troop_for_tracking(self, troop: Dict):
        """
        Add a new troop for tracking
        """
        troop_id = troop['id']
        
        if troop_id in self.trackers:
            self.logger.warning(f"Troop {troop_id} is already being tracked")
            return False
        
        try:
            # Create new tracker
            if self.tracker_factory is None:
                self.logger.error("No tracker factory available")
                return False
                
            tracker = self.tracker_factory()
            
            # Get bounding box coordinates
            x, y, w, h = troop['bbox']
            bbox = (x, y, w, h)
            
            # Initialize tracker with the frame and bbox
            # Note: We need the frame to initialize the tracker
            # This will be done when the first frame is available
            
            # Store tracker and initialize history
            self.trackers[troop_id] = {
                'tracker': tracker,
                'bbox': bbox,
                'initialized': False,
                'last_update_frame': troop['placement_frame'],
                'tracking_failures': 0
            }
            
            self.tracking_history[troop_id] = [{
                'frame': troop['placement_frame'],
                'bbox': bbox,
                'center': troop['center']
            }]
            
            self.logger.info(f"Added troop {troop_id} for tracking")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add troop {troop_id} for tracking: {str(e)}")
            return False
    
    def initialize_tracker(self, troop_id: int, frame: np.ndarray):
        """
        Initialize a tracker with the first frame
        """
        if troop_id not in self.trackers:
            return False
        
        tracker_info = self.trackers[troop_id]
        if tracker_info['initialized']:
            return True
        
        try:
            tracker = tracker_info['tracker']
            bbox = tracker_info['bbox']
            
            # Initialize tracker
            success = tracker.init(frame, bbox)
            
            if success:
                tracker_info['initialized'] = True
                self.logger.info(f"Tracker initialized for troop {troop_id}")
                return True
            else:
                self.logger.warning(f"Failed to initialize tracker for troop {troop_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error initializing tracker for troop {troop_id}: {str(e)}")
            return False
    
    def update_tracking(self, frame: np.ndarray, frame_number: int) -> Dict[int, Dict]:
        """
        Update all active trackers with the current frame
        """
        tracking_results = {}
        
        for troop_id in list(self.trackers.keys()):
            tracker_info = self.trackers[troop_id]
            
            if not tracker_info['initialized']:
                # Try to initialize if not already done
                if not self.initialize_tracker(troop_id, frame):
                    continue
            
            try:
                # Update tracker
                success, bbox = tracker_info['tracker'].update(frame)
                
                if success and bbox is not None:
                    # Tracking successful
                    x, y, w, h = [int(coord) for coord in bbox]
                    
                    # Validate bbox coordinates
                    if self._is_valid_bbox(x, y, w, h, frame.shape):
                        # Update tracker info
                        tracker_info['bbox'] = (x, y, w, h)
                        tracker_info['last_update_frame'] = frame_number
                        tracker_info['tracking_failures'] = 0
                        
                        # Add to tracking history
                        center = (x + w // 2, y + h // 2)
                        self._add_tracking_point(troop_id, frame_number, (x, y, w, h), center)
                        
                        # Store result
                        tracking_results[troop_id] = {
                            'bbox': (x, y, w, h),
                            'center': center,
                            'confidence': 1.0 - (tracker_info['tracking_failures'] * 0.1)
                        }
                    else:
                        # Invalid bbox, mark as failure
                        tracker_info['tracking_failures'] += 1
                        self.logger.warning(f"Invalid bbox for troop {troop_id}: {bbox}")
                else:
                    # Tracking failed
                    tracker_info['tracking_failures'] += 1
                    self.logger.warning(f"Tracking failed for troop {troop_id}")
                    
                    # Check if we should remove this tracker
                    if tracker_info['tracking_failures'] > 5:
                        self.logger.info(f"Removing failed tracker for troop {troop_id}")
                        self.remove_troop_tracking(troop_id)
                        
            except Exception as e:
                self.logger.error(f"Error updating tracker for troop {troop_id}: {str(e)}")
                tracker_info['tracking_failures'] += 1
        
        return tracking_results
    
    def _is_valid_bbox(self, x: int, y: int, w: int, h: int, frame_shape: Tuple) -> bool:
        """
        Check if bounding box coordinates are valid
        """
        if w <= 0 or h <= 0:
            return False
        
        if x < 0 or y < 0:
            return False
        
        if x + w > frame_shape[1] or y + h > frame_shape[0]:
            return False
        
        # Check reasonable size limits
        if w < 10 or h < 10 or w > 500 or h > 500:
            return False
        
        return True
    
    def _add_tracking_point(self, troop_id: int, frame_number: int, 
                           bbox: Tuple[int, int, int, int], center: Tuple[int, int]):
        """
        Add a tracking point to the history
        """
        if troop_id not in self.tracking_history:
            self.tracking_history[troop_id] = []
        
        history = self.tracking_history[troop_id]
        history.append({
            'frame': frame_number,
            'bbox': bbox,
            'center': center
        })
        
        # Limit history length
        if len(history) > self.max_history_length:
            history.pop(0)
    
    def get_troop_trajectory(self, troop_id: int) -> List[Dict]:
        """
        Get the tracking trajectory for a specific troop
        """
        return self.tracking_history.get(troop_id, [])
    
    def get_troop_current_position(self, troop_id: int) -> Optional[Dict]:
        """
        Get the current position of a tracked troop
        """
        if troop_id not in self.trackers:
            return None
        
        tracker_info = self.trackers[troop_id]
        if not tracker_info['initialized']:
            return None
        
        return {
            'bbox': tracker_info['bbox'],
            'center': (tracker_info['bbox'][0] + tracker_info['bbox'][2] // 2,
                      tracker_info['bbox'][1] + tracker_info['bbox'][3] // 2),
            'last_update_frame': tracker_info['last_update_frame'],
            'tracking_failures': tracker_info['tracking_failures']
        }
    
    def remove_troop_tracking(self, troop_id: int):
        """
        Remove a troop from tracking
        """
        if troop_id in self.trackers:
            del self.trackers[troop_id]
            self.logger.info(f"Removed troop {troop_id} from tracking")
        
        if troop_id in self.tracking_history:
            del self.tracking_history[troop_id]
    
    def get_tracking_statistics(self) -> Dict:
        """
        Get statistics about current tracking
        """
        active_trackers = len(self.trackers)
        total_history_points = sum(len(history) for history in self.tracking_history.values())
        
        # Count trackers by status
        initialized_count = sum(1 for info in self.trackers.values() if info['initialized'])
        failed_count = sum(1 for info in self.trackers.values() 
                          if info['tracking_failures'] > 0)
        
        return {
            'active_trackers': active_trackers,
            'initialized_trackers': initialized_count,
            'failed_trackers': failed_count,
            'total_history_points': total_history_points,
            'tracker_type': self.tracker_type
        }
    
    def visualize_tracking(self, frame: np.ndarray) -> np.ndarray:
        """
        Create visualization of tracking results
        """
        vis_frame = frame.copy()
        
        # Draw tracking results for each troop
        for troop_id, tracker_info in self.trackers.items():
            if not tracker_info['initialized']:
                continue
            
            bbox = tracker_info['bbox']
            x, y, w, h = bbox
            center = (x + w // 2, y + h // 2)
            
            # Color based on tracking failures
            failures = tracker_info['tracking_failures']
            if failures == 0:
                color = (0, 255, 0)  # Green for good tracking
            elif failures <= 2:
                color = (0, 255, 255)  # Yellow for some failures
            else:
                color = (0, 0, 255)  # Red for many failures
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw troop ID
            cv2.putText(vis_frame, f"ID:{troop_id}", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw center point
            cv2.circle(vis_frame, center, 3, color, -1)
            
            # Draw trajectory if available
            if troop_id in self.tracking_history:
                trajectory = self.tracking_history[troop_id]
                if len(trajectory) > 1:
                    points = [point['center'] for point in trajectory[-20:]]  # Last 20 points
                    for i in range(1, len(points)):
                        cv2.line(vis_frame, points[i-1], points[i], color, 1)
        
        # Add statistics
        stats = self.get_tracking_statistics()
        y_offset = 30
        cv2.putText(vis_frame, f"Active trackers: {stats['active_trackers']}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
        cv2.putText(vis_frame, f"Tracker type: {stats['tracker_type']}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return vis_frame
    
    def reset(self):
        """
        Reset all tracking state
        """
        self.trackers.clear()
        self.tracking_history.clear()
        self.logger.info("Troop tracker reset")
    
    def cleanup_old_trackers(self, current_frame: int, max_age_frames: int = 300):
        """
        Remove trackers that haven't been updated recently
        """
        troops_to_remove = []
        
        for troop_id, tracker_info in self.trackers.items():
            frames_since_update = current_frame - tracker_info['last_update_frame']
            if frames_since_update > max_age_frames:
                troops_to_remove.append(troop_id)
        
        for troop_id in troops_to_remove:
            self.remove_troop_tracking(troop_id)
            self.logger.info(f"Removed old tracker for troop {troop_id}") 