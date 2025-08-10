"""
Simple troop detector using motion detection and tracking with card tracking
"""

import cv2
import numpy as np
import logging
import os
import json
from typing import List, Dict, Tuple, Optional
import config_new as config
from inference_sdk import InferenceHTTPClient

class TroopDetector:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Background subtraction for motion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=False, 
            varThreshold=config.PIXEL_DIFF_THRESHOLD
        )
        
        # Tracking
        self.next_id = 0
        self.tracked_objects = {}  # id -> object info
        self.logger.info("Using detection-based tracking (industry standard for dynamic scenes)")
        
        # Arena background color for filtering
        self.arena_color = None
        self.arena_tolerance = 30
        
        # Card tracking (copied from original card_tracker.py)
        self.rf_client = None
        self.workspace_name = None
        self.card_names = self._load_card_names()
        self.ally_hand_coords = config.ALLY_HAND_COORDS
        self.enemy_hand_coords = config.ENEMY_HAND_COORDS
        self.frame_count = 0
        self.ally_hand_state = {}
        self.enemy_hand_state = {}
        self.recently_disappeared_ally = {}
        self.recently_disappeared_enemy = {}
        self.disappearance_memory_frames = 30
        
        # Setup Roboflow client
        try:
            self.rf_client = self.setup_roboflow()
            self.workspace_name = os.getenv('WORKSPACE_CARD_DETECTION')
            if not self.workspace_name:
                self.logger.warning("WORKSPACE_CARD_DETECTION not set; Roboflow calls may fail.")
            self.logger.info("Roboflow client initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to setup Roboflow client: {str(e)}")
            self.rf_client = None
            self.workspace_name = None
    
    # ========================
    # CARD TRACKING METHODS (copied exactly from original card_tracker.py)
    # ========================
    
    def setup_roboflow(self):
        api_key = os.getenv('ROBOFLOW_API_KEY')
        if not api_key:
            raise ValueError("ROBOFLOW_API_KEY environment variable is not set. Please check your .env file.")
        
        return InferenceHTTPClient(
            api_url="http://localhost:9001",
            api_key=api_key
        )
    
    def _load_card_names(self) -> Dict[int, str]:
        try:
            json_path = os.path.join(os.path.dirname(__file__), 'models', 'cards', 'card_names.json')
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    card_data = json.load(f)
                    return {int(k): v for k, v in card_data.items()}
            else:
                self.logger.warning(f"Card names JSON file not found at {json_path}")
                return self._get_default_card_names()
        except Exception as e:
            self.logger.error(f"Error loading card names from JSON: {e}")
            return self._get_default_card_names()

    def _get_default_card_names(self) -> Dict[int, str]:
        return {
            0: "archer queen", 1: "archers", 2: "arrows", 3: "baby dragon", 4: "balloon",
            5: "bandit", 6: "barbarian barrel", 7: "barbarian hut", 8: "barbarians", 9: "bats"
        }
    
    def detect_hand_cards(self, frame: np.ndarray, which: str = "ally") -> Dict[int, Dict]:
        """
        Unified call: detect cards in all configured regions for the given hand.
        Returns a mapping: slot_index -> card_dict
        card_dict: {
          'class_id': int,
          'card_name': str,
          'bbox': (x, y, w, h),
          'confidence': float,
          'center': (cx, cy)
        }
        which: "ally" or "enemy"
        """
        assert which in ("ally", "enemy"), "which must be 'ally' or 'enemy'"
        coords_list = self.ally_hand_coords if which == "ally" else self.enemy_hand_coords
        results: Dict[int, Dict] = {}

        if frame is None or frame.size == 0:
            self.logger.debug("Empty frame passed to detect_hand_cards")
            return results

        # For each hand slot region, call Roboflow (or fallback) and return first/top detection
        for i, coords in enumerate(coords_list):
            x, y, w, h = coords
            # bounds check
            fh, fw = frame.shape[:2]
            if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > fw or y + h > fh:
                self.logger.warning(f"Invalid region for {which} slot {i}: {coords}")
                continue

            region = frame[y:y+h, x:x+w]
            if region.size == 0:
                self.logger.debug(f"Empty region for {which} slot {i}")
                continue

            detection = self._run_roboflow_on_region(region, offset=(x, y))
            # detection is a list of detections in that region; pick the highest-confidence one (or None)
            if detection:
                # take top prediction
                top = max(detection, key=lambda d: d.get('confidence', 0.0))
                results[i] = top
            else:
                # no detection -> unknown slot
                results[i] = {
                    'class_id': -1,
                    'card_name': 'Unknown',
                    'bbox': (x, y, w, h),
                    'confidence': 0.0,
                    'center': (x + w // 2, y + h // 2)
                }

        return results
    
    def _run_roboflow_on_region(self, region: np.ndarray, offset: Tuple[int, int]) -> List[Dict]:
        """
        Runs Roboflow workflow on a region (bytes) and returns parsed detection dicts
        Offset is (x0, y0) to convert relative bbox -> absolute coords.
        """
        x_off, y_off = offset
        try:
            if self.rf_client and self.workspace_name:
                # Convert BGR->RGB bytes
                region_rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
                region_bytes = cv2.imencode('.jpg', region_rgb)[1].tobytes()

                # run_workflow expected return structure may vary; wrap defensively
                results = self.rf_client.run_workflow(
                    workspace_name=self.workspace_name,
                    workflow_id="custom-workflow",
                    images={"image": region_bytes}
                )

                return self._parse_roboflow_results(results, offset)
            else:
                # No workspace configured - return empty results
                self.logger.warning("No Roboflow workspace configured")
                return []
        except Exception as e:
            self.logger.error(f"Roboflow region inference failed: {e}")
            return []

    def _parse_roboflow_results(self, results, offset: Tuple[int, int]) -> List[Dict]:
        cards = []
        try:
            predictions = []
            # handle several possible response shapes
            if isinstance(results, list) and results:
                preds_dict = results[0].get("predictions", {})
                if isinstance(preds_dict, dict):
                    predictions = preds_dict.get("predictions", [])
            elif isinstance(results, dict):
                # sometimes it's returned straight as dict
                predictions = results.get("predictions", []) or []
            # parse
            for pred in predictions:
                card_name = pred.get("class", "unknown")
                confidence = float(pred.get("confidence", 0.0))
                bbox = pred.get("bbox", {})
                # Roboflow bbox format may be absolute or relative; we assume absolute relative to region:
                # support both style: 'x','y','width','height'
                bx = int(bbox.get("x", 0))
                by = int(bbox.get("y", 0))
                bw = int(bbox.get("width", bbox.get("w", 0)))
                bh = int(bbox.get("height", bbox.get("h", 0)))
                abs_x1 = offset[0] + bx
                abs_y1 = offset[1] + by
                abs_w = bw
                abs_h = bh
                abs_cx = abs_x1 + abs_w // 2
                abs_cy = abs_y1 + abs_h // 2
                class_id = self._get_class_id_from_name(card_name)
                cards.append({
                    'class_id': class_id,
                    'card_name': card_name,
                    'bbox': (abs_x1, abs_y1, abs_w, abs_h),
                    'confidence': confidence,
                    'center': (abs_cx, abs_cy)
                })
            return cards
        except Exception as e:
            self.logger.error(f"Error parsing Roboflow results: {e}")
            return []

    def _get_class_id_from_name(self, card_name: str) -> int:
        name_to_id = {name: id for id, name in self.card_names.items()}
        if card_name in name_to_id:
            return name_to_id[card_name]
        cl = card_name.lower()
        for name, id in name_to_id.items():
            if name.lower() == cl:
                return id
        return -1

    def update_hand_states(self, frame: np.ndarray) -> Tuple[Dict, Dict]:
        """
        Update the current state of both hands and detect card disappearances
        """
        self.frame_count += 1

        ally_cards = {}
        enemy_cards = {}

        # Unified detection per hand (returns slot-indexed dicts)
        ally_detections = self.detect_hand_cards(frame, which="ally")
        for slot_idx, card in ally_detections.items():
            card_id = f"ally_{card['card_name']}_{slot_idx}"
            ally_cards[card_id] = card
            self.ally_hand_state[card_id] = self.frame_count

        enemy_detections = self.detect_hand_cards(frame, which="enemy")
        for slot_idx, card in enemy_detections.items():
            card_id = f"enemy_{card['card_name']}_{slot_idx}"
            enemy_cards[card_id] = card
            self.enemy_hand_state[card_id] = self.frame_count

        # Detect disappeared cards
        self._detect_disappeared_cards(ally_cards, enemy_cards)

        return ally_cards, enemy_cards

    def _detect_disappeared_cards(self, current_ally: Dict, current_enemy: Dict):
        # same logic as your original code
        for card_id in list(self.ally_hand_state.keys()):
            if card_id not in current_ally:
                self.recently_disappeared_ally[card_id] = self.frame_count
                self.logger.info(f"Ally card disappeared: {card_id}")
                del self.ally_hand_state[card_id]

        for card_id in list(self.enemy_hand_state.keys()):
            if card_id not in current_enemy:
                self.recently_disappeared_enemy[card_id] = self.frame_count
                self.logger.info(f"Enemy card disappeared: {card_id}")
                del self.enemy_hand_state[card_id]

        self._cleanup_old_disappearances()

    def _cleanup_old_disappearances(self):
        current_frame = self.frame_count
        self.recently_disappeared_ally = {
            card_id: frame for card_id, frame in self.recently_disappeared_ally.items()
            if current_frame - frame <= self.disappearance_memory_frames
        }
        self.recently_disappeared_enemy = {
            card_id: frame for card_id, frame in self.recently_disappeared_enemy.items()
            if current_frame - frame <= self.disappearance_memory_frames
        }

    def get_recent_disappearances(self) -> Tuple[Dict, Dict]:
        return self.recently_disappeared_ally, self.recently_disappeared_enemy

    def detect_placements(self, current_frame: int, max_frames_back: int = 5) -> List[Dict]:
        placements = []
        recent_ally, recent_enemy = self.get_recent_disappearances()
        for card_id, frame_disappeared in recent_ally.items():
            if current_frame - frame_disappeared <= max_frames_back:
                placements.append({
                    'type': 'ally_placement',
                    'card_id': card_id,
                    'frame': frame_disappeared,
                    'card_name': card_id.split('_')[1] if '_' in card_id else 'unknown'
                })
        for card_id, frame_disappeared in recent_enemy.items():
            if current_frame - frame_disappeared <= max_frames_back:
                placements.append({
                    'type': 'enemy_placement',
                    'card_id': card_id,
                    'frame': frame_disappeared,
                    'card_name': card_id.split('_')[1] if '_' in card_id else 'unknown'
                })
        return placements
    
    # ========================
    # END CARD TRACKING METHODS
    # ========================
        
    def detect_arena_color(self, frame: np.ndarray) -> bool:
        """Sample arena background color from center region"""
        try:
            x, y, w, h = config.ARENA_SAMPLE_REGION
            
            # Ensure coordinates are within frame
            height, width = frame.shape[:2]
            x = max(0, min(x, width - w))
            y = max(0, min(y, height - h))
            
            # Sample the region
            arena_region = frame[y:y+h, x:x+w]
            self.arena_color = np.mean(arena_region, axis=(0, 1)).astype(np.uint8)
            
            self.logger.info(f"Arena color detected: {self.arena_color}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to detect arena color: {e}")
            return False
    
    def detect_motion(self, frame: np.ndarray) -> List[Dict]:
        """Detect moving objects in frame within the tracking region only"""
        # Extract tracking region from frame
        track_x, track_y, track_w, track_h = config.TRACKING_REGION
        
        # Ensure tracking region is within frame bounds
        height, width = frame.shape[:2]
        track_x = max(0, min(track_x, width - track_w))
        track_y = max(0, min(track_y, height - track_h))
        track_w = min(track_w, width - track_x)
        track_h = min(track_h, height - track_y)
        
        # Extract tracking region
        tracking_region = frame[track_y:track_y+track_h, track_x:track_x+track_w]
        
        # Apply background subtraction only to tracking region
        fg_mask = self.bg_subtractor.apply(tracking_region)
        
        # Clean up the mask with more aggressive morphological operations
        kernel_size = getattr(config, 'MOTION_BLUR_KERNEL_SIZE', 5)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Remove noise with opening (erosion followed by dilation)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        # Fill small gaps with closing (dilation followed by erosion)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        # Additional erosion to remove thin connections
        fg_mask = cv2.erode(fg_mask, kernel, iterations=1)
        # Dilation to restore size
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Debug: Track what gets filtered out
        debug_stats = {
            'raw_contours': len(contours),
            'size_filtered': 0,
            'shape_filtered': 0,
            'density_filtered': 0,
            'color_filtered': 0,
            'final_objects': 0
        }
        
        detected_objects = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by size (more restrictive now)
            if area < config.MIN_OBJECT_SIZE or area > config.MAX_OBJECT_SIZE:
                continue
            debug_stats['size_filtered'] += 1
            
            # Get bounding box (relative to tracking region)
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by aspect ratio (avoid very thin/wide objects like UI elements)
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio < 0.3 or aspect_ratio > 3.0:  # Keep roughly square-ish objects
                continue
            debug_stats['shape_filtered'] += 1
            
            # Filter by area vs bounding box ratio (avoid sparse/scattered detections)
            bbox_area = w * h
            if bbox_area > 0 and (area / bbox_area) < 0.4:  # Increased from 0.3 to 0.4 - more restrictive
                continue
            debug_stats['density_filtered'] += 1
            
            # Convert to absolute frame coordinates
            abs_x = track_x + x
            abs_y = track_y + y
            
            # Filter objects that are mostly arena-colored (background)
            if self._is_background_object(frame, abs_x, abs_y, w, h):
                continue
            debug_stats['color_filtered'] += 1
            
            obj = {
                'bbox': (abs_x, abs_y, w, h),
                'center': (abs_x + w // 2, abs_y + h // 2),
                'area': area,
                'confidence': min(area / config.MAX_OBJECT_SIZE, 1.0)
            }
            detected_objects.append(obj)
        
        debug_stats['final_objects'] = len(detected_objects)
        
        # Log debug info every 10 frames to avoid spam
        if self.frame_count % 10 == 0:
            self.logger.debug(f"Detection stats: {debug_stats['raw_contours']} raw -> "
                            f"{debug_stats['size_filtered']} size OK -> "
                            f"{debug_stats['shape_filtered']} shape OK -> "
                            f"{debug_stats['density_filtered']} density OK -> "
                            f"{debug_stats['color_filtered']} color OK -> "
                            f"{debug_stats['final_objects']} final objects")
        
        return detected_objects
    
    def _is_background_object(self, frame: np.ndarray, x: int, y: int, w: int, h: int) -> bool:
        """Check if detected object is mostly background/arena colored"""
        if self.arena_color is None:
            return False
        
        # Sample center of the detected object
        center_x, center_y = x + w // 2, y + h // 2
        sample_size = min(w, h) // 4
        
        x1 = max(0, center_x - sample_size)
        y1 = max(0, center_y - sample_size)
        x2 = min(frame.shape[1], center_x + sample_size)
        y2 = min(frame.shape[0], center_y + sample_size)
        
        if x2 <= x1 or y2 <= y1:
            return False
        
        sample_region = frame[y1:y2, x1:x2]
        avg_color = np.mean(sample_region, axis=(0, 1))
        
        # Check if similar to arena color
        color_diff = np.abs(avg_color - self.arena_color)
        return np.max(color_diff) < self.arena_tolerance
    
    def update_tracking(self, frame: np.ndarray, detected_objects: List[Dict]) -> List[Dict]:
        """Detection-based tracking with temporal persistence for continuous IDs"""
        # Filter detected objects to tracking region and prioritize by significance
        filtered_objects = self._filter_and_prioritize_objects(detected_objects)
        
        # Card-based tracking control - START AT 0, ONLY TRACK WHEN CARDS PLAYED
        cards_played_recently = len(self.recently_disappeared_ally) + len(self.recently_disappeared_enemy)
        
        if config.CARD_BASED_TRACKING:
            if cards_played_recently == 0:
                # No cards played = no tracking allowed
                self.logger.debug("No cards played recently - no tracking allowed")
                return []
            else:
                # Allow tracking proportional to cards played, but much more restrictive
                max_tracking_slots = min(config.MAX_TRACKED_OBJECTS, cards_played_recently + 1)  # Only +1 buffer
                self.logger.debug(f"Cards played: {cards_played_recently}, allowing up to {max_tracking_slots} tracked objects")
                # Only take the most significant objects (sorted by area)
                filtered_objects = sorted(filtered_objects, key=lambda x: x['area'], reverse=True)[:max_tracking_slots]
        
        # Step 1: Update existing tracked objects with velocity prediction
        for obj_id, obj_info in self.tracked_objects.items():
            if obj_info.get('active', False):
                # Predict where the object should be based on previous movement
                if 'velocity' in obj_info:
                    predicted_center = (
                        obj_info['center'][0] + obj_info['velocity'][0],
                        obj_info['center'][1] + obj_info['velocity'][1]
                    )
                    obj_info['predicted_center'] = predicted_center
                else:
                    obj_info['predicted_center'] = obj_info['center']
        
        # Step 2: Match detected objects to existing tracks using improved association
        matched_detections = set()
        current_objects = []
        
        for obj in filtered_objects:
            best_match_id = None
            best_distance = float('inf')
            obj_center = obj['center']
            
            for existing_id, existing_obj in self.tracked_objects.items():
                if existing_obj.get('active', False):
                    # Use predicted position for better matching
                    predicted_center = existing_obj.get('predicted_center', existing_obj['center'])
                    
                    # Calculate distance to predicted position
                    distance = np.sqrt((obj_center[0] - predicted_center[0])**2 + 
                                     (obj_center[1] - predicted_center[1])**2)
                    
                    # More lenient matching threshold for continuous tracking
                    max_distance = 120  # Increased from 80 pixels
                    
                    # If object hasn't been seen for a few frames, be more lenient
                    frames_since_seen = self.frame_count - existing_obj['last_seen_frame']
                    if frames_since_seen > 1:
                        max_distance = 150  # Even more lenient for objects missing for multiple frames
                    
                    if distance < max_distance and distance < best_distance:
                        best_distance = distance
                        best_match_id = existing_id
            
            if best_match_id is not None:
                # Update existing object
                old_center = self.tracked_objects[best_match_id]['center']
                new_center = obj['center']
                
                # Calculate velocity for next frame prediction
                velocity = (new_center[0] - old_center[0], new_center[1] - old_center[1])
                
                self.tracked_objects[best_match_id].update({
                    'bbox': obj['bbox'],
                    'center': new_center,
                    'area': obj['area'],
                    'frames_tracked': self.tracked_objects[best_match_id]['frames_tracked'] + 1,
                    'last_seen_frame': self.frame_count,
                    'velocity': velocity,
                    'active': True,
                    'confidence': obj.get('confidence', 1.0)
                })
                obj['id'] = best_match_id
                obj['frames_tracked'] = self.tracked_objects[best_match_id]['frames_tracked']
                matched_detections.add(best_match_id)
                current_objects.append(obj)
            else:
                # New object - assign new ID
                obj['id'] = self.next_id
                obj['frames_tracked'] = 1
                self.tracked_objects[self.next_id] = {
                    'bbox': obj['bbox'],
                    'center': obj['center'],
                    'area': obj['area'],
                    'frames_tracked': 1,
                    'first_seen_frame': self.frame_count,
                    'last_seen_frame': self.frame_count,
                    'velocity': (0, 0),  # No movement history yet
                    'active': True,
                    'confidence': obj.get('confidence', 1.0)
                }
                self.next_id += 1
                current_objects.append(obj)
        
        # Step 3: Handle unmatched tracked objects (temporal persistence)
        for obj_id, obj_info in self.tracked_objects.items():
            if obj_info.get('active', False) and obj_id not in matched_detections:
                frames_since_seen = self.frame_count - obj_info['last_seen_frame']
                
                # Keep objects alive for longer if no new cards have been played
                max_missing_frames = 8 if cards_played_recently == 0 else 5
                
                if frames_since_seen <= max_missing_frames:
                    # Object is temporarily missing but still considered active
                    # Add it to current objects with reduced confidence
                    missing_obj = {
                        'id': obj_id,
                        'bbox': obj_info['bbox'],
                        'center': obj_info.get('predicted_center', obj_info['center']),
                        'area': obj_info['area'],
                        'frames_tracked': obj_info['frames_tracked'],
                        'confidence': max(0.3, obj_info.get('confidence', 1.0) - 0.2),  # Reduced confidence
                        'status': 'predicted'  # Mark as predicted rather than detected
                    }
                    current_objects.append(missing_obj)
                    self.logger.debug(f"Object {obj_id} missing for {frames_since_seen} frames - maintaining with prediction")
                else:
                    # Object has been missing too long - deactivate
                    obj_info['active'] = False
                    self.logger.debug(f"Object {obj_id} missing for {frames_since_seen} frames - deactivating")
        
        # Step 4: Clean up very old inactive objects
        objects_to_remove = [
            obj_id for obj_id, obj_info in self.tracked_objects.items()
            if not obj_info.get('active', False) and (self.frame_count - obj_info['last_seen_frame']) > 50
        ]
        for obj_id in objects_to_remove:
            del self.tracked_objects[obj_id]
        
        return current_objects
    
    def _filter_and_prioritize_objects(self, detected_objects: List[Dict]) -> List[Dict]:
        """Filter objects to tracking region and sort by priority"""
        filtered_objects = []
        
        for obj in detected_objects:
            x, y, w, h = obj['bbox']
            
            # Check if object is in tracking region
            if self._is_in_tracking_region(x, y, w, h):
                # Calculate priority score (larger, more central objects get higher priority)
                area = w * h
                center_x, center_y = x + w // 2, y + h // 2
                
                # Distance from center of tracking region
                track_x, track_y, track_w, track_h = config.TRACKING_REGION
                region_center_x = track_x + track_w // 2
                region_center_y = track_y + track_h // 2
                distance_from_center = np.sqrt((center_x - region_center_x)**2 + (center_y - region_center_y)**2)
                
                # Priority: larger area + closer to center + existing confidence
                priority_score = (area / 1000) + (1000 / (distance_from_center + 1)) + (obj['confidence'] * 100)
                obj['priority_score'] = priority_score
                
                filtered_objects.append(obj)
        
        # Sort by priority (highest first)
        filtered_objects.sort(key=lambda x: x['priority_score'], reverse=True)
        return filtered_objects
    
    def _is_in_tracking_region(self, x: int, y: int, w: int, h: int) -> bool:
        """Check if object center is within the designated tracking region"""
        center_x, center_y = x + w // 2, y + h // 2
        track_x, track_y, track_w, track_h = config.TRACKING_REGION
        
        return (track_x <= center_x <= track_x + track_w and 
                track_y <= center_y <= track_y + track_h)
    
    def process_frame(self, frame: np.ndarray, frame_number: int) -> Tuple[List[Dict], np.ndarray, List[Dict]]:
        """Process single frame and return detected objects, debug frame, and card placements"""
        # Detect arena color on first frame
        if frame_number <= config.FRAME_SKIP and self.arena_color is None:
            self.detect_arena_color(frame)
        
        # Update card tracking
        ally_cards, enemy_cards = self.update_hand_states(frame)
        
        # Detect card placements (disappearances)
        placement_events = self.detect_placements(frame_number)
        
        # Detect motion
        detected_objects = self.detect_motion(frame)
        
        # Correlate detected motion with card placements
        correlated_objects = self._correlate_motion_with_placements(detected_objects, placement_events)
        
        # Update tracking
        tracked_objects = self.update_tracking(frame, correlated_objects)
        
        # Add debug overlay
        debug_frame = self._add_debug_overlay(frame, detected_objects, tracked_objects, ally_cards, enemy_cards, placement_events)
        
        return tracked_objects, debug_frame, placement_events
    
    def _correlate_motion_with_placements(self, detected_objects: List[Dict], placement_events: List[Dict]) -> List[Dict]:
        """Correlate detected motion with recent card placements to identify troop types"""
        correlated_objects = []
        
        for obj in detected_objects:
            obj_center = obj['center']
            best_match = None
            best_distance = float('inf')
            
            # Find closest recent card placement
            for placement in placement_events:
                # Simple distance-based correlation (could be improved)
                # For now, just assign the card name to nearby motion
                distance = 100  # Default distance if no specific logic
                if distance < best_distance:
                    best_distance = distance
                    best_match = placement
            
            # Add card information to the object
            if best_match:
                obj['card_name'] = best_match['card_name']
                obj['placement_type'] = best_match['type']
            else:
                obj['card_name'] = 'unknown'
                obj['placement_type'] = 'unknown'
            
            correlated_objects.append(obj)
        
        return correlated_objects
    
    def _add_debug_overlay(self, frame: np.ndarray, detected: List[Dict], tracked: List[Dict], 
                          ally_cards: Dict, enemy_cards: Dict, placements: List[Dict]) -> np.ndarray:
        """Add visual debugging information to frame"""
        debug_frame = frame.copy()
        
        # Draw arena sample region
        x, y, w, h = config.ARENA_SAMPLE_REGION
        cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(debug_frame, "Arena", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw tracking region
        track_x, track_y, track_w, track_h = config.TRACKING_REGION
        cv2.rectangle(debug_frame, (track_x, track_y), (track_x + track_w, track_y + track_h), (255, 255, 0), 2)
        cv2.putText(debug_frame, "Tracking Zone", (track_x, track_y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Draw ally hand regions
        for i, (x, y, w, h) in enumerate(self.ally_hand_coords):
            cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(debug_frame, f"Ally {i+1}", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        # Draw enemy hand regions
        for i, (x, y, w, h) in enumerate(self.enemy_hand_coords):
            cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(debug_frame, f"Enemy {i+1}", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # Draw detected cards in hands
        for card_id, card in ally_cards.items():
            if card['card_name'] != 'Unknown':
                x, y, w, h = card['bbox']
                cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
                cv2.putText(debug_frame, card['card_name'], (x, y+h+15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
        
        for card_id, card in enemy_cards.items():
            if card['card_name'] != 'Unknown':
                x, y, w, h = card['bbox']
                cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
                cv2.putText(debug_frame, card['card_name'], (x, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
        
        # Draw detected objects (yellow)
        for obj in detected:
            x, y, w, h = obj['bbox']
            cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            label = f"det {obj['confidence']:.2f}"
            if 'card_name' in obj and obj['card_name'] != 'unknown':
                label += f" ({obj['card_name']})"
            cv2.putText(debug_frame, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        # Draw tracked objects with status indication
        for obj in tracked:
            x, y, w, h = obj['bbox']
            
            # Color code based on status
            if obj.get('status') == 'predicted':
                # Orange for predicted/missing objects
                color = (0, 165, 255)
                thickness = 2
                style = "pred"
            else:
                # Green for actively detected objects
                color = (0, 255, 0)
                thickness = 3
                style = "trk"
            
            cv2.rectangle(debug_frame, (x, y), (x + w, y + h), color, thickness)
            
            label = f"{style} {obj['id']}"
            if obj.get('frames_tracked', 0) > 1:
                label += f" ({obj['frames_tracked']}f)"
            if 'card_name' in obj and obj['card_name'] != 'unknown':
                label += f" {obj['card_name']}"
            
            cv2.putText(debug_frame, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add statistics
        cards_played = len(self.recently_disappeared_ally) + len(self.recently_disappeared_enemy)
        cv2.putText(debug_frame, f"Detected: {len(detected)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(debug_frame, f"Tracked: {len(tracked)}/{config.MAX_TRACKED_OBJECTS}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(debug_frame, f"Cards Played: {cards_played}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(debug_frame, f"Card Placements: {len(placements)}", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if self.arena_color is not None:
            cv2.putText(debug_frame, f"Arena: {self.arena_color}", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show recent placements
        y_offset = 180
        for placement in placements:
            text = f"Placed: {placement['card_name']} ({placement['type']})"
            cv2.putText(debug_frame, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            y_offset += 20
        
        return debug_frame
