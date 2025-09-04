import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
import time
import config_new as config
import json
import os


class TroopTrack:
    """Represents a single troop being tracked across frames"""

    def __init__(self, track_id: int, initial_detection: Dict, frame_number: int):
        self.track_id = track_id
        self.positions = [initial_detection]  # History of positions
        self.last_seen_frame = frame_number
        self.creation_frame = frame_number
        self.confirmed = False  # Becomes True after surviving multiple frames
        self.card_type = initial_detection.get('card_type', 'Unknown')
        self.player = initial_detection.get('player', 'Unknown')

        # Motion prediction
        self.velocity = np.array([0.0, 0.0])  # dx, dy per frame
        self.predicted_position = None
        self.card_slot_buffers = {'ally': [], 'enemy': []}

    def update_position(self, detection: Dict, frame_number: int, is_real_detection: bool = True):
        """Update track with new detection. Only update last_seen_frame if real detection."""
        current_pos = np.array([detection['center_x'], detection['center_y']])

        if len(self.positions) >= 2:
            prev_pos = np.array(
                [self.positions[-1]['center_x'], self.positions[-1]['center_y']])
            frame_diff = frame_number - self.last_seen_frame
            if frame_diff > 0:
                self.velocity = (current_pos - prev_pos) / frame_diff

        self.positions.append(detection)
        if is_real_detection:
            self.last_seen_frame = frame_number

        if len(self.positions) >= 3:
            self.confirmed = True
        if len(self.positions) > 10:
            self.positions = self.positions[-10:]

    def predict_next_position(self, frame_number: int) -> Tuple[float, float]:
        """Predict where the troop should be based on motion"""
        if not self.positions:
            return None, None

        last_pos = np.array([self.positions[-1]['center_x'],
                            self.positions[-1]['center_y']])
        frames_ahead = frame_number - self.last_seen_frame

        predicted = last_pos + (self.velocity * frames_ahead)
        return predicted[0], predicted[1]

    def track_position(self, current_frame: np.ndarray, previous_frame: np.ndarray, frame_number: int) -> Optional[Dict]:
        """Track troop position using optical flow and motion prediction"""
        import cv2
        import numpy as np

        if not self.positions or previous_frame is None:
            return None

        last_pos = self.positions[-1]

        # Initialize feature points if needed
        if not hasattr(self, 'feature_points') or self.feature_points is None:
            # Create feature points around the last known position
            x, y, w, h = last_pos['x'], last_pos['y'], last_pos['w'], last_pos['h']

            try:
                # Convert to grayscale for feature detection
                gray_prev = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY) if len(
                    previous_frame.shape) == 3 else previous_frame

                # Define region of interest around the troop
                roi_x1 = max(0, x - 10)
                roi_y1 = max(0, y - 10)
                roi_x2 = min(gray_prev.shape[1], x + w + 10)
                roi_y2 = min(gray_prev.shape[0], y + h + 10)

                # Ensure ROI is valid
                if roi_x2 <= roi_x1 or roi_y2 <= roi_y1:
                    return None

                # Detect good features to track in the ROI
                roi = gray_prev[roi_y1:roi_y2, roi_x1:roi_x2]

                if roi.size == 0:  # Empty ROI
                    return None

                corners = cv2.goodFeaturesToTrack(
                    roi, maxCorners=15, qualityLevel=0.05, minDistance=10)

                if corners is not None and len(corners) > 0:
                    # Convert corners back to full frame coordinates
                    self.feature_points = corners + \
                        np.array([roi_x1, roi_y1], dtype=np.float32)
                else:
                    return None

            except Exception as e:
                print(
                    f"Feature detection error for track {self.track_id}: {e}")
                return None

        # Track features using Lucas-Kanade optical flow
        gray_prev = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY) if len(
            previous_frame.shape) == 3 else previous_frame
        gray_curr = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY) if len(
            current_frame.shape) == 3 else current_frame

        if self.feature_points is not None and len(self.feature_points) > 0:
            # Parameters for Lucas-Kanade optical flow
            lk_params = dict(winSize=(25, 25), maxLevel=3,
                             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

            try:
                # Calculate optical flow
                new_points, status, error = cv2.calcOpticalFlowPyrLK(
                    gray_prev, gray_curr, self.feature_points, None, **lk_params)

                # Select good points - fix dimension mismatch issue
                if status is not None and len(status) > 0:
                    # Flatten status to 1D to match point indexing
                    status_mask = status.flatten() == 1

                    # Ensure we have valid points to work with
                    if np.any(status_mask) and new_points.size > 0:
                        good_new = new_points[status_mask]
                        good_old = self.feature_points[status_mask]
                    else:
                        good_new = np.array([]).reshape(
                            0, 2)  # Ensure proper shape
                        good_old = np.array([]).reshape(0, 2)
                else:
                    good_new = np.array([]).reshape(0, 2)
                    good_old = np.array([]).reshape(0, 2)

                if len(good_new) >= 3:  # Need at least 3 points for reliable tracking
                    # Calculate displacement - ensure we have valid arrays
                    if good_new.size > 0 and good_old.size > 0 and good_new.shape == good_old.shape:
                        try:
                            displacement = np.mean(good_new - good_old, axis=0)

                            # Ensure displacement is a 1D array with 2 elements
                            displacement = np.asarray(displacement).flatten()

                            if len(displacement) >= 2:
                                # Use item() to extract scalar values safely
                                dx = displacement[0].item() if hasattr(
                                    displacement[0], 'item') else float(displacement[0])
                                dy = displacement[1].item() if hasattr(
                                    displacement[1], 'item') else float(displacement[1])
                            else:
                                dx, dy = 0.0, 0.0
                        except Exception as e:
                            print(
                                f"Displacement calculation error for track {self.track_id}: {e}")
                            dx, dy = 0.0, 0.0
                    else:
                        # Fallback if arrays are incompatible
                        dx, dy = 0.0, 0.0

                    # Update position based on displacement
                    new_x = int(last_pos['x'] + dx)
                    new_y = int(last_pos['y'] + dy)
                    new_w = last_pos['w']
                    new_h = last_pos['h']

                    # Ensure position is within frame bounds
                    new_x = max(0, min(current_frame.shape[1] - new_w, new_x))
                    new_y = max(0, min(current_frame.shape[0] - new_h, new_y))

                    center_x = new_x + new_w // 2
                    center_y = new_y + new_h // 2

                    # Update feature points for next frame
                    self.feature_points = good_new

                    # Calculate confidence based on tracking quality
                    # More points = higher confidence
                    confidence = min(1.0, len(good_new) / 20.0)

                    new_detection = {
                        'x': new_x, 'y': new_y, 'w': new_w, 'h': new_h,
                        'center_x': center_x, 'center_y': center_y,
                        'area': new_w * new_h, 'method': 'OPTICAL_FLOW',
                        'card_type': self.card_type, 'player': self.player,
                        'confidence': confidence
                    }

                    return new_detection
                else:
                    # Not enough good points, reset feature tracking
                    self.feature_points = None
                    return None

            except Exception as e:
                print(f"Optical flow error for track {self.track_id}: {e}")
                # Reset feature points on any error
                self.feature_points = None
                return None

        return None

    def analyze_tracking_region_content(self, current_frame: np.ndarray) -> Dict:
        """Analyze the content of the tracking region to determine if it's worth tracking"""
        if not self.positions:
            return {'has_content': False, 'score': 0.0}

        last_pos = self.positions[-1]
        x, y, w, h = last_pos['x'], last_pos['y'], last_pos['w'], last_pos['h']

        # Extract the tracking region
        roi = current_frame[y:y+h, x:x+w]

        if roi.size == 0:
            return {'has_content': False, 'score': 0.0}

        # Convert to grayscale for analysis
        if len(roi.shape) == 3:
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            roi_gray = roi

        # Multiple content analysis metrics
        content_score = 0.0

        try:
            # 1. Edge density - areas with troops should have more edges
            edges = cv2.Canny(roi_gray, 50, 150)
            edge_density = np.sum(edges > 0) / (w * h)
            edge_score = min(1.0, edge_density * 100)  # Normalize

            # 2. Texture variance - troops have more texture than background
            texture_score = np.var(roi_gray) / 1000.0  # Normalize variance
            texture_score = min(1.0, texture_score)

            # 3. Color variation (if color image)
            color_score = 0.0
            if len(roi.shape) == 3:
                # Standard deviation across color channels
                color_std = np.std(roi.reshape(-1, 3), axis=0)
                color_score = min(1.0, np.mean(color_std) / 50.0)

            # 4. Corner/feature density
            corners = cv2.goodFeaturesToTrack(
                roi_gray, maxCorners=50, qualityLevel=0.01, minDistance=5)
            corner_density = len(corners) / \
                (w * h * 0.001) if corners is not None else 0
            corner_score = min(1.0, corner_density)

            # Combine scores with weights
            content_score = (
                edge_score * 0.3 +
                texture_score * 0.3 +
                color_score * 0.2 +
                corner_score * 0.2
            )

            return {
                'has_content': content_score > 0.15,  # Threshold for "interesting" content
                'score': content_score,
                'edge_density': edge_density,
                'texture_var': np.var(roi_gray),
                'corner_count': len(corners) if corners is not None else 0
            }

        except Exception as e:
            print(f"Content analysis error for track {self.track_id}: {e}")
            return {'has_content': False, 'score': 0.0}

    def analyze_background_match(self, current_frame: np.ndarray, arena_bg_color: np.ndarray = None) -> Dict:
        """Analyze if tracking region matches background colors (for building removal)"""
        if not self.positions:
            return {'matches_background': False, 'color_diff': float('inf')}

        # Check if arena background color is available
        if arena_bg_color is None:
            return {'matches_background': False, 'color_diff': float('inf')}

        last_pos = self.positions[-1]
        x, y, w, h = last_pos['x'], last_pos['y'], last_pos['w'], last_pos['h']

        # Extract the tracking region
        track_roi = current_frame[y:y+h, x:x+w]
        if track_roi.size == 0:
            return {'matches_background': False, 'color_diff': float('inf')}

        try:
            # Calculate average color of tracking region
            track_avg_color = np.mean(track_roi.reshape(-1, 3), axis=0)

            # Use the provided arena background color (BGR format)
            bg_avg_color = arena_bg_color

            # Calculate color difference (Euclidean distance in RGB space)
            color_diff = np.linalg.norm(track_avg_color - bg_avg_color)

            # Consider it a background match if color difference is small
            threshold = config.BUILDING_BG_COLOR_THRESHOLD
            matches_background = color_diff < threshold

            return {
                'matches_background': matches_background,
                'color_diff': color_diff,
                'track_color': track_avg_color,
                'bg_color': bg_avg_color
            }

        except Exception as e:
            print(f"Background analysis error for track {self.track_id}: {e}")
            return {'matches_background': False, 'color_diff': float('inf')}

    def is_stale(self, current_frame: int, max_missing_frames: int = 0, troop_config: Dict = None, frame_data: np.ndarray = None, arena_bg_color: np.ndarray = None) -> bool:
        """Check if track should be removed due to being missing too long, exceeding duration, or lack of content"""
        # Check if missing too long (real detections only)
        missing_frames = current_frame - self.last_seen_frame
        if missing_frames > max_missing_frames:
            return True

        # MOVEMENT-BASED REMOVAL: Check if track is moving or just stuck
        if frame_data is not None and len(self.positions) >= 4:
            if not hasattr(self, 'low_activity_count'):
                self.low_activity_count = 0

            # Calculate movement over last few frames
            recent_positions = self.positions[-4:]  # Last 4 positions

            # Calculate total movement distance
            total_movement = 0
            for i in range(1, len(recent_positions)):
                prev_pos = recent_positions[i-1]
                curr_pos = recent_positions[i]

                # Distance between centers
                prev_center = (prev_pos['x'] + prev_pos['w'] //
                               2, prev_pos['y'] + prev_pos['h']//2)
                curr_center = (curr_pos['x'] + curr_pos['w'] //
                               2, curr_pos['y'] + curr_pos['h']//2)

                dx = curr_center[0] - prev_center[0]
                dy = curr_center[1] - prev_center[1]
                movement = (dx*dx + dy*dy) ** 0.5
                total_movement += movement

            # Average movement per frame
            avg_movement = total_movement / (len(recent_positions) - 1)

            # DEBUG: Print movement analysis
            print(f"MOVEMENT DEBUG Track {self.track_id} ({self.card_type}):")
            print(
                f"  Total movement over {len(recent_positions)-1} frames: {total_movement:.2f} pixels")
            print(f"  Average movement per frame: {avg_movement:.2f} pixels")
            print(f"  Low activity count: {self.low_activity_count}")

            # Consider low activity if moving less than 2 pixels per frame on average
            has_movement = avg_movement > 2.0

            # Count low activity frames
            if not has_movement:
                self.low_activity_count += 1
                print(
                    f"Track {self.track_id} LOW MOVEMENT frame {self.low_activity_count}/4 (avg: {avg_movement:.2f} px/frame)")

                # Check if this troop is a building
                is_building = False
                if troop_config and self.card_type.lower() in troop_config.get('troops', {}):
                    troop_info = troop_config['troops'][self.card_type.lower()]
                    biased_positions = troop_info.get("biased_positions", [])
                    is_building = any(
                        pos == "BUILDING" for pos in biased_positions)

                # BUILDING-SPECIFIC REMOVAL: Use background color matching
                if is_building:
                    if not hasattr(self, 'bg_match_count'):
                        self.bg_match_count = 0

                    bg_analysis = self.analyze_background_match(
                        frame_data, arena_bg_color)

                    if bg_analysis['matches_background']:
                        self.bg_match_count += 1
                        print(
                            f"Track {self.track_id} BUILDING BACKGROUND MATCH {self.bg_match_count}/6 (color_diff: {bg_analysis['color_diff']:.1f})")

                        # Remove building after configurable frames of background matching
                        if self.bg_match_count >= config.BUILDING_BG_MATCH_FRAMES:
                            print(
                                f"Track {self.track_id} ({self.card_type}) removed: background match for {self.bg_match_count} frames")
                            return True
                    else:
                        # Reset counter if building content is detected
                        if self.bg_match_count > 0:
                            print(
                                f"Track {self.track_id} BUILDING CONTENT FOUND - resetting bg counter from {self.bg_match_count}")
                        self.bg_match_count = 0

                # NON-BUILDING REMOVAL: Use movement-based removal
                else:
                    # Remove after 4 frames of low movement (but not for buildings)
                    if self.low_activity_count >= 4:
                        print(
                            f"Track {self.track_id} ({self.card_type}) removed: low movement for {self.low_activity_count} frames")
                        return True
            else:
                # Reset counter if movement is found
                if self.low_activity_count > 0:
                    print(
                        f"Track {self.track_id} MOVEMENT FOUND - resetting counter from {self.low_activity_count} (avg: {avg_movement:.2f} px/frame)")
                self.low_activity_count = 0

        # Check if troop has exceeded its duration (for spells/temporary effects)
        if troop_config and self.card_type.lower() in troop_config.get('troops', {}):
            troop_info = troop_config['troops'][self.card_type.lower()]
            duration = troop_info.get('duration')

            if duration is not None:
                # Calculate how many frames this troop has existed
                frames_alive = (current_frame - self.creation_frame)
                # Convert duration from seconds to frames (assuming 30 FPS)
                # You can adjust this FPS value based on your video
                fps = config.FPS
                max_duration_frames = (duration * fps)

                if frames_alive > max_duration_frames:
                    print(
                        f"Track {self.track_id} ({self.card_type}) exceeded duration: {frames_alive} frames > {max_duration_frames} frames ({duration}s)")
                    return True

        return False


class TroopTracker:
    """Lightweight tracker that associates detections across frames"""

    def __init__(self, max_distance: float = 100.0, max_missing_frames: int = 0):
        self.tracks: List[TroopTrack] = []
        self.next_track_id = 1
        self.max_distance = max_distance  # Max distance to associate detection with track
        self.max_missing_frames = max_missing_frames
        self.arena_background_color = None  # Will be set by detector

        # Load troop configuration for duration checking
        self.troop_config = self._load_troop_config()

    def set_arena_background_color(self, color: Tuple[float, float, float]):
        """Set the arena background color for building background matching"""
        self.arena_background_color = np.array(color)
        print(f"TroopTracker: Arena background color set to {color}")

    def _load_troop_config(self) -> Dict:
        """Load troop configuration from JSON file"""
        try:
            config_path = "troop_bias_config.json"
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load troop config: {e}")
        return {}

    def track_existing_troops(self, current_frame: np.ndarray, previous_frame: np.ndarray, frame_number: int) -> List[Dict]:
        """Track existing troops using optical flow"""
        tracking_detections = []

        for track in self.tracks:
            if not track.positions:
                continue

            # Try to track this troop's position using optical flow
            tracked_pos = track.track_position(
                current_frame, previous_frame, frame_number)

            if tracked_pos:
                tracking_detections.append(tracked_pos)

        return tracking_detections

    def update(self, detections: List[Dict], frame_number: int, current_frame: np.ndarray = None) -> List[TroopTrack]:
        """
        Update tracker with new detections from current frame

        Args:
            detections: List of detection dicts with keys: x, y, w, h, area, method
            frame_number: Current frame number
            current_frame: Current frame data for content analysis

        Returns:
            List of active tracks
        """
        # Convert detections to standardized format
        standardized_detections = []
        for det in detections:
            center_x = det['x'] + det['w'] / 2
            center_y = det['y'] + det['h'] / 2
            standardized_detections.append({
                'x': det['x'], 'y': det['y'], 'w': det['w'], 'h': det['h'],
                'center_x': center_x, 'center_y': center_y,
                'area': det['w']*det['h'],
                'card_type': det.get('card_type', 'Unknown'),
                'player': det.get('player', 'Unknown')
            })

        # Predict where existing tracks should be
        self._predict_tracks(frame_number)

        # Associate detections with existing tracks
        unmatched_detections = self._associate_detections(
            standardized_detections, frame_number)

        # Create new tracks for unmatched detections
        for detection in unmatched_detections:
            new_track = TroopTrack(self.next_track_id, detection, frame_number)
            self.tracks.append(new_track)
            print(
                f"TRACK CREATED: ID {self.next_track_id} ({detection.get('card_type', 'Unknown')}) at frame {frame_number}")
            self.next_track_id += 1

        # Remove stale tracks and tracks with low confidence for several frames
        initial_count = len(self.tracks)
        active_tracks = []
        for track in self.tracks:
            # Remove if stale (includes duration and content checking)
            if track.is_stale(frame_number, self.max_missing_frames, self.troop_config, current_frame, self.arena_background_color):
                print(
                    f"CONFIRMED REMOVAL: Track {track.track_id} ({track.card_type}) - Total tracks: {initial_count} -> {initial_count - 1}")
                continue
            # Remove if confidence is low for last 2 positions
            if len(track.positions) >= 2:
                last_confidences = [p.get('confidence', 1.0)
                                    for p in track.positions[-2:]]
                if any(c < config.TRACKING_CONFIDENCE for c in last_confidences):
                    print(
                        f"CONFIRMED REMOVAL: Track {track.track_id} ({track.card_type}) - Low confidence")
                    continue
            active_tracks.append(track)

        # Force update the tracks list and clear any stale references
        old_count = len(self.tracks)
        self.tracks = active_tracks
        new_count = len(self.tracks)

        # Debug: Confirm the change took effect
        if old_count != new_count:
            print(f"TRACKS UPDATED: {old_count} -> {new_count}")

        return self.tracks

    def _predict_tracks(self, frame_number: int):
        '''Update predicted positions for all tracks'''
        for track in self.tracks:
            pred_x, pred_y = track.predict_next_position(frame_number)
            track.predicted_position = (
                pred_x, pred_y) if pred_x is not None else None

    def _associate_detections(self, detections: List[Dict], frame_number: int) -> List[Dict]:
        """Associate detections with existing tracks using distance"""
        unmatched_detections = detections.copy()

        # Sort tracks by confidence (confirmed tracks first, then by recency)
        sorted_tracks = sorted(self.tracks, key=lambda t: (
            t.confirmed, -abs(frame_number - t.last_seen_frame)))

        for track in sorted_tracks:
            if not track.predicted_position or track.predicted_position[0] is None:
                continue

            best_detection = None
            best_distance = float('inf')

            pred_x, pred_y = track.predicted_position

            # Find closest detection to predicted position
            for detection in unmatched_detections:
                det_x, det_y = detection['center_x'], detection['center_y']
                distance = np.sqrt((det_x - pred_x)**2 + (det_y - pred_y)**2)

                if distance < self.max_distance and distance < best_distance:
                    best_distance = distance
                    best_detection = detection

            # Associate best match
            if best_detection:
                # Only update last_seen_frame if this is a real detection (not optical flow)
                is_real_detection = best_detection.get(
                    'method', '') != 'OPTICAL_FLOW'
                track.update_position(
                    best_detection, frame_number, is_real_detection=is_real_detection)
                unmatched_detections.remove(best_detection)
                # print(
                #    f"Updated track {track.track_id} at ({best_detection['center_x']:.0f}, {best_detection['center_y']:.0f}) distance={best_distance:.1f}")

        return unmatched_detections

    def get_active_troops(self) -> List[Dict]:
        """Get current positions of all confirmed troops"""
        troops = []
        for track in self.tracks:
            if track.confirmed and track.positions:
                last_pos = track.positions[-1]
                troops.append({
                    'track_id': track.track_id,
                    'x': last_pos['x'],
                    'y': last_pos['y'],
                    'w': last_pos['w'],
                    'h': last_pos['h'],
                    'center_x': last_pos['center_x'],
                    'center_y': last_pos['center_y'],
                    'card_type': track.card_type,
                    'player': track.player,
                    'age': len(track.positions),
                    'last_seen': track.last_seen_frame
                })
        return troops

    def draw_tracks(self, frame: np.ndarray, frame_number: int) -> np.ndarray:
        """Draw tracking visualization on frame"""
        debug_frame = frame.copy()

        # Force refresh - get current active tracks
        current_tracks = [t for t in self.tracks if t.positions]

        for track in current_tracks:
            last_pos = track.positions[-1]
            x, y, w, h = last_pos['x'], last_pos['y'], last_pos['w'], last_pos['h']

            # Color based on track status
            if track.confirmed:
                color = (0, 255, 0)  # Green for confirmed tracks
                thickness = 2
            else:
                color = (0, 255, 255)  # Yellow for new tracks
                thickness = 1

            # Draw bounding box
            cv2.rectangle(debug_frame, (x, y),
                          (x + w, y + h), color, thickness)

            # Draw track ID and info
            label = f"T{track.track_id}"
            if track.card_type != "Unknown":
                label += f" {track.card_type}"

            cv2.putText(debug_frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

            # Draw motion trail
            if len(track.positions) > 1:
                points = []
                for pos in track.positions[-5:]:  # Last 5 positions
                    points.append((int(pos['center_x']), int(pos['center_y'])))

                for i in range(len(points) - 1):
                    cv2.line(debug_frame, points[i], points[i + 1], color, 1)

            # Draw predicted position
            if track.predicted_position and track.predicted_position[0] is not None:
                pred_x, pred_y = track.predicted_position
                cv2.circle(debug_frame, (int(pred_x), int(pred_y)),
                           5, (255, 0, 255), 2)

        # Draw tracker stats - FORCE CLEAR BACKGROUND FIRST
        stats_region = debug_frame[0:50, 0:400]  # Clear stats area
        stats_region.fill(0)  # Black background to eliminate old text

        confirmed_count = sum(1 for t in self.tracks if t.confirmed)
        cv2.putText(debug_frame, f"Tracks: {len(self.tracks)} ({confirmed_count} confirmed)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return debug_frame
