import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
import time
import config_new as config


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

    def is_stale(self, current_frame: int, max_missing_frames: int = 0) -> bool:
        """Check if track should be removed due to being missing too long"""
        return (current_frame - self.last_seen_frame) > max_missing_frames


class TroopTracker:
    """Lightweight tracker that associates detections across frames"""

    def __init__(self, max_distance: float = 100.0, max_missing_frames: int = 0):
        self.tracks: List[TroopTrack] = []
        self.next_track_id = 1
        self.max_distance = max_distance  # Max distance to associate detection with track
        self.max_missing_frames = max_missing_frames

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

    def update(self, detections: List[Dict], frame_number: int) -> List[TroopTrack]:
        """
        Update tracker with new detections from current frame

        Args:
            detections: List of detection dicts with keys: x, y, w, h, area, method
            frame_number: Current frame number

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
                'area': det['area'], 'method': det['method'],
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
            self.next_track_id += 1
            print(
                f"Created new track {new_track.track_id} at ({detection['center_x']:.0f}, {detection['center_y']:.0f})")

        # Remove stale tracks and tracks with low confidence for several frames
        active_tracks = []
        for track in self.tracks:
            # Remove if stale
            if track.is_stale(frame_number, self.max_missing_frames):
                print(
                    f"Removed stale track {track.track_id} (missing {frame_number - track.last_seen_frame} frames)")
                continue
            # Remove if confidence is low for last 2 positions
            if len(track.positions) >= 2:
                last_confidences = [p.get('confidence', 1.0) for p in track.positions[-2:]]
                if any(c < config.TRACKING_CONFIDENCE for c in last_confidences):
                    print(f"Removed track {track.track_id} due to low confidence (last 2 positions)")
                    continue
            active_tracks.append(track)
        self.tracks = active_tracks
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
                is_real_detection = best_detection.get('method', '') != 'OPTICAL_FLOW'
                track.update_position(best_detection, frame_number, is_real_detection=is_real_detection)
                unmatched_detections.remove(best_detection)
                #print(
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

        for track in self.tracks:
            if not track.positions:
                continue

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

        # Draw tracker stats
        confirmed_count = sum(1 for t in self.tracks if t.confirmed)
        cv2.putText(debug_frame, f"Tracks: {len(self.tracks)} ({confirmed_count} confirmed)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return debug_frame
