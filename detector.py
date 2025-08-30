import logging
import os
from typing import List, Tuple
import numpy as np
from inference_sdk import InferenceHTTPClient
from dotenv import load_dotenv
from troop_tracker import TroopTracker
import config_new as config
import cv2

# Load environment variables from .env file
load_dotenv()


class TroopDetector:
    """Simplified detector for card detection only"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.card_model = self.setup_card_roboflow()

        # Initialize Actions for card capture
        from Actions import Actions
        self.actions = Actions()

        # Card state
        self.card_states = []

        # Arena tracking
        self.arena_background_color = None
        self.previous_arena_frame = None
        self.current_full_frame = None
        self.previous_full_frame = None
        self.bg_subtractor = None

        # Troop tracking
        self.troop_tracker = TroopTracker(
            max_distance=80.0, max_missing_frames=600)
        
    def setup_card_roboflow(self):
        api_key = os.getenv('ROBOFLOW_API_KEY')
        if not api_key:
            raise ValueError(
                "ROBOFLOW_API_KEY environment variable is not set. Please check your .env file.")

        return InferenceHTTPClient(
            api_url="http://localhost:9001",
            api_key=api_key
        )

    def update_card_states(self, frame, ally_cards, enemy_cards):
        # update card states
        new_state = {"frame": frame, "ally": ally_cards, "enemy": enemy_cards}

        if len(self.card_states) >= 10:  # rolling window max 10 items
            self.card_states = self.card_states[-9:] + [new_state]
        else:
            self.card_states.append(new_state)

    def check_card_changes(self, which):
        """Detect real card changes by looking for A -> Unknown(s) -> B pattern"""
        if len(self.card_states) < 2:
            return None

        changes = []
        curr_state = self.card_states[-1][which]

        # Check each card position
        for pos in range(len(curr_state)):
            # Current card is confident (not Unknown)
            if curr_state[pos] != "Unknown":
                # Look back to find the last confident card at this position
                prev_confident_card = self._find_last_confident_card(
                    which, pos)

                # If we found a different confident card before, it was placed
                if prev_confident_card and prev_confident_card != curr_state[pos]:
                    changes.append(prev_confident_card)
        return changes if changes else None

    def _find_last_confident_card(self, which, position):
        """Look back through states to find the last confident card at given position"""
        # Start from second-to-last state and go backwards
        for i in range(len(self.card_states) - 2, -1, -1):
            if i < 0:
                break
            state = self.card_states[i][which]
            if len(state) > position and state[position] != "Unknown":
                return state[position]
        return None

    def setup_arena_tracking(self, frame):
        """Initialize arena background color from the first frame"""
        import config_new as config

        x, y, w, h = config.ARENA_SAMPLE_REGION
        arena_sample = frame[y:y+h, x:x+w]
        # Get average color of the sample region
        self.arena_background_color = cv2.mean(arena_sample)[:3]  # BGR values
        print(f"Arena background color set to: {self.arena_background_color}")

    def track_arena_changes(self, frame, card_changes, ally_placed, enemy_placed):
        """Hybrid approach using MOG2 Background Subtraction + Frame Differencing"""

        # Initialize on first frame
        if self.arena_background_color is None:
            self.setup_arena_tracking(frame)
            self.previous_arena_frame = frame.copy()
            # Initialize MOG2 background subtractor - industry standard
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                detectShadows=False, varThreshold=16, history=20)

            debug_frame = frame.copy()
            track_x, track_y, track_w, track_h = config.TRACKING_REGION
            cv2.rectangle(debug_frame, (track_x, track_y),
                          (track_x + track_w, track_y + track_h), (0, 255, 0), 2)
            cv2.putText(debug_frame, "Hybrid Detection: MOG2 + Frame Diff", (track_x, track_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            return debug_frame

        track_x, track_y, track_w, track_h = config.TRACKING_REGION
        debug_frame = frame.copy()

        # Extract tracking region
        current_region = frame[track_y:track_y +
                               track_h, track_x:track_x + track_w]

        # Always show tracking region
        cv2.rectangle(debug_frame, (track_x, track_y),
                      (track_x + track_w, track_y + track_h), (0, 255, 0), 2)
        cv2.putText(debug_frame, "Hybrid Detection: MOG2 + Frame Diff", (track_x, track_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if card_changes:
            cv2.putText(debug_frame, f"Cards: {card_changes}", (track_x, track_y - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        # Always update background model
        fg_mask = self.bg_subtractor.apply(current_region)

        # Store current region for tracking
        self.current_tracking_region = current_region
        self.tracking_offset = (track_x, track_y)

        # Only detect NEW objects when cards are placed
        if not card_changes:
            self.latest_detection = None
            self.previous_arena_frame = frame.copy()
            return debug_frame

        print(f"DEBUG: Detecting objects for cards: {card_changes}")

        # Get previous region for frame differencing
        prev_region = self.previous_arena_frame[track_y:track_y +
                                                track_h, track_x:track_x + track_w]

        # METHOD 1: MOG2 Background Subtraction (Industry Standard)
        mog2_objects = self._detect_with_mog2(current_region, fg_mask)

        # METHOD 2: Frame Differencing (Reliable Backup)
        diff_objects = self._detect_with_frame_diff(
            current_region, prev_region)

        # Combine both methods
        all_objects = mog2_objects + diff_objects

        print(
            f"DEBUG: MOG2 found {len(mog2_objects)} objects, Frame Diff found {len(diff_objects)} objects")

        # Draw ALL detected objects for debugging (very permissive)
        for obj in all_objects:
            abs_x = track_x + obj['x']
            abs_y = track_y + obj['y']
            w, h = obj['w'], obj['h']

            # Yellow rectangles for all detections
            cv2.rectangle(debug_frame, (abs_x, abs_y),
                          (abs_x + w, abs_y + h), (0, 255, 255), 1)
            cv2.putText(debug_frame, f"{obj['method']}:{obj['area']:.0f}",
                        (abs_x, abs_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)

        # Select best detection for final output
        if all_objects:
            best_object = max(all_objects, key=lambda x: x['area'])

            abs_x = track_x + best_object['x']
            abs_y = track_y + best_object['y']
            w, h = best_object['w'], best_object['h']

            # Determine card type and player
            card_type = "Unknown"
            player = "Unknown"
            for card_name in card_changes:
                if card_name != "Unknown":
                    if ally_placed and card_name in ally_placed:
                        player = "ally"
                    elif enemy_placed and card_name in enemy_placed:
                        player = "enemy"
                    else:
                        player = "ally" if abs_y > frame.shape[0] // 2 else "enemy"

                    card_type = card_name
                    break

            label = f"{player}_{card_type}"

            # Add card type and player info to detection for tracker
            best_object['card_type'] = card_type
            best_object['player'] = player

            # Convert to absolute coordinates for tracker
            best_object['x'] = abs_x
            best_object['y'] = abs_y

            # Draw final detection (thick red rectangle)
            cv2.rectangle(debug_frame, (abs_x, abs_y),
                          (abs_x + w, abs_y + h), (0, 0, 255), 3)
            cv2.putText(debug_frame, label, (abs_x, abs_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(debug_frame, f"Area:{best_object['area']:.0f} Method:{best_object['method']}",
                        (abs_x, abs_y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            print(
                f"FINAL DETECTION: {label} at ({abs_x},{abs_y}) area={best_object['area']:.0f} via {best_object['method']}")

            # Store detection for return
            self.latest_detection = best_object
        else:
            self.latest_detection = None

        # Update for next frame
        self.previous_arena_frame = frame.copy()
        return debug_frame

    def _detect_with_mog2(self, region, fg_mask):
        """MOG2 Background Subtraction Detection - Industry Standard"""

        # Clean up mask with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        objects = []
        # Get frame height for bias logic
        frame_height = region.shape[0] if region is not None else 0
        # Use ally/enemy placed info from self (set externally)
        ally_placed = getattr(self, 'ally_placed', None)
        enemy_placed = getattr(self, 'enemy_placed', None)
        for contour in contours:
            area = cv2.contourArea(contour)
            if 100 < area < 10000:  # Very permissive size filter
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                # Apply bias boost
                if ally_placed and y > frame_height // 2:
                    area *= config.MOG2_BIAS_BOOST
                elif enemy_placed and y < frame_height // 2:
                    area *= config.MOG2_BIAS_BOOST
                if 0.1 < aspect_ratio < 10:  # Very permissive aspect ratio
                    objects.append({
                        'x': x, 'y': y, 'w': w, 'h': h,
                        'area': area, 'method': 'MOG2'
                    })
                    print(
                        f"MOG2 Detection: area={area:.0f}, pos=({x},{y}), size=({w}x{h}), ratio={aspect_ratio:.2f}")
        return objects

    def _detect_with_frame_diff(self, current, previous):
        """Frame Differencing Detection - Reliable Backup"""
        # Convert to grayscale and compute difference
        curr_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(curr_gray, prev_gray)
        _, diff_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        # Clean up with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_OPEN, kernel)
        diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            diff_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        objects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 100 < area < 10000:  # Very permissive size filter
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                if 0.1 < aspect_ratio < 10:  # Very permissive aspect ratio
                    objects.append({
                        'x': x, 'y': y, 'w': w, 'h': h,
                        'area': area, 'method': 'DIFF'
                    })
                    print(
                        f"Frame Diff Detection: area={area:.0f}, pos=({x},{y}), size=({w}x{h}), ratio={aspect_ratio:.2f}")
        return objects

    def detect_hand_cards(self, frame, which="ally"):
        print("RUNNING MODEL")
        try:
            # Capture individual cards with frame and player type
            card_data = self.actions.capture_individual_cards(
                frame=frame, player_type=which)
            card_paths = card_data['cards']
            # print(f"\nIndividual card predictions for {which}:")

            cards = []
            CARD_DETECTION = os.getenv('CARD_DETECTION')
            if not CARD_DETECTION:
                raise ValueError(
                    "CARD_DETECTION environment variable is not set. Please check your .env file.")

            for card_path in card_paths:
                results = self.card_model.infer(
                    card_path,
                    model_id=CARD_DETECTION
                )
                # print("Card detection raw results:", results)  # Debug print
                # Fix: parse nested structure
                if isinstance(results, dict) and results:
                    confidence = results.get('confidence', 0.0)
                    predictions = results.get('predictions', [])

                    # If no predictions or empty, treat as unknown
                    if not predictions or len(predictions) == 0:
                        cards.append("Unknown")
                    # If low confidence (uncertain), treat as unknown for now
                    elif confidence <= 0.8:
                        cards.append("Unknown")
                    # High confidence, use the top prediction
                    else:
                        cards.append(results['top'])
                else:
                    cards.append("Unknown")
            return cards
        except Exception as e:
            print(f"Error in detect_hand_cards for {which}: {e}")
            return []
        
    def should_run_card_detection(self, frame: np.ndarray, which, threshold=config.THRESHOLD, cooldown_frames=config.COOLDOWN_FRAMES):
        if not hasattr(self, 'card_detection_cooldown'):
            self.card_detection_cooldown = {'ally': 0, 'enemy': 0}
        
        # skip this frame if on cooldown to avoid counting the brief background appearance
        if self.card_detection_cooldown[which] > 0:
            self.card_detection_cooldown[which] -= 1
            return False
        
        # Define your card region, e.g., from config
        if which == 'ally':
            x, y, w, h = config.ALLY_REGION
        elif which == 'enemy':
            x, y, w, h = config.ENEMY_REGION
        else:
            print("invalid which passed in should_run_card_detection")
        
        curr_crop = frame[y:y+h, x:x+w]
        curr_mean = np.mean(curr_crop, axis=(0, 1))
        if self.previous_full_frame is None:
            return True  # Always run on first frame
        prev_crop = self.previous_full_frame[y:y+h, x:x+w]
        prev_mean = np.mean(prev_crop,axis=(0,1))
        #diff = cv2.absdiff(curr_crop, prev_crop) # pixel diff
        #mean_diff = np.mean(diff)
        color_dist = np.linalg.norm(curr_mean - prev_mean)

        if color_dist > threshold:
            self.card_detection_cooldown[which] = cooldown_frames # set cooldown frames
            print(f"DETECTING CARD on dist of {color_dist}")
            return True
        else:
            return False

    def process_frame(self, frame: np.ndarray, frame_number: int) -> Tuple[List, np.ndarray, List]:
        """Process single frame and detect cards"""

        # dynamic call of card detection if pixel difference noted.
        if self.should_run_card_detection(frame, which="ally"):
            ally_cards = self.detect_hand_cards(frame, "ally")
        elif self.card_states:
            ally_cards = self.card_states[-1]["ally"]
        else:
            ally_cards = []

        if self.should_run_card_detection(frame, which="enemy"):
            enemy_cards = self.detect_hand_cards(frame, "enemy")
        elif self.card_states:
            enemy_cards = self.card_states[-1]["enemy"]
        else:
            enemy_cards = []

        # Update card states
        self.update_card_states(frame_number, ally_cards, enemy_cards)

        # Check card states
        ally_placed = self.check_card_changes("ally")
        enemy_placed = self.check_card_changes("enemy")

        # Combine all card changes
        all_changes = []
        if ally_placed:
            all_changes.extend(ally_placed)
            print("ally placed:", ally_placed)
        if enemy_placed:
            all_changes.extend(enemy_placed)
            print("enemy placed:", enemy_placed)

        # Always track arena changes (tracking region always visible)
        debug_frame = self.track_arena_changes(
            frame, all_changes, ally_placed, enemy_placed)

        # Prepare detections for tracker
        detections_for_tracker = []

        # Add new detections from card placements
        if hasattr(self, 'latest_detection') and self.latest_detection:
            detections_for_tracker.append(self.latest_detection)

        # Add continuous tracking detections for existing troops using optical flow
        if self.previous_full_frame is not None:
            tracking_detections = self.troop_tracker.track_existing_troops(
                frame, self.previous_full_frame, frame_number)
            detections_for_tracker.extend(tracking_detections)

        # Update tracker with all detections (new + tracked)
        active_tracks = self.troop_tracker.update(
            detections_for_tracker, frame_number)

        # Store current frame for next iteration
        self.previous_full_frame = frame.copy()

        # ALWAYS draw tracking visualization to show persistent squares
        # Get current troop positions for output
        debug_frame = self.troop_tracker.draw_tracks(debug_frame, frame_number)
        detected_objects = self.troop_tracker.get_active_troops()

        # Return detected objects for compatibility
        placement_events = []

        return detected_objects, debug_frame, placement_events
