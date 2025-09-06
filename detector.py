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
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.card_model = self.setup_card_roboflow()
        self.troop_model = self.setup_troop_roboflow()

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

        # If a match is happening
        self.is_game = False
        self.is_game_counter = 0
        self.is_not_game_counter = 0

        # Troop tracking
        self.troop_tracker = TroopTracker(
            max_distance=80.0, max_missing_frames=600)

    def detect_game_active(self, frame):
        # extract region of interest
        x, y, w, h = config.ACTIVE_REGION
        # colors to match for
        target_colors = config.ACTIVE_COLORS
        # color threshold
        threshold = config.ACTIVE_COLOR_THRESHOLD
        # get area of interest
        sample = frame[y:y+h, x:x+w]
        # Get average color of the sample region
        region_color = cv2.mean(sample)[:3]
        # conditional logic to compare sample color mean to specified colors
        for target_color in target_colors:
            # L2, Euclidian distance by default. L1 is Manhatten (|| + ||) Linf is max(|| + || +..)
            distance = np.linalg.norm(
                np.array(region_color) - np.array(target_color))
            if distance < threshold:
                self.is_game_counter += 1
                if self.is_game_counter >= config.ACTIVE_STABLE:
                    # print(f"Active Counter: {self.is_game_counter}")
                    self.is_not_game_counter = 0
                    self.is_game = True
                # self.logger.info(
                #    f"Game is active, distance: {distance}, color: {region_color}")
                return True  # to prevent further detections
        else:
            # self.logger.info(
            #    f"Game is inactive, distance: {distance}, color: {region_color}")
            self.is_not_game_counter += 1
            if self.is_not_game_counter >= config.ACTIVE_STABLE:
                # print(f"Inactive Counter: {self.is_not_game_counter}")
                self.is_game_counter = 0
                self.is_game = False
            return False

    def score_detection(self, obj, troop_info, player, frame_height, pc, config):
        area = obj['area']
        w, h = obj['w'], obj['h']
        if 'abs_x' in obj and 'abs_y' in obj:
            x, y = obj['abs_x'], obj['abs_y']
        else:
            # Convert relative to absolute using tracking offset
            track_x, track_y, _, _ = config.TRACKING_REGION
        x, y = obj['x'] + track_x, obj['y'] + track_y
        aspect_ratio = h / w if w > 0 else 0
        size_score = 1.0 - \
            abs(area/1000.0 - troop_info.get('size_rank', 1))/10.0
        ar_score = 1.0 - abs(aspect_ratio -
                             troop_info.get('aspect_ratio', 1.0))/2.0
        pos_score = 0.5
        for pos in troop_info.get('biased_positions', []):
            if isinstance(pos, str) and hasattr(pc, pos.upper()):
                regions = getattr(pc, pos.upper())[player]
                for region in regions:
                    px, py, pw, ph = region
                    # Check if ENTIRE detection bounding box fits within region
                    detection_right = x + w
                    detection_bottom = y + h
                    region_right = px + pw
                    region_bottom = py + ph

                    if (x >= px and detection_right <= region_right and
                            y >= py and detection_bottom <= region_bottom):
                        print(
                            f"ENTIRE DETECTION IN REGION: ({x},{y},{w},{h}) within ({px},{py},{pw},{ph})")
                        pos_score = 1.0
                        break
                if pos_score == 1.0:
                    break
        # bias placements on the "right side" i.e. if not a tower troop then our side otherwise their side
        if (player == 'ally' and y > frame_height // 2) or (player == 'enemy' and y < frame_height // 2):
            if not any(isinstance(pos, str) and pos.upper() == "TOWER" for pos in troop_info.get('biased_positions', [])):
                # print(f"TROOP {troop_info} GETTING BIASED")
                size_score *= config.MOG2_BIAS_BOOST
                pos_score *= config.MOG2_BIAS_BOOST
        # for tower troops
        elif (player == 'ally' and y < frame_height // 2) or (player == 'enemy' and y > frame_height // 2):
            if any(isinstance(pos, str) and pos.upper() == "TOWER" for pos in troop_info.get('biased_positions', [])):
                # print(f"TOWER TROOP {troop_info} GETTING BIASED")
                size_score *= config.MOG2_BIAS_BOOST
                pos_score *= config.MOG2_BIAS_BOOST

        score = 0.7*size_score + 0.5*ar_score + 0.5*pos_score
        return score

    def setup_card_roboflow(self):
        api_key = os.getenv('ROBOFLOW_API_KEY')
        if not api_key:
            raise ValueError(
                "ROBOFLOW_API_KEY environment variable is not set. Please check your .env file.")

        return InferenceHTTPClient(
            api_url="http://localhost:9001",
            api_key=api_key
        )

    def setup_troop_roboflow(self):
        """Setup troop verification model"""
        api_key = os.getenv('ROBOFLOW_API_KEY')
        if not api_key:
            raise ValueError(
                "ROBOFLOW_API_KEY environment variable is not set. Please check your .env file.")

        return InferenceHTTPClient(
            api_url="http://localhost:9001",
            api_key=api_key
        )

    def verify_troop_detection(self, frame: np.ndarray, obj: dict, expected_troop: str) -> float:
        """
        Verify if detected object matches expected troop using troop model
        Returns confidence score (0-1) or 0 if no match
        """
        if not self.troop_model:
            raise ValueError("Error: No troop_model")

        try:
            # Extract ROI
            x, y, w, h = obj['x'], obj['y'], obj['w'], obj['h']
            roi = frame[y:y+h, x:x+w]

            if roi.size == 0:
                return 0.0

            # Run inference
            TROOP_DETECTION = os.getenv('TROOP_DETECTION')
            # Run inference on the ROI
            results = self.troop_model.infer(
                roi,
                model_id=TROOP_DETECTION
            )
            print(f"inference results: {results}")
            # Parse results using same pattern as detect_hand_cards
            if isinstance(results, dict) and results:
                confidence = results.get('confidence', 0.0)
                predictions = results.get('predictions', [])

                self.logger.info(
                    f"confidence: {confidence} and predictions: {predictions}")

                # If no predictions or empty, return 0
                if not predictions or len(predictions) == 0:
                    self.logger.info(f"no predictions")
                    return 0.0

                # Find expected troop in predictions
                for prediction in predictions:
                    if prediction.get('class', '').lower() == expected_troop.lower():
                        # Return confidence if above threshold
                        if confidence > 0.8:
                            return confidence
                        else:
                            return 0.0

            return 0.0

        except Exception as e:
            print(f"Troop verification error: {e}")
            return 0.0

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

        # Pass the background color to the troop tracker for building removal
        self.troop_tracker.set_arena_background_color(
            self.arena_background_color)

    def track_arena_changes(self, frame, card_changes, ally_placed, enemy_placed):
        import placement_config as pc
        track_x, track_y, track_w, track_h = config.TRACKING_REGION
        debug_frame = frame.copy()
        """Hybrid approach using MOG2 Background Subtraction + Frame Differencing"""

        # Initialize on first frame
        if self.arena_background_color is None:
            self.setup_arena_tracking(frame)
            self.previous_arena_frame = frame.copy()
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                detectShadows=False, varThreshold=config.VAR_THRESHOLD, history=config.HISTORY)
            debug_frame = frame.copy()
            track_x, track_y, track_w, track_h = config.TRACKING_REGION
            cv2.rectangle(debug_frame, (track_x, track_y),
                          (track_x + track_w, track_y + track_h), (0, 255, 0), 2)
            cv2.putText(debug_frame, "Hybrid Detection: MOG2 + Frame Diff", (track_x, track_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            return debug_frame

        track_x, track_y, track_w, track_h = config.TRACKING_REGION
        debug_frame = frame.copy()

        current_region = frame[track_y:track_y +
                               track_h, track_x:track_x + track_w]
        cv2.rectangle(debug_frame, (track_x, track_y),
                      (track_x + track_w, track_y + track_h), (0, 255, 0), 2)
        cv2.putText(debug_frame, "Hybrid Detection: MOG2 + Frame Diff", (track_x, track_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if card_changes:
            cv2.putText(debug_frame, f"Cards: {card_changes}", (track_x, track_y - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        fg_mask = self.bg_subtractor.apply(
            current_region, learningRate=config.LEARNING_RATE)
        self.current_tracking_region = current_region
        self.tracking_offset = (track_x, track_y)

        prev_region = self.previous_arena_frame[track_y:track_y +
                                                track_h, track_x:track_x + track_w]
        mog2_objects = self._detect_with_mog2(current_region, fg_mask)
        # Draw small yellow boxes for all MOG2 detections (debugging)
        for obj in mog2_objects:
            abs_x = track_x + obj['x']
            abs_y = track_y + obj['y']
            w, h = obj['w'], obj['h']
            cv2.rectangle(debug_frame, (abs_x, abs_y),
                          (abs_x + w, abs_y + h), (0, 255, 255), 1)
            cv2.putText(debug_frame, 'MOG2', (abs_x, abs_y - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)

        diff_objects = self._detect_with_frame_diff(
            current_region, prev_region)
        # Draw purple boxes for all Frame Diff detections (debugging)
        for obj in diff_objects:
            abs_x = track_x + obj['x']
            abs_y = track_y + obj['y']
            w, h = obj['w'], obj['h']
            cv2.rectangle(debug_frame, (abs_x, abs_y),
                          (abs_x + w, abs_y + h), (255, 0, 255), 1)
            cv2.putText(debug_frame, 'DIFF', (abs_x, abs_y - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 255), 1)

        if not card_changes and not config.MOG2_DEBUG_ALWAYS_RUN:
            self.latest_detection = None
            self.previous_arena_frame = frame.copy()
            return debug_frame

        if card_changes:
            print(f"DEBUG: Detecting objects for cards: {card_changes}")

        all_objects = mog2_objects + diff_objects
        # print(
        #    f"DEBUG: MOG2 found {len(mog2_objects)} objects, Frame Diff found {len(diff_objects)} objects")

        frame_height = frame.shape[0]
        # Loop over all troops in card_changes
        self.latest_detections = []
        processed_troops = {}  # Track how many detections per troop type

        for entry in card_changes:
            troop = entry['troop']
            player = entry['player']
            troop_key = f"{player}_{troop}"

            # Limit to 1 detection per troop type per frame
            if troop_key in processed_troops:
                print(f"SKIPPING: Already processed {troop_key} this frame")
                continue
            processed_troops[troop_key] = True
            import json
            with open('troop_bias_config.json', 'r') as f:
                troop_config = json.load(f)["troops"]
            troop_info = troop_config.get(troop, None)
            best_object = None
            best_score = -float('inf')

            if troop_info:
                # Score all objects first
                scored_objects = []
                for obj in all_objects:
                    if 'area' not in obj:
                        obj['area'] = obj.get('w', 1) * obj.get('h', 1)
                    score = self.score_detection(
                        obj, troop_info, player, frame_height, pc, config)
                    scored_objects.append((obj, score))

                # Sort by score (highest first)
                scored_objects.sort(key=lambda x: x[1], reverse=True)

                if scored_objects:
                    # Check if top candidates have similar scores (within threshold)
                    top_score = scored_objects[0][1]
                    close_candidates = [obj for obj, score in scored_objects[:config.TROOP_VERIFICATION_MAX_ATTEMPTS]
                                        if top_score - score <= config.TROOP_VERIFICATION_SCORE_THRESHOLD]

                    # Use troop verification if multiple close candidates
                    if len(close_candidates) > 1:
                        for candidate in close_candidates:
                            verification_confidence = self.verify_troop_detection(
                                frame, candidate, troop)
                            self.logger.info(
                                f"best verification confidence is: {verification_confidence}")
                            if verification_confidence >= config.TROOP_VERIFICATION_MIN_CONFIDENCE:
                                best_object = candidate
                                best_score = verification_confidence
                                break
                    else:
                        self.logger.info(f"no close candidates found")

                    # Fallback to highest scoring object if no verification match
                    if best_object is None:
                        best_object, best_score = scored_objects[0]
            if best_object:
                abs_x = track_x + best_object['x']
                abs_y = track_y + best_object['y']
                w, h = best_object['w'], best_object['h']

                # Expand small detections to minimum size
                min_width = config.MIN_DETECTION_WIDTH
                min_height = config.MIN_DETECTION_HEIGHT
                original_w, original_h = w, h

                if w < min_width or h < min_height:
                    # Calculate expansion needed
                    width_expansion = max(0, min_width - w)
                    height_expansion = max(0, min_height - h)

                    # Expand from center - adjust position and size
                    abs_x = abs_x - width_expansion // 2
                    abs_y = abs_y - height_expansion // 2
                    w = max(w, min_width)
                    h = max(h, min_height)

                    # Ensure we don't go outside frame bounds
                    abs_x = max(0, abs_x)
                    abs_y = max(0, abs_y)
                    # Assuming 720x1280 frame size from config
                    abs_x = min(720 - w, abs_x)
                    abs_y = min(1280 - h, abs_y)

                    print(
                        f"EXPANDED DETECTION: {troop} from {original_w}x{original_h} to {w}x{h} at ({abs_x},{abs_y})")

                label = f"{player}_{troop}"
                best_object['card_type'] = troop
                best_object['player'] = player
                best_object['x'] = abs_x
                best_object['y'] = abs_y
                best_object['w'] = w  # Update with expanded width
                best_object['h'] = h  # Update with expanded height
                best_object['area'] = w * h  # Update area calculation
                cv2.rectangle(debug_frame, (abs_x, abs_y),
                              (abs_x + w, abs_y + h), (0, 0, 255), 3)
                cv2.putText(debug_frame, label, (abs_x, abs_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(debug_frame, f"Area:{best_object.get('area', 0):.0f} Method:{best_object.get('method', '?')}", (
                    abs_x, abs_y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                print(
                    f"FINAL DETECTION: {label} at ({abs_x},{abs_y}) area={best_object.get('area', 0):.0f} via {best_object.get('method', '?')}")
                self.latest_detections.append(best_object)
        self.previous_arena_frame = frame.copy()

        # Print the boxes and their scores

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
        for contour in contours:
            area = cv2.contourArea(contour)
            if 100 < area < 10000:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = h / w if w > 0 else 0
                if 0.1 < aspect_ratio < 10:
                    objects.append({
                        'x': x, 'y': y, 'w': w, 'h': h,
                        'area': area, 'method': 'MOG2'
                    })
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
                    # print(
                    #    f"Frame Diff Detection: area={area:.0f}, pos=({x},{y}), size=({w}x{h}), ratio={aspect_ratio:.2f}")
        return objects

    def detect_hand_cards(self, frame, which="ally"):
        # print("RUNNING MODEL")
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
        prev_mean = np.mean(prev_crop, axis=(0, 1))
        # diff = cv2.absdiff(curr_crop, prev_crop) # pixel diff
        # mean_diff = np.mean(diff)
        color_dist = np.linalg.norm(curr_mean - prev_mean)

        if color_dist > threshold:
            # set cooldown frames
            self.card_detection_cooldown[which] = cooldown_frames
            print(f"DETECTING CARD on dist of {color_dist}")
            return True
        else:
            return False

    def process_frame(self, frame: np.ndarray, frame_number: int) -> Tuple[List, np.ndarray, List]:
        """Process single frame and detect cards"""
        # dynamic call of card detection if pixel difference noted.
        if self.should_run_card_detection(frame, which="ally"):
            print("Detecting hand cards ally")
            ally_cards = self.detect_hand_cards(frame, "ally")
        elif self.card_states:
            ally_cards = self.card_states[-1]["ally"]
        else:
            ally_cards = []

        if self.should_run_card_detection(frame, which="enemy"):
            print("Detecting hand cards enemy")
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

        # --- Troop delay logic (streamlined, before detection) ---
        if not hasattr(self, 'troop_delay_buffer'):
            self.troop_delay_buffer = {}
        delay_config = {}
        import json
        with open('troop_bias_config.json', 'r') as f:
            troop_config = json.load(f)["troops"]

        FPS = config.FPS
        skip = config.FRAME_SKIP
        for troop_name, info in troop_config.items():
            # frames = (frames/second * second)/skip
            delay_config[troop_name] = np.floor(
                (FPS * info.get("delay", 0))/skip)

        # Buffer new placements and decrement all delay counters every frame
        all_changes = []

        # Add new troops to buffer if not present, storing player info
        for troop in (ally_placed or []):
            if troop not in self.troop_delay_buffer:
                self.troop_delay_buffer[troop] = {
                    'delay': delay_config.get(troop, 0), 'player': 'ally'}
        for troop in (enemy_placed or []):
            if troop not in self.troop_delay_buffer:
                self.troop_delay_buffer[troop] = {
                    'delay': delay_config.get(troop, 0), 'player': 'enemy'}
        # Decrement all delay counters and release troops when ready
        for troop in list(self.troop_delay_buffer.keys()):
            entry = self.troop_delay_buffer[troop]
            if entry['delay'] > 0:
                print(
                    f"[DELAY] Withholding {troop} ({entry['player']}): {entry['delay']} frames left")
                entry['delay'] -= 1
                self.troop_delay_buffer[troop] = entry
            else:
                # CRITICAL FIX: Check if this troop type already has active tracks before creating new detection
                existing_tracks = [track for track in self.troop_tracker.tracks
                                   if track.card_type.lower() == troop.lower() and track.player == entry['player']]

                # Also check if any existing tracks are about to be removed (high bg_match_count for buildings)
                stable_tracks = []
                for track in existing_tracks:
                    is_about_to_be_removed = False

                    # Check if this track is close to being removed due to background matching
                    # Close to removal threshold (3)
                    if hasattr(track, 'bg_match_count') and track.bg_match_count >= 2:
                        is_about_to_be_removed = True
                        print(
                            f"[DELAY] Track {track.track_id} ({track.card_type}) has bg_match_count={track.bg_match_count} - about to be removed")

                    if not is_about_to_be_removed:
                        stable_tracks.append(track)

                if stable_tracks:
                    print(
                        f"[DELAY] SUPPRESSING {troop} ({entry['player']}) - {len(stable_tracks)} stable tracks already exist")
                    # Remove from buffer without creating detection
                    del self.troop_delay_buffer[troop]
                else:
                    all_changes.append(
                        {'troop': troop, 'player': entry['player']})
                    del self.troop_delay_buffer[troop]
                    print(
                        f"[DELAY] RELEASING {troop} ({entry['player']}) - no existing tracks")
        if all_changes:
            print("Ready for detection:", all_changes)

        # Always track arena changes (tracking region always visible)
        debug_frame = self.track_arena_changes(
            frame, all_changes, ally_placed, enemy_placed)

        # Prepare detections for tracker
        detections_for_tracker = []

        # Add all new detections from card placements
        if hasattr(self, 'latest_detections') and self.latest_detections:
            detections_for_tracker.extend(self.latest_detections)

        # Determine if we're expecting new cards (i.e., if all_changes has entries)
        expecting_new_cards = bool(all_changes)

        # Update tracker with detections and let it handle optical flow tracking internally
        active_tracks = self.troop_tracker.update(
            detections_for_tracker, frame_number, current_frame=frame, previous_frame=self.previous_full_frame, expecting_new_cards=expecting_new_cards)

        # Store current frame for next iteration
        self.previous_full_frame = frame.copy()

        # ALWAYS draw tracking visualization to show persistent squares
        # Get current troop positions for output
        debug_frame = self.troop_tracker.draw_tracks(debug_frame, frame_number)
        detected_objects = self.troop_tracker.get_active_troops()

        # Return detected objects for compatibility
        placement_events = []

        return detected_objects, debug_frame, placement_events
