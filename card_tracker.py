"""
Card tracking and hand management for Clash Royale troop annotation
"""

import cv2
import numpy as np
import logging
import os
import json
from typing import List, Tuple, Dict, Optional
import config
from inference_sdk import InferenceHTTPClient


class CardTracker:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Single Roboflow client instance (used for all region/hand detections)
        try:
            self.rf_model = self.setup_roboflow()
            self.card_model = self.setup_card_roboflow()
            if not self.workspace_name:
                self.logger.warning("WORKSPACE_CARD_DETECTION not set; Roboflow calls may fail.")
            self.logger.info("Roboflow client initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to setup Roboflow client: {str(e)}")
            self.rf_client = None
            self.workspace_name = None

        # Load card names from JSON file
        self.card_names = self._load_card_names()

        # Card detection regions (hand positions) - use config coordinates
        self.ally_hand_coords = config.ally_hand_coords
        self.enemy_hand_coords = config.enemy_hand_coords

        # Card tracking state
        self.detected_cards = []
        self.last_detection_frame = 0
        self.disappearance_memory_frames = 30  # Remember disappearances for N frames
        self.frame_count = 0

        # Hand state tracking
        self.ally_hand_state: Dict[str, Dict] = {}
        self.enemy_hand_state: Dict[str, Dict] = {}
        self.recently_disappeared_ally: Dict[str, int] = {}
        self.recently_disappeared_enemy: Dict[str, int] = {}

    def setup_roboflow(self):
        api_key = os.getenv('ROBOFLOW_API_KEY')
        if not api_key:
            raise ValueError("ROBOFLOW_API_KEY environment variable is not set. Please check your .env file.")
        
        return InferenceHTTPClient(
            api_url="http://localhost:9001",
            api_key=api_key
        )

    def setup_card_roboflow(self):
        api_key = os.getenv('ROBOFLOW_API_KEY')
        if not api_key:
            raise ValueError("ROBOFLOW_API_KEY environment variable is not set. Please check your .env file.")
        
        return InferenceHTTPClient(
            api_url="http://localhost:9001",
            api_key=api_key
        )

    # --------------------------
    # Card name helpers (unchanged)
    # --------------------------
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

    def reload_card_names(self):
        self.card_names = self._load_card_names()
        self.logger.info(f"Reloaded {len(self.card_names)} card names")

    def get_card_name(self, class_id: int) -> str:
        return self.card_names.get(class_id, "unknown")

    def get_class_id(self, card_name: str) -> int:
        for class_id, name in self.card_names.items():
            if name.lower() == card_name.lower():
                return class_id
        return -1

    def save_card_names(self):
        try:
            json_path = os.path.join(os.path.dirname(__file__), 'models', 'cards', 'card_names.json')
            os.makedirs(os.path.dirname(json_path), exist_ok=True)
            with open(json_path, 'w') as f:
                json.dump(self.card_names, f, indent=2)
            self.logger.info(f"Saved {len(self.card_names)} card names to {json_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving card names to JSON: {e}")
            return False

    def add_card(self, card_name: str) -> int:
        next_id = max(self.card_names.keys()) + 1 if self.card_names else 0
        self.card_names[next_id] = card_name
        if self.save_card_names():
            self.logger.info(f"Added new card: {card_name} with ID {next_id}")
            return next_id
        else:
            del self.card_names[next_id]
            self.logger.error(f"Failed to add card: {card_name}")
            return -1

    def update_card_name(self, class_id: int, new_name: str) -> bool:
        if class_id in self.card_names:
            old_name = self.card_names[class_id]
            self.card_names[class_id] = new_name
            if self.save_card_names():
                self.logger.info(f"Updated card {class_id}: {old_name} -> {new_name}")
                return True
            else:
                self.card_names[class_id] = old_name
                self.logger.error(f"Failed to update card {class_id}")
                return False
        return False

    # --------------------------
    # Roboflow client setup
    # --------------------------
    def setup_card_roboflow(self):
        api_key = os.getenv('ROBOFLOW_API_KEY')
        if not api_key:
            raise ValueError("ROBOFLOW_API_KEY environment variable is not set.")
        return Roboflow(api_key=api_key)

    # --------------------------
    # Unified detection entrypoint
    # --------------------------
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

    # --------------------------
    # Roboflow region runner and parser
    # --------------------------
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
                # fallback to template matching detection list
                return self._template_matching_fallback(region, offset)
        except Exception as e:
            self.logger.error(f"Roboflow region inference failed: {e}")
            return self._template_matching_fallback(region, offset)

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

    # --------------------------
    # Fallback / template matching (unchanged)
    # --------------------------
    def _template_matching_fallback(self, region: np.ndarray, offset: Tuple[int, int]) -> List[Dict]:
        try:
            cards = []
            if region is None or region.size == 0:
                return []
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(contour)
                    if 50 < w < 200 and 50 < h < 200:
                        center_x = offset[0] + x + w // 2
                        center_y = offset[1] + y + h // 2
                        cards.append({
                            'class_id': -1,
                            'card_name': 'unknown_card',
                            'bbox': (offset[0] + x, offset[1] + y, w, h),
                            'confidence': 0.5,
                            'center': (center_x, center_y)
                        })
            return cards
        except Exception as e:
            self.logger.error(f"Error in template matching fallback: {str(e)}")
            return []

    # --------------------------
    # Hand updating uses the unified detect_hand_cards
    # --------------------------
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

    def get_placement_label(self, card_id: str) -> Optional[str]:
        if card_id in self.recently_disappeared_ally:
            return f"ally_{card_id.split('_', 1)[1]}"
        elif card_id in self.recently_disappeared_enemy:
            return f"enemy_{card_id.split('_', 1)[1]}"
        return None

    def detect_placements(self, current_frame: int, max_frames_back: int = 5) -> List[Dict]:
        placements = []
        recent_ally, recent_enemy = self.get_recent_disappearances()
        for card_id, frame_disappeared in recent_ally.items():
            if current_frame - frame_disappeared <= max_frames_back:
                placements.append({
                    'type': 'ally_placement',
                    'card_id': card_id,
                    'frame': frame_disappeared,
                    'label': self.get_placement_label(card_id)
                })
        for card_id, frame_disappeared in recent_enemy.items():
            if current_frame - frame_disappeared <= max_frames_back:
                placements.append({
                    'type': 'enemy_placement',
                    'card_id': card_id,
                    'frame': frame_disappeared,
                    'label': self.get_placement_label(card_id)
                })
        return placements

    # --------------------------
    # Visualization (unchanged)
    # --------------------------
    def visualize_hands(self, frame: np.ndarray, ally_cards: Dict, enemy_cards: Dict) -> np.ndarray:
        vis_frame = frame.copy()
        for i, coords in enumerate(self.ally_hand_coords):
            x, y, w, h = coords
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(vis_frame, f"Ally {i+1}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        for i, coords in enumerate(self.enemy_hand_coords):
            x, y, w, h = coords
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(vis_frame, f"Enemy {i+1}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        for card_id, card in ally_cards.items():
            x, y, w, h = card['bbox']
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(vis_frame, card['card_name'], (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        for card_id, card in enemy_cards.items():
            x, y, w, h = card['bbox']
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(vis_frame, card['card_name'], (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        y_offset = 30
        for card_id, frame in self.recently_disappeared_ally.items():
            cv2.putText(vis_frame, f"Disappeared: {card_id}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y_offset += 20
        for card_id, frame in self.recently_disappeared_enemy.items():
            cv2.putText(vis_frame, f"Disappeared: {card_id}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            y_offset += 20
        return vis_frame

    def get_hand_statistics(self) -> Dict:
        return {
            'ally_cards_count': len(self.ally_hand_state),
            'enemy_cards_count': len(self.enemy_hand_state),
            'recent_disappearances_ally': len(self.recently_disappeared_ally),
            'recent_disappearances_enemy': len(self.recently_disappeared_enemy),
            'total_frames_processed': self.frame_count
        }
