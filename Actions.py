import pyautogui
import os
from datetime import datetime
import time
import platform
import config_new as config


class Actions:
    def __init__(self):
        self.os_type = platform.system()
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.images_folder = os.path.join(self.script_dir, 'main_images')

        # Define screen regions based on OS
        self.TOP_LEFT_X = 0
        self.TOP_LEFT_Y = 0
        self.BOTTOM_RIGHT_X = 1280
        self.BOTTOM_RIGHT_Y = 720
        self.FIELD_AREA = (self.TOP_LEFT_X, self.TOP_LEFT_Y,
                           self.BOTTOM_RIGHT_X, self.BOTTOM_RIGHT_Y)

        self.WIDTH = self.BOTTOM_RIGHT_X - self.TOP_LEFT_X
        self.HEIGHT = self.BOTTOM_RIGHT_Y - self.TOP_LEFT_Y

        # Card position to key mapping
        self.card_keys = {
            0: '1',  # Changed from 1 to 0
            1: '2',  # Changed from 2 to 1
            2: '3',  # Changed from 3 to 2
            3: '4'   # Changed from 4 to 3
        }

        # Card name to position mapping (will be updated during detection)
        self.current_ally_card_positions = {}
        self.current_enemy_card_positions = {}

    def capture_area(self, save_path):
        screenshot = pyautogui.screenshot(
            region=(self.TOP_LEFT_X, self.TOP_LEFT_Y, self.WIDTH, self.HEIGHT))
        screenshot.save(save_path)

    def capture_card_area(self, save_path, player_type="ally"):
        """Capture screenshot of card area"""
        if player_type.lower() == "ally":
            region = config.ALLY_REGION
        else:  # enemy
            region = config.ENEMY_REGION

        screenshot = pyautogui.screenshot(region=region)
        screenshot.save(save_path)

    def capture_individual_cards(self, frame=None, player_type="ally", slots=None):
        """
        Capture and split card bar into individual card images

        Args:
            frame: numpy array of the video frame (if None, uses screen capture)
            player_type: "ally" or "enemy"
        """
        player_type = player_type.lower()

        # Get coordinates and region based on player type
        if player_type == "ally":
            coords = config.ALLY_HAND_COORDS
            region = config.ALLY_REGION
        else:  # enemy
            coords = config.ENEMY_HAND_COORDS
            region = config.ENEMY_REGION

        if frame is not None:
            import cv2
            # Extract region from frame instead of screen
            x, y, w, h = region
            screenshot_array = frame[y:y+h, x:x+w]
            # Convert to PIL Image for compatibility with existing code
            from PIL import Image
            screenshot = Image.fromarray(cv2.cvtColor(
                screenshot_array, cv2.COLOR_BGR2RGB))
        else:
            screenshot = pyautogui.screenshot(region=region)

        cards = []
        # Create screenshots directory if needed
        screenshots_dir = os.path.join(self.script_dir, 'screenshots')
        os.makedirs(screenshots_dir, exist_ok=True)

        # Use exact coordinates from config instead of dividing by 4
        # Optional notice when capturing exactly one specific slot
        if slots is not None and isinstance(slots, (list, tuple)) and len(slots) == 1:
            print(f"[CARD CAPTURE] Capturing only slot {slots[0]} for {player_type}")

        for i, (x, y, w, h) in enumerate(coords):
            if slots is not None and i not in slots:
                continue  # skip non-requested slots
            # Calculate relative position within the captured screenshot
            rel_x = x - region[0]
            rel_y = y - region[1]

            # Crop individual card
            card_img = screenshot.crop((rel_x, rel_y, rel_x + w, rel_y + h))
            save_path = os.path.join(
                screenshots_dir, f"{player_type}_card_{i+1}.png")
            card_img.save(save_path)
            cards.append(save_path)

        return {
            'cards': cards,
            'player_type': player_type
        }

    def count_elixir(self):
        if self.os_type == "Darwin":
            for i in range(10, 0, -1):
                image_file = os.path.join(self.images_folder, f"{i}elixir.png")
                try:
                    location = pyautogui.locateOnScreen(
                        image_file, confidence=0.5, grayscale=True)
                    if location:
                        return i
                except Exception as e:
                    print(f"Error locating {image_file}: {e}")
            return 0
        elif self.os_type == "Windows":
            target = (225, 128, 229)
            tolerance = 80
            count = 0
            for x in range(1512, 1892, 38):
                r, g, b = pyautogui.pixel(x, 989)
                if (abs(r - target[0]) <= tolerance) and (abs(g - target[1]) <= tolerance) and (abs(b - target[2]) <= tolerance):
                    count += 1
            return count
        else:
            return 0

    def update_card_positions(self, detections, player_type="ally"):
        """
        Update card positions based on detection results
        detections: list of dictionaries with 'class' and 'x' position
        player_type: "ally" or "enemy"
        """
        # Sort detections by x position (left to right)
        sorted_cards = sorted(detections, key=lambda x: x['x'])

        # Map cards to positions 0-3
        card_positions = {
            card['class']: idx
            for idx, card in enumerate(sorted_cards)
        }

        # Store in appropriate dictionary based on player type
        if player_type.lower() == "ally":
            self.current_ally_card_positions = card_positions
        else:  # enemy
            self.current_enemy_card_positions = card_positions

    def card_play(self, x, y, card_index):
        print(f"Playing card {card_index} at position ({x}, {y})")
        if card_index in self.card_keys:
            key = self.card_keys[card_index]
            print(f"Pressing key: {key}")
            pyautogui.press(key)
            time.sleep(0.2)
            print(f"Moving mouse to: ({x}, {y})")
            pyautogui.moveTo(x, y, duration=0.2)
            print("Clicking")
            pyautogui.click()
        else:
            print(f"Invalid card index: {card_index}")

    def click_battle_start(self):
        button_image = os.path.join(
            self.images_folder, "battlestartbutton.png")
        confidences = [0.8, 0.7, 0.6, 0.5]  # Try multiple confidence levels

        # Define the region (left, top, width, height) for the correct battle button
        battle_button_region = (1486, 755, 1730-1486, 900-755)

        while True:
            for confidence in confidences:
                print(
                    f"Looking for battle start button (confidence: {confidence})")
                try:
                    location = pyautogui.locateOnScreen(
                        button_image,
                        confidence=confidence,
                        region=battle_button_region  # Only search in this region
                    )
                    if location:
                        x, y = pyautogui.center(location)
                        print(f"Found battle start button at ({x}, {y})")
                        pyautogui.moveTo(x, y, duration=0.2)
                        pyautogui.click()
                        return True
                except:
                    pass

            # If button not found, click to clear screens
            print("Button not found, clicking to clear screens...")
            pyautogui.moveTo(1705, 331, duration=0.2)
            pyautogui.click()
            time.sleep(1)

    def detect_game_end(self):
        try:
            winner_img = os.path.join(self.images_folder, "Winner.png")
            confidences = [0.8, 0.7, 0.6]

            winner_region = (1510, 121, 1678-1510, 574-121)

            for confidence in confidences:
                print(f"\nTrying detection with confidence: {confidence}")
                winner_location = None

                # Try to find Winner in region
                try:
                    winner_location = pyautogui.locateOnScreen(
                        winner_img, confidence=confidence, grayscale=True, region=winner_region
                    )
                except Exception as e:
                    print(f"Error locating Winner: {str(e)}")

                if winner_location:
                    _, y = pyautogui.center(winner_location)
                    print(
                        f"Found 'Winner' at y={y} with confidence {confidence}")
                    result = "victory" if y > 402 else "defeat"
                    time.sleep(3)
                    # Click the "Play Again" button at a fixed coordinate (adjust as needed)
                    play_again_x, play_again_y = 1522, 913  # Example coordinates
                    print(
                        f"Clicking Play Again at ({play_again_x}, {play_again_y})")
                    pyautogui.moveTo(play_again_x, play_again_y, duration=0.2)
                    pyautogui.click()
                    return result
        except Exception as e:
            print(f"Error in game end detection: {str(e)}")
        return None

    def detect_match_over(self):
        matchover_img = os.path.join(self.images_folder, "matchover.png")
        confidences = [0.8, 0.6, 0.4]
        # Define the region where the matchover image appears (adjust as needed)
        region = (1378, 335, 1808-1378, 411-335)
        for confidence in confidences:
            try:
                location = pyautogui.locateOnScreen(
                    matchover_img, confidence=confidence, grayscale=True, region=region
                )
                if location:
                    print("Match over detected!")
                    return True
            except Exception as e:
                print(f"Error locating matchover.png: {e}")
        return False
