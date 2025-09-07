"""
Configuration for Clash Royale Troop Detection Tool
"""

# Video processing settings
FRAME_SKIP = 20  # Process every xth frame (60 fps video)
# FRAME_SKIP_CARD_DETECTION = 60 # Only detects card in hand every 60th frame (1 second) # not used anymore
RESIZE_FACTOR = 1.0  # Keep original size
START_TIME_SECONDS = 6  # 247.0  # Start analysis at this time (in seconds)
FPS = 60

DELAY = 0.2  # s per frame delay
PIXEL_DIFF_THRESHOLD = 120
MIN_OBJECT_SIZE = 1200  # Much larger minimum (filters out small movements)
MAX_OBJECT_SIZE = 10000  # Smaller maximum (focus on troop-sized objects)

# Minimum detection size expansion
MIN_DETECTION_WIDTH = 75   # Minimum width for detection boxes
MIN_DETECTION_HEIGHT = 100  # Minimum height for detection boxes

# Detection confidence
DETECTION_CONFIDENCE = 0.5 # lower to allow for grayed out cards

# Detection overlap, anything more than 0.7 of overlap is discounted.
OVERLAP_THRESHOLD = 0.6

MOTION_BLUR_KERNEL_SIZE = 11  # Larger kernel for more smoothing
THRESHOLD = 3  # threshold for color difference to run card detection model
# amount of frames after a detection to wait before detecting the next change
COOLDOWN_FRAMES = 1

# Scoring boost
MOG2_BIAS_BOOST = 1.5  # boost troops on our side by 2x
# still used incase golden
GOLDEN = [253, 255, 69]
ALLY_BLUE = [51, 182, 229] # ally blue to look out for
ENEMY_RED = [228, 21, 76] # enemy red to look out for

# MOG2 settings
HISTORY = 20
VAR_THRESHOLD = 15
LEARNING_RATE = 0.1  # how fast changes are adapted into model background

# Debug setting: Enable/disable MOG2 detection on every frame
MOG2_DEBUG_ALWAYS_RUN = True

# Tracking settings
MAX_TRACKING_FRAMES = 60  # How long to track an object
TRACKING_CONFIDENCE = 0.9  # Minimum confidence to keep tracking
MAX_TRACKED_OBJECTS = 10  # Maximum objects to track simultaneously
MAX_DISTANCE = 150
TRACKING_REGION = (33, 304, 646, 747)  # (x, y, w, h) Arena region for tracking
CARD_BASED_TRACKING = True  # Only track when cards have been played recently
MIN_ACTIVITY_FRAMES = 1 # Remove tracks after minimal movement in these frames
MIN_MOVEMENT = 1.0 # Less than 1,0 pixels per frame on average

# Troop verification model settings
# Use verification when candidate scores are within this threshold
TROOP_VERIFICATION_SCORE_THRESHOLD = 0.6
# Maximum candidates to verify before defaulting to highest score
TROOP_VERIFICATION_MAX_ATTEMPTS = 3
# Minimum confidence from troop model to accept match
TROOP_VERIFICATION_MIN_CONFIDENCE = 0.3

# Output settings
OUTPUT_DIR = './output_dataset/'
IMAGES_DIR = './output_dataset/images/'
LABELS_DIR = './output_dataset/labels/'
SAVE_IMAGES = False  # Set to False to disable image saving during testing

# YouTube URLs to process (or set TEST_VIDEO_PATH for local file)
YOUTUBE_URLS = [
    "https://www.youtube.com/watch?v=fSRN4bQcoDo&ab_channel=TVroyale"
]
# "https://www.youtube.com/watch?v=R3A17nCHrDg&ab_channel=Ryley-ClashRoyale"
# "https://www.youtube.com/watch?v=Px0O-NFvfx8&ab_channel=Ryley-ClashRoyale",
# "https://www.youtube.com/watch?v=R3A17nCHrDg&ab_channel=Ryley-ClashRoyale",

# For testing with local video file, set this path and YOUTUBE_URLS to empty list
TEST_VIDEO_PATH = None  # "test_video.mp4"

# Arena region for background color sampling (x, y, width, height)
# This should be a central area of the game arena
ARENA_SAMPLE_REGION = (437, 828, 50, 50)  # Center of 720x1280 frame (portrait)

# Building removal parameters
# Color difference threshold for background matching
BUILDING_BG_COLOR_THRESHOLD = 10.0
BUILDING_BG_MATCH_FRAMES = 3        # Frames of background matching before removal

# Card hand detection coordinates (for 720x1280 video - portrait)
# Ally hand positions (bottom of screen) - adjust these based on your video
ALLY_HAND_COORDS = [
    (97, 1116, 85, 86),   # Card 1
    (192, 1119, 85, 83),   # Card 2
    (286, 1119, 88, 84),   # Card 3
    (383, 1119, 79, 85),   # Card 4
]

# Enemy hand positions (top of screen) - adjust these based on your video
ENEMY_HAND_COORDS = [
    (95, 126, 88, 88),    # Card 1
    (191, 129, 83, 86),    # Card 2
    (288, 130, 87, 88),    # Card 3
    (383, 131, 83, 84),    # Card 4
]

ALLY_CARD_BAR_X = ALLY_HAND_COORDS[0][0]
ALLY_CARD_BAR_Y = ALLY_HAND_COORDS[0][1]
# Proper width to include all 4 cards
ALLY_CARD_BAR_WIDTH = (
    ALLY_HAND_COORDS[3][0] + ALLY_HAND_COORDS[3][2]) - ALLY_HAND_COORDS[0][0]
ALLY_CARD_BAR_HEIGHT = ALLY_HAND_COORDS[0][3]

ALLY_REGION = (ALLY_CARD_BAR_X, ALLY_CARD_BAR_Y,
               ALLY_CARD_BAR_WIDTH, ALLY_CARD_BAR_HEIGHT)

ENEMY_CARD_BAR_X = ENEMY_HAND_COORDS[0][0]
ENEMY_CARD_BAR_Y = ENEMY_HAND_COORDS[0][1]
# Proper width to include all 4 cards
ENEMY_CARD_BAR_WIDTH = (
    ENEMY_HAND_COORDS[3][0] + ENEMY_HAND_COORDS[3][2]) - ENEMY_HAND_COORDS[0][0]
ENEMY_CARD_BAR_HEIGHT = ENEMY_HAND_COORDS[0][3]

ENEMY_REGION = (ENEMY_CARD_BAR_X, ENEMY_CARD_BAR_Y,
                ENEMY_CARD_BAR_WIDTH, ENEMY_CARD_BAR_HEIGHT)

# Game Activity Detection Settings
# (x, y, w, h) Region where game timer is displayed
# (x, y, w, h) Region where game timer is displayed
# (x, y, w, h) Region where game timer is displayed
# (x, y, w, h) Region where game timer is displayed
ACTIVE_REGION = (26, 1199, 40, 45)  # (x, y, w, h) Region where game timer is displayed
# RGB color when timer is active (either black i.e. regular or orange i.e. overtime background)
ACTIVE_COLORS = [(160.67, 46.60, 154.42), (160.67, 70.60, 154.42)]
# Color difference threshold for detecting active based on elixer
ACTIVE_COLOR_THRESHOLD = 35.0
# Amount of frames in a row to detect if game is active or not.
ACTIVE_STABLE = 3
