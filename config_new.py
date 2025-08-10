"""
Configuration for Clash Royale Troop Detection Tool
"""

# Video processing settings
FRAME_SKIP = 30  # Process every 30th frame (1 FPS from 30 FPS video)
RESIZE_FACTOR = 1.0  # Keep original size
START_TIME_SECONDS = 10.0  # Start analysis at this time (in seconds)

# Detection settings
PIXEL_DIFF_THRESHOLD = 120  # Very high threshold (very insensitive to background movement)
MIN_OBJECT_SIZE = 1200  # Much larger minimum (filters out small movements)
MAX_OBJECT_SIZE = 5000  # Smaller maximum (focus on troop-sized objects)
MOTION_BLUR_KERNEL_SIZE = 11  # Larger kernel for more smoothing

# Tracking settings
MAX_TRACKING_FRAMES = 300  # How long to track an object
TRACKING_CONFIDENCE = 0.3  # Minimum confidence to keep tracking
MAX_TRACKED_OBJECTS = 10  # Maximum objects to track simultaneously
TRACKING_REGION = (98, 305, 527, 746)  # (x, y, w, h) Arena region for tracking
CARD_BASED_TRACKING = True  # Only track when cards have been played recently

# Output settings
OUTPUT_DIR = './output_dataset/'
IMAGES_DIR = './output_dataset/images/'
LABELS_DIR = './output_dataset/labels/'
SAVE_IMAGES = False  # Set to False to disable image saving during testing

# YouTube URLs to process (or set TEST_VIDEO_PATH for local file)
YOUTUBE_URLS = [
    "https://www.youtube.com/watch?v=R3A17nCHrDg&ab_channel=Ryley-ClashRoyale",
    
]
#"https://www.youtube.com/watch?v=kYD3v_VAhBI&t=508s&ab_channel=Ryley-ClashRoyale",

# For testing with local video file, set this path and YOUTUBE_URLS to empty list
TEST_VIDEO_PATH = None  # "test_video.mp4"

# Arena region for background color sampling (x, y, width, height)
# This should be a central area of the game arena
ARENA_SAMPLE_REGION = (437, 828, 50, 50)  # Center of 720x1280 frame (portrait)

# Card hand detection coordinates (for 720x1280 video - portrait)
# Ally hand positions (bottom of screen) - adjust these based on your video
ALLY_HAND_COORDS = [
    (99, 1098, 85, 87),   # Card 1
    (190, 1094, 86, 89),   # Card 2
    (285, 1097, 85, 88),   # Card 3
    (383, 1099, 81, 89),   # Card 4
]

# Enemy hand positions (top of screen) - adjust these based on your video  
ENEMY_HAND_COORDS = [
    (94, 120, 91, 86),    # Card 1
    (192, 122, 81, 84),    # Card 2
    (288, 119, 82, 90),    # Card 3
    (386, 120, 78, 89),    # Card 4
]
