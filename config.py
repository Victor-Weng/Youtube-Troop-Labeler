"""
Configuration file for Clash Royale Troop Annotation Tool
"""

# Test mode - set to True to run with test video instead of YouTube
test_mode = False

# Test video path (used when test_mode = True)
test_video_path = "test_video.mp4"  # Change this to your test video file

# YouTube video URLs to process (used when test_mode = False)
youtube_urls = [
    "https://www.youtube.com/watch?v=kYD3v_VAhBI&t=508s&ab_channel=Ryley-ClashRoyale",
]

# Ally hand detection coordinates (list of rectangles)
# Each rectangle is (x, y, width, height) for card positions

res = 720

if res == 720:
    #conversion factor to 720p from 1080p boxing
    cf = 720/1080
elif res == 592:
    cf = 592/1080

# Arena color sampling coordinates (x, y, width, height)
# Sample from a central area of the arena to get baseline color
arena_color_sample_coords = tuple(int(v*cf) for v in [1128, 703, 28, 24])

ally_hand_coords = [
    tuple(int(v * cf) for v in (775, 922, 253, 71)),  # Card 1
    tuple(int(v * cf) for v in (844, 922, 253, 71)),  # Card 2
    tuple(int(v * cf) for v in (906, 922, 253, 71)),  # Card 3
    tuple(int(v * cf) for v in (977, 922, 253, 71)),  # Card 4
]


# Enemy hand detection coordinates
enemy_hand_coords = [
    tuple(int(v * cf) for v in (775, 98, 253, 71)),   # Card 1
    tuple(int(v * cf) for v in (844, 98, 253, 71)),   # Card 2
    tuple(int(v * cf) for v in (906, 98, 253, 71)),   # Card 3
    tuple(int(v * cf) for v in (977, 98, 253, 71)),   # Card 4
]

# Frame processing settings
frame_rate = 1  # FPS for frame sampling
frame_skip = 30   # Skip frames to achieve desired FPS (30 FPS video -> 1 FPS processing)

# Pixel difference detection settings
pixel_diff_threshold = 30      # Threshold for pixel difference detection
min_cluster_size = 100         # Minimum pixel cluster size to consider as troop
max_cluster_size = 10000       # Maximum pixel cluster size

# Tracking settings
tracker_type = 'CSRT'  # Options: 'CSRT', 'KCF', 'MOSSE'
tracking_confidence_threshold = 0.3
max_tracking_frames = 300  # Maximum frames to track a troop

# Stream handling settings
stream_retry_limit = 3
max_consecutive_failed_frames_before_retry = 10
stream_timeout_seconds = 30

# Output settings
output_dir = './output_dataset/'
images_dir = './output_dataset/images/'
labels_dir = './output_dataset/labels/'
classes_file = './output_dataset/classes.txt'

# Card recognition model settings
card_model_path = './models/cards/'
card_confidence_threshold = 0.7

# Video processing settings
video_width = 1280
video_height = 720
resize_factor = 1.0  # Scale factor for processing (1.0 = original size)

# Logging settings
log_level = 'INFO'
log_file = './troop_annotation.log'

# Performance settings
max_memory_usage_mb = 2048
enable_gpu = False  # Set to True if CUDA is available 