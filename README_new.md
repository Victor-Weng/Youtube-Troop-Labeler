# Clash Royale Troop Detection Tool - Simplified

A clean, straightforward implementation for detecting and tracking troops in Clash Royale videos.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_new.txt
```

### 2. Configure Card Regions (One-time setup)

Before running the main tool, configure the card detection regions:

```bash
python config_ui.py
```

This opens a visual interface where you can:

-   **Drag rectangles** to move card regions
-   **Drag corners** to resize regions
-   **Press 'S'** to save configuration
-   **Press 'R'** to reset to defaults
-   **Press 'ESC'** to exit

The configuration is saved to `config_new.py` and used automatically by the main tool.

### 3. Run the Tool

Edit `config_new.py` to set your video source (YouTube URL or local file), then:

```bash
python main_new.py
```

## Configuration

Edit `config_new.py` to adjust:

-   Frame processing rate
-   Detection sensitivity
-   Output directories
-   Video sources

## Controls

-   **ESC**: Exit
-   **SPACE**: Pause/Resume

## Output

Creates a YOLO-format dataset in `./output_dataset/`:

-   `images/` - Frame images with detected troops
-   `labels/` - YOLO format labels
-   `classes.txt` - Class definitions
-   `summary.txt` - Processing statistics

## Architecture

**5 simple files:**

-   `main_new.py` - Main application (120 lines)
-   `detector.py` - Motion detection & tracking (200 lines)
-   `video_handler.py` - Video processing (80 lines)
-   `output.py` - Dataset generation (80 lines)
-   `config_ui.py` - Visual configuration tool (300 lines)

**Total: ~780 lines vs 1500+ in original**

## How It Works

1. **Motion Detection**: Uses OpenCV background subtraction to find moving objects
2. **Card Tracking**: Monitors card hands and detects when cards are played
3. **Correlation**: Links detected motion with specific card placements
4. **Tracking**: Tracks detected objects across frames using OpenCV trackers
5. **Output**: Saves frames with objects in YOLO format

Simple, effective, and maintainable.
