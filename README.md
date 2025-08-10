# Clash Royale Troop Annotation Tool

A computer vision tool for automatically detecting and tracking troops in Clash Royale gameplay videos, generating YOLOv5-compatible datasets for machine learning.

## Features

- **Real-time video processing** with visual feedback
- **Automatic troop detection** using pixel difference analysis
- **Arena color sampling** for accurate troop identification
- **Card placement tracking** from hand positions
- **Troop tracking** across multiple frames
- **YOLOv5 dataset generation** with bounding boxes and labels
- **Test mode** for local video files
- **YouTube video processing** support
- **AI-powered card detection** using Roboflow (with fallback to template matching)

## Quick Start

### 1. Setup Environment

```bash
# Activate virtual environment
# Windows:
.\activate.bat
# PowerShell:
.\activate.ps1
# Manual:
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Test Basic Components

```bash
python test_basic.py
```

This will verify that all core components are working correctly.

### 3. Run in Test Mode (Recommended for first use)

1. **Update config.py**:
   ```python
   test_mode = True
   test_video_path = "your_video.mp4"  # Change to your video file
   ```

2. **Run the tool**:
   ```bash
   python main.py
   ```

3. **Controls**:
   - **ESC**: Stop processing
   - **SPACE**: Pause/resume
   - **Mouse**: Resize window as needed

### 4. Run with YouTube Videos

1. **Update config.py**:
   ```python
   test_mode = False
   youtube_urls = [
       "https://www.youtube.com/watch?v=your_video_id",
       # Add more URLs as needed
   ]
   ```

2. **Run the tool**:
   ```bash
   python main.py
   ```

## Demo Mode

For testing with local video files:

```bash
python demo.py your_video.mp4
```

## Configuration

Edit `config.py` to customize:

- **Video processing**: Frame rate, frame skip, resize factor
- **Detection settings**: Pixel difference threshold, cluster sizes
- **Output settings**: Output directory, dataset format

### Roboflow Setup (Optional)

For enhanced card detection using AI models:

1. **Get your Roboflow API key**:
   - Go to [https://app.roboflow.com/account/api](https://app.roboflow.com/account/api)
   - Copy your API key

2. **Set environment variables**:
   ```bash
   # Windows:
   set ROBOFLOW_API_KEY=your_api_key_here
   set WORKSPACE_CARD_DETECTION=your_workspace_name
   
   # Linux/Mac:
   export ROBOFLOW_API_KEY=your_api_key_here
   export WORKSPACE_CARD_DETECTION=your_workspace_name
   ```

3. **Ensure your Roboflow workspace exists**:
   - The tool will use the "custom-workflow" workflow ID
   - Make sure your workspace has the appropriate card detection model

4. **Test Roboflow integration**:
   ```bash
   python test_roboflow.py
   ```

5. **Create a .env file** (alternative):
   ```
   ROBOFLOW_API_KEY=your_api_key_here
   WORKSPACE_CARD_DETECTION=your_workspace_name
   ```

**Note**: If Roboflow is not configured, the tool will fall back to template matching for card detection.

## Output

The tool generates a YOLOv5-compatible dataset:

```
output_dataset/
├── images/           # Frame images with active troops
├── labels/           # YOLO format labels (.txt files)
├── classes.txt       # Class mapping file
└── dataset_info.txt  # Processing statistics
```

## Troubleshooting

### No Video Display
- Ensure you're running in a graphical environment
- Check that OpenCV is properly installed
- Try running `python test_basic.py` first

### No Troops Detected
- Verify arena color sampling coordinates in `config.py`
- Check that the video contains Clash Royale gameplay
- Adjust pixel difference thresholds if needed

### Performance Issues
- Reduce frame rate in `config.py`
- Increase frame skip value
- Use smaller video resolution

## System Requirements

- Python 3.7+
- OpenCV 4.5+
- NumPy
- yt-dlp (for YouTube processing)
- Windows/Linux/macOS with GUI support

## Architecture

- **StreamHandler**: Video input management (YouTube/local)
- **ArenaColorDetector**: Arena color sampling and masking
- **PixelDifferenceDetector**: Troop placement detection
- **CardTracker**: Hand card monitoring
- **TroopTracker**: Multi-frame troop tracking
- **OutputWriter**: Dataset generation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 