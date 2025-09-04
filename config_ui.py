#!/usr/bin/env python3
"""
Configuration UI for Card Region Setup
Run this separately to configure card detection regions visually
"""

import cv2
import numpy as np
import json
import os
import sys
from typing import List, Tuple, Dict, Optional
import config_new as config
from video_handler import VideoHandler


class CardRegionConfigurator:
    def __init__(self):
        self.frame = None
        self.original_frame = None
        self.regions = {
            'ally': list(config.ALLY_HAND_COORDS),
            'enemy': list(config.ENEMY_HAND_COORDS),
            'arena': config.ARENA_SAMPLE_REGION,
            'tracking': getattr(config, 'TRACKING_REGION', (50, 150, 620, 950)),
            'timer': getattr(config, 'ACTIVE_REGION', (310, 50, 100, 40))
        }
        self.dragging = False
        self.drag_region = None
        self.drag_handle = None  # 'move', 'resize_br', 'resize_tl'
        self.drag_start = None
        self.drag_offset = (0, 0)

        # Colors for different regions
        self.colors = {
            'ally': (255, 0, 0),      # Blue
            'enemy': (0, 0, 255),     # Red
            'arena': (0, 255, 0),     # Green
            'tracking': (0, 255, 255),  # Yellow
            'timer': (255, 0, 255)    # Magenta
        }

        self.window_name = "Card Region Configuration"
        self.help_text = [
            "Instructions:",
            "- Drag rectangles to move them",
            "- Drag corners to resize",
            "- Blue: Ally cards, Red: Enemy cards",
            "- Green: Arena sampling, Yellow: Tracking zone",
            "- Magenta: Timer region (for game activity detection)",
            "- Press 'S' to save, 'R' to reset, 'ESC' to exit"
        ]

    def load_frame_from_video(self, time_seconds: float = 15.0) -> bool:
        """Load a frame from the configured video source at a specific time"""
        try:
            video_handler = VideoHandler()

            # Try local file first, then YouTube
            if config.TEST_VIDEO_PATH:
                print(
                    f"Loading frame at {time_seconds}s from local video: {config.TEST_VIDEO_PATH}")
                cap = cv2.VideoCapture(config.TEST_VIDEO_PATH)
                if not cap.isOpened():
                    print(
                        f"Could not open video file: {config.TEST_VIDEO_PATH}")
                    return False
            elif config.YOUTUBE_URLS:
                print(
                    f"Loading frame at {time_seconds}s from YouTube video...")
                url = config.YOUTUBE_URLS[0]

                # Get stream URL using the correct method name
                stream_url = video_handler.get_youtube_stream_url(url)
                if not stream_url:
                    print("Failed to get stream URL")
                    return False

                cap = cv2.VideoCapture(stream_url)
                if not cap.isOpened():
                    print("Failed to open video stream")
                    return False
            else:
                print("No video source configured in config_new.py")
                return False

            # Seek to the specific time position (much more reliable than frame seeking)
            print(f"Seeking to {time_seconds} seconds...")
            # Convert to milliseconds
            cap.set(cv2.CAP_PROP_POS_MSEC, time_seconds * 1000)

            # Read the frame at that position
            ret, frame = cap.read()
            cap.release()

            if not ret or frame is None:
                print(f"Failed to read frame at {time_seconds} seconds")
                return False

            # Resize frame to 720x1280 if needed (portrait)
            height, width = frame.shape[:2]
            if width != 720 or height != 1280:
                frame = cv2.resize(frame, (720, 1280))
                print(f"Resized frame from {width}x{height} to 720x1280")

            self.frame = frame.copy()
            self.original_frame = frame.copy()
            print(f"Successfully loaded frame at {time_seconds} seconds")
            return True

        except Exception as e:
            print(f"Error loading frame: {e}")
            return False

    def point_in_rect(self, point: Tuple[int, int], rect: Tuple[int, int, int, int]) -> bool:
        """Check if point is inside rectangle"""
        x, y = point
        rx, ry, rw, rh = rect
        return rx <= x <= rx + rw and ry <= y <= ry + rh

    def get_resize_handle(self, point: Tuple[int, int], rect: Tuple[int, int, int, int]) -> Optional[str]:
        """Check if point is on a resize handle"""
        x, y = point
        rx, ry, rw, rh = rect
        handle_size = 10

        # Top-left corner
        if abs(x - rx) < handle_size and abs(y - ry) < handle_size:
            return 'resize_tl'
        # Bottom-right corner
        if abs(x - (rx + rw)) < handle_size and abs(y - (ry + rh)) < handle_size:
            return 'resize_br'

        return None

    def find_clicked_region(self, point: Tuple[int, int]) -> Tuple[Optional[str], Optional[int], Optional[str]]:
        """Find which region was clicked and what type of interaction"""
        x, y = point

        # Check arena region first
        arena_rect = self.regions['arena']
        handle = self.get_resize_handle(point, arena_rect)
        if handle:
            return 'arena', 0, handle
        if self.point_in_rect(point, arena_rect):
            return 'arena', 0, 'move'

        # Check tracking region
        tracking_rect = self.regions['tracking']
        handle = self.get_resize_handle(point, tracking_rect)
        if handle:
            return 'tracking', 0, handle
        if self.point_in_rect(point, tracking_rect):
            return 'tracking', 0, 'move'

        # Check timer region
        timer_rect = self.regions['timer']
        handle = self.get_resize_handle(point, timer_rect)
        if handle:
            return 'timer', 0, handle
        if self.point_in_rect(point, timer_rect):
            return 'timer', 0, 'move'

        # Check ally regions
        for i, rect in enumerate(self.regions['ally']):
            handle = self.get_resize_handle(point, rect)
            if handle:
                return 'ally', i, handle
            if self.point_in_rect(point, rect):
                return 'ally', i, 'move'

        # Check enemy regions
        for i, rect in enumerate(self.regions['enemy']):
            handle = self.get_resize_handle(point, rect)
            if handle:
                return 'enemy', i, handle
            if self.point_in_rect(point, rect):
                return 'enemy', i, 'move'

        return None, None, None

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for dragging and resizing"""
        if event == cv2.EVENT_LBUTTONDOWN:
            region_type, region_idx, handle = self.find_clicked_region((x, y))
            if region_type:
                self.dragging = True
                self.drag_region = (region_type, region_idx)
                self.drag_handle = handle
                self.drag_start = (x, y)

                if handle == 'move':
                    if region_type == 'arena':
                        rect = self.regions['arena']
                    elif region_type == 'tracking':
                        rect = self.regions['tracking']
                    elif region_type == 'timer':
                        rect = self.regions['timer']
                    else:
                        rect = self.regions[region_type][region_idx]
                    self.drag_offset = (x - rect[0], y - rect[1])

        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            if self.drag_region:
                region_type, region_idx = self.drag_region

                if region_type == 'arena':
                    rect = list(self.regions['arena'])
                elif region_type == 'tracking':
                    rect = list(self.regions['tracking'])
                elif region_type == 'timer':
                    rect = list(self.regions['timer'])
                else:
                    rect = list(self.regions[region_type][region_idx])

                if self.drag_handle == 'move':
                    # Move the entire rectangle
                    rect[0] = x - self.drag_offset[0]
                    rect[1] = y - self.drag_offset[1]
                elif self.drag_handle == 'resize_tl':
                    # Resize from top-left corner
                    old_x, old_y, old_w, old_h = rect
                    new_w = old_w + (old_x - x)
                    new_h = old_h + (old_y - y)
                    if new_w > 10 and new_h > 10:  # Minimum size
                        rect[0] = x
                        rect[1] = y
                        rect[2] = new_w
                        rect[3] = new_h
                elif self.drag_handle == 'resize_br':
                    # Resize from bottom-right corner
                    new_w = x - rect[0]
                    new_h = y - rect[1]
                    if new_w > 10 and new_h > 10:  # Minimum size
                        rect[2] = new_w
                        rect[3] = new_h

                # Update the region
                if region_type == 'arena':
                    self.regions['arena'] = tuple(rect)
                elif region_type == 'tracking':
                    self.regions['tracking'] = tuple(rect)
                elif region_type == 'timer':
                    self.regions['timer'] = tuple(rect)
                else:
                    self.regions[region_type][region_idx] = tuple(rect)

                self.update_display()

        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
            self.drag_region = None
            self.drag_handle = None

    def draw_regions(self):
        """Draw all regions on the frame"""
        self.frame = self.original_frame.copy()

        # Draw arena region
        x, y, w, h = self.regions['arena']
        cv2.rectangle(self.frame, (x, y), (x + w, y + h),
                      self.colors['arena'], 2)
        cv2.putText(self.frame, "Arena", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['arena'], 2)

        # Draw resize handles for arena
        cv2.circle(self.frame, (x, y), 5, self.colors['arena'], -1)  # Top-left
        cv2.circle(self.frame, (x + w, y + h), 5,
                   self.colors['arena'], -1)  # Bottom-right

        # Draw tracking region
        x, y, w, h = self.regions['tracking']
        cv2.rectangle(self.frame, (x, y), (x + w, y + h),
                      self.colors['tracking'], 2)
        cv2.putText(self.frame, "Tracking Zone", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['tracking'], 2)

        # Draw resize handles for tracking region
        cv2.circle(self.frame, (x, y), 5,
                   self.colors['tracking'], -1)  # Top-left
        cv2.circle(self.frame, (x + w, y + h), 5,
                   self.colors['tracking'], -1)  # Bottom-right

        # Draw timer region
        x, y, w, h = self.regions['timer']
        cv2.rectangle(self.frame, (x, y), (x + w, y + h),
                      self.colors['timer'], 2)
        cv2.putText(self.frame, "Timer", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['timer'], 2)

        # Draw resize handles for timer region
        cv2.circle(self.frame, (x, y), 5, self.colors['timer'], -1)  # Top-left
        cv2.circle(self.frame, (x + w, y + h), 5,
                   self.colors['timer'], -1)  # Bottom-right

        # Draw ally regions
        for i, (x, y, w, h) in enumerate(self.regions['ally']):
            cv2.rectangle(self.frame, (x, y), (x + w, y + h),
                          self.colors['ally'], 2)
            cv2.putText(self.frame, f"Ally {i+1}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['ally'], 1)
            # Resize handles
            cv2.circle(self.frame, (x, y), 4, self.colors['ally'], -1)
            cv2.circle(self.frame, (x + w, y + h), 4, self.colors['ally'], -1)

        # Draw enemy regions
        for i, (x, y, w, h) in enumerate(self.regions['enemy']):
            cv2.rectangle(self.frame, (x, y), (x + w, y + h),
                          self.colors['enemy'], 2)
            cv2.putText(self.frame, f"Enemy {i+1}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['enemy'], 1)
            # Resize handles
            cv2.circle(self.frame, (x, y), 4, self.colors['enemy'], -1)
            cv2.circle(self.frame, (x + w, y + h), 4, self.colors['enemy'], -1)

        # Draw help text
        for i, text in enumerate(self.help_text):
            cv2.putText(self.frame, text, (10, 30 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def update_display(self):
        """Update the display window"""
        self.draw_regions()
        cv2.imshow(self.window_name, self.frame)

    def save_configuration(self) -> bool:
        """Save current configuration to config_new.py"""
        try:
            # Read current config file
            with open('config_new.py', 'r') as f:
                lines = f.readlines()

            # Find and update the relevant lines
            updated_lines = []
            skip_until = None

            for i, line in enumerate(lines):
                if 'ARENA_SAMPLE_REGION = ' in line:
                    updated_lines.append(
                        f"ARENA_SAMPLE_REGION = {self.regions['arena']}  # Center of 720x1280 frame (portrait)\n")
                elif 'TRACKING_REGION = ' in line:
                    updated_lines.append(
                        f"TRACKING_REGION = {self.regions['tracking']}  # (x, y, w, h) Arena region for tracking\n")
                elif 'ACTIVE_REGION = ' in line:
                    updated_lines.append(
                        f"ACTIVE_REGION = {self.regions['timer']}  # (x, y, w, h) Region where game timer is displayed\n")
                elif 'ALLY_HAND_COORDS = [' in line:
                    updated_lines.append("ALLY_HAND_COORDS = [\n")
                    for j, coord in enumerate(self.regions['ally']):
                        updated_lines.append(f"    {coord},   # Card {j+1}\n")
                    updated_lines.append("]\n")
                    skip_until = ']'
                elif 'ENEMY_HAND_COORDS = [' in line:
                    updated_lines.append("ENEMY_HAND_COORDS = [\n")
                    for j, coord in enumerate(self.regions['enemy']):
                        updated_lines.append(f"    {coord},    # Card {j+1}\n")
                    updated_lines.append("]\n")
                    skip_until = ']'
                elif skip_until and skip_until in line:
                    skip_until = None
                    continue
                elif skip_until:
                    continue
                else:
                    updated_lines.append(line)

            # Write back to file
            with open('config_new.py', 'w') as f:
                f.writelines(updated_lines)

            print("Configuration saved to config_new.py!")
            return True

        except Exception as e:
            print(f"Error saving configuration: {e}")
            return False

    def reset_to_defaults(self):
        """Reset regions to default values"""
        self.regions = {
            'ally': [
                (160, 1200, 80, 60),
                (250, 1200, 80, 60),
                (340, 1200, 80, 60),
                (430, 1200, 80, 60)
            ],
            'enemy': [
                (160, 50, 80, 60),
                (250, 50, 80, 60),
                (340, 50, 80, 60),
                (430, 50, 80, 60)
            ],
            'arena': (360, 640, 50, 50),
            'tracking': (50, 150, 620, 950)
        }
        self.update_display()
        print("Reset to default configuration")

    def run(self, time_seconds: float = 15.0):
        """Main configuration loop"""
        print("Card Region Configurator")
        print("=" * 40)

        # Try loading frame, with fallback to earlier times if needed
        loaded = False
        # Try different time positions
        attempts = [time_seconds, 10.0, 5.0, 1.0]

        for attempt_time in attempts:
            print(f"Attempting to load frame at {attempt_time} seconds...")
            if self.load_frame_from_video(attempt_time):
                loaded = True
                print(f"Successfully loaded frame at {attempt_time} seconds")
                break
            else:
                print(f"Failed to load frame at {attempt_time} seconds")

        if not loaded:
            print("Failed to load any video frame. Please check your configuration.")
            return False

        # Setup window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 720, 1280)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        # Initial display
        self.update_display()

        print("\nConfiguration UI loaded. Use mouse to adjust regions.")
        print("Press 'S' to save, 'R' to reset, 'ESC' to exit")

        # Main loop
        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                break
            elif key == ord('s') or key == ord('S'):
                if self.save_configuration():
                    print("Configuration saved successfully!")
                else:
                    print("Failed to save configuration!")
            elif key == ord('r') or key == ord('R'):
                self.reset_to_defaults()

        cv2.destroyAllWindows()
        return True


def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        try:
            time_seconds = float(sys.argv[1])
        except ValueError:
            print("Invalid time value. Using default 15 seconds.")
            time_seconds = 15.0
    else:
        time_seconds = 15.0

    print(f"Starting Card Region Configurator (Time: {time_seconds}s)")

    configurator = CardRegionConfigurator()
    configurator.run(time_seconds)


if __name__ == "__main__":
    main()
