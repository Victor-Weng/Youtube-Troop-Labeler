#!/usr/bin/env python3
"""
Basic test script to verify components are working
"""

import cv2
import numpy as np
import logging
import sys

# Import our modules
from arena_color import ArenaColorDetector
from pixel_diff import PixelDifferenceDetector

def test_arena_color_detection():
    """Test arena color detection with a simple test image"""
    print("Testing Arena Color Detection...")
    
    # Create a test image with a known color
    test_image = np.full((480, 640, 3), [100, 150, 200], dtype=np.uint8)  # BGR format
    
    # Create detector
    detector = ArenaColorDetector()
    
    # Test sampling
    arena_color = detector.sample_arena_color(test_image)
    if arena_color is not None:
        print(f"✓ Arena color detected: BGR({arena_color[0]}, {arena_color[1]}, {arena_color[2]})")
        
        # Test color matching
        test_pixel = np.array([100, 150, 200])
        is_match = detector.is_arena_color(test_pixel)
        print(f"✓ Color matching: {is_match}")
        
        # Test mask generation
        mask = detector.get_arena_mask(test_image)
        print(f"✓ Mask generated: shape {mask.shape}, arena pixels: {np.sum(mask)}")
        
        return True
    else:
        print("✗ Failed to detect arena color")
        return False

def test_pixel_difference():
    """Test pixel difference detection"""
    print("\nTesting Pixel Difference Detection...")
    
    # Create test frames
    frame1 = np.full((480, 640, 3), [100, 150, 200], dtype=np.uint8)
    frame2 = frame1.copy()
    
    # Add a "troop" (different colored rectangle) to frame2
    frame2[200:250, 300:350] = [50, 100, 150]  # Different color
    
    # Create detector
    arena_detector = ArenaColorDetector()
    pixel_detector = PixelDifferenceDetector(arena_detector)
    
    # Set arena color
    arena_color = arena_detector.sample_arena_color(frame1)
    if arena_color is not None:
        pixel_detector.set_arena_color(arena_color)
        
        # Test change detection
        diff = pixel_detector.detect_changes(frame2)
        if diff is not None:
            print(f"✓ Pixel differences detected: shape {diff.shape}")
            
            # Test troop detection
            troops = pixel_detector.detect_troops(frame2)
            print(f"✓ Troops detected: {len(troops)}")
            
            if troops:
                for i, troop in enumerate(troops):
                    print(f"  Troop {i+1}: bbox={troop['bbox']}, confidence={troop['confidence']:.2f}")
            
            return True
        else:
            print("✗ Failed to detect pixel differences")
            return False
    else:
        print("✗ Failed to set arena color")
        return False

def test_opencv():
    """Test OpenCV functionality"""
    print("\nTesting OpenCV...")
    
    try:
        # Test basic OpenCV operations
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_image[25:75, 25:75] = [255, 255, 255]  # White square
        
        # Test drawing functions
        cv2.rectangle(test_image, (10, 10), (90, 90), (0, 255, 0), 2)
        cv2.putText(test_image, "Test", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        print("✓ OpenCV basic operations working")
        
        # Test window creation (this might fail in headless environments)
        try:
            cv2.namedWindow('Test', cv2.WINDOW_NORMAL)
            cv2.destroyWindow('Test')
            print("✓ OpenCV window operations working")
        except Exception as e:
            print(f"⚠ OpenCV window operations: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ OpenCV test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("BASIC COMPONENT TESTS")
    print("=" * 50)
    
    tests = [
        test_opencv,
        test_arena_color_detection,
        test_pixel_difference,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("=" * 50)
    
    if passed == total:
        print("✓ All tests passed! The system should work correctly.")
        return 0
    else:
        print("✗ Some tests failed. Check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 