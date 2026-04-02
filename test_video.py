#!/usr/bin/env python3
"""
Test script for video processing functionality
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image

# Add the app directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_test_video():
    """Create a simple test video from existing images."""
    print("Creating test video...")

    # Get some test images
    test_dir = "Offroad_Segmentation_testImages/Color_Images"
    if not os.path.exists(test_dir):
        print("❌ Test images directory not found")
        return None

    image_files = os.listdir(test_dir)[:10]  # Use first 10 images
    if len(image_files) < 5:
        print("❌ Not enough test images")
        return None

    # Create video
    video_path = "test_video.mp4"
    first_image = Image.open(os.path.join(test_dir, image_files[0]))
    height, width = first_image.size[::-1]  # PIL gives (width, height)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 2.0, (width, height))  # 2 FPS

    for img_file in image_files:
        img_path = os.path.join(test_dir, img_file)
        img = cv2.imread(img_path)
        if img is not None:
            out.write(img)

    out.release()
    print(f"✅ Created test video: {video_path} ({len(image_files)} frames)")
    return video_path

def test_video_processing():
    """Test the video processing function."""
    print("Testing video processing...")

    # Create test video
    video_path = create_test_video()
    if not video_path:
        return False

    try:
        # Import the processing function
        from app import process_video

        # Process video
        output_path = "test_output.mp4"
        result = process_video(video_path, output_path)

        print("✅ Video processing completed!")
        print(f"   Frames processed: {result['total_frames']}")
        print(f"   FPS: {result['fps']}")
        print(f"   Resolution: {result['width']}x{result['height']}")
        print(f"   Output: {output_path}")

        # Check if output exists
        if os.path.exists(output_path):
            print("✅ Output video created successfully!")
            return True
        else:
            print("❌ Output video not found")
            return False

    except Exception as e:
        print(f"❌ Video processing failed: {e}")
        return False
    finally:
        # Clean up
        try:
            if os.path.exists(video_path):
                os.unlink(video_path)
            if os.path.exists(output_path):
                os.unlink(output_path)
        except:
            pass

if __name__ == "__main__":
    success = test_video_processing()
    sys.exit(0 if success else 1)