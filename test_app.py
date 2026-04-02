#!/usr/bin/env python3
"""
Quick test script for the Flask app prediction functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import predict_segmentation, mask_to_color
from PIL import Image
import numpy as np

def test_prediction():
    """Test the prediction function with a sample image."""
    print("Testing segmentation prediction...")

    # Find a test image
    test_dir = "Offroad_Segmentation_testImages/Color_Images"
    if os.path.exists(test_dir):
        test_images = os.listdir(test_dir)
        if test_images:
            test_image_path = os.path.join(test_dir, test_images[0])
            print(f"Using test image: {test_image_path}")

            # Load and predict
            image = Image.open(test_image_path).convert("RGB")
            print(f"Image size: {image.size}")

            pred_mask = predict_segmentation(image)
            print(f"Prediction shape: {pred_mask.shape}")
            print(f"Unique classes in prediction: {np.unique(pred_mask)}")

            # Create colored mask
            color_mask = mask_to_color(pred_mask)
            print(f"Color mask shape: {color_mask.shape}")

            print("✅ Prediction test successful!")
            return True
        else:
            print("❌ No test images found")
            return False
    else:
        print("❌ Test images directory not found")
        return False

if __name__ == "__main__":
    success = test_prediction()
    sys.exit(0 if success else 1)