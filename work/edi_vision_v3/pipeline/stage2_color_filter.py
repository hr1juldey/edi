"""Stage 2: Color Pre-Filtering

This module implements a fast HSV-based color filter to narrow down the search space
before running expensive SAM segmentation. The purpose is to detect ALL regions matching
a specified color (e.g., "blue") in <100ms.
"""

import logging
import numpy as np
import cv2


# Define color ranges in HSV format (H: 0-180, S: 0-255, V: 0-255)
color_ranges = {
    "blue": [(90, 50, 50), (130, 255, 255)],
    "green": [(40, 40, 40), (80, 255, 255)],
    "red": [[(0, 50, 50), (10, 255, 255)],      # Red wraps around hue
            [(170, 50, 50), (180, 255, 255)]],
    "yellow": [(20, 50, 50), (30, 255, 255)],
    "orange": [(10, 50, 50), (20, 255, 255)],
    "white": [(0, 0, 200), (180, 30, 255)],
    "black": [(0, 0, 0), (180, 255, 30)],
    "gray": [(0, 0, 50), (180, 50, 200)]
}


def color_prefilter(image: np.ndarray, color_name: str) -> np.ndarray:
    """
    Create a binary mask of all regions matching the specified color.

    Args:
        image: RGB image (H x W x 3), numpy array
        color_name: Color to filter (e.g., "blue", "red", "green")

    Returns:
        Binary mask (H x W) where 1 = color match, dtype=np.uint8
    """
    logging.info(f"Starting color prefilter for color: {color_name}")
    
    # Convert image from RGB to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Initialize mask
    mask = None
    
    # Handle different color ranges
    if color_name not in color_ranges:
        logging.warning(f"Color '{color_name}' not in color_ranges. Returning mask of all ones.")
        # Return mask of all ones (no filtering) for unknown colors
        return np.ones((image.shape[0], image.shape[1]), dtype=np.uint8)
    
    color_range = color_ranges[color_name]
    
    if color_name == "red":
        # Red has dual ranges
        mask1 = cv2.inRange(hsv_image, color_range[0][0], color_range[0][1])
        mask2 = cv2.inRange(hsv_image, color_range[1][0], color_range[1][1])
        mask = cv2.bitwise_or(mask1, mask2)
    else:
        # Single range for other colors
        lower, upper = color_range
        mask = cv2.inRange(hsv_image, lower, upper)
    
    # Morphological cleanup
    # Create elliptical kernel for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # Close small holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # Remove noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Convert to binary (0 and 1) instead of (0 and 255)
    mask = (mask > 0).astype(np.uint8)
    
    logging.info(f"Color prefilter completed. Mask shape: {mask.shape}, "
                 f"Blue pixels: {np.sum(mask > 0)}, Coverage: {np.sum(mask > 0) / mask.size * 100:.2f}%")
    
    return mask