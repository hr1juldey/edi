"""Stage 1b: Color-Based Filtering

Filters YOLO-World detections by color using HSV analysis.

This solves the YOLO-World color limitation:
- YOLO-World can't detect "red vehicles" directly
- But it can detect "vehicles", then we filter by color

Usage:
    from pipeline.stage1b_color_filter import filter_boxes_by_color

    boxes = detect_entities_yolo_world(image, "vehicles")
    red_boxes = filter_boxes_by_color(image, boxes, "red")
"""

import logging
import numpy as np
import cv2
from typing import List, Tuple

from .stage1_yolo_world import DetectionBox


# HSV color ranges (Hue: 0-180, Saturation: 0-255, Value: 0-255)
# Format: (lower_bound, upper_bound) or tuple of ranges for wrap-around colors
COLOR_RANGES_HSV = {
    # Red wraps around 0/180 in HSV
    "red": [
        ((0, 100, 100), (10, 255, 255)),    # Lower red
        ((160, 100, 100), (180, 255, 255))  # Upper red
    ],
    # Primary colors
    "blue": [((100, 100, 100), (130, 255, 255))],
    "green": [((40, 50, 50), (80, 255, 255))],
    "yellow": [((20, 100, 100), (35, 255, 255))],
    "orange": [((10, 100, 100), (20, 255, 255))],

    # Secondary colors
    "purple": [((130, 50, 50), (160, 255, 255))],
    "violet": [((130, 50, 50), (160, 255, 255))],  # Same as purple
    "pink": [((150, 50, 100), (170, 255, 255))],
    "cyan": [((85, 100, 100), (95, 255, 255))],
    "magenta": [((140, 50, 50), (160, 255, 255))],

    # Browns and earth tones
    "brown": [((10, 50, 20), (20, 200, 150))],
    "tan": [((15, 40, 100), (25, 120, 200))],
    "beige": [((20, 20, 150), (30, 80, 220))],

    # Grayscale
    "white": [((0, 0, 200), (180, 30, 255))],
    "black": [((0, 0, 0), (180, 255, 50))],
    "gray": [((0, 0, 50), (180, 50, 200))],
    "grey": [((0, 0, 50), (180, 50, 200))],  # Same as gray
    "silver": [((0, 0, 120), (180, 30, 200))],

    # Other colors
    "turquoise": [((80, 100, 100), (90, 255, 255))],
    "teal": [((85, 100, 100), (95, 255, 200))],
    "indigo": [((110, 100, 100), (130, 255, 255))],
    "maroon": [((0, 100, 50), (10, 255, 150))],
    "navy": [((110, 100, 50), (130, 255, 150))],
    "olive": [((30, 50, 50), (40, 200, 150))],
    "gold": [((25, 100, 150), (35, 255, 255))],
    "cream": [((20, 20, 200), (30, 50, 255))],
}


def create_color_mask(
    region_hsv: np.ndarray,
    color_name: str
) -> np.ndarray:
    """
    Create a binary mask for pixels matching the target color.

    Args:
        region_hsv: HSV image region (H x W x 3)
        color_name: Target color name

    Returns:
        Binary mask (H x W) where 255 = color match, 0 = no match
    """
    if color_name not in COLOR_RANGES_HSV:
        logging.warning(f"Color '{color_name}' not in database")
        return np.zeros((region_hsv.shape[0], region_hsv.shape[1]), dtype=np.uint8)

    color_ranges = COLOR_RANGES_HSV[color_name]

    # Create masks for each range
    masks = []
    for lower, upper in color_ranges:
        mask = cv2.inRange(region_hsv, np.array(lower), np.array(upper))
        masks.append(mask)

    # Combine masks (OR operation)
    if len(masks) == 1:
        return masks[0]
    else:
        combined_mask = masks[0]
        for mask in masks[1:]:
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        return combined_mask


def analyze_color_match(
    image: np.ndarray,
    box: DetectionBox,
    target_color: str
) -> Tuple[float, np.ndarray]:
    """
    Analyze how well a box region matches the target color.

    Args:
        image: RGB image (H x W x 3)
        box: Detection box to analyze
        target_color: Target color name

    Returns:
        (match_percentage, color_mask)
        - match_percentage: 0-1, percentage of pixels matching color
        - color_mask: Binary mask of matching pixels
    """
    # Extract region
    x1, y1 = box.x, box.y
    x2, y2 = box.x + box.w, box.y + box.h

    # Ensure valid bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.shape[1], x2)
    y2 = min(image.shape[0], y2)

    # Extract region and convert to HSV
    region_rgb = image[y1:y2, x1:x2]

    if region_rgb.size == 0:
        return 0.0, np.zeros((1, 1), dtype=np.uint8)

    region_hsv = cv2.cvtColor(region_rgb, cv2.COLOR_RGB2HSV)

    # Create color mask
    color_mask = create_color_mask(region_hsv, target_color)

    # Calculate match percentage
    total_pixels = region_hsv.shape[0] * region_hsv.shape[1]
    matching_pixels = np.sum(color_mask > 0)
    match_percentage = matching_pixels / total_pixels if total_pixels > 0 else 0.0

    return match_percentage, color_mask


def filter_boxes_by_color(
    image: np.ndarray,
    boxes: List[DetectionBox],
    target_color: str,
    color_match_threshold: float = 0.25
) -> List[DetectionBox]:
    """
    Filter detection boxes by color using HSV analysis.

    Args:
        image: RGB image (H x W x 3)
        boxes: Detected boxes from YOLO-World
        target_color: Target color name (e.g., "red", "blue")
        color_match_threshold: Minimum percentage of pixels matching color (0-1)
                               Default 0.25 = 25% of box must match target color

    Returns:
        Filtered list of boxes that match the target color

    Examples:
        >>> boxes = detect_entities_yolo_world(image, "vehicles")
        >>> red_boxes = filter_boxes_by_color(image, boxes, "red", 0.25)
        >>> # Returns only boxes where 25%+ of pixels are red
    """
    if len(boxes) == 0:
        logging.info("No boxes to filter")
        return []

    if target_color not in COLOR_RANGES_HSV:
        logging.warning(
            f"Color '{target_color}' not in color database. "
            f"Available colors: {', '.join(sorted(COLOR_RANGES_HSV.keys()))}"
        )
        logging.warning("Returning all boxes without filtering")
        return boxes

    logging.info(f"Filtering {len(boxes)} boxes by color: {target_color}")
    logging.info(f"  Color match threshold: {color_match_threshold:.1%}")

    # Convert image to HSV once (optimization)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    filtered_boxes = []

    for i, box in enumerate(boxes):
        match_percentage, color_mask = analyze_color_match(image, box, target_color)

        if match_percentage >= color_match_threshold:
            filtered_boxes.append(box)
            logging.debug(
                f"  Box {i} ({box.label}): "
                f"{match_percentage:.1%} {target_color} → KEEP "
                f"(conf={box.confidence:.3f})"
            )
        else:
            logging.debug(
                f"  Box {i} ({box.label}): "
                f"{match_percentage:.1%} {target_color} → FILTER OUT "
                f"(below {color_match_threshold:.1%} threshold)"
            )

    logging.info(
        f"After color filtering: {len(filtered_boxes)}/{len(boxes)} boxes kept "
        f"({len(filtered_boxes)/len(boxes)*100:.0f}%)"
    )

    return filtered_boxes


def get_dominant_color(
    image: np.ndarray,
    box: DetectionBox,
    top_n: int = 3
) -> List[Tuple[str, float]]:
    """
    Get dominant colors in a box region.

    Useful for debugging and understanding why filtering failed.

    Args:
        image: RGB image
        box: Detection box
        top_n: Number of top colors to return

    Returns:
        List of (color_name, match_percentage) tuples, sorted by match

    Example:
        >>> colors = get_dominant_color(image, box)
        >>> # [("blue", 0.45), ("white", 0.30), ("black", 0.15)]
    """
    color_matches = []

    for color_name in COLOR_RANGES_HSV.keys():
        match_percentage, _ = analyze_color_match(image, box, color_name)
        if match_percentage > 0.05:  # Only include if >5%
            color_matches.append((color_name, match_percentage))

    # Sort by match percentage (descending)
    color_matches.sort(key=lambda x: x[1], reverse=True)

    return color_matches[:top_n]
