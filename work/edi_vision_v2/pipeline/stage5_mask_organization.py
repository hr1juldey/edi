"""Stage 5: Mask Organization

This module organizes CLIP-filtered masks into EntityMask objects with metadata.
Each mask remains SEPARATE with its own metadata.

CRITICAL: DO NOT MERGE MASKS! Each roof gets its own EntityMask object.
"""

import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class EntityMask:
    """Container for individual entity mask with metadata.

    CRITICAL: Each mask represents ONE entity and must stay separate.
    """
    mask: np.ndarray           # Binary mask (H x W), dtype=np.uint8
    entity_id: int             # Unique identifier for this entity
    similarity_score: float    # CLIP similarity score (0.0-1.0)
    bbox: Tuple[int, int, int, int]  # Bounding box (x_min, y_min, x_max, y_max)
    centroid: Tuple[float, float]    # Centroid coordinates (x, y)
    area: int                  # Number of pixels in mask
    dominant_color: Tuple[int, int, int]  # RGB color (most common in masked region)

    def __repr__(self):
        return (f"EntityMask(id={self.entity_id}, "
                f"similarity={self.similarity_score:.3f}, "
                f"area={self.area}px, "
                f"bbox={self.bbox})")


def extract_dominant_color(image: np.ndarray, mask: np.ndarray) -> Tuple[int, int, int]:
    """
    Extract the dominant color from the masked region.

    Args:
        image: RGB image (H x W x 3)
        mask: Binary mask (H x W)

    Returns:
        RGB tuple (r, g, b) of dominant color
    """
    # Get pixels in masked region
    masked_pixels = image[mask > 0]  # Shape: (N, 3)

    if len(masked_pixels) == 0:
        return (0, 0, 0)  # Black if empty

    # Calculate mean color
    mean_color = np.mean(masked_pixels, axis=0)
    dominant_color = tuple(int(c) for c in mean_color)

    return dominant_color


def organize_masks(image: np.ndarray,
                  filtered_masks: List[Tuple[np.ndarray, float]]) -> List[EntityMask]:
    """
    Organize CLIP-filtered masks into EntityMask objects with metadata.

    Args:
        image: Original RGB image (H x W x 3), numpy array
        filtered_masks: List of (mask, similarity_score) tuples from Stage 4

    Returns:
        List of EntityMask objects, one per input mask (SEPARATE, not merged!)
    """
    logging.info(f"Organizing {len(filtered_masks)} masks into EntityMask objects")

    if len(filtered_masks) == 0:
        logging.warning("No masks to organize")
        return []

    entity_masks = []

    for idx, (mask, similarity_score) in enumerate(filtered_masks):
        # Calculate bounding box
        y_coords, x_coords = np.where(mask > 0)

        if len(y_coords) == 0 or len(x_coords) == 0:
            logging.warning(f"Mask {idx}: Empty mask, skipping")
            continue

        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        bbox = (int(x_min), int(y_min), int(x_max), int(y_max))

        # Calculate centroid
        centroid_x = float(np.mean(x_coords))
        centroid_y = float(np.mean(y_coords))
        centroid = (centroid_x, centroid_y)

        # Calculate area
        area = int(np.sum(mask > 0))

        # Extract dominant color
        dominant_color = extract_dominant_color(image, mask)

        # Create EntityMask object
        entity_mask = EntityMask(
            mask=mask,
            entity_id=idx,
            similarity_score=similarity_score,
            bbox=bbox,
            centroid=centroid,
            area=area,
            dominant_color=dominant_color
        )

        entity_masks.append(entity_mask)

        logging.debug(f"Entity {idx}: {entity_mask}")

    # Sort by area (largest first)
    entity_masks.sort(key=lambda x: x.area, reverse=True)

    # Re-assign entity_id after sorting
    for new_id, entity_mask in enumerate(entity_masks):
        entity_mask.entity_id = new_id

    logging.info(f"Organized {len(entity_masks)} separate entity masks (sorted by area)")

    return entity_masks