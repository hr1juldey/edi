"""Stage 3: SAM Segmentation

This module implements SAM 2.1 automatic mask generation to find ALL objects,
then filters to only those overlapping with the color mask from Stage 2.
Each distinct object gets its own separate mask.
"""

import logging
import numpy as np
import cv2
import torch
from ultralytics import SAM
from typing import List


def segment_regions(image: np.ndarray, color_mask: np.ndarray,
                   min_area: int = 100,
                   color_overlap_threshold: float = 0.5) -> List[np.ndarray]:
    """
    Generate individual SAM masks for each object overlapping with color mask.

    Args:
        image: Original RGB image (H x W x 3), numpy array
        color_mask: Binary mask from Stage 2 (H x W), values 0 or 1
        min_area: Minimum mask area in pixels (filter noise)
        color_overlap_threshold: Minimum fraction of mask that must overlap color
                                 (0.5 = 50% of mask must be blue)

    Returns:
        List of binary masks (each H x W, dtype=np.uint8), one per object
    """
    logging.info("Starting SAM automatic segmentation")
    logging.info(f"Color mask coverage: {np.sum(color_mask > 0) / color_mask.size * 100:.2f}%")

    # Initialize SAM 2.1 in automatic mode
    sam_model = SAM("sam2.1_b.pt")
    if torch.cuda.is_available():
        sam_model.to('cuda')
        sam_model.half()

    try:
        # Run SAM automatic mask generation
        results = sam_model(image, task="segment")

        # Handle empty results
        if len(results) == 0 or results[0].masks is None:
            logging.warning("SAM found no objects in image")
            return []

        all_masks = results[0].masks.data
        logging.info(f"SAM generated {len(all_masks)} total masks")

        # Filter masks by color overlap
        filtered_masks = []

        for idx, mask_tensor in enumerate(all_masks):
            # Convert to numpy binary mask
            mask_np = mask_tensor.cpu().numpy()
            mask_binary = (mask_np > 0.5).astype(np.uint8)

            # Calculate area
            mask_area = np.sum(mask_binary)

            # Filter by minimum area
            if mask_area < min_area:
                continue

            # Calculate overlap with color mask
            overlap_pixels = np.sum((mask_binary > 0) & (color_mask > 0))

            # Avoid division by zero
            if mask_area == 0:
                continue

            overlap_fraction = overlap_pixels / mask_area

            # Keep mask if sufficient overlap
            if overlap_fraction >= color_overlap_threshold:
                filtered_masks.append(mask_binary)
                logging.debug(f"Mask {idx}: {mask_area} px, "
                            f"{overlap_fraction*100:.1f}% blue - KEPT")

        logging.info(f"Filtered to {len(filtered_masks)} masks with "
                    f"≥{color_overlap_threshold*100:.0f}% color overlap")

    except RuntimeError as e:
        if "out of memory" in str(e):
            logging.error("SAM OOM - try reducing image resolution")
            # Fallback: resize image and retry
            scale = 0.5
            height, width = image.shape[:2]
            new_height, new_width = int(height * scale), int(width * scale)

            resized_image = cv2.resize(image, (new_width, new_height))
            cv2.resize(color_mask, (new_width, new_height),
                                          interpolation=cv2.INTER_NEAREST)

            results = sam_model(resized_image, task="segment")

            if len(results) > 0 and results[0].masks is not None:
                all_masks = results[0].masks.data
                filtered_masks = []

                for mask_tensor in all_masks:
                    mask_np = mask_tensor.cpu().numpy()
                    mask_binary = (mask_np > 0.5).astype(np.uint8)

                    # Resize mask back to original size
                    mask_binary = cv2.resize(mask_binary, (width, height),
                                           interpolation=cv2.INTER_NEAREST)

                    mask_area = np.sum(mask_binary)
                    if mask_area < min_area:
                        continue

                    overlap_pixels = np.sum((mask_binary > 0) & (color_mask > 0))
                    overlap_fraction = overlap_pixels / mask_area if mask_area > 0 else 0

                    if overlap_fraction >= color_overlap_threshold:
                        filtered_masks.append(mask_binary)
            else:
                filtered_masks = []
        else:
            raise

    finally:
        # Always cleanup GPU memory
        del sam_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return filtered_masks