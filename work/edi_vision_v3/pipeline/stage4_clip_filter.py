"""Stage 4: CLIP Semantic Filtering

This module uses CLIP to filter masks based on semantic similarity to target entity.
Keeps only masks that match the target description (e.g., "tin roof").
"""

import logging
import numpy as np
import torch
import open_clip
from PIL import Image
from typing import List, Tuple


def clip_filter_masks(image: np.ndarray,
                     masks: List[np.ndarray],
                     target_description: str,
                     similarity_threshold: float = 0.22) -> List[Tuple[np.ndarray, float]]:
    """
    Filter masks using CLIP semantic similarity.

    Args:
        image: Original RGB image (H x W x 3), numpy array
        masks: List of binary masks from Stage 3
        target_description: Text description of target entity (e.g., "tin roof", "blue roof")
        similarity_threshold: Minimum CLIP similarity score to keep mask (0.22 = 22%)

    Returns:
        List of tuples: [(mask, similarity_score), ...] for masks above threshold
        Sorted by similarity score (highest first)
    """
    logging.info(f"Starting CLIP filtering for '{target_description}'")
    logging.info(f"Processing {len(masks)} masks with threshold {similarity_threshold:.2f}")

    if len(masks) == 0:
        logging.warning("No masks to filter")
        return []

    # Initialize CLIP model
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32',
        pretrained='openai'
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    logging.info(f"Initialized CLIP model on {device}")

    try:
        # Encode target text
        text = tokenizer([target_description]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Process each mask
        filtered_results = []

        for idx, mask in enumerate(masks):
            # Extract bounding box
            y_coords, x_coords = np.where(mask > 0)

            if len(y_coords) == 0 or len(x_coords) == 0:
                logging.warning(f"Mask {idx}: Empty mask, skipping")
                continue

            y_min, y_max = y_coords.min(), y_coords.max()
            x_min, x_max = x_coords.min(), x_coords.max()

            # Crop image and apply mask
            cropped_image = image[y_min:y_max+1, x_min:x_max+1]
            cropped_mask = mask[y_min:y_max+1, x_min:x_max+1]

            masked_region = cropped_image.copy()
            masked_region[cropped_mask == 0] = [128, 128, 128]  # Gray background

            # Convert to PIL and preprocess
            pil_image = Image.fromarray(masked_region)
            image_tensor = preprocess(pil_image).unsqueeze(0).to(device)

            # Encode image
            with torch.no_grad():
                image_features = model.encode_image(image_tensor)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Calculate similarity
            similarity = (image_features @ text_features.T).item()

            # Filter by threshold
            if similarity >= similarity_threshold:
                filtered_results.append((mask, similarity))
                logging.debug(f"Mask {idx}: similarity {similarity:.3f} - KEPT")
            else:
                logging.debug(f"Mask {idx}: similarity {similarity:.3f} - FILTERED")

        # Sort by similarity (highest first)
        filtered_results.sort(key=lambda x: x[1], reverse=True)

        logging.info(f"Filtered to {len(filtered_results)} masks with "
                    f"similarity ≥{similarity_threshold:.2f}")

    finally:
        # Cleanup
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return filtered_results