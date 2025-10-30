# Stage 3: SAM Segmentation - REVISED Implementation Instructions

**File to modify**: `pipeline/stage3_sam_segmentation.py`

**Reference**: See `/home/riju279/Documents/Code/Zonko/EDI/edi/docs/VISION_PIPELINE_RESEARCH.md` Section "Stage 3: SAM 2.1 Segmentation"

---

## CRITICAL CHANGE

**Problem with previous approach**: Using connected components + point prompts groups TOUCHING roofs into single masks. This violates the requirement: "all roofs will have separate masks".

**Solution**: Use **SAM Automatic Mask Generation** to detect ALL objects, then filter by color overlap.

---

## Overview

This stage uses SAM 2.1's automatic mask generation mode to find ALL objects in the image, then filters to only those that overlap with the blue color mask from Stage 2.

**Purpose**: Generate individual masks for EVERY blue object (each roof gets its own mask, even if touching)

**Key Point**: SAM automatic mode finds distinct objects based on visual boundaries, not just connected components.

---

## Requirements

### 1. Modified Function: `segment_regions()`

```python
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
```

### 2. Processing Steps

**Step 1: Initialize SAM in Automatic Mode**

```python
from ultralytics import SAM
import torch

# Load SAM 2.1 base model
sam_model = SAM("sam2.1_b.pt")
if torch.cuda.is_available():
    sam_model.to('cuda')
    sam_model.half()  # Half precision for speed

logging.info("Initialized SAM 2.1 in automatic mask generation mode")
```

**Step 2: Run SAM Automatic Mask Generation**

```python
# Run SAM in automatic mode (no prompts)
# This generates masks for ALL objects in the image
results = sam_model(image, task="segment")

# SAM returns results[0].masks.data as a torch tensor of shape (N, H, W)
# where N is the number of masks found
if len(results) == 0 or results[0].masks is None:
    logging.warning("SAM found no objects in image")
    return []

all_masks = results[0].masks.data  # Torch tensor (N, H, W)
logging.info(f"SAM generated {len(all_masks)} total masks")
```

**Step 3: Filter Masks by Color Overlap**

```python
# Filter to only masks that overlap significantly with the blue color mask
filtered_masks = []

for idx, mask_tensor in enumerate(all_masks):
    # Convert to numpy
    mask_np = mask_tensor.cpu().numpy()
    mask_binary = (mask_np > 0.5).astype(np.uint8)

    # Calculate area
    mask_area = np.sum(mask_binary)

    # Skip if too small
    if mask_area < min_area:
        continue

    # Calculate overlap with color mask
    overlap_pixels = np.sum((mask_binary > 0) & (color_mask > 0))
    overlap_fraction = overlap_pixels / mask_area

    # Keep mask if sufficient overlap with blue regions
    if overlap_fraction >= color_overlap_threshold:
        filtered_masks.append(mask_binary)
        logging.debug(f"Mask {idx}: {mask_area} pixels, "
                     f"{overlap_fraction*100:.1f}% blue overlap - KEPT")
    else:
        logging.debug(f"Mask {idx}: {mask_area} pixels, "
                     f"{overlap_fraction*100:.1f}% blue overlap - FILTERED")

logging.info(f"Filtered to {len(filtered_masks)} masks with ≥{color_overlap_threshold*100:.0f}% color overlap")
```

**Step 4: Cleanup and Return**

```python
# Unload SAM to free GPU memory
del sam_model
if torch.cuda.is_available():
    torch.cuda.empty_cache()

return filtered_masks
```

---

## Complete Revised Implementation

```python
"""Stage 3: SAM Segmentation

This module implements SAM 2.1 automatic mask generation to find ALL objects,
then filters to only those overlapping with the color mask from Stage 2.
Each distinct object gets its own separate mask.
"""

import logging
import numpy as np
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
    logging.info(f"Starting SAM automatic segmentation")
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
            import cv2
            scale = 0.5
            height, width = image.shape[:2]
            new_height, new_width = int(height * scale), int(width * scale)

            resized_image = cv2.resize(image, (new_width, new_height))
            resized_color_mask = cv2.resize(color_mask, (new_width, new_height),
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
```

---

## Edge Cases

### 1. No Objects Found

```python
if len(results) == 0 or results[0].masks is None:
    logging.warning("SAM found no objects in image")
    return []
```

### 2. All Masks Filtered Out

If no masks meet the color overlap threshold, return empty list and log warning.

### 3. Very Large Number of Masks

SAM can generate 100+ masks for complex scenes. This is OK - filtering will reduce them.

---

## Expected Improvements

**Previous approach (connected components)**:
- Generated: 8 masks (many roofs grouped together)
- Issue: Touching roofs treated as one mask

**Revised approach (automatic generation)**:
- Expected: 20-30+ masks (each roof separate)
- Benefit: SAM finds object boundaries, separates touching roofs

---

## Testing Updates

Update `tests/test_stage3.py` to reflect new behavior:

```python
def test_segment_regions():
    """Test that SAM automatic mode generates individual masks."""
    # ... load image and color mask ...

    masks = segment_regions(image_rgb, color_mask, min_area=500,
                           color_overlap_threshold=0.5)

    # Should generate MORE masks than connected components approach
    # Expect 20+ for test image with many roofs
    assert len(masks) >= 15, f"Expected ≥15 masks, got {len(masks)}"

    # Each mask should be binary
    for mask in masks:
        assert mask.dtype == np.uint8
        assert np.all((mask == 0) | (mask == 1))
```

---

## Performance Considerations

**Automatic mode is slower** than point prompts:
- Point prompts: ~180ms × 8 masks = 1.5s
- Automatic mode: ~3-7 seconds (generates all masks at once)

**Trade-off**: Slower but more accurate (gets all individual roofs)

**Optimization**: Use `sam2.1_t.pt` (tiny) for faster processing if needed.

---

## Visual Validation

After running revised implementation, expect to see:
- **20-30+ individual masks** instead of 8
- **Each roof has its own mask** even if roofs are touching
- **Sky still detected** (will be filtered in Stage 4 by CLIP)

---

## Command to Execute

Replace the current `pipeline/stage3_sam_segmentation.py` with the revised implementation above, then re-run validation:

```bash
python validate_stage3.py
```

Expected output:
```
SAM generated 150+ total masks
Filtered to 25-30 masks with ≥50% color overlap
```

---

## Acceptance Criteria

- [ ] Function uses SAM automatic mask generation (`task="segment"`)
- [ ] Filters masks by color overlap threshold
- [ ] Generates 20+ individual masks for test image
- [ ] Each touching roof gets separate mask
- [ ] Visual validation shows individual roofs highlighted
- [ ] Tests updated and passing
- [ ] Performance: <10 seconds (acceptable for automatic mode)

---

**Begin revised implementation of Stage 3 now!**
