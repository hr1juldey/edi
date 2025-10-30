# Stage 5: Mask Organization - Implementation Instructions

**File to create**: `pipeline/stage5_mask_organization.py`

**Reference**: See `/home/riju279/Documents/Code/Zonko/EDI/edi/docs/VISION_PIPELINE_RESEARCH.md` Section "Stage 5: Mask Organization"

---

## CRITICAL REQUIREMENT

**DO NOT MERGE MASKS!**

Each mask must remain SEPARATE with its own metadata. Return a list of individual `EntityMask` objects, NOT a single combined mask.

---

## Overview

This stage organizes the CLIP-filtered masks into structured `EntityMask` objects with rich metadata (bounding box, centroid, area, similarity score, color).

**Purpose**: Package each mask with metadata for downstream processing and validation

**Key Point**: Each roof gets its own `EntityMask` object - they stay SEPARATE

---

## Requirements

### 1. Data Structure: `EntityMask` Class

```python
from dataclasses import dataclass
from typing import Tuple
import numpy as np

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
```

### 2. Main Function: `organize_masks()`

```python
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
```

### 3. Processing Steps

**Step 1: Extract Metadata for Each Mask**

```python
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

logging.info(f"Organized {len(entity_masks)} separate entity masks")
```

**Step 2: Extract Dominant Color**

```python
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

    # Calculate mean color (simple approach)
    mean_color = np.mean(masked_pixels, axis=0)
    dominant_color = tuple(int(c) for c in mean_color)

    return dominant_color
```

**Alternative (median color)**:
```python
# Use median instead of mean for robustness to outliers
median_color = np.median(masked_pixels, axis=0)
dominant_color = tuple(int(c) for c in median_color)
```

**Step 3: Sort by Area (Largest First)**

```python
# Sort by area (largest roofs first)
entity_masks.sort(key=lambda x: x.area, reverse=True)

logging.info(f"Sorted entity masks by area (largest first)")
```

**Step 4: Return List of Separate Masks**

```python
return entity_masks  # List of individual EntityMask objects (NOT merged!)
```

---

## Complete Implementation

```python
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
```

---

## Edge Cases

### 1. No Masks Input
```python
if len(filtered_masks) == 0:
    logging.warning("No masks to organize")
    return []
```

### 2. Empty Mask
```python
if len(y_coords) == 0:
    logging.warning(f"Mask {idx}: Empty, skipping")
    continue
```

### 3. Single-Pixel Mask
Very small masks are still valid - include them with area=1.

---

## Testing

### Create `tests/test_stage5.py`

**Test Case 1**: Organize masks into EntityMask objects

```python
def test_organize_masks():
    """Test that masks are organized into EntityMask objects."""
    # Create synthetic masks
    image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)

    # Create 3 test masks
    mask1 = np.zeros((200, 200), dtype=np.uint8)
    mask1[50:100, 50:100] = 1

    mask2 = np.zeros((200, 200), dtype=np.uint8)
    mask2[120:150, 120:150] = 1

    mask3 = np.zeros((200, 200), dtype=np.uint8)
    mask3[10:30, 10:30] = 1

    filtered_masks = [(mask1, 0.8), (mask2, 0.6), (mask3, 0.9)]

    from pipeline.stage5_mask_organization import organize_masks, EntityMask
    entity_masks = organize_masks(image, filtered_masks)

    # Should get 3 separate EntityMask objects
    assert len(entity_masks) == 3

    # Each should be an EntityMask
    for entity_mask in entity_masks:
        assert isinstance(entity_mask, EntityMask)
        assert isinstance(entity_mask.mask, np.ndarray)
        assert isinstance(entity_mask.entity_id, int)
        assert isinstance(entity_mask.similarity_score, float)
        assert isinstance(entity_mask.bbox, tuple)
        assert len(entity_mask.bbox) == 4
        assert isinstance(entity_mask.centroid, tuple)
        assert len(entity_mask.centroid) == 2
        assert isinstance(entity_mask.area, int)
        assert entity_mask.area > 0
        assert isinstance(entity_mask.dominant_color, tuple)
        assert len(entity_mask.dominant_color) == 3
```

**Test Case 2**: Verify metadata correctness

```python
def test_metadata_correctness():
    """Test that metadata is calculated correctly."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[20:40, 20:40] = [100, 150, 200]  # Blue region

    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:40, 20:40] = 1

    from pipeline.stage5_mask_organization import organize_masks
    entity_masks = organize_masks(image, [(mask, 0.75)])

    assert len(entity_masks) == 1
    entity = entity_masks[0]

    # Verify bbox
    assert entity.bbox == (20, 20, 39, 39)

    # Verify centroid (center of 20:40 is 29.5)
    assert abs(entity.centroid[0] - 29.5) < 1.0
    assert abs(entity.centroid[1] - 29.5) < 1.0

    # Verify area
    assert entity.area == 20 * 20  # 400 pixels

    # Verify similarity score
    assert entity.similarity_score == 0.75

    # Verify dominant color (should be close to [100, 150, 200])
    assert abs(entity.dominant_color[0] - 100) < 5
    assert abs(entity.dominant_color[1] - 150) < 5
    assert abs(entity.dominant_color[2] - 200) < 5
```

**Test Case 3**: Sorted by area

```python
def test_sorted_by_area():
    """Test that EntityMasks are sorted by area (largest first)."""
    image = np.zeros((200, 200, 3), dtype=np.uint8)

    # Small mask (400 pixels)
    mask_small = np.zeros((200, 200), dtype=np.uint8)
    mask_small[10:30, 10:30] = 1

    # Large mask (2500 pixels)
    mask_large = np.zeros((200, 200), dtype=np.uint8)
    mask_large[50:100, 50:100] = 1

    # Medium mask (900 pixels)
    mask_medium = np.zeros((200, 200), dtype=np.uint8)
    mask_medium[120:150, 120:150] = 1

    filtered_masks = [(mask_small, 0.5), (mask_large, 0.6), (mask_medium, 0.7)]

    from pipeline.stage5_mask_organization import organize_masks
    entity_masks = organize_masks(image, filtered_masks)

    # Should be sorted by area: large, medium, small
    assert entity_masks[0].area == 2500
    assert entity_masks[1].area == 900
    assert entity_masks[2].area == 400

    # Entity IDs should be reassigned: 0, 1, 2
    assert entity_masks[0].entity_id == 0
    assert entity_masks[1].entity_id == 1
    assert entity_masks[2].entity_id == 2
```

**Test Case 4**: Empty masks list

```python
def test_empty_masks_list():
    """Test behavior with empty masks list."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    from pipeline.stage5_mask_organization import organize_masks
    entity_masks = organize_masks(image, [])

    assert entity_masks == []
```

**Test Case 5**: CRITICAL - Masks stay separate

```python
def test_masks_stay_separate():
    """CRITICAL TEST: Verify masks are NOT merged."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    # Create 5 separate masks
    masks = []
    for i in range(5):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[i*15:(i+1)*15, i*15:(i+1)*15] = 1
        masks.append((mask, 0.5 + i*0.05))

    from pipeline.stage5_mask_organization import organize_masks
    entity_masks = organize_masks(image, masks)

    # CRITICAL: Should get 5 separate EntityMask objects
    assert len(entity_masks) == 5, f"Expected 5 separate masks, got {len(entity_masks)}"

    # Each EntityMask should have a unique mask array
    for i in range(len(entity_masks)):
        for j in range(i+1, len(entity_masks)):
            # Masks should NOT be identical
            assert not np.array_equal(entity_masks[i].mask, entity_masks[j].mask), \
                f"Masks {i} and {j} are identical - they were merged!"
```

---

## Visual Validation

After implementation, create visualization:

```python
# Save visualization for supervisor review
import matplotlib.pyplot as plt

# Load test image and run full pipeline
test_img = cv2.imread("test_image.jpeg")
test_img_rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

# Run Stages 2-4
from pipeline.stage2_color_filter import color_prefilter
from pipeline.stage3_sam_segmentation import segment_regions
from pipeline.stage4_clip_filter import clip_filter_masks
from pipeline.stage5_mask_organization import organize_masks

color_mask = color_prefilter(test_img_rgb, "blue")
sam_masks = segment_regions(test_img_rgb, color_mask, min_area=500)
filtered_masks = clip_filter_masks(test_img_rgb, sam_masks, "tin roof", similarity_threshold=0.22)

# Stage 5: Organize
entity_masks = organize_masks(test_img_rgb, filtered_masks)

print(f"\nStage 5 Results:")
print(f"Organized {len(entity_masks)} separate entity masks")

# Create visualization showing each mask with metadata
num_masks = min(len(entity_masks), 10)  # Show top 10
cols = 5
rows = (num_masks + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
if rows == 1:
    axes = [axes]

for idx in range(rows * cols):
    row = idx // cols
    col = idx % cols

    if idx < num_masks:
        entity = entity_masks[idx]

        # Create overlay
        overlay = test_img_rgb.copy()
        overlay[entity.mask > 0] = overlay[entity.mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5

        # Draw bounding box
        x_min, y_min, x_max, y_max = entity.bbox
        cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (255, 255, 0), 2)

        # Draw centroid
        cx, cy = entity.centroid
        cv2.circle(overlay, (int(cx), int(cy)), 5, (255, 0, 0), -1)

        axes[row][col].imshow(overlay)
        axes[row][col].set_title(
            f"Entity {entity.entity_id}\n"
            f"Area: {entity.area}px\n"
            f"Score: {entity.similarity_score:.3f}\n"
            f"Color: RGB{entity.dominant_color}",
            fontsize=8
        )
        axes[row][col].axis('off')
    else:
        axes[row][col].axis('off')

plt.tight_layout()
plt.savefig("logs/stage5_entity_masks.png", dpi=150, bbox_inches='tight')
print("\nSaved: logs/stage5_entity_masks.png")

# Print entity details
print("\nTop 5 Entity Masks:")
for entity in entity_masks[:5]:
    print(f"  {entity}")
```

---

## Expected Results

For test_image.jpeg with 14 CLIP-filtered masks:

**Output**: 14 separate `EntityMask` objects
- Each with unique entity_id (0-13)
- Sorted by area (largest roofs first)
- Rich metadata: bbox, centroid, area, similarity, color

**Example output**:
```
Organized 14 separate entity masks

Top 5 Entity Masks:
  EntityMask(id=0, similarity=0.262, area=30961px, bbox=(234, 567, 389, 678))
  EntityMask(id=1, similarity=0.260, area=26429px, bbox=(123, 456, 234, 567))
  EntityMask(id=2, similarity=0.259, area=18768px, bbox=(456, 789, 567, 890))
  EntityMask(id=3, similarity=0.259, area=17285px, bbox=(678, 234, 789, 345))
  EntityMask(id=4, similarity=0.253, area=15123px, bbox=(345, 678, 456, 789))
```

---

## Acceptance Criteria

- [ ] Class `EntityMask` defined with all required fields
- [ ] Function `organize_masks()` implemented correctly
- [ ] All 5 test cases pass
- [ ] **CRITICAL TEST PASSES**: Masks stay separate (not merged)
- [ ] Visual validation: Shows individual masks with metadata
- [ ] Code has type hints and docstrings
- [ ] Each EntityMask has correct metadata (bbox, centroid, area, color)

---

## Report Format

After completion, report:

```
STAGE 5: Mask Organization - [COMPLETE/FAILED]

Implementation:
- File: pipeline/stage5_mask_organization.py
- Lines of code: XXX
- Class: EntityMask (dataclass)
- Function: organize_masks()

Test Results:
- Test Case 1 (Organize masks): [PASS/FAIL]
- Test Case 2 (Metadata correctness): [PASS/FAIL]
- Test Case 3 (Sorted by area): [PASS/FAIL]
- Test Case 4 (Empty masks): [PASS/FAIL]
- Test Case 5 (Masks stay separate - CRITICAL): [PASS/FAIL]

Results:
- Input: XX CLIP-filtered masks
- Output: XX separate EntityMask objects
- Largest mask: XXXX pixels
- Smallest mask: XXX pixels

Visual Validation:
- Saved to: logs/stage5_entity_masks.png
- Shows: Individual masks with metadata overlays

Top 3 Entities:
1. Entity 0: XXX px, score=0.XXX
2. Entity 1: XXX px, score=0.XXX
3. Entity 2: XXX px, score=0.XXX

Issues: [None / List any problems]

Status: Awaiting supervisor approval
```

---

**Begin implementation of Stage 5 now!**
