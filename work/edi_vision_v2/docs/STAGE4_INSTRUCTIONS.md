# Stage 4: CLIP Semantic Filtering - Implementation Instructions

**File to create**: `pipeline/stage4_clip_filter.py`

**Reference**: See `/home/riju279/Documents/Code/Zonko/EDI/edi/docs/VISION_PIPELINE_RESEARCH.md` Section "Stage 4: CLIP Semantic Filtering"

---

## Overview

This stage uses CLIP (Contrastive Language-Image Pre-training) to semantically filter masks from Stage 3, keeping only those that match the target entity description (e.g., "tin roof", "building roof").

**Purpose**: Filter out non-target objects (sky, trees, vehicles) and keep only semantically relevant masks

**Key Point**: Use threshold-based filtering (not top-k) to keep ALL masks above similarity threshold

---

## Requirements

### 1. Main Function: `clip_filter_masks()`

```python
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
```

### 2. Processing Steps

**Step 1: Initialize CLIP Model**

```python
import open_clip
import torch
from PIL import Image

# Load OpenCLIP model (ViT-B/32 is good balance of speed/accuracy)
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32',
    pretrained='openai'
)

# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()  # Set to evaluation mode

# Load tokenizer
tokenizer = open_clip.get_tokenizer('ViT-B-32')

logging.info(f"Initialized CLIP model on {device}")
```

**Step 2: Encode Target Text**

```python
# Tokenize target description
text = tokenizer([target_description]).to(device)

# Encode text to get text features
with torch.no_grad():
    text_features = model.encode_text(text)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # Normalize

logging.info(f"Encoded target description: '{target_description}'")
```

**Step 3: Process Each Mask**

```python
filtered_results = []

for idx, mask in enumerate(masks):
    # Extract masked region from image
    # Create a crop of the image using the mask bounding box
    y_coords, x_coords = np.where(mask > 0)

    if len(y_coords) == 0 or len(x_coords) == 0:
        logging.warning(f"Mask {idx}: Empty mask, skipping")
        continue

    # Get bounding box
    y_min, y_max = y_coords.min(), y_coords.max()
    x_min, x_max = x_coords.min(), x_coords.max()

    # Crop image to bounding box
    cropped_image = image[y_min:y_max+1, x_min:x_max+1]

    # Apply mask to cropped region (set non-masked pixels to white/gray)
    cropped_mask = mask[y_min:y_max+1, x_min:x_max+1]
    masked_region = cropped_image.copy()
    masked_region[cropped_mask == 0] = [128, 128, 128]  # Gray background

    # Convert to PIL Image
    pil_image = Image.fromarray(masked_region)

    # Preprocess for CLIP
    image_tensor = preprocess(pil_image).unsqueeze(0).to(device)

    # Encode image to get image features
    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    # Calculate cosine similarity
    similarity = (image_features @ text_features.T).item()

    # Keep if above threshold
    if similarity >= similarity_threshold:
        filtered_results.append((mask, similarity))
        logging.debug(f"Mask {idx}: similarity {similarity:.3f} - KEPT")
    else:
        logging.debug(f"Mask {idx}: similarity {similarity:.3f} - FILTERED")

# Sort by similarity (highest first)
filtered_results.sort(key=lambda x: x[1], reverse=True)

logging.info(f"Filtered {len(masks)} masks to {len(filtered_results)} "
            f"with similarity ≥{similarity_threshold:.2f}")
```

**Step 4: Cleanup and Return**

```python
# Cleanup CLIP model
del model
if torch.cuda.is_available():
    torch.cuda.empty_cache()

return filtered_results
```

---

## Complete Implementation

```python
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
```

---

## Edge Cases

### 1. No Masks Input
```python
if len(masks) == 0:
    logging.warning("No masks to filter")
    return []
```

### 2. Empty Mask
```python
y_coords, x_coords = np.where(mask > 0)
if len(y_coords) == 0:
    logging.warning(f"Mask {idx}: Empty, skipping")
    continue
```

### 3. All Masks Filtered Out
If no masks exceed threshold, return empty list and log warning.

### 4. Very Small Masked Region
CLIP should handle small crops, but very tiny regions (<10x10 pixels) may give unreliable scores.

---

## Threshold Tuning

**Default threshold: 0.22 (22%)**

Guidelines:
- **0.15-0.20**: Very lenient (keeps most objects)
- **0.22-0.25**: Balanced (recommended)
- **0.30+**: Strict (only very confident matches)

For "tin roof" or "blue roof", **0.22** is a good starting point.

---

## Testing

### Create `tests/test_stage4.py`

**Test Case 1**: Filter masks by target description

```python
def test_clip_filter_masks():
    """Test that CLIP filters masks semantically."""
    # Load test image
    image = cv2.imread("test_image.jpeg")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get masks from Stages 2 and 3
    from pipeline.stage2_color_filter import color_prefilter
    from pipeline.stage3_sam_segmentation import segment_regions

    color_mask = color_prefilter(image_rgb, "blue")
    masks = segment_regions(image_rgb, color_mask, min_area=500)

    # Apply CLIP filtering
    from pipeline.stage4_clip_filter import clip_filter_masks
    filtered = clip_filter_masks(image_rgb, masks, "tin roof", similarity_threshold=0.20)

    # Should filter out sky and keep roofs
    assert isinstance(filtered, list)
    assert len(filtered) > 0, "Should keep at least some masks"
    assert len(filtered) < len(masks), "Should filter out some masks"

    # Each result should be (mask, score) tuple
    for mask, score in filtered:
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == np.uint8
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert score >= 0.20, f"Score {score} below threshold"
```

**Test Case 2**: Verify sorted by similarity

```python
def test_sorted_by_similarity():
    """Test that results are sorted by similarity score."""
    # ... load image and get masks ...

    filtered = clip_filter_masks(image_rgb, masks, "roof", similarity_threshold=0.15)

    # Verify sorted descending
    scores = [score for _, score in filtered]
    assert scores == sorted(scores, reverse=True), "Results should be sorted by similarity"
```

**Test Case 3**: Empty masks list

```python
def test_empty_masks():
    """Test behavior with empty masks list."""
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    masks = []

    from pipeline.stage4_clip_filter import clip_filter_masks
    filtered = clip_filter_masks(image, masks, "roof")

    assert filtered == []
```

**Test Case 4**: High threshold filters all

```python
def test_high_threshold():
    """Test that very high threshold filters out all masks."""
    # ... load image and masks ...

    # Use impossibly high threshold
    filtered = clip_filter_masks(image_rgb, masks, "roof", similarity_threshold=0.95)

    # Should filter everything
    assert len(filtered) == 0
```

**Test Case 5**: Performance benchmark

```python
def test_performance():
    """Test that CLIP filtering completes in reasonable time."""
    import time

    # ... load image and get ~15-20 masks ...

    start = time.time()
    filtered = clip_filter_masks(image_rgb, masks, "tin roof")
    elapsed = time.time() - start

    # Target: <5 seconds for 15-20 masks
    assert elapsed < 5.0, f"Too slow: {elapsed:.1f}s (target: <5s)"
    print(f"Performance: {elapsed:.2f}s for {len(masks)} masks")
```

---

## Visual Validation

After implementation, create visualization:

```python
# Save visualization for supervisor review
import matplotlib.pyplot as plt

# Load test image and run pipeline
test_img = cv2.imread("test_image.jpeg")
test_img_rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

# Stage 2: Color filter
from pipeline.stage2_color_filter import color_prefilter
color_mask = color_prefilter(test_img_rgb, "blue")

# Stage 3: SAM segmentation
from pipeline.stage3_sam_segmentation import segment_regions
sam_masks = segment_regions(test_img_rgb, color_mask, min_area=500)

# Stage 4: CLIP filtering
from pipeline.stage4_clip_filter import clip_filter_masks
filtered_masks = clip_filter_masks(test_img_rgb, sam_masks, "tin roof",
                                  similarity_threshold=0.22)

print(f"\nStage 3: {len(sam_masks)} SAM masks")
print(f"Stage 4: {len(filtered_masks)} CLIP-filtered masks")

# Create comparison visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Before CLIP (all SAM masks)
overlay_before = test_img_rgb.copy()
for mask in sam_masks:
    overlay_before[mask > 0] = overlay_before[mask > 0] * 0.5 + np.array([255, 0, 0]) * 0.5

axes[0].imshow(overlay_before)
axes[0].set_title(f"Before CLIP: {len(sam_masks)} masks\n(includes sky, buildings, etc.)")
axes[0].axis('off')

# After CLIP (filtered masks)
overlay_after = test_img_rgb.copy()
for mask, score in filtered_masks:
    overlay_after[mask > 0] = overlay_after[mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5

axes[1].imshow(overlay_after)
axes[1].set_title(f"After CLIP: {len(filtered_masks)} masks\n(roofs only)")
axes[1].axis('off')

plt.tight_layout()
plt.savefig("logs/stage4_clip_filtering.png", dpi=150, bbox_inches='tight')
print("\nSaved: logs/stage4_clip_filtering.png")

# Print top similarity scores
print("\nTop 5 similarity scores:")
for i, (_, score) in enumerate(filtered_masks[:5]):
    print(f"  Mask {i+1}: {score:.3f}")
```

---

## Expected Results

For test_image.jpeg with "tin roof" query:

**Input**: 17 SAM masks (roofs + sky + other objects)
**Output**: 10-15 roof masks (sky and non-roof objects filtered out)

**Expected filtering**:
- ❌ Sky (large region, low similarity to "tin roof")
- ✅ Blue tin roofs (high similarity)
- ❌ Trees, vehicles (low similarity)
- ❌ Building walls (medium similarity, but below threshold)

---

## Acceptance Criteria

- [ ] Function `clip_filter_masks()` implemented correctly
- [ ] All 5 test cases pass
- [ ] Performance: <5 seconds for 15-20 masks
- [ ] Visual validation: Sky filtered out, roofs retained
- [ ] Returns list of (mask, score) tuples sorted by score
- [ ] Code has type hints and docstrings
- [ ] Proper GPU memory cleanup

---

## Report Format

After completion, report:

```
STAGE 4: CLIP Semantic Filtering - [COMPLETE/FAILED]

Implementation:
- File: pipeline/stage4_clip_filter.py
- Lines of code: XXX
- Function: clip_filter_masks()

Test Results:
- Test Case 1 (Filter masks): [PASS/FAIL]
- Test Case 2 (Sorted by similarity): [PASS/FAIL]
- Test Case 3 (Empty masks): [PASS/FAIL]
- Test Case 4 (High threshold): [PASS/FAIL]
- Test Case 5 (Performance <5s): [PASS/FAIL]

Performance:
- Execution time: X.XX seconds
- Input masks: XX (from Stage 3)
- Output masks: XX (filtered)
- Filter rate: XX% removed

Visual Validation:
- Saved to: logs/stage4_clip_filtering.png
- Sky filtered: [YES/NO]
- Roofs retained: [YES/NO]

Top 3 Similarity Scores:
1. Mask X: 0.XXX
2. Mask X: 0.XXX
3. Mask X: 0.XXX

Issues: [None / List any problems]

Status: Awaiting supervisor approval
```

---

**Begin implementation of Stage 4 now!**
