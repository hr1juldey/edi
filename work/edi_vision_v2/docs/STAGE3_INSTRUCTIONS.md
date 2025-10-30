# Stage 3: SAM Segmentation - Implementation Instructions

**File to create**: `pipeline/stage3_sam_segmentation.py`

**Reference**: See `/home/riju279/Documents/Code/Zonko/EDI/edi/docs/VISION_PIPELINE_RESEARCH.md` Section "Stage 3: SAM 2.1 Segmentation"

---

## Overview

This stage takes the binary color mask from Stage 2 and generates pixel-perfect segmentation masks for EACH distinct region using SAM 2.1.

**Purpose**: Convert coarse color mask into individual, precise masks (one per entity)

**Key Point**: ALL regions must be processed (not top-k). Each roof gets its own separate mask.

---

## Requirements

### 1. Main Function: `segment_regions()`

```python
def segment_regions(image: np.ndarray, color_mask: np.ndarray,
                   min_area: int = 100) -> List[np.ndarray]:
    """
    Generate individual SAM masks for each region in the color mask.

    Args:
        image: Original RGB image (H x W x 3), numpy array
        color_mask: Binary mask from Stage 2 (H x W), values 0 or 1
        min_area: Minimum region area in pixels (filter noise)

    Returns:
        List of binary masks (each H x W, dtype=np.uint8), one per region
    """
```

### 2. Processing Steps

**Step 1: Find Connected Components**

```python
# Convert mask to uint8 if needed (values 0 or 255 for connectedComponents)
color_mask_vis = (color_mask * 255).astype(np.uint8)

# Find all connected regions
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
    color_mask_vis,
    connectivity=8
)

# stats columns: [x, y, width, height, area]
# centroids shape: (num_labels, 2) - [x, y] coordinates
# labels[i, j] = component ID at pixel (i, j)
```

**Step 2: Filter Small Regions (Noise)**

```python
valid_regions = []
for label_id in range(1, num_labels):  # Skip label 0 (background)
    area = stats[label_id, cv2.CC_STAT_AREA]
    if area >= min_area:
        centroid = centroids[label_id]  # (x, y)
        bbox = stats[label_id, :4]      # [x, y, width, height]
        valid_regions.append({
            'label_id': label_id,
            'centroid': centroid,
            'bbox': bbox,
            'area': area
        })

logging.info(f"Found {len(valid_regions)} valid regions (filtered {num_labels - 1 - len(valid_regions)} noise regions)")
```

**Step 3: Initialize SAM 2.1**

```python
from ultralytics import SAM

# Load SAM 2.1 base model
sam_model = SAM("sam2.1_b.pt")  # Or sam2.1_t.pt for faster, sam2.1_h.pt for better quality
sam_model.to('cuda')  # Use GPU

# Enable half precision for speed
if torch.cuda.is_available():
    sam_model.half()
```

**Step 4: Run SAM on Each Region**

```python
individual_masks = []

for region in valid_regions:
    # Extract point prompt (centroid coordinates)
    point_x, point_y = region['centroid']
    point_prompt = [[int(point_x), int(point_y)]]  # SAM expects [[x, y]]

    # Run SAM with point prompt
    results = sam_model(
        image,
        points=point_prompt,
        labels=[1]  # 1 = foreground point
    )

    # Extract mask from results
    # SAM returns masks in results[0].masks.data (torch tensor)
    if len(results) > 0 and results[0].masks is not None:
        # Convert to numpy, take first mask (highest confidence)
        mask = results[0].masks.data[0].cpu().numpy()
        mask_binary = (mask > 0.5).astype(np.uint8)
        individual_masks.append(mask_binary)

        logging.debug(f"Region {region['label_id']}: Generated mask with {np.sum(mask_binary)} pixels")
    else:
        logging.warning(f"Region {region['label_id']}: SAM failed to generate mask")

logging.info(f"Generated {len(individual_masks)} individual masks")
```

**Step 5: Cleanup and Return**

```python
# Unload SAM to free GPU memory
del sam_model
torch.cuda.empty_cache()

return individual_masks
```

---

## Edge Cases

### 1. No Regions Found
```python
if num_labels <= 1:  # Only background
    logging.warning("No regions found in color mask")
    return []
```

### 2. SAM Out of Memory
```python
try:
    results = sam_model(image, points=point_prompt, labels=[1])
except RuntimeError as e:
    if "out of memory" in str(e):
        logging.error("SAM OOM - try reducing image resolution")
        # Fallback: resize image
        scale = 0.5
        resized_image = cv2.resize(image, None, fx=scale, fy=scale)
        results = sam_model(resized_image, points=[[int(point_x*scale), int(point_y*scale)]], labels=[1])
        # Resize mask back
        mask = cv2.resize(results[0].masks.data[0].cpu().numpy(),
                         (image.shape[1], image.shape[0]))
    else:
        raise
```

### 3. Very Large Number of Regions
```python
if len(valid_regions) > 100:
    logging.warning(f"{len(valid_regions)} regions detected - this may take a while")
    # Consider increasing min_area threshold
```

### 4. Overlapping Masks
```python
# SAM may generate overlapping masks
# This is OK - Stage 4 (CLIP) will filter by semantic relevance
# Stage 5 will organize them properly
```

---

## Implementation Checklist

- [ ] Import required libraries (cv2, numpy, ultralytics, torch, typing)
- [ ] Implement `segment_regions()` function
- [ ] Connected components analysis with cv2.connectedComponentsWithStats
- [ ] Filter regions by minimum area (default 100 pixels)
- [ ] Load SAM 2.1 model with half precision
- [ ] Iterate through each valid region
- [ ] Run SAM with point prompts (centroids)
- [ ] Extract and convert masks to numpy binary format
- [ ] Cleanup GPU memory after processing
- [ ] Add type hints and docstrings
- [ ] Add comprehensive logging

---

## Testing

### Create `tests/test_stage3.py`

**Test Case 1**: Generate masks from color mask

```python
def test_segment_regions():
    """Test that SAM generates individual masks from color mask."""
    # Load test image
    image = cv2.imread("test_image.jpeg")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Use Stage 2 to get color mask
    from pipeline.stage2_color_filter import color_prefilter
    color_mask = color_prefilter(image_rgb, "blue")

    # Run Stage 3
    from pipeline.stage3_sam_segmentation import segment_regions
    masks = segment_regions(image_rgb, color_mask, min_area=100)

    # Verify we got multiple masks
    assert isinstance(masks, list)
    assert len(masks) > 0, "Should generate at least one mask"

    # Each mask should be binary numpy array
    for i, mask in enumerate(masks):
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == np.uint8
        assert mask.shape[:2] == image_rgb.shape[:2], f"Mask {i} shape mismatch"
        assert np.all((mask == 0) | (mask == 1)), f"Mask {i} not binary"

        # Should have reasonable coverage
        coverage = np.sum(mask) / mask.size
        assert coverage > 0, f"Mask {i} is empty"
        assert coverage < 0.5, f"Mask {i} covers too much ({coverage*100:.1f}%)"
```

**Test Case 2**: Minimum area filtering

```python
def test_min_area_filtering():
    """Test that small regions are filtered out."""
    # Create synthetic mask with small and large regions
    color_mask = np.zeros((500, 500), dtype=np.uint8)

    # Large region (10,000 pixels)
    color_mask[100:200, 100:200] = 1

    # Small region (25 pixels - should be filtered)
    color_mask[300:305, 300:305] = 1

    # Create dummy image
    image = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)

    # Run with min_area=100
    from pipeline.stage3_sam_segmentation import segment_regions
    masks = segment_regions(image, color_mask, min_area=100)

    # Should only get 1 mask (large region), small region filtered
    assert len(masks) == 1, f"Expected 1 mask, got {len(masks)}"
```

**Test Case 3**: Empty mask handling

```python
def test_empty_mask():
    """Test behavior with empty color mask."""
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    color_mask = np.zeros((100, 100), dtype=np.uint8)

    from pipeline.stage3_sam_segmentation import segment_regions
    masks = segment_regions(image, color_mask)

    # Should return empty list
    assert masks == []
```

**Test Case 4**: Separate masks for separate regions

```python
def test_separate_masks():
    """Test that each region gets its own mask (CRITICAL REQUIREMENT)."""
    # Create mask with 3 distinct regions
    color_mask = np.zeros((300, 300), dtype=np.uint8)
    color_mask[50:100, 50:100] = 1    # Region 1
    color_mask[50:100, 200:250] = 1   # Region 2
    color_mask[200:250, 125:175] = 1  # Region 3

    image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)

    from pipeline.stage3_sam_segmentation import segment_regions
    masks = segment_regions(image, color_mask, min_area=100)

    # Should get 3 separate masks
    assert len(masks) == 3, f"Expected 3 masks, got {len(masks)}"

    # Verify masks don't overlap (they should be separate)
    # (SAM may create slight overlaps, but centroids should be in different locations)
    centroids = []
    for mask in masks:
        y_coords, x_coords = np.where(mask > 0)
        centroid_y = np.mean(y_coords)
        centroid_x = np.mean(x_coords)
        centroids.append((centroid_x, centroid_y))

    # Check that centroids are sufficiently separated
    for i in range(len(centroids)):
        for j in range(i+1, len(centroids)):
            dist = np.sqrt((centroids[i][0] - centroids[j][0])**2 +
                          (centroids[i][1] - centroids[j][1])**2)
            assert dist > 50, f"Masks {i} and {j} too close (distance {dist:.1f})"
```

**Test Case 5**: Performance benchmark

```python
def test_performance():
    """Test that SAM segmentation completes in reasonable time."""
    import time

    # Use actual test image
    image = cv2.imread("test_image.jpeg")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    from pipeline.stage2_color_filter import color_prefilter
    color_mask = color_prefilter(image_rgb, "blue")

    # Benchmark
    start = time.time()
    from pipeline.stage3_sam_segmentation import segment_regions
    masks = segment_regions(image_rgb, color_mask, min_area=100)
    elapsed = time.time() - start

    # Target: <5 seconds for typical image with 20-30 regions
    # Being generous with 10 seconds for test stability
    assert elapsed < 10.0, f"Too slow: {elapsed:.1f}s (target: <10s)"

    print(f"Performance: {elapsed:.2f}s for {len(masks)} masks")
```

---

## Visual Validation

After implementation, create visualization to verify individual masks:

```python
# Save visualization for supervisor review
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Load test image and run pipeline
test_img = cv2.imread("test_image.jpeg")
test_img_rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

# Stage 2: Color filter
from pipeline.stage2_color_filter import color_prefilter
color_mask = color_prefilter(test_img_rgb, "blue")

# Stage 3: SAM segmentation
from pipeline.stage3_sam_segmentation import segment_regions
individual_masks = segment_regions(test_img_rgb, color_mask, min_area=500)

# Create visualization grid
num_masks = len(individual_masks)
cols = min(5, num_masks)
rows = (num_masks + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
if rows == 1:
    axes = [axes]
if cols == 1:
    axes = [[ax] for ax in axes]

for idx, mask in enumerate(individual_masks):
    row = idx // cols
    col = idx % cols

    # Create overlay
    overlay = test_img_rgb.copy()
    overlay[mask > 0] = overlay[mask > 0] * 0.5 + np.array([255, 0, 0]) * 0.5

    axes[row][col].imshow(overlay)
    axes[row][col].set_title(f"Mask {idx+1} ({np.sum(mask)} px)")
    axes[row][col].axis('off')

# Hide unused subplots
for idx in range(num_masks, rows * cols):
    row = idx // cols
    col = idx % cols
    axes[row][col].axis('off')

plt.tight_layout()
plt.savefig("logs/stage3_individual_masks.png", dpi=150, bbox_inches='tight')
print(f"\nSaved: logs/stage3_individual_masks.png")
print(f"Generated {len(individual_masks)} individual masks")
```

---

## Acceptance Criteria

- [ ] Function `segment_regions()` implemented correctly
- [ ] All 5 test cases pass
- [ ] Performance: <10 seconds for test image with ~20 regions
- [ ] Visual validation: Each detected region has separate mask
- [ ] Code has type hints and docstrings
- [ ] Proper GPU memory cleanup (del model, torch.cuda.empty_cache())
- [ ] Handles edge cases (empty mask, OOM, no regions)
- [ ] **CRITICAL**: Masks stay SEPARATE (not merged)

---

## Expected Output

For test_image.jpeg with 20+ blue roofs:

```python
# Example usage
image = load_test_image()
color_mask = color_prefilter(image, "blue")
masks = segment_regions(image, color_mask, min_area=500)

print(f"Number of masks: {len(masks)}")  # Should be 20-30 (roofs + sky patches)
print(f"Mask shapes: {[m.shape for m in masks[:3]]}")  # All should match image shape
print(f"Coverage: {[np.sum(m) for m in masks]}")  # Pixels per mask
```

Expected output:
```
Number of masks: 27
Mask shapes: [(1080, 1920), (1080, 1920), (1080, 1920)]
Coverage: [4823, 3456, 2891, 1234, 5678, ...]  # Varies by entity size
```

---

## Report Format

After completion, report:

```
STAGE 3: SAM Segmentation - [COMPLETE/FAILED]

Implementation:
- File: pipeline/stage3_sam_segmentation.py
- Lines of code: XXX
- Function: segment_regions()

Test Results:
- Test Case 1 (Generate masks): [PASS/FAIL]
- Test Case 2 (Min area filtering): [PASS/FAIL]
- Test Case 3 (Empty mask): [PASS/FAIL]
- Test Case 4 (Separate masks - CRITICAL): [PASS/FAIL]
- Test Case 5 (Performance <10s): [PASS/FAIL]

Performance:
- Execution time: X.XX seconds
- Number of masks generated: XX
- Average mask area: XXXX pixels

Visual Validation:
- Saved to: logs/stage3_individual_masks.png
- Description: [Does each region have its own mask?]

Issues: [None / List any problems]

Status: Awaiting supervisor approval
```

---

## Notes

**SAM Model Selection**:
- `sam2.1_t.pt` (Tiny): Fastest, good for quick iteration
- `sam2.1_b.pt` (Base): Recommended - good balance
- `sam2.1_h.pt` (Huge): Best quality, slower

**Point Prompt Strategy**:
- Use centroid of each region as point prompt
- SAM works best with point prompts for individual objects
- Box prompts could be added later for refinement

**GPU Memory**:
- SAM 2.1 Base (FP16): ~3.5 GB VRAM
- Process one image at a time
- Unload model after completion

**Performance Tips**:
- Use half precision: `sam_model.half()`
- Process regions in batch if SAM API supports it
- Consider max image dimension (2048px) to avoid OOM

---

**Begin implementation of Stage 3 now!**
