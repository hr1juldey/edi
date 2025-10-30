# Stage 2: Color Pre-Filtering - Implementation Instructions

**File to create**: `pipeline/stage2_color_filter.py`

**Reference**: See `/home/riju279/Documents/Code/Zonko/EDI/edi/docs/VISION_PIPELINE_RESEARCH.md` Section "Stage 2: Color Pre-Filtering (HSV)"

---

## Overview

This stage implements a fast HSV-based color filter to narrow down the search space before running expensive SAM segmentation.

**Purpose**: Detect ALL regions matching a specified color (e.g., "blue") in <100ms

**Key Point**: High recall (catch all blue regions) is more important than precision (some false positives OK). CLIP will filter out false positives in Stage 4.

---

## Requirements

### 1. Main Function: `color_prefilter()`

```python
def color_prefilter(image: np.ndarray, color_name: str) -> np.ndarray:
    """
    Create a binary mask of all regions matching the specified color.

    Args:
        image: RGB image (H x W x 3), numpy array
        color_name: Color to filter (e.g., "blue", "red", "green")

    Returns:
        Binary mask (H x W) where 1 = color match, dtype=np.uint8
    """
```

### 2. Color Ranges (HSV)

Define these color ranges:

```python
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
```

**Note**: Red has TWO ranges because hue wraps around at 180°

### 3. Processing Steps

1. Convert image from RGB to HSV: `cv2.cvtColor(image, cv2.COLOR_RGB2HSV)`
2. Apply `cv2.inRange()` to create binary mask
3. Morphological cleanup:
   - `cv2.morphologyEx(..., cv2.MORPH_CLOSE, ...)` - Fill small holes
   - `cv2.morphologyEx(..., cv2.MORPH_OPEN, ...)` - Remove noise
4. Return cleaned binary mask

### 4. Edge Cases

- If `color_name` not in dictionary: Return mask of all ones (no filtering)
- Handle red's dual ranges with logical OR
- Use elliptical kernel for morphological ops: `cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))`

---

## Implementation Checklist

- [ ] Import required libraries (cv2, numpy, typing)
- [ ] Define `color_ranges` dictionary
- [ ] Implement `color_prefilter()` function
- [ ] Handle RGB to HSV conversion
- [ ] Handle red's dual-range case
- [ ] Implement morphological cleanup
- [ ] Add type hints
- [ ] Add docstring
- [ ] Add logging statements

---

## Testing

### Create `tests/test_stage2.py`

**Test Case 1**: Blue color detection on test_image.jpeg
```python
def test_blue_color_detection():
    """Test that blue regions are detected in test image."""
    image = cv2.imread("test_image.jpeg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask = color_prefilter(image, "blue")

    # Verify mask is binary
    assert mask.dtype == np.uint8
    assert np.all((mask == 0) | (mask == 1) | (mask == 255))

    # Verify significant blue regions detected (roofs should be found)
    blue_pixels = np.sum(mask > 0)
    total_pixels = mask.size
    blue_percentage = (blue_pixels / total_pixels) * 100

    # We expect at least 1% blue pixels (20 roofs)
    assert blue_percentage > 1.0, f"Only {blue_percentage:.2f}% blue pixels found"
    assert blue_percentage < 50.0, f"Too many: {blue_percentage:.2f}% blue pixels"
```

**Test Case 2**: Red color with dual ranges
```python
def test_red_color_dual_range():
    """Test that red color detection handles dual ranges."""
    # Create synthetic image with red pixels
    test_img = np.zeros((100, 100, 3), dtype=np.uint8)
    test_img[20:40, 20:40] = [255, 0, 0]  # Bright red

    mask = color_prefilter(test_img, "red")

    # Verify red region is detected
    red_pixels = np.sum(mask[20:40, 20:40] > 0)
    expected_pixels = 20 * 20
    assert red_pixels > expected_pixels * 0.8  # At least 80% detected
```

**Test Case 3**: Fallback for unknown color
```python
def test_unknown_color_fallback():
    """Test fallback behavior for unknown color."""
    test_img = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = color_prefilter(test_img, "unknown_color")

    # Should return all ones (no filtering)
    assert np.all(mask == 1)
```

**Test Case 4**: Performance benchmark
```python
def test_performance():
    """Test that color filtering is fast (<100ms)."""
    import time

    image = cv2.imread("test_image.jpeg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    start = time.time()
    mask = color_prefilter(image, "blue")
    elapsed = (time.time() - start) * 1000

    assert elapsed < 100.0, f"Too slow: {elapsed:.0f}ms (target: <100ms)"
```

---

## Visual Validation

After implementation, create a visualization to check results:

```python
# Save visualization for supervisor review
import matplotlib.pyplot as plt
from PIL import Image

# Load test image
test_img = cv2.imread("test_image.jpeg")
test_img_rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

# Apply color filter
mask = color_prefilter(test_img_rgb, "blue")

# Create visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(test_img_rgb)
axes[0].set_title("Original Image")
axes[0].axis('off')

axes[1].imshow(mask, cmap='gray')
axes[1].set_title("Blue Color Mask")
axes[1].axis('off')

# Overlay
overlay = test_img_rgb.copy()
overlay[mask > 0] = overlay[mask > 0] * 0.5 + np.array([255, 0, 0]) * 0.5
axes[2].imshow(overlay)
axes[2].set_title("Overlay (Red = Detected)")
axes[2].axis('off')

plt.tight_layout()
plt.savefig("logs/stage2_blue_detection.png", dpi=150, bbox_inches='tight')
print("Saved: logs/stage2_blue_detection.png")
```

---

## Acceptance Criteria

- [ ] Function `color_prefilter()` implemented correctly
- [ ] All 4 test cases pass
- [ ] Performance: <100ms for 1920x1080 image
- [ ] Visual validation: Blue roofs clearly highlighted in overlay
- [ ] Code has type hints and docstrings
- [ ] Handles edge cases (unknown color, red dual-range)

---

## Expected Output

```python
# Example usage
image = load_test_image()
mask = color_prefilter(image, "blue")

print(f"Mask shape: {mask.shape}")  # Should match image dimensions
print(f"Mask dtype: {mask.dtype}")  # Should be np.uint8
print(f"Blue pixels: {np.sum(mask > 0)}")  # Should be thousands
print(f"Coverage: {np.sum(mask > 0) / mask.size * 100:.1f}%")  # Should be 5-15%
```

Expected output:
```
Mask shape: (1080, 1920)
Mask dtype: uint8
Blue pixels: 94832
Coverage: 4.6%
```

---

## Report Format

After completion, report:

```
STAGE 2: Color Pre-Filter - [COMPLETE/FAILED]

Implementation:
- File: pipeline/stage2_color_filter.py
- Lines of code: XXX
- Function: color_prefilter()

Test Results:
- Test Case 1 (Blue detection): [PASS/FAIL]
- Test Case 2 (Red dual-range): [PASS/FAIL]
- Test Case 3 (Unknown color fallback): [PASS/FAIL]
- Test Case 4 (Performance <100ms): [PASS/FAIL]

Performance:
- Execution time: XX ms
- Blue coverage: X.X%

Visual Validation:
- Saved to: logs/stage2_blue_detection.png
- Description: [Does it cover all blue roofs?]

Issues: [None / List any problems]

Status: Awaiting supervisor approval
```

---

## Notes

- Use `cv2.morphologyEx()` for cleanup, not manual operations
- Kernel size (5, 5) is good default, but can be adjusted
- RGB to HSV: `cv2.cvtColor(image, cv2.COLOR_RGB2HSV)`
- Recall > Precision at this stage (better to over-detect than miss entities)

---

**Begin implementation of Stage 2 now!**
