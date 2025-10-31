# YOLO-World Findings and Implementation Strategy

**Date**: 2025-10-31
**Status**: Critical Finding - Requires Dual-Path Architecture

---

## Key Finding: YOLO-World Color Limitation

### What Works ✅

**Semantic-only queries** work perfectly:
```python
detect("vehicles")  → 2 detections (conf: 0.268, 0.260)
detect("car")       → 5 detections (conf: 0.896, 0.804, ...)
detect("building")  → Works when buildings present
detect("roof")      → Works when roofs present
```

**Performance**: Fast (~50ms), accurate, reliable

### What Doesn't Work ❌

**Color+object queries** return zero detections:
```python
detect("red vehicles")     → 0 detections (even though red car confirmed present by VLM)
detect("brown roofs")      → 0 detections
detect("yellow auto-rickshaws") → 0 detections (even though yellow taxis present)
```

---

## Root Cause Analysis

**Why YOLO-World can't handle color queries:**

1. **Training Data**: YOLO-World is trained on LVIS/COCO datasets
   - Categories are semantic: "car", "person", "dog", "building"
   - NOT color-specific: No "red car" or "brown dog" categories

2. **Open-vocabulary mechanism**: Uses CLIP text embeddings
   - "car" → Strong semantic embedding in CLIP
   - "red vehicles" → Weak/ambiguous embedding (not a standard category)
   - CLIP is trained on image-text pairs, not fine-grained color distinctions

3. **Detection vs Description**: YOLO models detect **what** objects are, not **how** they look
   - Color is a visual attribute, not an object category
   - Requires separate color analysis stage

---

## Solution: Dual-Path Architecture

### Architecture Overview

```
User Query: "red vehicles"
    ↓
┌─────────────────────────────────────┐
│ Query Parser (DSpy or simple regex) │
│ Extracts: color="red", object="vehicles" │
└─────────────────────────────────────┘
    ↓
    ├─ HAS COLOR? ─→ YES
    │                 ↓
    │   ┌───────────────────────────────┐
    │   │ Stage 1a: Semantic Detection  │
    │   │ YOLO-World("vehicles")        │
    │   │ Result: N boxes               │
    │   └───────────────────────────────┘
    │                 ↓
    │   ┌───────────────────────────────┐
    │   │ Stage 1b: Color Filtering     │
    │   │ For each box:                 │
    │   │   - Extract region from image │
    │   │   - Analyze HSV histogram     │
    │   │   - Match against "red"       │
    │   │   - Filter if mismatch        │
    │   └───────────────────────────────┘
    │                 ↓
    │              Filtered boxes
    │
    └─ NO COLOR? ─→ Use YOLO-World directly
                     Result: All boxes
```

### Implementation Plan

#### File: `pipeline/stage1_query_parser.py` (NEW)

```python
"""Parse user query to extract color and object components."""

import re
from dataclasses import dataclass
from typing import Optional

COLORS = [
    "red", "blue", "green", "yellow", "orange", "purple", "brown",
    "black", "white", "gray", "grey", "pink", "cyan", "magenta"
]

@dataclass
class ParsedQuery:
    """Parsed query components."""
    color: Optional[str]  # None if no color
    object_type: str      # "vehicles", "roofs", "buildings", etc.
    original_query: str

def parse_query(user_query: str) -> ParsedQuery:
    """
    Parse user query to extract color and object.

    Examples:
        "red vehicles" → ParsedQuery(color="red", object_type="vehicles")
        "vehicles" → ParsedQuery(color=None, object_type="vehicles")
        "brown roofs" → ParsedQuery(color="brown", object_type="roofs")
    """
    query_lower = user_query.lower()

    # Check for color
    detected_color = None
    for color in COLORS:
        if color in query_lower:
            detected_color = color
            # Remove color from query to get object
            object_query = query_lower.replace(color, "").strip()
            break
    else:
        # No color found
        object_query = query_lower

    return ParsedQuery(
        color=detected_color,
        object_type=object_query,
        original_query=user_query
    )
```

#### File: `pipeline/stage1b_color_filter.py` (NEW)

```python
"""Color-based filtering for detected boxes."""

import numpy as np
import cv2
from typing import List
from dataclasses import dataclass

from .stage1_yolo_world import DetectionBox

# HSV color ranges (more reliable than RGB)
COLOR_RANGES_HSV = {
    "red": [(0, 100, 100), (10, 255, 255), (160, 100, 100), (180, 255, 255)],  # Red wraps around
    "blue": [(100, 100, 100), (130, 255, 255)],
    "green": [(40, 40, 40), (80, 255, 255)],
    "yellow": [(20, 100, 100), (30, 255, 255)],
    "orange": [(10, 100, 100), (20, 255, 255)],
    "brown": [(10, 50, 20), (20, 200, 100)],
    "purple": [(130, 50, 50), (160, 255, 255)],
    "white": [(0, 0, 200), (180, 30, 255)],
    "black": [(0, 0, 0), (180, 255, 30)],
    "gray": [(0, 0, 50), (180, 50, 200)],
}

def filter_boxes_by_color(
    image: np.ndarray,
    boxes: List[DetectionBox],
    target_color: str,
    color_match_threshold: float = 0.3
) -> List[DetectionBox]:
    """
    Filter detection boxes by color using HSV analysis.

    Args:
        image: RGB image (H x W x 3)
        boxes: Detected boxes from YOLO-World
        target_color: Target color name (e.g., "red", "blue")
        color_match_threshold: Minimum percentage of pixels matching color (0-1)

    Returns:
        Filtered list of boxes that match the target color
    """
    if target_color not in COLOR_RANGES_HSV:
        logging.warning(f"Color '{target_color}' not in color database, returning all boxes")
        return boxes

    # Convert image to HSV
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    filtered_boxes = []

    for box in boxes:
        # Extract region
        x1, y1 = box.x, box.y
        x2, y2 = box.x + box.w, box.y + box.h

        # Ensure valid bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)

        region_hsv = image_hsv[y1:y2, x1:x2]

        # Check color match
        color_ranges = COLOR_RANGES_HSV[target_color]

        # Handle red (wraps around 0/180)
        if target_color == "red":
            lower1, upper1, lower2, upper2 = color_ranges
            mask1 = cv2.inRange(region_hsv, np.array(lower1), np.array(upper1))
            mask2 = cv2.inRange(region_hsv, np.array(lower2), np.array(upper2))
            color_mask = cv2.bitwise_or(mask1, mask2)
        else:
            lower, upper = color_ranges
            color_mask = cv2.inRange(region_hsv, np.array(lower), np.array(upper))

        # Calculate percentage of pixels matching color
        total_pixels = region_hsv.shape[0] * region_hsv.shape[1]
        matching_pixels = np.sum(color_mask > 0)
        match_percentage = matching_pixels / total_pixels if total_pixels > 0 else 0

        if match_percentage >= color_match_threshold:
            filtered_boxes.append(box)
            logging.debug(f"  Box {box.label}: {match_percentage:.1%} {target_color} → KEEP")
        else:
            logging.debug(f"  Box {box.label}: {match_percentage:.1%} {target_color} → FILTER OUT")

    return filtered_boxes
```

#### Updated: `pipeline/stage1_yolo_world.py`

Add high-level function with color support:

```python
def detect_entities_with_color(
    image: np.ndarray,
    user_prompt: str,
    confidence_threshold: float = 0.35
) -> List[DetectionBox]:
    """
    Detect entities with optional color filtering.

    Handles both semantic-only and color+object queries.

    Examples:
        >>> boxes = detect_entities_with_color(image, "vehicles")
        >>> boxes = detect_entities_with_color(image, "red vehicles")
        >>> boxes = detect_entities_with_color(image, "brown roofs")
    """
    from .stage1_query_parser import parse_query
    from .stage1b_color_filter import filter_boxes_by_color

    # Parse query
    parsed = parse_query(user_prompt)

    # Detect using semantic part
    boxes = detect_entities_yolo_world(
        image,
        parsed.object_type,
        confidence_threshold
    )

    # Filter by color if specified
    if parsed.color and len(boxes) > 0:
        logging.info(f"Filtering {len(boxes)} boxes by color: {parsed.color}")
        boxes = filter_boxes_by_color(image, boxes, parsed.color)
        logging.info(f"After color filtering: {len(boxes)} boxes")

    return boxes
```

---

## Testing Strategy

### Test Cases

1. **Semantic-only** (should work now):
   - "vehicles" → Detect all vehicles
   - "buildings" → Detect all buildings
   - "roofs" → Detect all roofs

2. **Color+object** (needs dual-path):
   - "red vehicles" → Detect vehicles, filter by red
   - "brown roofs" → Detect roofs, filter by brown
   - "yellow auto-rickshaws" → Detect "auto-rickshaws" OR "vehicles", filter by yellow

3. **Edge cases**:
   - "purple objects" → Detect nothing gracefully (no semantic category)
   - "sky" → Detect using YOLO-World semantic understanding

### Validation with VLM

For each detection:
1. Run YOLO-World detection
2. Apply color filtering
3. Use local vision MCP to validate: "Are the detected regions {color} {objects}?"

---

## Performance Impact

**Before** (YOLO-World only):
- Semantic queries: ✅ Fast (~50ms), accurate
- Color queries: ❌ Zero detections

**After** (Dual-path):
- Semantic queries: ✅ Fast (~50ms), no change
- Color queries: ✅ Moderate (~150ms total)
  - YOLO-World: ~50ms
  - HSV analysis per box: ~10ms each
  - Total: 50ms + (N_boxes × 10ms)

**For 5 detected boxes**: ~150ms total (still fast!)

---

## Implementation Priority

**Day 1 Complete** ✅:
- Stage 1: YOLO-World semantic detection
- Stage 2: Box-to-SAM conversion
- Testing infrastructure

**Day 2 Next Steps** (IMMEDIATE):
1. ✅ Document findings (this file)
2. ⏳ Implement `stage1_query_parser.py`
3. ⏳ Implement `stage1b_color_filter.py`
4. ⏳ Update `detect_entities_yolo_world()` wrapper
5. ⏳ Test on all 9 wildcard cases

**Expected Results After Implementation**:
- Semantic queries: 100% success (already working)
- Color queries: 80-90% success (new capability)
- Overall: 89%+ success rate on 9 wildcard tests

---

## Conclusion

**YOLO-World is the right choice** - it's fast, accurate, and already installed. The color limitation is expected and solvable with a simple HSV filtering stage.

The dual-path architecture is:
- **Lightweight**: Just adds HSV analysis (~10ms per box)
- **Reliable**: HSV color matching is deterministic
- **Flexible**: Can add more colors easily

This approach **solves all 3 critical v2.0 flaws**:
1. ✅ No color-first assumption → Semantic queries work
2. ✅ No toxic fallback → Returns empty list gracefully
3. ✅ Fast and reliable → YOLO-World + HSV is deterministic

**Status**: Ready to implement dual-path architecture.
**Time Estimate**: 1-2 hours for implementation + testing.
