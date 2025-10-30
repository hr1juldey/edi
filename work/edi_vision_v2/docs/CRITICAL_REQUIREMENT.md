# ⚠️ CRITICAL REQUIREMENT ⚠️

## DO NOT MERGE MASKS!

**User Requirement**: All roofs will have **SEPARATE masks**. Don't let the masks merge.

---

## What This Means

### ✅ CORRECT Approach

**Output**: List of 20 individual `EntityMask` objects

```python
[
  EntityMask(entity_id=0, mask=roof_1_mask, bbox=(10, 20, 50, 60), ...),
  EntityMask(entity_id=1, mask=roof_2_mask, bbox=(70, 30, 110, 70), ...),
  EntityMask(entity_id=2, mask=roof_3_mask, bbox=(120, 40, 160, 80), ...),
  ...
  EntityMask(entity_id=19, mask=roof_20_mask, bbox=(800, 400, 850, 450), ...)
]
```

Each roof maintains its own:
- Binary mask (pixel-perfect boundaries)
- Entity ID (0-19)
- Bounding box
- Centroid
- Area
- CLIP similarity score

### ❌ WRONG Approach

**DO NOT** create a single combined mask like this:

```python
# WRONG! Do not do this!
combined_mask = np.zeros_like(...)
for mask in masks:
    combined_mask |= mask  # This merges them!
return combined_mask  # Bad! We lost individual masks!
```

---

## Why Separate Masks Matter

1. **Selective Editing**: User can edit specific roofs individually
2. **Quality Control**: Validate each roof separately
3. **Metadata Tracking**: Know which mask corresponds to which roof
4. **Debugging**: Easier to debug issues with specific roofs
5. **Future Features**: Enable features like "edit only the 3 largest roofs"

---

## Implementation Guide

### Stage 3: SAM Segmentation
- Output: `List[np.ndarray]` - 20 separate masks ✅

### Stage 4: CLIP Filtering
- Input: List of masks
- Output: `List[Tuple[int, float, np.ndarray]]` - filtered masks with scores ✅

### Stage 5: Organization (NOT Aggregation!)
- Input: List of individual masks
- Output: `List[EntityMask]` - organized with metadata ✅
- **DO NOT** merge into single mask!

### Stage 6: Validation
- For visualization ONLY: Can create temporary combined mask
- But preserve original separate masks!

### Stage 7: Orchestrator
- Final output: `List[EntityMask]` - 20 separate masks ✅

---

## Validation with VLM

When validating, we can create a visualization:

```python
# For validation visualization ONLY (temporary)
vis_mask = np.zeros_like(image_shape)
for entity_mask in entity_masks:
    vis_mask |= entity_mask.mask

# Show to VLM for validation
overlay = create_overlay(image, vis_mask)
vlm_result = validate(overlay)

# But the actual output remains:
# entity_masks = [EntityMask(...), EntityMask(...), ...]
```

---

## Expected Output Structure

```python
class EntityMask:
    mask: np.ndarray          # Binary mask (H x W)
    entity_id: int            # Unique ID (0-19)
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    centroid: Tuple[int, int] # (cx, cy)
    area: int                 # Number of pixels
    similarity_score: float   # CLIP similarity (0.0-1.0)

# Final output
pipeline_result = {
    'entity_masks': List[EntityMask],  # 20 separate masks!
    'total_entities': 20,
    'validation': {
        'confidence': 0.92,
        'feedback': 'All 20 roofs correctly masked'
    }
}
```

---

## Testing

**Unit Test for Stage 5**:
```python
def test_masks_remain_separate():
    # Input: 5 masks
    masks = [create_test_mask(i) for i in range(5)]

    # Process
    entity_masks = organize_masks(masks, scores)

    # Verify
    assert len(entity_masks) == 5  # Still 5 separate!
    assert all(isinstance(m, EntityMask) for m in entity_masks)

    # Check they're not merged
    for i, em in enumerate(entity_masks):
        assert em.entity_id == i
        assert em.mask.shape == masks[i].shape
        assert np.array_equal(em.mask, masks[i])  # Original preserved!
```

---

## Summary

**REMEMBER**:
- ✅ Keep masks separate throughout the pipeline
- ✅ Each roof gets its own `EntityMask` object
- ✅ Track metadata (ID, bbox, centroid, score)
- ❌ DO NOT merge masks into one
- ⚠️ Visualization combines masks temporarily, but originals stay separate

**Final deliverable**: `List[EntityMask]` with 20 individual masks, not a single merged mask!

---

**This is a hard requirement. Qwen must implement this correctly.**
