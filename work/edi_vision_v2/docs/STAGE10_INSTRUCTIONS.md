# Stage 10: Wildcard Robustness Testing

**Objective**: Test the vision pipeline on diverse real-world images to validate robustness, identify edge cases, and establish baseline performance metrics across different scenarios.

**Implementation File**: `tests/test_wildcard.py`

---

## Overview

Stage 10 extends the integration testing from Stage 8 to include diverse, unseen images beyond the original test case. This validates the pipeline's ability to generalize across:

- Different image compositions (urban, rural, aerial, close-up)
- Various lighting conditions (daylight, golden hour, overcast, indoor)
- Multiple entity types (buildings, vehicles, people, objects, nature)
- Different image sizes and resolutions
- Edge cases (crowded scenes, minimal color contrast, similar adjacent objects)

---

## Test Image Dataset

### Available Images from `/images/` Folder

| Image | Description | Primary Colors | Entities | Difficulty |
|-------|-------------|----------------|----------|------------|
| `IP.jpeg` | Mountain village (test baseline) | Blue roofs, sky | Buildings, roofs | Medium |
| `Darjeeling.jpg` | Mountain settlement | Orange/brown roofs | Buildings, mountains | Medium |
| `kol_1.png` | Kolkata cityscape | Mixed urban colors | Buildings, vehicles | Hard |
| `mumbai-traffic.jpg` | Mumbai traffic scene | Red, yellow vehicles | Cars, buses, auto-rickshaws | Hard |
| `pondi_2.jpg` | Pondicherry coastal | Blue sky, buildings | Colonial buildings, palm trees | Medium |
| `Pondicherry.jpg` | Pondicherry street | Yellow, white buildings | Buildings, streets | Easy |
| `WP.jpg` | Large resolution image | TBD | TBD | Unknown (2MB file) |

---

## Test Scenarios

### Scenario 1: Multi-Color Detection (kol_1.png)

**Prompt**: `"highlight all red vehicles"`

**Expected Behavior**:
- Detect 5-15 red vehicles (cars, buses, autos)
- Filter out red building elements
- Maintain separate masks for adjacent vehicles
- Handle partial occlusion (vehicles behind others)

**Success Criteria**:
```python
assert len(result['entity_masks']) >= 5  # At least 5 red vehicles
assert result['metadata']['color_coverage'] > 2.0  # Red covers >2% of image
assert result['validation']['confidence'] >= 0.6  # Moderate confidence
```

---

### Scenario 2: Similar Adjacent Objects (Darjeeling.jpg)

**Prompt**: `"edit brown roofs"`

**Expected Behavior**:
- Detect 8-12 brown/orange tin roofs
- Maintain separate masks despite touching buildings
- Distinguish roofs from ground terrain (similar color)
- Filter out background mountains

**Success Criteria**:
```python
assert len(result['entity_masks']) >= 8  # Multiple separate roofs
# Verify masks are separate (check entity_ids are unique)
entity_ids = [e.entity_id for e in result['entity_masks']]
assert len(entity_ids) == len(set(entity_ids))  # All unique
```

---

### Scenario 3: High-Resolution Image (WP.jpg)

**Prompt**: `"edit sky regions"`

**Expected Behavior**:
- Handle 2MB image without OOM errors
- Auto-resize if necessary (per Stage 3 specs)
- Complete processing in <30 seconds
- Maintain mask precision after resizing

**Success Criteria**:
```python
assert result['success'] == True  # No OOM crash
assert result['metadata']['total_time'] < 30.0  # Within time budget
assert len(result['entity_masks']) >= 1  # Sky detected
```

---

### Scenario 4: Dense Scene (mumbai-traffic.jpg)

**Prompt**: `"detect yellow auto-rickshaws"`

**Expected Behavior**:
- Handle crowded scene with 20+ vehicles
- Distinguish yellow autos from yellow taxis/buses
- Filter out yellow road markings
- Handle motion blur (if present)

**Success Criteria**:
```python
assert len(result['entity_masks']) >= 3  # At least 3 autos
assert result['metadata']['clip_filtered_count'] > 0  # CLIP filtered properly
# False positive rate check
false_positive_entities = [e for e in result['entity_masks'] if e.area < 500]
assert len(false_positive_entities) < len(result['entity_masks']) * 0.2  # <20% small noise
```

---

### Scenario 5: Architectural Detail (Pondicherry.jpg)

**Prompt**: `"highlight yellow colonial buildings"`

**Expected Behavior**:
- Detect 2-4 yellow buildings
- Maintain building boundaries (not merge with adjacent)
- Filter out yellow street elements
- Handle architectural texture (windows, doors)

**Success Criteria**:
```python
assert len(result['entity_masks']) >= 2  # Multiple buildings
# Check building size consistency (should be large regions)
avg_area = np.mean([e.area for e in result['entity_masks']])
assert avg_area > 10000  # Buildings are large entities
```

---

### Scenario 6: Coastal Scene (pondi_2.jpg)

**Prompt**: `"edit blue sky"`

**Expected Behavior**:
- Detect large sky region (potentially 40-60% of image)
- Filter out blue water/ocean (semantic difference)
- Handle horizon line cleanly
- Manage color gradients (sky darkens near horizon)

**Success Criteria**:
```python
assert len(result['entity_masks']) == 1  # Single sky mask
sky_mask = result['entity_masks'][0]
assert sky_mask.area > image_pixels * 0.3  # Sky is >30% of image
# Check sky is in upper portion
assert sky_mask.centroid[1] < image_height * 0.5  # Y-coordinate in upper half
```

---

## Edge Case Testing

### Edge Case 1: No Color Match

**Prompt**: `"edit purple elements"` (on image with no purple)

**Expected**:
```python
assert len(result['entity_masks']) == 0
assert result['metadata']['color_coverage'] < 1.0  # <1% coverage
assert result['success'] == True  # Pipeline doesn't crash
```

---

### Edge Case 2: Ambiguous Semantic Query

**Prompt**: `"edit interesting objects"` (vague semantic)

**Expected**:
```python
# Pipeline should complete, but results may be inconsistent
assert result['success'] == True
assert 'intent' in result['metadata']
assert result['metadata']['intent']['confidence'] < 0.7  # Low confidence expected
```

---

### Edge Case 3: Very Small Entities

**Prompt**: `"detect small birds"` (if present in any image)

**Expected**:
```python
# Should filter by min_area threshold
if len(result['entity_masks']) > 0:
    assert all(e.area >= 500 for e in result['entity_masks'])  # Min area enforced
```

---

## Implementation Checklist

### Test File Structure

```python
# tests/test_wildcard.py

import pytest
import os
from pathlib import Path

@pytest.mark.slow
class TestWildcardRobustness:
    """Wildcard robustness tests on diverse images."""

    @pytest.fixture(scope="class")
    def images_dir(self):
        return Path("/home/riju279/Documents/Code/Zonko/EDI/edi/images/")

    @pytest.fixture(scope="class")
    def pipeline(self):
        return VisionPipeline(enable_validation=False, save_intermediate=False)

    def test_scenario_1_multi_color_detection(self, pipeline, images_dir):
        """Test multi-color vehicle detection in urban scene."""
        # Implementation
        pass

    def test_scenario_2_similar_adjacent_objects(self, pipeline, images_dir):
        """Test roof detection with touching buildings."""
        pass

    # ... Additional test methods
```

---

### Metrics to Collect

For each test scenario, collect:

```python
{
    "image": "kol_1.png",
    "prompt": "highlight red vehicles",
    "metrics": {
        "entities_detected": 12,
        "color_coverage_percent": 4.2,
        "total_time_seconds": 18.5,
        "stage_timings": {
            "stage1": 0.005,
            "stage2": 0.003,
            "stage3": 6.8,
            "stage4": 1.9,
            "stage5": 0.07
        },
        "clip_filter_rate": 0.15,  # 15% of SAM masks filtered
        "success": True,
        "validation_confidence": 0.72
    }
}
```

---

## Acceptance Criteria

### Quantitative Metrics

| Metric | Target | Rationale |
|--------|--------|-----------|
| Success rate (no crashes) | 100% | Pipeline must be robust |
| Entity detection rate | ≥60% | Reasonable for diverse scenes |
| False positive rate | <20% | More false negatives acceptable than false positives |
| Processing time (no VLM) | <25s | Slightly higher than test image due to complexity |
| Memory usage | <10GB VRAM | Within GPU constraints |

### Qualitative Assessment

For each test image, manually verify:
- ✅ Masks are **visually correct** (check intermediate outputs in `logs/`)
- ✅ Separate masks maintained for touching objects
- ✅ No catastrophic failures (entire image masked, pipeline hangs, etc.)
- ✅ Error messages are informative if failure occurs

---

## Deliverables

1. **`tests/test_wildcard.py`** - Complete test suite with 6 scenarios + 3 edge cases
2. **`logs/wildcard/`** - Intermediate visualizations for each test case
3. **`WILDCARD_RESULTS.md`** - Summary report with metrics table
4. **Performance baseline** - JSON file with timing/accuracy metrics for future comparison

---

## Example WILDCARD_RESULTS.md Format

```markdown
# Wildcard Robustness Test Results

**Test Date**: YYYY-MM-DD
**Hardware**: RTX 3060 12GB, 32GB RAM
**Pipeline Version**: v2.0.0

## Summary

| Image | Prompt | Entities | Time (s) | Success | Notes |
|-------|--------|----------|----------|---------|-------|
| kol_1.png | "red vehicles" | 12 | 18.5 | ✅ | Detected cars, buses, autos |
| Darjeeling.jpg | "brown roofs" | 10 | 17.2 | ✅ | Separate masks maintained |
| mumbai-traffic.jpg | "yellow autos" | 8 | 21.3 | ✅ | Dense scene handled well |
| WP.jpg | "sky" | 1 | 28.1 | ✅ | Auto-resized from 6000x4000 |
| Pondicherry.jpg | "yellow buildings" | 3 | 15.8 | ✅ | Clean architectural masks |
| pondi_2.jpg | "blue sky" | 1 | 16.4 | ✅ | Distinguished from water |

**Overall Success Rate**: 100% (6/6 scenarios passed)
**Average Processing Time**: 19.6 seconds
**Total Test Runtime**: 2.1 minutes

## Observations

### Strengths
- Robust handling of diverse image types
- Consistent performance across resolutions
- No OOM errors even on 2MB images
- Separate mask requirement maintained in all cases

### Weaknesses
- CLIP threshold (0.22) caused false positives in dense scenes
- Color gradients (sky) sometimes split into multiple masks
- Small objects (<500px) correctly filtered but user may expect detection

### Recommendations
1. Increase CLIP threshold to 0.25 for production use
2. Add gradient-aware merging for natural scenes (sky, water)
3. Add user-configurable min_area threshold in config.yaml
4. Consider adaptive thresholds based on scene complexity
```

---

## Notes for Implementation

1. **Use `@pytest.mark.slow`** - These tests take 5-10 minutes total
2. **Skip if images missing** - Use `@pytest.mark.skipif(not image.exists())`
3. **Save all intermediate outputs** - Enable `save_intermediate=True` for debugging
4. **Disable VLM validation** - Too slow for automated testing, use `enable_validation=False`
5. **Collect comprehensive logs** - Write all metrics to JSON for analysis

---

## Future Extensions

After Stage 10 completion, consider:
- **Automated dataset expansion**: Scrape creative commons images
- **Continuous integration**: Run wildcard tests on every commit
- **Performance regression testing**: Compare metrics against baseline
- **User feedback loop**: Let users submit challenging images for testing

---

**Success Metric**: Pipeline handles 100% of test scenarios without crashes, with ≥60% detection accuracy on diverse real-world images.
