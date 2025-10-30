# Stage 8: Integration Testing

**Objective**: Validate the complete vision pipeline with integration tests, edge cases, and performance benchmarks.

**Implementation File**: `tests/test_integration.py`

---

## Requirements

### 1. Full Pipeline Integration Test

**Test**: `test_full_pipeline_blue_roofs()`

**Input**:
- Image: `test_image.jpeg`
- Prompt: "turn the blue tin roofs of all those buildings to green"

**Expected Output**:
```python
assert len(result['entity_masks']) >= 14  # At least 14 blue roofs detected
assert result['validation']['confidence'] >= 0.7  # VLM validation passed
assert result['total_time'] < 20.0  # Completed in <20 seconds
assert result['metadata']['stage3_mask_count'] >= 14  # SAM found individual roofs
assert result['metadata']['stage4_filtered_count'] >= 14  # CLIP kept roofs, removed sky
```

**Validation**:
- Each roof has separate mask (not merged)
- No sky in final masks
- All entity_masks have valid bbox, centroid, area
- Intermediate images saved to logs/ (if enabled)

---

### 2. Edge Case Tests

#### Test A: No Color Match
**Test**: `test_pipeline_no_color_match()`

**Input**:
- Image: `test_image.jpeg`
- Prompt: "edit the purple structures"  # No purple in image

**Expected**:
```python
assert len(result['entity_masks']) == 0  # No matches
assert 'stage2_color_mask_coverage' in result['metadata']
assert result['metadata']['stage2_color_mask_coverage'] < 5.0  # <5% coverage
# Pipeline should complete without errors
```

---

#### Test B: Ambiguous Color (Multiple Targets)
**Test**: `test_pipeline_multiple_colors()`

**Input**:
- Image: `test_image.jpeg`
- Prompt: "edit the buildings"  # Generic, no color specified

**Expected**:
```python
# Should use DSpy to extract intent
# Fallback behavior: detect all colored regions or return structured response
assert 'entity_masks' in result
# Should NOT crash
```

---

#### Test C: Very Small Objects
**Test**: `test_pipeline_small_objects()`

**Setup**:
- Create synthetic test image with 50 tiny blue dots (10x10px each)

**Expected**:
```python
# Should filter out objects below min_area threshold
assert len(result['entity_masks']) == 0 or all(
    entity.area >= 500 for entity in result['entity_masks']
)
```

---

#### Test D: Touching Objects (Critical!)
**Test**: `test_pipeline_touching_objects()`

**Validation**:
```python
# Verify touching roofs get SEPARATE masks
entity_masks = result['entity_masks']
assert len(entity_masks) >= 14

# Check entity IDs are unique
entity_ids = [e.entity_id for e in entity_masks]
assert len(entity_ids) == len(set(entity_ids))  # All unique

# Check bboxes don't exactly match (separate objects)
bboxes = [e.bbox for e in entity_masks]
assert len(bboxes) == len(set(bboxes))  # All unique bounding boxes
```

---

### 3. Performance Benchmarks

**Test**: `test_performance_benchmarks()`

**Measure each stage**:
```python
result = pipeline.process(image_path, prompt)
timings = result['stage_timings']

# Stage time limits (RTX 3060 12GB)
assert timings['stage1_entity_extraction'] < 2.0  # <2s
assert timings['stage2_color_filter'] < 0.5  # <500ms
assert timings['stage3_sam_segmentation'] < 8.0  # <8s
assert timings['stage4_clip_filter'] < 3.0  # <3s
assert timings['stage5_organization'] < 0.5  # <500ms
assert timings['stage6_vlm_validation'] < 5.0  # <5s (if enabled)

# Total time
assert result['total_time'] < 20.0  # <20 seconds
```

**Memory benchmark**:
```python
import psutil
import os

process = psutil.Process(os.getpid())
mem_before = process.memory_info().rss / 1024 / 1024  # MB

result = pipeline.process(image_path, prompt)

mem_after = process.memory_info().rss / 1024 / 1024  # MB
mem_increase = mem_after - mem_before

assert mem_increase < 2000  # <2GB RAM increase
```

---

### 4. Validation Without VLM

**Test**: `test_pipeline_no_vlm()`

**Purpose**: Test pipeline works without Ollama running

**Setup**:
```python
pipeline = VisionPipeline(enable_validation=False)
result = pipeline.process(image_path, prompt)
```

**Expected**:
```python
assert 'entity_masks' in result
assert len(result['entity_masks']) >= 14
assert 'validation' not in result  # VLM skipped
# Should complete successfully
```

---

### 5. Intermediate Output Validation

**Test**: `test_intermediate_outputs()`

**Setup**:
```python
pipeline = VisionPipeline(save_intermediate=True, output_dir="logs/test_run")
result = pipeline.process(image_path, prompt)
```

**Verify files created**:
```python
import os
assert os.path.exists("logs/test_run/stage2_color_mask.png")
assert os.path.exists("logs/test_run/stage3_sam_masks.png")
assert os.path.exists("logs/test_run/stage4_clip_filtered.png")
assert os.path.exists("logs/test_run/stage5_entity_masks.png")
assert os.path.exists("logs/test_run/stage6_validation_overlay.png")  # If VLM enabled
assert os.path.exists("logs/test_run/result.json")
```

---

### 6. Error Handling Tests

#### Test A: Missing Image File
**Test**: `test_missing_image()`
```python
with pytest.raises(FileNotFoundError):
    pipeline.process("nonexistent.jpg", "edit blue roofs")
```

#### Test B: Corrupted Image
**Test**: `test_corrupted_image()`
```python
# Create corrupted file
with open("corrupted.jpg", "w") as f:
    f.write("not an image")

with pytest.raises((IOError, ValueError)):
    pipeline.process("corrupted.jpg", "edit blue roofs")
```

#### Test C: Empty Prompt
**Test**: `test_empty_prompt()`
```python
with pytest.raises(ValueError):
    pipeline.process(image_path, "")
```

#### Test D: SAM Out of Memory
**Test**: `test_sam_oom_handling()`
```python
# Test with very large image (8000x6000px)
# Should trigger OOM fallback in stage3_sam_segmentation.py
# Verify it resizes and retries instead of crashing
```

---

## Test Data Requirements

Create `tests/fixtures/` with:

1. **test_image.jpeg** (existing - ~20 blue roofs)
2. **test_no_match.jpeg** - Image with no blue (create or find)
3. **test_small_objects.png** - Synthetic 50 tiny blue dots
4. **test_large.jpg** - 8000x6000px image for OOM test
5. **corrupted.jpg** - Invalid image file

---

## Implementation Checklist

- [ ] Create `tests/test_integration.py`
- [ ] Implement `test_full_pipeline_blue_roofs()`
- [ ] Implement edge case tests (A-D)
- [ ] Implement performance benchmarks
- [ ] Implement VLM-disabled test
- [ ] Implement intermediate output test
- [ ] Implement error handling tests (A-D)
- [ ] Create test fixtures in `tests/fixtures/`
- [ ] Run all integration tests with pytest
- [ ] Generate coverage report
- [ ] Document any failures or performance issues

---

## Acceptance Criteria

**All tests must pass**:
```bash
pytest tests/test_integration.py -v
```

**Coverage targets**:
- Integration tests: 100% of orchestrator.py
- Edge cases: All error paths tested
- Performance: All stages within time limits

**Success Metrics**:
- ✅ Full pipeline detects 14-20 blue roofs
- ✅ Touching objects stay separate
- ✅ Total time <20 seconds
- ✅ Graceful handling of all edge cases
- ✅ All error conditions handled

---

## Deliverables

1. **tests/test_integration.py** - Complete integration test suite
2. **tests/fixtures/** - All test images
3. **Test report** - Pytest output showing all tests passing
4. **Performance metrics** - Timing breakdown for each stage

---

## Notes for Qwen

- Use `pytest` framework with fixtures
- Use `@pytest.mark.slow` for tests >5 seconds
- Use `tmpdir` fixture for temporary output directories
- Mock Ollama API if VLM tests are flaky
- Focus on **realistic scenarios** from actual use cases
- Test the **critical requirement**: Separate masks for touching objects

**Key Validation**: The primary success criterion is detecting ALL blue roofs (14-20) with SEPARATE masks, completing in <20 seconds.
