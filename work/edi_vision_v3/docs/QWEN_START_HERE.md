# Qwen: Start Here - Quick Implementation Guide

**Date**: 2025-10-31
**Your Mission**: Implement EDI Vision v3.0 to achieve 85%+ success rate

---

## 🎯 What You Need to Know

### The Problem
v2.0 has 11% true success rate (1/9 wildcard tests) due to:
1. Color-first assumption (crashes on 60% of queries)
2. Toxic fallback (returns garbage as "success")
3. No semantic-only detection path

### Your Goal
Implement v3.0 with dual-path architecture to achieve 85%+ success rate (7.5/9 tests)

### Your Role
- **Implement** all code following detailed specifications
- **Test** each component thoroughly
- **Report** completion status to Claude for validation
- **Fix** issues identified by Claude

### Claude's Role
- **Supervises** your work (doesn't write code)
- **Validates** your implementation
- **Approves** or requests revisions
- **Provides** detailed specifications and guidance

---

## 📖 Document Reading Order

**Read these documents in this exact order:**

1. **@CRITICAL_ARCHITECTURE_FLAWS.md** (30 min)
   - Understand what failed in v2.0 and why
   - Visual evidence of failures
   - Three solution paths (A, B, C)

2. **@V2_TO_V3_MIGRATION_ANALYSIS.md** (45 min)
   - Component-by-component analysis
   - What to copy vs redesign
   - Detailed implementation specifications

3. **@QWEN_SUPERVISION_PLAN.md** (60 min)
   - Your complete implementation guide
   - Phase-by-phase tasks
   - Validation checkpoints
   - Quality standards

4. **@README.md** (15 min)
   - v3.0 architecture overview
   - Performance targets
   - Success criteria

---

## 🚀 Phase 1: Start Implementation Now

### Task 1.1: Workspace Setup (15 min)

**Execute these commands:**
```bash
cd /home/riju279/Documents/Code/Zonko/EDI/edi/work/edi_vision_v3

# Copy test images
cp -r ../edi_vision_v2/images/ ./images/

# Copy model weights
cp ../edi_vision_v2/sam2.1_b.pt ./

# Create directories
mkdir -p pipeline tests validations logs/{orchestrator,test,wildcard}

# Create __init__.py files
touch pipeline/__init__.py tests/__init__.py validations/__init__.py

# Verify
ls -la
# Expected: docs/, pipeline/, tests/, validations/, logs/, images/, sam2.1_b.pt, README.md
```

**Report to Claude:**
```
Task 1.1 Complete:
- Directories created: [list]
- Images copied: [count] files
- sam2.1_b.pt size: [size in MB]
- Ready for validation: YES
```

---

### Task 1.2: Copy Stage 3 (SAM) with Enhancements (60 min)

**Specification**: See `@QWEN_SUPERVISION_PLAN.md` Task 1.2

**Key Requirements**:
1. Read `@../edi_vision_v2/pipeline/stage3_sam_segmentation.py`
2. Copy to `@pipeline/stage3_sam_segmentation.py`
3. Add 3 diagnostic blocks:
   - Input diagnostics (log image shape, color mask coverage)
   - Fallback SAM (relaxed parameters if 0 masks)
   - Failure diagnostics (detailed error logs)
4. Return empty list (not crash) if SAM fails completely

**Validation Test**:
```python
# Test on known failing image
import numpy as np
from pipeline.stage3_sam_segmentation import segment_regions

image = load_image("images/mumbai-traffic.jpg")
color_mask = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8) * 0.5  # 50% mask

result = segment_regions(image, color_mask, min_area=500)

# Should NOT crash, should log diagnostics
# Check logs for "SAM input diagnostics", "SAM fallback", etc.
```

**Report to Claude:**
```
Task 1.2 Complete:
- File created: pipeline/stage3_sam_segmentation.py ([X] lines)
- Enhancements added:
  - [ ] Input diagnostics
  - [ ] Fallback SAM with relaxed parameters
  - [ ] Failure diagnostics
- Test result: [PASS/FAIL]
- Issues encountered: [none/describe]
- Ready for validation: YES
```

---

### Task 1.3: Copy Stage 4 (CLIP) with Adaptive Threshold (45 min)

**Specification**: See `@QWEN_SUPERVISION_PLAN.md` Task 1.3

**Key Requirements**:
1. Read `@../edi_vision_v2/pipeline/stage4_clip_filter.py`
2. Copy to `@pipeline/stage4_clip_filter.py`
3. Add `get_adaptive_clip_threshold(detection_strategy)` function
4. Modify `filter_masks()` to accept `detection_strategy` parameter
5. Use adaptive threshold: color_guided=0.22, semantic_only=0.18, hybrid=0.20

**Validation Test**:
```python
from pipeline.stage4_clip_filter import filter_masks, get_adaptive_clip_threshold

# Test adaptive thresholds
assert get_adaptive_clip_threshold("color_guided") == 0.22
assert get_adaptive_clip_threshold("semantic_only") == 0.18
assert get_adaptive_clip_threshold("hybrid") == 0.20

# Test filter_masks with different strategies
# (semantic_only should return MORE entities due to lower threshold)
```

**Report to Claude:**
```
Task 1.3 Complete:
- File created: pipeline/stage4_clip_filter.py ([X] lines)
- Functions added:
  - [ ] get_adaptive_clip_threshold()
  - [ ] Modified filter_masks() signature
- Test result: [PASS/FAIL]
- Ready for validation: YES
```

---

### Task 1.4: Copy Stage 5 (Mask Organization) As-Is (10 min)

**Specification**: See `@QWEN_SUPERVISION_PLAN.md` Task 1.4

**Key Requirement**: Copy exactly, no modifications

```bash
cp ../edi_vision_v2/pipeline/stage5_mask_organization.py pipeline/

# Verify identical
diff ../edi_vision_v2/pipeline/stage5_mask_organization.py pipeline/stage5_mask_organization.py
# Expected: No differences
```

**Report to Claude:**
```
Task 1.4 Complete:
- File copied: pipeline/stage5_mask_organization.py
- Verification: Identical to v2.0
- Ready for validation: YES
```

---

### Task 1.5: Copy Stage 6 (VLM) with Structured Feedback (90 min)

**Specification**: See `@QWEN_SUPERVISION_PLAN.md` Task 1.5

**Key Requirements**:
1. Read `@../edi_vision_v2/pipeline/stage6_vlm_validation.py`
2. Copy to `@pipeline/stage6_vlm_validation.py`
3. Enhance `ValidationResult` dataclass with 4 new fields:
   - `missed_entities: List[str]`
   - `false_positive_entities: List[str]`
   - `correction_hints: Dict[str, Any]`
   - `spatial_accuracy: Optional[float]`
4. Parse structured feedback from VLM response using regex

**Validation Test**:
```python
from pipeline.stage6_vlm_validation import validate_with_vlm, ValidationResult

# Test on sample case
result = validate_with_vlm(image, masks, "highlight red vehicles")

# Check new fields are populated
assert isinstance(result.missed_entities, list)
assert isinstance(result.false_positive_entities, list)
assert isinstance(result.correction_hints, dict)
```

**Report to Claude:**
```
Task 1.5 Complete:
- File created: pipeline/stage6_vlm_validation.py ([X] lines)
- ValidationResult enhanced with 4 fields
- Structured feedback parsing implemented
- Test result: [PASS/FAIL]
- Ready for validation: YES
```

---

## ⏭️ What Comes Next

After completing Tasks 1.1-1.5, you will implement:

**Task 2.1**: Dynamic Color Mapper (2-3 hours)
- NEW component using DSpy + LLM
- Replaces static color dictionary
- Most critical component for v3.0

**Task 2.2**: Enhanced Intent Parser (2 hours)
- Adds detection strategy routing
- Key for dual-path orchestrator

**Task 2.3**: Post-Filters (1-2 hours)
- Size/position/shape filtering
- Improves semantic-only path

**Task 2.4**: Dual-Path Orchestrator (3-4 hours)
- Routes to appropriate detection path
- Merges results for hybrid queries

---

## 📋 Quality Checklist (Use This for EVERY Task)

Before reporting completion, verify:

### Code Quality
- [ ] Black formatting (max line 100)
- [ ] Type hints on all functions
- [ ] Google-style docstrings
- [ ] No hardcoded paths or magic numbers
- [ ] Comprehensive logging (DEBUG, INFO, WARNING, ERROR)

### Functionality
- [ ] All required functions implemented
- [ ] All enhancements added
- [ ] No regressions (v2 functionality preserved)
- [ ] Graceful error handling (no crashes)

### Testing
- [ ] Validation test passes
- [ ] Edge cases handled (empty inputs, invalid inputs)
- [ ] Performance acceptable (<5s for color, <8s for semantic)
- [ ] Memory usage reasonable (<10GB GPU)

### Documentation
- [ ] Docstrings complete and accurate
- [ ] Comments explain non-obvious decisions
- [ ] TODO markers for future enhancements
- [ ] No commented-out code (remove or explain)

---

## 🆘 When You Get Stuck

### If Code Doesn't Work
1. **Check logs**: Set logging level to DEBUG
2. **Print intermediate values**: Use print() liberally
3. **Test components individually**: Don't test entire pipeline at once
4. **Simplify**: Create minimal test case that reproduces issue

### If You're Unsure About Spec
1. **Re-read specification**: Often contains the answer
2. **Check v2 implementation**: See how it was done before
3. **Ask Claude specific question**: "Why does X fail with error Y?"
4. **Don't guess**: Better to ask than implement incorrectly

### If Tests Fail
1. **Document the failure**: Error message, stack trace, input values
2. **Investigate root cause**: What assumption was wrong?
3. **Propose fix**: Explain what needs to change
4. **Get approval**: Don't proceed until Claude validates fix

---

## 💬 Communication Template

### After Each Task
```
=== TASK [X.Y] COMPLETION REPORT ===

Task: [Name]
Status: COMPLETE / NEEDS REVISION

Files Created/Modified:
- [path/to/file.py] ([X] lines)

Enhancements Implemented:
- [X] [Enhancement 1]
- [X] [Enhancement 2]
- [ ] [Enhancement 3 - SKIPPED, reason: ...]

Tests Run:
- [Test 1]: PASS
- [Test 2]: FAIL (error: ...)

Performance:
- Execution time: [X]s
- Memory usage: [X]GB

Issues Encountered:
- [Issue 1]: [How resolved]
- [Issue 2]: [Needs guidance]

Ready for Validation: YES / NO (if NO, explain why)

=== END REPORT ===
```

---

## 🎯 Remember

1. **Quality > Speed**: Don't rush, get it right the first time
2. **Test Everything**: Write tests as you go, not after
3. **Log Everything**: When in doubt, add more logging
4. **Ask Questions**: Claude is here to help when stuck
5. **Document Issues**: Help future developers understand decisions

---

## 🚦 Current Status

- [x] Documentation complete
- [x] v3 workspace created
- [ ] **YOU ARE HERE**: Task 1.1 (Workspace Setup)
- [ ] Tasks 1.2-1.5 (Copy working components)
- [ ] Tasks 2.1-2.4 (Implement redesigned components)
- [ ] Phase 2: Integration (Week 2)
- [ ] Phase 3: Validation system (Week 3)
- [ ] Phase 4: Production (Week 4)

---

**Now**: Execute Task 1.1 and report completion to Claude.

**Goal**: By end of Phase 1, achieve 5-6/9 wildcard test success (up from 1/9).

*"Every line of code you write brings us closer to 85% success rate. Let's make v3.0 production-grade!"*
