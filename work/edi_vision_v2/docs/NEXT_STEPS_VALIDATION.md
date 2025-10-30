# EDI Vision V2 - Next Steps: Validation & Testing

**Date**: 2025-10-30
**Status**: Research Complete, Ready for Implementation

---

## Executive Summary

Two new capabilities have been designed for EDI Vision Pipeline v2:

1. **Stage 10: Wildcard Robustness Testing** (Easy - Ready for immediate implementation)
   - Test pipeline on 6+ diverse images from `/images/` folder
   - Validate robustness across urban, rural, high-res, dense scenes
   - Establish performance baselines

2. **Edit Validation & Auto-Correction System** (Complex - Research complete, implementation future)
   - Automatic edit quality determination (Valid/Partial/Invalid)
   - Structured feedback for reasoning and client subsystems
   - Iterative refinement with max 3 retries
   - Foundation for closed-loop editing

---

## What Was Delivered

### 1. Stage 10 Wildcard Testing Documentation ✅

**File**: `work/edi_vision_v2/docs/STAGE10_INSTRUCTIONS.md`

**Contents**:
- 6 test scenarios on diverse images:
  1. Multi-color detection (Kolkata cityscape)
  2. Similar adjacent objects (Darjeeling roofs)
  3. High-resolution handling (2MB image)
  4. Dense scenes (Mumbai traffic)
  5. Architectural detail (Pondicherry buildings)
  6. Coastal scenes (sky/water distinction)

- 3 edge cases:
  - No color match
  - Ambiguous semantic query
  - Very small entities

- Comprehensive metrics collection
- WILDCARD_RESULTS.md report template
- Success criteria: 100% no-crash rate, ≥60% detection accuracy

**Implementation Effort**: ~2-3 hours
**Dependencies**: None (uses existing pipeline)
**Priority**: **HIGH** - Should be implemented immediately after Stage 9

---

### 2. Edit Validation Research ✅

**File**: `work/edi_vision_v2/docs/EDIT_VALIDATION_RESEARCH.md`

**Contents**:
- Complete architecture for post-edit validation system
- Vision Delta Analysis (what changed visually)
- Semantic Alignment Analysis (VLM-based intent matching)
- Quality Scoring Engine (alignment score computation)
- Feedback Generation (hints for vision/reasoning subsystems)

**Key Innovations**:
```python
Alignment Score = (
    0.4 × (entities_preserved_correctly / total_to_preserve) +
    0.4 × (intended_changes_applied / total_intended) +
    0.2 × (1 - unintended_changes / total_entities)
)

Classification:
- Score ≥0.8: VALID (auto-accept)
- Score 0.6-0.8: PARTIAL (ask user)
- Score <0.6: INVALID (auto-retry with hints)
```

**Data Structures**:
- `ValidationReport` - Complete validation output
- `VisionCorrectionHints` - Actionable fixes for vision subsystem
- `ReasoningCorrectionHints` - Actionable fixes for reasoning subsystem
- `VisionDelta` - Detailed change analysis

**Integration Points**:
1. Vision Subsystem → Validation System
2. Reasoning Subsystem ← Validation Feedback
3. Client Subsystem (ComfyUI/Blender/Krita) ← Refined Parameters

**Implementation Effort**: ~3 weeks (phased)
**Dependencies**: ComfyUI client integration, reasoning subsystem
**Priority**: **MEDIUM** - After Stage 10, before full EDI integration

---

## Recommended Implementation Order

### Phase 1: Immediate (This Week)
**Goal**: Validate pipeline robustness

**Tasks**:
1. ✅ Read `STAGE10_INSTRUCTIONS.md`
2. ⏳ Implement `tests/test_wildcard.py`
3. ⏳ Run all 6 scenarios + 3 edge cases
4. ⏳ Generate `WILDCARD_RESULTS.md` report
5. ⏳ Tune thresholds based on results

**Deliverables**:
- Working wildcard test suite
- Performance baseline metrics
- Configuration recommendations

**Assignee**: Qwen CLI (supervised by Claude Code)

---

### Phase 2: Near-term (Next 1-2 Weeks)
**Goal**: Implement core validation

**Tasks**:
1. Read `EDIT_VALIDATION_RESEARCH.md`
2. Implement `stage7_edit_validation.py`:
   - `compute_vision_delta()`
   - `compute_alignment_score()`
   - `classify_validity()`
3. Implement `validation_scorer.py`
4. Create synthetic test cases (manually edited images)
5. Unit test validation components

**Deliverables**:
- Vision delta analysis module
- Quality scoring engine
- Test suite with 3 synthetic edits

**Assignee**: Qwen CLI or dedicated developer

---

### Phase 3: Integration (Week 3-4)
**Goal**: Close the validation loop

**Tasks**:
1. Implement `feedback_generator.py`
2. Implement `hint_applicators.py`
3. Integrate with `orchestrator.py`:
   - Add `process_with_validation()` function
   - Implement retry logic (max 3 attempts)
   - Apply vision hints to Stage 2-4 parameters
4. Create ComfyUI mock client for testing
5. End-to-end test: image → masks → mock edit → validation → retry

**Deliverables**:
- Complete validation loop
- Retry with hints working
- Integration tests

**Assignee**: Requires coordination between vision and reasoning subsystems

---

### Phase 4: Future (Post-POC)
**Goal**: Production-ready system

**Tasks**:
1. Real ComfyUI integration
2. User feedback UI (accept/reject/refine)
3. Learning from corrections (track hint effectiveness)
4. Blender/Krita integration
5. Advanced delta metrics (LPIPS, style consistency)

---

## File Summary

### New Documentation Files

| File | Purpose | Status | Lines |
|------|---------|--------|-------|
| `docs/STAGE10_INSTRUCTIONS.md` | Wildcard testing specs | ✅ Ready | 400+ |
| `docs/EDIT_VALIDATION_RESEARCH.md` | Validation architecture | ✅ Complete | 750+ |
| `docs/NEXT_STEPS_VALIDATION.md` | This file | ✅ Done | 200+ |

### Files to Create (Stage 10 Implementation)

| File | Purpose | Estimated Lines |
|------|---------|-----------------|
| `tests/test_wildcard.py` | Wildcard test suite | ~400 |
| `logs/wildcard/` | Test visualizations | N/A (images) |
| `WILDCARD_RESULTS.md` | Test report | ~100 |

### Files to Create (Validation Implementation)

| File | Purpose | Estimated Lines |
|------|---------|-----------------|
| `pipeline/stage7_edit_validation.py` | Vision delta analysis | ~300 |
| `pipeline/validation_scorer.py` | Quality scoring | ~200 |
| `pipeline/feedback_generator.py` | Hint generation | ~250 |
| `pipeline/hint_applicators.py` | Apply hints | ~200 |
| `tests/test_edit_validation.py` | Validation tests | ~300 |

---

## Key Decisions Made

### 1. Validation Scoring Formula
**Decision**: Use PRD Feature 4 formula (40% preservation, 40% intended changes, 20% unintended penalty)

**Rationale**:
- Balances multiple quality dimensions
- Aligns with user expectations (changes should match intent)
- Configurable weights for future tuning

### 2. Retry Strategy
**Decision**: Max 3 attempts with progressive hint application

**Attempts**:
1. Original masks + prompts
2. Improved masks (vision hints) + original prompts
3. Improved masks + refined prompts (reasoning hints)

**Rationale**:
- Gives pipeline multiple chances to succeed
- Learns from failures systematically
- Prevents infinite loops

### 3. Validity Thresholds
**Decision**:
- ≥0.8 = VALID (auto-accept)
- 0.6-0.8 = PARTIAL (ask user)
- <0.6 = INVALID (auto-retry)

**Rationale**:
- Conservative auto-accept prevents bad edits
- User involvement for borderline cases
- Auto-retry for clear failures

### 4. Hint Structure
**Decision**: Separate hints for vision and reasoning subsystems

**Rationale**:
- Different subsystems need different corrections
- Vision hints: threshold adjustments, re-scan regions
- Reasoning hints: prompt refinements, parameter tuning
- Enables targeted improvements

---

## Integration with EDI Architecture

### Current State (Post-Stage 9)
```
User Prompt → Vision Pipeline → Entity Masks
                                     ↓
                                (Future: Client Subsystem)
                                     ↓
                                (Future: Validation)
                                     ↓
                                (Future: Retry Loop)
                                     ↓
                                 Final Edit
```

### Target State (Post-Validation Implementation)
```
User Prompt → Vision Pipeline → Entity Masks
                                     ↓
                         Client Subsystem (ComfyUI/Blender/Krita)
                                     ↓
                              Edited Image
                                     ↓
                          Validation System
                          (Vision Delta + Semantic Analysis)
                                     ↓
                          Quality Scoring
                         (VALID/PARTIAL/INVALID)
                                     ↓
                  ┌──────────────────┼──────────────────┐
                  │                  │                  │
               VALID             PARTIAL           INVALID
                  │                  │                  │
           Auto-accept       Ask user          Auto-retry
                              Accept?          (Apply hints)
                              │  │  │                  │
                           Yes No Retry               │
                            │  │   │                  │
                        Accept │  Loop ───────────────┘
                               │         (Max 3 attempts)
                               └─> User rejects, end
```

---

## Questions for User

### 1. Implementation Priority
**Question**: Should we implement Stage 10 (wildcard testing) immediately, or wait?

**Recommendation**: Implement Stage 10 now - it's quick (2-3 hours) and will reveal any edge cases before moving to validation.

### 2. Validation Timeline
**Question**: Is the 3-week validation implementation timeline acceptable?

**Alternatives**:
- **Faster** (1 week): Skip learning/analytics, basic retry only
- **Slower** (6 weeks): Include user feedback UI, full integration

### 3. Test Image Selection
**Question**: Are the 8 images in `/images/` sufficient for wildcard testing, or should we add more?

**Current images**: Darjeeling, Kolkata, Mumbai, Pondicherry (2x), IP, WP
**Suggested additions**: Outdoor portraits, indoor scenes, night images

### 4. Validation Without Client Subsystem
**Question**: How to test validation without ComfyUI/Blender/Krita integration?

**Options**:
- Mock client (applies simple color transformations)
- Synthetic edits (manually create "edited" images)
- Wait for client integration

---

## Success Metrics

### Stage 10 (Wildcard Testing)
- ✅ 100% no-crash rate on diverse images
- ✅ ≥60% entity detection accuracy
- ✅ All processing times <30 seconds
- ✅ Performance baseline established

### Validation System
- ✅ Alignment score correlates ≥0.8 with human judgment
- ✅ Retry improves score by ≥0.15 on average
- ✅ Validation completes in <15 seconds
- ✅ Hints are actionable (can be applied programmatically)
- ✅ User override available at all decision points

---

## Conclusion

All research and documentation for the next two capabilities is complete:

1. **Stage 10 (Wildcard Testing)**: Ready for immediate implementation
   - Comprehensive test scenarios
   - Clear acceptance criteria
   - Baseline establishment

2. **Edit Validation**: Architecture designed, ready for phased implementation
   - Complete system design
   - Data structures defined
   - Integration points specified
   - 3-week implementation plan

**Recommended Next Step**: Implement Stage 10 (wildcard testing) to validate pipeline robustness before moving to validation system.

---

**Document Status**: Complete ✅
**Ready for**: Qwen CLI implementation (Stage 10) + Developer review (Validation system)
