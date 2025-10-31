# Edit Validation & Auto-Correction Research

**Purpose**: Design a system for automatic edit validation, quality determination, and feedback-driven correction loops

**Context**: Extension to EDI Vision Pipeline v2 for post-edit quality assessment and iterative refinement

**Related Documents**:
- `PRD.md` - Feature 4: Validation Loop with User Preference Learning
- `HLD.md` - Section 3.2: Validation & Retry Loop
- `VISION_PIPELINE_RESEARCH.md` - Base pipeline architecture

---

## 1. Problem Statement

### Current State
The EDI Vision Pipeline v2 successfully detects and segments entities before editing, but lacks:
1. **Post-edit validation**: No automatic quality check after ComfyUI/Blender/Krita applies edits
2. **Edit quality metrics**: No quantitative measure of how well the edit matches user intent
3. **Auto-correction feedback**: No structured data to guide reasoning subsystem for iterative improvements
4. **Subsystem integration**: No standardized interface between vision, reasoning, and client subsystems

### Required Capabilities
Based on PRD Feature 4, the system must:
- **Detect edit validity**: Valid / Invalid / Partially Valid classification
- **Quantify edit quality**: Alignment score (0.0-1.0) with component breakdown
- **Generate structured feedback**: Machine-readable corrections for reasoning subsystem
- **Enable iterative refinement**: Support max 3 automatic retry attempts
- **Learn from corrections**: Track which corrections improve results

---

## 2. Architecture Design

### 2.1 Edit Validation Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    EDIT VALIDATION SYSTEM                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input: Original Image + Edited Image + User Intent        │
│                           │                                 │
│            ┌──────────────┴──────────────┐                 │
│            │                             │                 │
│     ┌──────▼──────┐              ┌──────▼──────┐          │
│     │   Vision    │              │   Semantic  │          │
│     │   Delta     │              │   Alignment │          │
│     │  Analysis   │              │   Analysis  │          │
│     └──────┬──────┘              └──────┬──────┘          │
│            │                             │                 │
│            │   ┌─────────────────────┐   │                 │
│            └───►   Quality Scoring   ◄───┘                 │
│                └──────────┬──────────┘                     │
│                           │                                 │
│                ┌──────────▼──────────┐                     │
│                │ Feedback Generation │                     │
│                └──────────┬──────────┘                     │
│                           │                                 │
│            ┌──────────────┴──────────────┐                 │
│            │                             │                 │
│     ┌──────▼──────┐              ┌──────▼──────┐          │
│     │   Vision    │              │  Reasoning  │          │
│     │  Subsystem  │              │  Subsystem  │          │
│     │  Hints      │              │   Hints     │          │
│     └─────────────┘              └─────────────┘          │
│                                                             │
│  Output: ValidationReport + CorrectionHints                 │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Component Responsibilities

#### A. Vision Delta Analysis
**Purpose**: Detect what changed visually between original and edited images

**Method**:
1. **Entity Re-detection**:
   - Run SAM + CLIP on edited image (same as pre-edit)
   - Match entities by spatial overlap (IoU > 0.5)
   - Classify changes: `preserved`, `modified`, `removed`, `added`

2. **Change Quantification**:
   - **Color Delta**: ΔE2000 per entity (< 10 = preserved, > 30 = modified)
   - **Shape Delta**: IoU between before/after masks (> 0.85 = preserved)
   - **Position Delta**: Centroid shift (< 5% = preserved)
   - **Texture Delta**: SSIM within mask region

**Output**:
```python
{
    "preserved_entities": [
        {"entity_id": "roof_01", "match_confidence": 0.95, "color_delta": 3.2},
        ...
    ],
    "modified_entities": [
        {"entity_id": "roof_01", "intended": True, "color_delta": 45.7,
         "expected_color": "green", "actual_color": "teal"},
        ...
    ],
    "removed_entities": [],  # Should be empty for good edits
    "added_entities": [],    # New entities (artifacts, unwanted additions)
    "unexpected_changes": [
        {"entity_id": "building_wall", "type": "unintended_modification",
         "severity": "medium", "color_delta": 18.3},
        ...
    ]
}
```

---

#### B. Semantic Alignment Analysis
**Purpose**: Verify the edit matches user's semantic intent using VLM

**Method**:
1. **VLM Comparative Analysis**:
   - Send both images to VLM with specific questions:
     - "In the edited image, are the blue roofs now green as requested?"
     - "Were any other elements changed that shouldn't have been?"
     - "Rate the quality of the green color (natural/artificial/oversaturated)"

2. **Intent Parsing**:
   - Extract original intent from user prompt (DSpy)
   - Compare with actual changes detected
   - Identify mismatches

**Output**:
```python
{
    "intent_match": True/False,
    "intent_confidence": 0.85,
    "semantic_feedback": "Roofs changed to green successfully, but color is slightly oversaturated",
    "quality_ratings": {
        "color_naturalness": 0.7,
        "boundary_precision": 0.9,
        "texture_preservation": 0.85
    },
    "mismatches": [
        {"expected": "all roofs green", "actual": "17/20 roofs green, 3 unchanged"}
    ]
}
```

---

#### C. Quality Scoring Engine
**Purpose**: Compute quantitative alignment score from vision delta + semantic analysis

**Formula** (from PRD Feature 4):
```
Alignment Score = (
    0.4 × (entities_preserved_correctly / total_to_preserve) +
    0.4 × (intended_changes_applied / total_intended) +
    0.2 × (1 - unintended_changes / total_entities)
)
```

**Component Breakdown**:
```python
# 1. Preservation Score (40%)
preservation_score = sum([
    1 if (entity.color_delta < 10 and entity.shape_iou > 0.85)
    else 0
    for entity in preserved_entities
]) / total_entities_to_preserve

# 2. Intended Change Score (40%)
intended_change_score = sum([
    1 if (entity.modified and entity.intended and
          abs(entity.actual_color - entity.expected_color) < threshold)
    else 0
    for entity in modified_entities
]) / total_intended_changes

# 3. Unintended Change Penalty (20%)
unintended_score = 1 - (len(unexpected_changes) / total_entities)

# Final Score
alignment_score = (
    0.4 * preservation_score +
    0.4 * intended_change_score +
    0.2 * unintended_score
)
```

**Validity Classification**:
- `alignment_score >= 0.8`: **VALID** - Auto-accept
- `0.6 <= alignment_score < 0.8`: **PARTIAL** - Ask user
- `alignment_score < 0.6`: **INVALID** - Auto-retry with hints

---

#### D. Feedback Generation
**Purpose**: Generate structured hints for reasoning and vision subsystems

**Output Format**:
```python
@dataclass
class ValidationReport:
    """Complete validation report with actionable feedback."""

    # Classification
    validity: Literal["VALID", "PARTIAL", "INVALID"]
    alignment_score: float
    confidence: float

    # Component scores
    preservation_score: float
    intended_change_score: float
    unintended_change_penalty: float

    # Detailed delta
    vision_delta: VisionDelta
    semantic_analysis: SemanticAnalysis

    # Human-readable feedback
    summary: str  # "17/20 roofs changed to green. 3 roofs missed, likely due to..."

    # Machine-readable hints
    vision_hints: VisionCorrectionHints
    reasoning_hints: ReasoningCorrectionHints


@dataclass
class VisionCorrectionHints:
    """Hints for improving mask detection."""

    adjust_color_threshold: Optional[float]  # "Increase blue HSV range to [85, 50, 50] - [135, 255, 255]"
    adjust_clip_threshold: Optional[float]   # "Lower CLIP threshold to 0.20 to catch missed roofs"
    adjust_sam_params: Optional[Dict]        # "Increase points_per_side to 64"
    missed_regions: List[BBox]               # "Re-run SAM on these regions: [(x, y, w, h), ...]"


@dataclass
class ReasoningCorrectionHints:
    """Hints for improving prompt generation."""

    strengthen_positive: List[str]  # ["Add 'vibrant green'", "Add 'saturated color'"]
    strengthen_negative: List[str]  # ["Add 'preserve walls'", "Add 'do not modify ground'"]
    adjust_guidance_scale: Optional[float]  # "Increase guidance to 9.0 for stronger edits"
    adjust_strength: Optional[float]        # "Increase denoise strength to 0.8"
    regional_prompts: Optional[Dict]        # "Apply 'green roof' only to regions: [mask_ids]"
```

---

## 3. Integration Points

### 3.1 Vision Subsystem → Validation System

**Interface**: `validate_edit(original_image, edited_image, user_intent, entity_masks_before)`

**Called After**: ComfyUI/Blender/Krita returns edited image

**Flow**:
```python
# In orchestrator.py (future implementation)
def process_with_validation(image_path, user_prompt, max_retries=3):
    # Stage 1-6: Detect entities (existing)
    entity_masks = vision_pipeline.process(image_path, user_prompt)

    for attempt in range(max_retries):
        # Send to client subsystem for editing
        edited_image = comfyui_client.apply_edit(
            image_path,
            entity_masks,
            user_prompt
        )

        # Validate the edit
        validation_report = validate_edit(
            original_image=load_image(image_path),
            edited_image=edited_image,
            user_intent=user_prompt,
            entity_masks_before=entity_masks
        )

        # Decision tree
        if validation_report.validity == "VALID":
            return edited_image, validation_report

        elif validation_report.validity == "PARTIAL":
            user_response = ask_user("Edit is partial. Accept? [Y/n/Retry]")
            if user_response == "Y":
                return edited_image, validation_report
            elif user_response == "Retry":
                # Apply hints and retry
                entity_masks = apply_vision_hints(
                    entity_masks,
                    validation_report.vision_hints
                )
                # Continue loop
            else:
                return None, validation_report  # User rejected

        elif validation_report.validity == "INVALID":
            if attempt < max_retries - 1:
                # Auto-retry with corrections
                entity_masks = apply_vision_hints(
                    entity_masks,
                    validation_report.vision_hints
                )
                user_prompt = apply_reasoning_hints(
                    user_prompt,
                    validation_report.reasoning_hints
                )
                continue
            else:
                return None, validation_report  # Max retries exceeded
```

---

### 3.2 Reasoning Subsystem → Validation System

**Purpose**: Reasoning subsystem uses validation feedback to refine prompts

**Interface**: `apply_reasoning_hints(current_prompt_dict, hints) -> refined_prompt_dict`

**Example**:
```python
# Current prompts (from DSpy)
current_prompts = {
    "positive": "green tin roof, photorealistic, 8k",
    "negative": "low quality, artifacts"
}

# Hints from validation
hints = ReasoningCorrectionHints(
    strengthen_positive=["vibrant green", "natural lighting"],
    strengthen_negative=["preserve building walls", "preserve ground texture"],
    adjust_guidance_scale=9.0
)

# Refined prompts
refined_prompts = {
    "positive": "vibrant green tin roof, natural lighting, photorealistic, 8k",
    "negative": "low quality, artifacts, preserve building walls, preserve ground texture",
    "guidance_scale": 9.0  # Increased from 7.5
}
```

---

### 3.3 Client Subsystem (ComfyUI/Blender/Krita) Integration

**Purpose**: Client subsystems receive refined parameters for better results

**ComfyUI Workflow Adjustments**:
```python
# Based on validation hints, adjust workflow parameters
if hints.adjust_strength:
    workflow["denoise_strength"] = hints.adjust_strength

if hints.adjust_guidance_scale:
    workflow["guidance_scale"] = hints.adjust_guidance_scale

if hints.regional_prompts:
    # Apply different prompts to different regions
    for region_id, prompt in hints.regional_prompts.items():
        mask = entity_masks[region_id]
        workflow["regional_conditioning"].append({
            "mask": mask,
            "positive": prompt,
            "strength": 1.0
        })
```

---

## 4. Implementation Strategy

### Phase 1: Core Validation (Week 1)
**Goal**: Implement vision delta analysis and quality scoring

**Deliverables**:
1. `stage7_edit_validation.py` - Vision delta analysis module
2. `validation_scorer.py` - Alignment score computation
3. `test_edit_validation.py` - Unit tests with synthetic edits

**Key Functions**:
```python
def compute_vision_delta(original_image, edited_image, entity_masks_before):
    """Detect what changed between images."""
    pass

def compute_alignment_score(vision_delta, user_intent):
    """Calculate alignment score from delta and intent."""
    pass

def classify_validity(alignment_score):
    """Classify as VALID/PARTIAL/INVALID."""
    pass
```

---

### Phase 2: Feedback Generation (Week 2)
**Goal**: Generate actionable hints for subsystems

**Deliverables**:
1. `feedback_generator.py` - Hint generation logic
2. `hint_applicators.py` - Functions to apply hints
3. Integration with orchestrator.py

**Key Logic**:
```python
def generate_vision_hints(vision_delta, alignment_score):
    """Generate hints to improve mask detection."""
    hints = VisionCorrectionHints()

    # If missed entities
    if vision_delta.intended_change_score < 0.8:
        # Analyze why entities were missed
        missed_roofs = find_missed_entities(vision_delta)

        # Check color threshold
        if color_overlap_low(missed_roofs):
            hints.adjust_color_threshold = current_threshold * 1.1

        # Check CLIP threshold
        if clip_score_borderline(missed_roofs):
            hints.adjust_clip_threshold = current_threshold * 0.9

        # Provide regions to re-scan
        hints.missed_regions = [roof.bbox for roof in missed_roofs]

    return hints
```

---

### Phase 3: Iterative Refinement Loop (Week 3)
**Goal**: Implement max 3 retries with learning

**Deliverables**:
1. Full integration in orchestrator.py
2. Retry logic with hint application
3. User feedback collection

**Retry Strategy**:
```python
Attempt 1: Original masks + prompts
  → If INVALID: Apply vision hints (adjust thresholds)

Attempt 2: Improved masks + original prompts
  → If still INVALID: Apply reasoning hints (refine prompts)

Attempt 3: Improved masks + refined prompts
  → If still INVALID: Present to user with explanation
```

---

### Phase 4: Learning from Corrections (Future)
**Goal**: Track which corrections work best

**Data Collection**:
```python
correction_log = {
    "attempt_1": {
        "hints_applied": ["adjust_clip_threshold: 0.20"],
        "alignment_before": 0.55,
        "alignment_after": 0.72,
        "improvement": 0.17
    },
    "attempt_2": {
        "hints_applied": ["strengthen_positive: vibrant green"],
        "alignment_before": 0.72,
        "alignment_after": 0.88,
        "improvement": 0.16
    }
}
```

**ML Opportunity**: Train lightweight model to predict best hint given (vision_delta, semantic_feedback) → hint effectiveness

---

## 5. Data Structures

### Complete Schema

```python
from dataclasses import dataclass
from typing import List, Optional, Dict, Literal
import numpy as np

@dataclass
class EntityChange:
    """Represents a change to a single entity."""
    entity_id: str
    change_type: Literal["preserved", "modified", "removed", "added"]
    intended: bool
    confidence: float

    # Visual metrics
    color_delta: float  # ΔE2000
    shape_iou: float
    position_delta: float  # Centroid shift percentage
    texture_ssim: float

    # Expected vs actual
    expected_state: Optional[Dict]  # {"color": "green", "texture": "smooth"}
    actual_state: Dict
    mismatch_severity: Literal["none", "low", "medium", "high"]


@dataclass
class VisionDelta:
    """Complete vision-based delta analysis."""
    preserved_entities: List[EntityChange]
    modified_entities: List[EntityChange]
    removed_entities: List[EntityChange]
    added_entities: List[EntityChange]
    unexpected_changes: List[EntityChange]

    # Aggregate metrics
    total_entities_before: int
    total_entities_after: int
    intended_modifications: int
    unintended_modifications: int


@dataclass
class SemanticAnalysis:
    """VLM-based semantic analysis."""
    intent_match: bool
    intent_confidence: float
    semantic_feedback: str

    quality_ratings: Dict[str, float]  # color_naturalness, boundary_precision, etc.
    mismatches: List[Dict[str, str]]

    # Raw VLM response
    vlm_response: str


@dataclass
class VisionCorrectionHints:
    """Actionable hints for vision subsystem."""
    adjust_color_threshold: Optional[float]
    adjust_clip_threshold: Optional[float]
    adjust_sam_params: Optional[Dict]
    missed_regions: List[tuple]  # [(x, y, w, h), ...]
    recommended_rescans: List[str]  # ["region_top_left", "region_bottom_right"]


@dataclass
class ReasoningCorrectionHints:
    """Actionable hints for reasoning subsystem."""
    strengthen_positive: List[str]
    strengthen_negative: List[str]
    adjust_guidance_scale: Optional[float]
    adjust_strength: Optional[float]
    regional_prompts: Optional[Dict[str, str]]


@dataclass
class ValidationReport:
    """Complete validation report."""
    # Classification
    validity: Literal["VALID", "PARTIAL", "INVALID"]
    alignment_score: float
    confidence: float

    # Component scores
    preservation_score: float
    intended_change_score: float
    unintended_change_penalty: float

    # Detailed analysis
    vision_delta: VisionDelta
    semantic_analysis: SemanticAnalysis

    # Feedback
    summary: str
    vision_hints: VisionCorrectionHints
    reasoning_hints: ReasoningCorrectionHints

    # Metadata
    validation_time: float
    attempt_number: int
```

---

## 6. Test Cases

### Test Case 1: Perfect Edit
**Scenario**: All 20 blue roofs changed to green, no other changes

**Expected**:
```python
validation_report = ValidationReport(
    validity="VALID",
    alignment_score=0.95,
    preservation_score=1.0,  # All other entities preserved
    intended_change_score=1.0,  # All roofs changed
    unintended_change_penalty=1.0,  # No unexpected changes
    summary="Perfect edit: 20/20 roofs changed to green"
)
```

---

### Test Case 2: Partial Edit - Missed Entities
**Scenario**: 17/20 roofs changed, 3 missed

**Expected**:
```python
validation_report = ValidationReport(
    validity="PARTIAL",
    alignment_score=0.78,
    preservation_score=1.0,
    intended_change_score=0.85,  # 17/20 = 0.85
    unintended_change_penalty=1.0,
    summary="Partial: 17/20 roofs changed. 3 roofs missed (top-right corner)",
    vision_hints=VisionCorrectionHints(
        missed_regions=[(800, 50, 150, 100), (950, 50, 120, 90), (780, 150, 130, 95)],
        adjust_clip_threshold=0.20  # Lower from 0.22
    )
)
```

---

### Test Case 3: Invalid - Unintended Changes
**Scenario**: Roofs changed but walls also changed

**Expected**:
```python
validation_report = ValidationReport(
    validity="INVALID",
    alignment_score=0.52,
    preservation_score=0.7,  # 3/10 walls changed unintentionally
    intended_change_score=0.9,  # Roofs mostly changed
    unintended_change_penalty=0.7,  # 30% unexpected changes
    summary="Invalid: Walls also changed to greenish tint (unintended)",
    reasoning_hints=ReasoningCorrectionHints(
        strengthen_negative=["preserve building walls", "preserve vertical surfaces"],
        adjust_guidance_scale=7.0  # Lower from 9.0 to reduce spillover
    )
)
```

---

## 7. Performance Considerations

### Validation Speed
- **Vision Delta**: <3 seconds (SAM + CLIP re-analysis)
- **Semantic Analysis**: ~10 seconds (VLM call)
- **Scoring + Hints**: <0.5 seconds (CPU computation)
- **Total**: ~13-15 seconds per validation

### Memory Usage
- Peak VRAM: ~4GB (SAM re-analysis)
- Peak RAM: ~2GB (image comparison operations)

### Optimization Strategies
1. **Cached Entity Detection**: If masks don't need regeneration, skip Stage 1-5
2. **Parallel VLM**: Run semantic analysis while computing vision delta
3. **Incremental Delta**: Only analyze changed regions (diff-based masking)

---

## 8. Future Extensions

### 8.1 Advanced Delta Metrics
- **Perceptual Distance**: LPIPS instead of just color delta
- **Style Transfer Quality**: Compare texture statistics
- **Semantic Consistency**: Verify object identity (roof still looks like roof)

### 8.2 User Feedback Integration
- Track user acceptance patterns
- Learn user-specific quality thresholds
- Build personalized alignment score weights

### 8.3 Multi-Modal Validation
- Audio feedback for accessibility
- Haptic feedback for VR editing
- Real-time validation during interactive editing

---

## 9. Acceptance Criteria

**For Production Deployment**:
- ✅ Validation completes in <15 seconds
- ✅ Alignment score correlates ≥0.8 with human judgment
- ✅ Retry with hints improves score by ≥0.15 on average
- ✅ User override available at all decision points
- ✅ Zero crashes on malformed edits (robust error handling)
- ✅ Hints are actionable (can be applied programmatically)

---

## 10. Integration with EDI Roadmap

**Immediate** (After Stage 9):
- Stage 10: Wildcard testing (validate pipeline robustness)
- Stage 11: Edit validation implementation (this document)

**Near-term** (After validation system):
- ComfyUI client integration (apply edits with validation)
- Reasoning subsystem refinement (use hints for prompt improvement)
- User feedback UI (accept/reject/refine workflow)

**Long-term** (After POC):
- Blender/Krita integration
- Multi-step edit workflows
- Batch processing with validation
- Continuous learning from user corrections

---

**Document Status**: Research Complete - Ready for Implementation
**Next Step**: Create Stage 11 implementation instructions based on this research
