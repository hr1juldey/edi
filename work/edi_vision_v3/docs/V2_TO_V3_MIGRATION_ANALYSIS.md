# V2 to V3 Migration Analysis

**Document Version**: 1.0
**Date**: 2025-10-31
**Purpose**: Detailed analysis of what to copy, what to redesign, and why

---

## Executive Summary

**v2.0 Success**: 11% true success rate (1/9 wildcard tests)
**v3.0 Target**: 85%+ success rate (7.5+/9 wildcard tests)

**Strategy**: Keep working components, redesign broken architecture, add missing capabilities

---

## Component Analysis Matrix

| Component | Status in v2 | Action for v3 | Priority | Reasoning |
|-----------|-------------|---------------|----------|-----------|
| **stage3_sam_segmentation.py** | ✅ Working | COPY with diagnostics | P1 | SAM works perfectly when given good input |
| **stage4_clip_filter.py** | ✅ Working | COPY with tuning | P1 | CLIP filtering is sound, just needs threshold adjustment |
| **stage5_mask_organization.py** | ✅ Working | COPY as-is | P2 | Mask organization logic is correct |
| **stage6_vlm_validation.py** | ✅ Working | COPY + enhance | P2 | VLM validation provides valuable feedback |
| **stage2_color_filter.py** | ❌ BROKEN | REDESIGN completely | P0 | Toxic fallback, static dictionary, color-first assumption |
| **stage1_entity_extraction.py** | ⚠️ Limited | ENHANCE | P0 | Works but doesn't detect query type for routing |
| **orchestrator.py** | ⚠️ Limited | REDESIGN | P0 | No path routing, no strategy selection |
| **app.py (CLI)** | ✅ Working | COPY with updates | P2 | CLI interface is production-ready |
| **tui.py (TUI)** | ✅ Working | COPY with updates | P2 | TUI interface is functional |
| **config.yaml** | ⚠️ Limited | REDESIGN | P1 | Need dynamic color system config |
| **tests/** | ✅ Working | COPY + expand | P1 | Test infrastructure is solid, need more tests |
| **validations/** | ✅ Working | COPY as-is | P2 | Validation scripts are correct |

---

## What to COPY (Reusable Components)

### 1. Stage 3: SAM Segmentation (Copy with Enhancements)

**File**: `stage3_sam_segmentation.py`

**Why it works**:
- SAM 2.1 integration is correct
- Handles color mask overlap properly
- Filters by minimum area correctly
- Memory management is sound (GPU cleanup)

**What to add in v3**:
```python
# Add comprehensive diagnostics
def segment_regions(image, color_mask, min_area=100, color_overlap_threshold=0.5):
    logging.info(f"SAM input: image_shape={image.shape}, color_mask_coverage={color_mask.mean():.2%}")

    # Try primary parameters
    masks = sam_generator.generate(image)

    # ADD: Fallback with relaxed parameters if 0 masks
    if len(masks) == 0:
        logging.warning("SAM generated 0 masks, trying relaxed parameters")
        sam_generator_fallback = build_sam_generator(
            pred_iou_thresh=0.70,  # was 0.88
            stability_score_thresh=0.85,  # was 0.95
        )
        masks = sam_generator_fallback.generate(image)

    # ADD: Detailed diagnostics if still 0 masks
    if len(masks) == 0:
        log_detailed_diagnostics(image, color_mask)
```

**Action for Qwen**: Copy file, add diagnostic blocks, test on mumbai-traffic.jpg

---

### 2. Stage 4: CLIP Filtering (Copy with Threshold Tuning)

**File**: `stage4_clip_filter.py`

**Why it works**:
- CLIP model loading is correct
- Semantic matching logic is sound
- Processes masks in batches efficiently

**What to tune in v3**:
```python
# Current: Fixed threshold 0.22
similarity_threshold = 0.22

# v3: Adaptive threshold based on query type
def get_adaptive_threshold(detection_strategy):
    if detection_strategy == "color_guided":
        return 0.22  # High precision for color-guided
    elif detection_strategy == "semantic_only":
        return 0.18  # More permissive for semantic
    elif detection_strategy == "hybrid":
        return 0.20  # Middle ground
```

**Action for Qwen**: Copy file, add adaptive threshold logic, test on semantic queries

---

### 3. Stage 5: Mask Organization (Copy As-Is)

**File**: `stage5_mask_organization.py`

**Why it works**:
- Correctly creates separate masks for each entity
- Metadata generation is complete
- Bounding box calculations are accurate

**Action for Qwen**: Copy file directly, no changes needed

---

### 4. Stage 6: VLM Validation (Copy + Enhance)

**File**: `stage6_vlm_validation.py`

**Why it works**:
- Ollama integration is correct
- Validation overlay visualization works well
- JSON parsing with retry is robust

**What to enhance in v3**:
```python
# Add validation confidence scoring
@dataclass
class ValidationResult:
    covers_all_targets: bool
    confidence: float
    feedback: str

    # ADD: More structured feedback for auto-correction
    missed_entities: List[str]  # Entities user mentioned but not masked
    false_positive_entities: List[str]  # Masked entities not mentioned
    correction_hints: Dict[str, Any]  # Hints for pipeline adjustment
```

**Action for Qwen**: Copy file, add structured feedback fields, integrate with validation system

---

### 5. CLI & TUI Interfaces (Copy with Updates)

**Files**: `app.py`, `tui.py`, `tui.tcss`

**Why they work**:
- CLI argument parsing is comprehensive
- TUI layout and navigation are intuitive
- Visualization (2x2 grid) is effective
- Error handling is solid

**What to update in v3**:
- Add `--detection-strategy` flag (color-guided, semantic-only, hybrid, auto)
- Update help text to mention new capabilities
- Add progress indicator for slower semantic-only path
- Show confidence scores in TUI

**Action for Qwen**: Copy files, update CLI args and TUI display for new features

---

### 6. Test Infrastructure (Copy + Expand)

**Files**: All `tests/*.py` files

**Why it works**:
- Pytest fixtures are well-designed
- Integration tests cover happy paths
- Validation scripts are thorough

**What to add in v3**:
- Ground truth dataset (50+ annotated images)
- Precision/recall metrics
- Performance benchmarks
- Expanded wildcard tests (semantic queries, hybrid queries)

**Action for Qwen**: Copy test files, add new test categories

---

## What to REDESIGN (Broken Components)

### 1. Stage 2: Color Filtering (Complete Redesign) - P0 CRITICAL

**Current Issues**:
```python
# BROKEN: Toxic fallback
if color_name not in color_ranges:
    return np.ones((image.shape[0], image.shape[1]), dtype=np.uint8)  # ❌

# BROKEN: Static dictionary (8 colors)
color_ranges = {
    "blue": [...], "red": [...], "green": [...], ...  # Only 8 colors
}
```

**v3 Solution: Dynamic HSV Color Extraction**

**User Constraint**: No growing static dictionaries

**Approach**: Use LLM to convert color descriptions to HSV ranges on-the-fly

```python
# NEW: stage2_dynamic_color_mapper.py

import dspy
from typing import Optional, List, Tuple

class ColorDescriptionToHSV(dspy.Signature):
    """Convert natural language color descriptions to HSV ranges."""
    color_description = dspy.InputField(desc="Color from user prompt (e.g., 'brown', 'sky blue', 'dark purple')")

    hsv_ranges = dspy.OutputField(desc="JSON array of HSV ranges: [[[H_min, S_min, V_min], [H_max, S_max, V_max]], ...]")
    confidence = dspy.OutputField(desc="Float 0-1 indicating how confident this mapping is")
    alternative_names = dspy.OutputField(desc="Alternative color names (synonyms)")


class DynamicColorMapper(dspy.Module):
    def __init__(self):
        self.mapper = dspy.ChainOfThought(ColorDescriptionToHSV)
        self.cache = {}  # Cache common colors to avoid repeated LLM calls

    def get_hsv_ranges(self, color_description: str) -> Optional[List[Tuple]]:
        """
        Convert color description to HSV ranges using LLM.

        Returns None if color is not a valid color (e.g., "auto-rickshaw").
        """
        # Check cache first
        if color_description in self.cache:
            return self.cache[color_description]

        # Query LLM
        result = self.mapper(color_description=color_description)

        # Parse HSV ranges
        try:
            ranges = json.loads(result.hsv_ranges)
            confidence = float(result.confidence)

            if confidence < 0.6:
                logging.warning(f"Low confidence ({confidence}) mapping color '{color_description}'")
                return None  # Not a color or ambiguous

            # Cache result
            self.cache[color_description] = ranges
            return ranges

        except (json.JSONDecodeError, ValueError) as e:
            logging.error(f"Failed to parse HSV ranges for '{color_description}': {e}")
            return None


def color_prefilter_v3(image: np.ndarray, color_description: str,
                       mapper: DynamicColorMapper) -> Optional[np.ndarray]:
    """
    Create binary mask for color regions using dynamic HSV mapping.

    Returns:
        Binary mask if color is valid, None if color is invalid/ambiguous
    """
    # Get HSV ranges from LLM
    hsv_ranges = mapper.get_hsv_ranges(color_description)

    if hsv_ranges is None:
        logging.info(f"'{color_description}' is not a recognized color - will use semantic-only path")
        return None  # ✅ Clear failure signal

    # Convert image to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Create mask from HSV ranges
    mask = None
    for hsv_range in hsv_ranges:
        lower, upper = hsv_range[0], hsv_range[1]
        range_mask = cv2.inRange(hsv_image, tuple(lower), tuple(upper))
        mask = range_mask if mask is None else cv2.bitwise_or(mask, range_mask)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = (mask > 0).astype(np.uint8)

    coverage = np.sum(mask > 0) / mask.size
    logging.info(f"Dynamic color filter '{color_description}': {coverage*100:.2f}% coverage")

    return mask
```

**Advantages**:
- ✅ Handles ANY color description ("burgundy", "olive green", "sky blue", "ochre")
- ✅ No static dictionary to maintain
- ✅ Returns `None` for non-colors (no toxic fallback)
- ✅ Caches common colors (fast after first use)
- ✅ LLM can handle cultural color names ("saffron", "indigo")

**Disadvantages**:
- ⚠️ Requires LLM call for unknown colors (1-2 seconds first time)
- ⚠️ Depends on LLM quality for HSV mapping accuracy

**Mitigation**:
- Ship with pre-populated cache for 50 common colors
- Use fast local LLM (qwen3:8b already loaded)
- Fallback to semantic-only if LLM fails

**Action for Qwen**: Implement DynamicColorMapper with caching and DSpy signatures

---

### 2. Stage 1: Intent Extraction (Enhance for Routing) - P0 CRITICAL

**Current Issues**:
```python
# Current: Only extracts entities and color
@dataclass
class Intent:
    entities: List[str]
    color: Optional[str]
```

**v3 Enhancement: Add Detection Strategy**

```python
# NEW: Enhanced intent with routing strategy

@dataclass
class IntentV3:
    entities: List[str]  # e.g., ["auto-rickshaws", "vehicles"]
    color: Optional[str]  # e.g., "yellow" or None
    size_hint: Optional[str]  # "small", "large", None
    position_hint: Optional[str]  # "top", "bottom", "center", None
    detection_strategy: str  # "color_guided", "semantic_only", "hybrid"
    confidence: float  # 0-1


class EnhancedIntentParser(dspy.Module):
    def __init__(self):
        self.parser = dspy.ChainOfThought("prompt -> intent")

    def forward(self, prompt: str) -> IntentV3:
        result = self.parser(
            prompt=prompt,
            instructions="""
            Extract:
            1. entities: Objects to detect
            2. color: Color if mentioned (null if none)
            3. size_hint: "small"/"large"/null
            4. position_hint: "top"/"bottom"/"center"/null
            5. detection_strategy:
               - "color_guided": Color is PRIMARY discriminator
               - "semantic_only": No color OR color is descriptive only
               - "hybrid": Color helps but not sufficient alone

            Examples:
            "red vehicles" → color_guided (color distinguishes from other objects)
            "auto-rickshaws" → semantic_only (no color)
            "blue sky" → hybrid (sky has position, blue helps)
            "yellow colonial buildings" → hybrid (buildings have shape + color)
            "small birds" → semantic_only (size hint, no color)
            """
        )

        return IntentV3(
            entities=result.entities,
            color=result.color,
            size_hint=result.size_hint,
            position_hint=result.position_hint,
            detection_strategy=result.detection_strategy,
            confidence=result.confidence
        )
```

**Action for Qwen**: Enhance stage1 with routing strategy detection

---

### 3. Orchestrator (Complete Redesign) - P0 CRITICAL

**Current Issues**:
```python
# Current: Linear pipeline (no routing)
def process(image_path, prompt):
    intent = stage1_extract_intent(prompt)
    color_mask = stage2_color_filter(image, intent.color)  # ❌ Always runs
    sam_masks = stage3_sam(image, color_mask)
    ...
```

**v3 Solution: Dual-Path Orchestrator**

```python
# NEW: orchestrator_v3.py

class DualPathOrchestrator:
    def __init__(self):
        self.intent_parser = EnhancedIntentParser()
        self.color_mapper = DynamicColorMapper()
        self.sam_segmenter = SAMSegmenter()
        self.clip_filter = CLIPFilter()
        self.vlm_validator = VLMValidator()

    def process(self, image_path, user_prompt):
        image = load_image(image_path)

        # Stage 1: Parse intent and determine strategy
        intent = self.intent_parser.forward(user_prompt)

        logging.info(f"Intent: entities={intent.entities}, color={intent.color}, "
                    f"strategy={intent.detection_strategy}")

        # Route to appropriate detection path
        if intent.detection_strategy == "color_guided":
            entity_masks = self._color_guided_path(image, intent)
        elif intent.detection_strategy == "semantic_only":
            entity_masks = self._semantic_only_path(image, intent)
        elif intent.detection_strategy == "hybrid":
            entity_masks = self._hybrid_path(image, intent)
        else:
            raise ValueError(f"Unknown strategy: {intent.detection_strategy}")

        return entity_masks

    def _color_guided_path(self, image, intent):
        """Fast path: Color filter → SAM on regions → CLIP high threshold"""
        # Stage 2: Dynamic color mapping
        color_mask = self.color_mapper.get_color_mask(image, intent.color)

        if color_mask is None or color_mask.mean() < 0.01:
            logging.warning("Color filter failed or <1% coverage, falling back to semantic")
            return self._semantic_only_path(image, intent)

        # Stage 3: SAM on color regions
        sam_masks = self.sam_segmenter.segment_regions(image, color_mask)

        # Stage 4: CLIP with high threshold
        filtered_masks = self.clip_filter.filter_masks(
            image, sam_masks, intent.entities,
            threshold=0.22  # High precision
        )

        return filtered_masks

    def _semantic_only_path(self, image, intent):
        """Slow path: Full SAM → CLIP low threshold → Post-filtering"""
        # Stage 3: SAM on full image
        logging.info("Running SAM on full image (semantic-only mode)")
        sam_masks = self.sam_segmenter.segment_full_image(image)

        # Stage 4: CLIP with lower threshold
        filtered_masks = self.clip_filter.filter_masks(
            image, sam_masks, intent.entities,
            threshold=0.18  # More permissive
        )

        # Stage 4.5: Post-filtering by size/position
        if intent.size_hint:
            filtered_masks = filter_by_size(filtered_masks, intent.size_hint)
        if intent.position_hint:
            filtered_masks = filter_by_position(filtered_masks, intent.position_hint)

        return filtered_masks

    def _hybrid_path(self, image, intent):
        """Combined: Run both paths, merge results"""
        logging.info("Running hybrid detection (color + semantic)")

        # Run color-guided
        color_results = self._color_guided_path(image, intent)

        # Run semantic-only
        semantic_results = self._semantic_only_path(image, intent)

        # Merge: Remove duplicates by IoU, rank by CLIP score
        merged = merge_and_deduplicate(color_results, semantic_results, iou_threshold=0.5)

        return merged
```

**Action for Qwen**: Implement dual-path orchestrator with routing logic

---

## What to ADD (New Components)

### 1. Post-Filtering Module (New in v3)

**File**: `pipeline/stage4_5_post_filters.py`

**Purpose**: Filter entities by size, position, shape after CLIP

```python
def filter_by_size(entities: List[EntityMask], size_hint: str) -> List[EntityMask]:
    """Filter by relative size (small < 2%, large > 5%)."""
    if size_hint == "small":
        return [e for e in entities if e.area_ratio < 0.02]
    elif size_hint == "large":
        return [e for e in entities if e.area_ratio > 0.05]
    return entities

def filter_by_position(entities: List[EntityMask], position_hint: str) -> List[EntityMask]:
    """Filter by spatial position (top/center/bottom)."""
    for e in entities:
        e.position_y_ratio = e.bbox_center_y / e.image_height

    if position_hint == "top":
        return [e for e in entities if e.position_y_ratio < 0.33]
    elif position_hint == "bottom":
        return [e for e in entities if e.position_y_ratio > 0.67]
    return entities
```

**Action for Qwen**: Implement post-filtering module with size/position/shape filters

---

### 2. Result Merger (New in v3)

**File**: `pipeline/result_merger.py`

**Purpose**: Merge results from color-guided and semantic-only paths

```python
def merge_and_deduplicate(
    color_results: List[EntityMask],
    semantic_results: List[EntityMask],
    iou_threshold: float = 0.5
) -> List[EntityMask]:
    """
    Merge two result sets, removing duplicates by IoU.

    Keep entity with higher CLIP score when IoU > threshold.
    """
    merged = []
    used_semantic_indices = set()

    # For each color result
    for color_entity in color_results:
        best_match = None
        best_iou = 0
        best_idx = -1

        # Find semantic result with highest IoU
        for idx, semantic_entity in enumerate(semantic_results):
            if idx in used_semantic_indices:
                continue

            iou = compute_iou(color_entity.mask, semantic_entity.mask)
            if iou > best_iou:
                best_iou = iou
                best_match = semantic_entity
                best_idx = idx

        # If high IoU, it's the same entity - keep higher scoring one
        if best_iou > iou_threshold:
            if color_entity.clip_score > best_match.clip_score:
                merged.append(color_entity)
            else:
                merged.append(best_match)
            used_semantic_indices.add(best_idx)
        else:
            # No duplicate, add color result
            merged.append(color_entity)

    # Add remaining semantic results (no duplicates)
    for idx, semantic_entity in enumerate(semantic_results):
        if idx not in used_semantic_indices:
            merged.append(semantic_entity)

    # Sort by CLIP score
    merged.sort(key=lambda e: e.clip_score, reverse=True)

    return merged
```

**Action for Qwen**: Implement result merger with IoU-based deduplication

---

### 3. Validation System Integration (New in v3)

**File**: `validation/vision_delta_analysis.py`

**Purpose**: Implement edit validation from EDIT_VALIDATION_RESEARCH.md

**Action for Qwen**: Implement after core pipeline is working (Phase 2)

---

## Migration Checklist for Qwen

### Phase 1: Core Pipeline (Week 1-2)

- [ ] **Setup**
  - [ ] Copy v2 test images to v3/images/
  - [ ] Copy v2 model weights (sam2.1_b.pt) to v3/
  - [ ] Create v3 config.yaml with dynamic color system settings

- [ ] **Copy Working Components**
  - [ ] Copy stage3_sam_segmentation.py → Add diagnostics + fallback
  - [ ] Copy stage4_clip_filter.py → Add adaptive threshold
  - [ ] Copy stage5_mask_organization.py → No changes
  - [ ] Copy stage6_vlm_validation.py → Add structured feedback fields
  - [ ] Copy app.py → Update CLI args for new features
  - [ ] Copy tui.py → Update display for new capabilities
  - [ ] Copy tests/ → Add new test categories

- [ ] **Redesign Broken Components**
  - [ ] Implement stage2_dynamic_color_mapper.py (DSpy + HSV)
  - [ ] Enhance stage1_entity_extraction.py (add routing strategy)
  - [ ] Implement orchestrator_v3.py (dual-path routing)

- [ ] **Add New Components**
  - [ ] Implement stage4_5_post_filters.py (size/position filtering)
  - [ ] Implement result_merger.py (IoU-based deduplication)

- [ ] **Testing**
  - [ ] Run wildcard tests on v3 pipeline
  - [ ] Validate improvement over v2 (target: 5-6/9 pass)
  - [ ] Performance benchmarks (color-guided: <5s, semantic: <8s)

### Phase 2: Validation System (Week 3)

- [ ] Implement validation/vision_delta_analysis.py
- [ ] Implement validation/quality_scoring.py
- [ ] Integrate retry loop in orchestrator
- [ ] Test auto-correction with failed edits from v2

### Phase 3: Production Readiness (Week 4)

- [ ] Expand ground truth dataset (50+ images)
- [ ] Measure precision/recall on ground truth
- [ ] Performance optimization
- [ ] Documentation updates
- [ ] User testing

---

## Expected Outcomes

| Metric | v2 Actual | v3 Target | Measurement |
|--------|----------|-----------|-------------|
| **Success Rate** | 11% (1/9) | 75%+ (7/9) | Wildcard tests |
| **False Positive Rate** | 22% (2/9) | 0% | No toxic fallback |
| **Color Coverage** | 40% (8 colors) | 100% (dynamic) | Any color description |
| **Semantic Query Support** | 0% | 100% | "auto-rickshaws", "birds" work |
| **Processing Time (color)** | 10s | <5s | Benchmark suite |
| **Processing Time (semantic)** | N/A (crashed) | <8s | Acceptable for comprehensive path |

---

## Risk Mitigation

### Risk 1: LLM Color Mapping Inaccuracy

**Mitigation**:
- Pre-populate cache with 50 common colors
- Validate LLM HSV ranges against known good mappings
- Fallback to semantic-only if color confidence < 0.6

### Risk 2: Semantic-Only Path Too Slow

**Mitigation**:
- Optimize SAM batch processing
- Use SAM 2.1 tiny model (sam2.1_t.pt) for faster processing
- Show progress bar in TUI for user feedback

### Risk 3: Hybrid Path Produces Duplicates

**Mitigation**:
- IoU threshold tuning (0.4-0.6)
- Visual inspection of merge results
- Add "unique entities only" option in config

---

## Conclusion

v3 architecture addresses ALL P0 critical flaws identified in v2:
- ✅ No toxic fallback (returns `None` instead)
- ✅ Dynamic color handling (no static dictionary)
- ✅ Semantic-only detection path (handles 60% of missed queries)
- ✅ Comprehensive diagnostics (SAM failures explained)
- ✅ Dual-path routing (strategy-aware processing)

**Next Step**: Qwen implements Phase 1 components following this migration plan.
