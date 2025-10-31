# Qwen CLI Supervision Plan - EDI Vision V3

**Document Version**: 1.0
**Date**: 2025-10-31
**Supervisor**: Claude Code
**Implementer**: Qwen CLI
**Target**: 85%+ success rate (7.5/9 wildcard tests)

---

## Supervision Philosophy

**Claude's Role**: Research, planning, architecture, validation, quality assurance
**Qwen's Role**: Implementation, testing, debugging, documentation

**Key Principle**: Claude doesn't write code. Claude supervises by:
1. Providing extremely detailed specifications
2. Validating Qwen's output against quality standards
3. Identifying edge cases and failure modes
4. Measuring performance against benchmarks
5. Approving or rejecting implementation phases

---

## Implementation Phases

### Phase 1: Foundation Setup (Week 1)

**Objective**: Set up v3 workspace with working components from v2

#### Task 1.1: Workspace Setup

**Qwen Instructions**:
```bash
# Navigate to v3 directory
cd /home/riju279/Documents/Code/Zonko/EDI/edi/work/edi_vision_v3

# Copy test images
cp -r ../edi_vision_v2/images/ ./images/

# Copy model weights
cp ../edi_vision_v2/sam2.1_b.pt ./

# Create subdirectories
mkdir -p pipeline tests validations logs/{orchestrator,test,wildcard}

# Create empty __init__.py files
touch pipeline/__init__.py tests/__init__.py validations/__init__.py
```

**Validation Checkpoints**:
- [ ] All directories exist
- [ ] sam2.1_b.pt is 154MB (verify with `ls -lh sam2.1_b.pt`)
- [ ] Test images present (at least 6 images from wildcard tests)

**Claude Validation**:
```bash
# Check structure
ls -la edi_vision_v3/
# Expected: docs/, pipeline/, tests/, validations/, logs/, images/, sam2.1_b.pt
```

---

#### Task 1.2: Copy Stage 3 (SAM Segmentation) with Enhancements

**Qwen Instructions**:
1. Read `@edi_vision_v2/pipeline/stage3_sam_segmentation.py`
2. Copy to `@edi_vision_v3/pipeline/stage3_sam_segmentation.py`
3. Add diagnostic enhancements as specified in `@V2_TO_V3_MIGRATION_ANALYSIS.md` section "Stage 3: SAM Segmentation"

**Specific Enhancements Required**:
```python
# ADD at beginning of segment_regions()
logging.info(f"SAM input diagnostics: image_shape={image.shape}, "
            f"color_mask_coverage={color_mask.mean():.2%}, "
            f"color_mask_nonzero_pixels={np.sum(color_mask > 0)}")

# ADD after primary SAM generation
if len(all_masks) == 0:
    logging.warning("SAM generated 0 masks with primary parameters, trying fallback")

    # Fallback: Relaxed parameters
    sam_model_fallback = SAM("sam2.1_b.pt")
    if torch.cuda.is_available():
        sam_model_fallback.to('cuda')
        sam_model_fallback.half()

    results_fallback = sam_model_fallback(
        image,
        task="segment",
        pred_iou_thresh=0.70,  # Relaxed from 0.88
        stability_score_thresh=0.85,  # Relaxed from 0.95
    )

    if len(results_fallback) > 0 and results_fallback[0].masks is not None:
        all_masks = results_fallback[0].masks.data
        logging.info(f"SAM fallback succeeded: generated {len(all_masks)} masks")

# ADD if still 0 masks after fallback
if len(all_masks) == 0:
    logging.error("SAM failed even with fallback parameters")
    logging.error(f"Image diagnostics: dtype={image.dtype}, min={image.min()}, "
                 f"max={image.max()}, shape={image.shape}")
    logging.error(f"Color mask diagnostics: unique_values={np.unique(color_mask)}, "
                 f"largest_region_size={get_largest_region_size(color_mask)}px")
    # Return empty list instead of crashing
    return []
```

**Validation Checkpoints**:
- [ ] File copied successfully
- [ ] Diagnostic logging added at 3 locations (input, after primary, after fallback)
- [ ] Fallback SAM with relaxed parameters implemented
- [ ] Returns empty list (not crash) if SAM fails completely
- [ ] All original functionality preserved

**Claude Validation**:
```python
# Test on known SAM-failing image
result = segment_regions(
    image=load_image("images/mumbai-traffic.jpg"),
    color_mask=create_test_color_mask(),  # Simulated yellow mask
    min_area=500
)

# Expected: Either succeeds with fallback OR returns [] with detailed error logs
# Should NOT crash
```

**Quality Standards**:
- Code formatting: Black style, max line length 100
- Type hints: All function parameters and returns
- Docstrings: Google style with Args, Returns, Raises
- Logging: INFO for success, WARNING for fallback, ERROR for failure
- No hardcoded paths or magic numbers

---

#### Task 1.3: Copy Stage 4 (CLIP Filtering) with Adaptive Threshold

**Qwen Instructions**:
1. Read `@edi_vision_v2/pipeline/stage4_clip_filter.py`
2. Copy to `@edi_vision_v3/pipeline/stage4_clip_filter.py`
3. Add adaptive threshold logic as specified in migration analysis

**Specific Enhancements Required**:
```python
# ADD: Adaptive threshold function
def get_adaptive_clip_threshold(detection_strategy: str) -> float:
    """
    Get CLIP similarity threshold based on detection strategy.

    Args:
        detection_strategy: One of "color_guided", "semantic_only", "hybrid"

    Returns:
        Float threshold value (0.0 to 1.0)
    """
    thresholds = {
        "color_guided": 0.22,  # High precision - color already filtered
        "semantic_only": 0.18,  # More permissive - no color guidance
        "hybrid": 0.20,  # Middle ground
    }
    return thresholds.get(detection_strategy, 0.22)

# MODIFY: filter_masks() function signature
def filter_masks(image: np.ndarray, masks: List[np.ndarray], entities: List[str],
                detection_strategy: str = "color_guided",  # ADD this parameter
                top_k: int = 20) -> List[Dict]:
    """
    Filter SAM masks using CLIP semantic similarity with adaptive threshold.

    Args:
        image: Original RGB image
        masks: List of binary masks from SAM
        entities: Target entity names from user prompt
        detection_strategy: Detection strategy for threshold adaptation
        top_k: Maximum number of entities to return

    Returns:
        List of entity dictionaries with masks and metadata
    """
    threshold = get_adaptive_clip_threshold(detection_strategy)
    logging.info(f"CLIP filtering with strategy '{detection_strategy}', threshold={threshold:.3f}")

    # ... rest of function uses `threshold` variable
```

**Validation Checkpoints**:
- [ ] Adaptive threshold function implemented
- [ ] filter_masks() signature updated with detection_strategy parameter
- [ ] Threshold correctly applied based on strategy
- [ ] Logging shows which threshold is being used
- [ ] Backwards compatible (default to "color_guided")

**Claude Validation**:
```python
# Test adaptive thresholds
masks_color_guided = filter_masks(image, masks, ["vehicles"], "color_guided")
masks_semantic_only = filter_masks(image, masks, ["vehicles"], "semantic_only")
masks_hybrid = filter_masks(image, masks, ["vehicles"], "hybrid")

# Expected: semantic_only returns MORE entities (lower threshold)
assert len(masks_semantic_only) >= len(masks_color_guided)
```

---

#### Task 1.4: Copy Stage 5 (Mask Organization) As-Is

**Qwen Instructions**:
1. Read `@edi_vision_v2/pipeline/stage5_mask_organization.py`
2. Copy to `@edi_vision_v3/pipeline/stage5_mask_organization.py` WITHOUT modifications

**Validation Checkpoints**:
- [ ] File copied exactly as-is
- [ ] No changes made (this component works perfectly)
- [ ] File imports and runs without errors

**Claude Validation**:
```bash
# Verify file is identical
diff edi_vision_v2/pipeline/stage5_mask_organization.py edi_vision_v3/pipeline/stage5_mask_organization.py
# Expected: No differences
```

---

#### Task 1.5: Copy Stage 6 (VLM Validation) with Structured Feedback

**Qwen Instructions**:
1. Read `@edi_vision_v2/pipeline/stage6_vlm_validation.py`
2. Copy to `@edi_vision_v3/pipeline/stage6_vlm_validation.py`
3. Enhance ValidationResult dataclass with structured feedback fields

**Specific Enhancements Required**:
```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

@dataclass
class ValidationResult:
    # Existing fields
    covers_all_targets: bool
    confidence: float
    feedback: str
    target_coverage: float
    false_positive_ratio: float
    missing_targets: str
    suggestions: List[str]

    # NEW: Structured feedback for auto-correction
    missed_entities: List[str] = field(default_factory=list)
    """Entities mentioned by user but not masked"""

    false_positive_entities: List[str] = field(default_factory=list)
    """Entities masked but not mentioned by user"""

    correction_hints: Dict[str, Any] = field(default_factory=dict)
    """Hints for pipeline adjustment (e.g., {'clip_threshold': 0.20, 'min_area': 300})"""

    spatial_accuracy: Optional[float] = None
    """0-1 score for spatial accuracy (are masks in right locations?)"""

# MODIFY: validate_with_vlm() to populate new fields
def validate_with_vlm(...) -> ValidationResult:
    # ... existing code

    # ADD: Parse structured feedback from VLM
    try:
        # Extract missed entities
        missed_pattern = r"missed_entities:\s*\[(.*?)\]"
        missed_match = re.search(missed_pattern, vlm_response)
        missed_entities = (
            [e.strip() for e in missed_match.group(1).split(",") if e.strip()]
            if missed_match else []
        )

        # Extract false positives
        fp_pattern = r"false_positives:\s*\[(.*?)\]"
        fp_match = re.search(fp_pattern, vlm_response)
        false_positive_entities = (
            [e.strip() for e in fp_match.group(1).split(",") if e.strip()]
            if fp_match else []
        )

        # Extract correction hints
        hints = {}
        if "increase clip threshold" in vlm_response.lower():
            hints['clip_threshold'] = current_threshold + 0.02
        if "decrease clip threshold" in vlm_response.lower():
            hints['clip_threshold'] = current_threshold - 0.02
        if "increase min_area" in vlm_response.lower():
            hints['min_area'] = current_min_area * 2
        if "decrease min_area" in vlm_response.lower():
            hints['min_area'] = current_min_area // 2

    except Exception as e:
        logging.warning(f"Failed to parse structured feedback: {e}")
        missed_entities = []
        false_positive_entities = []
        hints = {}

    return ValidationResult(
        # ... existing fields
        missed_entities=missed_entities,
        false_positive_entities=false_positive_entities,
        correction_hints=hints,
    )
```

**Validation Checkpoints**:
- [ ] ValidationResult dataclass enhanced with 4 new fields
- [ ] validate_with_vlm() populates new fields
- [ ] Regex parsing for structured feedback implemented
- [ ] Graceful handling if VLM doesn't provide structured format
- [ ] Logging for parsed feedback values

**Claude Validation**:
```python
# Test on known case
validation = validate_with_vlm(
    image=image,
    entity_masks=[...],  # 3 masks of random objects
    user_prompt="highlight all red vehicles"
)

# Expected: ValidationResult with populated missed_entities and correction_hints
print(validation.missed_entities)  # Should list vehicles not detected
print(validation.correction_hints)  # Should suggest threshold adjustments
```

---

### Phase 2: Core Redesign (Week 1-2)

**Objective**: Implement redesigned components (color mapper, intent parser, orchestrator)

#### Task 2.1: Implement Dynamic Color Mapper

**Qwen Instructions**:
1. Create new file `@edi_vision_v3/pipeline/stage2_dynamic_color_mapper.py`
2. Implement as specified in `@V2_TO_V3_MIGRATION_ANALYSIS.md` section "Stage 2: Color Filtering"

**Detailed Implementation Specification**:

```python
"""Stage 2: Dynamic Color Mapping

This module uses DSpy + local LLM to convert natural language color descriptions
to HSV ranges on-the-fly, eliminating the need for static color dictionaries.
"""

import logging
import numpy as np
import cv2
import json
import dspy
from typing import Optional, List, Tuple
from dataclasses import dataclass

# Pre-populated cache for 50 common colors
COMMON_COLOR_CACHE = {
    "red": [[[0, 50, 50], [10, 255, 255]], [[170, 50, 50], [180, 255, 255]]],
    "blue": [[[90, 50, 50], [130, 255, 255]]],
    "green": [[[40, 40, 40], [80, 255, 255]]],
    "yellow": [[[20, 50, 50], [30, 255, 255]]],
    "orange": [[[10, 50, 50], [20, 255, 255]]],
    "white": [[[0, 0, 200], [180, 30, 255]]],
    "black": [[[0, 0, 0], [180, 255, 30]]],
    "gray": [[[0, 0, 50], [180, 50, 200]]],
    "brown": [[[10, 40, 40], [20, 255, 200]]],
    "purple": [[[130, 40, 40], [160, 255, 255]]],
    "pink": [[[150, 40, 100], [170, 255, 255]]],
    "cyan": [[[85, 100, 100], [95, 255, 255]]],
    "magenta": [[[140, 100, 100], [160, 255, 255]]],
    "maroon": [[[0, 100, 40], [10, 255, 100]]],
    "navy": [[[110, 100, 40], [130, 255, 100]]],
    "olive": [[[30, 40, 40], [40, 255, 200]]],
    "teal": [[[85, 100, 40], [95, 255, 200]]],
    "beige": [[[20, 20, 180], [30, 80, 255]]],
    "cream": [[[20, 10, 200], [30, 60, 255]]],
    "tan": [[[15, 30, 150], [25, 100, 220]]],
    "gold": [[[20, 100, 180], [30, 255, 255]]],
    "silver": [[[0, 0, 150], [180, 30, 220]]],
    "sky": [[[100, 40, 150], [120, 255, 255]]],  # Sky blue
    "burgundy": [[[0, 80, 60], [10, 255, 150]]],
    "turquoise": [[[85, 70, 100], [95, 255, 255]]],
}


class ColorDescriptionToHSV(dspy.Signature):
    """Convert natural language color descriptions to HSV ranges."""

    color_description = dspy.InputField(
        desc="Color from user prompt (e.g., 'brown', 'sky blue', 'dark purple')"
    )

    is_valid_color = dspy.OutputField(
        desc="Boolean: true if this is a color description, false if not (e.g., 'auto-rickshaw' is not a color)"
    )

    hsv_ranges = dspy.OutputField(
        desc=(
            "JSON array of HSV ranges in format: "
            "[[[H_min, S_min, V_min], [H_max, S_max, V_max]], ...]. "
            "H: 0-180, S: 0-255, V: 0-255. "
            "Return empty array [] if is_valid_color is false."
        )
    )

    confidence = dspy.OutputField(
        desc="Float 0-1 indicating confidence in HSV mapping"
    )

    reasoning = dspy.OutputField(
        desc="Brief explanation of HSV range selection"
    )


class DynamicColorMapper(dspy.Module):
    def __init__(self, ollama_url="http://localhost:11434", model="qwen3:8b"):
        super().__init__()

        # Initialize DSpy with Ollama
        self.lm = dspy.OllamaLocal(model=model, base_url=ollama_url)
        dspy.settings.configure(lm=self.lm)

        # Create CoT module
        self.mapper = dspy.ChainOfThought(ColorDescriptionToHSV)

        # Runtime cache (session-level)
        self.runtime_cache = {}

    def get_hsv_ranges(self, color_description: str) -> Optional[List[List[List[int]]]]:
        """
        Convert color description to HSV ranges.

        Args:
            color_description: Natural language color (e.g., "brown", "sky blue")

        Returns:
            List of HSV ranges, or None if not a valid color
        """
        color_lower = color_description.lower().strip()

        # Check pre-populated cache first (fast path)
        if color_lower in COMMON_COLOR_CACHE:
            logging.info(f"Color '{color_description}' found in common cache")
            return COMMON_COLOR_CACHE[color_lower]

        # Check runtime cache
        if color_lower in self.runtime_cache:
            logging.info(f"Color '{color_description}' found in runtime cache")
            return self.runtime_cache[color_lower]

        # Query LLM (slow path)
        logging.info(f"Querying LLM for color '{color_description}'")
        try:
            result = self.mapper(color_description=color_description)

            # Parse results
            is_valid_color = str(result.is_valid_color).lower() == "true"
            confidence = float(result.confidence)

            logging.info(
                f"LLM response: is_valid_color={is_valid_color}, "
                f"confidence={confidence:.2f}, reasoning={result.reasoning}"
            )

            # If not a valid color, return None
            if not is_valid_color or confidence < 0.6:
                logging.warning(
                    f"'{color_description}' is not a recognized color "
                    f"(confidence={confidence:.2f})"
                )
                return None

            # Parse HSV ranges
            hsv_ranges = json.loads(result.hsv_ranges)

            # Validate format
            if not isinstance(hsv_ranges, list) or len(hsv_ranges) == 0:
                logging.error(f"Invalid HSV ranges format: {hsv_ranges}")
                return None

            # Cache result for future use
            self.runtime_cache[color_lower] = hsv_ranges
            logging.info(f"Cached HSV ranges for '{color_description}': {hsv_ranges}")

            return hsv_ranges

        except Exception as e:
            logging.error(f"Failed to map color '{color_description}': {e}")
            return None

    def create_color_mask(
        self, image: np.ndarray, color_description: str
    ) -> Optional[np.ndarray]:
        """
        Create binary mask for regions matching color description.

        Args:
            image: RGB image (H x W x 3)
            color_description: Natural language color

        Returns:
            Binary mask (H x W), or None if color is invalid/ambiguous
        """
        # Get HSV ranges
        hsv_ranges = self.get_hsv_ranges(color_description)

        if hsv_ranges is None:
            logging.info(
                f"'{color_description}' is not a color - will use semantic-only path"
            )
            return None  # ✅ Clear failure signal (no toxic fallback)

        # Convert image to HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Create mask from HSV ranges
        mask = None
        for hsv_range in hsv_ranges:
            lower, upper = np.array(hsv_range[0]), np.array(hsv_range[1])
            range_mask = cv2.inRange(hsv_image, lower, upper)
            mask = range_mask if mask is None else cv2.bitwise_or(mask, range_mask)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = (mask > 0).astype(np.uint8)

        coverage = np.sum(mask > 0) / mask.size
        logging.info(
            f"Dynamic color filter '{color_description}': {coverage*100:.2f}% coverage"
        )

        return mask
```

**Validation Checkpoints**:
- [ ] File created with all required classes and functions
- [ ] COMMON_COLOR_CACHE pre-populated with 25+ colors
- [ ] DSpy signature ColorDescriptionToHSV correctly defined
- [ ] DynamicColorMapper implements caching (common + runtime)
- [ ] get_hsv_ranges() returns None for non-colors (no toxic fallback)
- [ ] create_color_mask() includes morphological cleanup
- [ ] Comprehensive logging at all decision points
- [ ] Type hints on all functions
- [ ] Google-style docstrings

**Claude Validation Tests**:

**Test 1: Common color (should use cache)**
```python
mapper = DynamicColorMapper()
ranges = mapper.get_hsv_ranges("brown")
assert ranges is not None
assert ranges == COMMON_COLOR_CACHE["brown"]
# Should NOT call LLM (check logs for "found in common cache")
```

**Test 2: Uncommon color (should query LLM)**
```python
mapper = DynamicColorMapper()
ranges = mapper.get_hsv_ranges("burgundy")
assert ranges is not None
assert len(ranges) > 0
# Should call LLM (check logs for "Querying LLM")
```

**Test 3: Non-color (should return None)**
```python
mapper = DynamicColorMapper()
ranges = mapper.get_hsv_ranges("auto-rickshaw")
assert ranges is None
# Should NOT create mask (check logs for "not a recognized color")
```

**Test 4: Create mask**
```python
mapper = DynamicColorMapper()
image = load_test_image("images/test_image.jpeg")
mask = mapper.create_color_mask(image, "brown")
assert mask is not None or mask is None  # Either is valid
if mask is not None:
    assert mask.shape == image.shape[:2]
    assert mask.dtype == np.uint8
    assert np.all((mask == 0) | (mask == 1))  # Binary mask
```

**Performance Requirements**:
- Common colors: <10ms (cache lookup)
- Uncommon colors (first call): <2 seconds (LLM query)
- Uncommon colors (cached): <10ms (cache lookup)
- No crashes on invalid input
- No toxic fallback (returns None instead of np.ones())

---

#### Task 2.2: Enhance Stage 1 (Intent Parser) for Routing

**Qwen Instructions**:
1. Read `@edi_vision_v2/pipeline/stage1_entity_extraction.py`
2. Create enhanced version at `@edi_vision_v3/pipeline/stage1_intent_parser_v3.py`
3. Add detection strategy routing as specified in migration analysis

**Specific Implementation Specification**:

```python
"""Stage 1: Enhanced Intent Parsing with Detection Strategy Routing

This module extracts user intent and determines the optimal detection strategy
(color-guided, semantic-only, or hybrid) for routing in the orchestrator.
"""

import logging
import dspy
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class IntentV3:
    """Enhanced intent with routing strategy."""

    entities: List[str]  # Objects to detect (e.g., ["auto-rickshaws", "vehicles"])
    color: Optional[str]  # Color if mentioned (e.g., "yellow" or None)
    size_hint: Optional[str]  # "small", "large", or None
    position_hint: Optional[str]  # "top", "bottom", "center", None
    detection_strategy: str  # "color_guided", "semantic_only", "hybrid"
    confidence: float  # 0-1
    reasoning: str  # Explanation of strategy choice


class ExtractIntent(dspy.Signature):
    """Extract structured intent from user prompt with detection strategy."""

    user_prompt = dspy.InputField(desc="User's natural language editing request")

    entities = dspy.OutputField(
        desc="Comma-separated list of target objects (e.g., 'auto-rickshaws, vehicles')"
    )

    color = dspy.OutputField(
        desc="Color if mentioned (e.g., 'yellow', 'blue'), or 'none' if no color"
    )

    size_hint = dspy.OutputField(
        desc="Size hint: 'small', 'large', or 'none'"
    )

    position_hint = dspy.OutputField(
        desc="Position hint: 'top', 'bottom', 'center', or 'none'"
    )

    detection_strategy = dspy.OutputField(
        desc=(
            "Choose detection strategy:\n"
            "- 'color_guided': Color is PRIMARY discriminator (e.g., 'red vehicles')\n"
            "- 'semantic_only': No color OR color is not useful (e.g., 'auto-rickshaws', 'birds')\n"
            "- 'hybrid': Color helps but insufficient alone (e.g., 'blue sky', 'brown roofs')"
        )
    )

    confidence = dspy.OutputField(
        desc="Float 0-1: How confident is this intent extraction?"
    )

    reasoning = dspy.OutputField(
        desc="Brief explanation of detection_strategy choice"
    )


class EnhancedIntentParser(dspy.Module):
    def __init__(self, ollama_url="http://localhost:11434", model="qwen3:8b"):
        super().__init__()

        # Initialize DSpy with Ollama
        self.lm = dspy.OllamaLocal(model=model, base_url=ollama_url)
        dspy.settings.configure(lm=self.lm)

        # Create CoT module with detailed instructions
        self.parser = dspy.ChainOfThought(ExtractIntent)

    def forward(self, user_prompt: str) -> IntentV3:
        """
        Parse user prompt and determine detection strategy.

        Args:
            user_prompt: User's natural language request

        Returns:
            IntentV3 object with routing strategy
        """
        logging.info(f"Parsing intent from prompt: '{user_prompt}'")

        # Call DSpy parser
        result = self.parser(user_prompt=user_prompt)

        # Parse entities
        entities = [e.strip() for e in result.entities.split(",") if e.strip()]

        # Parse color (handle "none")
        color = result.color if result.color.lower() != "none" else None

        # Parse size hint
        size_hint = result.size_hint if result.size_hint.lower() != "none" else None

        # Parse position hint
        position_hint = (
            result.position_hint if result.position_hint.lower() != "none" else None
        )

        # Parse confidence
        try:
            confidence = float(result.confidence)
        except ValueError:
            logging.warning(f"Invalid confidence value: {result.confidence}, using 0.8")
            confidence = 0.8

        # Validate detection strategy
        valid_strategies = ["color_guided", "semantic_only", "hybrid"]
        strategy = result.detection_strategy
        if strategy not in valid_strategies:
            logging.warning(
                f"Invalid strategy '{strategy}', defaulting to 'semantic_only'"
            )
            strategy = "semantic_only"

        intent = IntentV3(
            entities=entities,
            color=color,
            size_hint=size_hint,
            position_hint=position_hint,
            detection_strategy=strategy,
            confidence=confidence,
            reasoning=result.reasoning,
        )

        logging.info(
            f"Parsed intent: entities={intent.entities}, color={intent.color}, "
            f"strategy={intent.detection_strategy}, confidence={intent.confidence:.2f}"
        )
        logging.info(f"Reasoning: {intent.reasoning}")

        return intent
```

**Example Prompts and Expected Outputs**:

```python
# Test Case 1: Color-guided
prompt = "highlight all red vehicles"
intent = parser.forward(prompt)
assert intent.entities == ["vehicles"]
assert intent.color == "red"
assert intent.detection_strategy == "color_guided"
# Reasoning: "Red is primary discriminator to find vehicles among other objects"

# Test Case 2: Semantic-only
prompt = "detect auto-rickshaws"
intent = parser.forward(prompt)
assert intent.entities == ["auto-rickshaws"]
assert intent.color is None
assert intent.detection_strategy == "semantic_only"
# Reasoning: "No color mentioned, need semantic detection of auto-rickshaws"

# Test Case 3: Hybrid
prompt = "edit brown roofs"
intent = parser.forward(prompt)
assert intent.entities == ["roofs"]
assert intent.color == "brown"
assert intent.detection_strategy == "hybrid"
# Reasoning: "Brown helps identify roofs, but roof shape/position also important"

# Test Case 4: Hybrid with position
prompt = "highlight blue sky"
intent = parser.forward(prompt)
assert intent.entities == ["sky"]
assert intent.color == "blue"
assert intent.position_hint == "top"
assert intent.detection_strategy == "hybrid"
# Reasoning: "Sky is in top region, blue color helps but position is key"

# Test Case 5: Semantic with size
prompt = "detect small birds"
intent = parser.forward(prompt)
assert intent.entities == ["birds"]
assert intent.size_hint == "small"
assert intent.detection_strategy == "semantic_only"
# Reasoning: "No color, need to find bird shapes with size filtering"
```

**Validation Checkpoints**:
- [ ] IntentV3 dataclass with all 7 fields implemented
- [ ] ExtractIntent DSpy signature with detailed instructions
- [ ] EnhancedIntentParser correctly parses all fields
- [ ] Handles "none" values gracefully
- [ ] Validates detection_strategy (must be one of 3 valid options)
- [ ] Comprehensive logging of parsed intent
- [ ] All 5 test cases pass with reasonable outputs

**Claude Validation**:
Run all 5 test cases and verify:
1. Strategy selection is reasonable
2. Confidence scores are > 0.7
3. Reasoning explanations make sense
4. No crashes on edge cases

---

#### Task 2.3: Implement Post-Filtering Module

**Qwen Instructions**:
1. Create new file `@edi_vision_v3/pipeline/stage4_5_post_filters.py`
2. Implement size, position, and shape filtering as specified

**Full Implementation Specification**:

```python
"""Stage 4.5: Post-Filtering

This module provides additional filtering after CLIP for semantic-only and hybrid paths.
Filters by size, position, and shape to improve semantic detection accuracy.
"""

import logging
import numpy as np
from typing import List, Dict


def filter_by_size(
    entity_masks: List[Dict], size_hint: str, image_area: int
) -> List[Dict]:
    """
    Filter entities by relative size.

    Args:
        entity_masks: List of entity dictionaries with 'mask' and 'area' keys
        size_hint: "small" or "large"
        image_area: Total image area in pixels (H * W)

    Returns:
        Filtered list of entity masks
    """
    if size_hint not in ["small", "large"]:
        return entity_masks

    filtered = []
    for entity in entity_masks:
        area = entity.get("area", 0)
        area_ratio = area / image_area

        if size_hint == "small" and area_ratio < 0.02:  # < 2% of image
            filtered.append(entity)
            logging.debug(f"Entity {entity.get('id', '?')}: size OK (small, {area_ratio:.4f})")
        elif size_hint == "large" and area_ratio > 0.05:  # > 5% of image
            filtered.append(entity)
            logging.debug(f"Entity {entity.get('id', '?')}: size OK (large, {area_ratio:.4f})")
        else:
            logging.debug(
                f"Entity {entity.get('id', '?')}: filtered out by size "
                f"(area_ratio={area_ratio:.4f}, wanted {size_hint})"
            )

    logging.info(
        f"Size filtering ({size_hint}): {len(entity_masks)} → {len(filtered)} entities"
    )
    return filtered


def filter_by_position(
    entity_masks: List[Dict], position_hint: str, image_height: int
) -> List[Dict]:
    """
    Filter entities by vertical position.

    Args:
        entity_masks: List of entity dictionaries with 'bbox' key [x, y, w, h]
        position_hint: "top", "bottom", or "center"
        image_height: Image height in pixels

    Returns:
        Filtered list of entity masks
    """
    if position_hint not in ["top", "bottom", "center"]:
        return entity_masks

    filtered = []
    for entity in entity_masks:
        bbox = entity.get("bbox", [0, 0, 0, 0])
        x, y, w, h = bbox
        center_y = y + h / 2
        y_ratio = center_y / image_height

        if position_hint == "top" and y_ratio < 0.33:
            filtered.append(entity)
            logging.debug(f"Entity {entity.get('id', '?')}: position OK (top, y_ratio={y_ratio:.3f})")
        elif position_hint == "bottom" and y_ratio > 0.67:
            filtered.append(entity)
            logging.debug(f"Entity {entity.get('id', '?')}: position OK (bottom, y_ratio={y_ratio:.3f})")
        elif position_hint == "center" and 0.33 < y_ratio < 0.67:
            filtered.append(entity)
            logging.debug(f"Entity {entity.get('id', '?')}: position OK (center, y_ratio={y_ratio:.3f})")
        else:
            logging.debug(
                f"Entity {entity.get('id', '?')}: filtered out by position "
                f"(y_ratio={y_ratio:.3f}, wanted {position_hint})"
            )

    logging.info(
        f"Position filtering ({position_hint}): {len(entity_masks)} → {len(filtered)} entities"
    )
    return filtered


def filter_by_aspect_ratio(
    entity_masks: List[Dict], min_ratio: float = 0.2, max_ratio: float = 5.0
) -> List[Dict]:
    """
    Filter entities by aspect ratio (remove extreme elongations).

    Args:
        entity_masks: List of entity dictionaries with 'bbox' key [x, y, w, h]
        min_ratio: Minimum aspect ratio (width/height)
        max_ratio: Maximum aspect ratio (width/height)

    Returns:
        Filtered list of entity masks
    """
    filtered = []
    for entity in entity_masks:
        bbox = entity.get("bbox", [0, 0, 1, 1])
        x, y, w, h = bbox

        if h == 0:
            logging.warning(f"Entity {entity.get('id', '?')}: invalid bbox (h=0)")
            continue

        aspect_ratio = w / h

        if min_ratio <= aspect_ratio <= max_ratio:
            filtered.append(entity)
        else:
            logging.debug(
                f"Entity {entity.get('id', '?')}: filtered out by aspect ratio "
                f"({aspect_ratio:.2f} not in [{min_ratio}, {max_ratio}])"
            )

    logging.info(
        f"Aspect ratio filtering: {len(entity_masks)} → {len(filtered)} entities"
    )
    return filtered


def apply_post_filters(
    entity_masks: List[Dict],
    size_hint: Optional[str],
    position_hint: Optional[str],
    image_shape: Tuple[int, int],
) -> List[Dict]:
    """
    Apply all post-filters based on hints.

    Args:
        entity_masks: List of entity dictionaries
        size_hint: Size hint from intent ("small", "large", or None)
        position_hint: Position hint from intent ("top", "bottom", "center", or None)
        image_shape: (height, width) of image

    Returns:
        Filtered list of entity masks
    """
    image_height, image_width = image_shape
    image_area = image_height * image_width

    filtered = entity_masks

    # Apply size filter if hint present
    if size_hint:
        filtered = filter_by_size(filtered, size_hint, image_area)

    # Apply position filter if hint present
    if position_hint:
        filtered = filter_by_position(filtered, position_hint, image_height)

    # Always apply aspect ratio filter (remove extreme elongations)
    filtered = filter_by_aspect_ratio(filtered)

    return filtered
```

**Validation Checkpoints**:
- [ ] All 4 functions implemented (filter_by_size, filter_by_position, filter_by_aspect_ratio, apply_post_filters)
- [ ] Correct logic for size filtering (small < 2%, large > 5%)
- [ ] Correct logic for position filtering (top < 0.33, center 0.33-0.67, bottom > 0.67)
- [ ] Aspect ratio filter removes extreme shapes
- [ ] Comprehensive logging at DEBUG level for each entity
- [ ] INFO level summary logging
- [ ] Type hints and docstrings

**Claude Validation**:

```python
# Test size filtering
entities = [
    {"id": 0, "area": 50, "mask": ...},  # 0.5% of 10000px image
    {"id": 1, "area": 600, "mask": ...},  # 6% of image
]
small_only = filter_by_size(entities, "small", 10000)
assert len(small_only) == 1
assert small_only[0]["id"] == 0

large_only = filter_by_size(entities, "large", 10000)
assert len(large_only) == 1
assert large_only[0]["id"] == 1

# Test position filtering
entities = [
    {"id": 0, "bbox": [0, 50, 100, 100]},  # y_ratio = 0.1 (top)
    {"id": 1, "bbox": [0, 400, 100, 100]},  # y_ratio = 0.45 (center)
    {"id": 2, "bbox": [0, 900, 100, 100]},  # y_ratio = 0.95 (bottom)
]
top_only = filter_by_position(entities, "top", 1000)
assert len(top_only) == 1
assert top_only[0]["id"] == 0

center_only = filter_by_position(entities, "center", 1000)
assert len(center_only) == 1
assert center_only[0]["id"] == 1

bottom_only = filter_by_position(entities, "bottom", 1000)
assert len(bottom_only) == 1
assert bottom_only[0]["id"] == 2
```

---

## Quality Standards (ALL Tasks)

### Code Quality
- **Formatting**: Black style, max line length 100
- **Type hints**: All function parameters and returns
- **Docstrings**: Google style with Args, Returns, Raises
- **Logging**: Appropriate levels (DEBUG, INFO, WARNING, ERROR)
- **No magic numbers**: Use named constants
- **No hardcoded paths**: Use config or parameters

### Performance Requirements
- **Color-guided path**: < 5 seconds total
- **Semantic-only path**: < 8 seconds total
- **Memory**: < 10GB GPU VRAM
- **CPU**: Reasonable utilization (no infinite loops)

### Reliability Requirements
- **No crashes**: Graceful error handling with clear messages
- **No silent failures**: Log all errors and warnings
- **No toxic fallbacks**: Return None instead of garbage
- **Idempotent**: Running twice produces same result

### Testing Requirements
- **Unit tests**: For each function
- **Integration tests**: For each stage
- **Edge case tests**: Empty inputs, invalid inputs, extreme values
- **Performance tests**: Benchmark against requirements

---

## Claude's Validation Process

After each task, Claude will:

1. **Review Code Quality**
   - Check formatting, type hints, docstrings
   - Verify no hardcoded values or paths
   - Check logging is comprehensive

2. **Run Validation Tests**
   - Execute provided test cases
   - Verify expected outputs
   - Check performance (timing)

3. **Edge Case Testing**
   - Test with empty/null inputs
   - Test with extreme values
   - Test with known failure cases from v2

4. **Performance Benchmarking**
   - Measure execution time
   - Measure GPU memory usage
   - Compare against requirements

5. **Approval Decision**
   - **APPROVED**: Meets all quality standards → Proceed to next task
   - **NEEDS REVISION**: Issues found → Provide specific feedback for fixes
   - **BLOCKED**: Critical issue → Stop and investigate

---

## Success Metrics

### Phase 1 Success Criteria (End of Week 1)
- [ ] All working components copied successfully
- [ ] All enhancements implemented (diagnostics, adaptive thresholds, structured feedback)
- [ ] All redesigned components implemented (color mapper, intent parser)
- [ ] All new components implemented (post-filters)
- [ ] Unit tests pass for all modules
- [ ] No regressions from v2 (SAM still works on easy cases)

### Phase 2 Success Criteria (End of Week 2)
- [ ] Dual-path orchestrator implemented
- [ ] Integration tests pass
- [ ] Wildcard tests show improvement: 5-6/9 pass (up from 1/9)
- [ ] Zero false positives (down from 2/9)
- [ ] Performance meets targets

---

## Communication Protocol

### Qwen Reports After Each Task:
1. **What was implemented**: Brief summary
2. **Files created/modified**: List with line counts
3. **Tests run**: Which tests, pass/fail status
4. **Issues encountered**: Any challenges or edge cases
5. **Ready for validation**: Explicit statement

### Claude Responds With:
1. **Validation results**: Test outcomes
2. **Issues found**: Specific problems with code references
3. **Approval status**: APPROVED / NEEDS REVISION / BLOCKED
4. **Next steps**: Proceed or fix issues

---

## Emergency Protocols

### If Qwen Gets Stuck:
1. **Document the blocker**: What is preventing progress?
2. **Provide context**: Code snippets, error messages, logs
3. **Ask specific questions**: Not "how do I do this?" but "why does X fail with error Y?"
4. **Claude will**: Analyze, provide detailed guidance, update specs if needed

### If Tests Fail:
1. **Don't proceed**: Fix failing tests before moving on
2. **Investigate root cause**: Use logging, debugging, print statements
3. **Document findings**: What caused the failure?
4. **Propose fix**: Explain what needs to change
5. **Claude validates fix**: Before proceeding

---

**Next Step**: Qwen begins Task 1.1 (Workspace Setup) and reports completion.
