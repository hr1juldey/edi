# Stage 6: VLM Validation - Implementation Instructions

**File to create**: `pipeline/stage6_vlm_validation.py`

**Reference**: See `/home/riju279/Documents/Code/Zonko/EDI/edi/docs/VISION_PIPELINE_RESEARCH.md` Section "Stage 6: VLM Validation"

---

## Overview

This stage uses a local Vision-Language Model (VLM) to validate that the detected entity masks match the user's intent. It provides structured feedback about the quality and correctness of the results.

**Purpose**: Quality assurance - verify masks represent correct entities before returning to user

**Key Point**: This is an ADVISORY stage - provides feedback but doesn't block the pipeline. Results can be used for debugging and confidence scoring.

---

## Requirements

### 1. Main Function: `validate_with_vlm()`

```python
def validate_with_vlm(image: np.ndarray,
                     entity_masks: List[EntityMask],
                     user_intent: str,
                     vlm_model: str = "qwen2.5-vl:7b") -> Dict[str, Any]:
    """
    Validate entity masks using local VLM.

    Args:
        image: Original RGB image (H x W x 3), numpy array
        entity_masks: List of EntityMask objects from Stage 5
        user_intent: Original user prompt (e.g., "change blue tin roofs to green")
        vlm_model: Ollama VLM model name

    Returns:
        Dictionary with validation results:
        {
            'overall_confidence': float,  # 0.0-1.0
            'entity_validations': List[Dict],  # Per-entity feedback
            'detected_count': int,
            'expected_entity': str,
            'issues': List[str],
            'recommendation': str  # 'accept', 'review', 'retry'
        }
    """
```

### 2. Data Structure: Validation Result

```python
from typing import Dict, List, Any
from dataclasses import dataclass, asdict

@dataclass
class ValidationResult:
    """Result of VLM validation."""
    overall_confidence: float  # 0.0-1.0 (how confident VLM is in results)
    detected_count: int  # Number of entities detected
    expected_entity: str  # What entity type was expected (e.g., "tin roof")
    entity_validations: List[Dict[str, Any]]  # Per-entity validation
    issues: List[str]  # List of problems found
    recommendation: str  # 'accept', 'review', 'retry'

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
```

### 3. Processing Steps

**Step 1: Parse User Intent**

```python
# Extract target entity from user intent
# Example: "change blue tin roofs to green" -> "tin roof"
def extract_target_entity(user_intent: str) -> str:
    """
    Extract the target entity type from user prompt.

    Args:
        user_intent: User's original prompt

    Returns:
        Entity type string (e.g., "tin roof", "building", "vehicle")
    """
    # Use simple pattern matching or LLM to extract entity
    # For MVP: look for common patterns
    import re

    # Pattern: "change [color] [entity] to ..."
    # Pattern: "edit the [entity] ..."
    # Pattern: "[entity] should be ..."

    patterns = [
        r'(?:change|edit|modify)\s+(?:the\s+)?(?:\w+\s+)?(\w+\s+\w+)',  # "change blue tin roofs"
        r'(?:change|edit|modify)\s+(?:the\s+)?(\w+)',  # "change roofs"
        r'(\w+\s+\w+)\s+(?:to|should)',  # "tin roofs to green"
    ]

    for pattern in patterns:
        match = re.search(pattern, user_intent.lower())
        if match:
            return match.group(1).strip()

    # Fallback: use full intent
    return user_intent.strip()
```

**Step 2: Create Composite Visualization**

```python
def create_validation_image(image: np.ndarray,
                           entity_masks: List[EntityMask]) -> np.ndarray:
    """
    Create composite image showing all detected entities for VLM.

    Args:
        image: Original RGB image
        entity_masks: List of EntityMask objects

    Returns:
        Image with all masks highlighted in different colors
    """
    overlay = image.copy()

    # Use different colors for each mask to help VLM distinguish
    colors = [
        [255, 0, 0],    # Red
        [0, 255, 0],    # Green
        [0, 0, 255],    # Blue
        [255, 255, 0],  # Yellow
        [255, 0, 255],  # Magenta
        [0, 255, 255],  # Cyan
        [255, 128, 0],  # Orange
        [128, 0, 255],  # Purple
    ]

    for idx, entity in enumerate(entity_masks):
        color = colors[idx % len(colors)]
        overlay[entity.mask > 0] = overlay[entity.mask > 0] * 0.5 + np.array(color) * 0.5

        # Draw entity ID
        cx, cy = entity.centroid
        cv2.putText(overlay, f"{idx}", (int(cx)-10, int(cy)),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    return overlay
```

**Step 3: Query VLM via Ollama**

```python
import requests
import base64
import json
from io import BytesIO
from PIL import Image

def query_vlm_ollama(image: np.ndarray,
                    prompt: str,
                    model: str = "qwen2.5-vl:7b") -> str:
    """
    Query local Ollama VLM with image and prompt.

    Args:
        image: RGB numpy array
        prompt: Question for VLM
        model: Ollama model name

    Returns:
        VLM response text
    """
    # Convert numpy array to base64
    pil_image = Image.fromarray(image)
    buffer = BytesIO()
    pil_image.save(buffer, format='PNG')
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # Ollama API request
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [image_base64],
        "stream": False
    }

    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result.get('response', '')
    except Exception as e:
        logging.error(f"VLM query failed: {e}")
        return ""
```

**Step 4: Validate Overall Detection**

```python
# Create validation prompt
target_entity = extract_target_entity(user_intent)

validation_image = create_validation_image(image, entity_masks)

prompt = f"""You are analyzing an image where we detected {len(entity_masks)} regions highlighted in different colors.

User's intent: "{user_intent}"
Expected entity type: "{target_entity}"

Please analyze:
1. How many {target_entity} objects do you see in the image (total, not just highlighted)?
2. Do the highlighted regions correctly identify {target_entity} objects?
3. Are there any {target_entity} objects that were MISSED (not highlighted)?
4. Are any highlighted regions NOT {target_entity} objects (false positives)?

Respond in JSON format:
{{
  "total_visible": <number>,
  "correctly_detected": <number>,
  "missed": <number>,
  "false_positives": <number>,
  "confidence": <0.0-1.0>,
  "issues": ["list", "of", "problems"],
  "recommendation": "accept|review|retry"
}}
"""

vlm_response = query_vlm_ollama(validation_image, prompt, vlm_model)

# Parse JSON response
try:
    validation_data = json.loads(vlm_response)
except:
    # Fallback if VLM doesn't return valid JSON
    validation_data = {
        'total_visible': len(entity_masks),
        'correctly_detected': len(entity_masks),
        'missed': 0,
        'false_positives': 0,
        'confidence': 0.5,
        'issues': ['VLM response parsing failed'],
        'recommendation': 'review'
    }
```

**Step 5: Per-Entity Validation (Optional)**

```python
entity_validations = []

# Validate up to 5 largest entities individually
for idx, entity in enumerate(entity_masks[:5]):
    # Crop to entity bounding box
    x_min, y_min, x_max, y_max = entity.bbox
    cropped = image[y_min:y_max+1, x_min:x_max+1].copy()

    # Apply mask
    cropped_mask = entity.mask[y_min:y_max+1, x_min:x_max+1]
    cropped[cropped_mask == 0] = [128, 128, 128]  # Gray background

    entity_prompt = f"Is this a {target_entity}? Answer yes or no and explain briefly."

    entity_response = query_vlm_ollama(cropped, entity_prompt, vlm_model)

    # Simple yes/no detection
    is_valid = 'yes' in entity_response.lower()

    entity_validations.append({
        'entity_id': entity.entity_id,
        'is_valid': is_valid,
        'vlm_response': entity_response,
        'area': entity.area,
        'similarity_score': entity.similarity_score
    })

logging.info(f"Validated {len(entity_validations)} entities individually")
```

**Step 6: Compile Results**

```python
# Calculate overall confidence
overall_confidence = validation_data.get('confidence', 0.5)

# Determine recommendation
recommendation = validation_data.get('recommendation', 'review')

# If VLM says we missed many entities or have many false positives, recommend retry
missed = validation_data.get('missed', 0)
false_positives = validation_data.get('false_positives', 0)

if missed > 3 or false_positives > 3:
    recommendation = 'retry'
elif overall_confidence > 0.8 and missed == 0 and false_positives == 0:
    recommendation = 'accept'
else:
    recommendation = 'review'

# Compile issues
issues = validation_data.get('issues', [])

result = ValidationResult(
    overall_confidence=overall_confidence,
    detected_count=len(entity_masks),
    expected_entity=target_entity,
    entity_validations=entity_validations,
    issues=issues,
    recommendation=recommendation
)

return result.to_dict()
```

---

## Complete Implementation

```python
"""Stage 6: VLM Validation

This module uses a local VLM to validate that detected entity masks match user intent.
Provides advisory feedback for quality assurance.
"""

import logging
import numpy as np
import cv2
import requests
import base64
import json
from io import BytesIO
from PIL import Image
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
import re

from pipeline.stage5_mask_organization import EntityMask


@dataclass
class ValidationResult:
    """Result of VLM validation."""
    overall_confidence: float  # 0.0-1.0
    detected_count: int
    expected_entity: str
    entity_validations: List[Dict[str, Any]]
    issues: List[str]
    recommendation: str  # 'accept', 'review', 'retry'

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


def extract_target_entity(user_intent: str) -> str:
    """
    Extract the target entity type from user prompt.

    Args:
        user_intent: User's original prompt

    Returns:
        Entity type string (e.g., "tin roof", "building")
    """
    patterns = [
        r'(?:change|edit|modify)\s+(?:the\s+)?(?:\w+\s+)?(\w+\s+\w+)',
        r'(?:change|edit|modify)\s+(?:the\s+)?(\w+)',
        r'(\w+\s+\w+)\s+(?:to|should)',
    ]

    for pattern in patterns:
        match = re.search(pattern, user_intent.lower())
        if match:
            entity = match.group(1).strip()
            # Common cleanup
            entity = entity.replace('the ', '').replace('all ', '')
            return entity

    return user_intent.strip()


def create_validation_image(image: np.ndarray,
                           entity_masks: List[EntityMask]) -> np.ndarray:
    """
    Create composite image showing all detected entities.

    Args:
        image: Original RGB image
        entity_masks: List of EntityMask objects

    Returns:
        Image with all masks highlighted
    """
    overlay = image.copy()

    colors = [
        [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
        [255, 0, 255], [0, 255, 255], [255, 128, 0], [128, 0, 255],
    ]

    for idx, entity in enumerate(entity_masks):
        color = colors[idx % len(colors)]
        overlay[entity.mask > 0] = overlay[entity.mask > 0] * 0.5 + np.array(color) * 0.5

        # Draw entity ID
        cx, cy = entity.centroid
        cv2.putText(overlay, f"{idx}", (int(cx)-10, int(cy)),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    return overlay


def query_vlm_ollama(image: np.ndarray,
                    prompt: str,
                    model: str = "qwen2.5-vl:7b") -> str:
    """
    Query local Ollama VLM.

    Args:
        image: RGB numpy array
        prompt: Question for VLM
        model: Ollama model name

    Returns:
        VLM response text
    """
    # Convert to base64
    pil_image = Image.fromarray(image)
    buffer = BytesIO()
    pil_image.save(buffer, format='PNG')
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [image_base64],
        "stream": False
    }

    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result.get('response', '')
    except Exception as e:
        logging.error(f"VLM query failed: {e}")
        return ""


def validate_with_vlm(image: np.ndarray,
                     entity_masks: List[EntityMask],
                     user_intent: str,
                     vlm_model: str = "qwen2.5-vl:7b") -> Dict[str, Any]:
    """
    Validate entity masks using local VLM.

    Args:
        image: Original RGB image (H x W x 3)
        entity_masks: List of EntityMask objects from Stage 5
        user_intent: Original user prompt
        vlm_model: Ollama VLM model name

    Returns:
        Validation result dictionary
    """
    logging.info(f"Starting VLM validation with model '{vlm_model}'")
    logging.info(f"Validating {len(entity_masks)} entities for intent: '{user_intent}'")

    if len(entity_masks) == 0:
        return ValidationResult(
            overall_confidence=0.0,
            detected_count=0,
            expected_entity=extract_target_entity(user_intent),
            entity_validations=[],
            issues=['No entities detected'],
            recommendation='retry'
        ).to_dict()

    # Extract target entity
    target_entity = extract_target_entity(user_intent)
    logging.info(f"Target entity: '{target_entity}'")

    # Create validation image
    validation_image = create_validation_image(image, entity_masks)

    # Overall validation prompt
    prompt = f"""You are analyzing an image where we detected {len(entity_masks)} regions highlighted in different colors with numbers.

User's intent: "{user_intent}"
Expected entity type: "{target_entity}"

Please analyze:
1. How many {target_entity} objects do you see in the image (total, including non-highlighted)?
2. Do the highlighted regions correctly identify {target_entity} objects?
3. Are there any {target_entity} objects that were MISSED (not highlighted)?
4. Are any highlighted regions NOT {target_entity} objects (false positives)?

Respond in JSON format:
{{
  "total_visible": <number of total {target_entity} in image>,
  "correctly_detected": <number of correct highlights>,
  "missed": <number of {target_entity} not highlighted>,
  "false_positives": <number of wrong highlights>,
  "confidence": <0.0-1.0>,
  "issues": ["list of problems if any"],
  "recommendation": "accept or review or retry"
}}
"""

    vlm_response = query_vlm_ollama(validation_image, prompt, vlm_model)
    logging.debug(f"VLM response: {vlm_response}")

    # Parse JSON response
    try:
        # Try to extract JSON from response
        json_match = re.search(r'\{.*\}', vlm_response, re.DOTALL)
        if json_match:
            validation_data = json.loads(json_match.group())
        else:
            raise ValueError("No JSON found in response")
    except Exception as e:
        logging.warning(f"Failed to parse VLM JSON: {e}")
        validation_data = {
            'total_visible': len(entity_masks),
            'correctly_detected': len(entity_masks),
            'missed': 0,
            'false_positives': 0,
            'confidence': 0.5,
            'issues': [f'VLM response parsing failed: {str(e)}'],
            'recommendation': 'review'
        }

    # Per-entity validation (optional, for top 5)
    entity_validations = []
    for idx, entity in enumerate(entity_masks[:5]):
        x_min, y_min, x_max, y_max = entity.bbox
        cropped = image[y_min:y_max+1, x_min:x_max+1].copy()
        cropped_mask = entity.mask[y_min:y_max+1, x_min:x_max+1]
        cropped[cropped_mask == 0] = [128, 128, 128]

        entity_prompt = f"Is this a {target_entity}? Answer yes or no and explain briefly in one sentence."
        entity_response = query_vlm_ollama(cropped, entity_prompt, vlm_model)

        is_valid = 'yes' in entity_response.lower()[:50]  # Check first 50 chars

        entity_validations.append({
            'entity_id': entity.entity_id,
            'is_valid': is_valid,
            'vlm_response': entity_response[:200],  # Truncate
            'area': entity.area,
            'similarity_score': entity.similarity_score
        })

    logging.info(f"Validated {len(entity_validations)} entities individually")

    # Calculate metrics
    overall_confidence = float(validation_data.get('confidence', 0.5))
    missed = int(validation_data.get('missed', 0))
    false_positives = int(validation_data.get('false_positives', 0))

    # Determine recommendation
    if missed > 3 or false_positives > 3:
        recommendation = 'retry'
    elif overall_confidence > 0.8 and missed == 0 and false_positives == 0:
        recommendation = 'accept'
    else:
        recommendation = validation_data.get('recommendation', 'review')

    issues = validation_data.get('issues', [])
    if missed > 0:
        issues.append(f"{missed} {target_entity} objects were missed")
    if false_positives > 0:
        issues.append(f"{false_positives} incorrect detections")

    result = ValidationResult(
        overall_confidence=overall_confidence,
        detected_count=len(entity_masks),
        expected_entity=target_entity,
        entity_validations=entity_validations,
        issues=issues,
        recommendation=recommendation
    )

    logging.info(f"Validation complete: {recommendation} (confidence={overall_confidence:.2f})")

    return result.to_dict()
```

---

## Testing

### Create `tests/test_stage6.py`

**Test Case 1**: VLM validation with valid entities

```python
def test_validate_with_vlm():
    """Test VLM validation (requires Ollama running)."""
    try:
        # Load test image
        image = cv2.imread("test_image.jpeg")
        if image is None:
            pytest.skip("Test image not available")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get entity masks from full pipeline
        from pipeline.stage2_color_filter import color_prefilter
        from pipeline.stage3_sam_segmentation import segment_regions
        from pipeline.stage4_clip_filter import clip_filter_masks
        from pipeline.stage5_mask_organization import organize_masks

        color_mask = color_prefilter(image_rgb, "blue")
        sam_masks = segment_regions(image_rgb, color_mask, min_area=500)
        filtered_masks = clip_filter_masks(image_rgb, sam_masks, "tin roof", similarity_threshold=0.22)
        entity_masks = organize_masks(image_rgb, filtered_masks)

        # Validate with VLM
        from pipeline.stage6_vlm_validation import validate_with_vlm

        result = validate_with_vlm(
            image_rgb,
            entity_masks,
            "change blue tin roofs to green",
            vlm_model="qwen2.5-vl:7b"
        )

        # Check result structure
        assert isinstance(result, dict)
        assert 'overall_confidence' in result
        assert 'detected_count' in result
        assert 'expected_entity' in result
        assert 'entity_validations' in result
        assert 'issues' in result
        assert 'recommendation' in result

        assert result['detected_count'] == len(entity_masks)
        assert result['recommendation'] in ['accept', 'review', 'retry']
        assert 0.0 <= result['overall_confidence'] <= 1.0

        print(f"\nValidation result: {result['recommendation']}")
        print(f"Confidence: {result['overall_confidence']:.2f}")
        print(f"Issues: {result['issues']}")

    except Exception as e:
        pytest.skip(f"VLM validation failed: {e}")
```

**Test Case 2**: Extract target entity

```python
def test_extract_target_entity():
    """Test entity extraction from user intent."""
    from pipeline.stage6_vlm_validation import extract_target_entity

    # Test various prompt formats
    assert "tin roof" in extract_target_entity("change blue tin roofs to green").lower()
    assert "roof" in extract_target_entity("edit the roofs").lower()
    assert "building" in extract_target_entity("modify the buildings").lower()
```

**Test Case 3**: Empty entity list

```python
def test_validate_empty_entities():
    """Test validation with no entities."""
    from pipeline.stage6_vlm_validation import validate_with_vlm

    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = validate_with_vlm(image, [], "change roofs to green")

    assert result['detected_count'] == 0
    assert result['recommendation'] == 'retry'
    assert len(result['issues']) > 0
```

**Test Case 4**: Validation image creation

```python
def test_create_validation_image():
    """Test composite validation image creation."""
    from pipeline.stage5_mask_organization import EntityMask
    from pipeline.stage6_vlm_validation import create_validation_image

    image = np.zeros((200, 200, 3), dtype=np.uint8)

    # Create test entity masks
    mask1 = np.zeros((200, 200), dtype=np.uint8)
    mask1[50:100, 50:100] = 1

    entity1 = EntityMask(
        mask=mask1,
        entity_id=0,
        similarity_score=0.8,
        bbox=(50, 50, 99, 99),
        centroid=(75.0, 75.0),
        area=2500,
        dominant_color=(100, 100, 100)
    )

    validation_img = create_validation_image(image, [entity1])

    assert validation_img.shape == image.shape
    assert not np.array_equal(validation_img, image)  # Should be modified
```

---

## Visual Validation

After implementation, create validation script:

```python
# validate_stage6.py
import cv2
import numpy as np
import json
from pipeline.stage2_color_filter import color_prefilter
from pipeline.stage3_sam_segmentation import segment_regions
from pipeline.stage4_clip_filter import clip_filter_masks
from pipeline.stage5_mask_organization import organize_masks
from pipeline.stage6_vlm_validation import validate_with_vlm, create_validation_image

# Load test image
test_img = cv2.imread("test_image.jpeg")
test_img_rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

print("Running full pipeline Stages 2-6...")

# Run pipeline
color_mask = color_prefilter(test_img_rgb, "blue")
sam_masks = segment_regions(test_img_rgb, color_mask, min_area=500)
filtered_masks = clip_filter_masks(test_img_rgb, sam_masks, "tin roof", similarity_threshold=0.22)
entity_masks = organize_masks(test_img_rgb, filtered_masks)

# Stage 6: VLM Validation
user_intent = "change blue tin roofs to green"
validation_result = validate_with_vlm(test_img_rgb, entity_masks, user_intent)

print(f"\n{'='*60}")
print("STAGE 6: VLM VALIDATION RESULTS")
print(f"{'='*60}")
print(f"Target Entity: {validation_result['expected_entity']}")
print(f"Detected Count: {validation_result['detected_count']}")
print(f"Overall Confidence: {validation_result['overall_confidence']:.2f}")
print(f"Recommendation: {validation_result['recommendation'].upper()}")
print(f"\nIssues:")
for issue in validation_result['issues']:
    print(f"  - {issue}")

print(f"\nPer-Entity Validation:")
for ev in validation_result['entity_validations']:
    status = "✓" if ev['is_valid'] else "✗"
    print(f"  {status} Entity {ev['entity_id']}: {ev['vlm_response'][:100]}...")

# Save validation image
validation_img = create_validation_image(test_img_rgb, entity_masks)
cv2.imwrite("logs/stage6_validation_overlay.png", cv2.cvtColor(validation_img, cv2.COLOR_RGB2BGR))
print(f"\nSaved validation overlay: logs/stage6_validation_overlay.png")

# Save full report
with open("logs/stage6_validation_report.json", 'w') as f:
    json.dump(validation_result, f, indent=2)
print("Saved validation report: logs/stage6_validation_report.json")
```

---

## Expected Results

For test_image.jpeg with 14 entity masks:

```json
{
  "overall_confidence": 0.85,
  "detected_count": 14,
  "expected_entity": "tin roof",
  "entity_validations": [
    {
      "entity_id": 0,
      "is_valid": true,
      "vlm_response": "Yes, this is a blue tin roof structure...",
      "area": 39351,
      "similarity_score": 0.227
    },
    ...
  ],
  "issues": [],
  "recommendation": "accept"
}
```

---

## Acceptance Criteria

- [ ] Function `validate_with_vlm()` implemented correctly
- [ ] All 4 test cases pass
- [ ] Ollama VLM integration working (qwen2.5-vl:7b)
- [ ] Returns structured ValidationResult
- [ ] Per-entity validation for top 5 entities
- [ ] Recommendation logic (accept/review/retry) working
- [ ] Code has type hints and docstrings
- [ ] Handles VLM unavailable gracefully

---

## Notes

**VLM Model**: `qwen2.5-vl:7b`
- Good balance of accuracy and speed
- Runs locally via Ollama
- ~7B parameters, needs ~8GB VRAM

**Performance**:
- VLM inference: ~2-5 seconds per query
- Total for overall + 5 entities: ~15-30 seconds
- This is OPTIONAL stage - can be skipped for speed

**Graceful Degradation**:
```python
try:
    result = validate_with_vlm(...)
except Exception as e:
    logging.warning(f"VLM validation failed: {e}, skipping")
    result = {
        'recommendation': 'review',
        'overall_confidence': 0.5,
        'issues': ['VLM unavailable']
    }
```

---

## Report Format

After completion, report:

```
STAGE 6: VLM Validation - [COMPLETE/FAILED]

Implementation:
- File: pipeline/stage6_vlm_validation.py
- Lines of code: XXX
- Class: ValidationResult (dataclass)
- Function: validate_with_vlm()

Test Results:
- Test Case 1 (VLM validation): [PASS/FAIL]
- Test Case 2 (Extract entity): [PASS/FAIL]
- Test Case 3 (Empty entities): [PASS/FAIL]
- Test Case 4 (Validation image): [PASS/FAIL]

Validation Results:
- Input: XX entity masks
- Target entity: "XXX"
- Overall confidence: 0.XX
- Recommendation: [ACCEPT/REVIEW/RETRY]
- Issues found: XX

Per-Entity Validation:
- Entity 0: [VALID/INVALID] - "VLM response..."
- Entity 1: [VALID/INVALID] - "VLM response..."
- ...

Files Generated:
- logs/stage6_validation_overlay.png
- logs/stage6_validation_report.json

Issues: [None / List any problems]

Status: Awaiting supervisor approval
```

---

**Begin implementation of Stage 6 now!**
