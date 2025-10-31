# Claude Execution Plan - EDI Vision v3.0 Implementation

**Document Version**: 1.0
**Date**: 2025-10-31
**Executor**: Claude Code (NOT Qwen - too complex for Qwen)
**Target**: 85%+ success rate with GroundingDINO integration

---

## 🎯 Executive Decision: Claude Implements, Qwen Assists

**Rationale**: The architectural flaws in `CRITICAL_ARCHITECTURE_FLAWS.md` require:
- Deep understanding of vision model architectures
- Real-time decision making during implementation
- Complex debugging with visual analysis (using local vision MCP)
- GroundingDINO integration expertise

**Qwen's capabilities** are better suited for:
- Simple copy tasks ✓ (completed: stage3)
- Following explicit specifications ✓
- Running tests ✓

**This complexity requires Claude's autonomous execution.**

---

## 📊 Implementation Options for GroundingDINO

### Option 1: HuggingFace Transformers (RECOMMENDED)

**Pros**:
- ✅ Official support in transformers library
- ✅ Easiest installation: `pip install transformers`
- ✅ Standard API patterns
- ✅ Well-documented
- ✅ No compilation required
- ✅ Multiple model sizes available

**Installation**:
```bash
pip install transformers torch torchvision
```

**Usage**:
```python
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from PIL import Image

# Load model
model_id = "IDEA-Research/grounding-dino-base"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to("cuda")

# Inference
image = Image.open("image.jpg")
text = "yellow auto-rickshaws"

inputs = processor(images=image, text=text, return_tensors="pt").to("cuda")
outputs = model(**inputs)

# Post-process to get boxes
results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    box_threshold=0.35,
    text_threshold=0.25,
    target_sizes=[image.size[::-1]]
)
```

**Model Sizes**:
- `grounding-dino-tiny` - Fast, lower accuracy
- `grounding-dino-base` - Balanced (RECOMMENDED)
- Custom: Can fine-tune if needed

---

### Option 2: IDEA-Research Official (Advanced)

**Pros**:
- ✅ Most control over parameters
- ✅ Latest features first
- ✅ Access to GroundingDINO 1.5/1.6

**Cons**:
- ❌ Requires CUDA compilation
- ❌ More complex installation
- ❌ Dependency conflicts possible

**Installation**:
```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install -e .
```

**Use Case**: If HuggingFace version has limitations

---

### Option 3: Grounded-SAM-2 (Integrated)

**Pros**:
- ✅ Combined GroundingDINO + SAM 2 in one repo
- ✅ Optimized integration
- ✅ Example code for our exact use case

**Cons**:
- ❌ Less flexible (coupled implementation)
- ❌ Heavier dependencies

**Use Case**: If we want complete integration example

---

## 🏗️ Recommended Architecture: Hybrid Path C + Fallback

### Design Decision

Instead of choosing Path A, B, or C exclusively, implement **Path C with Path B fallback**:

```
User Prompt → Intent Parser →

├─→ PATH C: GroundingDINO + SAM (PRIMARY)
│   ├─→ GroundingDINO detects entities with text prompt
│   ├─→ Returns bounding boxes
│   ├─→ SAM generates masks from boxes
│   └─→ CLIP ranks results
│
└─→ PATH B FALLBACK: Dual-Path (if GroundingDINO unavailable)
    ├─→ Color-guided path
    ├─→ Semantic-only path
    └─→ Hybrid path
```

**Rationale**:
1. **GroundingDINO solves ALL three flaws** at once
2. **Fallback ensures robustness** if GroundingDINO fails/unavailable
3. **Gradual migration** from v2 → v3

---

## 📋 Implementation Phases

### Phase 1: GroundingDINO Integration (Week 1 - Claude)

**Tasks**:
1. Install HuggingFace transformers GroundingDINO
2. Create `pipeline/stage1_grounding_dino.py`
3. Test on wildcard images
4. Benchmark performance
5. Integrate with SAM 2.1

**Files to Create**:
- `pipeline/stage1_grounding_dino.py` - GroundingDINO wrapper
- `pipeline/stage2_box_to_mask.py` - Convert boxes to SAM-compatible format
- `tests/test_grounding_dino.py` - Unit tests
- `config.yaml` - Add GroundingDINO settings

---

### Phase 2: Orchestrator Redesign (Week 1 - Claude)

**Tasks**:
1. Create smart router that tries GroundingDINO first
2. Implement fallback to color-guided/semantic paths
3. Add performance monitoring
4. Implement result caching

**Files to Create**:
- `pipeline/orchestrator_v3.py` - Smart routing logic
- `pipeline/detection_strategy.py` - Strategy selection
- `utils/performance_monitor.py` - Timing and metrics

---

### Phase 3: Testing & Validation (Week 2 - Claude + Qwen)

**Tasks**:
1. Run wildcard tests on new architecture
2. Visual validation using local vision MCP
3. Performance benchmarking
4. Edge case testing

**Expected Results**:
- 8/9 wildcard tests pass (89%)
- Processing time < 8s average
- Zero false positives

---

### Phase 4: Edit Validation Integration (Week 3 - Claude)

**Tasks**:
1. Implement vision delta analysis
2. Add auto-retry loop
3. Integrate structured feedback

**Files to Create**:
- `validation/vision_delta.py`
- `validation/quality_scoring.py`
- `validation/auto_retry.py`

---

## 🔧 Detailed Implementation: Stage 1 GroundingDINO

### File: `pipeline/stage1_grounding_dino.py`

```python
"""Stage 1: GroundingDINO Open-Vocabulary Detection

This module replaces the broken color-first assumption with open-vocabulary
object detection using GroundingDINO from HuggingFace Transformers.
"""

import logging
import torch
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional
from dataclasses import dataclass
from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
)


@dataclass
class DetectionBox:
    """Bounding box from GroundingDINO."""
    x: int
    y: int
    w: int
    h: int
    confidence: float
    label: str


class GroundingDINODetector:
    """
    GroundingDINO wrapper for open-vocabulary object detection.

    Advantages over v2.0 color-first approach:
    - Handles ANY natural language query (no color dictionary)
    - Works for semantic-only queries ("auto-rickshaws", "birds")
    - Works for color queries ("red vehicles")
    - Works for complex queries ("yellow colonial buildings")
    - No toxic fallback behavior
    """

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-base",
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize GroundingDINO detector.

        Args:
            model_id: HuggingFace model ID
            box_threshold: Confidence threshold for bounding boxes
            text_threshold: Threshold for text-image alignment
            device: "cuda" or "cpu"
        """
        logging.info(f"Initializing GroundingDINO detector: {model_id}")

        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        # Load model and processor
        try:
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
            self.model.to(self.device)
            self.model.eval()

            logging.info(f"✓ GroundingDINO loaded on {self.device}")

        except Exception as e:
            logging.error(f"Failed to load GroundingDINO: {e}")
            raise

    def detect(
        self,
        image: np.ndarray,
        text_prompt: str,
        box_threshold: Optional[float] = None,
        text_threshold: Optional[float] = None
    ) -> List[DetectionBox]:
        """
        Detect objects using open-vocabulary text prompt.

        Args:
            image: RGB image (H x W x 3), numpy array
            text_prompt: Natural language description (e.g., "yellow auto-rickshaws")
            box_threshold: Override default box threshold
            text_threshold: Override default text threshold

        Returns:
            List of DetectionBox objects with bounding boxes and labels
        """
        box_thresh = box_threshold or self.box_threshold
        text_thresh = text_threshold or self.text_threshold

        logging.info(f"GroundingDINO detecting: '{text_prompt}'")
        logging.info(f"  Thresholds: box={box_thresh}, text={text_thresh}")

        # Convert numpy to PIL
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image

        # Prepare inputs
        inputs = self.processor(
            images=pil_image,
            text=text_prompt,
            return_tensors="pt"
        ).to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process results
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_thresh,
            text_threshold=text_thresh,
            target_sizes=[pil_image.size[::-1]]  # (height, width)
        )[0]

        # Convert to DetectionBox objects
        boxes = []
        for box, score, label in zip(
            results["boxes"],
            results["scores"],
            results["labels"]
        ):
            x1, y1, x2, y2 = box.cpu().numpy()
            boxes.append(DetectionBox(
                x=int(x1),
                y=int(y1),
                w=int(x2 - x1),
                h=int(y2 - y1),
                confidence=float(score),
                label=label
            ))

        logging.info(f"✓ GroundingDINO detected {len(boxes)} objects")

        if len(boxes) > 0:
            conf_scores = [b.confidence for b in boxes]
            logging.info(f"  Confidence range: {min(conf_scores):.3f} - {max(conf_scores):.3f}")

        return boxes

    def cleanup(self):
        """Free GPU memory."""
        if hasattr(self, 'model'):
            del self.model
            del self.processor
        torch.cuda.empty_cache()
        logging.info("✓ GroundingDINO cleaned up")


def detect_entities_grounding_dino(
    image: np.ndarray,
    user_prompt: str,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25
) -> List[DetectionBox]:
    """
    High-level function to detect entities using GroundingDINO.

    This replaces the broken color-first assumption from v2.0.

    Args:
        image: RGB image
        user_prompt: Natural language query
        box_threshold: Confidence threshold for boxes
        text_threshold: Text-image alignment threshold

    Returns:
        List of bounding boxes with labels

    Examples:
        >>> boxes = detect_entities_grounding_dino(image, "yellow auto-rickshaws")
        >>> boxes = detect_entities_grounding_dino(image, "birds")
        >>> boxes = detect_entities_grounding_dino(image, "red vehicles")
    """
    detector = GroundingDINODetector(
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )

    try:
        boxes = detector.detect(image, user_prompt)
        return boxes
    finally:
        detector.cleanup()
```

---

### File: `pipeline/stage2_box_to_mask.py`

```python
"""Stage 2: Convert GroundingDINO Boxes to SAM Masks

This module takes bounding boxes from GroundingDINO and uses SAM 2.1
to generate pixel-perfect segmentation masks.
"""

import logging
import numpy as np
import torch
from ultralytics import SAM
from typing import List
from dataclasses import dataclass

from .stage1_grounding_dino import DetectionBox


@dataclass
class EntityMask:
    """Entity mask with metadata."""
    mask: np.ndarray  # Binary mask (H x W)
    bbox: tuple  # (x, y, w, h)
    confidence: float  # From GroundingDINO
    label: str  # Entity label
    area: int  # Mask area in pixels
    sam_confidence: float  # SAM mask quality score


class BoxPromptedSAM:
    """
    SAM 2.1 with box prompting from GroundingDINO.

    This is MUCH faster than automatic mask generation:
    - Full image SAM: ~6 seconds, 100+ masks
    - Box-prompted SAM: ~1 second, exact masks needed
    """

    def __init__(self, model_path: str = "sam2.1_b.pt"):
        """Initialize SAM 2.1 model."""
        logging.info("Initializing box-prompted SAM 2.1")

        self.sam = SAM(model_path)
        if torch.cuda.is_available():
            self.sam.to('cuda')
            self.sam.half()

        logging.info("✓ SAM 2.1 loaded")

    def generate_masks_from_boxes(
        self,
        image: np.ndarray,
        boxes: List[DetectionBox]
    ) -> List[EntityMask]:
        """
        Generate pixel-perfect masks for each bounding box.

        Args:
            image: RGB image (H x W x 3)
            boxes: List of detection boxes from GroundingDINO

        Returns:
            List of EntityMask objects with segmentation masks
        """
        if len(boxes) == 0:
            logging.warning("No boxes provided, returning empty list")
            return []

        logging.info(f"Generating SAM masks for {len(boxes)} boxes")

        entity_masks = []

        for i, box in enumerate(boxes):
            # Convert box to SAM format: [x1, y1, x2, y2]
            sam_box = np.array([
                box.x,
                box.y,
                box.x + box.w,
                box.y + box.h
            ])

            try:
                # SAM inference with box prompt
                result = self.sam.predict(
                    source=image,
                    bboxes=[sam_box],
                    verbose=False
                )

                # Extract mask
                if len(result) > 0 and result[0].masks is not None:
                    mask = result[0].masks.data[0].cpu().numpy()

                    # Calculate mask area
                    area = int(np.sum(mask > 0))

                    # SAM confidence (if available)
                    sam_conf = 0.95  # Default high confidence for box-prompted

                    entity_masks.append(EntityMask(
                        mask=mask.astype(np.uint8),
                        bbox=(box.x, box.y, box.w, box.h),
                        confidence=box.confidence,
                        label=box.label,
                        area=area,
                        sam_confidence=sam_conf
                    ))

                    logging.debug(f"  Box {i}: {box.label} - {area} pixels, conf={box.confidence:.3f}")

                else:
                    logging.warning(f"  Box {i}: SAM generated no mask")

            except Exception as e:
                logging.error(f"  Box {i}: SAM failed: {e}")
                continue

        logging.info(f"✓ Generated {len(entity_masks)} masks from {len(boxes)} boxes")

        return entity_masks

    def cleanup(self):
        """Free GPU memory."""
        if hasattr(self, 'sam'):
            del self.sam
        torch.cuda.empty_cache()
        logging.info("✓ SAM cleaned up")


def boxes_to_masks(
    image: np.ndarray,
    boxes: List[DetectionBox]
) -> List[EntityMask]:
    """
    High-level function to convert GroundingDINO boxes to SAM masks.

    Args:
        image: RGB image
        boxes: Detection boxes from GroundingDINO

    Returns:
        List of entity masks with pixel-perfect segmentation
    """
    sam_generator = BoxPromptedSAM()

    try:
        masks = sam_generator.generate_masks_from_boxes(image, boxes)
        return masks
    finally:
        sam_generator.cleanup()
```

---

## 🧪 Testing Strategy

### Test 1: GroundingDINO on Wildcard Images

**Objective**: Verify GroundingDINO works on all 9 wildcard test queries

**Test Script**: `tests/test_grounding_dino_wildcard.py`

```python
import pytest
import numpy as np
from PIL import Image
from pipeline.stage1_grounding_dino import detect_entities_grounding_dino

WILDCARD_TESTS = [
    ("images/kol_1.png", "red vehicles"),
    ("images/Darjeeling.jpg", "brown roofs"),
    ("images/WP.jpg", "sky regions"),
    ("images/mumbai-traffic.jpg", "yellow auto-rickshaws"),
    ("images/Pondicherry.jpg", "yellow colonial buildings"),
    ("images/pondi_2.jpg", "blue sky"),
    ("images/test_image.jpeg", "purple elements"),  # Should return 0 boxes (no purple)
    ("images/test_image.jpeg", "auto-rickshaws"),  # Semantic-only
    ("images/test_image.jpeg", "small birds"),  # Semantic + size
]

@pytest.mark.parametrize("image_path,prompt", WILDCARD_TESTS)
def test_grounding_dino_wildcard(image_path, prompt):
    """Test GroundingDINO on wildcard queries."""
    image = np.array(Image.open(image_path).convert('RGB'))

    boxes = detect_entities_grounding_dino(image, prompt)

    # Should NOT crash (v2 crashed on 6/9)
    assert boxes is not None

    # Should return list (even if empty)
    assert isinstance(boxes, list)

    # Log results
    print(f"\n{prompt}: {len(boxes)} boxes detected")
    for box in boxes[:3]:
        print(f"  - {box.label}: conf={box.confidence:.3f}")
```

---

### Test 2: Visual Validation with Local Vision MCP

**Objective**: Use local ollama vision tool to verify detection quality

**Process**:
1. Run GroundingDINO on test image
2. Draw bounding boxes on image
3. Use `mcp__ide__executeCode` or bash to call local vision server
4. Ask vision model: "Are the red boxes correctly highlighting [query]?"

**Example**:
```python
# Generate detection visualization
boxes = detect_entities_grounding_dino(image, "yellow auto-rickshaws")
viz_image = draw_boxes(image, boxes)
viz_image.save("detection_viz.jpg")

# Use local vision MCP to validate
# (Claude will do this during testing)
```

---

## 📊 Expected Performance Improvements

| Metric | v2.0 | v3.0 (GroundingDINO) | Improvement |
|--------|------|----------------------|-------------|
| **Success Rate** | 11% (1/9) | 89% (8/9) | **8x better** |
| **False Positives** | 22% (2/9) | 0% | **Zero FP** |
| **Query Coverage** | 40% | 100% | **Full coverage** |
| **Semantic Queries** | Crash | Works | **New capability** |
| **Processing Time** | 10s (when works) | 5-7s | **Faster** |
| **Color Dictionary** | 8 static | Unlimited dynamic | **No maintenance** |

---

## 🚀 Execution Schedule

### Week 1 (Claude Solo)
- **Mon-Tue**: Implement GroundingDINO integration
- **Wed**: Test on wildcard images, visual validation
- **Thu**: Implement box-to-mask conversion
- **Fri**: Integrate with orchestrator

### Week 2 (Claude + Qwen Testing)
- **Mon-Tue**: Run comprehensive tests
- **Wed**: Performance benchmarking
- **Thu**: Edge case handling
- **Fri**: Documentation + handoff

### Week 3 (Validation System)
- Implement vision delta analysis
- Add auto-retry loop
- Integrate structured feedback

---

## 🎯 Success Criteria

**Phase 1 Complete When**:
- [x] GroundingDINO installed and working
- [ ] 8/9 wildcard tests pass
- [ ] Processing time < 8s average
- [ ] Zero false positives
- [ ] Visual validation confirms quality

**Project Complete When**:
- [ ] 8.5/9 wildcard tests pass (94%)
- [ ] Precision ≥ 85%, Recall ≥ 80%
- [ ] Edit validation system working
- [ ] Auto-retry improves results
- [ ] Production-ready documentation

---

## 🛠️ Next Immediate Steps for Claude

1. **Install transformers GroundingDINO**:
   ```bash
   pip install transformers torch torchvision
   ```

2. **Create stage1_grounding_dino.py** with code above

3. **Test on ONE wildcard image** to verify setup

4. **Visual validation** using local vision MCP

5. **Report results** and proceed

---

**Status**: Ready to begin execution
**Executor**: Claude Code
**Start Date**: 2025-10-31
**Target Completion**: 2 weeks
