"""
Multi-stage pipeline for EDI vision subsystem:
1. CLIP-based entity detection
2. YOLO-based refinement
3. SAM-based masking
4. Validation
5. Feedback processing
"""
from .stage1_clip.clip_entity_detector import get_entity_bounding_boxes
from .stage2_yolo.yolo_refiner import YOLORefiner
from .stage3_sam.sam_integration import SAMIntegration
from .stage4_validation.validation_system import validate_edit
from .stage5_feedback.feedback_processor import EditLogGenerator, DSPyRefiner

__all__ = [
    'get_entity_bounding_boxes',
    'YOLORefiner',
    'SAMIntegration',
    'validate_edit',
    'EditLogGenerator',
    'DSPyRefiner'
]