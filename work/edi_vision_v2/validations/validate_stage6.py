#!/usr/bin/env python3
"""Visual validation for Stage 6: VLM Validation"""

import cv2
import json
from pipeline.stage2_color_filter import color_prefilter
from pipeline.stage3_sam_segmentation import segment_regions
from pipeline.stage4_clip_filter import clip_filter_masks
from pipeline.stage5_mask_organization import organize_masks
from pipeline.stage6_vlm_validation import validate_with_vlm, create_validation_overlay

# Load test image
test_img = cv2.imread("test_image.jpeg")
test_img_rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

print("Running full pipeline Stages 2-6...")

# Run pipeline Stages 2-5
color_mask = color_prefilter(test_img_rgb, "blue")
sam_masks = segment_regions(test_img_rgb, color_mask, min_area=500)
filtered_masks = clip_filter_masks(test_img_rgb, sam_masks, "tin roof", similarity_threshold=0.22)
entity_masks = organize_masks(test_img_rgb, filtered_masks)

print(f"Stage 5 complete: {len(entity_masks)} entity masks organized")

# Stage 6: VLM Validation
# NOTE: validate_with_vlm expects List[np.ndarray], but Stage 5 returns List[EntityMask]
# We need to extract the mask arrays from EntityMask objects

print("\nStage 6: VLM Validation...")
user_intent = "change blue tin roofs to green"

# Extract mask arrays from EntityMask objects
mask_arrays = [entity.mask for entity in entity_masks]

try:
    validation_result = validate_with_vlm(test_img_rgb, mask_arrays, user_intent)

    print(f"\n{'='*60}")
    print("STAGE 6: VLM VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"User Intent: {user_intent}")
    print(f"Detected Count: {len(mask_arrays)}")
    print(f"Covers All Targets: {validation_result.covers_all_targets}")
    print(f"Overall Confidence: {validation_result.confidence:.2f}")
    print(f"Target Coverage: {validation_result.target_coverage:.2f}")
    print(f"False Positive Ratio: {validation_result.false_positive_ratio:.2f}")
    print("\nFeedback:")
    print(f"  {validation_result.feedback}")
    print("\nMissing Targets:")
    print(f"  {validation_result.missing_targets}")

    if validation_result.suggestions:
        print("\nSuggestions:")
        for suggestion in validation_result.suggestions:
            print(f"  - {suggestion}")

    # Save validation overlay
    validation_overlay = create_validation_overlay(test_img_rgb, mask_arrays)
    cv2.imwrite("logs/stage6_validation_overlay.png",
                cv2.cvtColor(validation_overlay, cv2.COLOR_RGB2BGR))
    print("\nSaved validation overlay: logs/stage6_validation_overlay.png")

    # Save validation report as JSON
    report = {
        'user_intent': user_intent,
        'detected_count': len(mask_arrays),
        'covers_all_targets': validation_result.covers_all_targets,
        'confidence': validation_result.confidence,
        'feedback': validation_result.feedback,
        'target_coverage': validation_result.target_coverage,
        'false_positive_ratio': validation_result.false_positive_ratio,
        'missing_targets': validation_result.missing_targets,
        'suggestions': validation_result.suggestions
    }

    with open("logs/stage6_validation_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    print("Saved validation report: logs/stage6_validation_report.json")

    # Determine recommendation
    if validation_result.covers_all_targets and validation_result.confidence >= 0.8:
        recommendation = "ACCEPT"
    elif validation_result.confidence >= 0.5:
        recommendation = "REVIEW"
    else:
        recommendation = "RETRY"

    print(f"\n{'='*60}")
    print(f"RECOMMENDATION: {recommendation}")
    print(f"{'='*60}")

except Exception as e:
    print(f"\n⚠️ VLM Validation failed: {e}")
    print("This is expected if Ollama is not running or qwen2.5-vl:7b is not installed")
    print("\nTo fix:")
    print("  1. Start Ollama: ollama serve")
    print("  2. Pull model: ollama pull qwen2.5-vl:7b")
    print("\nStage 6 is OPTIONAL - pipeline works without it!")
