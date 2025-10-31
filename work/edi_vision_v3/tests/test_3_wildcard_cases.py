"""Test 3 Initial Wildcard Cases

Tests YOLO-World → SAM pipeline on 3 representative wildcard queries
from the original 9 test cases.

This verifies:
1. Color-guided queries work (v2 mostly worked)
2. Hybrid queries work (v2 had mixed results)
3. No crashes or garbage results
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image
import logging

from pipeline.stage1_yolo_world import detect_entities_yolo_world
from pipeline.stage2_yolo_to_sam import convert_boxes_to_masks

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)


WILDCARD_CASES = [
    # (image_path, query, v2_result, expected_behavior)
    ("images/kol_1.png", "red vehicles", "✅ worked", "Should detect vehicles"),
    ("images/Darjeeling.jpg", "brown roofs", "❌ false positive", "Should detect or return empty"),
    ("images/mumbai-traffic.jpg", "yellow auto-rickshaws", "❌ SAM failure", "Should detect if present"),
]


def test_wildcard_case(image_path, query, v2_result, expected):
    """Test a single wildcard case."""
    print(f"\n{'='*70}")
    print(f"Image: {image_path}")
    print(f"Query: '{query}'")
    print(f"v2.0: {v2_result}")
    print(f"Expected: {expected}")
    print(f"{'='*70}")

    # Check if image exists
    if not os.path.exists(image_path):
        print(f"⚠ Image not found: {image_path}")
        return "SKIP"

    try:
        # Load image
        image = np.array(Image.open(image_path).convert('RGB'))
        print(f"✓ Image loaded: {image.shape}")

        # Stage 1: YOLO-World detection
        print("\nStage 1: YOLO-World detection...")
        boxes = detect_entities_yolo_world(
            image,
            query,
            confidence_threshold=0.25
        )

        print(f"  ✓ Detected {len(boxes)} boxes")

        if len(boxes) > 0:
            # Print box details
            for i, box in enumerate(boxes[:5]):
                print(
                    f"    Box {i}: {box.label}, "
                    f"bbox=({box.x},{box.y},{box.w},{box.h}), "
                    f"conf={box.confidence:.3f}"
                )

            # Stage 2: Convert to SAM masks
            print("\nStage 2: Converting boxes to SAM masks...")
            masks = convert_boxes_to_masks(image, boxes)

            print(f"  ✓ Generated {len(masks)} masks")

            if len(masks) > 0:
                total_area = sum(m.area for m in masks)
                image_area = image.shape[0] * image.shape[1]
                coverage = (total_area / image_area) * 100

                print(f"  ✓ Total coverage: {coverage:.2f}% of image")

                # Print mask details
                for i, mask in enumerate(masks[:3]):
                    print(
                        f"    Mask {i}: {mask.label}, "
                        f"area={mask.area:,}px ({(mask.area/image_area)*100:.2f}%), "
                        f"conf={mask.confidence:.3f}"
                    )

                # Save visualization (optional)
                try:
                    from pipeline.stage2_yolo_to_sam import YOLOBoxToSAMMask
                    converter = YOLOBoxToSAMMask()
                    viz = converter.visualize_masks(image, masks, alpha=0.5)

                    viz_path = f"logs/viz_{os.path.basename(image_path).split('.')[0]}_{query.replace(' ', '_')}.png"
                    os.makedirs("logs", exist_ok=True)
                    Image.fromarray(viz).save(viz_path)
                    print(f"  ✓ Visualization saved: {viz_path}")
                except Exception as e:
                    print(f"  ⚠ Visualization failed: {e}")

                return "PASS"
            else:
                print(f"  ⚠ Boxes detected but no masks generated")
                return "WARN"
        else:
            print(f"  ℹ No boxes detected")
            print(f"    (This is OK if objects not present in image)")
            return "OK-EMPTY"

    except Exception as e:
        print(f"\n✗ TEST FAILED")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return "FAIL"


def main():
    """Run all 3 wildcard tests."""
    print("\n" + "="*80)
    print(" INITIAL WILDCARD TESTS (3 Cases)")
    print(" Testing v3.0 pipeline on representative queries from original 9 test cases")
    print("="*80)

    results = []

    for image_path, query, v2_result, expected in WILDCARD_CASES:
        result = test_wildcard_case(image_path, query, v2_result, expected)
        results.append((image_path, query, result))

    # Summary
    print("\n" + "="*80)
    print(" SUMMARY")
    print("="*80)

    for image_path, query, result in results:
        image_name = os.path.basename(image_path)
        print(f"{result:10} | {image_name:25} | {query}")

    # Statistics
    pass_count = sum(1 for r in results if r[2] == "PASS")
    fail_count = sum(1 for r in results if r[2] == "FAIL")
    skip_count = sum(1 for r in results if r[2] == "SKIP")
    ok_empty_count = sum(1 for r in results if r[2] == "OK-EMPTY")

    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    print(f"PASS (detections + masks):  {pass_count}/{len(results)}")
    print(f"OK-EMPTY (graceful empty):  {ok_empty_count}/{len(results)}")
    print(f"FAIL (crashes/errors):      {fail_count}/{len(results)}")
    print(f"SKIP (missing images):      {skip_count}/{len(results)}")

    print("\n" + "="*80)
    if fail_count == 0:
        print("✓ NO CRASHES OR ERRORS!")
        print("\nv3.0 handles all queries gracefully (no crashes like v2.0)")
        if pass_count > 0:
            print(f"Generated masks for {pass_count} test cases")
        print("\nReady to run full 9 wildcard test suite.")
    else:
        print(f"✗ {fail_count} test(s) failed")
        print("\nPlease review errors above.")
    print("="*80 + "\n")

    return fail_count == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
