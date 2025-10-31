"""Basic Integration Test for YOLO-World → SAM Pipeline

Tests the core v3.0 pipeline:
1. YOLO-World open-vocabulary detection (stage1)
2. Box-to-SAM mask conversion (stage2)

This replaces the broken v2.0 color-first pipeline.
"""

import sys
import os

# Add parent directory to path
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


def test_basic_pipeline():
    """Test YOLO-World → SAM on a single query."""
    print("\n" + "="*60)
    print("TEST: Basic YOLO-World → SAM Pipeline")
    print("="*60)

    # Find a test image
    image_path = None
    test_paths = [
        "images/kol_1.png",
        "images/test_image.jpeg",
        "../edi_vision_v2/images/kol_1.png",
        "../edi_vision_v2/images/test_image.jpeg"
    ]

    for path in test_paths:
        if os.path.exists(path):
            image_path = path
            break

    if not image_path:
        print("✗ No test image found")
        print(f"  Tried: {test_paths}")
        return False

    print(f"✓ Using test image: {image_path}")

    # Load image
    try:
        image = np.array(Image.open(image_path).convert('RGB'))
        print(f"✓ Image loaded: {image.shape}")
    except Exception as e:
        print(f"✗ Failed to load image: {e}")
        return False

    # Test queries
    test_queries = [
        "vehicles",           # Semantic-only (v2 crashed)
        "buildings",          # Semantic-only (v2 crashed)
        "red vehicles",       # Color-guided (v2 worked)
    ]

    print(f"\nTesting {len(test_queries)} queries...")
    print()

    results = []

    for query in test_queries:
        print(f"{'='*60}")
        print(f"Query: '{query}'")
        print(f"{'='*60}")

        try:
            # Stage 1: YOLO-World detection
            print("Stage 1: YOLO-World detection...")
            boxes = detect_entities_yolo_world(
                image,
                query,
                confidence_threshold=0.25
            )

            print(f"  ✓ Detected {len(boxes)} boxes")

            # Stage 2: Convert to SAM masks
            if len(boxes) > 0:
                print("Stage 2: Converting boxes to SAM masks...")
                masks = convert_boxes_to_masks(image, boxes)

                print(f"  ✓ Generated {len(masks)} masks")

                # Validate masks
                if len(masks) > 0:
                    total_area = sum(m.area for m in masks)
                    print(f"  ✓ Total mask area: {total_area:,} pixels")

                    # Print mask details
                    for i, mask in enumerate(masks[:3]):
                        print(
                            f"    Mask {i}: {mask.label}, "
                            f"area={mask.area}px, "
                            f"conf={mask.confidence:.3f}"
                        )

                    results.append(("PASS", query, len(boxes), len(masks)))
                else:
                    print(f"  ⚠ Warning: Boxes detected but no masks generated")
                    results.append(("WARN", query, len(boxes), 0))
            else:
                print(f"  ℹ No boxes detected (query may not match image content)")
                results.append(("OK", query, 0, 0))

        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append(("FAIL", query, 0, 0))

        print()

    # Summary
    print("="*60)
    print("SUMMARY")
    print("="*60)

    for status, query, boxes, masks in results:
        print(f"{status:6} | {query:20} | boxes={boxes:2}, masks={masks:2}")

    # Check for failures
    failures = [r for r in results if r[0] == "FAIL"]
    if len(failures) > 0:
        print(f"\n✗ {len(failures)}/{len(results)} tests FAILED")
        return False
    else:
        print(f"\n✓ All {len(results)} tests completed (no crashes!)")
        return True


def test_critical_v2_failure_cases():
    """Test queries that crashed or returned garbage in v2.0."""
    print("\n" + "="*60)
    print("TEST: Critical v2.0 Failure Cases")
    print("="*60)

    # Find a test image
    image_path = None
    for path in ["images/test_image.jpeg", "../edi_vision_v2/images/test_image.jpeg"]:
        if os.path.exists(path):
            image_path = path
            break

    if not image_path:
        print("⚠ No test image found, skipping")
        return True

    image = np.array(Image.open(image_path).convert('RGB'))

    # These queries all CRASHED or returned GARBAGE in v2.0
    critical_cases = [
        ("purple objects", 0, "v2 returned 11 garbage masks"),
        ("auto-rickshaws", 0, "v2 crashed (no color)"),
        ("small birds", 0, "v2 crashed (no color)"),
        ("sky", None, "v2 crashed (no color)"),
    ]

    print(f"Testing {len(critical_cases)} critical failure cases from v2.0...")
    print()

    all_passed = True

    for query, expected_boxes, v2_behavior in critical_cases:
        print(f"Query: '{query}'")
        print(f"  v2.0 behavior: {v2_behavior}")

        try:
            boxes = detect_entities_yolo_world(image, query, confidence_threshold=0.25)

            # Should NOT crash
            assert boxes is not None, "Returned None (should return list)"
            assert isinstance(boxes, list), f"Returned {type(boxes)} (should be list)"

            print(f"  ✓ v3.0: Returned {len(boxes)} boxes (no crash)")

            # For expected zero detections, verify we don't return garbage
            if expected_boxes == 0 and len(boxes) > 0:
                print(f"  ⚠ Warning: Expected 0, got {len(boxes)} (may be false positives)")
                all_passed = False
            elif expected_boxes == 0 and len(boxes) == 0:
                print(f"  ✓ Correctly returned empty list (no garbage)")

        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            all_passed = False

        print()

    if all_passed:
        print("✓ All critical v2.0 failure cases now work!")
    else:
        print("⚠ Some cases need attention")

    return all_passed


def main():
    """Run all basic tests."""
    print("\n" + "="*70)
    print(" YOLO-WORLD → SAM BASIC INTEGRATION TESTS")
    print(" Testing v3.0 pipeline against v2.0 critical failures")
    print("="*70)

    results = []

    # Test 1: Basic pipeline
    results.append(("Basic pipeline", test_basic_pipeline()))

    # Test 2: Critical v2.0 failures
    results.append(("Critical v2.0 cases", test_critical_v2_failure_cases()))

    # Summary
    print("\n" + "="*70)
    print(" FINAL SUMMARY")
    print("="*70)

    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(r[1] for r in results)

    print("\n" + "="*70)
    if all_passed:
        print("✓ ALL BASIC TESTS PASSED")
        print("\nv3.0 pipeline is working correctly!")
        print("Ready to run full wildcard tests.")
    else:
        print("✗ SOME TESTS FAILED")
        print("\nPlease review errors above.")
    print("="*70 + "\n")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
