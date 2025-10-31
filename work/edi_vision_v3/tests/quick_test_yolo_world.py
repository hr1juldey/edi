"""Quick test to verify YOLO-World works with existing installation.

This test verifies:
1. YOLO-World is available in ultralytics
2. Model can be loaded
3. Custom classes (open-vocabulary) work
4. Inference runs successfully
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image
from ultralytics import YOLO


def test_yolo_world_available():
    """Test that YOLO-World is available in ultralytics."""
    print("\n" + "="*60)
    print("TEST 1: YOLO-World Model Availability")
    print("="*60)

    try:
        # Try to load YOLO-World model
        model = YOLO("yolov8s-world.pt")
        print("✓ YOLO-World model loaded successfully")
        print(f"  Model type: {type(model)}")
        return True
    except Exception as e:
        print(f"✗ Failed to load YOLO-World: {e}")
        return False


def test_custom_classes():
    """Test open-vocabulary detection with custom classes."""
    print("\n" + "="*60)
    print("TEST 2: Custom Classes (Open-Vocabulary)")
    print("="*60)

    try:
        model = YOLO("yolov8s-world.pt")

        # Set custom classes (any text prompts)
        custom_classes = ["vehicles", "buildings", "trees", "sky"]
        model.set_classes(custom_classes)

        print(f"✓ Set custom classes: {custom_classes}")
        return True
    except Exception as e:
        print(f"✗ Failed to set custom classes: {e}")
        return False


def test_inference_on_image():
    """Test inference on a real image."""
    print("\n" + "="*60)
    print("TEST 3: Inference on Test Image")
    print("="*60)

    # Check if test image exists
    image_path = "images/test_image.jpeg"
    if not os.path.exists(image_path):
        print(f"⚠ Test image not found: {image_path}")
        print("  Trying alternative paths...")

        # Try other common test images
        alternatives = [
            "../edi_vision_v2/images/test_image.jpeg",
            "images/kol_1.png",
            "../edi_vision_v2/images/kol_1.png"
        ]

        for alt_path in alternatives:
            if os.path.exists(alt_path):
                image_path = alt_path
                print(f"✓ Found image at: {image_path}")
                break
        else:
            print("✗ No test images found")
            return False

    try:
        # Load model
        model = YOLO("yolov8s-world.pt")

        # Set test classes
        test_classes = ["building", "roof", "vehicle"]
        model.set_classes(test_classes)
        print(f"  Classes: {test_classes}")

        # Load image
        image = Image.open(image_path)
        print(f"  Image: {image_path}")
        print(f"  Size: {image.size}")

        # Run inference
        results = model.predict(source=image, conf=0.25, verbose=False)

        # Check results
        if len(results) > 0 and results[0].boxes is not None:
            num_boxes = len(results[0].boxes)
            print(f"✓ Inference successful: {num_boxes} objects detected")

            # Print first few detections
            for i, box in enumerate(results[0].boxes[:5]):
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = test_classes[cls_id] if cls_id < len(test_classes) else f"class_{cls_id}"
                print(f"    Box {i}: {label}, confidence={conf:.3f}")

            return True
        else:
            print("✓ Inference successful: 0 objects detected (image may not contain specified classes)")
            return True

    except Exception as e:
        print(f"✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_different_queries():
    """Test various query types that failed in v2.0."""
    print("\n" + "="*60)
    print("TEST 4: Various Query Types")
    print("="*60)

    # Find a test image
    image_path = None
    for path in ["images/kol_1.png", "../edi_vision_v2/images/kol_1.png"]:
        if os.path.exists(path):
            image_path = path
            break

    if not image_path:
        print("⚠ No test image found, skipping")
        return True

    try:
        model = YOLO("yolov8s-world.pt")
        image = Image.open(image_path)

        # Test queries that failed in v2.0
        test_queries = [
            "red vehicles",           # Color-guided (v2: worked)
            "vehicles",              # Semantic-only (v2: crashed)
            "yellow auto-rickshaws", # Hybrid (v2: crashed if yellow not in dict)
            "buildings",             # Semantic-only (v2: crashed)
            "sky",                   # Semantic-only (v2: crashed)
        ]

        print(f"  Testing on: {image_path}")
        print(f"  Image size: {image.size}")
        print()

        for query in test_queries:
            model.set_classes([query])
            results = model.predict(source=image, conf=0.25, verbose=False)

            num_detected = len(results[0].boxes) if len(results) > 0 and results[0].boxes is not None else 0

            print(f"  '{query}': {num_detected} objects detected")

        print("\n✓ All query types completed without crashing")
        print("  (v2.0 would have crashed on semantic-only queries)")
        return True

    except Exception as e:
        print(f"✗ Query testing failed: {e}")
        return False


def main():
    """Run all quick tests."""
    print("\n" + "="*70)
    print(" YOLO-WORLD QUICK VERIFICATION TEST")
    print(" Verifying open-vocabulary detection with existing packages only")
    print("="*70)

    results = []

    # Run tests
    results.append(("Model availability", test_yolo_world_available()))
    results.append(("Custom classes", test_custom_classes()))
    results.append(("Inference", test_inference_on_image()))
    results.append(("Various queries", test_different_queries()))

    # Summary
    print("\n" + "="*70)
    print(" TEST SUMMARY")
    print("="*70)

    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(r[1] for r in results)

    print("\n" + "="*70)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("\nYOLO-World is working correctly with existing packages!")
        print("Ready to implement full v3.0 pipeline.")
    else:
        print("✗ SOME TESTS FAILED")
        print("\nPlease check the errors above.")
    print("="*70 + "\n")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
