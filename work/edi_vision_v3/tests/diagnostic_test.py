"""Diagnostic Test - Check if YOLO-World is detecting anything

This test uses very broad queries and low confidence threshold
to see if YOLO-World is working at all.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image
from pipeline.stage1_yolo_world import detect_entities_yolo_world

# Test with very broad queries
images_to_test = [
    "images/kol_1.png",
    "images/Darjeeling.jpg",
    "images/mumbai-traffic.jpg"
]

broad_queries = [
    "object",
    "thing",
    "vehicle",
    "building",
    "roof",
    "car",
    "house"
]

print("="*70)
print("DIAGNOSTIC TEST: Testing YOLO-World with broad queries")
print("="*70)

for image_path in images_to_test:
    if not os.path.exists(image_path):
        print(f"\n✗ Skipping {image_path} (not found)")
        continue

    print(f"\n{'='*70}")
    print(f"Image: {image_path}")
    print(f"{'='*70}")

    image = np.array(Image.open(image_path).convert('RGB'))
    print(f"Size: {image.shape}")

    for query in broad_queries:
        boxes = detect_entities_yolo_world(
            image,
            query,
            confidence_threshold=0.15  # Lower threshold
        )

        if len(boxes) > 0:
            print(f"  ✓ '{query}': {len(boxes)} detections")
            for i, box in enumerate(boxes[:2]):
                print(f"      Box {i}: conf={box.confidence:.3f}, bbox=({box.x},{box.y},{box.w}x{box.h})")
        else:
            print(f"  - '{query}': 0 detections")

print(f"\n{'='*70}")
print("END DIAGNOSTIC")
print(f"{'='*70}")
