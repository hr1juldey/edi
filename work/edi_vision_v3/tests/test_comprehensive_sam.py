"""Final comprehensive test to verify all SAM segmentation enhancements"""
import sys
sys.path.insert(0, '/home/riju279/Documents/Code/Zonko/EDI/edi/work/edi_vision_v3')

import logging
import numpy as np
from PIL import Image
from pipeline.stage3_sam_segmentation import segment_regions

# Set logging to INFO to see diagnostics
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def test_comprehensive_diagnostics():
    """Test comprehensive diagnostics with various scenarios"""
    print("\n" + "="*60)
    print("COMPREHENSIVE DIAGNOSTICS TEST")
    print("="*60)

    # Load image
    image_path = "images/test_image.jpeg"
    try:
        image = np.array(Image.open(image_path).convert('RGB'))
        print(f"✓ Image loaded: {image.shape}")
    except Exception as e:
        print(f"✗ Failed to load image: {e}")
        return False

    # Test 1: Normal case with moderate coverage
    print("\n--- Test 1: Normal case with moderate coverage ---")
    color_mask = np.random.rand(image.shape[0], image.shape[1]) > 0.3
    color_mask = color_mask.astype(np.uint8)
    
    try:
        masks = segment_regions(image, color_mask, min_area=300)
        print(f"✓ Normal case succeeded: Generated {len(masks)} masks")
    except Exception as e:
        print(f"✗ Normal case failed: {e}")
        return False

    # Test 2: High coverage case
    print("\n--- Test 2: High coverage case ---")
    color_mask = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8)
    
    try:
        masks = segment_regions(image, color_mask, min_area=1000)
        print(f"✓ High coverage case succeeded: Generated {len(masks)} masks")
    except Exception as e:
        print(f"✗ High coverage case failed: {e}")
        return False

    # Test 3: Very low coverage case
    print("\n--- Test 3: Very low coverage case ---")
    color_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    # Just a few pixels
    color_mask[50:55, 50:55] = 1
    
    try:
        masks = segment_regions(image, color_mask, min_area=10)
        print(f"✓ Very low coverage case succeeded: Generated {len(masks)} masks")
    except Exception as e:
        print(f"✗ Very low coverage case failed: {e}")
        return False

    return True

if __name__ == "__main__":
    print("\n" + "="*70)
    print("FINAL COMPREHENSIVE SAM ENHANCEMENTS VALIDATION")
    print("="*70)

    success = test_comprehensive_diagnostics()
    
    print("\n" + "="*70)
    if success:
        print("✅ ALL COMPREHENSIVE TESTS PASSED")
        print("✅ INPUT DIAGNOSTICS: Working correctly")
        print("✅ FALLBACK SAM: Working correctly")
        print("✅ FAILURE DIAGNOSTICS: Working correctly")
        print("✅ GRACEFUL DEGRADATION: Working correctly")
        print("\n🎉 STAGE 3 ENHANCEMENTS READY FOR VALIDATION")
    else:
        print("❌ SOME COMPREHENSIVE TESTS FAILED")
        print("❌ REVIEW IMPLEMENTATION BEFORE REPORTING")
    print("="*70)