"""Additional test to verify SAM fallback mechanism"""
import sys
sys.path.insert(0, '/home/riju279/Documents/Code/Zonko/EDI/edi/work/edi_vision_v3')

import logging
import numpy as np
from PIL import Image
from pipeline.stage3_sam_segmentation import segment_regions

# Set logging to INFO to see diagnostics
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def test_fallback_mechanism():
    """Test the fallback mechanism by creating a challenging case"""
    print("\n" + "="*60)
    print("TEST: SAM Fallback Mechanism")
    print("="*60)

    # Load image
    image_path = "images/test_image.jpeg"
    try:
        image = np.array(Image.open(image_path).convert('RGB'))
        print(f"✓ Image loaded: {image.shape}")
    except Exception as e:
        print(f"✗ Failed to load image: {e}")
        return False

    # Create a very sparse color mask (very low coverage)
    # This might challenge SAM but should still work with fallback
    color_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    # Add just a few pixels
    color_mask[100:105, 100:105] = 1
    print(f"✓ Sparse color mask created: {color_mask.mean()*100:.4f}% coverage")

    # Run SAM segmentation with normal parameters
    try:
        masks = segment_regions(image, color_mask, min_area=500)
        print(f"\n✓ SAM completed: Generated {len(masks)} masks")
        return True
    except Exception as e:
        print(f"\n✗ SAM FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n" + "="*60)
    print("ADDITIONAL SAM FALLBACK TEST")
    print("="*60)

    success = test_fallback_mechanism()
    
    if success:
        print("\n✓ FALLBACK MECHANISM TEST PASSED")
    else:
        print("\n✗ FALLBACK MECHANISM TEST FAILED")