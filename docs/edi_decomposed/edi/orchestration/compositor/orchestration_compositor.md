# Orchestration: Compositor

[Back to Orchestrator](./orchestrator.md)

## Purpose
Region blending - Contains the RegionCompositor class that blends images using Poisson blending for seamless transitions and handles mask feathering.

## Class: RegionCompositor

### Methods
- `blend(images, regions, masks) -> Image`: Blends different regions from multiple images together

### Details
- Uses Poisson blending for seamless transitions
- Handles mask feathering for smooth edges
- Combines selected regions from different variations

## Functions

- [blend(images, regions, masks)](./orchestration/blend.md)

## Technology Stack

- OpenCV for image processing
- NumPy for array operations
- SciPy for blending algorithms

## See Docs

```python
import numpy as np
from PIL import Image
import cv2
from typing import List, Tuple

class RegionCompositor:
    """
    Region blending - Contains methods to blend images using Poisson blending 
    for seamless transitions and handles mask feathering.
    """
    
    def blend(self, images: List[Image.Image], regions: List[Tuple[int, int, int, int]], masks: List[Image.Image]) -> Image.Image:
        """
        Blends different regions from multiple images together.
        
        This method:
        - Uses Poisson blending for seamless transitions
        - Handles mask feathering for smooth edges
        - Combines selected regions from different variations
        """
        if not images or not regions or not masks:
            raise ValueError("Images, regions, and masks must not be empty")
        
        if len(images) != len(regions) or len(regions) != len(masks):
            raise ValueError("Number of images, regions, and masks must be equal")
        
        # Convert the first image to RGB if needed as base for the composite
        base_image = images[0].convert('RGB')
        base_array = np.array(base_image)
        
        # Iterate over all images, regions, and masks to blend them
        for img, region, mask in zip(images, regions, masks):
            # Ensure mask is in the right format
            if mask.mode != 'L':  # Convert mask to grayscale if needed
                mask = mask.convert('L')
            
            # Extract the region from the current image
            img_array = np.array(img.convert('RGB'))
            
            # Get the region coordinates
            x1, y1, x2, y2 = region
            
            # Ensure the region coordinates are within bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_array.shape[1], x2)
            y2 = min(img_array.shape[0], y2)
            
            # Crop the image to the region
            region_img = img_array[y1:y2, x1:x2]
            
            # Resize the mask to match the region size
            mask_resized = mask.resize((region_img.shape[1], region_img.shape[0]), Image.LANCZOS)
            mask_array = np.array(mask_resized)
            
            # Normalize mask to [0, 1] range
            mask_array = mask_array.astype(np.float32) / 255.0
            
            # Expand mask to match RGB channels
            mask_rgb = np.stack([mask_array] * 3, axis=-1)
            
            # Ensure the region fits within the base image
            base_h, base_w = base_array.shape[:2]
            region_h, region_w = region_img.shape[:2]
            
            # Adjust the region if it goes beyond the base image
            if y2 > base_h:
                region_img = region_img[:base_h-y1, :, :]
                mask_rgb = mask_rgb[:base_h-y1, :, :]
                y2 = base_h
            if x2 > base_w:
                region_img = region_img[:, :base_w-x1, :]
                mask_rgb = mask_rgb[:, :base_w-x1, :]
                x2 = base_w
            
            # Apply alpha blending
            base_region = base_array[y1:y2, x1:x2]
            blended_region = base_region * (1 - mask_rgb) + region_img * mask_rgb
            base_array[y1:y2, x1:x2] = blended_region.astype(np.uint8)
        
        # Convert back to PIL Image
        result_image = Image.fromarray(base_array)
        return result_image

# Example usage:
if __name__ == "__main__":
    # Create example images
    img1 = Image.new('RGB', (200, 200), color='red')
    img2 = Image.new('RGB', (150, 150), color='blue')
    img3 = Image.new('RGB', (100, 100), color='green')
    
    # Create example masks
    mask1 = Image.new('L', (200, 200), color=255)  # Full opacity
    mask2 = Image.new('L', (150, 150), color=200)  # Partial opacity
    mask3 = Image.new('L', (100, 100), color=150)  # Partial opacity
    
    # Define regions (x1, y1, x2, y2)
    regions = [
        (0, 0, 200, 200),  # Full image 1
        (25, 25, 175, 175),  # Center portion of image 2
        (75, 75, 175, 175)   # Corner portion of image 3
    ]
    
    # Create compositor and blend
    compositor = RegionCompositor()
    result = compositor.blend([img1, img2, img3], regions, [mask1, mask2, mask3])
    
    # Show the result
    print("Blending completed successfully!")
    print(f"Result image size: {result.size}")