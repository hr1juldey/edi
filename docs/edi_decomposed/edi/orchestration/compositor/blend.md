# RegionCompositor.blend()

[Back to Compositor](../orchestration_compositor.md)

## Related User Story
"As a user, I want to see multiple variations and pick the best parts from each." (from PRD)

## Function Signature
`blend(images, regions, masks) -> Image`

## Parameters
- `images` - A list of images from which to extract regions
- `regions` - A list of region specifications indicating which parts to extract
- `masks` - A list of masks defining the boundaries of regions to blend

## Returns
- `Image` - A composite image with selected regions blended together

## Step-by-step Logic
1. Take images, regions, and masks as input
2. Extract the specified regions from each image using the provided masks
3. Apply Poisson blending to create seamless transitions between regions
4. Handle mask feathering to ensure smooth edges
5. Blend the selected regions maintaining color and lighting consistency
6. Return the final composite image

## Blending Technology
- Uses Poisson blending algorithm for seamless transitions
- Preserves gradients at region boundaries
- Maintains overall image quality
- Handles color correction between different source regions

## Feathering Process
- Applies smooth transitions at mask boundaries
- Reduces visible seams in the composite
- Maintains natural appearance of the result
- Adjusts alpha values for smooth blending

## Input/Output Data Structures
### Image Object
The function accepts and returns Pillow Image objects
- Input images: Original source images for region extraction
- Output image: Composite with selected regions blended together

### Mask Object
Mask objects define the boundaries for region extraction:
- Binary masks indicating which pixels to include
- Alpha channels for feathered edges

## See Docs

```python
import numpy as np
from PIL import Image, ImageDraw
import cv2
from typing import List, Tuple, Union

class RegionCompositor:
    def __init__(self):
        pass

    def blend(self, images: List[Image.Image], regions: List[Tuple[int, int, int, int]], masks: List[Image.Image]) -> Image.Image:
        """
        Blends selected regions from multiple images into a single composite image using Poisson blending.
        
        This method:
        1. Takes images, regions, and masks as input
        2. Extracts the specified regions from each image using the provided masks
        3. Applies Poisson blending to create seamless transitions between regions
        4. Handles mask feathering to ensure smooth edges
        5. Blends the selected regions maintaining color and lighting consistency
        6. Returns the final composite image
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
            
            # Apply Poisson blending using OpenCV
            center = (x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2)
            
            # Convert region image and mask to OpenCV format
            region_cv = cv2.cvtColor(region_img, cv2.COLOR_RGB2BGR)
            mask_cv = (mask_array * 255).astype(np.uint8)
            
            # Ensure the center point is within bounds
            center = (min(center[0], base_w - 1), min(center[1], base_h - 1))
            
            # Perform seamless cloning (Poisson blending)
            try:
                base_bgr = cv2.cvtColor(base_array, cv2.COLOR_RGB2BGR)
                blended_bgr = cv2.seamlessClone(region_cv, base_bgr, mask_cv, center, cv2.NORMAL_CLONE)
                base_array = cv2.cvtColor(blended_bgr, cv2.COLOR_BGR2RGB)
            except cv2.error:
                # If seamless cloning fails, use alpha blending as fallback
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
    
    # Draw some shapes on masks to indicate which regions to blend
    draw1 = ImageDraw.Draw(mask1)
    draw1.ellipse([(50, 50), (150, 150)], fill=255)  # Circle in the middle
    
    draw2 = ImageDraw.Draw(mask2)
    draw2.rectangle([(25, 25), (125, 125)], fill=200)  # Square in the middle
    
    draw3 = ImageDraw.Draw(mask3)
    draw3.polygon([(50, 0), (100, 50), (50, 100), (0, 50)], fill=150)  # Diamond shape
    
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
    result.show()
    print("Blending completed successfully!")
```