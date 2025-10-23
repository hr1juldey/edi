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