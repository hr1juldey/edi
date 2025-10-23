# resize_image()

[Back to Image Ops](../image_ops.md)

## Related User Story
"As a user, I want EDI to process my images efficiently without running out of memory." (from PRD - implied by performance requirements)

## Function Signature
`resize_image(image, max_size)`

## Parameters
- `image` - The input image to resize (PIL Image object)
- `max_size` - The maximum dimension (width or height) for the resized image

## Returns
- `image` - The resized image as a PIL Image object

## Step-by-step Logic
1. Get the original dimensions of the input image
2. Calculate the scaling factor to ensure the largest dimension doesn't exceed max_size
3. Calculate the new dimensions preserving the aspect ratio
4. Resize the image using appropriate resampling algorithm
5. Return the resized image while preserving the original aspect ratio
6. Handle edge cases where the image is already smaller than max_size

## Optimization Strategy
- Preserves aspect ratio during resizing
- Uses appropriate resampling for quality vs speed tradeoff
- Reduces memory usage for large images (>2048px)
- Enables processing of large images that would otherwise cause OOM errors

## Performance Considerations
- Efficient resampling algorithm selection
- Memory-conscious processing to avoid additional OOM errors
- Fast execution for real-time processing in the application
- Maintains image quality while reducing dimensions

## Input/Output Data Structures
### Input
- image: PIL Image object to be resized
- max_size: Integer representing maximum allowed dimension (width or height)

### Output
- PIL Image object with dimensions scaled to fit within max_size while preserving aspect ratio