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

## See Docs

### Python Implementation Example
Implementation of the resize_image function:

```python
from PIL import Image
from typing import Union, Tuple
import os

def resize_image(image: Image.Image, max_size: int) -> Image.Image:
    """
    Resize an image to fit within max_size while preserving aspect ratio.
    
    Args:
        image: PIL Image object to be resized
        max_size: Maximum allowed dimension (width or height)
    
    Returns:
        PIL Image object with dimensions scaled to fit within max_size while preserving aspect ratio
    """
    # Validate inputs
    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL Image object, got {type(image)}")
    
    if not isinstance(max_size, int) or max_size <= 0:
        raise ValueError(f"max_size must be a positive integer, got {max_size}")
    
    # Get original dimensions
    original_width, original_height = image.size
    
    # Handle edge case where image is already smaller than max_size
    if original_width <= max_size and original_height <= max_size:
        # Return a copy to avoid modifying the original
        return image.copy()
    
    # Calculate scaling factor to ensure largest dimension doesn't exceed max_size
    scale_factor = min(max_size / original_width, max_size / original_height)
    
    # Calculate new dimensions preserving aspect ratio
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)
    
    # Resize the image using appropriate resampling algorithm
    # Use LANCZOS for high quality or BICUBIC for good balance of quality and speed
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    return resized_image

def resize_image_advanced(
    image: Image.Image, 
    max_size: int, 
    quality: str = 'high',
    maintain_alpha: bool = True
) -> Image.Image:
    """
    Advanced resize function with additional options for quality and alpha handling.
    
    Args:
        image: PIL Image object to be resized
        max_size: Maximum allowed dimension (width or height)
        quality: Quality level ('low', 'medium', 'high', 'highest')
        maintain_alpha: Whether to maintain alpha channel for transparent images
    
    Returns:
        PIL Image object with dimensions scaled to fit within max_size while preserving aspect ratio
    """
    # Validate inputs
    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL Image object, got {type(image)}")
    
    if not isinstance(max_size, int) or max_size <= 0:
        raise ValueError(f"max_size must be a positive integer, got {max_size}")
    
    if quality not in ['low', 'medium', 'high', 'highest']:
        raise ValueError(f"Quality must be one of 'low', 'medium', 'high', 'highest', got {quality}")
    
    # Get original dimensions
    original_width, original_height = image.size
    
    # Handle edge case where image is already smaller than max_size
    if original_width <= max_size and original_height <= max_size:
        return image.copy()
    
    # Calculate scaling factor and new dimensions
    scale_factor = min(max_size / original_width, max_size / original_height)
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)
    
    # Select resampling algorithm based on quality setting
    resampling_map = {
        'low': Image.Resampling.NEAREST,
        'medium': Image.Resampling.BILINEAR,
        'high': Image.Resampling.BICUBIC,
        'highest': Image.Resampling.LANCZOS
    }
    resampling = resampling_map[quality]
    
    # Handle alpha channel if present and requested
    if maintain_alpha and image.mode in ('RGBA', 'LA', 'P'):
        # If the image has transparency, preserve it appropriately
        if image.mode == 'P':
            # Convert palette mode to RGBA to preserve transparency
            image = image.convert('RGBA')
        
        resized_image = image.resize((new_width, new_height), resampling)
    else:
        # For non-transparent images or when alpha maintenance is not needed
        resized_image = image.resize((new_width, new_height), resampling)
    
    return resized_image

def resize_image_by_percentage(image: Image.Image, percentage: float) -> Image.Image:
    """
    Resize an image by a percentage of its original size.
    
    Args:
        image: PIL Image object to be resized
        percentage: Percentage of original size (e.g., 50 for 50%)
    
    Returns:
        PIL Image object resized to the specified percentage
    """
    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL Image object, got {type(image)}")
    
    if not isinstance(percentage, (int, float)) or percentage <= 0:
        raise ValueError(f"Percentage must be a positive number, got {percentage}")
    
    original_width, original_height = image.size
    scale_factor = percentage / 100.0
    
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)
    
    # Use high-quality resampling for percentage-based scaling
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    return resized_image

def resize_image_with_memory_check(image: Image.Image, max_size: int, target_memory_mb: float = 50.0) -> Image.Image:
    """
    Resize an image with additional memory usage check.
    
    Args:
        image: PIL Image object to be resized
        max_size: Maximum allowed dimension (width or height)
        target_memory_mb: Target maximum memory usage in MB
    
    Returns:
        PIL Image object with dimensions scaled appropriately
    """
    # Calculate estimated memory usage of original image (in MB)
    original_width, original_height = image.size
    # Rough estimate: width * height * 4 bytes per pixel (RGBA)
    original_memory_mb = (original_width * original_height * 4) / (1024 * 1024)
    
    if original_memory_mb > target_memory_mb:
        # Calculate a more aggressive resize to meet memory constraints
        memory_scale_factor = (target_memory_mb / original_memory_mb) ** 0.5
        # Also apply max_size constraint
        max_scale_factor = min(max_size / original_width, max_size / original_height)
        final_scale_factor = min(memory_scale_factor, max_scale_factor)
        
        new_width = int(original_width * final_scale_factor)
        new_height = int(original_height * final_scale_factor)
        
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return resized_image
    
    else:
        # Use the standard resize function
        return resize_image(image, max_size)

# Example usage
if __name__ == "__main__":
    # Create a sample image for demonstration
    import numpy as np
    sample_array = (np.random.rand(3000, 2000, 3) * 255).astype(np.uint8)
    sample_image = Image.fromarray(sample_array)
    
    print(f"Original image size: {sample_image.size}")
    
    # Basic resize
    resized = resize_image(sample_image, max_size=1024)
    print(f"Resized image size: {resized.size}")
    
    # Advanced resize with quality options
    resized_high = resize_image_advanced(sample_image, max_size=1024, quality='highest')
    print(f"High-quality resized image size: {resized_high.size}")
    
    # Resize by percentage
    resized_pct = resize_image_by_percentage(sample_image, percentage=25)
    print(f"25% of original size: {resized_pct.size}")
    
    # Resize with memory check
    resized_mem = resize_image_with_memory_check(
        sample_image, max_size=2048, target_memory_mb=10.0
    )
    print(f"Memory-optimized resize: {resized_mem.size}")
```

### Alternative Implementation with Multiple Strategies
Different resizing approaches for various use cases:

```python
from PIL import Image, ImageOps
from typing import Tuple, Optional
import io

class ImageResizer:
    """
    A class containing different image resizing strategies for various use cases.
    """
    
    @staticmethod
    def resize_to_fit(image: Image.Image, max_width: int, max_height: int) -> Image.Image:
        """
        Resize image to fit within specified width and height, preserving aspect ratio.
        """
        original_width, original_height = image.size
        
        # Calculate the scale factor to fit within both dimensions
        width_ratio = max_width / original_width
        height_ratio = max_height / original_height
        scale_factor = min(width_ratio, height_ratio)
        
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)
        
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    @staticmethod
    def resize_to_fill(image: Image.Image, target_width: int, target_height: int) -> Image.Image:
        """
        Resize image to fill specified dimensions, cropping if necessary to maintain aspect ratio.
        """
        original_width, original_height = image.size
        
        # Calculate the scale factor to fill the space
        width_ratio = target_width / original_width
        height_ratio = target_height / original_height
        scale_factor = max(width_ratio, height_ratio)
        
        # Calculate new size
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)
        
        # Resize first
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Then crop to exact dimensions
        return ImageOps.fit(resized, (target_width, target_height), Image.Resampling.LANCZOS)
    
    @staticmethod
    def resize_contain(image: Image.Image, max_width: int, max_height: int, 
                      background_color: Tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
        """
        Resize image to fit within dimensions with letterboxing if necessary.
        """
        # First resize to fit
        resized = ImageResizer.resize_to_fit(image, max_width, max_height)
        
        # Create a new image with the target size and background color
        new_image = Image.new(image.mode, (max_width, max_height), background_color)
        
        # Calculate position to center the resized image
        x = (max_width - resized.width) // 2
        y = (max_height - resized.height) // 2
        
        # Paste the resized image onto the new image
        if resized.mode == 'RGBA':
            new_image.paste(resized, (x, y), resized)
        else:
            new_image.paste(resized, (x, y))
        
        return new_image
    
    @staticmethod
    def resize_smart_crop(image: Image.Image, target_width: int, target_height: int) -> Image.Image:
        """
        Resize image with smart cropping using the center of the image.
        This approach maintains important content in the center of the image.
        """
        original_width, original_height = image.size
        
        # Calculate aspect ratios
        target_ratio = target_width / target_height
        original_ratio = original_width / original_height
        
        if original_ratio > target_ratio:
            # Image is wider than target, crop width
            new_height = original_height
            new_width = int(original_height * target_ratio)
            x_offset = (original_width - new_width) // 2
            y_offset = 0
        else:
            # Image is taller than target, crop height
            new_width = original_width
            new_height = int(original_width / target_ratio)
            x_offset = 0
            y_offset = (original_height - new_height) // 2
        
        # Crop the image
        cropped = image.crop((x_offset, y_offset, x_offset + new_width, y_offset + new_height))
        
        # Resize to target dimensions
        return cropped.resize((target_width, target_height), Image.Resampling.LANCZOS)
    
    @staticmethod
    def resize_progressive(image: Image.Image, max_size: int, step_size: int = 100) -> Image.Image:
        """
        Resize image in steps to avoid memory issues with very large images.
        """
        original_width, original_height = image.size
        
        if max(original_width, original_height) <= max_size:
            return image.copy()
        
        # Calculate the target size
        scale_factor = min(max_size / original_width, max_size / original_height)
        target_width = int(original_width * scale_factor)
        target_height = int(original_height * scale_factor)
        
        # Start with the original image
        current_image = image
        
        # Calculate intermediate sizes for progressive resizing
        intermediate_width = original_width
        intermediate_height = original_height
        
        # Resize in steps to avoid memory issues
        while max(intermediate_width, intermediate_height) > max_size * 2:
            # Calculate next step
            step_scale = max_size * 2 / max(intermediate_width, intermediate_height)
            intermediate_width = int(intermediate_width * step_scale)
            intermediate_height = int(intermediate_height * step_scale)
            
            current_image = current_image.resize(
                (intermediate_width, intermediate_height), 
                Image.Resampling.LANCZOS
            )
        
        # Final resize to target size
        return current_image.resize((target_width, target_height), Image.Resampling.LANCZOS)

# Example usage of different strategies
if __name__ == "__main__":
    # Create a sample image
    import numpy as np
    sample_array = (np.random.rand(1600, 1200, 3) * 255).astype(np.uint8)
    sample_image = Image.fromarray(sample_array)
    
    # Test different resizing strategies
    resizer = ImageResizer()
    
    # Resize to fit
    fit_resized = resizer.resize_to_fit(sample_image, 800, 600)
    print(f"Fit to 800x600: {fit_resized.size}")
    
    # Resize to fill
    fill_resized = resizer.resize_to_fill(sample_image, 800, 600)
    print(f"Fill 800x600: {fill_resized.size}")
    
    # Resize with contain strategy
    contain_resized = resizer.resize_contain(sample_image, 800, 600)
    print(f"Contain in 800x600: {contain_resized.size}")
    
    # Smart crop
    crop_resized = resizer.resize_smart_crop(sample_image, 800, 600)
    print(f"Smart crop to 800x600: {crop_resized.size}")
    
    # Progressive resize
    progressive_resized = resizer.resize_progressive(sample_image, 400)
    print(f"Progressive resize to max 400: {progressive_resized.size}")
```