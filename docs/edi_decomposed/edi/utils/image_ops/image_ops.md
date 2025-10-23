# Utils: Image Ops

[Back to Index](./index.md)

## Purpose
Image manipulation utilities - Contains functions for resizing images, validating image files, computing image hashes, etc.

## Functions
- `resize_image(image, max_size)`: Resizes images to a maximum size
- `validate_image(path) -> bool`: Validates if a file is a proper image
- `compute_image_hash(path) -> str`: Computes a hash for an image file

### Details
- General purpose image operation utilities
- Used across multiple subsystems
- Provides consistent image handling

## Technology Stack

- Pillow for image processing
- Hash algorithms for image identification

## See Docs

```python
from PIL import Image
import hashlib
import os
from typing import Tuple, Union
import io

def resize_image(image: Union[str, Image.Image], max_size: Tuple[int, int]) -> Image.Image:
    """
    Resizes images to a maximum size while maintaining aspect ratio.
    
    This function:
    - Accepts either a file path or PIL Image object as input
    - Resizes the image to fit within the specified maximum dimensions
    - Maintains the original aspect ratio of the image
    - Uses high-quality resampling for best results
    """
    # Load image if a file path is provided
    if isinstance(image, str):
        img = Image.open(image)
    elif isinstance(image, Image.Image):
        img = image
    else:
        raise TypeError("Input must be a file path (str) or PIL Image object")
    
    # Get original dimensions
    original_width, original_height = img.size
    
    # Calculate the scaling factor to maintain aspect ratio
    width_ratio = max_size[0] / original_width
    height_ratio = max_size[1] / original_height
    scale_factor = min(width_ratio, height_ratio)
    
    # Calculate new dimensions
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)
    
    # Resize the image using high-quality resampling (Lanczos)
    resized_img = img.resize((new_width, new_height), Image.LANCZOS)
    
    return resized_img

def validate_image(path: str) -> bool:
    """
    Validates if a file is a proper image.
    
    This function:
    - Checks if the file exists
    - Attempts to open and verify the image file
    - Validates that the file has a proper image format
    """
    if not os.path.exists(path):
        return False
    
    try:
        with Image.open(path) as img:
            # Try to load the image to check if it's valid
            img.load()
            # Verify the image has valid dimensions
            if img.width <= 0 or img.height <= 0:
                return False
            return True
    except Exception:
        # If any exception occurs during image loading/validation, it's not a valid image
        return False

def compute_image_hash(path: str) -> str:
    """
    Computes a hash for an image file.
    
    This function:
    - Reads the image file contents
    - Computes a SHA-256 hash of the image data
    - Returns the hash as a hexadecimal string
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file does not exist: {path}")
    
    # Calculate hash of image contents
    sha256_hash = hashlib.sha256()
    
    with open(path, "rb") as f:
        # Read the file in chunks to handle large files efficiently
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    
    return sha256_hash.hexdigest()

# Example usage and tests
if __name__ == "__main__":
    # Since we don't have actual image files to work with in this context,
    # we'll demonstrate the functionality with a created image
    
    # Create a sample image for testing
    sample_img = Image.new('RGB', (100, 100), color='red')
    sample_img.save('sample_test_image.jpg')
    
    print("Testing image operations:")
    
    # Test resize_image
    try:
        resized = resize_image('sample_test_image.jpg', (50, 50))
        print(f"Original size: (100, 100)")
        print(f"Resized size: {resized.size}")
        print(f"Resize successful: {resized.size[0] <= 50 and resized.size[1] <= 50}")
    except Exception as e:
        print(f"Resize test error: {e}")
    
    # Test validate_image
    is_valid = validate_image('sample_test_image.jpg')
    print(f"Image validation result: {is_valid}")
    
    # Test compute_image_hash
    try:
        img_hash = compute_image_hash('sample_test_image.jpg')
        print(f"Image hash (first 16 chars): {img_hash[:16]}...")
    except Exception as e:
        print(f"Hash computation error: {e}")
    
    # Clean up test file
    os.remove('sample_test_image.jpg')
    
    # Test with invalid file
    print(f"Validation of non-existent file: {validate_image('nonexistent.jpg')}")
    print(f"Hash of non-existent file: ", end="")
    try:
        compute_image_hash('nonexistent.jpg')
    except FileNotFoundError:
        print("FileNotFoundError raised as expected")