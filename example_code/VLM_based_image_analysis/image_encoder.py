"""
Image encoding utilities for preparing images for VLM processing.
"""

import base64
import io
from typing import List
from PIL import Image


def encode_image_to_base64(image_path: str) -> str:
    """
    Encode a single image file to base64 string.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        str: Base64 encoded string of the image
    """
    try:
        with Image.open(image_path) as img:
            buffer = io.BytesIO()
            img.save(buffer, format=img.format or "PNG")
            byte_data = buffer.getvalue()
            base64_str = base64.b64encode(byte_data).decode("utf-8").replace("\n", "")
            return base64_str
    except Exception as e:
        raise ValueError(f"Failed to encode image '{image_path}': {e}")


def encode_images_batch(image_paths: List[str]) -> List[str]:
    """
    Batch encode multiple image files to base64 strings.
    
    Args:
        image_paths (List[str]): List of paths to image files
        
    Returns:
        List[str]: List of base64 encoded strings
    """
    encoded_images = []
    for path in image_paths:
        try:
            encoded_images.append(encode_image_to_base64(path))
        except Exception as e:
            print(f"[WARN] Skipping '{path}' - {e}")
            encoded_images.append(None)  # Keep indexing consistent
    return encoded_images


def validate_image_path(image_path: str) -> bool:
    """
    Validate that an image path points to a valid image file.
    
    Args:
        image_path (str): Path to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        with Image.open(image_path) as img:
            img.verify()  # Verify it's a valid image
        return True
    except Exception:
        return False


def get_image_dimensions(image_path: str) -> tuple:
    """
    Get the dimensions of an image.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        tuple: (width, height) of the image
    """
    try:
        with Image.open(image_path) as img:
            return img.size
    except Exception as e:
        raise ValueError(f"Failed to get dimensions for '{image_path}': {e}")


def resize_image_if_needed(image_path: str, max_dimension: int = 2048) -> str:
    """
    Resize an image if it exceeds maximum dimensions, maintaining aspect ratio.
    
    Args:
        image_path (str): Path to the image file
        max_dimension (int): Maximum dimension (width or height)
        
    Returns:
        str: Path to resized image (may be same as input if no resize needed)
    """
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            
            # Check if resize is needed
            if width <= max_dimension and height <= max_dimension:
                return image_path  # No resize needed
            
            # Calculate new dimensions maintaining aspect ratio
            if width > height:
                new_width = max_dimension
                new_height = int(height * (max_dimension / width))
            else:
                new_height = max_dimension
                new_width = int(width * (max_dimension / height))
            
            # Resize image
            resized_img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # Save to temporary file
            import tempfile
            
            temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            resized_img.save(temp_file.name, "JPEG", quality=85)
            temp_file.close()
            
            return temp_file.name
    except Exception as e:
        raise ValueError(f"Failed to resize image '{image_path}': {e}")