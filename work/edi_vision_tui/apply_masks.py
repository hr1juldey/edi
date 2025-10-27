"""
Script to apply masks to the original image and create superimposed outputs
"""
import cv2
import numpy as np
from PIL import Image
import os
import glob

def apply_mask_to_image(image_path, mask_path, output_path):
    """
    Apply a mask to an image and save the superimposed result.
    
    Args:
        image_path: Path to the original image
        mask_path: Path to the mask image
        output_path: Path to save the superimposed image
    """
    # Load the original image
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # Load the mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # Resize mask to match image dimensions if needed
    if mask.shape[:2] != original_image.shape[:2]:
        mask = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Create a colored overlay (red for masked regions)
    overlay = original_image.copy()
    overlay[mask > 0] = [255, 0, 0]  # Set masked pixels to red
    
    # Blend the original and overlay (50% original, 50% overlay)
    superimposed = cv2.addWeighted(original_image, 0.7, overlay, 0.3, 0)
    
    # Save the result
    result_image = Image.fromarray(superimposed.astype('uint8'))
    result_image.save(output_path)
    print(f"Saved superimposed image to {output_path}")

def main():
    # Define paths
    image_path = "test_image.jpeg"
    mask_pattern = "*mask*.png"  # Find all mask files
    
    # Find all mask files
    mask_files = glob.glob(mask_pattern)
    
    if not mask_files:
        print("No mask files found!")
        return
    
    print(f"Found {len(mask_files)} mask files:")
    for mask_file in mask_files:
        print(f"  - {mask_file}")
    
    # Process each mask
    for i, mask_file in enumerate(mask_files):
        # Generate output filename
        base_name = os.path.splitext(mask_file)[0]
        output_path = f"superimposed_{base_name}.png"
        
        print(f"\nProcessing {mask_file}...")
        apply_mask_to_image(image_path, mask_file, output_path)
        print(f"Created superimposed image: {output_path}")

if __name__ == "__main__":
    main()