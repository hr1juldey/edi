"""
Script to analyze mask files and determine what regions they cover
"""
import cv2
import numpy as np
import os

def analyze_mask(mask_path, image_path=None):
    """
    Analyze a mask file to determine what regions it covers.
    
    Args:
        mask_path: Path to the mask file
        image_path: Optional path to the original image for visual reference
    """
    # Load the mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Error: Could not load mask {mask_path}")
        return
    
    print(f"\nAnalyzing mask: {mask_path}")
    print(f"Mask dimensions: {mask.shape}")
    
    # Calculate coverage percentage
    total_pixels = mask.size
    masked_pixels = np.sum(mask > 0)
    coverage = (masked_pixels / total_pixels) * 100
    print(f"Coverage: {coverage:.2f}% ({masked_pixels} out of {total_pixels} pixels)")
    
    # Find bounding box of the mask
    ys, xs = np.where(mask > 0)
    if len(ys) > 0 and len(xs) > 0:
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        
        # Calculate bounding box dimensions
        width = x2 - x1 + 1
        height = y2 - y1 + 1
        bbox_area = width * height
        
        print(f"Bounding box: ({x1}, {y1}) to ({x2}, {y2})")
        print(f"Bounding box area: {bbox_area} pixels")
        
        # Calculate ratio of actual mask to bounding box
        ratio = masked_pixels / bbox_area if bbox_area > 0 else 0
        print(f"Mask to bounding box ratio: {ratio:.2f}")
        
        # If an image path is provided, try to identify what's in the bounding box
        if image_path:
            image = cv2.imread(image_path)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Extract the region from the image
                region = image[y1:y2+1, x1:x2+1]
                
                # Convert to PIL for easier analysis
                from PIL import Image
                img_pil = Image.fromarray(region)
                
                # Simple color analysis
                colors = []
                for i in range(region.shape[0]):
                    for j in range(region.shape[1]):
                        pixel = region[i, j]
                        # Check if pixel is predominantly red, blue, or green
                        r, g, b = pixel[0], pixel[1], pixel[2]
                        if r > 150 and g < 100 and b < 100:  # Red dominant
                            colors.append('red')
                        elif r < 100 and g < 100 and b > 150:  # Blue dominant
                            colors.append('blue')
                        elif r < 100 and g > 150 and b < 100:  # Green dominant
                            colors.append('green')
                        elif r > 100 and g > 100 and b > 100:  # White/gray
                            colors.append('white')
                        else:
                            colors.append('other')
                
                # Count color occurrences
                color_counts = {}
                for color in colors:
                    color_counts[color] = color_counts.get(color, 0) + 1
                
                print("Color distribution in masked region:")
                for color, count in sorted(color_counts.items(), key=lambda x: x[1], reverse=True):
                    percentage = (count / len(colors)) * 100
                    print(f"  {color}: {percentage:.1f}% ({count} pixels)")
    else:
        print("No pixels selected in mask")

if __name__ == "__main__":
    # Define paths
    image_path = "test_image.jpeg"
    
    # List of mask files to analyze
    mask_files = [
        "adaptive_result_orange_to_blue.png",
        "advanced_result_orange_to_blue.png",
        "m_adaptive_result_orange_to_blue.png"
    ]
    
    for mask_file in mask_files:
        if os.path.exists(mask_file):
            analyze_mask(mask_file, image_path)
        else:
            print(f"\nMask file {mask_file} not found")