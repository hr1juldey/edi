"""
Script to find orange roofs in the test image
"""
import cv2
import numpy as np
from PIL import Image
import os

def find_orange_roofs(image_path):
    """
    Find orange roofs in the image.
    
    Args:
        image_path: Path to the image file
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    
    # Create a mask for orange roofs
    # Orange color range (in RGB) - adjusting for typical roof colors
    # Orange: high red, medium green, low blue
    orange_mask = np.zeros((height, width), dtype=np.uint8)
    
    # Define orange color thresholds
    # For orange roofs, we want high red, medium green, and low blue
    for i in range(height):
        for j in range(width):
            r, g, b = image[i, j]
            
            # Orange detection: high red, medium green, low blue
            # Adjust these thresholds based on what orange looks like in the image
            if r > 150 and g > 50 and b < 100 and (r > g + 30):  # Orange condition
                orange_mask[i, j] = 255
    
    # Calculate coverage percentage
    total_pixels = height * width
    orange_pixels = np.sum(orange_mask > 0)
    coverage = (orange_pixels / total_pixels) * 100
    
    print(f"Orange roof detection results:")
    print(f"Coverage: {coverage:.2f}% ({orange_pixels} out of {total_pixels} pixels)")
    
    # Find bounding box of orange regions
    ys, xs = np.where(orange_mask > 0)
    if len(ys) > 0 and len(xs) > 0:
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        
        print(f"Bounding box: ({x1}, {y1}) to ({x2}, {y2})")
        
        # Create a visualization with orange highlighted
        visualization = image.copy()
        visualization[orange_mask > 0] = [255, 165, 0]  # Orange color for visualization
        
        # Save the visualization
        vis_image = Image.fromarray(visualization.astype('uint8'))
        vis_image.save("orange_roofs_visualization.png")
        print(f"Saved visualization to orange_roofs_visualization.png")
    else:
        print("No orange roofs detected")
    
    # Analyze color distribution in the entire image
    print("\nColor distribution analysis:")
    colors = []
    for i in range(height):
        for j in range(width):
            r, g, b = image[i, j]
            if r > 150 and g > 50 and b < 100 and (r > g + 30):
                colors.append('orange')
            elif r > 150 and g < 100 and b < 100:
                colors.append('red')
            elif r < 100 and g < 100 and b > 150:
                colors.append('blue')
            elif r > 100 and g > 100 and b > 100:
                colors.append('white')
            else:
                colors.append('other')
    
    # Count color occurrences
    color_counts = {}
    for color in colors:
        color_counts[color] = color_counts.get(color, 0) + 1
    
    print("Overall color distribution:")
    for color, count in sorted(color_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(colors)) * 100
        print(f"  {color}: {percentage:.1f}% ({count} pixels)")

if __name__ == "__main__":
    image_path = "test_image.jpeg"
    find_orange_roofs(image_path)