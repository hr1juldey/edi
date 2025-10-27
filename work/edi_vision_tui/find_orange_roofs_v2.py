"""
Script to find orange/red roofed structures in the test image
"""
import cv2
import numpy as np
from PIL import Image
import os

def find_orange_roofs(image_path):
    """
    Find orange/red roofed structures in the image.
    
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
    
    # Create a mask for orange/red roofs
    roof_mask = np.zeros((height, width), dtype=np.uint8)
    
    # Define color ranges for orange/red roofs
    # We'll use a broader range to capture different shades of orange and red
    for i in range(height):
        for j in range(width):
            r, g, b = image[i, j]
            
            # Orange/red detection: high red, lower green and blue
            # This will catch orange, red, and reddish-brown roofs
            if r > 100 and (r > g + 20) and (r > b + 20):
                roof_mask[i, j] = 255
    
    # Calculate coverage percentage
    total_pixels = height * width
    roof_pixels = np.sum(roof_mask > 0)
    coverage = (roof_pixels / total_pixels) * 100
    
    print(f"Orange/Red roof detection results:")
    print(f"Coverage: {coverage:.2f}% ({roof_pixels} out of {total_pixels} pixels)")
    
    # Find bounding box of roof regions
    ys, xs = np.where(roof_mask > 0)
    if len(ys) > 0 and len(xs) > 0:
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        
        print(f"Bounding box: ({x1}, {y1}) to ({x2}, {y2})")
        
        # Create a visualization with orange/red roofs highlighted
        visualization = image.copy()
        visualization[roof_mask > 0] = [255, 165, 0]  # Orange color for visualization
        
        # Save the visualization
        vis_image = Image.fromarray(visualization.astype('uint8'))
        vis_image.save("orange_red_roofs_visualization.png")
        print(f"Saved visualization to orange_red_roofs_visualization.png")
    else:
        print("No orange/red roofs detected")
    
    # Analyze specific roof areas (look for clusters of pixels)
    print("\nAnalyzing potential roof structures:")
    
    # Use connected components to find separate roof structures
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(roof_mask, connectivity=8)
    
    roof_structures = []
    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area > 100:  # Filter out small noise
            # Get bounding box for this component
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            
            roof_structures.append({
                'area': area,
                'bbox': (x, y, x + w, y + h),
                'centroid': (int(centroids[i][0]), int(centroids[i][1]))
            })
    
    print(f"Found {len(roof_structures)} potential roof structures (area > 100 pixels)")
    
    # Sort by area (largest first)
    roof_structures.sort(key=lambda x: x['area'], reverse=True)
    
    # Show top 5 largest structures
    for i, structure in enumerate(roof_structures[:5]):
        print(f"  Structure {i+1}: Area={structure['area']} pixels, Bounding Box={structure['bbox']}")

if __name__ == "__main__":
    image_path = "test_image.jpeg"
    find_orange_roofs(image_path)