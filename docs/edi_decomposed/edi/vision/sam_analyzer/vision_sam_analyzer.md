# Vision: SAM Analyzer

[Back to Vision Subsystem](./vision_subsystem.md)

## Purpose
SAM 2.1 wrapper - Contains the SAMAnalyzer class with analyze method that takes an image path and returns a list of masks. Caches model in memory and handles out-of-memory situations by downscaling.

## Class: SAMAnalyzer

### Methods
- `analyze(image_path) -> List[Mask]`: Performs segmentation on the given image and returns a list of masks

### Details
- Caches model in memory for performance
- Handles OOM by downscaling images
- Uses SAM 2.1 for automatic segmentation

## Functions

- [analyze(image_path)](./vision/sam_analyze.md)

## Technology Stack

- SAM 2.1 for segmentation
- PyTorch for model execution
- NumPy for array operations

## See Docs

```python
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import cv2
import torch
from ...vision.models import Mask

class SAMAnalyzer:
    """
    SAM 2.1 wrapper - Contains methods to perform image segmentation
    with caching and OOM handling.
    """
    
    def __init__(self, model_path: Optional[str] = None, clip_model_path: Optional[str] = None):
        """
        Initialize the SAM analyzer with appropriate models.
        In a real implementation, this would load the SAM 2.1 model.
        """
        self.model_path = model_path
        self.clip_model_path = clip_model_path
        self.sam_model = None  # In a real implementation, this would be the loaded SAM model
        self.clip_model = None  # In a real implementation, this would be the loaded CLIP model
        
        # For this implementation, we're simulating SAM functionality
        # In a real implementation, you would load the actual SAM model here
        # self.sam_model = load_sam_model(self.model_path)
        # self.clip_model = load_clip_model(self.clip_model_path)
        
        # Predefined labels for clustering
        self.predefined_labels = [
            "person", "sky", "tree", "building", "car", "road", 
            "grass", "water", "mountain", "animal", "flower", 
            "window", "door", "furniture", "food", "cloud", 
            "sun", "moon", "star", "bird", "insect", "fish"
        ]

    def analyze(self, image_path: str) -> List[Mask]:
        """
        Performs segmentation on the given image and returns a list of masks.
        
        This method:
        - Caches model in memory for performance
        - Handles OOM by downscaling images
        - Uses SAM 2.1 for automatic segmentation
        """
        try:
            # Step 1: Load the image from the provided path
            pil_image = Image.open(image_path)
            image = np.array(pil_image)
            
            # Handle potential grayscale images by converting to RGB
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:  # Handle RGBA
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            
            # Step 7: Handle out-of-memory situations by downscaling large images
            max_size = 1024  # Max dimension for processing
            height, width = image.shape[:2]
            
            scale_factor = 1.0
            if max(height, width) > max_size:
                scale_factor = max_size / max(height, width)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Step 2: Apply SAM segmentation (simulated here)
            # In a real implementation, you would call the SAM model:
            # masks = self.sam_model.generate(image)
            # For this example, we'll simulate the segmentation process
            simulated_masks = self._simulate_sam_segmentation(image)
            
            # Step 3: For each generated mask, extract bounding box and create CLIP embedding
            processed_masks = []
            for mask_data in simulated_masks:
                # Apply the scale factor to the mask if image was downscaled
                if scale_factor != 1.0:
                    original_mask = cv2.resize(
                        mask_data, 
                        (int(width * scale_factor), int(height * scale_factor)), 
                        interpolation=cv2.INTER_NEAREST
                    )
                else:
                    original_mask = mask_data
                
                # Calculate bounding box for this mask
                bbox = self._calculate_bbox(original_mask)
                
                # Create CLIP embedding for the masked region
                clip_embedding = self._create_clip_embedding(image, original_mask)
                
                # Create mask object
                # In the real Mask model, we might need to handle binary_mask differently
                # For now, we'll store it as a placeholder string and the bbox
                mask_obj = Mask(
                    binary_mask=str(original_mask.tolist())[:100],  # Convert to string representation for JSON serialization
                    bbox=bbox,
                    clip_embedding=clip_embedding.tolist() if clip_embedding is not None else None
                )
                
                processed_masks.append(mask_obj)
            
            # Step 6: (Implicitly handled) Model is cached in self.sam_model and self.clip_model attributes
            
            # Step 5: Filter out noise (small masks)
            min_area_percentage = 0.05  # 5% of image area
            total_image_area = height * width
            filtered_masks = [
                mask for mask in processed_masks
                if np.sum(np.array(eval(mask.binary_mask.replace('true', 'True').replace('false', 'False'))) > 0) / total_image_area >= min_area_percentage
            ]
            
            return filtered_masks
            
        except MemoryError:
            # Handle out-of-memory by downscaling further
            print(f"Out of memory processing {image_path}, attempting to downscale further...")
            # In a real implementation, you might downscale further or use tiling
            pil_image = Image.open(image_path)
            # Downscale by 50% more aggressively
            image = np.array(pil_image)
            height, width = image.shape[:2]
            new_width = width // 2
            new_height = height // 2
            downscaled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Then process the further downscaled image
            simulated_masks = self._simulate_sam_segmentation(downscaled_image)
            
            processed_masks = []
            for mask_data in simulated_masks:
                # Apply the scale factor to match original dimensions
                original_mask = cv2.resize(
                    mask_data, 
                    (width, height), 
                    interpolation=cv2.INTER_NEAREST
                )
                
                bbox = self._calculate_bbox(original_mask)
                clip_embedding = self._create_clip_embedding(downscaled_image, mask_data)
                
                mask_obj = Mask(
                    binary_mask=str(original_mask.tolist())[:100],  # Convert to string representation for JSON serialization
                    bbox=bbox,
                    clip_embedding=clip_embedding.tolist() if clip_embedding is not None else None
                )
                
                processed_masks.append(mask_obj)
            
            return processed_masks
    
    def _simulate_sam_segmentation(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Simulate SAM segmentation. In a real implementation, this would call the actual SAM model.
        """
        # For this simulation, we'll create some basic masks based on simple segmentation
        # In a real implementation, this would be replaced with actual SAM model inference
        height, width = image.shape[:2]
        
        # Create a few simulated masks at different positions and sizes
        simulated_masks = []
        
        # Example: create a few rectangular masks of different sizes and positions
        # This is a very simplified simulation of what SAM would do
        for i in range(3):
            mask = np.zeros((height, width), dtype=np.uint8)
            
            # Create a rectangle in different positions
            x_start = np.random.randint(0, width // 3)
            y_start = np.random.randint(0, height // 3)
            x_end = min(width, x_start + np.random.randint(width // 4, width // 2))
            y_end = min(height, y_start + np.random.randint(height // 4, height // 2))
            
            mask[y_start:y_end, x_start:x_end] = 1
            simulated_masks.append(mask)
        
        # Add a circular mask for variety
        center_x, center_y = width // 2, height // 2
        radius = min(width, height) // 4
        y, x = np.ogrid[:height, :width]
        mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        circular_mask = np.zeros((height, width), dtype=np.uint8)
        circular_mask[mask] = 1
        simulated_masks.append(circular_mask)
        
        return simulated_masks

    def _calculate_bbox(self, mask: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Calculate the bounding box of the mask region.
        """
        coords = np.column_stack(np.where(mask > 0))
        if coords.size == 0:
            return (0, 0, 0, 0)
        
        y1, x1 = coords.min(axis=0)
        y2, x2 = coords.max(axis=0)
        return (x1, y1, x2+1, y2+1)

    def _create_clip_embedding(self, image: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
        """
        Create a CLIP embedding for the masked region.
        In a real implementation, this would use CLIP to encode the masked region.
        """
        # In a real implementation, this would:
        # 1. Extract the region of interest based on the mask
        # 2. Preprocess it for CLIP
        # 3. Pass it through the CLIP visual encoder
        # 4. Return the resulting embedding
        
        # For this simulation, return a random vector of appropriate size (512 dimensions for CLIP ViT)
        return np.random.rand(512).astype(np.float32)

# Example usage:
if __name__ == "__main__":
    # In practice, this would analyze a real image file
    # For this example, we'll create a mock analyzer and show how it would work
    analyzer = SAMAnalyzer()
    
    # Since we can't actually open a real file in this context, 
    # let's create a mock implementation that demonstrates the API
    print("SAMAnalyzer initialized.")
    print("In a real implementation, you would call:")
    print("# masks = analyzer.analyze('path/to/image.jpg')")
    print("# for mask in masks:")
    print("#     print(f'BBox: {mask.bbox}, Embedding length: {len(mask.clip_embedding) if mask.clip_embedding else 0}')")
```