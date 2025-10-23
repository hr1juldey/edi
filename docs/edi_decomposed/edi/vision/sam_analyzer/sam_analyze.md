# SAMAnalyzer.analyze()

[Back to SAM Analyzer](../vision_sam_analyzer.md)

## Related User Story
"As a user, I want EDI to understand my image's composition so it knows what can be safely edited." (from PRD)

## Function Signature
`analyze(image_path: str) -> List[Mask]`

## Parameters
- `image_path: str` - The file path to the image that needs to be analyzed for objects and segments

## Returns
- `List[Mask]` - A list of mask objects that represent the segmented regions of the image

## Step-by-step Logic
1. Load the image from the provided path using PIL
2. Apply SAM 2.1 automatic segmentation to generate a list of masks
3. For each generated mask: extract bounding box, crop the region, and create embedding vector using CLIP
4. Cluster masks by semantic similarity using CLIP text similarity
5. Label clusters using predefined labels like "sky", "building", "person", etc.
6. Cache the SAM model in memory to avoid reloading for subsequent operations
7. Handle out-of-memory situations by downscaling the image if needed

## Performance Optimizations
- Model caching to avoid reloading
- Image downscaling for large images to reduce processing time
- Noise filtering for masks smaller than 5% of image area

## Input/Output Data Structures
### Mask Object
A Mask object contains:
- Binary mask data
- Bounding box coordinates
- CLIP embedding vector
- Associated label and confidence score

## See Docs

```python
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from PIL import Image
import cv2
import torch

@dataclass
class Mask:
    """
    A Mask object contains:
    - Binary mask data
    - Bounding box coordinates
    - CLIP embedding vector
    - Associated label and confidence score
    """
    binary_mask: np.ndarray
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    clip_embedding: Optional[np.ndarray]
    label: Optional[str] = None
    confidence: Optional[float] = None

class SAMAnalyzer:
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
        Analyze an image using SAM for segmentation.
        
        This method:
        1. Load the image from the provided path using PIL
        2. Apply SAM 2.1 automatic segmentation to generate a list of masks
        3. For each generated mask: extract bounding box, crop the region, and create embedding vector using CLIP
        4. Cluster masks by semantic similarity using CLIP text similarity
        5. Label clusters using predefined labels like "sky", "building", "person", etc.
        6. Cache the SAM model in memory to avoid reloading for subsequent operations
        7. Handle out-of-memory situations by downscaling the image if needed
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
                mask_obj = Mask(
                    binary_mask=original_mask,
                    bbox=bbox,
                    clip_embedding=clip_embedding
                )
                
                processed_masks.append(mask_obj)
            
            # Step 4 & 5: Cluster masks by semantic similarity and assign labels using simulated CLIP labeling
            labeled_masks = self._assign_labels_to_masks(processed_masks, image)
            
            # Step 6: (Implicitly handled) Model is cached in self.sam_model and self.clip_model attributes
            
            # Step 5: Filter out noise (small masks)
            min_area_percentage = 0.05  # 5% of image area
            total_image_area = height * width
            filtered_masks = [
                mask for mask in labeled_masks
                if np.sum(mask.binary_mask > 0) / total_image_area >= min_area_percentage
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
                    binary_mask=original_mask,
                    bbox=bbox,
                    clip_embedding=clip_embedding
                )
                
                processed_masks.append(mask_obj)
            
            labeled_masks = self._assign_labels_to_masks(processed_masks, downscaled_image)
            return labeled_masks

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

    def _assign_labels_to_masks(self, masks: List[Mask], image: np.ndarray) -> List[Mask]:
        """
        Assign labels to masks based on CLIP similarity.
        This simulates clustering and labeling using predefined labels.
        """
        labeled_masks = []
        
        for mask_obj in masks:
            # Calculate the dominant color in the masked region to help with label assignment
            masked_pixels = image[mask_obj.binary_mask > 0]
            if len(masked_pixels) > 0:
                avg_color = np.mean(masked_pixels, axis=0)
                
                # Simple heuristic to assign labels based on color and position
                label, confidence = self._heuristic_label_assignment(avg_color, mask_obj.bbox, image.shape)
            else:
                label = "object"
                confidence = 0.5
            
            # Update the mask object with the assigned label and confidence
            labeled_mask = Mask(
                binary_mask=mask_obj.binary_mask,
                bbox=mask_obj.bbox,
                clip_embedding=mask_obj.clip_embedding,
                label=label,
                confidence=confidence
            )
            
            labeled_masks.append(labeled_mask)
        
        return labeled_masks

    def _heuristic_label_assignment(self, avg_color: np.ndarray, bbox: Tuple[int, int, int, int], 
                                   image_shape: Tuple[int, ...]) -> Tuple[str, float]:
        """
        Simple heuristic to assign labels based on average color, position, and size.
        In a real implementation, this would use CLIP text-image similarity.
        """
        # Extract bounding box coordinates
        x1, y1, x2, y2 = bbox
        img_height, img_width = image_shape[0:2]
        
        # Calculate position ratios (0-1) to help with classification
        y_center = (y1 + y2) / 2
        y_position_ratio = y_center / img_height
        area_ratio = ((x2 - x1) * (y2 - y1)) / (img_width * img_height)
        
        # Use color and position heuristics to determine label
        # This is a simplified approach - real CLIP would use semantic similarity
        avg_r, avg_g, avg_b = avg_color
        
        # Simple heuristic rules
        if y_position_ratio < 0.3:  # Top of image, likely sky
            if avg_r > avg_g and avg_r > avg_b:  # More red
                return "cloud", 0.7
            else:  # More blue
                return "sky", 0.8
        
        elif avg_r > avg_g and avg_r > avg_b:  # Reddish area
            if area_ratio > 0.1:  # Large area
                return "building", 0.7
            else:
                return "flower", 0.6
        
        elif avg_g > avg_r and avg_g > avg_b:  # Greenish area, likely vegetation
            return "tree", 0.8 if area_ratio > 0.05 else "grass", 0.75
        
        elif y_position_ratio > 0.7 and area_ratio > 0.2:  # Large area in bottom of image
            return "grass", 0.75
        
        elif avg_b > avg_r and avg_b > avg_g:  # Bluish area
            return "water", 0.7 if area_ratio > 0.1 else "sky", 0.6
        
        else:  # Default to person if the area is moderate and in center
            if 0.2 < y_position_ratio < 0.8 and 0.05 < area_ratio < 0.3:
                return "person", 0.6
            else:
                return "object", 0.55

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
    print("#     print(f'Label: {mask.label}, Confidence: {mask.confidence:.2f}')")