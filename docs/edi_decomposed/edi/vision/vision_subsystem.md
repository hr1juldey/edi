# Vision Subsystem

[Back to Index](../index.md)

## Purpose
Image analysis, object detection, change detection using SAM 2.1 and OpenCLIP

## Component Design

### 1. Vision Subsystem

**Purpose**: Transform images into structured scene understanding

#### 1.1 Object Detection Module

**Inputs**:

- Image file path
- Optional region-of-interest hints from user

**Processing**:

```
1. Load image → PIL Image
2. SAM 2.1 automatic segmentation → List[Mask]
3. For each mask:
   a. Extract bounding box
   b. Crop region
   c. CLIP encode → embedding vector
4. Cluster masks by semantic similarity
5. Label clusters using CLIP text similarity
   (compare to predefined labels: "sky", "building", "person", etc.)
```

**Outputs**:

```python
SceneAnalysis(
    entities=[
        Entity(
            id="sky_0",
            label="sky",
            confidence=0.94,
            bbox=(0, 0, 1920, 760),  # XYXY format
            mask=ndarray,  # Binary mask
            color_dominant="#87CEEB",
            area_percent=39.6
        ),
        Entity(id="building_0", ...),
        ...
    ],
    spatial_layout="sky (top 40%), building (center 55%), grass (bottom 5%)"
)
```

**Performance Optimization**:

- Cache SAM model in memory (load once per session)
- Resize images >2048px to reduce processing time
- Skip fine-grained segmentation if <5% area (noise filtering)

#### 1.2 Change Detection Module

**Purpose**: Compare before/after images to validate edits

**Algorithm**:

```python
def compute_delta(before: SceneAnalysis, after: SceneAnalysis) -> EditDelta:
    # Match entities by spatial overlap (IoU > 0.5)
    matches = match_entities(before.entities, after.entities)
    
    preserved = []
    modified = []
    removed = []
    added = []
    
    for before_entity, after_entity in matches:
        if after_entity is None:
            removed.append(before_entity)
        elif entities_similar(before_entity, after_entity):
            preserved.append((before_entity, after_entity))
        else:
            modified.append((before_entity, after_entity))
    
    for entity in after.entities:
        if entity not in [m[1] for m in matches]:
            added.append(entity)
    
    return EditDelta(
        preserved=preserved,
        modified=modified,
        removed=removed,
        added=added,
        alignment_score=calculate_alignment(...)
    )
```

**Similarity Metrics**:

- Color: ΔE2000 < 10 (perceptually similar)
- Position: Center shift < 5% of image dimension
- Shape: Mask IoU > 0.85

## Sub-modules

This component includes the following modules:

- [vision/sam_analyzer.py](./sam_analyzer/sam_analyzer.md)
- [vision/clip_labeler.py](./clip_labeler/clip_labeler.md)
- [vision/scene_builder.py](./scene_builder/scene_builder.md)
- [vision/change_detector.py](./change_detector/change_detector.md)
- [vision/models.py](./models.md)

## Technology Stack

- SAM 2.1 for segmentation
- OpenCLIP (ViT-B/32) for labeling
- Pillow for image processing
- NumPy for array operations
- Pydantic for data validation

## See Docs

### SAM 2.1 and OpenCLIP Implementation Example
Vision subsystem implementation for EDI:

```python
import numpy as np
from PIL import Image
import torch
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
from dataclasses import dataclass
from enum import Enum
import cv2
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image
import clip  # OpenCLIP
import torchvision.transforms as transforms
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import colorsys
import tempfile

class EntityType(str, Enum):
    SKY = "sky"
    BUILDING = "building"
    PERSON = "person"
    VEHICLE = "vehicle"
    ANIMAL = "animal"
    PLANT = "plant"
    WATER = "water"
    GROUND = "ground"
    FURNITURE = "furniture"
    OTHER = "other"

class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    
    @property
    def width(self) -> float:
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        return self.y2 - self.y1
    
    @property
    def area(self) -> float:
        return self.width * self.height

class Entity(BaseModel):
    id: str
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BoundingBox
    mask: Optional[np.ndarray] = None  # Binary mask
    color_dominant: Optional[str] = None  # Hex color
    area_percent: float = Field(ge=0.0, le=100.0)
    embedding: Optional[List[float]] = None  # CLIP embedding

class SceneAnalysis(BaseModel):
    entities: List[Entity]
    spatial_layout: str

class EditDelta(BaseModel):
    preserved: List[Tuple[Entity, Entity]]
    modified: List[Tuple[Entity, Entity]]
    removed: List[Entity]
    added: List[Entity]
    alignment_score: float = Field(ge=0.0, le=1.0)

class VisionSubsystem:
    """
    Vision subsystem for EDI - handles image analysis, object detection, and change detection.
    """
    
    def __init__(self, sam_checkpoint_path: str = None, clip_model_name: str = "ViT-B/32"):
        self.sam_checkpoint_path = sam_checkpoint_path
        self.clip_model_name = clip_model_name
        
        # Initialize SAM model
        self.sam_model = None
        self.sam_predictor = None
        self._load_sam_model()
        
        # Initialize CLIP model
        self.clip_model, self.clip_preprocess = clip.load(clip_model_name)
        self.clip_model.eval()
        
        # Predefined labels for classification
        self.possible_labels = [
            "sky", "building", "person", "car", "tree", "water", "grass", 
            "road", "mountain", "animal", "furniture", "object"
        ]
        
        # Create text embeddings for predefined labels
        self.label_embeddings = self._encode_texts(self.possible_labels)
    
    def _load_sam_model(self):
        """Load the SAM model for segmentation."""
        if self.sam_checkpoint_path:
            model_type = "vit_h"  # Default to vit_h, could be inferred from checkpoint
            self.sam_model = sam_model_registry[model_type](checkpoint=self.sam_checkpoint_path)
            self.sam_predictor = SamPredictor(self.sam_model)
    
    def _encode_texts(self, texts: List[str]) -> torch.Tensor:
        """Encode texts using CLIP model."""
        text_tokens = clip.tokenize(texts).to(self.clip_model.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features
    
    def preprocess_image(self, image_path: str, max_size: int = 2048) -> Image.Image:
        """
        Preprocess image by resizing if too large, maintaining aspect ratio.
        """
        image = Image.open(image_path).convert("RGB")
        
        # Resize if image is too large
        width, height = image.size
        if max(width, height) > max_size:
            if width > height:
                new_width = max_size
                new_height = int(height * (max_size / width))
            else:
                new_height = max_size
                new_width = int(width * (max_size / height))
            
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return image
    
    def analyze(self, image_path: str, roi_hints: Optional[List[Tuple[int, int, int, int]]] = None) -> SceneAnalysis:
        """
        Perform comprehensive scene analysis on an image.
        """
        # Preprocess image
        pil_image = self.preprocess_image(image_path)
        image_np = np.array(pil_image)
        
        # Set image for SAM predictor
        self.sam_predictor.set_image(image_np)
        
        # Generate masks using SAM
        masks, scores, logits = self.sam_predictor.predict()
        
        # Filter masks by confidence and size
        min_area = (pil_image.width * pil_image.height) * 0.001  # 0.1% of total area
        valid_indices = [i for i in range(len(masks)) if 
                        masks[i].sum() > min_area and  # Skip tiny masks
                        scores[i] > 0.5]  # Skip low-confidence masks
        
        entities = []
        for idx in valid_indices:
            mask = masks[idx]
            score = scores[idx]
            
            # Get bounding box
            bbox = self._get_bounding_box(mask)
            
            # Create entity ID
            entity_id = f"{bbox.width * bbox.height:.0f}_{int(score*100)}"
            
            # Calculate area percentage
            area_percent = (mask.sum() / (pil_image.width * pil_image.height)) * 100
            
            # Get dominant color
            color_hex = self._get_dominant_color(image_np, mask)
            
            # CLIP encode the masked region
            region_image = self._extract_region_image(image_np, mask)
            embedding = self._encode_image(region_image).cpu().numpy().tolist()
            
            # Label the entity using CLIP similarity
            label, confidence = self._label_entity(embedding)
            
            entity = Entity(
                id=entity_id,
                label=label,
                confidence=confidence,
                bbox=bbox,
                mask=mask,
                color_dominant=color_hex,
                area_percent=area_percent,
                embedding=embedding
            )
            
            entities.append(entity)
        
        # Cluster similar entities (optional)
        clustered_entities = self._cluster_entities(entities)
        
        # Create spatial layout description
        spatial_layout = self._describe_spatial_layout(clustered_entities, pil_image.size)
        
        return SceneAnalysis(
            entities=clustered_entities,
            spatial_layout=spatial_layout
        )
    
    def _get_bounding_box(self, mask: np.ndarray) -> BoundingBox:
        """
        Calculate bounding box from a binary mask.
        """
        rows, cols = np.where(mask)
        if len(rows) == 0 or len(cols) == 0:
            return BoundingBox(x1=0, y1=0, x2=0, y2=0)
        
        y1, y2 = rows.min(), rows.max()
        x1, x2 = cols.min(), cols.max()
        
        return BoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2))
    
    def _get_dominant_color(self, image: np.ndarray, mask: np.ndarray) -> str:
        """
        Calculate dominant color of masked region.
        """
        # Apply mask to image to get the region
        masked_image = image.copy()
        masked_image[mask == 0] = [0, 0, 0]  # Set non-region to black
        
        # Get non-black pixels
        non_black = masked_image.reshape(-1, 3)
        non_black = non_black[~np.all(non_black == 0, axis=1)]
        
        if len(non_black) == 0:
            return "#808080"  # Default gray
        
        # Use K-means to find dominant color
        n_colors = min(3, len(non_black))
        if n_colors == 0:
            return "#808080"
        
        kmeans = KMeans(n_clusters=n_colors, random_state=42)
        kmeans.fit(non_black)
        dominant_color = kmeans.cluster_centers_[0]
        
        # Convert to hex
        hex_color = "#{:02x}{:02x}{:02x}".format(
            int(dominant_color[0]), 
            int(dominant_color[1]), 
            int(dominant_color[2])
        )
        
        return hex_color
    
    def _extract_region_image(self, image: np.ndarray, mask: np.ndarray) -> Image.Image:
        """
        Extract a region from an image using a mask.
        """
        # Apply mask
        masked_image = image.copy()
        masked_image[mask == 0] = 0  # Set non-region to black
        
        # Find bounding box to crop
        rows, cols = np.where(mask)
        if len(rows) == 0 or len(cols) == 0:
            # Return a small black image if mask is empty
            return Image.new("RGB", (10, 10), color="black")
        
        y1, y2 = rows.min(), rows.max()
        x1, x2 = cols.min(), cols.max()
        
        # Crop the region
        cropped = masked_image[y1:y2+1, x1:x2+1]
        
        return Image.fromarray(cropped)
    
    def _encode_image(self, image: Image.Image) -> torch.Tensor:
        """
        Encode an image using CLIP model.
        """
        # Preprocess image
        image_tensor = self.clip_preprocess(image).unsqueeze(0).to(self.clip_model.device)
        
        # Encode
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        
        return image_features
    
    def _label_entity(self, embedding: List[float]) -> Tuple[str, float]:
        """
        Label an entity based on CLIP similarity to predefined labels.
        """
        # Convert embedding to tensor
        entity_tensor = torch.tensor([embedding]).to(self.clip_model.device)
        
        # Calculate similarity with predefined labels
        similarities = (entity_tensor @ self.label_embeddings.T).squeeze(0)
        max_idx = similarities.argmax().item()
        
        label = self.possible_labels[max_idx]
        confidence = float(similarities[max_idx])
        
        return label, confidence
    
    def _cluster_entities(self, entities: List[Entity]) -> List[Entity]:
        """
        Cluster similar entities based on spatial and semantic features.
        """
        # For now, just return the entities as-is
        # In a real implementation, this would cluster similar entities
        # that are close together spatially
        return entities
    
    def _describe_spatial_layout(self, entities: List[Entity], image_size: Tuple[int, int]) -> str:
        """
        Create a text description of the spatial layout.
        """
        width, height = image_size
        entity_layouts = []
        
        for entity in entities:
            # Determine spatial position
            center_x = (entity.bbox.x1 + entity.bbox.x2) / 2
            center_y = (entity.bbox.y1 + entity.bbox.y2) / 2
            
            x_pos = "left" if center_x < width/3 else "center" if center_x < 2*width/3 else "right"
            y_pos = "top" if center_y < height/3 else "middle" if center_y < 2*height/3 else "bottom"
            
            entity_layouts.append(f"{entity.label} ({x_pos} {y_pos}, {entity.area_percent:.1f}%)")
        
        return ", ".join(entity_layouts[:5])  # Limit to first 5 for brevity
    
    def compute_delta(self, 
                     before_analysis: SceneAnalysis, 
                     after_analysis: SceneAnalysis) -> EditDelta:
        """
        Compare before/after images to validate edits and compute differences.
        """
        # Match entities between before and after
        matches = self._match_entities(before_analysis.entities, after_analysis.entities)
        
        preserved = []
        modified = []
        removed = []
        added = []
        
        # Process matches
        matched_after_ids = set()
        for before_entity, after_entity in matches:
            if after_entity is None:
                removed.append(before_entity)
            elif self._entities_similar(before_entity, after_entity):
                preserved.append((before_entity, after_entity))
            else:
                modified.append((before_entity, after_entity))
                matched_after_ids.add(after_entity.id)
        
        # Find added entities (those not matched in before)
        for after_entity in after_analysis.entities:
            if after_entity.id not in [m[1].id if m[1] else None for m in matches if m[1]]:
                added.append(after_entity)
        
        # Calculate alignment score based on preservation
        total_entities = len(before_analysis.entities)
        preserved_count = len(preserved)
        modified_count = len(modified)
        removed_count = len(removed)
        added_count = len(added)
        
        # Scoring: High preservation = high score, significant changes = lower score
        preservation_ratio = preserved_count / max(total_entities, 1)
        change_ratio = (modified_count + removed_count + added_count) / max(total_entities, 1)
        
        alignment_score = max(0.0, preservation_ratio - (change_ratio * 0.3))
        
        return EditDelta(
            preserved=preserved,
            modified=modified,
            removed=removed,
            added=added,
            alignment_score=min(alignment_score, 1.0)
        )
    
    def _match_entities(self, 
                       before_entities: List[Entity], 
                       after_entities: List[Entity],
                       iou_threshold: float = 0.5) -> List[Tuple[Entity, Optional[Entity]]]:
        """
        Match entities between before and after analyses based on spatial overlap.
        """
        matches = []
        
        for before_entity in before_entities:
            best_match = None
            best_iou = 0
            
            for after_entity in after_entities:
                if after_entity.mask is not None and before_entity.mask is not None:
                    iou = self._calculate_iou(before_entity.mask, after_entity.mask)
                    if iou > best_iou and iou > iou_threshold:
                        best_iou = iou
                        best_match = after_entity
            
            matches.append((before_entity, best_match))
        
        return matches
    
    def _calculate_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """
        Calculate Intersection over Union between two masks.
        """
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _entities_similar(self, 
                         entity1: Entity, 
                         entity2: Entity,
                         color_threshold: float = 10.0,
                         position_threshold: float = 0.05,
                         shape_threshold: float = 0.85) -> bool:
        """
        Determine if two entities are similar based on color, position, and shape.
        """
        # Check color similarity (simplified - in reality would use better color distance)
        color_similar = self._colors_similar(entity1.color_dominant, entity2.color_dominant, color_threshold)
        
        # Check position similarity
        pos_similar = self._positions_similar(entity1.bbox, entity2.bbox, position_threshold)
        
        # Check shape similarity (IoU of masks if available)
        shape_similar = self._shapes_similar(entity1, entity2, shape_threshold)
        
        # Entity is similar if all aspects are similar
        return color_similar and pos_similar and shape_similar
    
    def _colors_similar(self, color1: str, color2: str, threshold: float) -> bool:
        """
        Check if two hex colors are perceptually similar.
        """
        # Convert hex to RGB
        def hex_to_rgb(hex_color):
            hex_color = hex_color.lstrip('#')
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
        # Simplified color similarity check
        # In reality, would use ΔE2000 or other perceptual color difference
        try:
            rgb1 = hex_to_rgb(color1)
            rgb2 = hex_to_rgb(color2)
            
            # Euclidean distance in RGB space (simplified)
            diff = sum((c1 - c2) ** 2 for c1, c2 in zip(rgb1, rgb2)) ** 0.5
            return diff < threshold * 255 / 100  # Threshold as percentage
        except:
            return True  # Default to similar if color parsing fails
    
    def _positions_similar(self, bbox1: BoundingBox, bbox2: BoundingBox, threshold: float) -> bool:
        """
        Check if two entities have similar positions.
        """
        # Calculate centers
        center1_x = (bbox1.x1 + bbox1.x2) / 2
        center1_y = (bbox1.y1 + bbox1.y2) / 2
        center2_x = (bbox2.x1 + bbox2.x2) / 2
        center2_y = (bbox2.y1 + bbox2.y2) / 2
        
        # Calculate distance as percentage of image size
        # This is a placeholder - would need actual image dimensions
        # For now, assume a standard size
        max_dim = max(bbox1.x2, bbox1.y2, bbox2.x2, bbox2.y2)
        distance = ((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2) ** 0.5
        
        return (distance / max_dim) < threshold
    
    def _shapes_similar(self, entity1: Entity, entity2: Entity, threshold: float) -> bool:
        """
        Check if two entities have similar shapes (using mask IoU).
        """
        if entity1.mask is not None and entity2.mask is not None:
            iou = self._calculate_iou(entity1.mask, entity2.mask)
            return iou > threshold
        
        # If no masks available, assume similar
        return True

# Example usage
if __name__ == "__main__":
    # Initialize the vision subsystem
    # Note: This would require actual model files in a real implementation
    try:
        vision_system = VisionSubsystem()
        
        # Example: Analyze an image (replace with actual image path)
        # analysis = vision_system.analyze("example.jpg")
        # print(f"Found {len(analysis.entities)} entities")
        
        print("Vision subsystem initialized successfully")
        
    except Exception as e:
        print(f"Error initializing vision subsystem: {e}")
        print("This example shows the structure but requires actual model files")
```

### Pydantic and NumPy Implementation Example
Data validation and array operations for the vision subsystem:

```python
import numpy as np
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Union
from enum import Enum
import cv2
from PIL import Image

class ColorType(str, Enum):
    RGB = "RGB"
    HSV = "HSV"
    LAB = "LAB"

class BoundingBox3D(BaseModel):
    x1: float = Field(..., description="X coordinate of top-left corner")
    y1: float = Field(..., description="Y coordinate of top-left corner")
    z1: float = Field(0, description="Z coordinate (depth) of top-left corner")
    x2: float = Field(..., description="X coordinate of bottom-right corner")
    y2: float = Field(..., description="Y coordinate of bottom-right corner")
    z2: float = Field(0, description="Z coordinate (depth) of bottom-right corner")
    
    @validator('x2', 'y2', 'z2')
    def validate_coordinates(cls, v, values, field):
        """Ensure that bottom-right coordinates are greater than top-left."""
        if field.name.startswith('x'):
            other = 'x1' if field.name == 'x2' else 'x1'
        elif field.name.startswith('y'):
            other = 'y1' if field.name == 'y2' else 'y1'
        else:  # z
            other = 'z1' if field.name == 'z2' else 'z1'
        
        if other in values and v <= values[other]:
            raise ValueError(f'{field.name} must be greater than {other}')
        return v
    
    @property
    def width(self) -> float:
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        return self.y2 - self.y1
    
    @property
    def depth(self) -> float:
        return self.z2 - self.z1
    
    @property
    def volume(self) -> float:
        return self.width * self.height * self.depth
    
    @property
    def area(self) -> float:
        return self.width * self.height

class ImageMask(BaseModel):
    """Represents a binary or labeled image mask."""
    data: np.ndarray = Field(..., description="Binary or labeled mask data")
    width: int = Field(..., description="Width of the mask")
    height: int = Field(..., description="Height of the mask")
    data_type: str = Field(default="binary", description="Type of mask data (binary, labeled)")
    
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            np.ndarray: lambda x: x.tolist()
        }
    
    @validator('data', pre=True)
    def validate_mask_data(cls, v):
        """Validate the mask data format."""
        if isinstance(v, list):
            v = np.array(v)
        
        if not isinstance(v, np.ndarray):
            raise ValueError("Mask data must be a numpy array")
        
        if len(v.shape) not in [2, 3]:
            raise ValueError("Mask must be 2D or 3D array")
        
        return v
    
    def get_region_properties(self, label: int = 1) -> Dict[str, Any]:
        """Calculate properties of a specific region in the mask."""
        if self.data_type == "binary":
            region = self.data.astype(bool)
        else:
            region = self.data == label
        
        # Calculate region properties
        area = region.sum()
        pixels = np.where(region)
        
        if len(pixels[0]) == 0:  # Empty region
            return {"area": 0, "centroid": (0, 0), "bbox": (0, 0, 0, 0)}
        
        y_min, y_max = pixels[0].min(), pixels[0].max()
        x_min, x_max = pixels[1].min(), pixels[1].max()
        
        centroid_y = pixels[0].mean()
        centroid_x = pixels[1].mean()
        
        return {
            "area": float(area),
            "centroid": (float(centroid_x), float(centroid_y)),
            "bbox": (int(x_min), int(y_min), int(x_max), int(y_max)),
            "height": int(y_max - y_min),
            "width": int(x_max - x_min)
        }

class ColorInformation(BaseModel):
    """Color information for an image region."""
    dominant_color: str = Field(..., regex=r'^#([A-Fa-f0-9]{6}))  # Hex color
    color_space: ColorType = ColorType.RGB
    color_values: List[float] = Field(..., min_items=3, max_items=4)  # RGB/HSV/LAB values
    color_variance: float = Field(ge=0, description="Variance of colors in region")
    
    @validator('color_values')
    def validate_color_values(cls, v, values):
        """Validate color values based on color space."""
        color_space = values.get('color_space', ColorType.RGB)
        
        if color_space == ColorType.RGB:
            if len(v) != 3:
                raise ValueError("RGB must have 3 values")
            if not all(0 <= val <= 255 for val in v):
                raise ValueError("RGB values must be between 0 and 255")
        elif color_space == ColorType.HSV:
            if len(v) != 3:
                raise ValueError("HSV must have 3 values")
            if not (0 <= v[0] <= 360):  # Hue
                raise ValueError("Hue must be between 0 and 360")
            if not all(0 <= val <= 100 for val in v[1:]):  # Saturation, Value
                raise ValueError("Saturation and Value must be between 0 and 100")
        elif color_space == ColorType.LAB:
            if len(v) != 3:
                raise ValueError("LAB must have 3 values")
            # L* is typically 0-100, a* and b* are typically -128 to +127
            if not (0 <= v[0] <= 100):  # L*
                raise ValueError("L* value must be between 0 and 100")
            if not all(-128 <= val <= 127 for val in v[1:]):  # a*, b*
                raise ValueError("a* and b* values must be between -128 and 127")
        
        return v

class AdvancedEntity(BaseModel):
    """Enhanced entity with more detailed information."""
    id: str
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BoundingBox3D
    mask: Optional[ImageMask] = None
    color_info: Optional[ColorInformation] = None
    area_percent: float = Field(ge=0.0, le=100.0)
    embedding: Optional[List[float]] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    @validator('embedding')
    def validate_embedding(cls, v):
        """Validate embedding is a list of floats."""
        if not all(isinstance(x, (int, float)) for x in v):
            raise ValueError("Embedding must contain numeric values")
        return [float(x) for x in v]

class AdvancedSceneAnalysis(BaseModel):
    """Enhanced scene analysis with additional information."""
    entities: List[AdvancedEntity]
    spatial_layout: str
    image_dimensions: Tuple[int, int]  # (width, height)
    overall_color_pallete: List[str] = Field(default_factory=list)
    dominant_colors: List[ColorInformation] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class VisionDataProcessor:
    """Handles advanced data processing for the vision subsystem."""
    
    @staticmethod
    def calculate_image_statistics(image: np.ndarray) -> Dict[str, float]:
        """Calculate statistical properties of an image."""
        # Convert to float for calculations
        img_float = image.astype(np.float64)
        
        stats = {
            "mean": np.mean(img_float),
            "std": np.std(img_float),
            "min": np.min(img_float),
            "max": np.max(img_float),
            "median": np.median(img_float),
            "skewness": 0,  # Would require scipy.stats
            "kurtosis": 0   # Would require scipy.stats
        }
        
        # Calculate per-channel statistics if image is color
        if len(image.shape) == 3:
            channel_stats = {}
            for i in range(image.shape[2]):
                channel_data = image[:, :, i].astype(np.float64)
                channel_stats[f"channel_{i}"] = {
                    "mean": float(np.mean(channel_data)),
                    "std": float(np.std(channel_data)),
                    "min": float(np.min(channel_data)),
                    "max": float(np.max(channel_data))
                }
            stats["channels"] = channel_stats
        
        return stats
    
    @staticmethod
    def compute_gradients(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute image gradients using Sobel operators."""
        if len(image.shape) == 3:
            # Convert to grayscale if color image
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Calculate gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        return grad_x, grad_y
    
    @staticmethod
    def extract_features(image: np.ndarray) -> Dict[str, Any]:
        """Extract various features from an image."""
        features = {}
        
        # Color features
        features["color_histogram"] = VisionDataProcessor._compute_color_histogram(image)
        
        # Texture features
        features["texture_contrast"] = VisionDataProcessor._compute_contrast(image)
        
        # Edge features
        grad_x, grad_y = VisionDataProcessor.compute_gradients(image)
        features["edge_density"] = np.mean(np.abs(grad_x) + np.abs(grad_y))
        features["edge_strength"] = np.sqrt(grad_x**2 + grad_y**2).mean()
        
        # Statistical features
        features["statistics"] = VisionDataProcessor.calculate_image_statistics(image)
        
        return features
    
    @staticmethod
    def _compute_color_histogram(image: np.ndarray, bins: int = 256) -> Dict[str, np.ndarray]:
        """Compute color histogram for an image."""
        histograms = {}
        
        if len(image.shape) == 3:
            # Multi-channel image
            for i in range(image.shape[2]):
                histograms[f"channel_{i}"] = np.histogram(image[:, :, i], bins=bins)[0]
        else:
            # Single channel image
            histograms["channel_0"] = np.histogram(image, bins=bins)[0]
        
        return histograms
    
    @staticmethod
    def _compute_contrast(image: np.ndarray) -> float:
        """Compute image contrast using standard deviation."""
        if len(image.shape) == 3:
            # For color images, compute contrast per channel then average
            contrasts = []
            for i in range(image.shape[2]):
                channel_contrast = np.std(image[:, :, i])
                contrasts.append(channel_contrast)
            return float(sum(contrasts) / len(contrasts))
        else:
            return float(np.std(image))

# Example usage
if __name__ == "__main__":
    # Example of BoundingBox3D validation
    bbox = BoundingBox3D(x1=10, y1=10, x2=100, y2=100)
    print(f"Bounding box area: {bbox.area}, volume: {bbox.volume}")
    
    # Example of mask processing
    # Create a sample binary mask
    mask_data = np.zeros((100, 100), dtype=np.uint8)
    mask_data[20:60, 20:60] = 1  # White square in black background
    
    image_mask = ImageMask(
        data=mask_data,
        width=100,
        height=100,
        data_type="binary"
    )
    
    region_props = image_mask.get_region_properties()
    print(f"Region properties: {region_props}")
    
    # Example of color information
    color_info = ColorInformation(
        dominant_color="#FF5733",
        color_space=ColorType.RGB,
        color_values=[255, 87, 51],
        color_variance=10.5
    )
    
    print(f"Color info: {color_info}")
    
    # Example of advanced entity
    advanced_entity = AdvancedEntity(
        id="test_123",
        label="object",
        confidence=0.85,
        bbox=BoundingBox3D(x1=0, y1=0, x2=100, y2=100),
        area_percent=15.0,
        embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
        color_info=color_info
    )
    
    print(f"Advanced entity: {advanced_entity}")
    
    print("Vision data validation and processing examples completed!")
```