# Vision: Models

[Back to Vision Subsystem](./vision_subsystem.md)

## Purpose
Pydantic models for vision subsystem - Contains SceneAnalysis, Entity, EditDelta, and Mask data structures for type-safe operations.

## Models
- `SceneAnalysis`: Contains entities and spatial layout information
- `Entity`: Represents a detected object with ID, label, confidence, etc.
- `EditDelta`: Represents changes between before/after analysis
- `Mask`: Binary mask data structure

### Details
- Type-safe data structures
- Pydantic validation
- JSON serialization support

## Technology Stack

- Pydantic for data validation
- Type hints for type safety

## See Docs

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Tuple
import numpy as np

class Entity(BaseModel):
    """
    Represents a detected object with ID, label, confidence, etc.
    """
    id: str = Field(
        ..., 
        description="Unique identifier for the entity (e.g., 'sky_0')"
    )
    name: str = Field(
        ..., 
        description="Semantic label for the entity (e.g., 'sky', 'tree')"
    )
    bbox: Tuple[int, int, int, int] = Field(
        ..., 
        description="Bounding box coordinates as (x1, y1, x2, y2)"
    )
    confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="Detection confidence score (0-1)"
    )
    color: Optional[Tuple[int, int, int]] = Field(
        None,
        description="Dominant RGB color of the entity (R, G, B values 0-255)"
    )
    centroid: Optional[Tuple[float, float]] = Field(
        None,
        description="Centroid coordinates of the entity (x, y)"
    )
    area: Optional[float] = Field(
        None,
        description="Area of the entity as percentage of total image"
    )

class Relationship(BaseModel):
    """
    Represents a relationship between two entities.
    """
    subject_id: str = Field(
        ..., 
        description="ID of the subject entity"
    )
    object_id: str = Field(
        ..., 
        description="ID of the object entity"
    )
    relationship: str = Field(
        ..., 
        description="Type of relationship (e.g., 'above', 'left_of', 'part_of')"
    )

class SceneAnalysis(BaseModel):
    """
    Contains entities and spatial layout information.
    """
    entities: List[Entity] = Field(
        ..., 
        description="List of detected entities in the scene"
    )
    spatial_layout: str = Field(
        ..., 
        description="Description of the spatial arrangement of entities"
    )
    relationships: List[Relationship] = Field(
        default_factory=list,
        description="List of relationships between entities"
    )
    composition_info: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional information about scene composition"
    )

class Mask(BaseModel):
    """
    Binary mask data structure.
    """
    binary_mask: str = Field(  # In practice, you might use a different representation
        ..., 
        description="Binary mask data (base64 encoded or file path in practice)"
    )
    bbox: Tuple[int, int, int, int] = Field(
        ..., 
        description="Bounding box coordinates as (x1, y1, x2, y2)"
    )
    clip_embedding: Optional[List[float]] = Field(
        None,
        description="CLIP embedding vector for the masked region"
    )
    label: Optional[str] = Field(
        None,
        description="Associated label from CLIP or other labeling system"
    )
    confidence: Optional[float] = Field(
        None,
        ge=0.0, 
        le=1.0,
        description="Confidence score for the mask label"
    )

class EditDelta(BaseModel):
    """
    Represents changes between before/after analysis.
    """
    preserved_entities: List[Entity] = Field(
        default_factory=list,
        description="List of entities that were preserved in the edit"
    )
    modified_entities: List[Entity] = Field(
        default_factory=list,
        description="List of entities that were modified in the edit"
    )
    removed_entities: List[Entity] = Field(
        default_factory=list,
        description="List of entities that were removed in the edit"
    )
    added_entities: List[Entity] = Field(
        default_factory=list,
        description="List of entities that were added in the edit"
    )
    alignment_score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="Overall alignment score between edit and intent"
    )
    entities_preserved_correctly: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="Score for how well non-target entities were preserved"
    )
    intended_changes_applied: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="Score for how well intended changes were applied"
    )
    unintended_changes: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="Score for how many unintended changes occurred"
    )

# Example usage
if __name__ == "__main__":
    # Example of creating an Entity
    entity = Entity(
        id="sky_0",
        name="sky",
        bbox=(0, 0, 500, 200),
        confidence=0.95,
        color=(135, 206, 235),  # Light blue
        centroid=(250.0, 100.0),
        area=0.4  # 40% of image
    )
    print("Entity:", entity.json(indent=2))
    
    # Example of creating a SceneAnalysis
    scene = SceneAnalysis(
        entities=[entity],
        spatial_layout="sky occupies the upper 40% of the image",
        relationships=[
            Relationship(
                subject_id="sky_0",
                object_id="tree_1",
                relationship="above"
            )
        ],
        composition_info={
            "entity_count": 1,
            "scene_type": "outdoor",
            "dominant_colors": ["blue"]
        }
    )
    print("\\nScene Analysis:", scene.json(indent=2))
    
    # Example of creating a Mask
    mask = Mask(
        binary_mask="base64_encoded_mask_data",
        bbox=(100, 150, 200, 250),
        clip_embedding=[0.1, 0.3, 0.5] + [0.0] * 509,  # Simulated embedding
        label="tree",
        confidence=0.88
    )
    print("\\nMask:", mask.json(indent=2))
    
    # Example of creating an EditDelta
    edit_delta = EditDelta(
        preserved_entities=[entity],
        modified_entities=[],
        removed_entities=[],
        added_entities=[],
        alignment_score=0.85,
        entities_preserved_correctly=0.9,
        intended_changes_applied=0.8,
        unintended_changes=0.1
    )
    print("\\nEdit Delta:", edit_delta.json(indent=2))
```