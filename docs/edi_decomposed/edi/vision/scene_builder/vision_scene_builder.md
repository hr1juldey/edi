# Vision: Scene Builder

[Back to Vision Subsystem](./vision_subsystem.md)

## Purpose
Assembles SceneAnalysis - Contains the SceneBuilder class that clusters related entities and computes spatial layout description.

## Class: SceneBuilder

### Methods
- `build(masks, labels) -> SceneAnalysis`: Takes masks and labels and returns a SceneAnalysis object

### Details
- Clusters related entities together
- Computes spatial layout description
- Creates the final structured scene understanding

## Functions

- [build(masks, labels)](./vision/scene_build.md)

## Technology Stack

- Pydantic for data validation
- NumPy for array operations

## See Docs

```python
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from ...vision.models import Entity, Relationship, SceneAnalysis
from dataclasses import dataclass
from collections import defaultdict
import cv2

class SceneBuilder:
    """
    Assembles SceneAnalysis - Contains methods to cluster related entities 
    and compute spatial layout description.
    """
    
    def __init__(self, proximity_threshold: float = 0.3, similarity_threshold: float = 0.8):
        self.proximity_threshold = proximity_threshold
        self.similarity_threshold = similarity_threshold

    def build(self, masks: List[np.ndarray], labels: List[Dict[str, Any]]) -> SceneAnalysis:
        """
        Takes masks and labels and returns a SceneAnalysis object.
        
        This method:
        - Clusters related entities together
        - Computes spatial layout description
        - Creates the final structured scene understanding
        """
        # Validate inputs
        if len(masks) != len(labels):
            raise ValueError("Number of masks must match number of labels")
        
        # Create initial entity objects from masks and labels
        entities = []
        for i, (mask, label) in enumerate(zip(masks, labels)):
            entity = self._create_entity(mask, label, i)
            if entity:
                entities.append(entity)
        
        # Cluster related entities (e.g., group body parts into a person)
        clustered_entities = self._cluster_entities(entities)
        
        # Compute spatial layout
        spatial_layout = self._compute_spatial_layout(clustered_entities)
        
        # Determine relationships between entities
        relationships = self._compute_relationships(clustered_entities)
        
        # Calculate composition information
        composition_info = self._compute_composition_info(clustered_entities, spatial_layout, relationships)
        
        return SceneAnalysis(
            entities=clustered_entities,
            spatial_layout=spatial_layout,
            relationships=relationships,
            composition_info=composition_info
        )
    
    def _create_entity(self, mask: np.ndarray, label: Dict[str, Any], idx: int) -> Optional[Entity]:
        """Create an Entity object from a mask and label."""
        # Calculate bounding box
        coords = np.column_stack(np.where(mask > 0))
        if coords.size == 0:
            # If mask is empty, skip this entity
            return None
        
        y1, x1 = coords.min(axis=0)
        y2, x2 = coords.max(axis=0)
        bbox = (x1, y1, x2, y2)
        
        # Calculate centroid
        centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
        
        # Calculate area as percentage of mask
        area = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])
        
        # Calculate dominant color from the masked region
        color = self._calculate_dominant_color(mask, label)
        
        # Create and return entity
        entity_id = f"{label.get('name', 'unknown')}_{idx}"
        return Entity(
            id=entity_id,
            name=label.get('name', 'unknown'),
            bbox=bbox,
            confidence=label.get('confidence', 0.9),  # Default confidence if not provided
            color=color,
            centroid=centroid,
            area=area
        )
    
    def _calculate_dominant_color(self, mask: np.ndarray, label: Dict[str, Any]) -> Optional[Tuple[int, int, int]]:
        """Calculate the dominant color in the masked area."""
        # This is a placeholder implementation
        # In a real implementation, you would calculate the dominant color from the original image
        # using the mask to limit the calculation to the entity region
        return (128, 128, 128)  # Placeholder color
    
    def _cluster_entities(self, entities: List[Entity]) -> List[Entity]:
        """Cluster related entities based on spatial proximity and semantic similarity."""
        # In this implementation, we'll group entities that likely belong to the same object
        # For example: face, eyes, nose, mouth could be grouped into a person
        clustered_entities = []
        used_entities = set()
        
        for i, entity in enumerate(entities):
            if entity.id in used_entities:
                continue
                
            # Find related entities (close spatially and semantically related)
            related_entities = [entity]
            used_entities.add(entity.id)
            
            # Look for other entities that might be parts of the same object
            for j, other_entity in enumerate(entities[i+1:], i+1):
                if other_entity.id in used_entities:
                    continue
                
                # Check spatial proximity
                center_dist = np.sqrt(
                    (entity.centroid[0] - other_entity.centroid[0])**2 + 
                    (entity.centroid[1] - other_entity.centroid[1])**2
                )
                
                # Normalize by image size for relative distance
                max_dim = max(entity.bbox[2], entity.bbox[3])  # Use bbox dimensions as reference
                if max_dim > 0:
                    normalized_dist = center_dist / max_dim
                else:
                    normalized_dist = float('inf')
                
                # Check semantic similarity (simplified - just check if one is a part of the other)
                is_part = self._are_semantically_related(entity.name, other_entity.name)
                
                if normalized_dist < self.proximity_threshold or is_part:
                    related_entities.append(other_entity)
                    used_entities.add(other_entity.id)
            
            # Combine related entities into a single entity if needed
            if len(related_entities) > 1:
                # Create a new entity that represents the combined object
                combined_entity = self._combine_entities(related_entities)
                clustered_entities.append(combined_entity)
            else:
                clustered_entities.append(entity)
        
        return clustered_entities
    
    def _are_semantically_related(self, name1: str, name2: str) -> bool:
        """Check if two entity names are semantically related."""
        # Simplified semantic relationship check
        body_parts = {"face", "eye", "nose", "mouth", "head", "torso", "arm", "leg", "hand", "foot"}
        vehicle_parts = {"wheel", "door", "window", "hood", "trunk", "headlight", "taillight"}
        
        # Check if both names are body parts
        if name1 in body_parts and name2 in body_parts:
            return True
            
        # Check if both names are vehicle parts
        if name1 in vehicle_parts and name2 in vehicle_parts:
            return True
            
        # Check for common object-part relationships
        common_parts = {
            "face": ["eye", "nose", "mouth", "ear"],
            "person": ["face", "head", "torso", "arm", "leg", "hand", "foot"],
            "car": ["wheel", "door", "window", "hood", "trunk", "headlight", "taillight"],
            "building": ["window", "door", "roof"],
            "tree": ["trunk", "leaf", "branch"],
        }
        
        if name1 in common_parts and name2 in common_parts[name1]:
            return True
        if name2 in common_parts and name1 in common_parts[name2]:
            return True
            
        return False
    
    def _combine_entities(self, entities: List[Entity]) -> Entity:
        """Combine multiple related entities into a single entity."""
        if len(entities) == 1:
            return entities[0]
        
        # Determine the primary entity (likely the largest or most general)
        primary_entity = max(entities, key=lambda e: e.area if e.area else 0)
        
        # Expand the bounding box to encompass all related entities
        min_x = min(e.bbox[0] for e in entities)
        min_y = min(e.bbox[1] for e in entities)
        max_x = max(e.bbox[2] for e in entities)
        max_y = max(e.bbox[3] for e in entities)
        
        # Calculate new centroid
        avg_x = sum(e.centroid[0] for e in entities) / len(entities)
        avg_y = sum(e.centroid[1] for e in entities) / len(entities)
        
        # Combine areas (but this is approximate)
        total_area = sum(e.area if e.area else 0 for e in entities)
        
        # Use the primary entity's name but indicate it's a combination
        combined_name = f"{primary_entity.name}_group"
        
        return Entity(
            id=combined_name,
            name=combined_name,
            bbox=(min_x, min_y, max_x, max_y),
            confidence=max(e.confidence if e.confidence else 0 for e in entities),  # Use highest confidence
            color=primary_entity.color,  # Use primary entity's color
            centroid=(avg_x, avg_y),
            area=min(total_area, 1.0)  # Ensure area doesn't exceed 100%
        )
    
    def _compute_spatial_layout(self, entities: List[Entity]) -> str:
        """Compute the spatial layout description of the scene."""
        if not entities:
            return "Scene contains no detectable entities"
        
        # Sort entities by vertical position (y-coordinate)
        sorted_entities = sorted(entities, key=lambda e: e.centroid[1])
        
        # Calculate layout regions (top, middle, bottom)
        height_percentages = []
        for entity in sorted_entities:
            # Calculate vertical position as percentage of image height
            # This is a simplified approach; in a real implementation, you'd use image dimensions
            y_pos = entity.centroid[1]
            # For this example, we'll use a placeholder height of 100%
            height_percent = (y_pos / 100.0) * 100  # Placeholder calculation
            
            height_percentages.append((entity.name, height_percent, entity.area))
        
        # Group entities by their vertical position
        top_entities = []
        middle_entities = []
        bottom_entities = []
        
        for name, y_pos, area in height_percentages:
            if y_pos < 33:
                top_entities.append(f"{name} ({area*100:.1f}%)")
            elif y_pos < 66:
                middle_entities.append(f"{name} ({area*100:.1f}%)")
            else:
                bottom_entities.append(f"{name} ({area*100:.1f}%)")
        
        # Create spatial layout description
        layout_parts = []
        if top_entities:
            layout_parts.append(f"top: {', '.join(top_entities)}")
        if middle_entities:
            layout_parts.append(f"middle: {', '.join(middle_entities)}")
        if bottom_entities:
            layout_parts.append(f"bottom: {', '.join(bottom_entities)}")
        
        return "; ".join(layout_parts) if layout_parts else "Scene layout could not be determined"
    
    def _compute_relationships(self, entities: List[Entity]) -> List[Relationship]:
        """Determine relationships between entities."""
        relationships = []
        
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                # Calculate relationship based on spatial position
                rel = self._determine_spatial_relationship(entity1, entity2)
                
                if rel:
                    relationships.append(Relationship(
                        subject_id=entity1.id,
                        object_id=entity2.id,
                        relationship=rel
                    ))
        
        return relationships
    
    def _determine_spatial_relationship(self, entity1: Entity, entity2: Entity) -> Optional[str]:
        """Determine the spatial relationship between two entities."""
        x1, y1 = entity1.centroid
        x2, y2 = entity2.centroid
        
        # Calculate horizontal and vertical distances
        dx = x2 - x1
        dy = y2 - y1
        
        # Determine spatial relationship
        if abs(dx) > abs(dy):  # Horizontal relationship is stronger
            if dx > 0:
                return "right_of"
            else:
                return "left_of"
        else:  # Vertical relationship is stronger
            if dy > 0:
                return "below"
            else:
                return "above"
    
    def _compute_composition_info(self, entities: List[Entity], spatial_layout: str, relationships: List[Relationship]) -> Dict[str, Any]:
        """Compute overall scene composition information."""
        composition_info = {
            "entity_count": len(entities),
            "spatial_layout": spatial_layout,
            "relationship_count": len(relationships),
            "dominant_colors": self._get_dominant_colors(entities),
            "scene_type": self._infer_scene_type(entities),
            "depth_layers": self._infer_depth_layers(entities)
        }
        
        return composition_info
    
    def _get_dominant_colors(self, entities: List[Entity]) -> List[Tuple[int, int, int]]:
        """Get dominant colors in the scene."""
        colors = []
        for entity in entities:
            if entity.color:
                colors.append(entity.color)
        return colors
    
    def _infer_scene_type(self, entities: List[Entity]) -> str:
        """Infer the scene type based on detected entities."""
        # Simple scene type inference based on common entities
        entity_names = [e.name.lower() for e in entities]
        
        # Define keywords for different scene types
        indoor_keywords = ["chair", "table", "sofa", "bed", "couch", "kitchen", "bathroom", "room"]
        outdoor_keywords = ["tree", "sky", "grass", "mountain", "water", "road", "street", "building"]
        nature_keywords = ["tree", "grass", "flower", "river", "lake", "mountain", "forest", "beach"]
        urban_keywords = ["car", "building", "road", "street", "traffic", "city"]
        
        # Count occurrences
        indoor_count = sum(1 for name in entity_names if any(kw in name for kw in indoor_keywords))
        outdoor_count = sum(1 for name in entity_names if any(kw in name for kw in outdoor_keywords))
        nature_count = sum(1 for name in entity_names if any(kw in name for kw in nature_keywords))
        urban_count = sum(1 for name in entity_names if any(kw in name for kw in urban_keywords))
        
        # Determine scene type
        if nature_count >= max(indoor_count, outdoor_count, urban_count) and nature_count > 0:
            return "nature"
        elif urban_count >= max(indoor_count, outdoor_count, nature_count) and urban_count > 0:
            return "urban"
        elif indoor_count >= max(outdoor_count, nature_count, urban_count) and indoor_count > 0:
            return "indoor"
        elif outdoor_count >= max(indoor_count, nature_count, urban_count) and outdoor_count > 0:
            return "outdoor"
        else:
            return "unknown"
    
    def _infer_depth_layers(self, entities: List[Entity]) -> List[str]:
        """Infer depth layers in the scene (background, middle ground, foreground)."""
        if not entities:
            return ["no entities detected"]
        
        # Sort entities by vertical position (Y-coordinate)
        sorted_entities = sorted(entities, key=lambda e: e.centroid[1])
        
        # Divide into three layers: bottom (foreground), middle, top (background)
        n = len(sorted_entities)
        layers = {
            "background": [],
            "middle_ground": [],
            "foreground": []
        }
        
        # Assign entities to layers based on position
        for i, entity in enumerate(sorted_entities):
            if i < n / 3:  # Bottom third
                layers["foreground"].append(entity.name)
            elif i < 2 * n / 3:  # Middle third
                layers["middle_ground"].append(entity.name)
            else:  # Top third
                layers["background"].append(entity.name)
        
        # Return layer information
        result = []
        if layers["background"]:
            result.append(f"background: {', '.join(layers['background'])}")
        if layers["middle_ground"]:
            result.append(f"middle_ground: {', '.join(layers['middle_ground'])}")
        if layers["foreground"]:
            result.append(f"foreground: {', '.join(layers['foreground'])}")
        
        return result

# Example usage:
if __name__ == "__main__":
    # Create mock masks and labels
    # In a real implementation, these would come from SAM and CLIP
    mock_masks = [
        np.ones((100, 100)),  # Sky
        np.zeros((100, 100)),  # Placeholder
        np.ones((100, 100)) * 0.5,  # Tree
        np.ones((100, 100)) * 0.8,  # Ground
    ]
    
    mock_labels = [
        {"name": "sky", "confidence": 0.95},
        {"name": "person", "confidence": 0.89},
        {"name": "tree", "confidence": 0.92},
        {"name": "grass", "confidence": 0.87}
    ]
    
    # Create scene builder and build scene analysis
    builder = SceneBuilder()
    scene_analysis = builder.build(mock_masks, mock_labels)
    
    print("Scene Analysis Results:")
    print(f"Entities: {[e.name for e in scene_analysis.entities]}")
    print(f"Spatial Layout: {scene_analysis.spatial_layout}")
    print(f"Relationships: {[(r.subject_id, r.relationship, r.object_id) for r in scene_analysis.relationships]}")
    print(f"Composition Info: {scene_analysis.composition_info}")
```