# Vision: Change Detector

[Back to Vision Subsystem](./vision_subsystem.md)

## Purpose
Before/after comparison - Contains the ChangeDetector class that matches entities by IoU and calculates alignment scores.

## Class: ChangeDetector

### Methods
- `compute_delta(before, after) -> EditDelta`: Compares before and after SceneAnalysis objects and returns an EditDelta

### Details
- Matches entities by Intersection over Union (IoU)
- Calculates alignment scores for validation
- Identifies preserved, modified, removed, and added entities

## Functions

- [compute_delta(before, after)](./vision/change_compute_delta.md)

## Technology Stack

- NumPy for array operations
- Pydantic for data validation

## See Docs

```python
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import colorsys
from scipy.spatial.distance import euclidean
from shapely.geometry import Polygon
import cv2
from ...vision.models import SceneAnalysis, Entity, EditDelta

class ChangeDetector:
    """
    Before/after comparison - Contains methods to match entities by IoU 
    and calculate alignment scores.
    """
    
    def __init__(self, iou_threshold: float = 0.5, similarity_threshold: float = 0.85):
        self.iou_threshold = iou_threshold
        self.similarity_threshold = similarity_threshold

    def compute_delta(self, before: SceneAnalysis, after: SceneAnalysis) -> EditDelta:
        """
        Compares before and after SceneAnalysis objects and returns an EditDelta.
        
        This method:
        - Matches entities by Intersection over Union (IoU)
        - Calculates alignment scores for validation
        - Identifies preserved, modified, removed, and added entities
        """
        # Match entities between before and after scenes
        matched_pairs, unmatched_before, unmatched_after = self._match_entities(
            before.entities, after.entities
        )
        
        # Initialize lists for different types of changes
        preserved_entities = []
        modified_entities = []
        removed_entities = unmatched_before  # All unmatched before entities are removed
        added_entities = unmatched_after    # All unmatched after entities are added
        
        # Analyze matched pairs to determine if they're preserved or modified
        for before_entity, after_entity in matched_pairs:
            if self._are_similar(before_entity, after_entity):
                preserved_entities.append(after_entity)  # Use the after entity as it's current
            else:
                modified_entities.append(after_entity)  # Modified after entity
        
        # Calculate metrics for alignment scoring
        total_before_entities = len(before.entities)
        preserved_count = len(preserved_entities)
        modified_count = len(modified_entities)
        removed_count = len(removed_entities)
        added_count = len(added_entities)
        
        # Calculate alignment score components
        entities_preserved_correctly = preserved_count / total_before_entities if total_before_entities > 0 else 1.0
        # For simplicity in this example, we'll consider modified entities as applied changes
        # In a real implementation, this would be based on the original intent
        intended_changes_applied = (modified_count + preserved_count) / total_before_entities if total_before_entities > 0 else 1.0
        unintended_changes = (removed_count + added_count) / total_before_entities if total_before_entities > 0 else 0.0
        
        # Calculate alignment score using the specified formula
        # Alignment Score = (0.4 × Entities Preserved Correctly + 0.4 × Intended Changes Applied + 0.2 × (1 - Unintended Changes))
        alignment_score = (
            0.4 * entities_preserved_correctly +
            0.4 * intended_changes_applied +
            0.2 * (1 - unintended_changes)
        )
        
        return EditDelta(
            preserved_entities=preserved_entities,
            modified_entities=modified_entities,
            removed_entities=removed_entities,
            added_entities=added_entities,
            alignment_score=alignment_score,
            entities_preserved_correctly=entities_preserved_correctly,
            intended_changes_applied=intended_changes_applied,
            unintended_changes=unintended_changes
        )
    
    def _match_entities(self, before_entities: List[Entity], after_entities: List[Entity]):
        """
        Match entities between before and after scenes based on spatial overlap (IoU).
        Returns matched pairs and unmatched entities.
        """
        matched_pairs = []
        unmatched_before = before_entities.copy()
        unmatched_after = after_entities.copy()
        
        # Compare each before entity with each after entity
        for before_entity in before_entities[:]:  # Use slice to avoid modification during iteration
            best_match_idx = None
            best_iou = 0
            
            for i, after_entity in enumerate(after_entities):
                iou = self._calculate_iou(before_entity.bbox, after_entity.bbox)
                
                if iou > self.iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_match_idx = i
            
            if best_match_idx is not None:
                # Found a match
                matched_pairs.append((before_entity, after_entities[best_match_idx]))
                # Remove from unmatched lists
                if before_entity in unmatched_before:
                    unmatched_before.remove(before_entity)
                if after_entities[best_match_idx] in unmatched_after:
                    unmatched_after.remove(after_entities[best_match_idx])
        
        return matched_pairs, unmatched_before, unmatched_after
    
    def _calculate_iou(self, bbox1: tuple, bbox2: tuple) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        Each bbox is (x1, y1, x2, y2).
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_int = max(x1_1, x1_2)
        y1_int = max(y1_1, y1_2)
        x2_int = min(x2_1, x2_2)
        y2_int = min(y2_1, y2_1)
        
        # Calculate intersection area
        intersection_area = max(0, x2_int - x1_int) * max(0, y2_int - y1_int)
        
        # Calculate areas of both boxes
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Calculate union area
        union_area = area1 + area2 - intersection_area
        
        # Return IoU
        return intersection_area / union_area if union_area > 0 else 0.0
    
    def _are_similar(self, entity1: Entity, entity2: Entity) -> bool:
        """
        Determine if two entities are similar based on color, position, and shape metrics.
        
        Similarity Metrics:
        - Color: ΔE2000 < 10 (perceptually similar)
        - Position: Center shift < 5% of image dimension
        - Shape: Mask IoU > 0.85
        """
        # Check color similarity if both have color information
        if entity1.color and entity2.color:
            color_distance = self._color_distance(entity1.color, entity2.color)
            if color_distance >= 10:  # ΔE2000 >= 10 means not similar
                return False
        
        # Check position similarity (centroid shift)
        if entity1.centroid and entity2.centroid:
            max_shift = max(entity1.centroid[0], entity1.centroid[1]) * 0.05  # 5% of image dimension
            position_distance = euclidean(entity1.centroid, entity2.centroid)
            if position_distance > max_shift:
                return False
        
        # Check shape similarity if both have masks
        if hasattr(entity1, 'mask') and hasattr(entity2, 'mask') and entity1.mask is not None and entity2.mask is not None:
            mask_iou = self._calculate_mask_iou(entity1.mask, entity2.mask)
            if mask_iou < 0.85:  # Shape IoU threshold
                return False
        
        return True
    
    def _color_distance(self, color1: tuple, color2: tuple) -> float:
        """
        Calculate perceptual color distance using delta E (ΔE2000) approximation.
        This is a simplified implementation; a full ΔE2000 implementation is complex.
        """
        # Convert RGB to LAB for better perceptual distance
        # For simplicity, using a Euclidean distance in RGB space as an approximation
        return euclidean(color1, color2)
    
    def _calculate_mask_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """
        Calculate IoU between two binary masks.
        """
        # Ensure masks are binary
        mask1 = (mask1 > 0).astype(np.uint8)
        mask2 = (mask2 > 0).astype(np.uint8)
        
        # Calculate intersection and union
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        
        return intersection / union if union > 0 else 0.0

# Example usage:
if __name__ == "__main__":
    # Create mock entities for before scene (using the models we defined)
    from ...vision.models import Entity as EntityModel
    
    before_entities = [
        EntityModel(
            id="sky_1", 
            name="sky", 
            bbox=(0, 0, 500, 200), 
            confidence=0.9,
            color=(135, 206, 235), centroid=(250, 100), area=0.4
        ),
        EntityModel(
            id="tree_1", 
            name="tree", 
            bbox=(100, 200, 200, 400),
            confidence=0.85,
            color=(34, 139, 34), centroid=(150, 300), area=0.15
        ),
        EntityModel(
            id="mountain_1", 
            name="mountain", 
            bbox=(300, 200, 500, 400),
            confidence=0.92,
            color=(105, 105, 105), centroid=(400, 300), area=0.25
        )
    ]
    
    after_entities = [
        EntityModel(
            id="sky_1", 
            name="sky", 
            bbox=(0, 0, 500, 200),
            confidence=0.88,
            color=(255, 165, 0), centroid=(250, 100), area=0.4  # Changed: more dramatic orange color
        ),
        EntityModel(
            id="tree_1", 
            name="tree", 
            bbox=(100, 200, 200, 400),
            confidence=0.85,
            color=(34, 139, 34), centroid=(150, 300), area=0.15  # Preserved
        ),
        EntityModel(
            id="mountain_1", 
            name="mountain", 
            bbox=(300, 200, 500, 400),
            confidence=0.89,
            color=(169, 169, 169), centroid=(400, 300), area=0.25  # Changed: lighter gray
        )
    ]
    
    # Create scene analyses
    before_scene = SceneAnalysis(
        entities=before_entities, 
        spatial_layout="mountains in background, trees in middle ground, sky above"
    )
    after_scene = SceneAnalysis(
        entities=after_entities, 
        spatial_layout="mountains in background, trees in middle ground, sky above"
    )
    
    # Create change detector and compute delta
    detector = ChangeDetector()
    delta = detector.compute_delta(before_scene, after_scene)
    
    print("Edit Delta Results:")
    print(f"Preserved entities: {[e.name for e in delta.preserved_entities]}")
    print(f"Modified entities: {[e.name for e in delta.modified_entities]}")
    print(f"Removed entities: {[e.name for e in delta.removed_entities]}")
    print(f"Added entities: {[e.name for e in delta.added_entities]}")
    print(f"Alignment Score: {delta.alignment_score:.2f}")