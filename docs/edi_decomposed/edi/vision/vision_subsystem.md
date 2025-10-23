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