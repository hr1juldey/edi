# Vision: CLIP Labeler

[Back to Vision Subsystem](./vision_subsystem.md)

## Purpose
CLIP-based entity labeling - Contains the CLIPLabeler class that compares mask regions to text labels via CLIP and returns confidence scores.

## Class: CLIPLabeler

### Methods
- `label_masks(image, masks) -> List[Entity]`: Compares mask regions to text labels via CLIP and returns a list of entities with confidence scores

### Details
- Compares mask regions to text labels via CLIP
- Returns confidence scores for each label
- Uses OpenCLIP for labeling

## Functions

- [label_masks(image, masks)](./vision/clip_label_masks.md)

## Technology Stack

- OpenCLIP (ViT-B/32) for labeling
- PyTorch for model execution
- Pydantic for data validation