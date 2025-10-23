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