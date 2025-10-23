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