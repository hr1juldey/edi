# Orchestration: Compositor

[Back to Orchestrator](./orchestrator.md)

## Purpose
Region blending - Contains the RegionCompositor class that blends images using Poisson blending for seamless transitions and handles mask feathering.

## Class: RegionCompositor

### Methods
- `blend(images, regions, masks) -> Image`: Blends different regions from multiple images together

### Details
- Uses Poisson blending for seamless transitions
- Handles mask feathering for smooth edges
- Combines selected regions from different variations

## Functions

- [blend(images, regions, masks)](./orchestration/blend.md)

## Technology Stack

- OpenCV for image processing
- NumPy for array operations
- SciPy for blending algorithms