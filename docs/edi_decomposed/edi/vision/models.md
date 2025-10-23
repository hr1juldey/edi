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