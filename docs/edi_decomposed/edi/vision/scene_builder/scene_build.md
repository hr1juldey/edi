# SceneBuilder.build()

[Back to Scene Builder](../vision_scene_builder.md)

## Related User Story
"As a user, I want EDI to understand my image's composition so it knows what can be safely edited." (from PRD)

## Function Signature
`build(masks, labels) -> SceneAnalysis`

## Parameters
- `masks` - A list of mask objects from the SAM analyzer
- `labels` - A list of labels and entities from the CLIP labeler

## Returns
- `SceneAnalysis` - A comprehensive analysis object that represents the structured understanding of the scene

## Step-by-step Logic
1. Take the masks and labels as input from the previous processing steps
2. Cluster related entities together based on spatial proximity and semantic similarity
3. Compute the spatial layout description (e.g., "sky (top 40%), building (center 55%), grass (bottom 5%)")
4. Group related masks (e.g., person mask includes face, torso, limbs)
5. Create a structured representation of the scene with entities and their relationships
6. Calculate area percentages and spatial relationships between entities
7. Generate a comprehensive SceneAnalysis object that encapsulates all detected elements

## Performance Optimizations
- Efficient clustering algorithms for entity grouping
- Spatial relationship calculations optimized for common layouts
- Memory management during scene assembly

## Input/Output Data Structures
### SceneAnalysis Object
A SceneAnalysis object contains:
- List of Entity objects
- Spatial layout description
- Relationships between entities
- Overall scene composition information