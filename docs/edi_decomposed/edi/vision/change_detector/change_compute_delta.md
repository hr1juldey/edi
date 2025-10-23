# ChangeDetector.compute_delta()

[Back to Change Detector](../vision_change_detector.md)

## Related User Story
"As a user, I want EDI to check if the edit matches my intent and learn from my corrections." (from PRD)

## Function Signature
`compute_delta(before: SceneAnalysis, after: SceneAnalysis) -> EditDelta`

## Parameters
- `before: SceneAnalysis` - The scene analysis of the original image before editing
- `after: SceneAnalysis` - The scene analysis of the edited image after processing

## Returns
- `EditDelta` - An object that describes the changes between the before and after scenes

## Step-by-step Logic
1. Take the before and after SceneAnalysis objects as input
2. Match entities between the before and after scenes by spatial overlap (IoU > 0.5)
3. For each matched pair of entities:
   - If after entity is None → add to removed entities
   - If entities are similar (by similarity metrics) → add to preserved entities
   - Otherwise → add to modified entities
4. For entities in after scene that weren't in before → add to added entities
5. Calculate alignment score based on preservation and intended changes
6. Return an EditDelta object with preserved, modified, removed, and added entities

## Similarity Metrics
- Color: ΔE2000 < 10 (perceptually similar)
- Position: Center shift < 5% of image dimension
- Shape: Mask IoU > 0.85

## Validation Metrics
- Alignment Score = (0.4 × Entities Preserved Correctly + 0.4 × Intended Changes Applied + 0.2 × (1 - Unintended Changes))

## Input/Output Data Structures
### EditDelta Object
An EditDelta object contains:
- List of preserved entities
- List of modified entities
- List of removed entities
- List of added entities
- Alignment score