# SAMAnalyzer.analyze()

[Back to SAM Analyzer](../vision_sam_analyzer.md)

## Related User Story
"As a user, I want EDI to understand my image's composition so it knows what can be safely edited." (from PRD)

## Function Signature
`analyze(image_path: str) -> List[Mask]`

## Parameters
- `image_path: str` - The file path to the image that needs to be analyzed for objects and segments

## Returns
- `List[Mask]` - A list of mask objects that represent the segmented regions of the image

## Step-by-step Logic
1. Load the image from the provided path using PIL
2. Apply SAM 2.1 automatic segmentation to generate a list of masks
3. For each generated mask: extract bounding box, crop the region, and create embedding vector using CLIP
4. Cluster masks by semantic similarity using CLIP text similarity
5. Label clusters using predefined labels like "sky", "building", "person", etc.
6. Cache the SAM model in memory to avoid reloading for subsequent operations
7. Handle out-of-memory situations by downscaling the image if needed

## Performance Optimizations
- Model caching to avoid reloading
- Image downscaling for large images to reduce processing time
- Noise filtering for masks smaller than 5% of image area

## Input/Output Data Structures
### Mask Object
A Mask object contains:
- Binary mask data
- Bounding box coordinates
- CLIP embedding vector
- Associated label and confidence score