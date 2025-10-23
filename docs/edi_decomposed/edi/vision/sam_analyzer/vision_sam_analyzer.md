# Vision: SAM Analyzer

[Back to Vision Subsystem](./vision_subsystem.md)

## Purpose
SAM 2.1 wrapper - Contains the SAMAnalyzer class with analyze method that takes an image path and returns a list of masks. Caches model in memory and handles out-of-memory situations by downscaling.

## Class: SAMAnalyzer

### Methods
- `analyze(image_path) -> List[Mask]`: Performs segmentation on the given image and returns a list of masks

### Details
- Caches model in memory for performance
- Handles OOM by downscaling images
- Uses SAM 2.1 for automatic segmentation

## Functions

- [analyze(image_path)](./vision/sam_analyze.md)

## Technology Stack

- SAM 2.1 for segmentation
- PyTorch for model execution
- NumPy for array operations