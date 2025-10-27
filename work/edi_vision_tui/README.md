# Adaptive Mask Generator for Precise Image Editing

This application implements an advanced mask generation system that combines computer vision and AI to create precise masks for image editing tasks.

## Features

- Extracts entities and edit requests from natural language prompts
- Uses a multi-stage approach with CLIP, YOLO, and SAM for mask generation
- Validates results with VLM to ensure accurate targeting
- Implements iterative refinement with debug logs
- Applies masks to images for visualization

## Usage

### Basic Usage
```bash
python app.py --image input.jpg --prompt "change orange roofs to blue" --output mask.png
```

### With Mask Application
```bash
python app.py --image input.jpg --prompt "change orange roofs to blue" --output mask.png --apply-to-image
```

### With Verbose Logging
```bash
python app.py --image input.jpg --prompt "change orange roofs to blue" --output mask.png --apply-to-image --verbose
```

## File Structure

- `app.py` - Main application with adaptive mask generation
- `test_image.jpeg` - Original test image
- `result_adaptive.png` - Reference result image
- Model files (.pt) - Pre-trained models for SAM, YOLO, CLIP

## Key Components

1. **Entity Extraction**: Parses user prompts to identify target entities and edit requests
2. **Multi-Stage Masking**: Uses CLIP for semantic matching, YOLO for object detection, and SAM for precise segmentation
3. **VLM Validation**: Uses Vision-Language Models to validate that the generated mask matches the intended edit
4. **Iterative Refinement**: Attempts up to 3 times with validation feedback to ensure quality
5. **Post-processing**: Applies morphological operations to clean up the final mask

## Output

- `mask.png`: The generated binary mask
- `superimposed_mask.png`: The original image with the mask overlaid in red (when using `--apply-to-image`)

## Logging

The application provides detailed logs when using the `--verbose` flag, showing:
- Entity extraction results
- Each stage of the mask generation process
- Validation results
- Final statistics

## Requirements

- Python 3.12+
- PyTorch
- OpenCLIP
- Ultralytics
- OpenCV
- Pillow
- Requests (for VLM validation)

## Note

This application requires running Ollama locally with a vision-language model available at `http://localhost:11434` for VLM validation.