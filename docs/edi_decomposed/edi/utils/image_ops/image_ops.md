# Utils: Image Ops

[Back to Index](./index.md)

## Purpose
Image manipulation utilities - Contains functions for resizing images, validating image files, computing image hashes, etc.

## Functions
- `resize_image(image, max_size)`: Resizes images to a maximum size
- `validate_image(path) -> bool`: Validates if a file is a proper image
- `compute_image_hash(path) -> str`: Computes a hash for an image file

### Details
- General purpose image operation utilities
- Used across multiple subsystems
- Provides consistent image handling

## Technology Stack

- Pillow for image processing
- Hash algorithms for image identification