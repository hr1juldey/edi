# Utils: Validators

[Back to Index](./index.md)

## Purpose
Input validation - Contains functions for validating prompts, model names, and sanitizing filenames.

## Functions
- `validate_prompt(text) -> bool`: Validates if a prompt is properly formatted
- `validate_model_name(name) -> bool`: Validates if a model name is valid
- `sanitize_filename(name) -> str`: Sanitizes a filename for safe use

### Details
- Provides input validation across the application
- Prevents invalid inputs from causing errors
- Sanitizes user inputs for security

## Technology Stack

- Input validation libraries
- String manipulation utilities