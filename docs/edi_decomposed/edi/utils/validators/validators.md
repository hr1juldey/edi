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

## See Docs

```python
import re
import os
import string
from typing import Union

def validate_prompt(text: str) -> bool:
    """
    Validates if a prompt is properly formatted.
    
    This function:
    - Checks if the prompt is not empty
    - Ensures the prompt is not too long (> 1000 characters)
    - Verifies the prompt contains reasonable characters
    - Checks for any potentially harmful patterns
    """
    if not text or not isinstance(text, str):
        return False
    
    # Check length
    if len(text) > 1000:
        return False
    
    # Check for potentially harmful patterns (basic injection prevention)
    harmful_patterns = [
        r'<script',  # Script tags
        r'javascript:',  # JavaScript URLs
        r'vbscript:',  # VBScript URLs
        r'on\w+\s*=',  # Event handlers
    ]
    
    for pattern in harmful_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return False
    
    # Ensure the prompt contains at least some printable characters
    if not any(c.isprintable() and not c.isspace() for c in text):
        return False
    
    return True

def validate_model_name(name: str) -> bool:
    """
    Validates if a model name is valid.
    
    This function:
    - Checks if the name is not empty
    - Ensures the name contains only valid characters (alphanumeric, hyphens, underscores, dots)
    - Verifies the name does not contain potentially harmful patterns
    """
    if not name or not isinstance(name, str):
        return False
    
    # Check length
    if len(name) < 1 or len(name) > 200:
        return False
    
    # Check for valid characters (alphanumeric, hyphens, underscores, dots)
    if not re.match(r'^[a-zA-Z0-9._-]+, name):
        return False
    
    # Check for potentially problematic patterns
    if '..' in name or name.startswith('.') or name.startswith('-'):
        return False
    
    # Make sure it doesn't contain path traversal sequences
    if '../' in name or '/..' in name:
        return False
    
    return True

def sanitize_filename(name: str) -> str:
    """
    Sanitizes a filename for safe use.
    
    This function:
    - Removes or replaces invalid characters for filenames
    - Ensures the filename is safe to use across different platforms
    - Limits the filename length to prevent issues
    """
    if not name:
        return "unnamed"
    
    # Replace invalid characters across platforms
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        name = name.replace(char, '_')
    
    # Remove control characters
    name = ''.join(char for char in name if ord(char) >= 32 and ord(char) != 127)
    
    # Remove leading/trailing spaces and dots (which have special meaning in some systems)
    name = name.strip(' .')
    
    # Limit length to prevent filesystem issues (255 is common max, but we'll be more conservative)
    if len(name) > 200:
        name, ext = os.path.splitext(name)
        # Keep the extension and first 200 characters of the name
        name = name[:200 - len(ext)] + ext
    
    # If the resulting name is empty, provide a default
    if not name:
        name = "unnamed"
    
    return name

# Example usage and tests
if __name__ == "__main__":
    # Test validate_prompt
    print("Testing validate_prompt:")
    print(f"  'Make the sky blue': {validate_prompt('Make the sky blue')}")
    print(f"  '': {validate_prompt('')}")
    print(f"  None: {validate_prompt(None)}")
    print(f"  '<script>alert</script>': {validate_prompt('<script>alert</script>')}")
    
    # Test validate_model_name
    print("\nTesting validate_model_name:")
    print(f"  'llama-2-7b': {validate_model_name('llama-2-7b')}")
    print(f"  'gpt-3.5-turbo': {validate_model_name('gpt-3.5-turbo')}")
    print(f"  '': {validate_model_name('')}")
    print(f"  '../etc/passwd': {validate_model_name('../etc/passwd')}")
    
    # Test sanitize_filename
    print("\nTesting sanitize_filename:")
    print(f"  'my<file>:name.txt': {sanitize_filename('my<file>:name.txt')}")
    print(f"  'normal_filename.jpg': {sanitize_filename('normal_filename.jpg')}")
    print(f"  'file with spaces & symbols!.png': {sanitize_filename('file with spaces & symbols!.png')}")
    print(f"  '': {sanitize_filename('')}")
    print(f"  '...filename...': {sanitize_filename('...filename...')}")