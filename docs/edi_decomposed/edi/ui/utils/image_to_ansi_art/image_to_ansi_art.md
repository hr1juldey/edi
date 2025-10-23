# image_to_ansi_art()

[Back to UI Utils](../ui_utils.md)

## Related User Story
"As a user, I want to see my images displayed in the terminal interface." (from PRD - implied by TUI requirements)

## Function Signature
`image_to_ansi_art(path, max_width) -> str`

## Parameters
- `path` - The file path to the image to be converted
- `max_width` - The maximum width of the output in terminal characters

## Returns
- `str` - A string containing ANSI escape codes and characters that represent the image visually in the terminal

## Step-by-step Logic
1. Load the image from the provided file path
2. Calculate the appropriate dimensions to fit within max_width while preserving aspect ratio
3. Resize the image to the calculated dimensions
4. Map the image pixels to terminal colors using ANSI color codes
5. Convert the image to a character representation using techniques like half-block characters or braille characters
6. Generate the ANSI escape sequences needed to display colors in the terminal
7. Return the complete string that displays the image when printed to the terminal

## Terminal Display Technology
- Uses ANSI escape codes for color representation
- Employs special Unicode characters for better resolution (half-blocks, braille)
- Optimizes for 80×24 minimum terminal size
- Handles color depth limitations of terminal environments

## Performance Considerations
- Resizes images to prevent excessive character output
- Balances visual quality with terminal performance
- Optimizes for fast rendering in the TUI
- Handles various image formats efficiently

## Input/Output Data Structures
### Input
- Image path: String path to an image file (JPG, PNG, etc.)
- Max width: Integer representing maximum width in terminal characters

### Output
- String containing ANSI escape codes and characters that visually represent the input image when rendered in a terminal

## See Docs

### Python Implementation Example
Implementation of the image_to_ansi_art function:

```python
from PIL import Image
import numpy as np
from typing import Union, Tuple
import os

def image_to_ansi_art(path: str, max_width: int = 80) -> str:
    """
    Convert an image to ANSI art representation for terminal display.
    
    Args:
        path: The file path to the image to be converted
        max_width: The maximum width of the output in terminal characters
    
    Returns:
        String containing ANSI escape codes and characters that represent the image visually in the terminal
    """
    # Validate the input file path exists and is accessible
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found: {path}")
    
    if not os.path.isfile(path):
        raise ValueError(f"Path is not a file: {path}")
    
    # Load the image from the provided file path
    try:
        image = Image.open(path)
    except Exception as e:
        raise ValueError(f"Failed to load image: {str(e)}")
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Calculate the appropriate dimensions to fit within max_width while preserving aspect ratio
    original_width, original_height = image.size
    
    # Account for terminal character aspect ratio (characters are taller than wide)
    # Typical terminal character aspect ratio is about 2:1 (height:width)
    char_aspect_ratio = 2.0
    
    # Calculate new dimensions
    aspect_ratio = original_width / (original_height * char_aspect_ratio)
    new_width = min(max_width, original_width)
    new_height = int(new_width / aspect_ratio)
    
    # Ensure minimum height
    new_height = max(1, new_height)
    
    # Resize the image to the calculated dimensions
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Convert to numpy array for easier processing
    img_array = np.array(resized_image)
    
    # Map the image pixels to terminal colors using ANSI color codes
    ansi_art_lines = []
    
    # Process each row of pixels
    for y in range(new_height):
        line_chars = []
        
        for x in range(new_width):
            # Get RGB values for the pixel
            r, g, b = img_array[y, x]
            
            # Convert RGB to ANSI 256-color code
            ansi_color = _rgb_to_ansi256(r, g, b)
            
            # Use a character that fills the space well
            char = "█"  # Full block character
            
            # Generate the ANSI escape sequences needed to display colors in the terminal
            ansi_escape = f"\033[38;5;{ansi_color}m{char}\033[0m"
            line_chars.append(ansi_escape)
        
        # Join characters for this line
        ansi_art_lines.append("".join(line_chars))
    
    # Return the complete string that displays the image when printed to the terminal
    return "\n".join(ansi_art_lines)

def _rgb_to_ansi256(r: int, g: int, b: int) -> int:
    """
    Convert RGB values to ANSI 256-color code.
    
    Args:
        r, g, b: RGB values (0-255)
    
    Returns:
        ANSI 256-color code (0-255)
    """
    # Simple RGB to ANSI 256 conversion
    # For more accurate conversion, consider using a proper algorithm
    if r == g == b:
        # Grayscale
        if r < 8:
            return 16
        elif r > 248:
            return 231
        else:
            return round(((r - 8) / 247) * 24) + 232
    else:
        # Color cube conversion
        r = round((r / 255) * 5)
        g = round((g / 255) * 5)
        b = round((b / 255) * 5)
        return 16 + (36 * r) + (6 * g) + b

def image_to_ansi_art_optimized(path: str, 
                               max_width: int = 80,
                               use_half_blocks: bool = True,
                               color_depth: str = "256") -> str:
    """
    Optimized version with half-block characters and configurable color depth.
    
    Args:
        path: The file path to the image to be converted
        max_width: The maximum width of the output in terminal characters
        use_half_blocks: Whether to use half-block characters for better resolution
        color_depth: Color depth - "16", "256", or "truecolor"
    
    Returns:
        String containing ANSI escape codes and characters that represent the image visually in the terminal
    """
    # Validate the input file path exists and is accessible
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found: {path}")
    
    if not os.path.isfile(path):
        raise ValueError(f"Path is not a file: {path}")
    
    # Load the image from the provided file path
    try:
        image = Image.open(path)
    except Exception as e:
        raise ValueError(f"Failed to load image: {str(e)}")
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Calculate the appropriate dimensions
    original_width, original_height = image.size
    
    # Adjust for character aspect ratio and half-blocks
    char_aspect_ratio = 2.0
    if use_half_blocks:
        # Half blocks allow for double vertical resolution
        char_aspect_ratio = 1.0
    
    # Calculate new dimensions
    aspect_ratio = original_width / (original_height * char_aspect_ratio)
    new_width = min(max_width, original_width)
    new_height = int((new_width / aspect_ratio) * (0.5 if use_half_blocks else 1.0))
    
    # Ensure minimum dimensions
    new_width = max(1, new_width)
    new_height = max(1, new_height)
    
    # Resize the image
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    img_array = np.array(resized_image)
    
    if use_half_blocks:
        # Use half-block characters for better resolution
        return _generate_half_block_ansi_art(img_array, color_depth)
    else:
        # Use standard block characters
        return _generate_standard_ansi_art(img_array, color_depth)

def _generate_half_block_ansi_art(img_array: np.ndarray, color_depth: str) -> str:
    """
    Generate ANSI art using half-block characters for better resolution.
    """
    height, width, _ = img_array.shape
    lines = []
    
    # Process pairs of rows
    for y in range(0, height, 2):
        line_chars = []
        
        for x in range(width):
            # Upper pixel
            r1, g1, b1 = img_array[y, x]
            
            # Lower pixel (if exists)
            if y + 1 < height:
                r2, g2, b2 = img_array[y + 1, x]
            else:
                # Use background color for lower half if no pixel
                r2, g2, b2 = r1, g1, b1
            
            # Convert to ANSI colors
            if color_depth == "16":
                fg_color = _rgb_to_ansi16(r1, g1, b1)
                bg_color = _rgb_to_ansi16(r2, g2, b2)
                char = "▀"  # Upper half block
                ansi_escape = f"\033[38;5;{fg_color};48;5;{bg_color}m{char}\033[0m"
            elif color_depth == "256":
                fg_color = _rgb_to_ansi256(r1, g1, b1)
                bg_color = _rgb_to_ansi256(r2, g2, b2)
                char = "▀"  # Upper half block
                ansi_escape = f"\033[38;5;{fg_color};48;5;{bg_color}m{char}\033[0m"
            else:  # truecolor
                char = "▀"  # Upper half block
                ansi_escape = f"\033[38;2;{r1};{g1};{b1};48;2;{r2};{g2};{b2}m{char}\033[0m"
            
            line_chars.append(ansi_escape)
        
        lines.append("".join(line_chars))
    
    return "\n".join(lines)

def _generate_standard_ansi_art(img_array: np.ndarray, color_depth: str) -> str:
    """
    Generate standard ANSI art using full block characters.
    """
    height, width, _ = img_array.shape
    lines = []
    
    for y in range(height):
        line_chars = []
        
        for x in range(width):
            r, g, b = img_array[y, x]
            
            # Convert to ANSI colors
            if color_depth == "16":
                ansi_color = _rgb_to_ansi16(r, g, b)
                char = "█"
                ansi_escape = f"\033[38;5;{ansi_color}m{char}\033[0m"
            elif color_depth == "256":
                ansi_color = _rgb_to_ansi256(r, g, b)
                char = "█"
                ansi_escape = f"\033[38;5;{ansi_color}m{char}\033[0m"
            else:  # truecolor
                char = "█"
                ansi_escape = f"\033[38;2;{r};{g};{b}m{char}\033[0m"
            
            line_chars.append(ansi_escape)
        
        lines.append("".join(line_chars))
    
    return "\n".join(lines)

def _rgb_to_ansi16(r: int, g: int, b: int) -> int:
    """
    Convert RGB to nearest ANSI 16-color code.
    """
    # Simple conversion based on brightness and hue
    brightness = (r + g + b) / 3
    
    if brightness < 48:
        return 30  # Black
    elif brightness < 115:
        if r > g and r > b:
            return 31  # Red
        elif g > r and g > b:
            return 32  # Green
        elif b > r and b > g:
            return 34  # Blue
        else:
            return 33  # Yellow
    else:
        if r > g and r > b:
            return 91  # Bright Red
        elif g > r and g > b:
            return 92  # Bright Green
        elif b > r and b > g:
            return 94  # Bright Blue
        else:
            return 93  # Bright Yellow

# Example usage
if __name__ == "__main__":
    # Example usage (commented out since we don't have an actual image file)
    try:
        # This would work with an actual image file
        # ansi_art = image_to_ansi_art("example.jpg", max_width=40)
        # print(ansi_art)
        
        print("ANSI Art Generator")
        print("==================")
        print("To use this function, provide a path to an image file.")
        print("Example: image_to_ansi_art('path/to/image.jpg', max_width=80)")
        
    except Exception as e:
        print(f"Error: {e}")
```

### Advanced ANSI Art Implementation
Enhanced implementation with additional features:

```python
from PIL import Image, ImageOps
import numpy as np
from typing import Union, Tuple, Optional
import os
import sys

class ANSIArtConverter:
    """
    Advanced ANSI art converter with multiple rendering modes and optimizations.
    """
    
    def __init__(self):
        # Character sets for different rendering styles
        self.character_sets = {
            "blocks": "█▓▒░ ",  # From darkest to lightest
            "ascii": "@%#*+=-:. ",  # ASCII characters from darkest to lightest
            "braille": "⣿⣶⣤⣀⡀ ",  # Braille characters for high resolution
            "simple": "██░░ "  # Simple block characters
        }
        
        # ANSI color codes for 16-color mode
        self.ansi16_colors = {
            "black": 30, "red": 31, "green": 32, "yellow": 33,
            "blue": 34, "magenta": 35, "cyan": 36, "white": 37,
            "bright_black": 90, "bright_red": 91, "bright_green": 92,
            "bright_yellow": 93, "bright_blue": 94, "bright_magenta": 95,
            "bright_cyan": 96, "bright_white": 97
        }
    
    def convert(self, 
                path: str, 
                max_width: int = 80,
                render_mode: str = "half_blocks",
                color_mode: str = "256",
                character_set: str = "blocks",
                invert: bool = False,
                dither: bool = False) -> str:
        """
        Convert image to ANSI art with advanced options.
        
        Args:
            path: Path to the image file
            max_width: Maximum width in terminal characters
            render_mode: "half_blocks", "full_blocks", "characters", "braille"
            color_mode: "16", "256", "truecolor"
            character_set: Character set to use for rendering
            invert: Whether to invert the image colors
            dither: Whether to apply dithering for better quality
        
        Returns:
            ANSI art string
        """
        # Validate input
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image file not found: {path}")
        
        # Load and preprocess image
        image = self._load_image(path)
        
        # Apply preprocessing
        if invert:
            image = ImageOps.invert(image.convert('RGB'))
        
        # Calculate dimensions
        new_width, new_height = self._calculate_dimensions(image, max_width, render_mode)
        
        # Resize image
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Apply dithering if requested
        if dither:
            resized_image = self._apply_dithering(resized_image)
        
        # Convert to ANSI art
        if render_mode == "half_blocks":
            return self._render_half_blocks(resized_image, color_mode)
        elif render_mode == "full_blocks":
            return self._render_full_blocks(resized_image, color_mode, character_set)
        elif render_mode == "characters":
            return self._render_characters(resized_image, color_mode, character_set)
        elif render_mode == "braille":
            return self._render_braille(resized_image, color_mode)
        else:
            raise ValueError(f"Unsupported render mode: {render_mode}")
    
    def _load_image(self, path: str) -> Image.Image:
        """Load image with error handling."""
        try:
            image = Image.open(path)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return image
        except Exception as e:
            raise ValueError(f"Failed to load image '{path}': {str(e)}")
    
    def _calculate_dimensions(self, 
                            image: Image.Image, 
                            max_width: int, 
                            render_mode: str) -> Tuple[int, int]:
        """Calculate appropriate dimensions for rendering."""
        original_width, original_height = image.size
        
        # Adjust for character aspect ratio
        # Terminal characters are typically 2:1 height:width
        char_aspect_ratio = 2.0
        
        # Adjust for render mode
        if render_mode == "half_blocks":
            # Half blocks provide double vertical resolution
            char_aspect_ratio = 1.0
        elif render_mode == "braille":
            # Braille characters are 2x4 dots
            char_aspect_ratio = 0.5
        
        # Calculate new dimensions
        aspect_ratio = original_width / (original_height * char_aspect_ratio)
        new_width = min(max_width, original_width)
        new_height = int((new_width / aspect_ratio) * (0.5 if render_mode == "half_blocks" else 1.0))
        
        # Ensure minimum dimensions
        new_width = max(1, new_width)
        new_height = max(1, new_height)
        
        return new_width, new_height
    
    def _apply_dithering(self, image: Image.Image) -> Image.Image:
        """Apply Floyd-Steinberg dithering."""
        # Convert to grayscale for dithering
        grayscale = image.convert('L')
        
        # Apply Floyd-Steinberg dithering
        dithered = grayscale.convert('1')
        
        # Convert back to RGB
        return dithered.convert('RGB')
    
    def _render_half_blocks(self, image: Image.Image, color_mode: str) -> str:
        """Render using half-block characters for better resolution."""
        img_array = np.array(image)
        height, width, _ = img_array.shape
        lines = []
        
        # Process pairs of rows
        for y in range(0, height, 2):
            line_chars = []
            
            for x in range(width):
                # Upper pixel
                r1, g1, b1 = img_array[y, x]
                
                # Lower pixel (if exists)
                if y + 1 < height:
                    r2, g2, b2 = img_array[y + 1, x]
                else:
                    # Use background color for lower half if no pixel
                    r2, g2, b2 = r1, g1, b1
                
                # Generate ANSI escape sequence
                if color_mode == "16":
                    fg_color = self._rgb_to_ansi16(r1, g1, b1)
                    bg_color = self._rgb_to_ansi16(r2, g2, b2)
                    char = "▀"  # Upper half block
                    ansi_escape = f"\033[38;5;{fg_color};48;5;{bg_color}m{char}\033[0m"
                elif color_mode == "256":
                    fg_color = self._rgb_to_ansi256(r1, g1, b1)
                    bg_color = self._rgb_to_ansi256(r2, g2, b2)
                    char = "▀"  # Upper half block
                    ansi_escape = f"\033[38;5;{fg_color};48;5;{bg_color}m{char}\033[0m"
                else:  # truecolor
                    char = "▀"  # Upper half block
                    ansi_escape = f"\033[38;2;{r1};{g1};{b1};48;2;{r2};{g2};{b2}m{char}\033[0m"
                
                line_chars.append(ansi_escape)
            
            lines.append("".join(line_chars))
        
        return "\n".join(lines)
    
    def _render_full_blocks(self, image: Image.Image, color_mode: str, character_set: str) -> str:
        """Render using full block characters."""
        img_array = np.array(image)
        height, width, _ = img_array.shape
        lines = []
        
        chars = self.character_sets.get(character_set, self.character_sets["blocks"])
        
        for y in range(height):
            line_chars = []
            
            for x in range(width):
                r, g, b = img_array[y, x]
                
                # Calculate brightness for character selection
                brightness = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
                char_index = int(brightness * (len(chars) - 1))
                char = chars[char_index]
                
                # Generate ANSI escape sequence
                if color_mode == "16":
                    ansi_color = self._rgb_to_ansi16(r, g, b)
                    ansi_escape = f"\033[38;5;{ansi_color}m{char}\033[0m"
                elif color_mode == "256":
                    ansi_color = self._rgb_to_ansi256(r, g, b)
                    ansi_escape = f"\033[38;5;{ansi_color}m{char}\033[0m"
                else:  # truecolor
                    ansi_escape = f"\033[38;2;{r};{g};{b}m{char}\033[0m"
                
                line_chars.append(ansi_escape)
            
            lines.append("".join(line_chars))
        
        return "\n".join(lines)
    
    def _render_characters(self, image: Image.Image, color_mode: str, character_set: str) -> str:
        """Render using character mapping."""
        return self._render_full_blocks(image, color_mode, character_set)
    
    def _render_braille(self, image: Image.Image, color_mode: str) -> str:
        """Render using braille characters for high resolution."""
        # This is a simplified implementation
        img_array = np.array(image)
        height, width, _ = img_array.shape
        lines = []
        
        # Process in 4x2 blocks for braille dots
        for y in range(0, height, 4):
            line_chars = []
            
            for x in range(0, width, 2):
                # Create braille character based on 8-dot pattern
                dots = [False] * 8
                
                # Fill dots based on pixel values
                for dy in range(min(4, height - y)):
                    for dx in range(min(2, width - x)):
                        if y + dy < height and x + dx < width:
                            r, g, b = img_array[y + dy, x + dx]
                            brightness = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
                            # Simple threshold for dot activation
                            dots[dy * 2 + dx] = brightness > 0.5
                
                # Convert dots to braille character
                braille_char = self._dots_to_braille(dots)
                
                # Use average color of the block
                block_pixels = []
                for dy in range(min(4, height - y)):
                    for dx in range(min(2, width - x)):
                        if y + dy < height and x + dx < width:
                            block_pixels.append(img_array[y + dy, x + dx])
                
                if block_pixels:
                    avg_r = int(np.mean([p[0] for p in block_pixels]))
                    avg_g = int(np.mean([p[1] for p in block_pixels]))
                    avg_b = int(np.mean([p[2] for p in block_pixels]))
                    
                    # Generate ANSI escape sequence
                    if color_mode == "16":
                        ansi_color = self._rgb_to_ansi16(avg_r, avg_g, avg_b)
                        ansi_escape = f"\033[38;5;{ansi_color}m{braille_char}\033[0m"
                    elif color_mode == "256":
                        ansi_color = self._rgb_to_ansi256(avg_r, avg_g, avg_b)
                        ansi_escape = f"\033[38;5;{ansi_color}m{braille_char}\033[0m"
                    else:  # truecolor
                        ansi_escape = f"\033[38;2;{avg_r};{avg_g};{avg_b}m{braille_char}\033[0m"
                else:
                    ansi_escape = braille_char
                
                line_chars.append(ansi_escape)
            
            lines.append("".join(line_chars))
        
        return "\n".join(lines)
    
    def _dots_to_braille(self, dots: list) -> str:
        """Convert 8-dot pattern to braille character."""
        # Braille Unicode starts at U+2800
        braille_base = 0x2800
        
        # Map dots to braille bits
        # Positions: 1 4    0 3
        #           2 5 -> 1 4
        #           3 6    2 5
        #           7 8    6 7
        bit_positions = [0, 1, 2, 6, 3, 4, 5, 7]
        
        value = 0
        for i, dot in enumerate(dots):
            if dot:
                value |= (1 << bit_positions[i])
        
        return chr(braille_base + value)
    
    def _rgb_to_ansi16(self, r: int, g: int, b: int) -> int:
        """Convert RGB to nearest ANSI 16-color code."""
        # Simple conversion based on brightness and hue
        brightness = (r + g + b) / 3
        
        if brightness < 48:
            return 30  # Black
        elif brightness < 115:
            if r > g and r > b:
                return 31  # Red
            elif g > r and g > b:
                return 32  # Green
            elif b > r and b > g:
                return 34  # Blue
            else:
                return 33  # Yellow
        else:
            if r > g and r > b:
                return 91  # Bright Red
            elif g > r and g > b:
                return 92  # Bright Green
            elif b > r and b > g:
                return 94  # Bright Blue
            else:
                return 93  # Bright Yellow
    
    def _rgb_to_ansi256(self, r: int, g: int, b: int) -> int:
        """Convert RGB values to ANSI 256-color code."""
        # Simple RGB to ANSI 256 conversion
        if r == g == b:
            # Grayscale
            if r < 8:
                return 16
            elif r > 248:
                return 231
            else:
                return round(((r - 8) / 247) * 24) + 232
        else:
            # Color cube conversion
            r = round((r / 255) * 5)
            g = round((g / 255) * 5)
            b = round((b / 255) * 5)
            return 16 + (36 * r) + (6 * g) + b

# Example usage
if __name__ == "__main__":
    converter = ANSIArtConverter()
    
    # Example usage information
    print("ANSI Art Converter")
    print("==================")
    print("To convert an image to ANSI art:")
    print("converter.convert('path/to/image.jpg', max_width=80, render_mode='half_blocks')")
    print()
    print("Available options:")
    print("- render_mode: 'half_blocks', 'full_blocks', 'characters', 'braille'")
    print("- color_mode: '16', '256', 'truecolor'")
    print("- character_set: 'blocks', 'ascii', 'braille', 'simple'")
```