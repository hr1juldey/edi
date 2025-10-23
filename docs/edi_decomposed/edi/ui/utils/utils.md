# UI: Utils

[Back to TUI Layer](./tui_layer.md)

## Purpose
UI utilities - Contains helper functions for UI operations like image to ANSI art conversion, duration formatting, etc.

## Functions
- `image_to_ansi_art(path, max_width) -> str`: Converts images to ANSI art representation
- `format_duration(seconds) -> "1m 23s"`: Formats time durations for display
- `color_code_score(score) -> Rich markup`: Applies color coding to scores

### Details
- Helper functions for UI rendering
- Terminal-specific formatting utilities
- Provides visual enhancements

## Functions

- [image_to_ansi_art(path, max_width)](./ui_utils/image_to_ansi_art.md)
- [format_duration(seconds)](./ui_utils/format_duration.md)
- [color_code_score(score)](./ui_utils/color_code_score.md)

## Technology Stack

- Rich for terminal formatting
- Pillow for image processing
- Time utilities

## See Docs

### Rich Implementation Example
UI utilities implementation for the EDI application:

```python
from rich.console import Console
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich import print as rprint
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging
from datetime import datetime, timedelta
import time

class RichUIUtils:
    """
    UI utilities using Rich for terminal formatting.
    """
    
    def __init__(self):
        self.console = Console()
        self.logger = logging.getLogger(__name__)
    
    def image_to_ansi_art(self, path: str, max_width: int = 80) -> str:
        """
        Converts images to ANSI art representation.
        
        Args:
            path: Path to the image to convert
            max_width: Maximum width in terminal characters
            
        Returns:
            String containing ANSI art representation
        """
        # Validate input parameters
        if not Path(path).exists():
            raise FileNotFoundError(f"Image file not found: {path}")
        
        # Load image using Pillow
        try:
            from PIL import Image
            image = Image.open(path)
        except ImportError:
            raise ImportError("Pillow is required for image processing")
        except Exception as e:
            raise ValueError(f"Failed to load image: {str(e)}")
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Calculate appropriate dimensions to fit within max_width while preserving aspect ratio
        original_width, original_height = image.size
        char_aspect_ratio = 2.0  # Terminal characters are typically 2:1 height:width
        
        # Calculate new dimensions
        aspect_ratio = original_width / (original_height * char_aspect_ratio)
        new_width = min(max_width, original_width)
        new_height = int((new_width / aspect_ratio) / 2)  # Divide by 2 for character aspect ratio
        
        # Ensure minimum dimensions
        new_width = max(1, new_width)
        new_height = max(1, new_height)
        
        # Resize image to calculated dimensions
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert image pixels to terminal characters using ANSI color codes
        img_array = list(resized_image.getdata())
        ansi_art_lines = []
        
        # Character set for different intensities (from darkest to lightest)
        chars = " .:-=+*#%@"
        
        # Process each row
        for y in range(new_height):
            line_chars = []
            
            for x in range(new_width):
                # Get RGB values for the pixel
                r, g, b = img_array[y * new_width + x]
                
                # Calculate brightness for character selection
                brightness = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
                char_index = int(brightness * (len(chars) - 1))
                char = chars[char_index]
                
                # Generate ANSI escape sequences to display colors in the terminal
                ansi_escape = f"\\033[38;2;{r};{g};{b}m{char}\\033[0m"
                line_chars.append(ansi_escape)
            
            # Join characters for this line
            ansi_art_lines.append("".join(line_chars))
        
        # Return complete string that displays the image when printed to the terminal
        return "\\n".join(ansi_art_lines)
    
    def format_duration(self, seconds: float) -> str:
        """
        Formats time durations for display.
        
        Args:
            seconds: Time duration in seconds
            
        Returns:
            String representation like "1m 23s", "2h 5m", "45s"
        """
        # Validate input
        if not isinstance(seconds, (int, float)) or seconds < 0:
            raise ValueError(f"Seconds must be a non-negative number, got {seconds}")
        
        # Convert to integer for calculation
        total_seconds = int(seconds)
        
        # Calculate hours, minutes, and seconds
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        
        # Format the values into a human-readable string
        components = []
        
        # Add hours if significant
        if hours > 0:
            components.append(f"{hours}h")
        
        # Add minutes if significant or if we have hours
        if minutes > 0 or hours > 0:
            components.append(f"{minutes}m")
        
        # Add seconds if significant or if no other components
        if secs > 0 or len(components) == 0:
            components.append(f"{secs}s")
        
        # Omit zero-value components to keep the output concise
        # (already handled by conditional logic above)
        
        # Return the formatted duration string
        return " ".join(components)
    
    def color_code_score(self, score: float) -> str:
        """
        Applies color coding to scores.
        
        Args:
            score: Numeric score between 0 and 1
            
        Returns:
            Rich markup string that displays the score with appropriate color coding
        """
        # Validate input
        if not isinstance(score, (int, float)) or not (0 <= score <= 1):
            raise ValueError(f"Score must be between 0 and 1, got {score}")
        
        # Determine the appropriate color based on the score range
        if score >= 0.8:
            # High score (>0.8): Green (indicates good quality)
            color = "green"
        elif score >= 0.6:
            # Medium score (0.6-0.8): Yellow (indicates moderate quality)
            color = "yellow"
        else:
            # Low score (<0.6): Red (indicates poor quality)
            color = "red"
        
        # Create Rich markup that combines the score text with the appropriate color
        # Use appropriate symbols along with the color coding
        if score >= 0.8:
            symbol = "✓"  # Checkmark for high scores
        elif score >= 0.6:
            symbol = "⚠"  # Warning symbol for medium scores
        else:
            symbol = "✗"  # Cross for low scores
        
        # Return the colored markup string that will display with color when rendered
        return f"[{color}]{symbol} {score:.2f}[/{color}]"
    
    def display_image_preview(self, image_path: str, max_width: int = 40) -> None:
        """
        Display an image preview using ANSI art.
        
        Args:
            image_path: Path to the image to preview
            max_width: Maximum width in terminal characters
        """
        try:
            # Convert image to ANSI art
            ansi_art = self.image_to_ansi_art(image_path, max_width)
            
            # Display in a panel
            panel = Panel(
                ansi_art,
                title=f"[bold]Image Preview: {Path(image_path).name}[/bold]",
                border_style="blue",
                expand=False
            )
            
            self.console.print(panel)
            
        except Exception as e:
            self.logger.error(f"Failed to display image preview: {str(e)}")
            self.console.print(f"[red]Error displaying image preview: {str(e)}[/red]")
    
    def display_progress_bar(self, total: int = 100, description: str = "Processing") -> Progress:
        """
        Display a progress bar with Rich formatting.
        
        Args:
            total: Total number of steps
            description: Description of the process
            
        Returns:
            Progress object for updating
        """
        progress = Progress(
            TextColumn(f"[bold blue]{description}"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=self.console
        )
        
        return progress
    
    def display_session_summary(self, session_data: Dict[str, Any]) -> None:
        """
        Display session summary with Rich formatting.
        
        Args:
            session_data: Dictionary containing session information
        """
        # Create main panel
        panel_content = Text()
        
        # Add session information
        session_id = session_data.get("id", "Unknown")
        created_at = session_data.get("created_at", "Unknown")
        image_path = session_data.get("image_path", "Unknown")
        naive_prompt = session_data.get("naive_prompt", "Unknown")
        status = session_data.get("status", "Unknown")
        final_alignment_score = session_data.get("final_alignment_score", 0.0)
        
        panel_content.append(f"Session ID: {session_id}\\n", style="bold blue")
        panel_content.append(f"Created: {created_at}\\n", style="dim")
        panel_content.append(f"Image: {image_path}\\n", style="green")
        panel_content.append(f"Prompt: {naive_prompt}\\n", style="yellow")
        panel_content.append(f"Status: {status}\\n", style="magenta")
        panel_content.append(f"Alignment Score: {self.color_code_score(final_alignment_score)}\\n", style="cyan")
        
        # Create and display panel
        panel = Panel(
            panel_content,
            title="[bold]Session Summary[/bold]",
            border_style="blue",
            expand=False
        )
        
        self.console.print(panel)
    
    def display_prompt_history(self, prompt_history: List[Dict[str, Any]]) -> None:
        """
        Display prompt history with Rich formatting.
        
        Args:
            prompt_history: List of prompt history dictionaries
        """
        # Create table for prompt history
        table = Table(title="Prompt History", show_header=True, header_style="bold magenta")
        table.add_column("Iteration", style="dim", width=12)
        table.add_column("Positive Prompt", min_width=30)
        table.add_column("Negative Prompt", min_width=30)
        table.add_column("Quality Score", justify="right", style="green")
        
        # Add rows for each prompt
        for prompt_data in prompt_history:
            iteration = prompt_data.get("iteration", 0)
            positive_prompt = prompt_data.get("positive_prompt", "")
            negative_prompt = prompt_data.get("negative_prompt", "")
            quality_score = prompt_data.get("quality_score", 0.0)
            
            table.add_row(
                str(iteration),
                positive_prompt[:50] + "..." if len(positive_prompt) > 50 else positive_prompt,
                negative_prompt[:50] + "..." if len(negative_prompt) > 50 else negative_prompt,
                self.color_code_score(quality_score)
            )
        
        self.console.print(table)
    
    def display_entity_list(self, entities: List[Dict[str, Any]]) -> None:
        """
        Display entity list with Rich formatting.
        
        Args:
            entities: List of entity dictionaries
        """
        # Create tree for entities
        from rich.tree import Tree
        tree = Tree("Detected Entities", style="bold blue", guide_style="bold black")
        
        # Add each entity to the tree
        for i, entity in enumerate(entities):
            entity_id = entity.get("entity_id", f"entity_{i}")
            label = entity.get("label", "Unknown")
            confidence = entity.get("confidence", 0.0)
            
            # Create entity node
            entity_node = tree.add(f"[bold]{label}[/bold] ({entity_id})", style="green")
            
            # Add entity details
            entity_node.add(f"Confidence: {self.color_code_score(confidence)}")
            
            # Add bounding box if available
            bbox = entity.get("bbox", {})
            if bbox:
                x1, y1, x2, y2 = bbox.get("x1", 0), bbox.get("y1", 0), bbox.get("x2", 0), bbox.get("y2", 0)
                entity_node.add(f"Bounding Box: ({x1}, {y1}) to ({x2}, {y2})")
            
            # Add color if available
            color_hex = entity.get("color_hex", "")
            if color_hex:
                entity_node.add(f"Color: [bold]{color_hex}[/bold]")
            
            # Add area percentage if available
            area_percent = entity.get("area_percent", 0.0)
            if area_percent > 0:
                entity_node.add(f"Area: [yellow]{area_percent:.1f}%[/yellow]")
        
        self.console.print(tree)
    
    def display_validation_metrics(self, validation_data: Dict[str, Any]) -> None:
        """
        Display validation metrics with Rich formatting.
        
        Args:
            validation_data: Dictionary containing validation information
        """
        # Create table for validation metrics
        table = Table(title="Validation Metrics", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="dim", width=20)
        table.add_column("Value", justify="right", style="green")
        table.add_column("Threshold", justify="right", style="blue")
        table.add_column("Status", justify="center")
        
        # Add metrics to table
        metrics = {
            "Alignment Score": {
                "value": validation_data.get("alignment_score", 0.0),
                "threshold": 0.7,
                "status": "pass" if validation_data.get("alignment_score", 0.0) >= 0.7 else "fail"
            },
            "Preserved Count": {
                "value": validation_data.get("preserved_count", 0),
                "threshold": 2,
                "status": "pass" if validation_data.get("preserved_count", 0) >= 2 else "warn"
            },
            "Modified Count": {
                "value": validation_data.get("modified_count", 0),
                "threshold": 1,
                "status": "pass" if validation_data.get("modified_count", 0) >= 1 else "fail"
            },
            "Unintended Count": {
                "value": validation_data.get("unintended_count", 0),
                "threshold": 0,
                "status": "pass" if validation_data.get("unintended_count", 0) == 0 else "fail"
            }
        }
        
        for metric_name, metric_info in metrics.items():
            value = metric_info["value"]
            threshold = metric_info["threshold"]
            status = metric_info["status"]
            
            # Format status with color
            if status == "pass":
                status_text = "[green]PASS[/green]"
            elif status == "fail":
                status_text = "[red]FAIL[/red]"
            elif status == "warn":
                status_text = "[yellow]WARN[/yellow]"
            else:
                status_text = status.upper()
            
            table.add_row(
                metric_name,
                f"{value:.3f}" if isinstance(value, float) else str(value),
                f"{threshold:.3f}" if isinstance(threshold, float) else str(threshold),
                status_text
            )
        
        self.console.print(table)
    
    def display_system_info(self, system_info: Dict[str, Any]) -> None:
        """
        Display system information with Rich formatting.
        
        Args:
            system_info: Dictionary containing system information
        """
        # Create panel for system information
        panel_content = Text()
        
        # Add system information
        platform_info = system_info.get("platform", "Unknown")
        python_version = system_info.get("python_version", "Unknown")
        cpu_count = system_info.get("cpu_count", 0)
        memory_info = system_info.get("memory_info", {})
        disk_info = system_info.get("disk_info", {})
        
        panel_content.append(f"Platform: {platform_info}\\n", style="bold blue")
        panel_content.append(f"Python Version: {python_version}\\n", style="green")
        panel_content.append(f"CPU Cores: {cpu_count}\\n", style="yellow")
        panel_content.append(f"Memory: {memory_info.get('available_gb', 0):.1f}GB available\\n", style="magenta")
        panel_content.append(f"Disk Space: {disk_info.get('free_gb', 0):.1f}GB free\\n", style="cyan")
        
        # Create and display panel
        panel = Panel(
            panel_content,
            title="[bold]System Information[/bold]",
            border_style="green",
            expand=False
        )
        
        self.console.print(panel)
    
    def display_error_message(self, message: str, exception: Optional[Exception] = None) -> None:
        """
        Display error message with Rich formatting.
        
        Args:
            message: Error message to display
            exception: Optional exception for additional details
        """
        # Create error panel
        panel_content = Text()
        panel_content.append(f"Error: {message}\\n", style="bold red")
        
        if exception:
            panel_content.append(f"\\nDetails: {str(exception)}\\n", style="dim")
        
        # Create and display panel
        panel = Panel(
            panel_content,
            title="[bold red]Error[/bold red]",
            border_style="red",
            expand=False
        )
        
        self.console.print(panel)
    
    def display_warning_message(self, message: str) -> None:
        """
        Display warning message with Rich formatting.
        
        Args:
            message: Warning message to display
        """
        # Create warning panel
        panel_content = Text()
        panel_content.append(f"Warning: {message}\\n", style="bold yellow")
        
        # Create and display panel
        panel = Panel(
            panel_content,
            title="[bold yellow]Warning[/bold yellow]",
            border_style="yellow",
            expand=False
        )
        
        self.console.print(panel)
    
    def display_success_message(self, message: str) -> None:
        """
        Display success message with Rich formatting.
        
        Args:
            message: Success message to display
        """
        # Create success panel
        panel_content = Text()
        panel_content.append(f"Success: {message}\\n", style="bold green")
        
        # Create and display panel
        panel = Panel(
            panel_content,
            title="[bold green]Success[/bold green]",
            border_style="green",
            expand=False
        )
        
        self.console.print(panel)
    
    def display_confirmation_dialog(self, message: str) -> bool:
        """
        Display confirmation dialog with Rich formatting.
        
        Args:
            message: Confirmation message to display
            
        Returns:
            Boolean indicating user response
        """
        # Display confirmation message
        self.console.print(f"[bold yellow]Confirm:[/bold yellow] {message}")
        
        # Get user confirmation
        from rich.prompt import Confirm
        response = Confirm.ask("Continue?", default=False, console=self.console)
        return response
    
    def display_prompt_dialog(self, message: str, default: str = "") -> str:
        """
        Display prompt dialog with Rich formatting.
        
        Args:
            message: Prompt message to display
            default: Default response
            
        Returns:
            User response string
        """
        # Get user input
        from rich.prompt import Prompt
        response = Prompt.ask(message, default=default, console=self.console)
        return response

# Example usage
if __name__ == "__main__":
    # Initialize rich UI utils
    ui_utils = RichUIUtils()
    
    print("Rich UI Utils initialized")
    
    # Example: Format duration
    durations = [0, 30, 90, 150, 3661, 7265, 90061]
    print("Duration formatting examples:")
    for duration in durations:
        formatted = ui_utils.format_duration(duration)
        print(f"  {duration:>8} seconds → {formatted}")
    
    # Example: Color code scores
    scores = [0.95, 0.75, 0.45, 0.88, 0.33]
    print("\\nScore coloring examples:")
    for score in scores:
        colored = ui_utils.color_code_score(score)
        print(f"  Score {score}: {colored}")
    
    # Example: Create test image for ANSI art conversion
    try:
        from PIL import Image, ImageDraw
        import numpy as np
        
        # Create a simple test image
        img = Image.new('RGB', (100, 100), color=(73, 109, 137))
        d = ImageDraw.Draw(img)
        d.text((10, 10), "Test", fill=(255, 255, 0))
        img.save("test_image.jpg")
        
        # Convert to ANSI art
        ansi_art = ui_utils.image_to_ansi_art("test_image.jpg", max_width=20)
        print(f"\\nANSI art conversion example:\\n{ansi_art}")
        
        # Display image preview
        print("\\nDisplaying image preview:")
        ui_utils.display_image_preview("test_image.jpg", max_width=20)
        
        # Clean up test image
        Path("test_image.jpg").unlink(missing_ok=True)
        
    except ImportError:
        print("Pillow not available for image processing examples")
    except Exception as e:
        print(f"Error in image processing example: {e}")
    
    # Example: Display session summary
    print("\\nDisplaying session summary:")
    session_data = {
        "id": "session-123",
        "created_at": "2023-10-23T10:00:00",
        "image_path": "/path/to/image.jpg",
        "naive_prompt": "make the sky more dramatic",
        "status": "completed",
        "final_alignment_score": 0.85
    }
    ui_utils.display_session_summary(session_data)
    
    # Example: Display prompt history
    print("\\nDisplaying prompt history:")
    prompt_history = [
        {
            "iteration": 0,
            "positive_prompt": "dramatic sky with storm clouds",
            "negative_prompt": "sunny sky, clear weather",
            "quality_score": 0.92
        },
        {
            "iteration": 1,
            "positive_prompt": "storm clouds with lighting and dark atmosphere",
            "negative_prompt": "sunny sky, clear weather, no clouds",
            "quality_score": 0.88
        }
    ]
    ui_utils.display_prompt_history(prompt_history)
    
    # Example: Display entity list
    print("\\nDisplaying entity list:")
    entities = [
        {
            "entity_id": "sky_0",
            "label": "sky",
            "confidence": 0.95,
            "bbox": {"x1": 0, "y1": 0, "x2": 1920, "y2": 768},
            "mask_path": "/path/to/mask.png",
            "color_hex": "#87CEEB",
            "area_percent": 39.6
        },
        {
            "entity_id": "mountain_1",
            "label": "mountain",
            "confidence": 0.87,
            "bbox": {"x1": 20, "y1": 768, "x2": 1900, "y2": 1080},
            "mask_path": "/path/to/mask2.png",
            "color_hex": "#556B2F",
            "area_percent": 25.3
        }
    ]
    ui_utils.display_entity_list(entities)
    
    # Example: Display validation metrics
    print("\\nDisplaying validation metrics:")
    validation_data = {
        "alignment_score": 0.85,
        "preserved_count": 3,
        "modified_count": 1,
        "unintended_count": 0
    }
    ui_utils.display_validation_metrics(validation_data)
    
    # Example: Display system info
    print("\\nDisplaying system info:")
    system_info = {
        "platform": "Linux-5.15.0-amd64",
        "python_version": "3.9.2",
        "cpu_count": 8,
        "memory_info": {"available_gb": 12.5},
        "disk_info": {"free_gb": 450.2}
    }
    ui_utils.display_system_info(system_info)
    
    # Example: Display messages
    print("\\nDisplaying messages:")
    ui_utils.display_success_message("Operation completed successfully!")
    ui_utils.display_warning_message("Low disk space - only 1.2GB remaining")
    ui_utils.display_error_message("Failed to load image", FileNotFoundError("image.jpg not found"))
    
    # Example: Display dialogs (these would require user input in a real application)
    print("\\nDisplaying dialogs:")
    # confirmed = ui_utils.display_confirmation_dialog("Do you want to delete this session?")
    # print(f"User confirmed: {confirmed}")
    
    # user_input = ui_utils.display_prompt_dialog("Enter your name:", default="Anonymous")
    # print(f"User input: {user_input}")
    
    print("\\nRich UI utils example completed!")
```

### Pillow Implementation Example
Image processing utilities for EDI:

```python
from PIL import Image, ImageOps, ImageEnhance, ImageDraw, ImageFont
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json
import logging
from datetime import datetime
import tempfile
import shutil
import os

class PillowImageProcessor:
    """
    Image processing utilities using Pillow.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def resize_image(self, image_path: str, max_size: int = 1024) -> str:
        """
        Resizes an image to fit within max_size x max_size while maintaining aspect ratio.
        
        Args:
            image_path: Path to the image to resize
            max_size: Maximum dimension (width or height) for the resized image
            
        Returns:
            Path to the resized image
        """
        # Validate input parameters
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        if not isinstance(max_size, int) or max_size <= 0:
            raise ValueError(f"max_size must be a positive integer, got {max_size}")
        
        # Get original dimensions of the input image
        with Image.open(image_path) as img:
            original_width, original_height = img.size
            
            # Handle edge case where image is already smaller than max_size
            if original_width <= max_size and original_height <= max_size:
                # Return original image path
                return image_path
            
            # Calculate the scaling factor to ensure the largest dimension doesn't exceed max_size
            scale_factor = min(max_size / original_width, max_size / original_height)
            
            # Calculate the new dimensions preserving the aspect ratio
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)
            
            # Resize the image using appropriate resampling algorithm
            # Use LANCZOS for high quality or BICUBIC for good balance of quality and speed
            resized_image = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Save to temporary file
            temp_dir = Path(tempfile.gettempdir())
            temp_path = temp_dir / f"resized_{Path(image_path).name}"
            
            # Save resized image
            resized_image.save(temp_path, optimize=True, quality=85)
            
            # Return the path to the resized image
            return str(temp_path)
    
    def enhance_image(self, 
                     image_path: str, 
                     brightness: float = 1.0, 
                     contrast: float = 1.0, 
                     saturation: float = 1.0,
                     sharpness: float = 1.0) -> str:
        """
        Enhance an image with brightness, contrast, saturation, and sharpness adjustments.
        
        Args:
            image_path: Path to the image to enhance
            brightness: Brightness adjustment factor (1.0 = no change)
            contrast: Contrast adjustment factor (1.0 = no change)
            saturation: Saturation adjustment factor (1.0 = no change)
            sharpness: Sharpness adjustment factor (1.0 = no change)
            
        Returns:
            Path to the enhanced image
        """
        # Validate input parameters
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Validate enhancement factors
        for factor_name, factor_value in [("brightness", brightness), ("contrast", contrast), 
                                        ("saturation", saturation), ("sharpness", sharpness)]:
            if not isinstance(factor_value, (int, float)) or factor_value < 0:
                raise ValueError(f"{factor_name} must be a non-negative number, got {factor_value}")
        
        # Load image
        with Image.open(image_path) as img:
            # Enhance brightness
            if brightness != 1.0:
                enhancer = ImageEnhance.Brightness(img)
                img = enhancer.enhance(brightness)
            
            # Enhance contrast
            if contrast != 1.0:
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(contrast)
            
            # Enhance saturation
            if saturation != 1.0:
                enhancer = ImageEnhance.Color(img)
                img = enhancer.enhance(saturation)
            
            # Enhance sharpness
            if sharpness != 1.0:
                enhancer = ImageEnhance.Sharpness(img)
                img = enhancer.enhance(sharpness)
            
            # Save to temporary file
            temp_dir = Path(tempfile.gettempdir())
            temp_path = temp_dir / f"enhanced_{Path(image_path).name}"
            
            # Save enhanced image
            img.save(temp_path, optimize=True, quality=85)
            
            # Return the path to the enhanced image
            return str(temp_path)
    
    def crop_image(self, 
                  image_path: str, 
                  bbox: Tuple[int, int, int, int],
                  expand: bool = False) -> str:
        """
        Crop an image to a bounding box.
        
        Args:
            image_path: Path to the image to crop
            bbox: Bounding box as (x1, y1, x2, y2)
            expand: Whether to expand the crop area
            
        Returns:
            Path to the cropped image
        """
        # Validate input parameters
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        if not isinstance(bbox, (tuple, list)) or len(bbox) != 4:
            raise ValueError(f"bbox must be a tuple of 4 integers, got {bbox}")
        
        # Validate bounding box coordinates
        x1, y1, x2, y2 = bbox
        if not all(isinstance(coord, int) for coord in bbox):
            raise ValueError(f"bbox coordinates must be integers, got {bbox}")
        
        if x1 >= x2 or y1 >= y2:
            raise ValueError(f"Invalid bounding box coordinates: {bbox}")
        
        # Load image
        with Image.open(image_path) as img:
            # Validate bounding box is within image bounds
            img_width, img_height = img.size
            if x1 < 0 or y1 < 0 or x2 > img_width or y2 > img_height:
                raise ValueError(f"Bounding box {bbox} is outside image bounds ({img_width}x{img_height})")
            
            # Crop the image to the bounding box
            cropped_img = img.crop((x1, y1, x2, y2))
            
            # Expand the cropped area if requested
            if expand:
                # This would expand the crop area by a certain percentage
                # For simplicity, we'll just add a border
                expanded_img = ImageOps.expand(cropped_img, border=10, fill='black')
                cropped_img = expanded_img
            
            # Save to temporary file
            temp_dir = Path(tempfile.gettempdir())
            temp_path = temp_dir / f"cropped_{Path(image_path).name}"
            
            # Save cropped image
            cropped_img.save(temp_path, optimize=True, quality=85)
            
            # Return the path to the cropped image
            return str(temp_path)
    
    def rotate_image(self, image_path: str, angle: float, expand: bool = True) -> str:
        """
        Rotate an image by a specified angle.
        
        Args:
            image_path: Path to the image to rotate
            angle: Angle to rotate in degrees (counter-clockwise)
            expand: Whether to expand canvas to fit rotated image
            
        Returns:
            Path to the rotated image
        """
        # Validate input parameters
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        if not isinstance(angle, (int, float)):
            raise ValueError(f"angle must be a number, got {angle}")
        
        # Load image
        with Image.open(image_path) as img:
            # Rotate the image by the specified angle
            # The expand parameter determines whether to expand canvas to fit rotated image
            rotated_img = img.rotate(angle, expand=expand, fillcolor=(0, 0, 0, 0))
            
            # Save to temporary file
            temp_dir = Path(tempfile.gettempdir())
            temp_path = temp_dir / f"rotated_{Path(image_path).name}"
            
            # Save rotated image
            rotated_img.save(temp_path, optimize=True, quality=85)
            
            # Return the path to the rotated image
            return str(temp_path)
    
    def flip_image(self, image_path: str, horizontal: bool = True, vertical: bool = False) -> str:
        """
        Flip an image horizontally and/or vertically.
        
        Args:
            image_path: Path to the image to flip
            horizontal: Whether to flip horizontally
            vertical: Whether to flip vertically
            
        Returns:
            Path to the flipped image
        """
        # Validate input parameters
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        if not horizontal and not vertical:
            raise ValueError("At least one of horizontal or vertical must be True")
        
        # Load image
        with Image.open(image_path) as img:
            # Flip the image horizontally if requested
            if horizontal:
                img = ImageOps.mirror(img)
            
            # Flip the image vertically if requested
            if vertical:
                img = ImageOps.flip(img)
            
            # Save to temporary file
            temp_dir = Path(tempfile.gettempdir())
            temp_path = temp_dir / f"flipped_{Path(image_path).name}"
            
            # Save flipped image
            img.save(temp_path, optimize=True, quality=85)
            
            # Return the path to the flipped image
            return str(temp_path)
    
    def apply_color_filter(self, image_path: str, color: str = "grayscale") -> str:
        """
        Apply a color filter to an image.
        
        Args:
            image_path: Path to the image to filter
            color: Color filter to apply ("grayscale", "sepia", "invert", "solarize")
            
        Returns:
            Path to the filtered image
        """
        # Validate input parameters
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        valid_filters = ["grayscale", "sepia", "invert", "solarize"]
        if color not in valid_filters:
            raise ValueError(f"color must be one of {valid_filters}, got {color}")
        
        # Load image
        with Image.open(image_path) as img:
            # Apply color filter based on type
            if color == "grayscale":
                # Apply grayscale filter
                filtered_img = ImageOps.grayscale(img)
            elif color == "sepia":
                # Apply sepia effect
                filtered_img = self._apply_sepia_effect(img)
            elif color == "invert":
                # Invert colors
                filtered_img = ImageOps.invert(img.convert('RGB'))
            elif color == "solarize":
                # Solarize image
                filtered_img = ImageOps.solarize(img, threshold=128)
            else:
                # Default to grayscale if unknown filter
                filtered_img = ImageOps.grayscale(img)
            
            # Save to temporary file
            temp_dir = Path(tempfile.gettempdir())
            temp_path = temp_dir / f"filtered_{Path(image_path).name}"
            
            # Save filtered image
            filtered_img.save(temp_path, optimize=True, quality=85)
            
            # Return the path to the filtered image
            return str(temp_path)
    
    def _apply_sepia_effect(self, img: Image.Image) -> Image.Image:
        """
        Apply sepia effect to an image.
        
        Args:
            img: PIL Image to apply sepia effect to
            
        Returns:
            Sepia-filtered PIL Image
        """
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Apply sepia matrix
        sepia_matrix = [
            0.393, 0.769, 0.189, 0,
            0.349, 0.686, 0.168, 0,
            0.272, 0.534, 0.131, 0,
            0, 0, 0, 1
        ]
        
        # Apply the transformation
        sepia_img = img.convert('RGB', sepia_matrix[:12])
        
        return sepia_img
    
    def overlay_text(self, 
                    image_path: str, 
                    text: str, 
                    position: Tuple[int, int] = (10, 10),
                    font_size: int = 24,
                    color: str = "white",
                    background: Optional[str] = None) -> str:
        """
        Overlay text on an image.
        
        Args:
            image_path: Path to the image to overlay text on
            text: Text to overlay
            position: Position as (x, y) coordinates
            font_size: Font size for the text
            color: Color of the text
            background: Optional background color for text
            
        Returns:
            Path to the image with overlaid text
        """
        # Validate input parameters
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        if not text.strip():
            raise ValueError("Text cannot be empty")
        
        if not isinstance(position, (tuple, list)) or len(position) != 2:
            raise ValueError(f"Position must be a tuple of 2 integers, got {position}")
        
        x, y = position
        if not all(isinstance(coord, int) for coord in position):
            raise ValueError(f"Position coordinates must be integers, got {position}")
        
        # Load image
        with Image.open(image_path) as img:
            # Create drawing context
            draw = ImageDraw.Draw(img)
            
            # Try to load a font, fall back to default if unavailable
            try:
                # Try to use a system font
                font = ImageFont.truetype("DejaVuSans.ttf", font_size)
            except IOError:
                # Fall back to default font
                font = ImageFont.load_default()
            
            # Draw background rectangle if specified
            if background:
                # Calculate text size
                bbox = draw.textbbox((x, y), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                # Draw background rectangle
                draw.rectangle(
                    [x, y, x + text_width, y + text_height],
                    fill=background
                )
            
            # Draw text
            draw.text(position, text, fill=color, font=font)
            
            # Save to temporary file
            temp_dir = Path(tempfile.gettempdir())
            temp_path = temp_dir / f"text_overlay_{Path(image_path).name}"
            
            # Save image with text overlay
            img.save(temp_path, optimize=True, quality=85)
            
            # Return the path to the image with overlaid text
            return str(temp_path)
    
    def get_image_info(self, image_path: str) -> Dict[str, Any]:
        """
        Get detailed information about an image.
        
        Args:
            image_path: Path to the image to analyze
            
        Returns:
            Dictionary with image information
        """
        # Validate input parameters
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Load image
        with Image.open(image_path) as img:
            # Get basic image information
            width, height = img.size
            mode = img.mode
            format_name = img.format
            
            # Get file information
            stat = Path(image_path).stat()
            file_size = stat.st_size
            created_time = datetime.fromtimestamp(stat.st_ctime)
            modified_time = datetime.fromtimestamp(stat.st_mtime)
            
            # Calculate aspect ratio
            aspect_ratio = width / height if height > 0 else 0
            
            # Calculate megapixels
            megapixels = (width * height) / (1000 * 1000)
            
            # Get EXIF data if available
            exif_data = {}
            try:
                if hasattr(img, '_getexif') and img._getexif():
                    exif_data = img._getexif()
            except Exception:
                pass
            
            # Return structured image information
            return {
                "path": image_path,
                "width": width,
                "height": height,
                "mode": mode,
                "format": format_name,
                "file_size_bytes": file_size,
                "file_size_mb": round(file_size / (1024 * 1024), 2),
                "aspect_ratio": round(aspect_ratio, 2),
                "megapixels": round(megapixels, 2),
                "created_time": created_time.isoformat(),
                "modified_time": modified_time.isoformat(),
                "exif_data": exif_data if exif_data else {},
                "color_profile": img.info.get("icc_profile", None),
                "compression": img.info.get("compression", "unknown")
            }

# Example usage
if __name__ == "__main__":
    # Initialize pillow image processor
    image_processor = PillowImageProcessor()
    
    print("Pillow Image Processor initialized")
    
    # Example: Create test image
    try:
        from PIL import Image, ImageDraw
        
        # Create a simple test image
        img = Image.new('RGB', (200, 100), color=(73, 109, 137))
        d = ImageDraw.Draw(img)
        d.text((10, 10), "Test Image", fill=(255, 255, 0))
        img.save("test_image.jpg")
        
        print("Created test image for processing examples")
        
        # Example: Resize image
        resized_path = image_processor.resize_image("test_image.jpg", max_size=50)
        print(f"Resized image: {resized_path}")
        
        # Example: Enhance image
        enhanced_path = image_processor.enhance_image(
            "test_image.jpg",
            brightness=1.2,
            contrast=1.1,
            saturation=1.3,
            sharpness=1.2
        )
        print(f"Enhanced image: {enhanced_path}")
        
        # Example: Crop image
        cropped_path = image_processor.crop_image("test_image.jpg", (10, 10, 190, 90))
        print(f"Cropped image: {cropped_path}")
        
        # Example: Rotate image
        rotated_path = image_processor.rotate_image("test_image.jpg", angle=45)
        print(f"Rotated image: {rotated_path}")
        
        # Example: Flip image
        flipped_path = image_processor.flip_image("test_image.jpg", horizontal=True, vertical=False)
        print(f"Flipped image: {flipped_path}")
        
        # Example: Apply color filter
        grayscale_path = image_processor.apply_color_filter("test_image.jpg", color="grayscale")
        print(f"Grayscale filtered image: {grayscale_path}")
        
        sepia_path = image_processor.apply_color_filter("test_image.jpg", color="sepia")
        print(f"Sepia filtered image: {sepia_path}")
        
        # Example: Overlay text
        text_overlay_path = image_processor.overlay_text(
            "test_image.jpg",
            "Sample Text",
            position=(20, 20),
            font_size=20,
            color="white",
            background="black"
        )
        print(f"Text overlay image: {text_overlay_path}")
        
        # Example: Get image info
        image_info = image_processor.get_image_info("test_image.jpg")
        print(f"\\nImage info:")
        print(f"  Dimensions: {image_info['width']}x{image_info['height']}")
        print(f"  Format: {image_info['format']}")
        print(f"  Mode: {image_info['mode']}")
        print(f"  File size: {image_info['file_size_mb']} MB")
        print(f"  Aspect ratio: {image_info['aspect_ratio']}")
        print(f"  Megapixels: {image_info['megapixels']}")
        print(f"  Created: {image_info['created_time']}")
        print(f"  Modified: {image_info['modified_time']}")
        
        # Clean up test images
        test_images = [
            "test_image.jpg",
            resized_path,
            enhanced_path,
            cropped_path,
            rotated_path,
            flipped_path,
            grayscale_path,
            sepia_path,
            text_overlay_path
        ]
        
        for image_path in test_images:
            if image_path and Path(image_path).exists():
                try:
                    Path(image_path).unlink()
                except Exception:
                    pass  # Ignore cleanup errors
        
        print("\\nPillow image processing examples completed!")
        
    except ImportError:
        print("Pillow not available for image processing examples")
    except Exception as e:
        print(f"Error in image processing example: {e}")
```

### Time Utilities Implementation Example
Time formatting and duration utilities for EDI:

```python
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import json
import logging
from pathlib import Path

class TimeUtilities:
    """
    Time utilities for formatting durations and timestamps.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def format_duration(self, seconds: float) -> str:
        """
        Formats time durations for display.
        
        Args:
            seconds: Time duration in seconds
            
        Returns:
            String representation like "1m 23s", "2h 5m", "45s"
        """
        # Validate input
        if not isinstance(seconds, (int, float)) or seconds < 0:
            raise ValueError(f"Seconds must be a non-negative number, got {seconds}")
        
        # Convert to integer for calculation
        total_seconds = int(seconds)
        
        # Calculate hours, minutes, and seconds
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        
        # Format the values into a human-readable string
        components = []
        
        # Add hours if significant
        if hours > 0:
            components.append(f"{hours}h")
        
        # Add minutes if significant or if we have hours
        if minutes > 0 or hours > 0:
            components.append(f"{minutes}m")
        
        # Add seconds if significant or if no other components
        if secs > 0 or len(components) == 0:
            components.append(f"{secs}s")
        
        # Omit zero-value components to keep the output concise
        # (already handled by conditional logic above)
        
        # Return the formatted duration string
        return " ".join(components)
    
    def format_duration_precise(self, seconds: float, include_ms: bool = False) -> str:
        """
        Format duration with more precise control over formatting.
        
        Args:
            seconds: Time duration in seconds
            include_ms: Whether to include milliseconds for sub-second precision
            
        Returns:
            Formatted duration string
        """
        # Validate input
        if not isinstance(seconds, (int, float)):
            raise TypeError(f"Seconds must be numeric, got {type(seconds)}")
        
        if seconds < 0:
            raise ValueError(f"Seconds must be non-negative, got {seconds}")
        
        # Handle zero seconds
        if seconds == 0:
            return "0s"
        
        # Handle fractional seconds
        if include_ms and seconds < 1:
            ms = int(seconds * 1000)
            if ms > 0:
                return f"{ms}ms"
            else:
                return "<1ms"
        
        # Calculate time units
        total_seconds = seconds
        days = int(total_seconds // 86400)
        total_seconds %= 86400
        hours = int(total_seconds // 3600)
        total_seconds %= 3600
        minutes = int(total_seconds // 60)
        secs = total_seconds % 60
        
        # Build components
        components = []
        
        if days > 0:
            components.append(f"{days}d")
        
        if hours > 0:
            components.append(f"{hours}h")
        
        if minutes > 0:
            components.append(f"{minutes}m")
        
        # For seconds, include decimal if requested and it's a small duration
        if (secs > 0 or len(components) == 0) and len(components) < 2:  # Limit components
            if include_ms and secs < 60 and len(components) == 0:
                # Show seconds with decimals for sub-minute durations
                components.append(f"{secs:.1f}s")
            else:
                # Show whole seconds
                components.append(f"{int(secs)}s")
        
        return " ".join(components) if components else "0s"
    
    def format_duration_alternative(self, seconds: float, style: str = "standard") -> str:
        """
        Format duration using alternative styles.
        
        Args:
            seconds: Time duration in seconds
            style: Formatting style - "standard", "verbose", "compact", "digital"
            
        Returns:
            Formatted duration string
        """
        # Validate input
        if not isinstance(seconds, (int, float)):
            raise TypeError(f"Seconds must be numeric, got {type(seconds)}")
        
        if seconds < 0:
            raise ValueError(f"Seconds must be non-negative, got {seconds}")
        
        if seconds == 0:
            if style == "verbose":
                return "zero seconds"
            elif style == "compact":
                return "0"
            elif style == "digital":
                return "00:00"
            else:
                return "0s"
        
        total_seconds = int(seconds)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        
        if style == "verbose":
            # Verbose format: "2 hours, 15 minutes, 30 seconds"
            parts = []
            if hours > 0:
                parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
            if minutes > 0:
                parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
            if secs > 0 or len(parts) == 0:
                parts.append(f"{secs} second{'s' if secs != 1 else ''}")
            
            if len(parts) == 1:
                return parts[0]
            elif len(parts) == 2:
                return f"{parts[0]} and {parts[1]}"
            else:
                return ", ".join(parts[:-1]) + f", and {parts[-1]}"
        
        elif style == "compact":
            # Compact format: "2h15m30s"
            result = ""
            if hours > 0:
                result += f"{hours}h"
            if minutes > 0:
                result += f"{minutes}m"
            if secs > 0 or result == "":
                result += f"{secs}s"
            return result
        
        elif style == "digital":
            # Digital format: "02:15:30" or "15:30"
            if hours > 0:
                return f"{hours:02d}:{minutes:02d}:{secs:02d}"
            else:
                return f"{minutes:02d}:{secs:02d}"
        
        else:
            # Standard format (same as main function)
            components = []
            if hours > 0:
                components.append(f"{hours}h")
            if minutes > 0:
                components.append(f"{minutes}m")
            if secs > 0 or len(components) == 0:
                components.append(f"{secs}s")
            return " ".join(components)
    
    def parse_duration(self, duration_str: str) -> float:
        """
        Parse a duration string back into seconds.
        
        Args:
            duration_str: Duration string like "2h 15m 30s"
            
        Returns:
            Total seconds as float
        """
        if not isinstance(duration_str, str):
            raise TypeError(f"Duration must be a string, got {type(duration_str)}")
        
        # Handle digital format first
        if ":" in duration_str:
            parts = duration_str.split(":")
            if len(parts) == 2:  # MM:SS
                minutes, seconds = map(int, parts)
                return minutes * 60 + seconds
            elif len(parts) == 3:  # HH:MM:SS
                hours, minutes, seconds = map(int, parts)
                return hours * 3600 + minutes * 60 + seconds
        
        # Handle standard format
        total_seconds = 0.0
        parts = duration_str.replace(",", " ").split()
        
        for part in parts:
            part = part.strip().lower()
            if part.endswith('d'):
                total_seconds += int(part[:-1]) * 86400
            elif part.endswith('h'):
                total_seconds += int(part[:-1]) * 3600
            elif part.endswith('m'):
                total_seconds += int(part[:-1]) * 60
            elif part.endswith('s'):
                total_seconds += int(part[:-1])
            elif part.endswith('ms'):
                total_seconds += int(part[:-2]) / 1000.0
        
        return total_seconds
    
    def format_timestamp(self, timestamp: str, format_style: str = "relative") -> str:
        """
        Format timestamp for display.
        
        Args:
            timestamp: ISO format timestamp string
            format_style: Style - "relative", "absolute", "short", "long"
            
        Returns:
            Formatted timestamp string
        """
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        except ValueError:
            # If parsing fails, use current time
            dt = datetime.now()
        
        now = datetime.now()
        diff = now - dt
        
        if format_style == "relative":
            # Relative time (e.g., "2 hours ago")
            if diff.days > 0:
                return f"{diff.days} day{'s' if diff.days != 1 else ''} ago"
            elif diff.seconds >= 3600:
                hours = diff.seconds // 3600
                return f"{hours} hour{'s' if hours != 1 else ''} ago"
            elif diff.seconds >= 60:
                minutes = diff.seconds // 60
                return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
            else:
                return "Just now"
        
        elif format_style == "absolute":
            # Absolute time (e.g., "2023-10-23 14:30:00")
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        
        elif format_style == "short":
            # Short format (e.g., "Oct 23, 2023")
            return dt.strftime("%b %d, %Y")
        
        elif format_style == "long":
            # Long format (e.g., "October 23, 2023 at 2:30 PM")
            return dt.strftime("%B %d, %Y at %I:%M %p")
        
        else:
            # Default to relative
            return self.format_timestamp(timestamp, "relative")
    
    def get_elapsed_time(self, start_time: float) -> str:
        """
        Get elapsed time since start time.
        
        Args:
            start_time: Start time as returned by time.time()
            
        Returns:
            Formatted elapsed time string
        """
        elapsed_seconds = time.time() - start_time
        return self.format_duration(elapsed_seconds)
    
    def measure_execution_time(self, func: callable, *args, **kwargs) -> Dict[str, Any]:
        """
        Measure execution time of a function.
        
        Args:
            func: Function to measure
            *args: Arguments to pass to function
            **kwargs: Keyword arguments to pass to function
            
        Returns:
            Dictionary with execution time and result
        """
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        return {
            "result": result,
            "success": success,
            "error": error,
            "execution_time_seconds": execution_time,
            "execution_time_formatted": self.format_duration(execution_time)
        }

# Example usage
if __name__ == "__main__":
    # Initialize time utilities
    time_utils = TimeUtilities()
    
    print("Time Utilities initialized")
    
    # Example: Format durations
    test_durations = [0, 30, 90, 150, 3661, 7265, 90061, 0.5, 1.75]
    
    print("\\nDuration formatting examples:")
    for duration in test_durations:
        formatted = time_utils.format_duration(duration)
        print(f"  {duration:>8} seconds → {formatted}")
    
    # Example: Precise formatting
    print("\\nPrecise formatting examples:")
    precise_durations = [0.123, 0.5, 1.75, 30.25, 90.99]
    for duration in precise_durations:
        formatted = time_utils.format_duration_precise(duration, include_ms=True)
        print(f"  {duration:>6} seconds → {formatted}")
    
    # Example: Alternative styles
    print("\\nAlternative style examples:")
    sample_duration = 3661  # 1 hour, 1 minute, 1 second
    styles = ["standard", "verbose", "compact", "digital"]
    for style in styles:
        formatted = time_utils.format_duration_alternative(sample_duration, style)
        print(f"  {style:>8}: {formatted}")
    
    # Example: Parsing durations
    print("\\nParsing duration strings:")
    test_strings = ["1h 2m 3s", "45m 30s", "120s", "02:30", "01:02:03"]
    for duration_str in test_strings:
        parsed = time_utils.parse_duration(duration_str)
        print(f"  '{duration_str}' → {parsed} seconds")
    
    # Example: Formatting timestamps
    print("\\nTimestamp formatting examples:")
    test_timestamp = datetime.now().isoformat()
    styles = ["relative", "absolute", "short", "long"]
    for style in styles:
        formatted = time_utils.format_timestamp(test_timestamp, style)
        print(f"  {style:>8}: {formatted}")
    
    # Example: Measuring execution time
    print("\\nMeasuring execution time:")
    
    def slow_function():
        """Simulate a slow function."""
        time.sleep(0.1)  # Sleep for 100ms
        return "Slow function result"
    
    execution_result = time_utils.measure_execution_time(slow_function)
    print(f"  Function result: {execution_result['result']}")
    print(f"  Success: {execution_result['success']}")
    print(f"  Execution time: {execution_result['execution_time_formatted']}")
    print(f"  Raw time: {execution_result['execution_time_seconds']:.3f} seconds")
    
    if execution_result['error']:
        print(f"  Error: {execution_result['error']}")
    
    print("\\nTime utilities example completed!")
```