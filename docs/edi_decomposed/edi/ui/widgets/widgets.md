# UI: Widgets

[Back to TUI Layer](./tui_layer.md)

## Purpose
Custom Widgets - Contains specialized UI components like ImageComparisonPane, PromptDiffViewer, EntitySelectorList, etc.

## Widgets
- `ImageComparisonPane`: Side-by-side image viewer with ANSI art conversion
- `PromptDiffViewer`: Shows prompt evolution with highlighted changes
- `EntitySelectorList`: Checkbox list for entity selection
- `ValidationMetricsTable`: DataTable for validation metrics
- `ProgressSpinner`: Animated spinner with status text

### Details
- Custom-built for EDI's specific needs
- Implements visual feedback mechanisms
- Provides rich terminal user experience

## Technology Stack

- Textual for widget creation
- Rich for terminal rendering
- ANSI art conversion for images

## See Docs

### Textual Implementation Example
Custom widget implementations for the EDI application:

```python
from textual.app import App
from textual.widgets import Static, Button, ListView, ListItem, DataTable, ProgressBar
from textual.containers import Container, Vertical, Horizontal
from textual.screen import Screen
from textual import work
import asyncio
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
from datetime import datetime
from rich.text import Text
from rich.console import Console
from rich.panel import Panel
import logging

class ImageComparisonPane(Static):
    """
    Side-by-side image viewer with ANSI art conversion.
    """
    
    def __init__(self, 
                 before_image_path: str = None, 
                 after_image_path: str = None,
                 name: str = None,
                 id: str = None,
                 classes: str = None):
        super().__init__(name=name, id=id, classes=classes)
        self.before_image_path = before_image_path
        self.after_image_path = after_image_path
        self.logger = logging.getLogger(__name__)
    
    def on_mount(self) -> None:
        """Called when the widget is mounted."""
        self.update_comparison_view()
    
    def update_comparison_view(self) -> None:
        """Update the comparison view with current images."""
        if not self.before_image_path or not self.after_image_path:
            self.update("No images to compare")
            return
        
        try:
            # Convert images to ANSI art
            before_ansi = self._image_to_ansi_art(self.before_image_path)
            after_ansi = self._image_to_ansi_art(self.after_image_path)
            
            # Create side-by-side comparison
            comparison_text = self._create_side_by_side_view(before_ansi, after_ansi)
            self.update(comparison_text)
            
        except Exception as e:
            self.logger.error(f"Error updating comparison view: {str(e)}")
            self.update(f"Error: {str(e)}")
    
    def _image_to_ansi_art(self, image_path: str) -> str:
        """
        Convert image to ANSI art representation.
        
        Args:
            image_path: Path to image file
            
        Returns:
            ANSI art string representation
        """
        # In a real implementation, this would use an actual ANSI art conversion library
        # For this example, we'll create a simple representation
        try:
            from PIL import Image
            
            # Load image
            with Image.open(image_path) as img:
                # Resize to terminal-friendly dimensions
                max_width = 40
                max_height = 20
                
                # Calculate new dimensions maintaining aspect ratio
                aspect_ratio = img.width / img.height
                new_width = min(max_width, img.width)
                new_height = int(new_width / aspect_ratio / 2)  # Divide by 2 for character aspect ratio
                
                # Ensure minimum dimensions
                new_height = max(1, new_height)
                
                # Resize image
                resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Convert to ANSI art (simplified)
                ansi_art = self._convert_to_ansi_simple(resized_img)
                return ansi_art
                
        except Exception as e:
            self.logger.error(f"Error converting image to ANSI art: {str(e)}")
            return f"[Error converting {Path(image_path).name}]"
    
    def _convert_to_ansi_simple(self, image) -> str:
        """
        Simple conversion to ANSI art.
        
        Args:
            image: PIL Image object
            
        Returns:
            ANSI art string
        """
        # Simple character-based representation
        chars = " .:-=+*#%@"
        
        # Convert to grayscale if needed
        if image.mode != 'L':
            grayscale = image.convert('L')
        else:
            grayscale = image
        
        # Build ANSI art
        ansi_lines = []
        for y in range(grayscale.height):
            line = ""
            for x in range(grayscale.width):
                pixel = grayscale.getpixel((x, y))
                char_index = int(pixel / 255 * (len(chars) - 1))
                line += chars[char_index]
            ansi_lines.append(line)
        
        return "\n".join(ansi_lines)
    
    def _create_side_by_side_view(self, before_ansi: str, after_ansi: str) -> Text:
        """
        Create side-by-side comparison view.
        
        Args:
            before_ansi: ANSI art for before image
            after_ansi: ANSI art for after image
            
        Returns:
            Text object with formatted comparison view
        """
        # Split ANSI art into lines
        before_lines = before_ansi.split('\n')
        after_lines = after_ansi.split('\n')
        
        # Pad lines to same length
        max_lines = max(len(before_lines), len(after_lines))
        before_lines.extend([''] * (max_lines - len(before_lines)))
        after_lines.extend([''] * (max_lines - len(after_lines)))
        
        # Pad lines to same width
        max_width = max(
            max(len(line) for line in before_lines),
            max(len(line) for line in after_lines)
        )
        
        before_lines = [line.ljust(max_width) for line in before_lines]
        after_lines = [line.ljust(max_width) for line in after_lines]
        
        # Create comparison text
        comparison_lines = []
        comparison_lines.append(Text("BEFORE".center(max_width) + "  " + "AFTER".center(max_width), style="bold"))
        comparison_lines.append(Text("-" * max_width + "  " + "-" * max_width))
        
        for before_line, after_line in zip(before_lines, after_lines):
            line = Text()
            line.append(before_line, style="dim")
            line.append("  ")
            line.append(after_line, style="bright")
            comparison_lines.append(line)
        
        # Combine all lines
        result = Text()
        for i, line in enumerate(comparison_lines):
            if i > 0:
                result.append("\n")
            result.append(line)
        
        return result
    
    def set_images(self, before_image_path: str, after_image_path: str) -> None:
        """
        Set images for comparison.
        
        Args:
            before_image_path: Path to before image
            after_image_path: Path to after image
        """
        self.before_image_path = before_image_path
        self.after_image_path = after_image_path
        self.update_comparison_view()

class PromptDiffViewer(Static):
    """
    Shows prompt evolution with highlighted changes.
    """
    
    def __init__(self, 
                 prompt_history: List[Dict[str, Any]] = None,
                 name: str = None,
                 id: str = None,
                 classes: str = None):
        super().__init__(name=name, id=id, classes=classes)
        self.prompt_history = prompt_history or []
        self.logger = logging.getLogger(__name__)
    
    def on_mount(self) -> None:
        """Called when the widget is mounted."""
        self.update_diff_view()
    
    def update_diff_view(self) -> None:
        """Update the diff view with current prompt history."""
        if not self.prompt_history:
            self.update("No prompt history to display")
            return
        
        try:
            # Create diff view
            diff_text = self._create_diff_view(self.prompt_history)
            self.update(diff_text)
            
        except Exception as e:
            self.logger.error(f"Error updating diff view: {str(e)}")
            self.update(f"Error: {str(e)}")
    
    def _create_diff_view(self, prompt_history: List[Dict[str, Any]]) -> Text:
        """
        Create diff view from prompt history.
        
        Args:
            prompt_history: List of prompt history dictionaries
            
        Returns:
            Text object with formatted diff view
        """
        # Create diff text
        diff_text = Text()
        
        # Add header
        diff_text.append("Prompt Evolution\n", style="bold")
        diff_text.append("=" * 20 + "\n\n")
        
        # Add each prompt iteration
        for i, prompt_data in enumerate(prompt_history):
            iteration = prompt_data.get("iteration", i)
            positive_prompt = prompt_data.get("positive_prompt", "")
            negative_prompt = prompt_data.get("negative_prompt", "")
            quality_score = prompt_data.get("quality_score", 0.0)
            
            # Add iteration header
            diff_text.append(f"Iteration {iteration}\n", style="bold underline")
            
            # Add positive prompt
            diff_text.append("Positive: ", style="green")
            diff_text.append(f"{positive_prompt}\n")
            
            # Add negative prompt
            diff_text.append("Negative: ", style="red")
            diff_text.append(f"{negative_prompt}\n")
            
            # Add quality score
            diff_text.append("Quality: ", style="blue")
            diff_text.append(f"{quality_score:.2f}\n")
            
            # Add separator
            if i < len(prompt_history) - 1:
                diff_text.append("\n" + "-" * 10 + "\n\n")
        
        return diff_text
    
    def set_prompt_history(self, prompt_history: List[Dict[str, Any]]) -> None:
        """
        Set prompt history for diff view.
        
        Args:
            prompt_history: List of prompt history dictionaries
        """
        self.prompt_history = prompt_history
        self.update_diff_view()

class EntitySelectorList(ListView):
    """
    Checkbox list for entity selection.
    """
    
    def __init__(self, 
                 entities: List[Dict[str, Any]] = None,
                 name: str = None,
                 id: str = None,
                 classes: str = None):
        super().__init__(name=name, id=id, classes=classes)
        self.entities = entities or []
        self.selected_entities = set()
        self.logger = logging.getLogger(__name__)
    
    def on_mount(self) -> None:
        """Called when the widget is mounted."""
        self.populate_entities()
    
    def populate_entities(self) -> None:
        """
        Populate the list with entities.
        """
        # Clear existing items
        self.clear()
        
        # Add entities to list
        for i, entity in enumerate(self.entities):
            entity_id = entity.get("entity_id", f"entity_{i}")
            label = entity.get("label", "Unknown")
            confidence = entity.get("confidence", 0.0)
            
            # Create list item with checkbox
            item_text = f"[{'x' if entity_id in self.selected_entities else ' '}] {label} ({confidence:.2f})"
            item = ListItem(Static(item_text), id=f"entity_{entity_id}")
            self.append(item)
    
    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """
        Handle list item selection (toggles checkbox).
        
        Args:
            event: ListView.Selected event
        """
        # Get selected item
        selected_item = event.item
        item_id = selected_item.id
        
        # Extract entity ID from item ID
        if item_id and item_id.startswith("entity_"):
            entity_id = item_id[len("entity_"):]
            
            # Toggle selection
            if entity_id in self.selected_entities:
                self.selected_entities.remove(entity_id)
            else:
                self.selected_entities.add(entity_id)
            
            # Update item text
            for i, entity in enumerate(self.entities):
                if entity.get("entity_id") == entity_id:
                    label = entity.get("label", "Unknown")
                    confidence = entity.get("confidence", 0.0)
                    item_text = f"[{'x' if entity_id in self.selected_entities else ' '}] {label} ({confidence:.2f})"
                    selected_item.children[0].update(item_text)
                    break
    
    def get_selected_entities(self) -> List[str]:
        """
        Get list of selected entity IDs.
        
        Returns:
            List of selected entity IDs
        """
        return list(self.selected_entities)
    
    def set_entities(self, entities: List[Dict[str, Any]]) -> None:
        """
        Set entities for the selector list.
        
        Args:
            entities: List of entity dictionaries
        """
        self.entities = entities
        self.populate_entities()
    
    def select_all(self) -> None:
        """
        Select all entities.
        """
        self.selected_entities = {entity.get("entity_id") for entity in self.entities}
        self.populate_entities()
    
    def deselect_all(self) -> None:
        """
        Deselect all entities.
        """
        self.selected_entities.clear()
        self.populate_entities()

class ValidationMetricsTable(DataTable):
    """
    DataTable for validation metrics.
    """
    
    def __init__(self, 
                 name: str = None,
                 id: str = None,
                 classes: str = None):
        super().__init__(name=name, id=id, classes=classes)
        self.logger = logging.getLogger(__name__)
        self._initialize_columns()
    
    def _initialize_columns(self) -> None:
        """
        Initialize table columns.
        """
        self.add_column("Metric", width=20)
        self.add_column("Value", width=15)
        self.add_column("Threshold", width=15)
        self.add_column("Status", width=10)
    
    def update_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Update table with validation metrics.
        
        Args:
            metrics: Dictionary of validation metrics
        """
        # Clear existing rows
        self.clear()
        
        # Add metrics to table
        for metric_name, metric_data in metrics.items():
            value = metric_data.get("value", 0.0)
            threshold = metric_data.get("threshold", 0.0)
            status = metric_data.get("status", "unknown")
            
            # Format value and threshold
            value_str = f"{value:.3f}"
            threshold_str = f"{threshold:.3f}"
            
            # Format status with color
            if status == "pass":
                status_str = "[green]PASS[/green]"
            elif status == "fail":
                status_str = "[red]FAIL[/red]"
            elif status == "warn":
                status_str = "[yellow]WARN[/yellow]"
            else:
                status_str = status.upper()
            
            # Add row
            self.add_row(metric_name, value_str, threshold_str, status_str)
    
    def set_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Set metrics for the table.
        
        Args:
            metrics: Dictionary of validation metrics
        """
        self.update_metrics(metrics)

class ProgressSpinner(Static):
    """
    Animated spinner with status text.
    """
    
    def __init__(self, 
                 status_text: str = "Processing...",
                 name: str = None,
                 id: str = None,
                 classes: str = None):
        super().__init__(name=name, id=id, classes=classes)
        self.status_text = status_text
        self.spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        self.spinner_index = 0
        self.is_spinning = False
        self.logger = logging.getLogger(__name__)
    
    def on_mount(self) -> None:
        """Called when the widget is mounted."""
        self.start_spinner()
    
    def start_spinner(self) -> None:
        """
        Start the spinner animation.
        """
        if not self.is_spinning:
            self.is_spinning = True
            self._animate_spinner()
    
    def stop_spinner(self) -> None:
        """
        Stop the spinner animation.
        """
        self.is_spinning = False
        self.update("")
    
    def _animate_spinner(self) -> None:
        """
        Animate the spinner.
        """
        if self.is_spinning:
            # Get current spinner character
            spinner_char = self.spinner_chars[self.spinner_index]
            self.spinner_index = (self.spinner_index + 1) % len(self.spinner_chars)
            
            # Update display
            self.update(f"{spinner_char} {self.status_text}")
            
            # Schedule next animation frame
            self.set_timer(0.1, self._animate_spinner)
    
    def set_status_text(self, text: str) -> None:
        """
        Set the status text.
        
        Args:
            text: New status text
        """
        self.status_text = text
        if self.is_spinning:
            spinner_char = self.spinner_chars[self.spinner_index]
            self.update(f"{spinner_char} {self.status_text}")

# Example usage
if __name__ == "__main__":
    # Example of using the custom widgets
    
    # Initialize widgets
    print("Initializing custom widgets...")
    
    # Example: ImageComparisonPane
    comparison_pane = ImageComparisonPane(
        before_image_path="before.jpg",
        after_image_path="after.jpg"
    )
    print(f"Created ImageComparisonPane: {comparison_pane}")
    
    # Example: PromptDiffViewer
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
    
    diff_viewer = PromptDiffViewer(prompt_history=prompt_history)
    print(f"Created PromptDiffViewer: {diff_viewer}")
    
    # Example: EntitySelectorList
    entities = [
        {
            "entity_id": "sky_0",
            "label": "sky",
            "confidence": 0.95
        },
        {
            "entity_id": "mountain_1",
            "label": "mountain",
            "confidence": 0.87
        },
        {
            "entity_id": "tree_2",
            "label": "tree",
            "confidence": 0.92
        }
    ]
    
    entity_selector = EntitySelectorList(entities=entities)
    print(f"Created EntitySelectorList: {entity_selector}")
    
    # Example: ValidationMetricsTable
    metrics = {
        "alignment_score": {
            "value": 0.85,
            "threshold": 0.7,
            "status": "pass"
        },
        "preserved_count": {
            "value": 3,
            "threshold": 2,
            "status": "pass"
        },
        "modified_count": {
            "value": 1,
            "threshold": 1,
            "status": "pass"
        },
        "unintended_count": {
            "value": 0,
            "threshold": 0,
            "status": "pass"
        }
    }
    
    metrics_table = ValidationMetricsTable()
    metrics_table.set_metrics(metrics)
    print(f"Created ValidationMetricsTable: {metrics_table}")
    
    # Example: ProgressSpinner
    spinner = ProgressSpinner("Processing images...")
    print(f"Created ProgressSpinner: {spinner}")
    
    # Start spinner
    spinner.start_spinner()
    print("Spinner started")
    
    # Update status text
    spinner.set_status_text("Analyzing image...")
    print("Spinner status updated")
    
    # Stop spinner
    spinner.stop_spinner()
    print("Spinner stopped")
    
    print("Custom widget examples completed!")
```

### Rich Implementation Example
Terminal rendering with Rich for the EDI application:

```python
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.text import Text
from rich.syntax import Syntax
from rich.tree import Tree
from rich.markdown import Markdown
from rich.prompt import Prompt, Confirm
from rich.live import Live
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import logging

class RichRenderer:
    """
    Rich terminal renderer for EDI application.
    """
    
    def __init__(self):
        self.console = Console()
        self.logger = logging.getLogger(__name__)
    
    def render_session_summary(self, session_data: Dict[str, Any]) -> None:
        """
        Render session summary with Rich formatting.
        
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
        
        panel_content.append(f"Session ID: {session_id}\n", style="bold blue")
        panel_content.append(f"Created: {created_at}\n", style="dim")
        panel_content.append(f"Image: {image_path}\n", style="green")
        panel_content.append(f"Prompt: {naive_prompt}\n", style="yellow")
        panel_content.append(f"Status: {status}\n", style="magenta")
        panel_content.append(f"Alignment Score: {final_alignment_score:.2f}\n", style="cyan")
        
        # Create and display panel
        panel = Panel(
            panel_content,
            title="[bold]Session Summary[/bold]",
            border_style="blue",
            expand=False
        )
        
        self.console.print(panel)
    
    def render_prompt_history(self, prompt_history: List[Dict[str, Any]]) -> None:
        """
        Render prompt history with Rich formatting.
        
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
                f"{quality_score:.2f}"
            )
        
        self.console.print(table)
    
    def render_entity_list(self, entities: List[Dict[str, Any]]) -> None:
        """
        Render entity list with Rich formatting.
        
        Args:
            entities: List of entity dictionaries
        """
        # Create tree for entities
        tree = Tree("Detected Entities", style="bold blue", guide_style="bold black")
        
        # Add each entity to the tree
        for i, entity in enumerate(entities):
            entity_id = entity.get("entity_id", f"entity_{i}")
            label = entity.get("label", "Unknown")
            confidence = entity.get("confidence", 0.0)
            
            # Create entity node
            entity_node = tree.add(f"[bold]{label}[/bold] ({entity_id})", style="green")
            
            # Add entity details
            entity_node.add(f"Confidence: [cyan]{confidence:.2f}[/cyan]")
            
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
    
    def render_validation_metrics(self, validation_data: Dict[str, Any]) -> None:
        """
        Render validation metrics with Rich formatting.
        
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
    
    def render_progress_bar(self, total: int = 100, description: str = "Processing") -> Progress:
        """
        Render a progress bar with Rich formatting.
        
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
    
    def render_system_info(self, system_info: Dict[str, Any]) -> None:
        """
        Render system information with Rich formatting.
        
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
        
        panel_content.append(f"Platform: {platform_info}\n", style="bold blue")
        panel_content.append(f"Python Version: {python_version}\n", style="green")
        panel_content.append(f"CPU Cores: {cpu_count}\n", style="yellow")
        panel_content.append(f"Memory: {memory_info.get('available_gb', 0):.1f}GB available\n", style="magenta")
        panel_content.append(f"Disk Space: {disk_info.get('free_gb', 0):.1f}GB free\n", style="cyan")
        
        # Create and display panel
        panel = Panel(
            panel_content,
            title="[bold]System Information[/bold]",
            border_style="green",
            expand=False
        )
        
        self.console.print(panel)
    
    def render_help_info(self) -> None:
        """
        Render help information with Rich formatting.
        """
        # Create help text with markdown
        help_text = """
# EDI (Edit with Intelligence) - Help

## Commands

- `edi edit <image> <prompt>` - Edit an image with AI
- `edi setup` - Setup EDI environment
- `edi doctor` - Diagnose system issues
- `edi clear` - Clear session data
- `edi --help` - Show this help

## Keyboard Shortcuts

- `Q` - Quit application
- `H` - Show help
- `B` - Go back
- `Ctrl+S` - Save session
- `Ctrl+L` - Load session
- `Ctrl+N` - New session
- `Ctrl+D` - Delete session
- `Ctrl+R` - Refresh sessions
- `ESC` - Go back to previous screen

## Navigation

- Arrow keys - Move cursor
- Tab/Shift+Tab - Cycle through elements
- Enter - Confirm selection

## Editing Shortcuts

- `E` - Edit prompt
- `R` - Retry edit
- `A` - Accept result
- `V` - View variations
        """
        
        # Render markdown
        md = Markdown(help_text)
        self.console.print(md)
    
    def render_error_message(self, message: str, exception: Optional[Exception] = None) -> None:
        """
        Render error message with Rich formatting.
        
        Args:
            message: Error message to display
            exception: Optional exception for additional details
        """
        # Create error panel
        panel_content = Text()
        panel_content.append(f"Error: {message}\n", style="bold red")
        
        if exception:
            panel_content.append(f"\nDetails: {str(exception)}\n", style="dim")
        
        # Create and display panel
        panel = Panel(
            panel_content,
            title="[bold red]Error[/bold red]",
            border_style="red",
            expand=False
        )
        
        self.console.print(panel)
    
    def render_warning_message(self, message: str) -> None:
        """
        Render warning message with Rich formatting.
        
        Args:
            message: Warning message to display
        """
        # Create warning panel
        panel_content = Text()
        panel_content.append(f"Warning: {message}\n", style="bold yellow")
        
        # Create and display panel
        panel = Panel(
            panel_content,
            title="[bold yellow]Warning[/bold yellow]",
            border_style="yellow",
            expand=False
        )
        
        self.console.print(panel)
    
    def render_success_message(self, message: str) -> None:
        """
        Render success message with Rich formatting.
        
        Args:
            message: Success message to display
        """
        # Create success panel
        panel_content = Text()
        panel_content.append(f"Success: {message}\n", style="bold green")
        
        # Create and display panel
        panel = Panel(
            panel_content,
            title="[bold green]Success[/bold green]",
            border_style="green",
            expand=False
        )
        
        self.console.print(panel)
    
    def render_confirmation_dialog(self, message: str) -> bool:
        """
        Render confirmation dialog with Rich formatting.
        
        Args:
            message: Confirmation message to display
            
        Returns:
            Boolean indicating user response
        """
        # Render confirmation message
        self.console.print(f"[bold yellow]Confirm:[/bold yellow] {message}")
        
        # Get user confirmation
        response = Confirm.ask("Continue?", default=False, console=self.console)
        return response
    
    def render_prompt_dialog(self, message: str, default: str = "") -> str:
        """
        Render prompt dialog with Rich formatting.
        
        Args:
            message: Prompt message to display
            default: Default response
            
        Returns:
            User response string
        """
        # Get user input
        response = Prompt.ask(message, default=default, console=self.console)
        return response

# Example usage
if __name__ == "__main__":
    # Initialize rich renderer
    renderer = RichRenderer()
    
    print("Rich Renderer initialized")
    
    # Example: Render session summary
    session_data = {
        "id": "session-123",
        "created_at": "2023-10-23T10:00:00",
        "image_path": "/path/to/image.jpg",
        "naive_prompt": "make the sky more dramatic",
        "status": "completed",
        "final_alignment_score": 0.85
    }
    
    print("\\nRendering session summary...")
    renderer.render_session_summary(session_data)
    
    # Example: Render prompt history
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
    
    print("\\nRendering prompt history...")
    renderer.render_prompt_history(prompt_history)
    
    # Example: Render entity list
    entities = [
        {
            "entity_id": "sky_0",
            "label": "sky",
            "confidence": 0.95,
            "bbox": {"x1": 0, "y1": 0, "x2": 1920, "y2": 768},
            "color_hex": "#87CEEB",
            "area_percent": 39.6
        },
        {
            "entity_id": "mountain_1",
            "label": "mountain",
            "confidence": 0.87,
            "bbox": {"x1": 20, "y1": 768, "x2": 1900, "y2": 1080},
            "color_hex": "#556B2F",
            "area_percent": 25.3
        },
        {
            "entity_id": "tree_2",
            "label": "tree",
            "confidence": 0.92,
            "bbox": {"x1": 100, "y1": 800, "x2": 200, "y2": 1000},
            "color_hex": "#228B22",
            "area_percent": 3.7
        }
    ]
    
    print("\\nRendering entity list...")
    renderer.render_entity_list(entities)
    
    # Example: Render validation metrics
    validation_data = {
        "alignment_score": 0.85,
        "preserved_count": 3,
        "modified_count": 1,
        "unintended_count": 0
    }
    
    print("\\nRendering validation metrics...")
    renderer.render_validation_metrics(validation_data)
    
    # Example: Render progress bar
    print("\\nRendering progress bar...")
    with renderer.render_progress_bar(total=100, description="Processing Images") as progress:
        task = progress.add_task("Processing...", total=100)
        for i in range(100):
            progress.update(task, advance=1)
            import time
            time.sleep(0.01)  # Simulate work
    
    # Example: Render system info
    system_info = {
        "platform": "Linux-5.15.0-amd64",
        "python_version": "3.9.2",
        "cpu_count": 8,
        "memory_info": {"available_gb": 12.5},
        "disk_info": {"free_gb": 450.2}
    }
    
    print("\\nRendering system info...")
    renderer.render_system_info(system_info)
    
    # Example: Render help info
    print("\\nRendering help info...")
    renderer.render_help_info()
    
    # Example: Render error message
    print("\\nRendering error message...")
    renderer.render_error_message("Failed to load image", FileNotFoundError("image.jpg not found"))
    
    # Example: Render warning message
    print("\\nRendering warning message...")
    renderer.render_warning_message("Low disk space - only 1.2GB remaining")
    
    # Example: Render success message
    print("\\nRendering success message...")
    renderer.render_success_message("Image processed successfully!")
    
    # Example: Render confirmation dialog
    print("\\nRendering confirmation dialog...")
    # Note: This would require user input in a real application
    # confirmed = renderer.render_confirmation_dialog("Do you want to delete this session?")
    # print(f"User confirmed: {confirmed}")
    
    # Example: Render prompt dialog
    print("\\nRendering prompt dialog...")
    # Note: This would require user input in a real application
    # user_input = renderer.render_prompt_dialog("Enter your name:", default="Anonymous")
    # print(f"User input: {user_input}")
    
    print("\\nRich rendering examples completed!")
```

### ANSI Art Conversion Implementation Example
Image to ANSI art conversion for terminal display:

```python
from PIL import Image
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import logging
from rich.console import Console
from rich.text import Text

class ANSIConverter:
    """
    Converts images to ANSI art for terminal display.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.console = Console()
        
        # Character sets for different rendering styles
        self.character_sets = {
            "blocks": "█▓▒░ ",  # From darkest to lightest
            "ascii": "@%#*+=-:. ",  # ASCII characters from darkest to lightest
            "braille": "⣿⣶⣤⣀⡀ ",  # Braille characters for high resolution
            "simple": "██░░ "  # Simple block characters
        }
    
    def image_to_ansi_art(self, 
                         image_path: str, 
                         max_width: int = 80,
                         character_set: str = "blocks",
                         invert: bool = False,
                         color: bool = False) -> str:
        """
        Convert an image to ANSI art representation for terminal display.
        
        Args:
            image_path: Path to the image to convert
            max_width: Maximum width in terminal characters
            character_set: Character set to use for conversion
            invert: Whether to invert the image colors
            color: Whether to use color ANSI codes
            
        Returns:
            String containing ANSI art representation
        """
        # Validate the input file path exists and is accessible
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        if not Path(image_path).is_file():
            raise ValueError(f"Path is not a file: {image_path}")
        
        # Load the image from the provided file path
        try:
            image = Image.open(image_path)
        except Exception as e:
            raise ValueError(f"Failed to load image: {str(e)}")
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Invert colors if requested
        if invert:
            image = Image.eval(image, lambda x: 255 - x)
        
        # Calculate the appropriate dimensions to fit within max_width while preserving aspect ratio
        original_width, original_height = image.size
        
        # Account for terminal character aspect ratio (characters are taller than wide)
        # Typical terminal character aspect ratio is about 2:1 (height:width)
        char_aspect_ratio = 2.0
        
        # Calculate new dimensions
        aspect_ratio = original_width / (original_height * char_aspect_ratio)
        new_width = min(max_width, original_width)
        new_height = int((new_width / aspect_ratio) / 2)  # Divide by 2 for character aspect ratio
        
        # Ensure minimum height
        new_height = max(1, new_height)
        
        # Resize the image to the calculated dimensions
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to numpy array for easier processing
        img_array = np.array(resized_image)
        
        # Map the image pixels to terminal characters using ANSI color codes
        ansi_art_lines = []
        
        # Get character set
        chars = self.character_sets.get(character_set, self.character_sets["blocks"])
        
        # Process each row of pixels
        for y in range(new_height):
            line_chars = []
            
            for x in range(new_width):
                # Get RGB values for the pixel
                r, g, b = img_array[y, x]
                
                # Calculate brightness for character selection
                brightness = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
                char_index = int(brightness * (len(chars) - 1))
                char = chars[char_index]
                
                # Generate ANSI escape sequences if color is enabled
                if color:
                    ansi_escape = f"\\033[38;2;{r};{g};{b}m{char}\\033[0m"
                else:
                    ansi_escape = char
                
                line_chars.append(ansi_escape)
            
            # Join characters for this line
            ansi_art_lines.append("".join(line_chars))
        
        # Return the complete string that displays the image when printed to the terminal
        return "\\n".join(ansi_art_lines)
    
    def image_to_ansi_art_advanced(self,
                                 image_path: str,
                                 max_width: int = 80,
                                 character_set: str = "blocks",
                                 invert: bool = False,
                                 color: bool = False,
                                 dither: bool = False,
                                 background_color: Optional[tuple] = None) -> str:
        """
        Advanced image to ANSI art conversion with additional options.
        
        Args:
            image_path: Path to the image to convert
            max_width: Maximum width in terminal characters
            character_set: Character set to use for conversion
            invert: Whether to invert the image colors
            color: Whether to use color ANSI codes
            dither: Whether to apply dithering for better quality
            background_color: Optional background color (R, G, B)
            
        Returns:
            String containing ANSI art representation
        """
        # Validate the input file path exists and is accessible
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        if not Path(image_path).is_file():
            raise ValueError(f"Path is not a file: {image_path}")
        
        # Load the image from the provided file path
        try:
            image = Image.open(image_path)
        except Exception as e:
            raise ValueError(f"Failed to load image: {str(e)}")
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Invert colors if requested
        if invert:
            image = Image.eval(image, lambda x: 255 - x)
        
        # Apply dithering if requested
        if dither:
            image = image.convert('P', palette=Image.ADAPTIVE, colors=256)
            image = image.convert('RGB')
        
        # Calculate the appropriate dimensions
        original_width, original_height = image.size
        char_aspect_ratio = 2.0  # Terminal character aspect ratio
        
        # Calculate new dimensions
        aspect_ratio = original_width / (original_height * char_aspect_ratio)
        new_width = min(max_width, original_width)
        new_height = int((new_width / aspect_ratio) / 2)  # Divide by 2 for character aspect ratio
        new_height = max(1, new_height)
        
        # Resize the image
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        img_array = np.array(resized_image)
        
        # Map pixels to terminal characters
        ansi_art_lines = []
        chars = self.character_sets.get(character_set, self.character_sets["blocks"])
        
        # Process each row of pixels
        for y in range(new_height):
            line_chars = []
            
            for x in range(new_width):
                # Get RGB values
                r, g, b = img_array[y, x]
                
                # Calculate brightness for character selection
                brightness = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
                char_index = int(brightness * (len(chars) - 1))
                char = chars[char_index]
                
                # Generate ANSI escape sequences
                if color:
                    if background_color:
                        bg_r, bg_g, bg_b = background_color
                        ansi_escape = f"\\033[38;2;{r};{g};{b};48;2;{bg_r};{bg_g};{bg_b}m{char}\\033[0m"
                    else:
                        ansi_escape = f"\\033[38;2;{r};{g};{b}m{char}\\033[0m"
                else:
                    ansi_escape = char
                
                line_chars.append(ansi_escape)
            
            # Join characters for this line
            ansi_art_lines.append("".join(line_chars))
        
        # Return the complete string
        return "\\n".join(ansi_art_lines)
    
    def image_to_ansi_art_half_blocks(self,
                                    image_path: str,
                                    max_width: int = 80,
                                    color: bool = True) -> str:
        """
        Convert image to ANSI art using half-block characters for better resolution.
        
        Args:
            image_path: Path to the image to convert
            max_width: Maximum width in terminal characters
            color: Whether to use color ANSI codes
            
        Returns:
            String containing ANSI art representation
        """
        # Validate the input file path exists and is accessible
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        if not Path(image_path).is_file():
            raise ValueError(f"Path is not a file: {image_path}")
        
        # Load the image from the provided file path
        try:
            image = Image.open(image_path)
        except Exception as e:
            raise ValueError(f"Failed to load image: {str(e)}")
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Calculate dimensions
        original_width, original_height = image.size
        char_aspect_ratio = 1.0  # Half blocks provide better resolution
        
        # Calculate new dimensions
        aspect_ratio = original_width / (original_height * char_aspect_ratio)
        new_width = min(max_width, original_width)
        new_height = int((new_width / aspect_ratio) / 2)  # Divide by 2 for character aspect ratio
        new_height = max(2, new_height)  # Ensure at least 2 rows for half-block processing
        
        # Resize the image
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        img_array = np.array(resized_image)
        
        # Use half-block characters for better resolution
        ansi_art_lines = []
        
        # Process pairs of rows
        for y in range(0, new_height, 2):
            line_chars = []
            
            for x in range(new_width):
                # Upper pixel
                r1, g1, b1 = img_array[y, x]
                
                # Lower pixel (if exists)
                if y + 1 < new_height:
                    r2, g2, b2 = img_array[y + 1, x]
                else:
                    # Use background color for lower half if no pixel
                    r2, g2, b2 = r1, g1, b1
                
                # Convert RGB to ANSI 256-color code
                if color:
                    # Use half-block character with foreground and background colors
                    char = "▀"  # Upper half block
                    ansi_escape = f"\\033[38;2;{r1};{g1};{b1};48;2;{r2};{g2};{b2}m{char}\\033[0m"
                else:
                    # Use simple character mapping
                    brightness1 = (0.299 * r1 + 0.587 * g1 + 0.114 * b1) / 255.0
                    brightness2 = (0.299 * r2 + 0.587 * g2 + 0.114 * b2) / 255.0
                    
                    # Use different characters based on brightness combination
                    if brightness1 > 0.7 and brightness2 > 0.7:
                        char = " "
                    elif brightness1 > 0.7:
                        char = "▀"
                    elif brightness2 > 0.7:
                        char = "▄"
                    else:
                        char = "█"
                    
                    ansi_escape = char
                
                line_chars.append(ansi_escape)
            
            # Join characters for this line
            ansi_art_lines.append("".join(line_chars))
        
        # Return the complete string
        return "\\n".join(ansi_art_lines)
    
    def save_ansi_art(self, ansi_art: str, output_path: str) -> bool:
        """
        Save ANSI art to a file.
        
        Args:
            ansi_art: ANSI art string to save
            output_path: Path to save the ANSI art to
            
        Returns:
            Boolean indicating success
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(ansi_art)
            self.logger.info(f"ANSI art saved to: {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save ANSI art: {str(e)}")
            return False
    
    def display_ansi_art(self, ansi_art: str) -> None:
        """
        Display ANSI art in the terminal.
        
        Args:
            ansi_art: ANSI art string to display
        """
        print(ansi_art)
    
    def get_ansi_art_info(self, image_path: str) -> Dict[str, Any]:
        """
        Get information about ANSI art conversion.
        
        Args:
            image_path: Path to the image to analyze
            
        Returns:
            Dictionary with conversion information
        """
        try:
            image = Image.open(image_path)
            original_width, original_height = image.size
            
            return {
                "original_size": f"{original_width}x{original_height}",
                "original_format": image.format,
                "original_mode": image.mode,
                "file_size_kb": round(Path(image_path).stat().st_size / 1024, 2),
                "aspect_ratio": round(original_width / original_height, 2),
                "color_channels": len(image.getbands()),
                "estimated_terminal_width": min(80, original_width),
                "estimated_terminal_height": min(24, int(original_height / 2))
            }
        except Exception as e:
            self.logger.error(f"Failed to get ANSI art info: {str(e)}")
            return {
                "error": str(e)
            }

# Example usage
if __name__ == "__main__":
    # Initialize ANSI converter
    converter = ANSIConverter()
    
    print("ANSI Converter initialized")
    
    # Example: Convert image to ANSI art
    try:
        # Create a simple test image for demonstration
        test_image = Image.new('RGB', (100, 100), color=(73, 109, 137))
        test_image.save("test_image.jpg")
        
        # Convert to ANSI art
        ansi_art = converter.image_to_ansi_art(
            "test_image.jpg",
            max_width=40,
            character_set="blocks",
            invert=False,
            color=True
        )
        
        print(f"Converted ANSI art length: {len(ansi_art)} characters")
        print("ANSI art preview (first 200 chars):")
        print(ansi_art[:200])
        
        # Example: Advanced conversion
        advanced_ansi_art = converter.image_to_ansi_art_advanced(
            "test_image.jpg",
            max_width=50,
            character_set="ascii",
            invert=False,
            color=True,
            dither=True,
            background_color=(0, 0, 0)
        )
        
        print(f"\\nAdvanced ANSI art length: {len(advanced_ansi_art)} characters")
        
        # Example: Half-block conversion
        half_block_ansi_art = converter.image_to_ansi_art_half_blocks(
            "test_image.jpg",
            max_width=30,
            color=True
        )
        
        print(f"\\nHalf-block ANSI art length: {len(half_block_ansi_art)} characters")
        
        # Example: Get conversion info
        info = converter.get_ansi_art_info("test_image.jpg")
        print(f"\\nConversion info: {info}")
        
        # Example: Save ANSI art
        if converter.save_ansi_art(ansi_art, "test_ansi_art.txt"):
            print("\\nANSI art saved successfully")
        else:
            print("\\nFailed to save ANSI art")
        
        # Example: Display ANSI art
        print("\\nDisplaying ANSI art:")
        converter.display_ansi_art(ansi_art[:200])  # Show first 200 chars
        
        # Clean up test image
        Path("test_image.jpg").unlink(missing_ok=True)
        Path("test_ansi_art.txt").unlink(missing_ok=True)
        
        print("\\nANSI conversion examples completed!")
        
    except Exception as e:
        print(f"Error in ANSI conversion example: {e}")
```