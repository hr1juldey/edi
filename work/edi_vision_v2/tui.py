#!/usr/bin/env python3
"""
EDI Vision Pipeline - Text User Interface

This module provides the TUI interface for the vision pipeline,
designed for use by human users with interactive exploration.
"""

from textual.app import App, ComposeResult
from textual.screen import Screen
from textual.widgets import (
    Header, Footer, Button, Input, Label, ProgressBar, DirectoryTree,
    DataTable, Static, Markdown, Switch
)
from textual.containers import Container, Vertical, Horizontal, ScrollableContainer
from textual.reactive import reactive
import asyncio
from pathlib import Path
import cv2
import sys
import os

# Add the project directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.orchestrator import VisionPipeline


def image_to_ansi_art(image_path: str, width: int = 50) -> str:
    """Convert an image to ANSI art for terminal display."""
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return "Image could not be loaded"
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Calculate height to maintain aspect ratio
        h, w = image.shape[:2]
        aspect_ratio = h / w
        height = int(width * aspect_ratio * 0.5)  # Multiply by 0.5 to account for character height
        
        # Resize image
        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        
        # Convert to ANSI art
        ansi_art = []
        for y in range(height):
            line = []
            for x in range(width):
                if x < width and y < height:
                    r, g, b = resized[y, x]
                    # Use ANSI color code
                    color_code = f"\033[48;2;{r};{g};{b}m \033[0m"
                    line.append(color_code)
            ansi_art.append("".join(line))
        
        return "\n".join(ansi_art)
    except Exception:
        return "Could not convert image to ANSI art"


class EDIVisionApp(App):
    """Main application class for the EDI Vision TUI."""
    
    CSS_PATH = "tui.tcss"
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("h", "help", "Help"),
        ("escape", "back", "Back"),
        ("ctrl+d", "toggle_dark", "Toggle Dark Mode"),
    ]
    
    # Reactive variables
    current_image: reactive[str | None] = reactive(None)
    current_prompt: reactive[str] = reactive("")
    pipeline_result: reactive[dict | None] = reactive(None)


class WelcomeScreen(Screen):
    """Welcome screen with introduction and start button."""
    
    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="welcome-container"):
            yield Label("Welcome to EDI Vision Pipeline", id="welcome-title")
            yield Label("Detect and segment entities for image editing", id="welcome-subtitle")
            
            markdown_content = """
# EDI Vision Pipeline

**What it does:**
- Analyzes images to find specific objects based on your description
- Creates precise masks for each detected object
- Enables targeted editing of specific image elements

**How to use:**
1. Select an image file
2. Describe what you want to edit (e.g., "blue roofs", "red cars")
3. View the detected entities
4. Use the results in your image editing workflow

**Features:**
- Color-based object detection
- Semantic filtering with CLIP
- Pixel-perfect segmentation masks
- Validation with VLM
            """
            yield Markdown(markdown_content)
            
            yield Button("Start New Analysis", id="start-button", variant="primary")
        
        yield Footer()
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "start-button":
            self.app.push_screen(ImageSelectionScreen())


class ImageSelectionScreen(Screen):
    """Screen for selecting an image file."""
    
    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="image-selection-layout"):
            # Directory tree for file selection
            yield DirectoryTree(self.app.directory, id="file-tree")
            
            # Image preview and info panel
            with Vertical(id="preview-container"):
                yield Label("Image Preview", id="preview-title")
                yield Static("", id="image-preview", classes="preview-box")
                yield Label("No image selected", id="image-info")
                yield Button("Select Image", id="select-button", disabled=True)
        
        # Navigation controls
        with Horizontal(id="nav-buttons"):
            yield Button("Back", id="back-button", variant="warning")
            yield Button("Next", id="next-button", variant="primary", disabled=True)
        
        yield Footer()
    
    def on_mount(self) -> None:
        # Set starting directory to current working directory
        self.app.directory = Path.cwd()
        if hasattr(self.app, 'starting_image') and self.app.starting_image:
            # If we were given a starting image, preselect it
            starting_path = Path(self.app.starting_image).parent
            self.app.directory = starting_path
            self.query_one("#file-tree").path = str(starting_path)
    
    def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected) -> None:
        """Handle when a file is selected in the directory tree."""
        selected_path = Path(event.path)
        
        # Check if it's an image file
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}
        if selected_path.suffix.lower() in image_extensions:
            # Update image preview
            preview = self.query_one("#image-preview")
            preview.update(image_to_ansi_art(str(selected_path), width=40))
            
            # Update image info
            info_label = self.query_one("#image-info")
            try:
                img = cv2.imread(str(selected_path))
                if img is not None:
                    h, w = img.shape[:2]
                    size_mb = selected_path.stat().st_size / (1024 * 1024)
                    info_label.update(f"Dimensions: {w}x{h}, Size: {size_mb:.2f}MB")
                    
                    # Enable the select button
                    select_btn = self.query_one("#select-button")
                    select_btn.disabled = False
                    self.selected_image_path = str(selected_path)
                    
                    # Enable next button
                    next_btn = self.query_one("#next-button")
                    next_btn.disabled = False
                else:
                    info_label.update("Invalid image file")
                    self.query_one("#select-button").disabled = True
                    self.query_one("#next-button").disabled = True
            except Exception:
                info_label.update("Error reading image")
                self.query_one("#select-button").disabled = True
                self.query_one("#next-button").disabled = True
        else:
            # Not an image file
            preview = self.query_one("#image-preview")
            preview.update("Select an image file")
            info_label = self.query_one("#image-info")
            info_label.update("Not an image file")
            self.query_one("#select-button").disabled = True
            self.query_one("#next-button").disabled = True
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back-button":
            self.app.pop_screen()
        elif event.button.id == "select-button":
            # Store the selected image in the app
            self.app.current_image = self.selected_image_path
            # The image was already previewed and validated, so just enable the next button
        elif event.button.id == "next-button" and hasattr(self, 'selected_image_path'):
            self.app.current_image = self.selected_image_path
            self.app.push_screen(PromptInputScreen())


class PromptInputScreen(Screen):
    """Screen for entering the edit prompt."""
    
    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="prompt-input-container"):
            yield Label("Enter your edit request", id="prompt-title")
            
            with Container(id="input-wrapper"):
                # Text input for the prompt
                yield Input(
                    placeholder="e.g., 'change blue roofs to green', 'highlight red cars'",
                    id="prompt-input",
                    value=self.app.current_prompt
                )
            
            # Settings panel
            with Vertical(id="settings-panel"):
                yield Label("Settings", id="settings-title")
                
                # Validation toggle
                yield Switch(value=True, id="validation-switch", animate=False)
                yield Label("Enable VLM validation", id="validation-label")
                
                # Save steps toggle
                yield Switch(value=False, id="save-steps-switch", animate=False)
                yield Label("Save intermediate steps", id="save-steps-label")
        
        # Navigation controls
        with Horizontal(id="nav-buttons"):
            yield Button("Back", id="back-button", variant="warning")
            yield Button("Analyze Image", id="analyze-button", variant="primary")
        
        yield Footer()
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back-button":
            self.app.pop_screen()
        elif event.button.id == "analyze-button":
            # Get the prompt
            prompt_input = self.query_one("#prompt-input", Input)
            prompt = prompt_input.value.strip()
            
            if not prompt:
                # Show error message
                error_label = Label("Please enter a prompt", id="error-label")
                self.mount(error_label)
                return
            
            # Store the prompt in the app
            self.app.current_prompt = prompt
            
            # Collect settings
            validation_enabled = self.query_one("#validation-switch", Switch).value
            save_steps = self.query_one("#save-steps-switch", Switch).value
            
            # Store settings in app for use in processing
            self.app.validation_enabled = validation_enabled
            self.app.save_steps = save_steps
            
            # Navigate to processing screen
            self.app.push_screen(ProcessingScreen())
    
    def on_input_changed(self, event: Input.Changed) -> None:
        # Enable/disable analyze button based on input
        analyze_btn = self.query_one("#analyze-button")
        analyze_btn.disabled = not bool(event.value.strip())


class ProcessingScreen(Screen):
    """Screen showing live progress during pipeline execution."""
    
    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="processing-container"):
            yield Label("Processing Image...", id="processing-title")
            
            # Progress bars for each stage
            yield Label("Stage 1: Entity Extraction", id="stage1-label")
            yield ProgressBar(id="stage1-progress", total=1)
            
            yield Label("Stage 2: Color Filtering", id="stage2-label")
            yield ProgressBar(id="stage2-progress", total=1)
            
            yield Label("Stage 3: SAM Segmentation", id="stage3-label")
            yield ProgressBar(id="stage3-progress", total=1)
            
            yield Label("Stage 4: CLIP Filtering", id="stage4-label")
            yield ProgressBar(id="stage4-progress", total=1)
            
            yield Label("Stage 5: Mask Organization", id="stage5-label")
            yield ProgressBar(id="stage5-progress", total=1)
            
            yield Label("Stage 6: VLM Validation", id="stage6-label")
            yield ProgressBar(id="stage6-progress", total=1)
            
            # Log messages panel
            yield Label("Log Messages:", id="log-title")
            with ScrollableContainer(id="log-container"):
                yield Static(id="log-output", markup=True)
        
        # Cancel button
        with Horizontal(id="nav-buttons"):
            yield Button("Cancel", id="cancel-button", variant="error")
        
        yield Footer()
    
    async def on_mount(self) -> None:
        # Start the pipeline in a worker thread
        self.run_worker(self.run_pipeline(), exclusive=True)
    
    async def run_pipeline(self) -> None:
        """Run the vision pipeline in the background."""
        # Initialize pipeline with settings from the app
        pipeline = VisionPipeline(
            enable_validation=getattr(self.app, 'validation_enabled', True),
            save_intermediate=getattr(self.app, 'save_steps', False),
            output_dir="logs"
        )
        
        # Update log
        self.update_log("[bold blue]Starting pipeline...[/bold blue]")
        
        try:
            # Start processing
            result = await self.run_task(pipeline.process, self.app.current_image, self.app.current_prompt)
            
            # Store the result in the app
            self.app.pipeline_result = result
            
            if result['success']:
                self.update_log(f"[bold green]✅ Success! Detected {len(result['entity_masks'])} entities[/bold green]")
                self.update_log(f"[bold green]Total time: {result['metadata']['total_time']:.1f} seconds[/bold green]")
                
                # After a short delay, navigate to results screen
                await asyncio.sleep(1.5)  # Brief delay to show success
                self.app.push_screen(ResultsScreen())
            else:
                self.update_log(f"[bold red]❌ Pipeline failed: {result['error']}[/bold red]")
                # TODO: Show error screen or allow retry
        
        except Exception as e:
            self.update_log(f"[bold red]❌ Error: {str(e)}[/bold red]")
    
    def update_log(self, message: str) -> None:
        """Update the log with a new message."""
        log_output = self.query_one("#log-output", Static)
        current_content = log_output.renderable if log_output.renderable else ""
        new_content = current_content + f"\n{message}" if current_content else message
        log_output.update(new_content)
        
        # Scroll to bottom
        log_container = self.query_one("#log-container")
        log_container.scroll_end(animate=False)
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel-button":
            # For now, just go back. In a real implementation, 
            # this would cancel the underlying pipeline process
            self.app.pop_screen()


class ResultsScreen(Screen):
    """Screen showing the results of the pipeline."""
    
    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="results-container"):
            yield Label("Analysis Results", id="results-title")
            
            # Summary panel
            with Horizontal(id="summary-panel"):
                yield Label("Entities detected: --", id="entity-count")
                yield Label("Total time: --s", id="total-time")
                if self.app.pipeline_result and 'validation' in self.app.pipeline_result and self.app.pipeline_result['validation']:
                    val_conf = self.app.pipeline_result['validation'].confidence if hasattr(self.app.pipeline_result['validation'], 'confidence') else 0
                    yield Label(f"VLM confidence: {val_conf:.2f}", id="validation-conf")
            
            # Results table
            yield DataTable(id="results-table", show_header=True, zebra_stripes=True)
        
        # Action buttons
        with Horizontal(id="action-buttons"):
            yield Button("New Analysis", id="new-button", variant="primary")
            yield Button("Save Report", id="save-button", variant="success")
            yield Button("Export Masks", id="export-button", variant="default")
        
        yield Footer()
    
    def on_mount(self) -> None:
        """Populate the results when the screen is mounted."""
        result = self.app.pipeline_result
        if not result or not result['success']:
            return
        
        # Update summary
        entity_count = len(result['entity_masks'])
        self.query_one("#entity-count").update(f"Entities detected: {entity_count}")
        
        total_time = result['metadata']['total_time']
        self.query_one("#total-time").update(f"Total time: {total_time:.1f}s")
        
        # Populate table
        table = self.query_one("#results-table", DataTable)
        
        # Add headers
        table.add_columns("ID", "Area (px)", "Confidence", "Centroid", "Bounding Box")
        
        # Add rows for each entity mask
        for entity in result['entity_masks']:
            centroid_str = f"({entity.centroid[0]:.1f}, {entity.centroid[1]:.1f})"
            bbox_str = f"({entity.bbox[0]}, {entity.bbox[1]}, {entity.bbox[2]}, {entity.bbox[3]})"
            table.add_row(
                str(entity.entity_id),
                str(entity.area),
                f"{entity.similarity_score:.3f}",
                centroid_str,
                bbox_str
            )
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "new-button":
            # Go back to the beginning
            self.app.pop_screen()  # Go back to prompt input
            self.app.pop_screen()  # Go back to image selection 
            self.app.pop_screen()  # Go back to welcome screen
        elif event.button.id == "save-button":
            # Save a report of the results
            # For now just show a confirmation
            self.notify("Report saved successfully")
        elif event.button.id == "export-button":
            # Export the masks
            # For now just show a confirmation
            self.notify("Masks exported successfully")


def main():
    """Main entry point for the TUI application."""
    import argparse
    
    parser = argparse.ArgumentParser(description="EDI Vision Pipeline - Text User Interface")
    parser.add_argument("--image", help="Pre-select an image file")
    parser.add_argument("--config", help="Path to config file (not used in TUI but kept for consistency)")
    
    args = parser.parse_args()
    
    app = EDIVisionApp()
    
    # Store starting image if provided
    if args.image:
        app.starting_image = args.image
    
    app.run()


if __name__ == "__main__":
    main()