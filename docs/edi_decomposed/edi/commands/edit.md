# Commands: Edit

[Back to Index](./index.md)

## Purpose
Main edit command - Contains the edit_command async function that serves as the entry point for `edi edit` command, launching the Textual app or running in headless mode.

## Functions
- `async def edit_command(image_path, prompt, **kwargs)`: Main entry point for editing functionality

### Details
- Entry point for `edi edit` CLI command
- Launches Textual app or runs headless mode
- Handles command line arguments and options

## Technology Stack

- AsyncIO for asynchronous operations
- CLI argument parsing
- Textual for TUI interface

## See Docs

### AsyncIO Implementation Example
Async edit command for the EDI application:

```python
import asyncio
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
import sys
import json
from dataclasses import dataclass

class EDIEditManager:
    """Manages the EDI edit process asynchronously."""
    
    def __init__(self):
        self.session_data: Dict[str, Any] = {}
    
    async def validate_input(self, image_path: str, prompt: str) -> bool:
        """Validate input parameters."""
        # Check if image exists
        if not Path(image_path).exists():
            print(f"Error: Image file does not exist: {image_path}")
            return False
        
        # Check if prompt is not empty
        if not prompt.strip():
            print("Error: Prompt cannot be empty")
            return False
        
        # Validate image format
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        ext = Path(image_path).suffix.lower()
        if ext not in valid_extensions:
            print(f"Error: Invalid image format. Supported formats: {valid_extensions}")
            return False
        
        return True
    
    async def run_edit_pipeline(self, image_path: str, prompt: str, headless: bool = False) -> Dict[str, Any]:
        """Run the complete edit pipeline."""
        print(f"Starting edit pipeline for '{image_path}' with prompt: '{prompt}'")
        
        # This would call the orchestrator in a real implementation
        # For now, we'll simulate the process
        result = await self.simulate_edit_process(image_path, prompt)
        
        if headless:
            # In headless mode, return results directly
            return result
        else:
            # In TUI mode, this would launch the Textual interface
            await self.launch_tui_interface(result)
            return result
    
    async def simulate_edit_process(self, image_path: str, prompt: str) -> Dict[str, Any]:
        """Simulate the edit process (in a real implementation, this would call the orchestrator)."""
        # Simulate various steps of the edit process
        print("  - Analyzing image...")
        await asyncio.sleep(0.5)  # Simulate processing time
        
        print("  - Parsing user intent...")
        await asyncio.sleep(0.3)  # Simulate processing time
        
        print("  - Generating prompts...")
        await asyncio.sleep(0.4)  # Simulate processing time
        
        print("  - Processing edit...")
        await asyncio.sleep(1.0)  # Simulate processing time
        
        print("  - Validating results...")
        await asyncio.sleep(0.2)  # Simulate processing time
        
        # Generate a mock result
        output_path = str(Path(image_path).with_suffix('')) + '_edited' + Path(image_path).suffix
        return {
            "status": "success",
            "output_path": output_path,
            "prompt_used": prompt,
            "processing_time": 2.4,
            "quality_score": 0.85
        }
    
    async def launch_tui_interface(self, initial_result: Dict[str, Any]):
        """Launch the Textual TUI interface for interactive editing."""
        # This would import and run the main Textual app in a real implementation
        print("\nLaunching Textual TUI interface...")
        print(f"Initial result: {initial_result}")
        
        # In a real implementation, this would run something like:
        # from edi.tui.main import EDIMainApp
        # app = EDIMainApp(initial_result=initial_result)
        # app.run()
        
        print("TUI interface would start here in a real implementation")
    
    async def save_session_data(self, session_path: str):
        """Save session data to a file."""
        async with asyncio.Lock():
            with open(session_path, 'w') as f:
                json.dump(self.session_data, f, indent=2)
    
    async def load_session_data(self, session_path: str) -> Dict[str, Any]:
        """Load session data from a file."""
        try:
            with open(session_path, 'r') as f:
                self.session_data = json.load(f)
                return self.session_data
        except FileNotFoundError:
            return {}

async def edit_command(image_path: str, prompt: str, **kwargs) -> Dict[str, Any]:
    """Main entry point for editing functionality."""
    # Extract additional options
    headless = kwargs.get('headless', False)
    output_path = kwargs.get('output_path', None)
    quality = kwargs.get('quality', 'medium')
    
    print(f"EDI Edit Command")
    print(f"  Image: {image_path}")
    print(f"  Prompt: {prompt}")
    print(f"  Headless: {headless}")
    print(f"  Quality: {quality}")
    
    # Validate inputs
    edit_manager = EDIEditManager()
    is_valid = await edit_manager.validate_input(image_path, prompt)
    
    if not is_valid:
        print("Input validation failed. Exiting.")
        return {"status": "error", "message": "Input validation failed"}
    
    # Run the edit pipeline
    result = await edit_manager.run_edit_pipeline(image_path, prompt, headless)
    
    if result["status"] == "success":
        if output_path:
            # If output path specified, save to that path instead
            Path(result["output_path"]).rename(output_path)
            result["output_path"] = output_path
            print(f"Result saved to: {output_path}")
        else:
            print(f"Result saved to: {result['output_path']}")
        
        print(f"Quality score: {result['quality_score']:.2f}")
        print(f"Processing time: {result['processing_time']:.1f}s")
    
    return result

# Example usage for command line interface
async def main():
    parser = argparse.ArgumentParser(description='EDI - Edit with Intelligence')
    parser.add_argument('image_path', help='Path to the image to edit')
    parser.add_argument('prompt', help='Prompt describing the desired edit')
    parser.add_argument('--headless', action='store_true', 
                       help='Run in headless mode without TUI')
    parser.add_argument('--quality', choices=['low', 'medium', 'high'], 
                       default='medium', help='Quality level for processing')
    parser.add_argument('--output', '-o', help='Output path for the edited image')
    parser.add_argument('--model', help='Model to use for editing')
    
    args = parser.parse_args()
    
    # Run the edit command
    result = await edit_command(
        image_path=args.image_path,
        prompt=args.prompt,
        headless=args.headless,
        quality=args.quality,
        output_path=args.output,
        model=args.model
    )
    
    # Exit with appropriate code
    if result.get("status") == "error":
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    # This would normally be called from the main CLI entry point
    # For this example, we'll simulate a call
    async def simulate_call():
        # Example of calling the edit command
        result = await edit_command(
            image_path="example.jpg",
            prompt="make the sky more dramatic with storm clouds",
            headless=True,
            quality="high"
        )
        
        print(f"\nEdit command result: {result}")
    
    asyncio.run(simulate_call())
```

### CLI Argument Parsing Implementation Example
Command line interface for EDI edit command:

```python
import argparse
import sys
from pathlib import Path
from typing import Optional
import json
import asyncio

class EDICLI:
    """Command line interface for EDI."""
    
    def __init__(self):
        self.parser = self.create_parser()
    
    def create_parser(self) -> argparse.ArgumentParser:
        """Create the main argument parser."""
        parser = argparse.ArgumentParser(
            prog='edi',
            description='EDI - Edit with Intelligence: AI-powered image editing',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  edi edit image.jpg "make the sky more dramatic"
  edi edit image.png "enhance colors and contrast" --headless
  edi edit photo.jpg "change background to blue" --output result.jpg
  edi edit image.jpg "improve lighting" --quality high --model qwen3:8b
            """
        )
        
        # Add subcommands
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Edit command
        edit_parser = subparsers.add_parser('edit', help='Edit an image with AI')
        edit_parser.add_argument('image_path', help='Path to the image to edit')
        edit_parser.add_argument('prompt', help='Prompt describing the desired edit')
        edit_parser.add_argument('--headless', action='store_true', 
                                help='Run in headless mode without TUI')
        edit_parser.add_argument('--quality', choices=['low', 'medium', 'high'], 
                                default='medium', help='Quality level for processing')
        edit_parser.add_argument('-o', '--output', help='Output path for the edited image')
        edit_parser.add_argument('--model', help='Model to use for editing')
        edit_parser.add_argument('--batch', action='store_true', 
                                help='Process multiple images in batch mode')
        edit_parser.add_argument('--save-session', help='Save session to a file')
        edit_parser.add_argument('--load-session', help='Load session from a file')
        edit_parser.add_argument('--verbose', '-v', action='store_true', 
                                help='Enable verbose output')
        
        # Setup command
        setup_parser = subparsers.add_parser('setup', help='Setup EDI environment')
        setup_parser.add_argument('--download-models', action='store_true', 
                                 help='Download default models during setup')
        setup_parser.add_argument('--force', action='store_true', 
                                 help='Force reinstallation')
        
        # Doctor command
        doctor_parser = subparsers.add_parser('doctor', help='Check EDI environment')
        doctor_parser.add_argument('--verbose', '-v', action='store_true', 
                                  help='Enable verbose output')
        
        # Clear command
        clear_parser = subparsers.add_parser('clear', help='Clear EDI cache and temporary files')
        clear_parser.add_argument('--sessions', action='store_true', 
                                 help='Clear session data')
        clear_parser.add_argument('--cache', action='store_true', 
                                 help='Clear cache files')
        clear_parser.add_argument('--all', action='store_true', 
                                 help='Clear all data')
        
        return parser
    
    def validate_args(self, args) -> bool:
        """Validate command line arguments."""
        if args.command == 'edit':
            # Validate image path
            if not Path(args.image_path).exists():
                print(f"Error: Image file does not exist: {args.image_path}")
                return False
            
            # Validate image format for edit command
            valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
            ext = Path(args.image_path).suffix.lower()
            if ext not in valid_extensions:
                print(f"Error: Invalid image format: {ext}. Supported formats: {valid_extensions}")
                return False
            
            # Validate prompt
            if not args.prompt.strip():
                print("Error: Prompt cannot be empty")
                return False
        
        return True
    
    def run(self, args=None):
        """Run the CLI with the given arguments."""
        if args is None:
            args = sys.argv[1:]
        
        if not args:
            self.parser.print_help()
            return 1
        
        parsed_args = self.parser.parse_args(args)
        
        # Validate arguments
        if not self.validate_args(parsed_args):
            return 1
        
        # Dispatch to appropriate command handler
        if parsed_args.command == 'edit':
            return self.handle_edit(parsed_args)
        elif parsed_args.command == 'setup':
            return self.handle_setup(parsed_args)
        elif parsed_args.command == 'doctor':
            return self.handle_doctor(parsed_args)
        elif parsed_args.command == 'clear':
            return self.handle_clear(parsed_args)
        else:
            self.parser.print_help()
            return 1
    
    def handle_edit(self, args):
        """Handle the edit command."""
        print(f"Running edit command...")
        print(f"  Image: {args.image_path}")
        print(f"  Prompt: {args.prompt}")
        print(f"  Headless: {args.headless}")
        print(f"  Quality: {args.quality}")
        
        if args.output:
            print(f"  Output: {args.output}")
        
        if args.model:
            print(f"  Model: {args.model}")
        
        if args.verbose:
            print("  Verbose mode enabled")
        
        # In a real implementation, this would call the async edit command
        # For this example, we'll simulate the call
        import asyncio
        result = asyncio.run(
            edit_command(
                image_path=args.image_path,
                prompt=args.prompt,
                headless=args.headless,
                quality=args.quality,
                output_path=args.output,
                model=args.model
            )
        )
        
        if result.get("status") == "success":
            print(f"✓ Edit completed successfully!")
            print(f"  Output: {result.get('output_path')}")
            print(f"  Quality score: {result.get('quality_score')}")
            return 0
        else:
            print(f"✗ Edit failed: {result.get('message', 'Unknown error')}")
            return 1
    
    def handle_setup(self, args):
        """Handle the setup command."""
        print(f"Running setup command...")
        print(f"  Download models: {args.download_models}")
        print(f"  Force reinstallation: {args.force}")
        
        # In a real implementation, this would call the setup command
        import asyncio
        result = asyncio.run(setup_command(download_models=args.download_models))
        
        if result:
            print("✓ Setup completed successfully!")
            return 0
        else:
            print("✗ Setup failed!")
            return 1
    
    def handle_doctor(self, args):
        """Handle the doctor command."""
        print(f"Running doctor command...")
        if args.verbose:
            print("  Verbose mode enabled")
        
        # In a real implementation, this would diagnose the system
        print("  ✓ Checking Ollama connection...")
        print("  ✓ Checking ComfyUI connection...")
        print("  ✓ Checking model availability...")
        print("  ✓ Checking configuration...")
        
        print("✓ Doctor check completed. All systems operational!")
        return 0
    
    def handle_clear(self, args):
        """Handle the clear command."""
        print(f"Running clear command...")
        print(f"  Clear sessions: {args.sessions}")
        print(f"  Clear cache: {args.cache}")
        print(f"  Clear all: {args.all}")
        
        # In a real implementation, this would clear the appropriate data
        print("  Data cleared successfully!")
        return 0

# Mock functions to make the example work
async def edit_command(image_path: str, prompt: str, **kwargs) -> Dict[str, Any]:
    """Mock edit command function."""
    return {
        "status": "success",
        "output_path": image_path.replace(".", "_edited."),
        "quality_score": 0.85
    }

async def setup_command(download_models: bool = False) -> bool:
    """Mock setup command function."""
    return True

from typing import Dict, Any  # Import needed for type hints

# Example usage of the CLI
if __name__ == "__main__":
    # Example of using the CLI programmatically
    cli = EDICLI()
    
    # Example: Parse and handle a command line
    # This would normally be called as: python -m edi edit image.jpg "enhance colors"
    # For this example, we'll simulate it:
    
    # Simulate: edi edit example.jpg "make it brighter"
    print("Simulating: edi edit example.jpg 'make it brighter'")
    exit_code = cli.run(['edit', 'example.jpg', 'make it brighter'])
    print(f"Exit code: {exit_code}\n")
    
    # Simulate: edi edit image.png "enhance" --headless --quality high
    print("Simulating: edi edit image.png 'enhance' --headless --quality high")
    exit_code = cli.run(['edit', 'image.png', 'enhance', '--headless', '--quality', 'high'])
    print(f"Exit code: {exit_code}\n")
    
    # Simulate: edi setup --download-models
    print("Simulating: edi setup --download-models")
    exit_code = cli.run(['setup', '--download-models'])
    print(f"Exit code: {exit_code}\n")
```

### Textual Implementation Example
TUI interface for the edit command:

```python
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Static, Button, TextArea, Input
from textual.containers import Container, Vertical, Horizontal
from textual.screen import ModalScreen
from textual import work
import asyncio
from pathlib import Path
from typing import Optional

class EditConfirmationModal(ModalScreen):
    """Modal screen to confirm edit parameters."""
    
    def __init__(self, image_path: str, prompt: str, quality: str):
        super().__init__()
        self.image_path = image_path
        self.prompt = prompt
        self.quality = quality
    
    def compose(self) -> ComposeResult:
        yield Container(
            Static("Confirm Edit Parameters", classes="modal-title"),
            Container(
                Static(f"Image: {Path(self.image_path).name}"),
                Static(f"Prompt: {self.prompt}"),
                Static(f"Quality: {self.quality}"),
                classes="modal-content"
            ),
            Horizontal(
                Button("Confirm", variant="success", id="confirm"),
                Button("Cancel", variant="error", id="cancel"),
                classes="modal-buttons"
            ),
            classes="modal-container"
        )
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "confirm":
            self.dismiss(True)
        else:
            self.dismiss(False)

class EditScreen(App):
    """Main screen for the edit command TUI."""
    
    CSS = """
    Screen {
        background: $surface;
    }
    
    #edit-container {
        layout: vertical;
        align: center middle;
        width: 100%;
        height: 100%;
        padding: 2;
    }
    
    #image-preview {
        content-align: center middle;
        width: 1fr;
        height: 1fr;
        border: solid $primary;
        margin: 1;
    }
    
    #prompt-container {
        width: 1fr;
        height: auto;
        margin: 1 0;
    }
    
    #quality-selector {
        width: 1fr;
        height: auto;
        margin: 1 0;
    }
    
    #action-buttons {
        width: 1fr;
        height: auto;
        align: center middle;
    }
    
    .modal-container {
        width: 60%;
        height: 60%;
        background: $panel;
        border: thick $accent;
        content-align: center middle;
    }
    
    .modal-title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }
    
    .modal-content {
        width: 1fr;
        height: 1fr;
        margin: 1;
    }
    
    .modal-buttons {
        width: 1fr;
        height: auto;
        margin-top: 1;
    }
    """
    
    def __init__(self, image_path: str, initial_prompt: str = ""):
        super().__init__()
        self.image_path = image_path
        self.initial_prompt = initial_prompt
        self.current_quality = "medium"
    
    def compose(self) -> ComposeResult:
        """Create child widgets."""
        yield Header()
        yield Container(
            Static(f"Editing: {Path(self.image_path).name}", id="image-title"),
            Container(
                Static("Image Preview", id="image-preview"),
                Container(
                    Static("Edit Prompt:", classes="prompt-label"),
                    TextArea(
                        text=self.initial_prompt,
                        id="edit-prompt",
                        classes="prompt-input"
                    ),
                    id="prompt-container"
                ),
                Container(
                    Static("Quality:", classes="quality-label"),
                    Horizontal(
                        Button("Low", id="quality-low"),
                        Button("Medium", variant="primary", id="quality-medium"),
                        Button("High", id="quality-high"),
                        id="quality-selector"
                    ),
                    id="quality-container"
                ),
                Horizontal(
                    Button("Preview", variant="default", id="preview"),
                    Button("Edit", variant="success", id="edit", disabled=True),
                    Button("Cancel", variant="error", id="cancel"),
                    id="action-buttons"
                ),
                id="edit-container"
            ),
            id="main-container"
        )
        yield Footer()
    
    def on_mount(self) -> None:
        """Called when the widget is mounted."""
        # Enable the edit button when there's text in the prompt
        prompt_widget = self.query_one("#edit-prompt", TextArea)
        
        def enable_edit_if_prompted(text_area):
            edit_button = self.query_one("#edit", Button)
            edit_button.disabled = not bool(text_area.text.strip())
        
        prompt_widget.watch("text", enable_edit_if_prompted)
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        if event.button.id == "quality-low":
            self.current_quality = "low"
            self.update_quality_buttons()
        elif event.button.id == "quality-medium":
            self.current_quality = "medium"
            self.update_quality_buttons()
        elif event.button.id == "quality-high":
            self.current_quality = "high"
            self.update_quality_buttons()
        elif event.button.id == "edit":
            self.start_edit_process()
        elif event.button.id == "cancel":
            self.exit()
    
    def update_quality_buttons(self):
        """Update the appearance of quality buttons based on current selection."""
        low_btn = self.query_one("#quality-low", Button)
        medium_btn = self.query_one("#quality-medium", Button)
        high_btn = self.query_one("#quality-high", Button)
        
        # Reset all buttons
        low_btn.variant = "default"
        medium_btn.variant = "default"
        high_btn.variant = "default"
        
        # Set selected button to primary
        if self.current_quality == "low":
            low_btn.variant = "primary"
        elif self.current_quality == "medium":
            medium_btn.variant = "primary"
        elif self.current_quality == "high":
            high_btn.variant = "primary"
    
    def start_edit_process(self):
        """Start the edit process after confirming parameters."""
        prompt_widget = self.query_one("#edit-prompt", TextArea)
        prompt = prompt_widget.text
        
        # Show confirmation modal
        def handle_confirmation(confirmed):
            if confirmed:
                # In a real implementation, this would start the async edit process
                self.process_edit_async(prompt)
        
        self.push_screen(
            EditConfirmationModal(self.image_path, prompt, self.current_quality),
            callback=handle_confirmation
        )
    
    @work(exclusive=True, thread=True)
    def process_edit_async(self, prompt: str):
        """Process the edit asynchronously to not block the UI."""
        # This would call the actual edit pipeline in a real implementation
        # For this example, we'll simulate the process
        import time
        time.sleep(2)  # Simulate processing time
        
        # In a real implementation, this would be:
        # result = await edit_command(self.image_path, prompt, quality=self.current_quality)
        
        # After completion, show result
        self.call_from_thread(self.show_completion_message, prompt)
    
    def show_completion_message(self, prompt: str):
        """Show a message when editing is complete."""
        output_path = str(Path(self.image_path).with_suffix('')) + '_edited' + Path(self.image_path).suffix
        self.notify(f"Edit completed! Result saved to: {output_path}")

# Example usage
if __name__ == "__main__":
    import sys
    
    # Example of how this would be called from the edit command
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        initial_prompt = sys.argv[2] if len(sys.argv) > 2 else ""
    else:
        image_path = "example.jpg"
        initial_prompt = "Enhance the colors and lighting"
    
    app = EditScreen(image_path, initial_prompt)
    app.run()
```