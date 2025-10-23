# TUI Layer

[Back to Index](../index.md)

## Purpose

User interaction, display, navigation using Textual 0.87+

## Component Design

### Architecture

Screen-based navigation with reactive widgets

#### Screen Hierarchy

```bash
HomeScreen
├── ImageUploadScreen
│   ├── FileInput widget
│   ├── PreviewPane widget
│   └── AnalysisProgressBar widget
├── PromptInputScreen
│   ├── TextArea widget (for naive prompt)
│   ├── EntitySelectorList widget (checkboxes)
│   └── SubmitButton widget
├── ClarificationScreen
│   ├── QuestionLabel widget
│   ├── OptionsRadioSet widget
│   └── ConfirmButton widget
├── RefinementScreen
│   ├── IterationProgressBar widget
│   ├── PromptDiffViewer widget (shows evolution)
│   └── ApproveRejectButtons widget
├── ResultsScreen
│   ├── ImageComparisonPane (side-by-side)
│   ├── ValidationMetricsTable widget
│   ├── AcceptRetryButtons widget
│   └── FeedbackTextArea widget
└── MultiVariationScreen
    ├── GridLayout (3 columns)
    ├── VariationCards (A/B/C)
    └── SelectionControls widget
```

#### Key Widgets

**ImageComparisonPane**:

```python
class ImageComparisonPane(Widget):
    """
    Side-by-side image viewer with overlay support.
    """
    def compose(self):
        yield Container(
            Container(id="before-pane"),
            Container(id="after-pane"),
            id="comparison-container"
        )
    
    def render_images(self, before_path, after_path):
        # Convert images to ANSI art using Rich
        before_art = image_to_ansi_art(before_path, max_width=40)
        after_art = image_to_ansi_art(after_path, max_width=40)
        
        self.query_one("#before-pane").update(before_art)
        self.query_one("#after-pane").update(after_art)
```

**PromptDiffViewer**:

```python
class PromptDiffViewer(Widget):
    """
    Shows prompt evolution across refinement iterations.
    """
    def display_refinement(self, iteration, positive, negative):
        # Highlight added tokens in green, removed in red
        diff_positive = self.compute_diff(
            previous=self.prompts[iteration-1].positive,
            current=positive
        )
        
        self.render_diff(diff_positive, panel_title=f"Positive (v{iteration})")
```

#### Navigation Flow

```bash
┌──────────────────────────────────────────────────────────┐
│  [1] Upload Image       [2] Recent Sessions   [Q] Quit   │
├──────────────────────────────────────────────────────────┤
│                                                          │
│                   EDI: Edit with Intelligence            │
│                                                          │
│  Welcome! Let's edit your image together.                │
│                                                          │
│  > [1] Start new edit                                    │
│    [2] Resume session                                    │
│    [3] View examples                                     │
│    [H] Help                                              │
│                                                          │
│                                                          │
│  Navigation: Arrow keys / Numbers                        │
│  Quick actions: [Q]uit  [H]elp  [B]ack                   │
└──────────────────────────────────────────────────────────┘
```

**Keyboard Shortcuts**:

- Global: `Q` quit, `H` help, `B` back, `Ctrl+C` cancel operation
- Navigation: Arrow keys, Tab/Shift+Tab
- Actions: Numbers (1-9) for quick selection, Enter to confirm
- Editing: `E` edit prompt, `R` retry, `A` accept, `V` view variations

## Sub-modules

This component includes the following modules:

- [app.py](./app.md)
- [ui/screens/](./screens/screens.md)
- [ui/widgets/](./widgets/widgets.md)
- [ui/styles/](./styles.md)
- [ui/utils.py](./utils/utils.md)

## Technology Stack

- Textual 0.87+ for TUI framework
- Rich for terminal rendering
- AsyncIO for non-blocking operations

## See Docs

### Textual Implementation Example

Here's a basic Textual application example for an image editing interface:

```python
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Static, Button
from textual.containers import Horizontal, Vertical
from rich.panel import Panel

class EDIApp(App):
    """EDI (Edit with Intelligence) Textual Application."""
    
    TITLE = "EDI: Edit with Intelligence"
    SUB_TITLE = "Image Editing Assistant"
    
    CSS = """
    Screen {
        background: $surface;
        align: center middle;
    }
    #image-container {
        width: 1fr;
        height: 1fr;
        border: round $primary;
        content-align: center middle;
    }
    #controls {
        height: auto;
        width: 1fr;
        margin: 1 0;
    }
    Button {
        margin: 0 1;
    }
    """

    def compose(self) -> ComposeResult:
        """Create child widgets."""
        yield Header()
        yield Vertical(
            Static(Panel("Upload your image to begin editing", title="EDI"), id="image-container"),
            Horizontal(
                Button("Upload Image", id="upload", variant="primary"),
                Button("Edit Prompt", id="prompt", variant="default"),
                Button("Generate", id="generate", variant="success"),
                id="controls"
            ),
            id="main-content"
        )
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        if event.button.id == "upload":
            self.notify("Upload functionality would go here")
        elif event.button.id == "generate":
            self.notify("Generation starting...")

if __name__ == "__main__":
    app = EDIApp()
    app.run()
```

### Rich Formatting Example

Terminal output formatting for EDI application:

```python
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn
from rich import print

console = Console()

def display_image_comparison(before_path: str, after_path: str):
    """Display image comparison results in the terminal."""
    print(Panel(f"[bold]Before:[/] {before_path}\n[bold]After:[/] {after_path}", title="Image Edit Results"))

def display_metrics_table(metrics: dict):
    """Display validation metrics in a table."""
    table = Table(title="Edit Validation Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="magenta")
    
    for key, value in metrics.items():
        table.add_row(key, str(value))
    
    print(table)

def progress_tracker(total_steps: int):
    """Display progress during image processing."""
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    ) as progress:
        task = progress.add_task("[green]Processing...", total=total_steps)
        for step in range(total_steps):
            progress.update(task, advance=1)
            # Simulate processing step
            import time
            time.sleep(0.1)  # Simulate work being done

# Example usage:
if __name__ == "__main__":
    display_image_comparison("input.jpg", "output.jpg")
    
    metrics = {
        "Quality Score": 94.5,
        "Processing Time": "2.3s",
        "File Size Change": "-15%"
    }
    display_metrics_table(metrics)
    
    print("[bold blue]Starting image processing...[/bold blue]")
    progress_tracker(10)
```

### AsyncIO Implementation Example

Asynchronous handling of image processing operations:

```python
import asyncio
from typing import List, Dict
import aiofiles
import aiohttp

async def fetch_image_edit_result(session: aiohttp.ClientSession, url: str, prompt: str) -> Dict:
    """Asynchronously fetch image edit result from backend."""
    payload = {"image_url": url, "prompt": prompt}
    async with session.post("http://localhost:8000/edit", json=payload) as response:
        return await response.json()

async def process_multiple_edits(image_urls: List[str], prompt: str) -> List[Dict]:
    """Process multiple image edits concurrently."""
    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_image_edit_result(session, url, prompt)
            for url in image_urls
        ]
        results = await asyncio.gather(*tasks)
        return results

async def save_edit_result(result_data: Dict, filepath: str):
    """Asynchronously save edit result to file."""
    async with aiofiles.open(filepath, 'wb') as f:
        await f.write(result_data['image_bytes'])

async def handle_image_edit_workflow(image_path: str, prompt: str):
    """Complete async workflow for image editing."""
    print(f"Starting async processing for {image_path}")
    
    # Simulate API call to edit image
    async with aiohttp.ClientSession() as session:
        payload = {"image_path": image_path, "prompt": prompt}
        async with session.post("http://localhost:8000/process", json=payload) as response:
            result = await response.json()
    
    # Save result
    await save_edit_result(result, f"edited_{image_path}")
    print(f"Completed processing for {image_path}")
    
    return result

# Example usage:
if __name__ == "__main__":
    async def main():
        # Example of processing a single edit
        result = await handle_image_edit_workflow("input.jpg", "make it brighter")
        print(f"Edit result: {result}")
        
        # Example of processing multiple edits concurrently
        image_urls = ["img1.jpg", "img2.jpg", "img3.jpg"]
        results = await process_multiple_edits(image_urls, "enhance quality")
        print(f"Processed {len(results)} images")
    
    asyncio.run(main())
```
