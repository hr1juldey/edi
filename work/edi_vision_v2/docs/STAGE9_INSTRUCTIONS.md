# Stage 9: Main Application Entry Points

**Objective**: Create dual interfaces for the vision pipeline - TUI for humans, CLI for AI agents and code-controlled executions.

**Implementation Files**:
- `app.py` - CLI interface (for AI agents, scripts, automation)
- `tui.py` - TUI interface (for human users, interactive use)

---

## Overview: Two Interfaces, Two Use Cases

### Interface 1: CLI (Command-Line Interface) - `app.py`
**Target Users**: AI agents, scripts, automation, CI/CD pipelines
**Characteristics**:
- Non-interactive batch processing
- Structured output (JSON-compatible)
- Exit codes for success/failure
- Scriptable and pipeable
- Minimal user interaction

### Interface 2: TUI (Text User Interface) - `tui.py`
**Target Users**: Human users, interactive exploration
**Characteristics**:
- Interactive widget-based interface
- Real-time visual feedback
- Progressive disclosure of information
- Mouse and keyboard navigation
- Built with Textual framework

---

## Part A: CLI Interface Specification (app.py)

### Basic Usage
```bash
python app.py --image <path> --prompt "<description>" --output <path>
```

### Full Options
```bash
python app.py \
  --image test_image.jpeg \
  --prompt "turn the blue tin roofs to green" \
  --output result_masks.png \
  --verbose \
  --save-steps \
  --no-validation \
  --config config.yaml
```

---

## Argument Specification

### Required Arguments

**--image** / **-i**
- Path to input image (JPG, PNG, JPEG)
- Validates file exists and is readable
- Example: `--image /path/to/image.jpg`

**--prompt** / **-p**
- User's edit request (natural language)
- Must be non-empty string
- Example: `--prompt "change blue roofs to green"`

**--output** / **-o**
- Path for output visualization
- Creates parent directories if needed
- Example: `--output results/masks.png`

---

### Optional Arguments

**--verbose** / **-v**
- Enable detailed logging (INFO level)
- Shows timing for each stage
- Default: False (WARNING level only)

**--debug** / **-d**
- Enable debug logging (DEBUG level)
- Shows all intermediate data
- Default: False

**--save-steps**
- Save intermediate outputs to logs/
- Creates timestamped directory: `logs/run_YYYYMMDD_HHMMSS/`
- Saves: color_mask, sam_masks, clip_filtered, entity_masks, validation_overlay
- Default: False

**--no-validation**
- Skip VLM validation (Stage 6)
- Faster execution (saves ~5 seconds)
- Use when Ollama not available
- Default: False (validation enabled)

**--config** / **-c**
- Path to YAML config file
- Overrides default parameters
- Example: `--config custom_config.yaml`
- Default: None (use hardcoded defaults)

**--min-area**
- Minimum mask area in pixels
- Filters out noise
- Default: 500

**--color-threshold**
- HSV color match threshold (0.0-1.0)
- Stage 2 color filtering sensitivity
- Default: 0.5

**--clip-threshold**
- CLIP similarity threshold (0.0-1.0)
- Stage 4 semantic filtering cutoff
- Default: 0.22

---

## Output Specification

### Console Output (Default Mode)

**During execution**:
```
[Stage 1/6] Extracting entities from prompt...
  → Detected: blue, tin roof
[Stage 2/6] Filtering by color (blue)...
  → Color mask coverage: 66.3%
[Stage 3/6] SAM segmentation...
  → Generated 17 individual masks
[Stage 4/6] CLIP semantic filtering...
  → Filtered to 14 masks (removed sky, other objects)
[Stage 5/6] Organizing entity masks...
  → 14 entities with metadata
[Stage 6/6] VLM validation...
  → Confidence: 0.85 | Coverage: 95% | False positives: 5%

✅ Success! Detected 14 blue tin roofs
   Total time: 15.3 seconds
   Output saved: result_masks.png
```

**On failure**:
```
❌ Error: No blue regions found in image
   Suggestion: Check if image contains blue objects
   Color mask coverage: 0.2% (threshold: 5%)
```

---

### Verbose Mode Output

**With --verbose**:
```
[INFO] Loading image: test_image.jpeg (1920x1080)
[INFO] Initializing VisionPipeline
[INFO] Stage 1: Entity extraction (qwen3:8b)
[DEBUG] DSpy extraction: {"color": "blue", "target": "tin roof", "edit_type": "recolor"}
[INFO] Stage 1 complete: 1.2s

[INFO] Stage 2: Color pre-filter
[DEBUG] HSV range for blue: [(90, 50, 50), (130, 255, 255)]
[DEBUG] Color mask coverage: 66.3% of image
[INFO] Stage 2 complete: 0.004s

[INFO] Stage 3: SAM segmentation
[DEBUG] Loading SAM model: sam2.1_b.pt
[DEBUG] SAM generated 17 masks
[DEBUG] Filtering by 50% color overlap threshold
[INFO] Stage 3 complete: 6.5s

...
```

---

### Visual Output (result_masks.png)

**Layout**: Multi-panel visualization

```
┌─────────────────┬─────────────────┐
│   Original      │   Color Mask    │
│   Image         │   (Stage 2)     │
├─────────────────┼─────────────────┤
│   SAM Masks     │   Final Masks   │
│   (Stage 3)     │   (Stage 5)     │
└─────────────────┴─────────────────┘
```

**Final Masks Panel**:
- Original image with colored overlays
- Each entity has unique color (rainbow palette)
- Entity ID labels on each mask
- Title: "14 entities detected"

---

### Saved Intermediate Steps (--save-steps)

**Directory structure**:
```
logs/run_20241028_183045/
├── input_image.png           # Original input
├── stage2_color_mask.png     # Binary mask from HSV filtering
├── stage3_sam_masks.png      # Grid of SAM masks
├── stage4_clip_filtered.png  # Grid of CLIP-filtered masks
├── stage5_entity_masks.png   # Final organized entities
├── stage6_validation.png     # VLM validation overlay (if enabled)
├── result.json               # Full pipeline output
└── metadata.json             # Stage timings, parameters
```

---

## Configuration File Format (config.yaml)

```yaml
# Vision pipeline configuration
pipeline:
  enable_validation: true      # Run VLM validation
  save_intermediate: false     # Save intermediate outputs
  output_dir: "logs"           # Where to save logs

# Stage 2: Color filtering
color_filter:
  blue: [[90, 50, 50], [130, 255, 255]]
  green: [[40, 50, 50], [80, 255, 255]]
  # ... more colors

# Stage 3: SAM segmentation
sam:
  model: "sam2.1_b.pt"
  min_area: 500
  color_overlap_threshold: 0.5

# Stage 4: CLIP filtering
clip:
  model: "ViT-B-32"
  similarity_threshold: 0.22

# Stage 6: VLM validation
vlm:
  ollama_url: "http://localhost:11434/api/generate"
  model: "qwen2.5vl:7b"
  timeout: 30

# Logging
logging:
  level: "WARNING"  # DEBUG, INFO, WARNING, ERROR
  format: "[%(levelname)s] %(message)s"
```

---

## Implementation Structure

### Main Function Flow

```python
def main():
    # 1. Parse arguments
    args = parse_arguments()

    # 2. Setup logging
    setup_logging(args.verbose, args.debug)

    # 3. Load config
    config = load_config(args.config) if args.config else get_default_config()

    # 4. Validate inputs
    validate_image_path(args.image)
    validate_prompt(args.prompt)

    # 5. Initialize pipeline
    pipeline = VisionPipeline(
        enable_validation=not args.no_validation,
        save_intermediate=args.save_steps,
        output_dir=config['pipeline']['output_dir']
    )

    # 6. Process image
    try:
        result = pipeline.process(
            image_path=args.image,
            user_prompt=args.prompt
        )
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        sys.exit(1)

    # 7. Create visualization
    visualization = create_output_visualization(
        image_path=args.image,
        result=result
    )

    # 8. Save output
    save_output(visualization, args.output)

    # 9. Print summary
    print_summary(result)
```

---

### Helper Functions

**parse_arguments()**
```python
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EDI Vision Pipeline - Detect and segment entities for image editing",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Add all arguments
    return parser.parse_args()
```

**setup_logging()**
```python
def setup_logging(verbose: bool, debug: bool):
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING

    logging.basicConfig(
        level=level,
        format='[%(levelname)s] %(message)s'
    )
```

**create_output_visualization()**
```python
def create_output_visualization(image_path: str, result: Dict) -> np.ndarray:
    """
    Create 2x2 grid:
    - Top-left: Original image
    - Top-right: Stage 2 color mask
    - Bottom-left: Stage 3 SAM masks (grid)
    - Bottom-right: Final entity masks (colored overlays)
    """
    # Load images
    original = cv2.imread(image_path)
    # Create subplots
    # Return combined visualization
```

**print_summary()**
```python
def print_summary(result: Dict):
    """Print human-readable summary to console"""
    num_entities = len(result['entity_masks'])
    total_time = result['total_time']

    if num_entities == 0:
        print("❌ No entities detected")
        print(f"   Color mask coverage: {result['metadata']['stage2_color_mask_coverage']:.1f}%")
        if result['metadata']['stage2_color_mask_coverage'] < 5:
            print("   Suggestion: Image may not contain target color")
    else:
        print(f"✅ Success! Detected {num_entities} entities")
        print(f"   Total time: {total_time:.1f} seconds")

        if 'validation' in result:
            val = result['validation']
            print(f"   VLM confidence: {val['confidence']:.2f}")
            print(f"   Coverage: {val['target_coverage']:.0f}%")
```

---

## Error Handling

### User-Friendly Error Messages

**File not found**:
```
❌ Error: Image file not found
   Path: /path/to/image.jpg
   Suggestion: Check file path and try again
```

**Invalid image format**:
```
❌ Error: Invalid image format
   File: image.txt
   Supported formats: JPG, PNG, JPEG, WEBP
```

**Empty prompt**:
```
❌ Error: Prompt cannot be empty
   Usage: --prompt "edit blue roofs"
```

**Ollama not available** (with validation enabled):
```
⚠️  Warning: Ollama not available at http://localhost:11434
   Skipping VLM validation
   Use --no-validation to suppress this warning
```

**Out of memory**:
```
❌ Error: GPU out of memory
   Image size: 8000x6000 (very large)
   Suggestion: Resize image to <2048px or use smaller SAM model
```

---

## Implementation Checklist (CLI - app.py)

- [ ] Create `app.py` with argparse
- [ ] Implement all CLI arguments
- [ ] Implement logging setup (verbose/debug modes)
- [ ] Implement config file loading (YAML)
- [ ] Implement input validation
- [ ] Implement output visualization (2x2 grid)
- [ ] Implement summary printing
- [ ] Implement error handling with helpful messages
- [ ] Test with all argument combinations
- [ ] Test error scenarios (missing file, invalid format, etc.)
- [ ] Add docstrings and type hints
- [ ] Create default config.yaml template

---

## Part B: TUI Interface Specification (tui.py)

### Overview

The TUI provides an interactive, visual interface for human users using the Textual framework. It guides users through the pipeline with real-time feedback.

### Basic Usage

```bash
# Launch interactive TUI
python tui.py

# Launch with pre-filled image path
python tui.py --image test_image.jpeg

# Launch with config
python tui.py --config custom.yaml
```

---

### Screen Flow

```
┌─────────────────────────────────────────────┐
│ 1. WelcomeScreen                            │
│    - Brief introduction                     │
│    - "Start New Analysis" button            │
│    - Recent sessions list (if any)          │
└─────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────┐
│ 2. ImageSelectionScreen                     │
│    - File browser widget                    │
│    - Image preview (ANSI art)               │
│    - Image info (dimensions, size)          │
│    - "Next" button                          │
└─────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────┐
│ 3. PromptInputScreen                        │
│    - Text input for edit request            │
│    - Examples dropdown                      │
│    - Settings panel (optional):             │
│      - Enable/disable VLM validation        │
│      - Save intermediate steps              │
│    - "Analyze Image" button                 │
└─────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────┐
│ 4. ProcessingScreen                         │
│    - Live progress bar per stage            │
│    - Current stage indicator                │
│    - Stage timings                          │
│    - Log messages (scrollable)              │
│    - "Cancel" button                        │
└─────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────┐
│ 5. ResultsScreen                            │
│    - Summary panel:                         │
│      • Entities detected: X                 │
│      • Total time: X.Xs                     │
│      • Validation confidence: X%            │
│    - Visualization preview (ANSI)           │
│    - Entity list (table):                   │
│      • ID | Area | Color | Bbox             │
│    - Actions:                               │
│      • "Save Visualization"                 │
│      • "View Details"                       │
│      • "Start New Analysis"                 │
│      • "Quit"                               │
└─────────────────────────────────────────────┘
```

---

### Key Widgets

#### 1. **Header** (all screens)
```
╭─ EDI Vision Pipeline ─────────────────────────────────────────╮
│ Detect and segment entities for image editing                │
╰───────────────────────────────────────────────────────────────╯
```

#### 2. **Footer** (all screens)
```
[Q] Quit  [H] Help  [←] Back  [→] Next  [Enter] Confirm
```

#### 3. **StageProgressBar** (ProcessingScreen)
```
[Stage 1/6] Entity Extraction ━━━━━━━━━━━━━━━━━━━━ 100% (1.2s)
[Stage 2/6] Color Filtering   ━━━━━━━━━━━━━━━━━━━━ 100% (0.0s)
[Stage 3/6] SAM Segmentation  ━━━━━━━━━━━━━━━━━━━━  45% (3.2s)
[Stage 4/6] CLIP Filtering    ━━━━━━━━━━━━━━━━━━━━   0%
[Stage 5/6] Organization      ━━━━━━━━━━━━━━━━━━━━   0%
[Stage 6/6] VLM Validation    ━━━━━━━━━━━━━━━━━━━━   0%
```

#### 4. **EntityTable** (ResultsScreen)
```
┏━━━━┳━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
┃ ID ┃  Area  ┃   Color   ┃       BBox           ┃
┡━━━━╇━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
│  0 │ 12456  │  #3A5FCD  │ (120, 80, 340, 210)  │
│  1 │ 11892  │  #3B60CE  │ (450, 85, 660, 215)  │
│  2 │ 10234  │  #3C61CF  │ (120, 320, 330, 440) │
│ .. │   ...  │    ...    │        ...           │
└────┴────────┴───────────┴──────────────────────┘
```

#### 5. **LogPanel** (ProcessingScreen)
```
╭─ Processing Log ──────────────────────────────╮
│ [INFO] Loading image: test_image.jpeg         │
│ [INFO] Image dimensions: 1024x1024            │
│ [INFO] Starting Stage 1: Entity Extraction    │
│ [INFO] Detected entities: blue, tin roof      │
│ [INFO] Starting Stage 2: Color Filtering      │
│ [INFO] Color mask coverage: 66.3%             │
│ [INFO] Starting Stage 3: SAM Segmentation     │
│ [DEBUG] Loading SAM model: sam2.1_b.pt        │
│ ↓ (scrollable)                                │
╰───────────────────────────────────────────────╯
```

---

### Textual App Structure

```python
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import (
    Header, Footer, Button, Input, Label,
    ProgressBar, DataTable, Static, DirectoryTree
)
from textual.screen import Screen

class EDIVisionApp(App):
    """Main TUI application for EDI Vision Pipeline."""

    CSS_PATH = "tui.tcss"  # Textual CSS for styling

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("h", "help", "Help"),
        ("escape", "back", "Back"),
    ]

    def __init__(self, config_path: str | None = None):
        super().__init__()
        self.config_path = config_path
        self.current_image = None
        self.current_prompt = None
        self.pipeline_result = None

    def on_mount(self) -> None:
        """Called when app starts."""
        self.push_screen(WelcomeScreen())


class WelcomeScreen(Screen):
    """Welcome screen with introduction."""

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Label("Welcome to EDI Vision Pipeline", id="title"),
            Label("Detect and segment entities for image editing"),
            Button("Start New Analysis", variant="primary", id="start"),
            id="welcome-container"
        )
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "start":
            self.app.push_screen(ImageSelectionScreen())


class ImageSelectionScreen(Screen):
    """Image file selection screen."""

    def compose(self) -> ComposeResult:
        yield Header()
        with Container(id="main-container"):
            yield Label("Select an image to analyze")
            yield DirectoryTree("./", id="file-browser")
            with Horizontal(id="preview-panel"):
                yield Static("No image selected", id="preview")
                yield Container(
                    Label("Image Info", id="info-title"),
                    Label("", id="image-info"),
                    id="info-container"
                )
            with Horizontal(id="button-row"):
                yield Button("Back", variant="default", id="back")
                yield Button("Next", variant="primary", id="next")
        yield Footer()

    def on_directory_tree_file_selected(self, event) -> None:
        """Handle file selection."""
        file_path = event.path
        # Validate image format
        # Update preview (convert to ANSI art)
        # Update info panel


class PromptInputScreen(Screen):
    """Prompt input and settings screen."""

    def compose(self) -> ComposeResult:
        yield Header()
        with Container(id="main-container"):
            yield Label("Describe the edit you want to make")
            yield Input(
                placeholder="e.g., change blue roofs to green",
                id="prompt-input"
            )
            yield Label("Examples:")
            # Example buttons
            with Horizontal(id="settings-panel"):
                yield Label("Settings:")
                # Checkboxes for validation, save-steps
            yield Button("Analyze Image", variant="primary", id="analyze")
        yield Footer()


class ProcessingScreen(Screen):
    """Live processing with progress bars."""

    def compose(self) -> ComposeResult:
        yield Header()
        with Container(id="main-container"):
            yield Label("Processing your image...", id="status")
            # 6 progress bars (one per stage)
            yield Container(id="log-panel")
            yield Button("Cancel", variant="error", id="cancel")
        yield Footer()

    async def on_mount(self) -> None:
        """Start pipeline processing."""
        # Run pipeline in background worker
        # Update progress bars in real-time


class ResultsScreen(Screen):
    """Results display with actions."""

    def compose(self) -> ComposeResult:
        yield Header()
        with Container(id="main-container"):
            with Horizontal(id="summary-panel"):
                yield Label(f"✅ Detected {num_entities} entities")
                # Summary stats
            yield Static(id="visualization-preview")
            yield DataTable(id="entity-table")
            with Horizontal(id="action-buttons"):
                yield Button("Save Visualization", id="save")
                yield Button("New Analysis", id="new")
                yield Button("Quit", id="quit")
        yield Footer()
```

---

### Textual CSS (tui.tcss)

```css
/* Global styles */
Screen {
    align: center middle;
}

#title {
    text-align: center;
    text-style: bold;
    color: $accent;
    padding: 1;
}

#welcome-container {
    width: 60;
    height: auto;
    border: solid $primary;
    padding: 2;
}

/* Buttons */
Button {
    margin: 1 2;
}

Button.-primary {
    background: $success;
}

Button.-error {
    background: $error;
}

/* Progress bars */
ProgressBar {
    margin: 0 2;
    height: 1;
}

/* Data table */
DataTable {
    height: 20;
    margin: 1 2;
}

/* File browser */
DirectoryTree {
    width: 40%;
    height: 100%;
}

#preview-panel {
    height: 30;
}

#preview {
    width: 60%;
    border: solid $accent;
    padding: 1;
}
```

---

### Implementation Checklist (TUI - tui.py)

- [ ] Create `tui.py` with Textual App structure
- [ ] Implement WelcomeScreen
- [ ] Implement ImageSelectionScreen with file browser
- [ ] Implement PromptInputScreen with validation
- [ ] Implement ProcessingScreen with live progress
- [ ] Implement ResultsScreen with entity table
- [ ] Create `tui.tcss` stylesheet
- [ ] Add keyboard shortcuts (Q, H, ESC, arrows)
- [ ] Implement ANSI art image preview
- [ ] Add error handling with modal dialogs
- [ ] Test navigation flow (forward/back)
- [ ] Test with real pipeline execution
- [ ] Add help screen
- [ ] Add settings persistence

---

## Combined Acceptance Criteria

### CLI Acceptance Criteria

**Basic functionality**:
```bash
python app.py --image test_image.jpeg --prompt "blue roofs" --output result.png
# Should complete successfully and show summary
```

**Verbose mode**:
```bash
python app.py --image test_image.jpeg --prompt "blue roofs" --output result.png --verbose
# Should show detailed stage-by-stage progress
```

**Save intermediate steps**:
```bash
python app.py --image test_image.jpeg --prompt "blue roofs" --output result.png --save-steps
# Should create logs/run_YYYYMMDD_HHMMSS/ with all intermediate images
```

**Error handling**:
```bash
python app.py --image nonexistent.jpg --prompt "blue roofs" --output result.png
# Should show clear error message (not stack trace)
```

**CLI Success Metrics**:
- ✅ Clean, non-interactive interface
- ✅ Helpful error messages (no raw exceptions)
- ✅ Visual output clearly shows results
- ✅ Console output is concise and actionable
- ✅ All options work as documented
- ✅ Scriptable and pipeable
- ✅ Proper exit codes (0 success, 1 failure)

---

### TUI Acceptance Criteria

**Basic launch**:
```bash
python tui.py
# Should open interactive TUI with WelcomeScreen
```

**Pre-filled launch**:
```bash
python tui.py --image test_image.jpeg
# Should open TUI with image pre-selected
```

**Navigation flow**:
- ✅ Can navigate forward through all screens
- ✅ Can navigate back to previous screens
- ✅ Keyboard shortcuts work (Q, H, ESC, arrows)
- ✅ Mouse clicks work on all buttons

**Pipeline execution**:
- ✅ Progress bars update in real-time
- ✅ Log messages appear as pipeline runs
- ✅ Can cancel mid-processing
- ✅ Results screen shows entity table
- ✅ Can save visualization from results screen

**TUI Success Metrics**:
- ✅ Intuitive navigation flow
- ✅ Real-time visual feedback
- ✅ No crashes or freezes
- ✅ Responsive keyboard and mouse input
- ✅ Clear visual hierarchy
- ✅ ANSI art preview works
- ✅ Can complete full workflow without CLI

---

## Deliverables

### Required Files

1. **app.py** - Complete CLI application (400-500 lines)
2. **tui.py** - Complete TUI application (600-800 lines)
3. **tui.tcss** - Textual CSS stylesheet (100-150 lines)
4. **config.yaml** - Default configuration template
5. **README.md** - Updated usage instructions (both interfaces)

### Test Outputs

6. **CLI test report** - Output from running all CLI tests
7. **TUI screenshots** - Screenshots of each screen (if possible)
8. **Example visualizations** - Sample output images

---

## Implementation Order

### Phase 1: CLI Implementation (Priority)
1. Create `app.py` with complete argparse structure
2. Implement validation, logging, config loading
3. Implement main() and output functions
4. Test all CLI functionality
5. Get supervisor approval

### Phase 2: TUI Implementation
1. Create `tui.py` with basic App structure
2. Implement WelcomeScreen and ImageSelectionScreen
3. Implement PromptInputScreen
4. Implement ProcessingScreen with live progress
5. Implement ResultsScreen
6. Create `tui.tcss` stylesheet
7. Test full navigation flow
8. Get supervisor approval

### Phase 3: Integration and Documentation
1. Update README.md with both interfaces
2. Create example outputs
3. Final testing of both interfaces
4. Prepare deliverables

---

## Notes for Qwen

### CLI Notes
- Use `argparse` for CLI (not click or typer)
- Use `logging` module (not print statements)
- Use `pathlib.Path` for file operations
- Graceful fallbacks for missing dependencies (Ollama)
- Focus on **non-interactive batch processing**
- Proper exit codes for scripting

### TUI Notes
- Use `Textual` framework (already in dependencies)
- Study reference examples in `@/home/riju279/Documents/Code/Zonko/EDI/edi/example_code/textual/`
- Use `Worker` for background pipeline processing
- Handle CTRL+C gracefully
- Focus on **interactive user experience**
- Real-time progress updates

**Philosophy**:
- **CLI**: Simple, scriptable, non-interactive - for automation
- **TUI**: Rich, interactive, guided - for human exploration
