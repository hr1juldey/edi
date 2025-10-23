# UI: Screens

[Back to TUI Layer](./tui_layer.md)

## Purpose
Screen definitions - Contains all the screen classes for the Textual TUI including HomeScreen, ImageUploadScreen, PromptInputScreen, etc.

## Screens
- `HomeScreen`: Welcome screen with main menu options
- `ImageUploadScreen`: File selection and image preview
- `PromptInputScreen`: TextArea for prompts and entity selection
- `AnalysisScreen`: Shows progress during SAM+CLIP analysis
- `ClarificationScreen`: Displays clarifying questions
- `RefinementScreen`: Shows prompt evolution
- `ExecutionScreen`: Shows progress when sending to ComfyUI
- `ResultsScreen`: Side-by-side image comparison
- `VariationsScreen`: Grid layout for multiple results
- `FeedbackScreen`: User rating and feedback collection

### Details
- Each screen handles specific UI functionality
- Implements keyboard navigation
- Provides visual feedback to user

## Technology Stack

- Textual for screen creation
- Rich for terminal rendering
- AsyncIO for non-blocking operations

## See Docs

### Textual Implementation Example
Screen implementations for the EDI application:

```python
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Static, Button, Input, TextArea, DataTable, ProgressBar
from textual.containers import Container, Vertical, Horizontal
from textual.screen import Screen
from textual import work
import asyncio
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
from datetime import datetime
import logging

class HomeScreen(Screen):
    """
    Welcome screen with main menu options.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logger = logging.getLogger(__name__)
    
    def compose(self) -> ComposeResult:
        """Create child widgets."""
        yield Header()
        yield Vertical(
            Static("EDI: Edit with Intelligence", classes="title"),
            Static("AI-Powered Image Editing Assistant", classes="subtitle"),
            Container(
                Static("[1] Start New Edit", id="start-new-edit", classes="menu-item"),
                Static("[2] Load Recent Session", id="load-session", classes="menu-item"),
                Static("[3] View Examples", id="view-examples", classes="menu-item"),
                Static("[4] System Diagnostics", id="diagnostics", classes="menu-item"),
                Static("[Q] Quit", id="quit", classes="menu-item"),
                classes="menu-container"
            ),
            classes="main-content"
        )
        yield Footer()
    
    def on_mount(self) -> None:
        """Called when the screen is mounted."""
        self.logger.info("Home screen mounted")
    
    def on_static_pressed(self, event: Static.Pressed) -> None:
        """Handle menu item selection."""
        if event.static.id == "start-new-edit":
            self.app.push_screen(ImageUploadScreen())
        elif event.static.id == "load-session":
            self.app.push_screen(SessionHistoryScreen())
        elif event.static.id == "view-examples":
            self.app.push_screen(ExamplesScreen())
        elif event.static.id == "diagnostics":
            self.app.push_screen(DiagnosticsScreen())
        elif event.static.id == "quit":
            self.app.exit()

class ImageUploadScreen(Screen):
    """
    File selection and image preview.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.selected_image: Optional[str] = None
        self.logger = logging.getLogger(__name__)
    
    def compose(self) -> ComposeResult:
        """Create child widgets."""
        yield Header()
        yield Vertical(
            Static("Upload Image", classes="screen-title"),
            Horizontal(
                Vertical(
                    Static("Select Image File:", classes="instructions"),
                    Input(placeholder="Enter image path...", id="image-path-input"),
                    Button("Browse...", id="browse-btn", variant="primary"),
                    Button("Upload", id="upload-btn", variant="success", disabled=True),
                    Button("Back", id="back-btn", variant="error"),
                    classes="upload-controls"
                ),
                Vertical(
                    Static("Image Preview", classes="preview-title"),
                    Static("No image selected", id="image-preview", classes="preview-content"),
                    classes="image-preview-section"
                ),
                classes="upload-container"
            ),
            classes="main-content"
        )
        yield Footer()
    
    def on_mount(self) -> None:
        """Called when the screen is mounted."""
        self.logger.info("Image upload screen mounted")
    
    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle input changes."""
        if event.input.id == "image-path-input":
            self.selected_image = event.value
            upload_btn = self.query_one("#upload-btn", Button)
            upload_btn.disabled = not bool(self.selected_image.strip())
            
            # Update preview
            if self.selected_image and Path(self.selected_image).exists():
                self._update_image_preview(self.selected_image)
            else:
                self._clear_image_preview()
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "browse-btn":
            self._browse_for_image()
        elif event.button.id == "upload-btn":
            self._upload_image()
        elif event.button.id == "back-btn":
            self.app.pop_screen()
    
    def _browse_for_image(self) -> None:
        """Browse for an image file."""
        # In a real implementation, this would open a file dialog
        # For this example, we'll just simulate it
        self.logger.info("Browsing for image file")
        self.notify("File browser would open in a real implementation")
    
    def _upload_image(self) -> None:
        """Upload the selected image."""
        if not self.selected_image:
            self.logger.warning("No image selected for upload")
            self.notify("Please select an image first", severity="warning")
            return
        
        if not Path(self.selected_image).exists():
            self.logger.error(f"Image file not found: {self.selected_image}")
            self.notify(f"Image file not found: {self.selected_image}", severity="error")
            return
        
        # Validate image format
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        ext = Path(self.selected_image).suffix.lower()
        if ext not in valid_extensions:
            self.logger.error(f"Invalid image format: {ext}")
            self.notify(f"Invalid image format. Supported: {valid_extensions}", severity="error")
            return
        
        self.logger.info(f"Uploading image: {self.selected_image}")
        self.notify("Image uploaded successfully!", severity="information")
        
        # Push to next screen
        self.app.push_screen(PromptInputScreen(image_path=self.selected_image))
    
    def _update_image_preview(self, image_path: str) -> None:
        """Update the image preview."""
        preview_widget = self.query_one("#image-preview", Static)
        file_size = Path(image_path).stat().st_size / (1024 * 1024)  # MB
        preview_widget.update(f"Preview: {Path(image_path).name} ({file_size:.1f} MB)")
    
    def _clear_image_preview(self) -> None:
        """Clear the image preview."""
        preview_widget = self.query_one("#image-preview", Static)
        preview_widget.update("No image selected")

class PromptInputScreen(Screen):
    """
    TextArea for prompts and entity selection.
    """
    
    def __init__(self, image_path: str, **kwargs):
        super().__init__(**kwargs)
        self.image_path = image_path
        self.logger = logging.getLogger(__name__)
    
    def compose(self) -> ComposeResult:
        """Create child widgets."""
        yield Header()
        yield Vertical(
            Static(f"Editing: {Path(self.image_path).name}", classes="screen-title"),
            Horizontal(
                Vertical(
                    Static("Enter your edit prompt:", classes="instructions"),
                    TextArea(id="prompt-input", placeholder="Describe the changes you want..."),
                    Static("Detected Entities:", classes="instructions"),
                    Container(
                        Static("Loading entities...", id="entities-list"),
                        classes="entities-container"
                    ),
                    Horizontal(
                        Button("Back", id="back-btn", variant="error"),
                        Button("Analyze", id="analyze-btn", variant="primary", disabled=True),
                        Button("Execute", id="execute-btn", variant="success", disabled=True),
                        classes="action-buttons"
                    ),
                    classes="prompt-section"
                ),
                Vertical(
                    Static("Image Preview", classes="preview-title"),
                    Static("Image would be shown here", id="image-preview", classes="preview-content"),
                    classes="image-preview-section"
                ),
                classes="prompt-container"
            ),
            classes="main-content"
        )
        yield Footer()
    
    def on_mount(self) -> None:
        """Called when the screen is mounted."""
        self.logger.info(f"Prompt input screen mounted for image: {self.image_path}")
        
        # Start entity detection in background
        self._detect_entities()
    
    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Handle text area changes."""
        if event.text_area.id == "prompt-input":
            analyze_btn = self.query_one("#analyze-btn", Button)
            execute_btn = self.query_one("#execute-btn", Button)
            
            # Enable analyze button if prompt is not empty
            has_prompt = bool(event.text_area.text.strip())
            analyze_btn.disabled = not has_prompt
            
            # Enable execute button only if analysis is complete
            # In a real implementation, we would check analysis status
            execute_btn.disabled = not has_prompt or True  # Disabled until analysis complete
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "back-btn":
            self.app.pop_screen()
        elif event.button.id == "analyze-btn":
            self._analyze_prompt()
        elif event.button.id == "execute-btn":
            self._execute_edit()
    
    @work(exclusive=True, thread=True)
    def _detect_entities(self) -> None:
        """Detect entities in the background."""
        # In a real implementation, this would call the vision subsystem
        # For this example, we'll simulate it
        import time
        time.sleep(2)  # Simulate processing time
        
        # Simulate detected entities
        entities = [
            {"id": "sky_0", "label": "sky", "confidence": 0.95},
            {"id": "mountain_1", "label": "mountain", "confidence": 0.87},
            {"id": "tree_2", "label": "tree", "confidence": 0.92}
        ]
        
        # Update UI on main thread
        self.call_from_thread(self._update_entities_list, entities)
    
    def _update_entities_list(self, entities: List[Dict[str, Any]]) -> None:
        """Update the entities list on the UI thread."""
        entities_widget = self.query_one("#entities-list", Static)
        
        if not entities:
            entities_widget.update("No entities detected")
            return
        
        # Create entity list with checkboxes
        entity_lines = []
        for entity in entities:
            entity_lines.append(f"[ ] {entity['label']} ({entity['confidence']:.2f})")
        
        entities_widget.update("\\n".join(entity_lines))
        self.logger.info(f"Updated entities list with {len(entities)} entities")
    
    def _analyze_prompt(self) -> None:
        """Analyze the user prompt."""
        prompt_widget = self.query_one("#prompt-input", TextArea)
        prompt_text = prompt_widget.text.strip()
        
        if not prompt_text:
            self.logger.warning("Empty prompt provided for analysis")
            self.notify("Please enter a prompt first", severity="warning")
            return
        
        self.logger.info(f"Analyzing prompt: {prompt_text}")
        self.notify("Analyzing prompt...", severity="information")
        
        # In a real implementation, this would call the reasoning subsystem
        # For this example, we'll simulate it
        self._simulate_prompt_analysis(prompt_text)
    
    @work(exclusive=True, thread=True)
    def _simulate_prompt_analysis(self, prompt: str) -> None:
        """Simulate prompt analysis in background."""
        import time
        time.sleep(3)  # Simulate processing time
        
        # Simulate analysis result
        analysis_result = {
            "target_entities": ["sky_0"],
            "edit_type": "transform",
            "confidence": 0.85,
            "clarifying_questions": []
        }
        
        # Update UI on main thread
        self.call_from_thread(self._finish_prompt_analysis, analysis_result)
    
    def _finish_prompt_analysis(self, analysis_result: Dict[str, Any]) -> None:
        """Finish prompt analysis on UI thread."""
        execute_btn = self.query_one("#execute-btn", Button)
        execute_btn.disabled = False
        
        self.logger.info("Prompt analysis completed")
        self.notify("Prompt analysis completed!", severity="success")
    
    def _execute_edit(self) -> None:
        """Execute the edit."""
        self.logger.info("Executing edit")
        self.notify("Starting edit execution...", severity="information")
        
        # Push to execution screen
        self.app.push_screen(ExecutionScreen(image_path=self.image_path))

class AnalysisScreen(Screen):
    """
    Shows progress during SAM+CLIP analysis.
    """
    
    def __init__(self, image_path: str, **kwargs):
        super().__init__(**kwargs)
        self.image_path = image_path
        self.logger = logging.getLogger(__name__)
    
    def compose(self) -> ComposeResult:
        """Create child widgets."""
        yield Header()
        yield Vertical(
            Static("Image Analysis in Progress", classes="screen-title"),
            Container(
                Static("Analyzing image with SAM and CLIP...", id="analysis-status"),
                ProgressBar(id="analysis-progress", total=100),
                Static("0%", id="progress-percent"),
                classes="analysis-container"
            ),
            Horizontal(
                Button("Cancel", id="cancel-btn", variant="error"),
                classes="action-buttons"
            ),
            classes="main-content"
        )
        yield Footer()
    
    def on_mount(self) -> None:
        """Called when the screen is mounted."""
        self.logger.info(f"Analysis screen mounted for image: {self.image_path}")
        
        # Start analysis
        self._start_analysis()
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "cancel-btn":
            self._cancel_analysis()
    
    @work(exclusive=True)
    async def _start_analysis(self) -> None:
        """Start the analysis process."""
        progress_bar = self.query_one("#analysis-progress", ProgressBar)
        status_widget = self.query_one("#analysis-status", Static)
        percent_widget = self.query_one("#progress-percent", Static)
        
        # Simulate analysis steps
        steps = [
            ("Loading image...", 10),
            ("Detecting objects with SAM...", 30),
            ("Analyzing with CLIP...", 60),
            ("Generating masks...", 80),
            ("Finalizing results...", 100)
        ]
        
        for status_text, progress_value in steps:
            # Update UI
            status_widget.update(status_text)
            progress_bar.update(progress=progress_value)
            percent_widget.update(f"{progress_value}%")
            
            # Simulate processing time
            await asyncio.sleep(1.5)
        
        # Complete analysis
        self.call_from_thread(self._complete_analysis)
    
    def _complete_analysis(self) -> None:
        """Complete the analysis process."""
        self.logger.info("Image analysis completed")
        self.notify("Image analysis completed!", severity="success")
        
        # Push to next screen
        self.app.push_screen(ClarificationScreen())
    
    def _cancel_analysis(self) -> None:
        """Cancel the analysis process."""
        self.logger.info("Analysis cancelled by user")
        self.notify("Analysis cancelled", severity="warning")
        
        # Pop back to previous screen
        self.app.pop_screen()

class ClarificationScreen(Screen):
    """
    Displays clarifying questions.
    """
    
    def __init__(self, questions: List[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.questions = questions or [
            "Did you want to preserve the original colors of the mountains?",
            "Should the trees remain unchanged in the foreground?",
            "Do you prefer a more realistic or artistic style for the changes?"
        ]
        self.selected_answers: Dict[int, str] = {}
        self.logger = logging.getLogger(__name__)
    
    def compose(self) -> ComposeResult:
        """Create child widgets."""
        yield Header()
        yield Vertical(
            Static("Clarification Questions", classes="screen-title"),
            Static("Please answer the following questions to help us better understand your intent:", classes="instructions"),
            Container(
                *[Static(f"{i+1}. {q}", id=f"question-{i}") for i, q in enumerate(self.questions)],
                classes="questions-container"
            ),
            Container(
                *[Input(placeholder=f"Answer to question {i+1}...", id=f"answer-{i}") for i in range(len(self.questions))],
                classes="answers-container"
            ),
            Horizontal(
                Button("Back", id="back-btn", variant="error"),
                Button("Submit", id="submit-btn", variant="success"),
                classes="action-buttons"
            ),
            classes="main-content"
        )
        yield Footer()
    
    def on_mount(self) -> None:
        """Called when the screen is mounted."""
        self.logger.info("Clarification screen mounted")
    
    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle input changes."""
        if event.input.id.startswith("answer-"):
            # Extract question index
            question_index = int(event.input.id.split("-")[1])
            self.selected_answers[question_index] = event.value
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "back-btn":
            self.app.pop_screen()
        elif event.button.id == "submit-btn":
            self._submit_answers()
    
    def _submit_answers(self) -> None:
        """Submit the answers."""
        # Check if all questions have answers
        unanswered = [i for i in range(len(self.questions)) if i not in self.selected_answers]
        if unanswered:
            self.logger.warning(f"Unanswered questions: {unanswered}")
            self.notify("Please answer all questions", severity="warning")
            return
        
        self.logger.info("Clarification answers submitted")
        self.notify("Answers submitted successfully!", severity="success")
        
        # Push to refinement screen
        self.app.push_screen(RefinementScreen())

class RefinementScreen(Screen):
    """
    Shows prompt evolution.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logger = logging.getLogger(__name__)
        self.prompts_history = []
    
    def compose(self) -> ComposeResult:
        """Create child widgets."""
        yield Header()
        yield Vertical(
            Static("Prompt Evolution", classes="screen-title"),
            Static("Review how your prompt evolved through iterations:", classes="instructions"),
            Container(
                Static("Loading prompt history...", id="prompt-history"),
                classes="history-container"
            ),
            Horizontal(
                Button("Back", id="back-btn", variant="error"),
                Button("Accept", id="accept-btn", variant="success"),
                Button("Retry", id="retry-btn", variant="primary"),
                classes="action-buttons"
            ),
            classes="main-content"
        )
        yield Footer()
    
    def on_mount(self) -> None:
        """Called when the screen is mounted."""
        self.logger.info("Refinement screen mounted")
        
        # Load prompt history
        self._load_prompt_history()
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "back-btn":
            self.app.pop_screen()
        elif event.button.id == "accept-btn":
            self._accept_refinement()
        elif event.button.id == "retry-btn":
            self._retry_refinement()
    
    def _load_prompt_history(self) -> None:
        """Load prompt history."""
        # In a real implementation, this would load from the database
        # For this example, we'll create mock data
        self.prompts_history = [
            {
                "iteration": 0,
                "positive": "dramatic sky with storm clouds",
                "negative": "sunny sky, clear weather",
                "quality": 0.85
            },
            {
                "iteration": 1,
                "positive": "storm clouds with lighting and dark atmosphere",
                "negative": "sunny sky, clear weather, no clouds",
                "quality": 0.92
            },
            {
                "iteration": 2,
                "positive": "dramatic storm clouds with lightning strikes and dark sky",
                "negative": "sunny sky, clear weather, no clouds, bright lighting",
                "quality": 0.88
            }
        ]
        
        # Update UI
        self._update_prompt_history_display()
    
    def _update_prompt_history_display(self) -> None:
        """Update the prompt history display."""
        history_widget = self.query_one("#prompt-history", Static)
        
        if not self.prompts_history:
            history_widget.update("No prompt history available")
            return
        
        # Create history display
        history_lines = []
        for prompt in self.prompts_history:
            history_lines.append(f"Iteration {prompt['iteration']} (Quality: {prompt['quality']:.2f})")
            history_lines.append(f"  Positive: {prompt['positive']}")
            history_lines.append(f"  Negative: {prompt['negative']}")
            history_lines.append("")  # Empty line for spacing
        
        history_widget.update("\\n".join(history_lines))
    
    def _accept_refinement(self) -> None:
        """Accept the refined prompts."""
        self.logger.info("Refined prompts accepted")
        self.notify("Prompts accepted!", severity="success")
        
        # Push to execution screen
        self.app.push_screen(ExecutionScreen())
    
    def _retry_refinement(self) -> None:
        """Retry the refinement process."""
        self.logger.info("Refinement retry requested")
        self.notify("Restarting refinement process...", severity="information")
        
        # Clear history and restart
        self.prompts_history = []
        self._load_prompt_history()

class ExecutionScreen(Screen):
    """
    Shows progress when sending to ComfyUI.
    """
    
    def __init__(self, image_path: str = None, **kwargs):
        super().__init__(**kwargs)
        self.image_path = image_path
        self.logger = logging.getLogger(__name__)
    
    def compose(self) -> ComposeResult:
        """Create child widgets."""
        yield Header()
        yield Vertical(
            Static("Edit Execution in Progress", classes="screen-title"),
            Container(
                Static("Sending request to ComfyUI...", id="execution-status"),
                ProgressBar(id="execution-progress", total=100),
                Static("0%", id="progress-percent"),
                classes="execution-container"
            ),
            Horizontal(
                Button("Cancel", id="cancel-btn", variant="error"),
                classes="action-buttons"
            ),
            classes="main-content"
        )
        yield Footer()
    
    def on_mount(self) -> None:
        """Called when the screen is mounted."""
        self.logger.info("Execution screen mounted")
        
        # Start execution
        self._start_execution()
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "cancel-btn":
            self._cancel_execution()
    
    @work(exclusive=True)
    async def _start_execution(self) -> None:
        """Start the execution process."""
        progress_bar = self.query_one("#execution-progress", ProgressBar)
        status_widget = self.query_one("#execution-status", Static)
        percent_widget = self.query_one("#progress-percent", Static)
        
        # Simulate execution steps
        steps = [
            ("Preparing request...", 10),
            ("Sending to ComfyUI server...", 30),
            ("Processing on GPU...", 60),
            ("Generating results...", 80),
            ("Downloading results...", 100)
        ]
        
        for status_text, progress_value in steps:
            # Update UI
            status_widget.update(status_text)
            progress_bar.update(progress=progress_value)
            percent_widget.update(f"{progress_value}%")
            
            # Simulate processing time
            await asyncio.sleep(2.0)
        
        # Complete execution
        self.call_from_thread(self._complete_execution)
    
    def _complete_execution(self) -> None:
        """Complete the execution process."""
        self.logger.info("Edit execution completed")
        self.notify("Edit execution completed!", severity="success")
        
        # Push to results screen
        self.app.push_screen(ResultsScreen())
    
    def _cancel_execution(self) -> None:
        """Cancel the execution process."""
        self.logger.info("Execution cancelled by user")
        self.notify("Execution cancelled", severity="warning")
        
        # Pop back to previous screen
        self.app.pop_screen()

class ResultsScreen(Screen):
    """
    Side-by-side image comparison.
    """
    
    def __init__(self, before_image: str = None, after_image: str = None, **kwargs):
        super().__init__(**kwargs)
        self.before_image = before_image or "before.jpg"
        self.after_image = after_image or "after.jpg"
        self.logger = logging.getLogger(__name__)
    
    def compose(self) -> ComposeResult:
        """Create child widgets."""
        yield Header()
        yield Vertical(
            Static("Edit Results", classes="screen-title"),
            Horizontal(
                Vertical(
                    Static("Original Image", classes="image-title"),
                    Static("Image preview would be shown here", id="before-preview", classes="image-preview"),
                    classes="image-container"
                ),
                Vertical(
                    Static("Edited Image", classes="image-title"),
                    Static("Image preview would be shown here", id="after-preview", classes="image-preview"),
                    classes="image-container"
                ),
                classes="images-container"
            ),
            Container(
                Static("Quality Score: 0.85", id="quality-score", classes="score-display"),
                Static("Processing Time: 12.3s", id="processing-time", classes="time-display"),
                classes="metrics-container"
            ),
            Horizontal(
                Button("Back", id="back-btn", variant="error"),
                Button("Accept", id="accept-btn", variant="success"),
                Button("Reject", id="reject-btn", variant="primary"),
                Button("Retry", id="retry-btn", variant="warning"),
                classes="action-buttons"
            ),
            classes="main-content"
        )
        yield Footer()
    
    def on_mount(self) -> None:
        """Called when the screen is mounted."""
        self.logger.info("Results screen mounted")
        
        # Load images
        self._load_images()
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "back-btn":
            self.app.pop_screen()
        elif event.button.id == "accept-btn":
            self._accept_result()
        elif event.button.id == "reject-btn":
            self._reject_result()
        elif event.button.id == "retry-btn":
            self._retry_result()
    
    def _load_images(self) -> None:
        """Load images for comparison."""
        # In a real implementation, this would load actual images
        # For this example, we'll just update the UI
        before_widget = self.query_one("#before-preview", Static)
        after_widget = self.query_one("#after-preview", Static)
        score_widget = self.query_one("#quality-score", Static)
        time_widget = self.query_one("#processing-time", Static)
        
        before_widget.update(f"Original: {Path(self.before_image).name}")
        after_widget.update(f"Edited: {Path(self.after_image).name}")
        score_widget.update("Quality Score: 0.85")
        time_widget.update("Processing Time: 12.3s")
    
    def _accept_result(self) -> None:
        """Accept the edit result."""
        self.logger.info("Edit result accepted")
        self.notify("Result accepted!", severity="success")
        
        # Push to feedback screen
        self.app.push_screen(FeedbackScreen())
    
    def _reject_result(self) -> None:
        """Reject the edit result."""
        self.logger.info("Edit result rejected")
        self.notify("Result rejected", severity="warning")
        
        # Pop back to previous screen
        self.app.pop_screen()
    
    def _retry_result(self) -> None:
        """Retry the edit."""
        self.logger.info("Edit retry requested")
        self.notify("Restarting edit process...", severity="information")
        
        # Pop back and restart
        self.app.pop_screen()
        self.app.pop_screen()  # Pop ResultsScreen
        self.app.pop_screen()  # Pop ExecutionScreen

class VariationsScreen(Screen):
    """
    Grid layout for multiple results.
    """
    
    def __init__(self, variations: List[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.variations = variations or [f"variation_{i}.jpg" for i in range(4)]
        self.logger = logging.getLogger(__name__)
    
    def compose(self) -> ComposeResult:
        """Create child widgets."""
        yield Header()
        yield Vertical(
            Static("Edit Variations", classes="screen-title"),
            Static("Select your preferred variation:", classes="instructions"),
            Container(
                *[Static(f"Variation {i+1}", id=f"variation-{i}", classes="variation-preview") 
                  for i in range(len(self.variations))],
                classes="variations-grid"
            ),
            Horizontal(
                Button("Back", id="back-btn", variant="error"),
                Button("Accept Selected", id="accept-btn", variant="success"),
                classes="action-buttons"
            ),
            classes="main-content"
        )
        yield Footer()
    
    def on_mount(self) -> None:
        """Called when the screen is mounted."""
        self.logger.info("Variations screen mounted")
        
        # Load variations
        self._load_variations()
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "back-btn":
            self.app.pop_screen()
        elif event.button.id == "accept-btn":
            self._accept_variation()
    
    def _load_variations(self) -> None:
        """Load variations for display."""
        # In a real implementation, this would load actual variations
        # For this example, we'll just update the UI
        for i, variation in enumerate(self.variations):
            variation_widget = self.query_one(f"#variation-{i}", Static)
            variation_widget.update(f"Variation {i+1}: {Path(variation).name}")
    
    def _accept_variation(self) -> None:
        """Accept the selected variation."""
        self.logger.info("Variation accepted")
        self.notify("Variation accepted!", severity="success")
        
        # Push to feedback screen
        self.app.push_screen(FeedbackScreen())

class FeedbackScreen(Screen):
    """
    User rating and feedback collection.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logger = logging.getLogger(__name__)
        self.rating = 0
    
    def compose(self) -> ComposeResult:
        """Create child widgets."""
        yield Header()
        yield Vertical(
            Static("Feedback", classes="screen-title"),
            Static("How would you rate this edit?", classes="instructions"),
            Container(
                *[Button(str(i), id=f"rating-{i}", variant="default") for i in range(1, 6)],
                id="rating-buttons",
                classes="rating-container"
            ),
            Static("Additional Comments:", classes="instructions"),
            TextArea(id="comments-input", placeholder="Enter your feedback here..."),
            Horizontal(
                Button("Back", id="back-btn", variant="error"),
                Button("Submit", id="submit-btn", variant="success", disabled=True),
                classes="action-buttons"
            ),
            classes="main-content"
        )
        yield Footer()
    
    def on_mount(self) -> None:
        """Called when the screen is mounted."""
        self.logger.info("Feedback screen mounted")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id.startswith("rating-"):
            rating_value = int(event.button.id.split("-")[1])
            self._set_rating(rating_value)
        elif event.button.id == "back-btn":
            self.app.pop_screen()
        elif event.button.id == "submit-btn":
            self._submit_feedback()
    
    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Handle text area changes."""
        if event.text_area.id == "comments-input":
            # Enable submit button if rating is set
            submit_btn = self.query_one("#submit-btn", Button)
            submit_btn.disabled = self.rating == 0
    
    def _set_rating(self, rating: int) -> None:
        """Set the rating."""
        self.rating = rating
        self.logger.info(f"Rating set to: {rating}")
        
        # Update button styles
        for i in range(1, 6):
            btn = self.query_one(f"#rating-{i}", Button)
            if i <= rating:
                btn.variant = "primary"
            else:
                btn.variant = "default"
        
        # Enable submit button if comments are provided
        comments_widget = self.query_one("#comments-input", TextArea)
        submit_btn = self.query_one("#submit-btn", Button)
        submit_btn.disabled = not bool(comments_widget.text.strip())
    
    def _submit_feedback(self) -> None:
        """Submit the feedback."""
        comments_widget = self.query_one("#comments-input", TextArea)
        comments = comments_widget.text.strip()
        
        if self.rating == 0:
            self.logger.warning("No rating selected")
            self.notify("Please select a rating", severity="warning")
            return
        
        self.logger.info(f"Feedback submitted: Rating {self.rating}, Comments: {comments[:50]}...")
        self.notify("Feedback submitted successfully!", severity="success")
        
        # In a real implementation, this would save to database
        # For this example, we'll just exit
        self.app.exit()

class SessionHistoryScreen(Screen):
    """
    Session history viewer.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logger = logging.getLogger(__name__)
        self.sessions = []
    
    def compose(self) -> ComposeResult:
        """Create child widgets."""
        yield Header()
        yield Vertical(
            Static("Session History", classes="screen-title"),
            DataTable(id="sessions-table"),
            Horizontal(
                Button("Back", id="back-btn", variant="error"),
                Button("Load Selected", id="load-btn", variant="primary", disabled=True),
                Button("Delete Selected", id="delete-btn", variant="error", disabled=True),
                classes="action-buttons"
            ),
            classes="main-content"
        )
        yield Footer()
    
    def on_mount(self) -> None:
        """Called when the screen is mounted."""
        self.logger.info("Session history screen mounted")
        
        # Load sessions
        self._load_sessions()
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "back-btn":
            self.app.pop_screen()
        elif event.button.id == "load-btn":
            self._load_selected_session()
        elif event.button.id == "delete-btn":
            self._delete_selected_session()
    
    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle row selection in the sessions table."""
        load_btn = self.query_one("#load-btn", Button)
        delete_btn = self.query_one("#delete-btn", Button)
        load_btn.disabled = False
        delete_btn.disabled = False
    
    def _load_sessions(self) -> None:
        """Load session history."""
        # In a real implementation, this would load from database
        # For this example, we'll create mock data
        self.sessions = [
            {
                "id": "session-001",
                "image": "sunset.jpg",
                "date": "2023-10-20 14:30:00",
                "prompt": "make the sky more dramatic",
                "status": "completed",
                "score": 0.85
            },
            {
                "id": "session-002",
                "image": "portrait.jpg",
                "date": "2023-10-21 09:15:00",
                "prompt": "enhance colors and contrast",
                "status": "completed",
                "score": 0.92
            },
            {
                "id": "session-003",
                "image": "landscape.jpg",
                "date": "2023-10-22 16:45:00",
                "prompt": "change background to blue",
                "status": "failed",
                "score": 0.32
            }
        ]
        
        # Update table
        self._update_sessions_table()
    
    def _update_sessions_table(self) -> None:
        """Update the sessions table."""
        table = self.query_one("#sessions-table", DataTable)
        
        # Clear existing columns and rows
        table.clear()
        
        # Add columns
        table.add_column("ID", width=12)
        table.add_column("Image", width=15)
        table.add_column("Date", width=20)
        table.add_column("Prompt", width=30)
        table.add_column("Status", width=12)
        table.add_column("Score", width=8)
        
        # Add rows
        for session in self.sessions:
            table.add_row(
                session["id"][:8],
                session["image"][:15],
                session["date"][:19],
                session["prompt"][:30],
                session["status"],
                f"{session['score']:.2f}"
            )
    
    def _load_selected_session(self) -> None:
        """Load the selected session."""
        table = self.query_one("#sessions-table", DataTable)
        selected_row = table.cursor_row
        
        if selected_row < len(self.sessions):
            session = self.sessions[selected_row]
            self.logger.info(f"Loading session: {session['id']}")
            self.notify(f"Loading session {session['id'][:8]}...", severity="information")
            
            # In a real implementation, this would load the session
            # For this example, we'll just go back to home
            self.app.pop_screen()
        else:
            self.logger.warning("No session selected")
            self.notify("Please select a session first", severity="warning")
    
    def _delete_selected_session(self) -> None:
        """Delete the selected session."""
        table = self.query_one("#sessions-table", DataTable)
        selected_row = table.cursor_row
        
        if selected_row < len(self.sessions):
            session = self.sessions[selected_row]
            self.logger.info(f"Deleting session: {session['id']}")
            self.notify(f"Deleting session {session['id'][:8]}...", severity="information")
            
            # In a real implementation, this would delete the session from database
            # For this example, we'll just remove from list
            del self.sessions[selected_row]
            self._update_sessions_table()
        else:
            self.logger.warning("No session selected")
            self.notify("Please select a session first", severity="warning")

class ExamplesScreen(Screen):
    """
    Examples viewer.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logger = logging.getLogger(__name__)
        self.examples = []
    
    def compose(self) -> ComposeResult:
        """Create child widgets."""
        yield Header()
        yield Vertical(
            Static("Example Edits", classes="screen-title"),
            Static("Browse example edits to get inspiration:", classes="instructions"),
            Container(
                *[Static(f"Example {i+1}", id=f"example-{i}", classes="example-preview") 
                  for i in range(5)],
                classes="examples-grid"
            ),
            Horizontal(
                Button("Back", id="back-btn", variant="error"),
                Button("Try Selected", id="try-btn", variant="primary", disabled=True),
                classes="action-buttons"
            ),
            classes="main-content"
        )
        yield Footer()
    
    def on_mount(self) -> None:
        """Called when the screen is mounted."""
        self.logger.info("Examples screen mounted")
        
        # Load examples
        self._load_examples()
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "back-btn":
            self.app.pop_screen()
        elif event.button.id == "try-btn":
            self._try_selected_example()
    
    def _load_examples(self) -> None:
        """Load example edits."""
        # In a real implementation, this would load from examples directory
        # For this example, we'll create mock data
        self.examples = [
            {
                "id": "example-001",
                "title": "Dramatic Sky Enhancement",
                "description": "Transform a normal sky into a stormy, dramatic scene",
                "image_before": "sky_normal.jpg",
                "image_after": "sky_dramatic.jpg",
                "prompt": "make the sky more dramatic with storm clouds and lighting"
            },
            {
                "id": "example-002",
                "title": "Color Enhancement",
                "description": "Enhance colors and contrast for better visual appeal",
                "image_before": "photo_normal.jpg",
                "image_after": "photo_enhanced.jpg",
                "prompt": "enhance colors and contrast for better visual appeal"
            },
            {
                "id": "example-003",
                "title": "Background Modification",
                "description": "Change background while preserving subject",
                "image_before": "subject_normal.jpg",
                "image_after": "subject_blue_bg.jpg",
                "prompt": "change background to blue while preserving subject"
            },
            {
                "id": "example-004",
                "title": "Style Transfer",
                "description": "Apply artistic style to a photograph",
                "image_before": "photo_real.jpg",
                "image_after": "photo_artistic.jpg",
                "prompt": "apply artistic painting style to photograph"
            },
            {
                "id": "example-005",
                "title": "Object Removal",
                "description": "Remove unwanted objects from image",
                "image_before": "scene_with_object.jpg",
                "image_after": "scene_without_object.jpg",
                "prompt": "remove unwanted object from scene"
            }
        ]
        
        # Update previews
        self._update_example_previews()
    
    def _update_example_previews(self) -> None:
        """Update example previews."""
        for i, example in enumerate(self.examples):
            if i < 5:  # Only show first 5 examples
                example_widget = self.query_one(f"#example-{i}", Static)
                example_widget.update(f"{example['title']}: {example['description']}")
    
    def _try_selected_example(self) -> None:
        """Try the selected example."""
        # In a real implementation, this would load the example and start editing
        # For this example, we'll just show a notification
        self.logger.info("Selected example would be loaded for editing")
        self.notify("Example would be loaded for editing", severity="information")

class DiagnosticsScreen(Screen):
    """
    System diagnostics viewer.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logger = logging.getLogger(__name__)
        self.diagnostics = {}
    
    def compose(self) -> ComposeResult:
        """Create child widgets."""
        yield Header()
        yield Vertical(
            Static("System Diagnostics", classes="screen-title"),
            Static("Checking system status...", id="diagnostics-status"),
            Container(
                Static("Loading diagnostics...", id="diagnostics-content"),
                classes="diagnostics-container"
            ),
            Horizontal(
                Button("Back", id="back-btn", variant="error"),
                Button("Refresh", id="refresh-btn", variant="primary"),
                classes="action-buttons"
            ),
            classes="main-content"
        )
        yield Footer()
    
    def on_mount(self) -> None:
        """Called when the screen is mounted."""
        self.logger.info("Diagnostics screen mounted")
        
        # Run diagnostics
        self._run_diagnostics()
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "back-btn":
            self.app.pop_screen()
        elif event.button.id == "refresh-btn":
            self._run_diagnostics()
    
    @work(exclusive=True)
    async def _run_diagnostics(self) -> None:
        """Run system diagnostics."""
        status_widget = self.query_one("#diagnostics-status", Static)
        content_widget = self.query_one("#diagnostics-content", Static)
        
        status_widget.update("Running diagnostics...")
        
        # Simulate diagnostics process
        await asyncio.sleep(2)  # Simulate processing time
        
        # In a real implementation, this would call the doctor command
        # For this example, we'll create mock diagnostics
        self.diagnostics = {
            "python_version": "3.9.2",
            "ollama_status": "Connected",
            "comfyui_status": "Connected",
            "gpu_available": True,
            "memory_available_gb": 12.5,
            "disk_space_gb": 450.2,
            "models_available": 3,
            "session_count": 15,
            "last_session": "2023-10-22 16:45:00"
        }
        
        # Update UI on main thread
        self.call_from_thread(self._update_diagnostics_display)
    
    def _update_diagnostics_display(self) -> None:
        """Update the diagnostics display."""
        status_widget = self.query_one("#diagnostics-status", Static)
        content_widget = self.query_one("#diagnostics-content", Static)
        
        status_widget.update("Diagnostics Complete")
        
        # Create diagnostics display
        diagnostics_lines = [
            f"Python Version: {self.diagnostics['python_version']}",
            f"Ollama Status: {self.diagnostics['ollama_status']}",
            f"ComfyUI Status: {self.diagnostics['comfyui_status']}",
            f"GPU Available: {'Yes' if self.diagnostics['gpu_available'] else 'No'}",
            f"Memory Available: {self.diagnostics['memory_available_gb']:.1f} GB",
            f"Disk Space: {self.diagnostics['disk_space_gb']:.1f} GB",
            f"Models Available: {self.diagnostics['models_available']}",
            f"Session Count: {self.diagnostics['session_count']}",
            f"Last Session: {self.diagnostics['last_session']}"
        ]
        
        content_widget.update("\\n".join(diagnostics_lines))

# Example usage
if __name__ == "__main__":
    # Example of using the advanced screen implementations
    
    # Initialize theme manager
    theme_manager = ThemeManager("edi_themes")
    print("Theme manager initialized")
    
    # Example: Get available themes
    themes = theme_manager.get_available_themes()
    print(f"Available themes: {themes}")
    
    # Example: Get dark theme CSS
    dark_css = theme_manager.get_theme_css("dark")
    print(f"Dark theme CSS length: {len(dark_css)} characters")
    
    # Example: Get light theme CSS
    light_css = theme_manager.get_theme_css("light")
    print(f"Light theme CSS length: {len(light_css)} characters")
    
    # Example: Get color systems
    dark_colors = theme_manager.get_color_system("dark")
    light_colors = theme_manager.get_color_system("light")
    print(f"Dark theme primary color: {dark_colors.primary}")
    print(f"Light theme primary color: {light_colors.primary}")
    
    # Example: Create custom theme
    custom_css = """
    /* Custom Theme */
    App {
        background: #f0f0f0;
        color: #333333;
    }
    
    Button {
        background: #cccccc;
        color: #333333;
        border: tall #999999;
    }
    
    Button:hover {
        background: #bbbbbb;
    }
    """
    
    custom_colors = ColorSystem(
        primary="#333333",
        secondary="#666666",
        accent="#999999",
        background="#f0f0f0",
        surface="#e0e0e0",
        panel="#e0e0e0",
        error="#ff0000",
        warning="#ff9900",
        success="#00cc00",
        info="#0099cc",
        text="#333333",
        text_secondary="#666666",
        text_disabled="#999999",
        border="#cccccc",
        highlight="#dddddd"
    )
    
    theme_manager.create_custom_theme("custom", custom_css, custom_colors)
    print("Created custom theme")
    
    # Example: List available themes after creating custom
    available_themes = theme_manager.get_available_themes()
    print(f"Available themes after creating custom: {available_themes}")
    
    # Example: Delete custom theme
    theme_manager.delete_custom_theme("custom")
    print("Deleted custom theme")
    
    print("Advanced screen implementations example completed!")
```