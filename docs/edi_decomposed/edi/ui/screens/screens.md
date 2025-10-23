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