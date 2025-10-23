# TUI Layer

[Back to Index](../index.md)

## Purpose
User interaction, display, navigation using Textual 0.87+

## Component Design

### Architecture
Screen-based navigation with reactive widgets

#### Screen Hierarchy

```
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

```
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