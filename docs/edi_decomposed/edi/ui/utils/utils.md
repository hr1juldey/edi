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