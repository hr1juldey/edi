# UI: Styles

[Back to TUI Layer](./tui_layer.md)

## Purpose
CSS styling for the Textual TUI - Contains theme files for dark and light mode styling.

## Style Files
- `dark_theme.tcss`: Styling definitions for dark mode
- `light_theme.tcss`: Styling definitions for light mode

### Details
- Textual CSS (TCSS) styling files
- Provides consistent visual appearance
- Supports both dark and light themes

## Technology Stack

- Textual CSS (TCSS) for styling
- Theme management

## See Docs

### Textual CSS (TCSS) Implementation Example
UI styling implementation for the EDI application:

```python
from textual.app import App
from textual.widgets import Header, Footer, Static, Button, Input
from textual.containers import Container, Vertical, Horizontal
from textual.css.stylesheet import Stylesheet
from textual.design import ColorSystem
from textual.color import Color
import json
from typing import Dict, Any
from pathlib import Path

class EDIDarkTheme:
    """
    Dark theme styling for the EDI application.
    """
    
    # Color palette for dark theme
    COLORS = {
        "background": "#1e1e1e",
        "surface": "#2d2d2d",
        "primary": "#64b5f6",
        "secondary": "#81c784",
        "accent": "#ff8a65",
        "error": "#f44336",
        "warning": "#ffca28",
        "success": "#4caf50",
        "info": "#2196f3",
        "text_primary": "#ffffff",
        "text_secondary": "#bbbbbb",
        "text_disabled": "#777777",
        "border": "#444444",
        "highlight": "#333333"
    }
    
    # CSS styling definitions for dark mode
    TCSS = """
    /* Dark Theme Styles */
    
    /* Base application styling */
    App {
        background: $background;
        color: $text_primary;
    }
    
    /* Header styling */
    Header {
        background: $surface;
        color: $text_primary;
        height: 3;
        border-bottom: solid $border;
    }
    
    Header.-tall {
        height: 5;
    }
    
    /* Footer styling */
    Footer {
        background: $surface;
        color: $text_secondary;
        height: 3;
        border-top: solid $border;
    }
    
    Footer > .footer--highlight {
        background: $highlight;
        color: $primary;
    }
    
    Footer > .footer--key {
        text-style: bold;
        background: $surface;
        color: $primary;
    }
    
    /* Container styling */
    Container {
        background: $background;
        padding: 1;
    }
    
    Vertical {
        background: $background;
        padding: 0 1;
    }
    
    Horizontal {
        background: $background;
        padding: 1 0;
    }
    
    /* Widget styling */
    Static {
        background: $background;
        color: $text_primary;
    }
    
    Static.title {
        text-align: center;
        text-style: bold;
        color: $primary;
        margin: 1 0;
    }
    
    Static.subtitle {
        text-align: center;
        color: $text_secondary;
        margin: 0 0 2 0;
    }
    
    Static.section-header {
        text-style: bold underline;
        color: $secondary;
        margin: 1 0;
    }
    
    Static.modal-title {
        text-align: center;
        text-style: bold;
        color: $accent;
        margin: 1 0;
    }
    
    Static.modal-content {
        color: $text_primary;
        margin: 1 0;
    }
    
    /* Button styling */
    Button {
        background: $surface;
        color: $text_primary;
        border: tall $border;
        height: 3;
        margin: 0 1;
    }
    
    Button:hover {
        background: $highlight;
    }
    
    Button:focus {
        border: tall $primary;
    }
    
    Button.-disabled {
        background: $surface;
        color: $text_disabled;
        border: tall $border;
    }
    
    Button.success {
        background: $success;
        color: $text_primary;
        border: tall $success;
    }
    
    Button.success:hover {
        background: #388e3c;
        border: tall #388e3c;
    }
    
    Button.error {
        background: $error;
        color: $text_primary;
        border: tall $error;
    }
    
    Button.error:hover {
        background: #d32f2f;
        border: tall #d32f2f;
    }
    
    Button.primary {
        background: $primary;
        color: $text_primary;
        border: tall $primary;
    }
    
    Button.primary:hover {
        background: #1976d2;
        border: tall #1976d2;
    }
    
    Button.warning {
        background: $warning;
        color: #000000;
        border: tall $warning;
    }
    
    Button.warning:hover {
        background: #ffb300;
        border: tall #ffb300;
    }
    
    /* Input styling */
    Input {
        background: $surface;
        color: $text_primary;
        border: tall $border;
        height: 3;
        padding: 0 1;
    }
    
    Input:focus {
        border: tall $primary;
    }
    
    Input.-invalid {
        border: tall $error;
    }
    
    /* Text area styling */
    TextArea {
        background: $surface;
        color: $text_primary;
        border: tall $border;
        padding: 1;
    }
    
    TextArea:focus {
        border: tall $primary;
    }
    
    /* List styling */
    ListView {
        background: $background;
        border: tall $border;
    }
    
    ListItem {
        background: $surface;
        color: $text_primary;
        height: 3;
        padding: 0 1;
    }
    
    ListItem.--highlight {
        background: $highlight;
        color: $primary;
    }
    
    ListItem:hover {
        background: $highlight;
    }
    
    /* Modal styling */
    .modal-container {
        background: $surface;
        border: thick $accent;
        padding: 2;
    }
    
    .modal-title {
        text-align: center;
        text-style: bold;
        color: $accent;
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
        align: center middle;
    }
    
    /* Status indicator styling */
    .status-ok {
        color: $success;
    }
    
    .status-warning {
        color: $warning;
    }
    
    .status-error {
        color: $error;
    }
    
    .status-info {
        color: $info;
    }
    
    /* Progress bar styling */
    ProgressBar {
        background: $surface;
        bar-color: $primary;
        height: 1;
    }
    
    ProgressBar.--complete {
        bar-color: $success;
    }
    
    ProgressBar.--error {
        bar-color: $error;
    }
    
    /* Notification styling */
    .notification-success {
        background: $success;
        color: $text_primary;
        border: tall $success;
    }
    
    .notification-error {
        background: $error;
        color: $text_primary;
        border: tall $error;
    }
    
    .notification-warning {
        background: $warning;
        color: #000000;
        border: tall $warning;
    }
    
    .notification-info {
        background: $info;
        color: $text_primary;
        border: tall $info;
    }
    
    /* Card styling */
    .card {
        background: $surface;
        border: tall $border;
        padding: 1;
        margin: 1;
    }
    
    .card-title {
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
    }
    
    .card-content {
        color: $text_primary;
    }
    
    /* Tooltip styling */
    Tooltip {
        background: $surface;
        color: $text_primary;
        border: tall $border;
    }
    
    /* Scrollbar styling */
    .-scrollbar-thumb {
        background: $border;
    }
    
    .-scrollbar-thumb:hover {
        background: $primary;
    }
    
    /* Focus styling */
    :focus {
        tint: $primary 10%;
    }
    """
    
    @classmethod
    def get_color_system(cls) -> ColorSystem:
        """
        Get the color system for the dark theme.
        
        Returns:
            ColorSystem object with dark theme colors
        """
        return ColorSystem(
            primary=cls.COLORS["primary"],
            secondary=cls.COLORS["secondary"],
            accent=cls.COLORS["accent"],
            background=cls.COLORS["background"],
            surface=cls.COLORS["surface"],
            panel=cls.COLORS["surface"],
            error=cls.COLORS["error"],
            warning=cls.COLORS["warning"],
            success=cls.COLORS["success"],
            info=cls.COLORS["info"],
            text=cls.COLORS["text_primary"],
            text_secondary=cls.COLORS["text_secondary"],
            text_disabled=cls.COLORS["text_disabled"],
            border=cls.COLORS["border"],
            highlight=cls.COLORS["highlight"]
        )

class EDILightTheme:
    """
    Light theme styling for the EDI application.
    """
    
    # Color palette for light theme
    COLORS = {
        "background": "#ffffff",
        "surface": "#f5f5f5",
        "primary": "#1976d2",
        "secondary": "#388e3c",
        "accent": "#f57c00",
        "error": "#d32f2f",
        "warning": "#ffb300",
        "success": "#388e3c",
        "info": "#1976d2",
        "text_primary": "#212121",
        "text_secondary": "#757575",
        "text_disabled": "#bdbdbd",
        "border": "#e0e0e0",
        "highlight": "#eeeeee"
    }
    
    # CSS styling definitions for light mode
    TCSS = """
    /* Light Theme Styles */
    
    /* Base application styling */
    App {
        background: $background;
        color: $text_primary;
    }
    
    /* Header styling */
    Header {
        background: $surface;
        color: $text_primary;
        height: 3;
        border-bottom: solid $border;
    }
    
    Header.-tall {
        height: 5;
    }
    
    /* Footer styling */
    Footer {
        background: $surface;
        color: $text_secondary;
        height: 3;
        border-top: solid $border;
    }
    
    Footer > .footer--highlight {
        background: $highlight;
        color: $primary;
    }
    
    Footer > .footer--key {
        text-style: bold;
        background: $surface;
        color: $primary;
    }
    
    /* Container styling */
    Container {
        background: $background;
        padding: 1;
    }
    
    Vertical {
        background: $background;
        padding: 0 1;
    }
    
    Horizontal {
        background: $background;
        padding: 1 0;
    }
    
    /* Widget styling */
    Static {
        background: $background;
        color: $text_primary;
    }
    
    Static.title {
        text-align: center;
        text-style: bold;
        color: $primary;
        margin: 1 0;
    }
    
    Static.subtitle {
        text-align: center;
        color: $text_secondary;
        margin: 0 0 2 0;
    }
    
    Static.section-header {
        text-style: bold underline;
        color: $secondary;
        margin: 1 0;
    }
    
    Static.modal-title {
        text-align: center;
        text-style: bold;
        color: $accent;
        margin: 1 0;
    }
    
    Static.modal-content {
        color: $text_primary;
        margin: 1 0;
    }
    
    /* Button styling */
    Button {
        background: $surface;
        color: $text_primary;
        border: tall $border;
        height: 3;
        margin: 0 1;
    }
    
    Button:hover {
        background: $highlight;
    }
    
    Button:focus {
        border: tall $primary;
    }
    
    Button.-disabled {
        background: $surface;
        color: $text_disabled;
        border: tall $border;
    }
    
    Button.success {
        background: $success;
        color: $text_primary;
        border: tall $success;
    }
    
    Button.success:hover {
        background: #2e7d32;
        border: tall #2e7d32;
        color: $text_primary;
    }
    
    Button.error {
        background: $error;
        color: $text_primary;
        border: tall $error;
    }
    
    Button.error:hover {
        background: #c62828;
        border: tall #c62828;
        color: $text_primary;
    }
    
    Button.primary {
        background: $primary;
        color: $text_primary;
        border: tall $primary;
    }
    
    Button.primary:hover {
        background: #1565c0;
        border: tall #1565c0;
        color: $text_primary;
    }
    
    Button.warning {
        background: $warning;
        color: $text_primary;
        border: tall $warning;
    }
    
    Button.warning:hover {
        background: #ffa000;
        border: tall #ffa000;
        color: $text_primary;
    }
    
    /* Input styling */
    Input {
        background: $surface;
        color: $text_primary;
        border: tall $border;
        height: 3;
        padding: 0 1;
    }
    
    Input:focus {
        border: tall $primary;
    }
    
    Input.-invalid {
        border: tall $error;
    }
    
    /* Text area styling */
    TextArea {
        background: $surface;
        color: $text_primary;
        border: tall $border;
        padding: 1;
    }
    
    TextArea:focus {
        border: tall $primary;
    }
    
    /* List styling */
    ListView {
        background: $background;
        border: tall $border;
    }
    
    ListItem {
        background: $surface;
        color: $text_primary;
        height: 3;
        padding: 0 1;
    }
    
    ListItem.--highlight {
        background: $highlight;
        color: $primary;
    }
    
    ListItem:hover {
        background: $highlight;
    }
    
    /* Modal styling */
    .modal-container {
        background: $surface;
        border: thick $accent;
        padding: 2;
    }
    
    .modal-title {
        text-align: center;
        text-style: bold;
        color: $accent;
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
        align: center middle;
    }
    
    /* Status indicator styling */
    .status-ok {
        color: $success;
    }
    
    .status-warning {
        color: $warning;
    }
    
    .status-error {
        color: $error;
    }
    
    .status-info {
        color: $info;
    }
    
    /* Progress bar styling */
    ProgressBar {
        background: $surface;
        bar-color: $primary;
        height: 1;
    }
    
    ProgressBar.--complete {
        bar-color: $success;
    }
    
    ProgressBar.--error {
        bar-color: $error;
    }
    
    /* Notification styling */
    .notification-success {
        background: $success;
        color: $text_primary;
        border: tall $success;
    }
    
    .notification-error {
        background: $error;
        color: $text_primary;
        border: tall $error;
    }
    
    .notification-warning {
        background: $warning;
        color: $text_primary;
        border: tall $warning;
    }
    
    .notification-info {
        background: $info;
        color: $text_primary;
        border: tall $info;
    }
    
    /* Card styling */
    .card {
        background: $surface;
        border: tall $border;
        padding: 1;
        margin: 1;
    }
    
    .card-title {
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
    }
    
    .card-content {
        color: $text_primary;
    }
    
    /* Tooltip styling */
    Tooltip {
        background: $surface;
        color: $text_primary;
        border: tall $border;
    }
    
    /* Scrollbar styling */
    .-scrollbar-thumb {
        background: $border;
    }
    
    .-scrollbar-thumb:hover {
        background: $primary;
    }
    
    /* Focus styling */
    :focus {
        tint: $primary 10%;
    }
    """
    
    @classmethod
    def get_color_system(cls) -> ColorSystem:
        """
        Get the color system for the light theme.
        
        Returns:
            ColorSystem object with light theme colors
        """
        return ColorSystem(
            primary=cls.COLORS["primary"],
            secondary=cls.COLORS["secondary"],
            accent=cls.COLORS["accent"],
            background=cls.COLORS["background"],
            surface=cls.COLORS["surface"],
            panel=cls.COLORS["surface"],
            error=cls.COLORS["error"],
            warning=cls.COLORS["warning"],
            success=cls.COLORS["success"],
            info=cls.COLORS["info"],
            text=cls.COLORS["text_primary"],
            text_secondary=cls.COLORS["text_secondary"],
            text_disabled=cls.COLORS["text_disabled"],
            border=cls.COLORS["border"],
            highlight=cls.COLORS["highlight"]
        )

class ThemeManager:
    """
    Manages theme switching and application of styling for EDI.
    """
    
    def __init__(self, theme_dir: str = "themes"):
        self.theme_dir = Path(theme_dir)
        self.theme_dir.mkdir(exist_ok=True)
        self.current_theme = "dark"
        self.logger = logging.getLogger(__name__)
        
        # Initialize default themes
        self._create_default_themes()
    
    def _create_default_themes(self):
        """
        Create default theme files if they don't exist.
        """
        # Create dark theme file
        dark_theme_path = self.theme_dir / "dark_theme.tcss"
        if not dark_theme_path.exists():
            with open(dark_theme_path, 'w') as f:
                f.write(EDIDarkTheme.TCSS)
            self.logger.info(f"Created dark theme at: {dark_theme_path}")
        
        # Create light theme file
        light_theme_path = self.theme_dir / "light_theme.tcss"
        if not light_theme_path.exists():
            with open(light_theme_path, 'w') as f:
                f.write(EDILightTheme.TCSS)
            self.logger.info(f"Created light theme at: {light_theme_path}")
    
    def get_theme_css(self, theme_name: str) -> str:
        """
        Get CSS for a specific theme.
        
        Args:
            theme_name: Name of theme to get CSS for
            
        Returns:
            CSS string for the theme
        """
        theme_path = self.theme_dir / f"{theme_name}_theme.tcss"
        
        if theme_path.exists():
            with open(theme_path, 'r') as f:
                return f.read()
        else:
            # Return default theme CSS
            if theme_name == "dark":
                return EDIDarkTheme.TCSS
            elif theme_name == "light":
                return EDILightTheme.TCSS
            else:
                self.logger.warning(f"Unknown theme {theme_name}, returning dark theme")
                return EDIDarkTheme.TCSS
    
    def get_color_system(self, theme_name: str) -> ColorSystem:
        """
        Get color system for a specific theme.
        
        Args:
            theme_name: Name of theme to get color system for
            
        Returns:
            ColorSystem object for the theme
        """
        if theme_name == "dark":
            return EDIDarkTheme.get_color_system()
        elif theme_name == "light":
            return EDILightTheme.get_color_system()
        else:
            self.logger.warning(f"Unknown theme {theme_name}, returning dark theme color system")
            return EDIDarkTheme.get_color_system()
    
    def switch_theme(self, app: App, theme_name: str):
        """
        Switch the application theme.
        
        Args:
            app: Textual application instance
            theme_name: Name of theme to switch to
        """
        if theme_name not in ["dark", "light"]:
            self.logger.error(f"Invalid theme name: {theme_name}")
            return
        
        # Apply new theme CSS
        css_content = self.get_theme_css(theme_name)
        app.stylesheet.parse(css_content)
        app.stylesheet.apply(app)
        
        # Apply new color system
        color_system = self.get_color_system(theme_name)
        app.design = color_system
        
        # Update current theme
        self.current_theme = theme_name
        self.logger.info(f"Switched to {theme_name} theme")
    
    def get_available_themes(self) -> List[str]:
        """
        Get list of available themes.
        
        Returns:
            List of available theme names
        """
        themes = []
        for theme_file in self.theme_dir.glob("*_theme.tcss"):
            theme_name = theme_file.stem.replace("_theme", "")
            themes.append(theme_name)
        return themes
    
    def create_custom_theme(self, theme_name: str, css_content: str, color_system: ColorSystem):
        """
        Create a custom theme.
        
        Args:
            theme_name: Name for the custom theme
            css_content: CSS content for the theme
            color_system: ColorSystem object for the theme
        """
        theme_path = self.theme_dir / f"{theme_name}_theme.tcss"
        
        with open(theme_path, 'w') as f:
            f.write(css_content)
        
        self.logger.info(f"Created custom theme: {theme_name}")
    
    def delete_custom_theme(self, theme_name: str) -> bool:
        """
        Delete a custom theme.
        
        Args:
            theme_name: Name of theme to delete
            
        Returns:
            Boolean indicating success
        """
        if theme_name in ["dark", "light"]:
            self.logger.error("Cannot delete built-in themes")
            return False
        
        theme_path = self.theme_dir / f"{theme_name}_theme.tcss"
        
        if theme_path.exists():
            theme_path.unlink()
            self.logger.info(f"Deleted custom theme: {theme_name}")
            return True
        else:
            self.logger.warning(f"Theme {theme_name} not found")
            return False

# Example usage
if __name__ == "__main__":
    # Example of using the theme manager
    theme_manager = ThemeManager("edi_themes")
    
    print("Theme Manager initialized")
    print(f"Available themes: {theme_manager.get_available_themes()}")
    
    # Get dark theme CSS
    dark_css = theme_manager.get_theme_css("dark")
    print(f"Dark theme CSS length: {len(dark_css)} characters")
    
    # Get light theme CSS
    light_css = theme_manager.get_theme_css("light")
    print(f"Light theme CSS length: {len(light_css)} characters")
    
    # Get color systems
    dark_colors = theme_manager.get_color_system("dark")
    light_colors = theme_manager.get_color_system("light")
    
    print(f"Dark theme primary color: {dark_colors.primary}")
    print(f"Light theme primary color: {light_colors.primary}")
    
    # Create a custom theme
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
    
    # List available themes
    available_themes = theme_manager.get_available_themes()
    print(f"Available themes after creating custom: {available_themes}")
    
    # Delete custom theme
    theme_manager.delete_custom_theme("custom")
    print("Deleted custom theme")
    
    print("Theme management example completed!")
```

### Advanced Theme Management Implementation
Enhanced theme management with dynamic loading and user preferences:

```python
import asyncio
import aiofiles
from pathlib import Path
import json
from typing import Dict, Any, List, Optional
from textual.app import App
from textual.design import ColorSystem
from textual.css.stylesheet import Stylesheet
import logging
from dataclasses import dataclass
from enum import Enum

class ThemeType(Enum):
    """Enumeration for theme types."""
    DARK = "dark"
    LIGHT = "light"
    CUSTOM = "custom"

@dataclass
class ThemeDefinition:
    """Definition of a theme with metadata."""
    name: str
    type: ThemeType
    css_path: Path
    color_system: ColorSystem
    author: str = "Unknown"
    version: str = "1.0"
    description: str = ""
    preview_image: Optional[Path] = None
    created_at: str = ""

class AdvancedThemeManager:
    """
    Advanced theme manager with dynamic loading and user preferences.
    """
    
    def __init__(self, themes_dir: str = "themes", user_prefs_path: str = "user_preferences.json"):
        self.themes_dir = Path(themes_dir)
        self.user_prefs_path = Path(user_prefs_path)
        self.themes_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.themes: Dict[str, ThemeDefinition] = {}
        self.current_theme: Optional[str] = None
        self.user_preferences: Dict[str, Any] = {}
        
        # Initialize theme manager
        self._load_user_preferences()
        self._discover_themes()
        self._load_builtin_themes()
    
    def _load_user_preferences(self):
        """
        Load user theme preferences.
        """
        try:
            if self.user_prefs_path.exists():
                with open(self.user_prefs_path, 'r') as f:
                    self.user_preferences = json.load(f)
                
                # Set current theme from preferences
                self.current_theme = self.user_preferences.get("theme", "dark")
            else:
                # Create default preferences
                self.user_preferences = {
                    "theme": "dark",
                    "theme_auto_switch": True,
                    "preferred_themes": ["dark", "light"]
                }
                self._save_user_preferences()
                self.current_theme = "dark"
                
        except Exception as e:
            self.logger.error(f"Error loading user preferences: {str(e)}")
            self.user_preferences = {"theme": "dark"}
            self.current_theme = "dark"
    
    def _save_user_preferences(self):
        """
        Save user theme preferences.
        """
        try:
            with open(self.user_prefs_path, 'w') as f:
                json.dump(self.user_preferences, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving user preferences: {str(e)}")
    
    def _discover_themes(self):
        """
        Discover custom themes in the themes directory.
        """
        try:
            # Look for theme definition files
            for theme_file in self.themes_dir.glob("*.json"):
                try:
                    with open(theme_file, 'r') as f:
                        theme_data = json.load(f)
                    
                    # Create theme definition
                    theme_def = ThemeDefinition(
                        name=theme_data["name"],
                        type=ThemeType(theme_data.get("type", "custom")),
                        css_path=self.themes_dir / theme_data["css_file"],
                        color_system=ColorSystem(**theme_data["color_system"]),
                        author=theme_data.get("author", "Unknown"),
                        version=theme_data.get("version", "1.0"),
                        description=theme_data.get("description", ""),
                        preview_image=self.themes_dir / theme_data["preview_image"] if "preview_image" in theme_data else None,
                        created_at=theme_data.get("created_at", "")
                    )
                    
                    # Register theme
                    self.themes[theme_def.name] = theme_def
                    self.logger.info(f"Discovered theme: {theme_def.name}")
                    
                except Exception as e:
                    self.logger.error(f"Error loading theme {theme_file}: {str(e)}")
                    
        except Exception as e:
            self.logger.error(f"Error discovering themes: {str(e)}")
    
    def _load_builtin_themes(self):
        """
        Load builtin themes (dark and light).
        """
        # Dark theme
        dark_theme = ThemeDefinition(
            name="dark",
            type=ThemeType.DARK,
            css_path=self.themes_dir / "dark_theme.tcss",
            color_system=EDIDarkTheme.get_color_system(),
            author="EDI Team",
            version="1.0",
            description="Default dark theme for EDI"
        )
        self.themes["dark"] = dark_theme
        
        # Light theme
        light_theme = ThemeDefinition(
            name="light",
            type=ThemeType.LIGHT,
            css_path=self.themes_dir / "light_theme.tcss",
            color_system=EDILightTheme.get_color_system(),
            author="EDI Team",
            version="1.0",
            description="Default light theme for EDI"
        )
        self.themes["light"] = light_theme
    
    async def apply_theme(self, app: App, theme_name: str) -> bool:
        """
        Apply a theme to the application.
        
        Args:
            app: Textual application instance
            theme_name: Name of theme to apply
            
        Returns:
            Boolean indicating success
        """
        if theme_name not in self.themes:
            self.logger.error(f"Theme {theme_name} not found")
            return False
        
        try:
            theme_def = self.themes[theme_name]
            
            # Load CSS content
            async with aiofiles.open(theme_def.css_path, 'r') as f:
                css_content = await f.read()
            
            # Apply theme CSS
            app.stylesheet.parse(css_content)
            app.stylesheet.apply(app)
            
            # Apply color system
            app.design = theme_def.color_system
            
            # Update current theme
            self.current_theme = theme_name
            
            # Save preference
            self.user_preferences["theme"] = theme_name
            self._save_user_preferences()
            
            self.logger.info(f"Applied theme: {theme_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error applying theme {theme_name}: {str(e)}")
            return False
    
    async def get_theme_preview(self, theme_name: str) -> Optional[str]:
        """
        Get a preview of a theme.
        
        Args:
            theme_name: Name of theme to preview
            
        Returns:
            Preview image path or None if no preview available
        """
        if theme_name not in self.themes:
            return None
        
        theme_def = self.themes[theme_name]
        if theme_def.preview_image and theme_def.preview_image.exists():
            return str(theme_def.preview_image)
        
        return None
    
    async def list_themes(self) -> List[Dict[str, Any]]:
        """
        List all available themes with metadata.
        
        Returns:
            List of theme information dictionaries
        """
        theme_list = []
        
        for name, theme_def in self.themes.items():
            theme_info = {
                "name": name,
                "type": theme_def.type.value,
                "author": theme_def.author,
                "version": theme_def.version,
                "description": theme_def.description,
                "has_preview": theme_def.preview_image is not None and theme_def.preview_image.exists(),
                "is_current": name == self.current_theme
            }
            theme_list.append(theme_info)
        
        return theme_list
    
    async def create_theme(self, 
                         theme_name: str, 
                         css_content: str, 
                         color_system: ColorSystem,
                         theme_type: ThemeType = ThemeType.CUSTOM,
                         description: str = "",
                         author: str = "User") -> bool:
        """
        Create a new custom theme.
        
        Args:
            theme_name: Name for the new theme
            css_content: CSS content for the theme
            color_system: ColorSystem object for the theme
            theme_type: Type of theme (default: CUSTOM)
            description: Description of the theme
            author: Author of the theme
            
        Returns:
            Boolean indicating success
        """
        try:
            # Create CSS file
            css_path = self.themes_dir / f"{theme_name}_theme.tcss"
            async with aiofiles.open(css_path, 'w') as f:
                await f.write(css_content)
            
            # Create theme definition
            theme_def = ThemeDefinition(
                name=theme_name,
                type=theme_type,
                css_path=css_path,
                color_system=color_system,
                author=author,
                version="1.0",
                description=description,
                created_at=datetime.now().isoformat()
            )
            
            # Save theme definition
            theme_def_path = self.themes_dir / f"{theme_name}_theme.json"
            theme_def_data = {
                "name": theme_name,
                "type": theme_type.value,
                "css_file": f"{theme_name}_theme.tcss",
                "color_system": {
                    "primary": str(color_system.primary),
                    "secondary": str(color_system.secondary),
                    "accent": str(color_system.accent),
                    "background": str(color_system.background),
                    "surface": str(color_system.surface),
                    "panel": str(color_system.panel),
                    "error": str(color_system.error),
                    "warning": str(color_system.warning),
                    "success": str(color_system.success),
                    "info": str(color_system.info),
                    "text": str(color_system.text),
                    "text_secondary": str(color_system.text_secondary),
                    "text_disabled": str(color_system.text_disabled),
                    "border": str(color_system.border),
                    "highlight": str(color_system.highlight)
                },
                "author": author,
                "version": "1.0",
                "description": description,
                "created_at": datetime.now().isoformat()
            }
            
            async with aiofiles.open(theme_def_path, 'w') as f:
                await f.write(json.dumps(theme_def_data, indent=2))
            
            # Register theme
            self.themes[theme_name] = theme_def
            
            self.logger.info(f"Created theme: {theme_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating theme {theme_name}: {str(e)}")
            return False
    
    async def delete_theme(self, theme_name: str) -> bool:
        """
        Delete a custom theme.
        
        Args:
            theme_name: Name of theme to delete
            
        Returns:
            Boolean indicating success
        """
        if theme_name in ["dark", "light"]:
            self.logger.error("Cannot delete builtin themes")
            return False
        
        if theme_name not in self.themes:
            self.logger.error(f"Theme {theme_name} not found")
            return False
        
        try:
            theme_def = self.themes[theme_name]
            
            # Delete CSS file
            if theme_def.css_path.exists():
                theme_def.css_path.unlink()
            
            # Delete definition file
            theme_def_path = self.themes_dir / f"{theme_name}_theme.json"
            if theme_def_path.exists():
                theme_def_path.unlink()
            
            # Remove from registry
            del self.themes[theme_name]
            
            # If this was the current theme, switch to default
            if self.current_theme == theme_name:
                await self.apply_theme(App(), "dark")
            
            self.logger.info(f"Deleted theme: {theme_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting theme {theme_name}: {str(e)}")
            return False
    
    async def export_theme(self, theme_name: str, export_path: str) -> bool:
        """
        Export a theme to a file for sharing.
        
        Args:
            theme_name: Name of theme to export
            export_path: Path to export to
            
        Returns:
            Boolean indicating success
        """
        if theme_name not in self.themes:
            self.logger.error(f"Theme {theme_name} not found")
            return False
        
        try:
            theme_def = self.themes[theme_name]
            
            # Create export package
            export_data = {
                "theme": {
                    "name": theme_name,
                    "type": theme_def.type.value,
                    "author": theme_def.author,
                    "version": theme_def.version,
                    "description": theme_def.description,
                    "created_at": theme_def.created_at
                },
                "color_system": {
                    "primary": str(theme_def.color_system.primary),
                    "secondary": str(theme_def.color_system.secondary),
                    "accent": str(theme_def.color_system.accent),
                    "background": str(theme_def.color_system.background),
                    "surface": str(theme_def.color_system.surface),
                    "panel": str(theme_def.color_system.panel),
                    "error": str(theme_def.color_system.error),
                    "warning": str(theme_def.color_system.warning),
                    "success": str(theme_def.color_system.success),
                    "info": str(theme_def.color_system.info),
                    "text": str(theme_def.color_system.text),
                    "text_secondary": str(theme_def.color_system.text_secondary),
                    "text_disabled": str(theme_def.color_system.text_disabled),
                    "border": str(theme_def.color_system.border),
                    "highlight": str(theme_def.color_system.highlight)
                }
            }
            
            # Read CSS content
            async with aiofiles.open(theme_def.css_path, 'r') as f:
                css_content = await f.read()
            export_data["css_content"] = css_content
            
            # Save export package
            async with aiofiles.open(export_path, 'w') as f:
                await f.write(json.dumps(export_data, indent=2))
            
            self.logger.info(f"Exported theme {theme_name} to {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting theme {theme_name}: {str(e)}")
            return False
    
    async def import_theme(self, import_path: str) -> bool:
        """
        Import a theme from a file.
        
        Args:
            import_path: Path to import from
            
        Returns:
            Boolean indicating success
        """
        try:
            # Read import package
            async with aiofiles.open(import_path, 'r') as f:
                import_data = json.loads(await f.read())
            
            theme_info = import_data["theme"]
            color_system_data = import_data["color_system"]
            css_content = import_data["css_content"]
            
            # Create color system
            color_system = ColorSystem(
                primary=color_system_data["primary"],
                secondary=color_system_data["secondary"],
                accent=color_system_data["accent"],
                background=color_system_data["background"],
                surface=color_system_data["surface"],
                panel=color_system_data["panel"],
                error=color_system_data["error"],
                warning=color_system_data["warning"],
                success=color_system_data["success"],
                info=color_system_data["info"],
                text=color_system_data["text"],
                text_secondary=color_system_data["text_secondary"],
                text_disabled=color_system_data["text_disabled"],
                border=color_system_data["border"],
                highlight=color_system_data["highlight"]
            )
            
            # Create theme
            theme_name = theme_info["name"]
            success = await self.create_theme(
                theme_name=theme_name,
                css_content=css_content,
                color_system=color_system,
                theme_type=ThemeType(theme_info.get("type", "custom")),
                description=theme_info.get("description", ""),
                author=theme_info.get("author", "Imported")
            )
            
            if success:
                self.logger.info(f"Imported theme: {theme_name}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error importing theme from {import_path}: {str(e)}")
            return False
    
    async def get_theme_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about available themes.
        
        Returns:
            Dictionary with theme statistics
        """
        builtin_count = sum(1 for theme in self.themes.values() if theme.type in [ThemeType.DARK, ThemeType.LIGHT])
        custom_count = sum(1 for theme in self.themes.values() if theme.type == ThemeType.CUSTOM)
        
        return {
            "total_themes": len(self.themes),
            "builtin_themes": builtin_count,
            "custom_themes": custom_count,
            "current_theme": self.current_theme,
            "preferred_themes": self.user_preferences.get("preferred_themes", []),
            "auto_switch_enabled": self.user_preferences.get("theme_auto_switch", True)
        }
    
    async def auto_switch_theme(self, is_dark_mode: bool) -> bool:
        """
        Automatically switch theme based on system preferences.
        
        Args:
            is_dark_mode: Whether system is in dark mode
            
        Returns:
            Boolean indicating if theme was switched
        """
        if not self.user_preferences.get("theme_auto_switch", True):
            return False
        
        target_theme = "dark" if is_dark_mode else "light"
        
        if self.current_theme != target_theme:
            # Apply new theme
            app = App()  # In real usage, this would be the actual app instance
            success = await self.apply_theme(app, target_theme)
            
            if success:
                self.logger.info(f"Auto-switched to {target_theme} theme")
                return True
            else:
                self.logger.error(f"Failed to auto-switch to {target_theme} theme")
                return False
        
        return False

# Example usage
async def main():
    # Initialize advanced theme manager
    theme_manager = AdvancedThemeManager("edi_themes_advanced", "user_prefs_advanced.json")
    
    print("Advanced Theme Manager initialized")
    
    # List available themes
    themes = await theme_manager.list_themes()
    print(f"Available themes: {len(themes)}")
    for theme in themes:
        print(f"  - {theme['name']} ({theme['type']}): {theme['description']}")
        print(f"    Author: {theme['author']}, Version: {theme['version']}")
        print(f"    Current: {'Yes' if theme['is_current'] else 'No'}")
        print(f"    Has preview: {'Yes' if theme['has_preview'] else 'No'}")
    
    # Get theme statistics
    stats = await theme_manager.get_theme_statistics()
    print(f"\nTheme Statistics:")
    print(f"  Total themes: {stats['total_themes']}")
    print(f"  Built-in themes: {stats['builtin_themes']}")
    print(f"  Custom themes: {stats['custom_themes']}")
    print(f"  Current theme: {stats['current_theme']}")
    print(f"  Auto-switch enabled: {stats['auto_switch_enabled']}")
    
    # Example: Create a custom theme
    custom_css = """
    /* Custom Theme */
    App {
        background: #f8f9fa;
        color: #212529;
    }
    
    Button {
        background: #e9ecef;
        color: #212529;
        border: tall #dee2e6;
    }
    
    Button:hover {
        background: #dde0e3;
    }
    
    Button.primary {
        background: #0d6efd;
        color: #ffffff;
        border: tall #0d6efd;
    }
    
    Button.primary:hover {
        background: #0b5ed7;
        border: tall #0b5ed7;
    }
    """
    
    custom_color_system = ColorSystem(
        primary="#0d6efd",
        secondary="#6c757d",
        accent="#d63384",
        background="#f8f9fa",
        surface="#ffffff",
        panel="#ffffff",
        error="#dc3545",
        warning="#ffc107",
        success="#198754",
        info="#0dcaf0",
        text="#212529",
        text_secondary="#6c757d",
        text_disabled="#adb5bd",
        border="#dee2e6",
        highlight="#e9ecef"
    )
    
    success = await theme_manager.create_theme(
        theme_name="custom_light",
        css_content=custom_css,
        color_system=custom_color_system,
        description="A custom light theme with modern colors",
        author="User"
    )
    
    if success:
        print("\nCreated custom theme successfully")
    else:
        print("\nFailed to create custom theme")
    
    # Example: Export a theme
    export_success = await theme_manager.export_theme("custom_light", "custom_light_theme.json")
    if export_success:
        print("Exported custom theme")
    else:
        print("Failed to export custom theme")
    
    # Example: Import a theme (would work if we had an export file)
    # import_success = await theme_manager.import_theme("imported_theme.json")
    # if import_success:
    #     print("Imported theme successfully")
    # else:
    #     print("Failed to import theme")
    
    # Example: Delete a custom theme
    delete_success = await theme_manager.delete_theme("custom_light")
    if delete_success:
        print("Deleted custom theme")
    else:
        print("Failed to delete custom theme")
    
    print("Advanced theme management example completed!")

if __name__ == "__main__":
    asyncio.run(main())
```