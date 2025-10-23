# color_code_score()

[Back to UI Utils](../ui_utils.md)

## Related User Story
"As a user, I want clear visual feedback about the quality of my edits." (from PRD - implied by user experience requirements)

## Function Signature
`color_code_score(score) -> Rich markup`

## Parameters
- `score` - A numeric value representing a score (typically between 0 and 1, or 0 and 100)

## Returns
- `str` - Rich markup string that displays the score with appropriate color coding (green/yellow/red)

## Step-by-step Logic
1. Take the input score value
2. Determine the appropriate color based on the score range:
   - High score (>0.8 or >80): Green (indicates good quality)
   - Medium score (0.6-0.8 or 60-80): Yellow (indicates moderate quality)
   - Low score (<0.6 or <60): Red (indicates poor quality)
3. Create Rich markup that combines the score text with the appropriate color
4. Return the colored markup string that will display with color when rendered

## Color Mapping Strategy
- Green (>0.8): Indicates high quality or success (✓)
- Yellow (0.6-0.8): Indicates moderate quality or caution (⚠)
- Red (<0.6): Indicates low quality or failure (✗)
- Uses Rich's color system for terminal coloration
- May include symbols or icons along with the color coding

## Threshold Customization
- Default thresholds work for normalized scores (0-1)
- Can be adapted for percentage scores (0-100) or other ranges
- Thresholds can be configured based on the specific use case
- Consistent with validation metrics throughout the system

## Input/Output Data Structures
### Input
- score: Numeric value representing a quality score, alignment score, or other metric

### Output
- Rich-formatted string with color coding applied:
  - Example: "[green]0.85[/green]", "[yellow]0.72[/yellow]", "[red]0.45[/red]"
  - The string will display with appropriate colors when rendered by Rich

## See Docs

### Python Implementation Example
Implementation of the color_code_score function:

```python
from typing import Union, Optional
import math

def color_code_score(score: Union[int, float], 
                    score_range: str = "normalized",
                    thresholds: Optional[tuple] = None,
                    show_symbol: bool = True,
                    precision: int = 2) -> str:
    """
    Color codes a score based on quality using Rich markup.
    
    Args:
        score: Numeric value representing a quality score, alignment score, or other metric
        score_range: Range type - "normalized" (0-1), "percentage" (0-100), or "custom"
        thresholds: Custom thresholds as (yellow_threshold, green_threshold) for normalized scores
        show_symbol: Whether to include symbols along with color coding
        precision: Number of decimal places to display
    
    Returns:
        Rich-formatted string with color coding applied
    """
    # Validate input
    if not isinstance(score, (int, float)):
        raise TypeError(f"Score must be numeric, got {type(score)}")
    
    if math.isnan(score) or math.isinf(score):
        return "[magenta]N/A[/magenta]" if not show_symbol else "[magenta]?[/magenta]"
    
    # Determine thresholds based on score range
    if thresholds:
        yellow_threshold, green_threshold = thresholds
    elif score_range == "percentage":
        # For percentage scores (0-100)
        yellow_threshold = 60.0
        green_threshold = 80.0
    else:
        # For normalized scores (0-1)
        yellow_threshold = 0.6
        green_threshold = 0.8
    
    # Format the score with specified precision
    formatted_score = f"{score:.{precision}f}"
    
    # Determine the appropriate color based on the score range
    if score >= green_threshold:
        # High score (>0.8 or >80): Green (indicates good quality)
        color = "green"
        symbol = "✓" if show_symbol else ""
    elif score >= yellow_threshold:
        # Medium score (0.6-0.8 or 60-80): Yellow (indicates moderate quality)
        color = "yellow"
        symbol = "⚠" if show_symbol else ""
    else:
        # Low score (<0.6 or <60): Red (indicates poor quality)
        color = "red"
        symbol = "✗" if show_symbol else ""
    
    # Create Rich markup that combines the score text with the appropriate color
    if symbol:
        return f"[{color}]{symbol} {formatted_score}[/{color}]"
    else:
        return f"[{color}]{formatted_score}[/{color}]"

def color_code_score_advanced(score: Union[int, float],
                              score_range: str = "normalized",
                              custom_thresholds: Optional[dict] = None,
                              show_percentage: bool = False,
                              show_bar: bool = False,
                              bar_length: int = 10) -> str:
    """
    Advanced color coding with additional visualization options.
    
    Args:
        score: Numeric value representing a quality score
        score_range: Range type - "normalized", "percentage", or "custom"
        custom_thresholds: Custom thresholds dictionary {'red': val, 'yellow': val, 'green': val}
        show_percentage: Whether to show percentage alongside normalized score
        show_bar: Whether to show a visual bar representation
        bar_length: Length of the visual bar
    
    Returns:
        Rich-formatted string with advanced color coding and visualization
    """
    # Validate input
    if not isinstance(score, (int, float)):
        raise TypeError(f"Score must be numeric, got {type(score)}")
    
    if math.isnan(score) or math.isinf(score):
        return "[magenta]N/A[/magenta]"
    
    # Determine thresholds
    if custom_thresholds:
        red_threshold = custom_thresholds.get('red', 0.6)
        yellow_threshold = custom_thresholds.get('yellow', 0.7)
        green_threshold = custom_thresholds.get('green', 0.85)
    elif score_range == "percentage":
        red_threshold = 30.0
        yellow_threshold = 60.0
        green_threshold = 80.0
    else:
        red_threshold = 0.3
        yellow_threshold = 0.6
        green_threshold = 0.8
    
    # Format the score
    if score_range == "percentage":
        formatted_score = f"{score:.1f}%"
    else:
        formatted_score = f"{score:.2f}"
        if show_percentage and 0 <= score <= 1:
            formatted_score += f" ({score*100:.0f}%)"
    
    # Determine color and symbol
    if score >= green_threshold:
        color = "green"
        symbol = "✓"
    elif score >= yellow_threshold:
        color = "yellow"
        symbol = "⚠"
    elif score >= red_threshold:
        color = "orange"  # Orange for medium-low
        symbol = "▼"
    else:
        color = "red"
        symbol = "✗"
    
    # Build the result string
    result_parts = [f"[{color}]{symbol} {formatted_score}[/{color}]"]
    
    # Add visual bar if requested
    if show_bar:
        # Clamp score to 0-1 range for bar calculation
        clamped_score = max(0, min(1, score if score_range != "percentage" else score/100))
        filled_length = int(clamped_score * bar_length)
        
        # Create bar with gradient colors
        bar_parts = []
        for i in range(bar_length):
            if i < filled_length:
                # Determine segment color based on position
                segment_ratio = i / bar_length
                if segment_ratio < red_threshold:
                    bar_parts.append("[red]█[/red]")
                elif segment_ratio < yellow_threshold:
                    bar_parts.append("[orange]█[/orange]")
                elif segment_ratio < green_threshold:
                    bar_parts.append("[yellow]█[/yellow]")
                else:
                    bar_parts.append("[green]█[/green]")
            else:
                bar_parts.append("[dim]░[/dim]")
        
        bar_string = "".join(bar_parts)
        result_parts.append(f" {bar_string}")
    
    return " ".join(result_parts)

def batch_color_code(scores: list, 
                    score_range: str = "normalized",
                    **kwargs) -> list:
    """
    Color codes multiple scores at once.
    
    Args:
        scores: List of numeric scores to color code
        score_range: Range type for all scores
        **kwargs: Additional arguments to pass to color_code_score
    
    Returns:
        List of Rich-formatted strings
    """
    return [color_code_score(score, score_range, **kwargs) for score in scores]

def color_legend(score_range: str = "normalized") -> str:
    """
    Generates a color legend for score interpretation.
    
    Returns:
        Rich-formatted string with color legend
    """
    if score_range == "percentage":
        red_range = "< 60%"
        yellow_range = "60-80%"
        green_range = "> 80%"
    else:
        red_range = "< 0.6"
        yellow_range = "0.6-0.8"
        green_range = "> 0.8"
    
    legend = (
        f"[red]✗ {red_range}[/red]  "
        f"[yellow]⚠ {yellow_range}[/yellow]  "
        f"[green]✓ {green_range}[/green]"
    )
    
    return legend

# Example usage
if __name__ == "__main__":
    # Test basic color coding
    test_scores = [0.95, 0.75, 0.45, 0.88, 0.33]
    
    print("Basic Color Coding Examples:")
    for score in test_scores:
        coded = color_code_score(score)
        print(f"  Score {score}: {coded}")
    
    # Test percentage scores
    print("\nPercentage Score Examples:")
    percentage_scores = [95, 75, 45, 88, 33]
    for score in percentage_scores:
        coded = color_code_score(score, score_range="percentage")
        print(f"  Score {score}%: {coded}")
    
    # Test advanced features
    print("\nAdvanced Features:")
    score = 0.78
    advanced_coded = color_code_score_advanced(
        score, 
        show_percentage=True, 
        show_bar=True,
        bar_length=15
    )
    print(f"  Advanced score: {advanced_coded}")
    
    # Test batch processing
    print("\nBatch Processing:")
    batch_results = batch_color_code([0.9, 0.65, 0.4, 0.85])
    for i, result in enumerate(batch_results):
        print(f"  Score {i+1}: {result}")
    
    # Show legend
    print(f"\nLegend: {color_legend()}")
```

### Rich Integration Implementation Example
Integration with Rich library for terminal UI:

```python
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn
from rich.panel import Panel
from rich.text import Text
from typing import Union, List, Dict
import math

class ScoreVisualizer:
    """
    Visualizer for color-coded scores using Rich library.
    """
    
    def __init__(self):
        self.console = Console()
    
    def display_score_panel(self, 
                          score: Union[int, float], 
                          title: str = "Score",
                          show_details: bool = True) -> None:
        """
        Display a score in a rich panel with color coding.
        """
        # Color code the score
        coded_score = color_code_score(score)
        
        if show_details:
            # Add additional details based on score
            if score >= 0.8:
                description = "Excellent quality"
                detail_color = "green"
            elif score >= 0.6:
                description = "Good quality"
                detail_color = "yellow"
            else:
                description = "Needs improvement"
                detail_color = "red"
            
            content = Text()
            content.append(f"{title}: ", style="bold")
            content.append(f"{coded_score}\n")
            content.append(f"{description}", style=f"italic {detail_color}")
            
            panel = Panel(
                content,
                title=title,
                border_style=detail_color
            )
        else:
            panel = Panel(coded_score, title=title)
        
        self.console.print(panel)
    
    def display_score_table(self, 
                          scores: Dict[str, Union[int, float]],
                          title: str = "Scores Overview") -> None:
        """
        Display multiple scores in a table format.
        """
        table = Table(title=title)
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Score", style="magenta")
        table.add_column("Status", style="green")
        
        for metric, score in scores.items():
            coded_score = color_code_score(score)
            
            # Determine status text
            if score >= 0.8:
                status = "[green]Excellent[/green]"
            elif score >= 0.6:
                status = "[yellow]Good[/yellow]"
            else:
                status = "[red]Poor[/red]"
            
            table.add_row(metric, coded_score, status)
        
        self.console.print(table)
    
    def display_score_progress(self, 
                             scores: List[Union[int, float]],
                             labels: List[str] = None,
                             title: str = "Score Progress") -> None:
        """
        Display scores as a progress bar visualization.
        """
        if labels is None:
            labels = [f"Metric {i+1}" for i in range(len(scores))]
        
        # Create table for scores
        table = Table(title=title)
        table.add_column("Metric", style="cyan")
        table.add_column("Score", style="magenta", justify="right")
        table.add_column("Visualization")
        
        max_width = 30  # Maximum bar width
        
        for label, score in zip(labels, scores):
            coded_score = color_code_score(score)
            
            # Create progress bar
            clamped_score = max(0, min(1, score))
            filled_width = int(clamped_score * max_width)
            empty_width = max_width - filled_width
            
            # Color the bar segments
            bar_parts = []
            
            if filled_width > 0:
                # Determine color based on score
                if score >= 0.8:
                    bar_parts.append(f"[green]{'█' * filled_width}[/green]")
                elif score >= 0.6:
                    bar_parts.append(f"[yellow]{'█' * filled_width}[/yellow]")
                else:
                    bar_parts.append(f"[red]{'█' * filled_width}[/red]")
            
            if empty_width > 0:
                bar_parts.append(f"[dim]{'░' * empty_width}[/dim]")
            
            bar_visual = "".join(bar_parts)
            
            table.add_row(label, coded_score, bar_visual)
        
        self.console.print(table)
    
    def display_score_comparison(self,
                               before_score: Union[int, float],
                               after_score: Union[int, float],
                               title: str = "Score Comparison") -> None:
        """
        Display before/after score comparison.
        """
        before_coded = color_code_score(before_score)
        after_coded = color_code_score(after_score)
        
        # Calculate improvement
        improvement = after_score - before_score
        improvement_coded = color_code_score(
            abs(improvement), 
            thresholds=(0.05, 0.1)  # Different thresholds for improvement
        )
        
        # Determine improvement direction
        if improvement > 0:
            direction = f"[green]↑ +{improvement_coded}[/green]"
        elif improvement < 0:
            direction = f"[red]↓ -{improvement_coded}[/red]"
        else:
            direction = "[blue]→ 0[/blue]"
        
        comparison_text = Text()
        comparison_text.append("Before: ", style="bold")
        comparison_text.append(f"{before_coded}  ")
        comparison_text.append("After: ", style="bold")
        comparison_text.append(f"{after_coded}  ")
        comparison_text.append(direction)
        
        panel = Panel(
            comparison_text,
            title=title,
            border_style="blue"
        )
        
        self.console.print(panel)

# Example usage with Rich integration
if __name__ == "__main__":
    # Initialize visualizer
    visualizer = ScoreVisualizer()
    
    # Example 1: Single score panel
    print("Example 1: Single Score Panel")
    visualizer.display_score_panel(0.85, "Alignment Score")
    
    # Example 2: Score table
    print("\nExample 2: Score Table")
    scores_dict = {
        "Alignment": 0.87,
        "Preservation": 0.92,
        "Quality": 0.78,
        "Speed": 0.95
    }
    visualizer.display_score_table(scores_dict, "Editing Results")
    
    # Example 3: Score progress visualization
    print("\nExample 3: Score Progress Visualization")
    progress_scores = [0.92, 0.78, 0.65, 0.88, 0.45]
    progress_labels = ["Sky", "Building", "Trees", "Water", "Ground"]
    visualizer.display_score_progress(
        progress_scores, 
        progress_labels, 
        "Entity Scores"
    )
    
    # Example 4: Score comparison
    print("\nExample 4: Score Comparison")
    visualizer.display_score_comparison(0.65, 0.87, "Quality Improvement")
    
    # Show legend
    print(f"\nLegend: {color_legend()}")
```