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