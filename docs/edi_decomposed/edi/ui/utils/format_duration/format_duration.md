# format_duration()

[Back to UI Utils](../ui_utils.md)

## Related User Story
"As a user, I want to see clear time estimates for long operations." (from PRD - implied by user experience requirements)

## Function Signature
`format_duration(seconds) -> "1m 23s"`

## Parameters
- `seconds` - A numeric value representing duration in seconds

## Returns
- `str` - A human-readable string representation of the duration (e.g., "1m 23s", "2h 5m", "45s")

## Step-by-step Logic
1. Take the input value in seconds
2. Calculate the number of complete hours, minutes, and remaining seconds
3. Format the values into a human-readable string:
   - If more than 1 hour: format as "Xh Ym" or "Xh Ym Zs"
   - If more than 1 minute: format as "Xm Ys"
   - If less than 1 minute: format as "Xs"
4. Use appropriate abbreviations (h for hours, m for minutes, s for seconds)
5. Omit zero-value components to keep the output concise
6. Return the formatted duration string

## Formatting Rules
- Shows hours only when duration is 1 hour or more
- Shows minutes when duration is 1 minute or more (in addition to hours if applicable)
- Shows seconds unless the duration is exactly an hour or multiple thereof
- Uses pluralization properly (1s vs 2s, 1m vs 2m, etc.)

## Edge Cases
- Handles zero seconds (returns "0s")
- Handles fractional seconds by rounding down to whole seconds
- Handles very large durations efficiently
- Maintains readability for all duration ranges

## Input/Output Data Structures
### Input
- seconds: Numeric value representing time in seconds (integer or float)

### Output
- String representation of duration in format like "2h 15m 30s", "45m 12s", "32s", etc.