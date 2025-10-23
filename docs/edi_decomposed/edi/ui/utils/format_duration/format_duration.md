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

## See Docs

### Python Implementation Example
Implementation of the format_duration function:

```python
import math
from typing import Union, Optional

def format_duration(seconds: Union[int, float], 
                   precision: str = "auto",
                   max_components: int = 3,
                   show_zero: bool = False) -> str:
    """
    Format a duration in seconds into a human-readable string.
    
    Args:
        seconds: Numeric value representing time in seconds (integer or float)
        precision: Precision level - "auto", "seconds", "minutes", "hours"
        max_components: Maximum number of time components to show (e.g., 2 = "2h 15m")
        show_zero: Whether to show zero components when they're significant
    
    Returns:
        String representation of duration in format like "2h 15m 30s", "45m 12s", "32s", etc.
    """
    # Validate input
    if not isinstance(seconds, (int, float)):
        raise TypeError(f"Seconds must be numeric, got {type(seconds)}")
    
    if seconds < 0:
        raise ValueError(f"Seconds must be non-negative, got {seconds}")
    
    # Handle special case of zero seconds
    if seconds == 0:
        return "0s"
    
    # Handle fractional seconds by rounding down to whole seconds
    total_seconds = int(math.floor(seconds))
    
    # Calculate hours, minutes, and seconds
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    
    # Build components based on precision and significance
    components = []
    
    # Add hours if significant or required
    if hours > 0 or (show_zero and precision in ["auto", "hours"]):
        components.append(f"{hours}h")
    
    # Add minutes if significant or required
    if minutes > 0 or (show_zero and precision in ["auto", "minutes", "seconds"] and hours > 0):
        # Only add minutes if we're within component limit
        if len(components) < max_components:
            components.append(f"{minutes}m")
    
    # Add seconds if significant or required
    if secs > 0 or show_zero or precision == "seconds":
        # Only add seconds if we're within component limit
        if len(components) < max_components:
            components.append(f"{secs}s")
    
    # Handle case where no components were added (e.g., small fractions)
    if not components:
        return "0s"
    
    # Join components with spaces
    return " ".join(components)

def format_duration_precise(seconds: Union[int, float],
                           include_ms: bool = False,
                           compact: bool = False) -> str:
    """
    Format duration with more precise control over formatting.
    
    Args:
        seconds: Time in seconds
        include_ms: Whether to include milliseconds for sub-second precision
        compact: Whether to use compact notation (single letters)
    
    Returns:
        Formatted duration string
    """
    if not isinstance(seconds, (int, float)):
        raise TypeError(f"Seconds must be numeric, got {type(seconds)}")
    
    if seconds < 0:
        raise ValueError(f"Seconds must be non-negative, got {seconds}")
    
    # Handle zero
    if seconds == 0:
        return "0s" if not compact else "0"
    
    # Handle milliseconds if requested
    if include_ms and seconds < 1:
        ms = int(seconds * 1000)
        if ms > 0:
            return f"{ms}ms" if not compact else f"{ms}m"
        else:
            return "<1ms" if not compact else "<1m"
    
    # Calculate time units
    total_seconds = seconds
    days = int(total_seconds // 86400)
    total_seconds %= 86400
    hours = int(total_seconds // 3600)
    total_seconds %= 3600
    minutes = int(total_seconds // 60)
    secs = total_seconds % 60
    
    # Build components
    components = []
    
    if days > 0:
        components.append(f"{days}d" if not compact else f"{days}d")
    
    if hours > 0 and len(components) < 2:  # Limit to 2 components for readability
        components.append(f"{hours}h" if not compact else f"{hours}h")
    
    if minutes > 0 and len(components) < 2:
        components.append(f"{minutes}m" if not compact else f"{minutes}m")
    
    # For seconds, include decimal if requested and it's a small duration
    if (secs > 0 or len(components) == 0) and len(components) < 2:
        if include_ms and secs < 60 and len(components) == 0:
            # Show seconds with decimals for sub-minute durations
            components.append(f"{secs:.1f}s" if not compact else f"{secs:.1f}s")
        else:
            # Show whole seconds
            components.append(f"{int(secs)}s" if not compact else f"{int(secs)}s")
    
    return " ".join(components) if components else "0s"

def format_duration_alternative(seconds: Union[int, float],
                               style: str = "standard") -> str:
    """
    Format duration using alternative styles.
    
    Args:
        seconds: Time in seconds
        style: Formatting style - "standard", "verbose", "compact", "digital"
    
    Returns:
        Formatted duration string
    """
    if not isinstance(seconds, (int, float)):
        raise TypeError(f"Seconds must be numeric, got {type(seconds)}")
    
    if seconds < 0:
        raise ValueError(f"Seconds must be non-negative, got {seconds}")
    
    if seconds == 0:
        if style == "verbose":
            return "zero seconds"
        elif style == "compact":
            return "0"
        elif style == "digital":
            return "00:00"
        else:
            return "0s"
    
    total_seconds = int(math.floor(seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    
    if style == "verbose":
        # Verbose format: "2 hours, 15 minutes, 30 seconds"
        parts = []
        if hours > 0:
            parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
        if minutes > 0:
            parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
        if secs > 0 or len(parts) == 0:
            parts.append(f"{secs} second{'s' if secs != 1 else ''}")
        
        if len(parts) == 1:
            return parts[0]
        elif len(parts) == 2:
            return f"{parts[0]} and {parts[1]}"
        else:
            return ", ".join(parts[:-1]) + f", and {parts[-1]}"
    
    elif style == "compact":
        # Compact format: "2h15m30s"
        result = ""
        if hours > 0:
            result += f"{hours}h"
        if minutes > 0:
            result += f"{minutes}m"
        if secs > 0 or result == "":
            result += f"{secs}s"
        return result
    
    elif style == "digital":
        # Digital format: "02:15:30" or "15:30"
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"
    
    else:
        # Standard format (same as main function)
        components = []
        if hours > 0:
            components.append(f"{hours}h")
        if minutes > 0:
            components.append(f"{minutes}m")
        if secs > 0 or len(components) == 0:
            components.append(f"{secs}s")
        return " ".join(components)

def parse_duration(duration_str: str) -> float:
    """
    Parse a duration string back into seconds.
    
    Args:
        duration_str: Duration string like "2h 15m 30s"
    
    Returns:
        Total seconds as float
    """
    if not isinstance(duration_str, str):
        raise TypeError(f"Duration must be a string, got {type(duration_str)}")
    
    # Handle digital format first
    if ":" in duration_str:
        parts = duration_str.split(":")
        if len(parts) == 2:  # MM:SS
            minutes, seconds = map(int, parts)
            return minutes * 60 + seconds
        elif len(parts) == 3:  # HH:MM:SS
            hours, minutes, seconds = map(int, parts)
            return hours * 3600 + minutes * 60 + seconds
    
    # Handle standard format
    total_seconds = 0.0
    parts = duration_str.replace(",", " ").split()
    
    for part in parts:
        part = part.strip().lower()
        if part.endswith('d'):
            total_seconds += int(part[:-1]) * 86400
        elif part.endswith('h'):
            total_seconds += int(part[:-1]) * 3600
        elif part.endswith('m'):
            total_seconds += int(part[:-1]) * 60
        elif part.endswith('s'):
            total_seconds += int(part[:-1])
        elif part.endswith('ms'):
            total_seconds += int(part[:-2]) / 1000.0
    
    return total_seconds

# Example usage
if __name__ == "__main__":
    # Test basic formatting
    test_durations = [0, 30, 90, 150, 3661, 7265, 90061, 0.5, 1.75]
    
    print("Basic Duration Formatting:")
    for duration in test_durations:
        formatted = format_duration(duration)
        print(f"  {duration:>8} seconds → {formatted}")
    
    # Test precise formatting
    print("\nPrecise Formatting (with milliseconds):")
    precise_durations = [0.123, 0.5, 1.75, 30.25, 90.99]
    for duration in precise_durations:
        formatted = format_duration_precise(duration, include_ms=True)
        print(f"  {duration:>6} seconds → {formatted}")
    
    # Test alternative styles
    print("\nAlternative Styles:")
    sample_duration = 3661  # 1 hour, 1 minute, 1 second
    styles = ["standard", "verbose", "compact", "digital"]
    for style in styles:
        formatted = format_duration_alternative(sample_duration, style)
        print(f"  {style:>8}: {formatted}")
    
    # Test parsing
    print("\nParsing Duration Strings:")
    test_strings = ["1h 2m 3s", "45m 30s", "120s", "02:30", "01:02:03"]
    for duration_str in test_strings:
        parsed = parse_duration(duration_str)
        print(f"  '{duration_str}' → {parsed} seconds")
```

### Advanced Duration Formatting Implementation
Enhanced formatting with localization and custom formatting:

```python
import math
from typing import Union, Dict, Optional
from datetime import timedelta
import locale

class DurationFormatter:
    """
    Advanced duration formatter with localization and custom formatting options.
    """
    
    def __init__(self, 
                 language: str = "en",
                 custom_units: Optional[Dict[str, str]] = None):
        self.language = language
        self.custom_units = custom_units or self._default_units()
        
        # Unit translations
        self.translations = {
            "en": {
                "second": "second", "seconds": "seconds",
                "minute": "minute", "minutes": "minutes",
                "hour": "hour", "hours": "hours",
                "day": "day", "days": "days",
                "abbrev": {"s": "s", "m": "m", "h": "h", "d": "d"}
            },
            "es": {
                "second": "segundo", "seconds": "segundos",
                "minute": "minuto", "minutes": "minutos",
                "hour": "hora", "hours": "horas",
                "day": "día", "days": "días",
                "abbrev": {"s": "s", "m": "m", "h": "h", "d": "d"}
            },
            # Add more languages as needed
        }
    
    def _default_units(self) -> Dict[str, str]:
        """Default unit abbreviations."""
        return {
            "second": "s", "seconds": "s",
            "minute": "m", "minutes": "m",
            "hour": "h", "hours": "h",
            "day": "d", "days": "d"
        }
    
    def format(self, 
               seconds: Union[int, float],
               style: str = "short",
               max_units: int = 2,
               granularity: str = "auto") -> str:
        """
        Format duration with advanced options.
        
        Args:
            seconds: Time in seconds
            style: "short", "long", "abbreviated", "digital"
            max_units: Maximum number of time units to show
            granularity: "seconds", "minutes", "hours", "days", "auto"
        """
        if not isinstance(seconds, (int, float)):
            raise TypeError(f"Seconds must be numeric, got {type(seconds)}")
        
        if seconds < 0:
            raise ValueError(f"Seconds must be non-negative, got {seconds}")
        
        if seconds == 0:
            return self._format_zero(style)
        
        # Calculate time components
        total_seconds = int(math.floor(seconds))
        days = total_seconds // 86400
        hours = (total_seconds % 86400) // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        
        # Build components based on style and granularity
        components = []
        
        # Add components based on granularity
        if granularity == "auto" or granularity == "days":
            if days > 0:
                components.append((days, "day" if days == 1 else "days"))
        
        if granularity == "auto" or granularity in ["hours", "days"]:
            if hours > 0 or (components and style != "digital"):
                components.append((hours, "hour" if hours == 1 else "hours"))
        
        if granularity == "auto" or granularity in ["minutes", "hours", "days"]:
            if minutes > 0 or (components and style != "digital"):
                components.append((minutes, "minute" if minutes == 1 else "minutes"))
        
        if granularity == "auto" or granularity == "seconds":
            if secs > 0 or len(components) == 0:
                components.append((secs, "second" if secs == 1 else "seconds"))
        
        # Limit components
        components = components[:max_units]
        
        if not components:
            return self._format_zero(style)
        
        # Format based on style
        if style == "digital":
            return self._format_digital(components)
        elif style == "long":
            return self._format_long(components)
        elif style == "abbreviated":
            return self._format_abbreviated(components)
        else:  # short/default
            return self._format_short(components)
    
    def _format_zero(self, style: str) -> str:
        """Format zero duration."""
        if style == "long":
            return self.translations[self.language]["seconds"]
        elif style == "digital":
            return "00:00"
        else:
            return f"0{self.custom_units['second']}"
    
    def _format_short(self, components: list) -> str:
        """Format in short style (e.g., '1h 30m')."""
        abbrev = self.translations[self.language]["abbrev"]
        parts = []
        for value, unit_key in components:
            unit_abbr = abbrev.get(unit_key[0], unit_key[0])  # First letter abbreviation
            parts.append(f"{value}{unit_abbr}")
        return " ".join(parts)
    
    def _format_long(self, components: list) -> str:
        """Format in long style (e.g., '1 hour 30 minutes')."""
        trans = self.translations[self.language]
        parts = []
        for value, unit_key in components:
            translated_unit = trans[unit_key]
            parts.append(f"{value} {translated_unit}")
        
        if len(parts) == 1:
            return parts[0]
        elif len(parts) == 2:
            return f"{parts[0]} and {parts[1]}"
        else:
            return ", ".join(parts[:-1]) + f", and {parts[-1]}"
    
    def _format_abbreviated(self, components: list) -> str:
        """Format in abbreviated style (e.g., '1h30m')."""
        abbrev = self.translations[self.language]["abbrev"]
        parts = []
        for value, unit_key in components:
            unit_abbr = abbrev.get(unit_key[0], unit_key[0])
            parts.append(f"{value}{unit_abbr}")
        return "".join(parts)
    
    def _format_digital(self, components: list) -> str:
        """Format in digital style (e.g., '01:30:45')."""
        # For digital format, we need to reconstruct the time
        total_seconds = 0
        for value, unit in components:
            if "day" in unit:
                total_seconds += value * 86400
            elif "hour" in unit:
                total_seconds += value * 3600
            elif "minute" in unit:
                total_seconds += value * 60
            elif "second" in unit:
                total_seconds += value
        
        td = timedelta(seconds=total_seconds)
        hours, remainder = divmod(td.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        # Add days to hours
        hours += td.days * 24
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes:02d}:{seconds:02d}"

# Example usage with advanced formatter
if __name__ == "__main__":
    # Initialize formatter
    formatter = DurationFormatter()
    
    # Test durations
    test_durations = [0, 30, 90, 150, 3661, 90061, 86401]
    
    print("Advanced Duration Formatting:")
    print("=" * 50)
    
    for duration in test_durations:
        print(f"\nDuration: {duration} seconds")
        
        # Test different styles
        styles = ["short", "long", "abbreviated", "digital"]
        for style in styles:
            try:
                formatted = formatter.format(duration, style=style)
                print(f"  {style:>12}: {formatted}")
            except Exception as e:
                print(f"  {style:>12}: Error - {e}")
    
    # Test with different max units
    print(f"\nMax Units Test (3661 seconds):")
    for max_units in [1, 2, 3]:
        formatted = formatter.format(3661, max_units=max_units)
        print(f"  Max {max_units} units: {formatted}")
    
    # Test with different granularity
    print(f"\nGranularity Test (3661 seconds):")
    granularities = ["seconds", "minutes", "hours", "days", "auto"]
    for granularity in granularities:
        formatted = formatter.format(3661, granularity=granularity)
        print(f"  {granularity:>8}: {formatted}")
```