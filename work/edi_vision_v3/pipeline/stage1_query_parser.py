"""Stage 1 Helper: Query Parser

Parses user queries to extract color and object components.

This enables dual-path detection:
- Semantic-only: "vehicles" → No color, use YOLO-World directly
- Color+object: "red vehicles" → Extract color + object, use dual-path

Usage:
    from pipeline.stage1_query_parser import parse_query

    parsed = parse_query("red vehicles")
    # ParsedQuery(color="red", object_type="vehicles", original="red vehicles")

    parsed = parse_query("vehicles")
    # ParsedQuery(color=None, object_type="vehicles", original="vehicles")
"""

import re
import logging
from dataclasses import dataclass
from typing import Optional, List


# Color vocabulary (expandable)
COLORS = [
    "red", "blue", "green", "yellow", "orange", "purple", "brown",
    "black", "white", "gray", "grey", "pink", "cyan", "magenta",
    "turquoise", "violet", "indigo", "maroon", "navy", "olive",
    "teal", "silver", "gold", "beige", "tan", "cream"
]

# Common object categories (helps with query cleaning)
OBJECT_CATEGORIES = [
    "vehicle", "vehicles", "car", "cars", "auto-rickshaw", "auto-rickshaws",
    "rickshaw", "rickshaws", "taxi", "taxis", "bus", "buses", "truck", "trucks",
    "building", "buildings", "house", "houses", "roof", "roofs",
    "person", "people", "pedestrian", "pedestrians",
    "tree", "trees", "plant", "plants", "bird", "birds", "animal", "animals",
    "sky", "cloud", "clouds", "water", "road", "street", "bridge"
]


@dataclass
class ParsedQuery:
    """Parsed query components.

    Attributes:
        color: Detected color (None if no color in query)
        object_type: Object to detect (cleaned query)
        original_query: Original user query
        confidence: Confidence that parsing is correct (0-1)
    """
    color: Optional[str]
    object_type: str
    original_query: str
    confidence: float = 1.0


def parse_query(user_query: str) -> ParsedQuery:
    """
    Parse user query to extract color and object components.

    Strategy:
    1. Check if query contains a color word
    2. If yes: Extract color, remove from query to get object
    3. If no: Use entire query as object
    4. Clean up whitespace

    Args:
        user_query: Natural language query from user

    Returns:
        ParsedQuery with color and object components

    Examples:
        >>> parse_query("red vehicles")
        ParsedQuery(color="red", object_type="vehicles", original="red vehicles")

        >>> parse_query("vehicles")
        ParsedQuery(color=None, object_type="vehicles", original="vehicles")

        >>> parse_query("brown roofs")
        ParsedQuery(color="brown", object_type="roofs", original="brown roofs")

        >>> parse_query("yellow auto-rickshaws")
        ParsedQuery(color="yellow", object_type="auto-rickshaws", original="...")
    """
    query_lower = user_query.lower().strip()

    logging.debug(f"Parsing query: '{user_query}'")

    # Check for color
    detected_color = None
    object_query = query_lower

    for color in COLORS:
        # Use word boundaries to avoid partial matches
        pattern = r'\b' + re.escape(color) + r'\b'
        if re.search(pattern, query_lower):
            detected_color = color
            # Remove color from query to get object
            object_query = re.sub(pattern, '', query_lower).strip()
            logging.debug(f"  Detected color: {color}")
            logging.debug(f"  Object part: '{object_query}'")
            break

    # Clean up object query
    # Remove extra whitespace
    object_query = ' '.join(object_query.split())

    # If object query is empty after color removal, use original
    if not object_query:
        logging.warning(f"Empty object query after removing color, using original")
        object_query = query_lower

    # Calculate confidence
    confidence = 1.0
    if detected_color and not object_query:
        confidence = 0.5  # Color only, no object
    elif not detected_color and not any(obj in query_lower for obj in OBJECT_CATEGORIES):
        confidence = 0.7  # Unknown object category

    result = ParsedQuery(
        color=detected_color,
        object_type=object_query,
        original_query=user_query,
        confidence=confidence
    )

    logging.debug(f"  Parsed result: color={result.color}, object='{result.object_type}'")

    return result


def extract_object_synonyms(object_query: str) -> List[str]:
    """
    Generate synonym queries for better detection.

    For complex queries like "auto-rickshaws", provide simpler alternatives.

    Args:
        object_query: Object part of query

    Returns:
        List of query variations (original + synonyms)

    Examples:
        >>> extract_object_synonyms("auto-rickshaws")
        ["auto-rickshaws", "auto-rickshaw", "rickshaws", "rickshaw", "vehicle"]

        >>> extract_object_synonyms("vehicles")
        ["vehicles", "vehicle", "car"]
    """
    synonyms = [object_query]

    # Auto-rickshaw variations
    if "auto-rickshaw" in object_query or "autorickshaw" in object_query:
        synonyms.extend(["auto-rickshaw", "rickshaw", "vehicle", "three-wheeler"])

    # Vehicle variations
    elif "vehicle" in object_query:
        synonyms.extend(["vehicle", "car", "automobile"])

    # Building variations
    elif "building" in object_query or "house" in object_query:
        synonyms.extend(["building", "house", "structure"])

    # Remove duplicates while preserving order
    seen = set()
    unique_synonyms = []
    for syn in synonyms:
        if syn not in seen:
            seen.add(syn)
            unique_synonyms.append(syn)

    return unique_synonyms


def is_semantic_only(parsed: ParsedQuery) -> bool:
    """
    Check if query is semantic-only (no color).

    Args:
        parsed: Parsed query

    Returns:
        True if semantic-only, False if color+object

    Examples:
        >>> is_semantic_only(parse_query("vehicles"))
        True

        >>> is_semantic_only(parse_query("red vehicles"))
        False
    """
    return parsed.color is None
