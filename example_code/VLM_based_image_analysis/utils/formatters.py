"""
Output formatting utilities for consistent structured data.
"""

import json
from typing import Dict, List, Any, Union
import re


def format_entity_as_json(entity_object: Dict[str, Any], indent: int = 2) -> str:
    """
    Convert entity object to JSON string.
    
    Args:
        entity_object (Dict[str, Any]): Entity dictionary to format
        indent (int): Indentation level for JSON formatting
        
    Returns:
        str: Formatted JSON string
    """
    try:
        return json.dumps(entity_object, indent=indent, ensure_ascii=False)
    except Exception as e:
        raise ValueError(f"Failed to format entity as JSON: {e}")


def format_scene_as_json(scene_object: Dict[str, Any], indent: int = 2) -> str:
    """
    Convert scene object to JSON string.
    
    Args:
        scene_object (Dict[str, Any]): Scene dictionary to format
        indent (int): Indentation level for JSON formatting
        
    Returns:
        str: Formatted JSON string
    """
    try:
        return json.dumps(scene_object, indent=indent, ensure_ascii=False)
    except Exception as e:
        raise ValueError(f"Failed to format scene as JSON: {e}")


def format_entities_as_markdown(entities_list: List[Dict[str, Any]]) -> str:
    """
    Format entities as readable markdown.
    
    Args:
        entities_list (List[Dict[str, Any]]): List of entity dictionaries
        
    Returns:
        str: Markdown formatted string
    """
    if not entities_list:
        return "# Entities\n\nNo entities found."
    
    markdown = "# Entities\n\n"
    
    for i, entity in enumerate(entities_list, 1):
        if not isinstance(entity, dict):
            continue
            
        name = entity.get('name', 'Unknown')
        category = entity.get('category', 'unknown')
        markdown += f"## {i}. {name} ({category})\n\n"
        
        # Position
        position = entity.get('position', {})
        if isinstance(position, dict):
            markdown += f"**Position**: {position.get('description', 'unknown position')}"
            if position.get('depth_layer'):
                markdown += f" [{position['depth_layer']}]"
            markdown += "\n\n"
        
        # Colors
        colors = entity.get('colors', [])
        if colors:
            color_items = []
            for color in colors:
                if isinstance(color, dict):
                    color_str = color.get('name', 'color')
                    if color.get('hex'):
                        color_str += f" ({color['hex']})"
                    color_items.append(color_str)
            if color_items:
                markdown += f"**Colors**: {', '.join(color_items)}\n\n"
        
        # Description
        description = entity.get('description', '')
        if description:
            # Truncate long descriptions
            if len(description) > 200:
                description = description[:197] + "..."
            markdown += f"**Description**: {description}\n\n"
        
        # Quantity
        quantity = entity.get('quantity')
        if quantity:
            markdown += f"**Quantity**: {quantity}\n\n"
        
        # Children
        children = entity.get('children', [])
        if children:
            markdown += f"**Children**: {len(children)} sub-entities\n\n"
            for child in children:
                if isinstance(child, dict):
                    child_name = child.get('name', 'Unknown')
                    child_category = child.get('category', 'unknown')
                    markdown += f"  - {child_name} ({child_category})\n"
            markdown += "\n"
        
        markdown += "---\n\n"
    
    return markdown


def clean_json_string(json_str: str) -> str:
    """
    Clean common JSON formatting issues.
    
    Args:
        json_str (str): JSON string to clean
        
    Returns:
        str: Cleaned JSON string
    """
    if not isinstance(json_str, str):
        return "{}"
    
    # Remove trailing commas before closing braces/brackets
    json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
    
    # Fix escaped quotes
    json_str = json_str.replace('\\"', '"')
    
    # Remove extra newlines and spaces
    json_str = re.sub(r'\s+', ' ', json_str)
    
    # Ensure proper JSON wrapping
    json_str = json_str.strip()
    
    # If it looks like it might be JSON but isn't wrapped, try to wrap it
    if json_str and not (json_str.startswith('{') or json_str.startswith('[')):
        # Try to find JSON-like content
        obj_match = re.search(r'\{.*\}', json_str)
        arr_match = re.search(r'\[.*\]', json_str)
        
        if obj_match:
            json_str = obj_match.group(0)
        elif arr_match:
            json_str = arr_match.group(0)
    
    return json_str


def format_scene_summary(scene_dict: Dict[str, Any]) -> str:
    """
    Create a human-readable summary of the scene.
    
    Args:
        scene_dict (Dict[str, Any]): Scene dictionary to summarize
        
    Returns:
        str: Human-readable summary
    """
    if not isinstance(scene_dict, dict):
        return "Invalid scene data"
    
    summary = "=" * 50 + "\n"
    summary += "SCENE SUMMARY\n"
    summary += "=" * 50 + "\n"
    summary += f"Type: {scene_dict.get('scene_type', 'unknown')}\n"
    summary += f"Description: {scene_dict.get('overall_description', 'No description')}\n"
    summary += f"Total Entities: {len(scene_dict.get('entities', []))}\n"
    summary += "=" * 50 + "\n\n"
    
    entities = scene_dict.get('entities', [])
    for i, entity in enumerate(entities[:10]):  # Limit to first 10 entities
        if isinstance(entity, dict):
            name = entity.get('name', 'Unknown')
            category = entity.get('category', 'unknown')
            summary += f"▸ {name} ({category})\n"
            
            position = entity.get('position', {})
            if isinstance(position, dict):
                summary += f"  Position: {position.get('description', 'unknown position')}"
                if position.get('depth_layer'):
                    summary += f" [{position['depth_layer']}]"
                summary += "\n"
            
            colors = entity.get('colors', [])
            if colors:
                color_items = []
                for color in colors[:3]:  # Limit to first 3 colors
                    if isinstance(color, dict):
                        color_name = color.get('name', 'color')
                        color_hex = color.get('hex', '')
                        if color_hex:
                            color_items.append(f"{color_name}:{color_hex}")
                        else:
                            color_items.append(color_name)
                if color_items:
                    summary += f"  Colors: {', '.join(color_items)}\n"
            
            description = entity.get('description', '')
            if description:
                # Truncate long descriptions
                if len(description) > 80:
                    description = description[:77] + "..."
                summary += f"  Description: {description}\n"
            
            quantity = entity.get('quantity')
            if quantity:
                summary += f"  Quantity: {quantity}\n"
            
            children = entity.get('children', [])
            if children:
                summary += f"  ↳ {len(children)} sub-entities:\n"
            
            summary += "\n"
    
    return summary


def format_entity_tree(entities: List[Dict[str, Any]], indent_level: int = 0) -> str:
    """
    Format entities as a hierarchical tree structure.
    
    Args:
        entities (List[Dict[str, Any]]): List of entity dictionaries
        indent_level (int): Current indentation level
        
    Returns:
        str: Hierarchical tree formatted string
    """
    if not entities:
        return ""
    
    tree = ""
    indent = "  " * indent_level
    
    for entity in entities:
        if not isinstance(entity, dict):
            continue
            
        name = entity.get('name', 'Unknown')
        category = entity.get('category', 'unknown')
        tree += f"{indent}• {name} ({category})\n"
        
        # Add position info
        position = entity.get('position', {})
        if isinstance(position, dict) and position.get('description'):
            tree += f"{indent}  Position: {position['description']}\n"
        
        # Add color info
        colors = entity.get('colors', [])
        if colors:
            color_strs = []
            for color in colors[:2]:  # Limit to 2 colors
                if isinstance(color, dict):
                    color_name = color.get('name', 'color')
                    color_hex = color.get('hex', '')
                    if color_hex:
                        color_strs.append(f"{color_name}({color_hex})")
                    else:
                        color_strs.append(color_name)
            if color_strs:
                tree += f"{indent}  Colors: {', '.join(color_strs)}\n"
        
        # Process children recursively
        children = entity.get('children', [])
        if children:
            tree += format_entity_tree(children, indent_level + 1)
    
    return tree


def truncate_long_fields(obj: Union[Dict, List, Any], max_length: int = 500) -> Union[Dict, List, Any]:
    """
    Recursively truncate long string fields to prevent oversized outputs.
    
    Args:
        obj (Union[Dict, List, Any]): Object to process
        max_length (int): Maximum length for string fields
        
    Returns:
        Union[Dict, List, Any]: Processed object with truncated strings
    """
    if isinstance(obj, dict):
        result = {}
        for key, value in obj.items():
            if isinstance(value, str) and len(value) > max_length:
                result[key] = value[:max_length] + "..."
            else:
                result[key] = truncate_long_fields(value, max_length)
        return result
    elif isinstance(obj, list):
        return [truncate_long_fields(item, max_length) for item in obj]
    else:
        return obj


def prettify_json_output(json_obj: Union[Dict, List]) -> str:
    """
    Prettify JSON output for better readability.
    
    Args:
        json_obj (Union[Dict, List]): JSON object to prettify
        
    Returns:
        str: Pretty formatted JSON string
    """
    try:
        # Truncate long fields first
        truncated_obj = truncate_long_fields(json_obj, max_length=300)
        return json.dumps(truncated_obj, indent=2, ensure_ascii=False)
    except Exception as e:
        return f"Error prettifying JSON: {e}"