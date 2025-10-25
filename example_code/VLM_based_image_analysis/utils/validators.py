"""
Data validation utilities for ensuring structured output quality.
"""

from typing import Dict, List, Any, Union
import json


def validate_entity_structure(entity_dict: Dict[str, Any]) -> bool:
    """
    Validate that an entity dictionary has required fields and correct types.
    
    Args:
        entity_dict (Dict[str, Any]): Entity dictionary to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not isinstance(entity_dict, dict):
        return False
    
    # Check required top-level fields
    required_fields = ['name', 'category', 'position', 'colors', 'description']
    for field in required_fields:
        if field not in entity_dict:
            print(f"[WARN] Missing required field '{field}' in entity")
            return False
    
    # Validate name and category are strings
    if not isinstance(entity_dict['name'], str) or not entity_dict['name']:
        print("[WARN] Entity name must be a non-empty string")
        return False
    
    if not isinstance(entity_dict['category'], str):
        print("[WARN] Entity category must be a string")
        return False
    
    # Validate position
    if not validate_position_data(entity_dict.get('position')):
        return False
    
    # Validate colors
    if not validate_color_data(entity_dict.get('colors')):
        return False
    
    # Validate description is string
    if not isinstance(entity_dict['description'], str):
        print("[WARN] Entity description must be a string")
        return False
    
    # Validate optional fields if present
    if 'children' in entity_dict and not isinstance(entity_dict['children'], list):
        print("[WARN] Entity children must be a list")
        return False
    
    if 'attributes' in entity_dict and not isinstance(entity_dict['attributes'], dict):
        print("[WARN] Entity attributes must be a dictionary")
        return False
    
    if 'quantity' in entity_dict and entity_dict['quantity'] is not None:
        if not isinstance(entity_dict['quantity'], int) or entity_dict['quantity'] < 0:
            print("[WARN] Entity quantity must be a non-negative integer or None")
            return False
    
    # Validate children recursively
    for child in entity_dict.get('children', []):
        if isinstance(child, dict) and not validate_entity_structure(child):
            return False
    
    return True


def validate_position_data(position_dict: Union[Dict[str, Any], str]) -> bool:
    """
    Validate position data has required fields.
    
    Args:
        position_dict (Union[Dict[str, Any], str]): Position data to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    # If position is a string, that's acceptable
    if isinstance(position_dict, str):
        return True
    
    # If position is a dict, validate its structure
    if not isinstance(position_dict, dict):
        print("[WARN] Position must be a string or dictionary")
        return False
    
    # Check required fields in position dict
    if 'description' not in position_dict:
        print("[WARN] Position must have a 'description' field")
        return False
    
    if not isinstance(position_dict['description'], str):
        print("[WARN] Position description must be a string")
        return False
    
    # Optional fields should have correct types if present
    if 'depth_layer' in position_dict:
        if position_dict['depth_layer'] not in ['foreground', 'midground', 'background']:
            print("[WARN] Position depth_layer must be 'foreground', 'midground', or 'background'")
            return False
    
    if 'x_percent' in position_dict:
        if position_dict['x_percent'] is not None:
            if not isinstance(position_dict['x_percent'], (int, float)):
                print("[WARN] Position x_percent must be a number or None")
                return False
            if not (0 <= position_dict['x_percent'] <= 100):
                print("[WARN] Position x_percent must be between 0 and 100")
                return False
    
    if 'y_percent' in position_dict:
        if position_dict['y_percent'] is not None:
            if not isinstance(position_dict['y_percent'], (int, float)):
                print("[WARN] Position y_percent must be a number or None")
                return False
            if not (0 <= position_dict['y_percent'] <= 100):
                print("[WARN] Position y_percent must be between 0 and 100")
                return False
    
    return True


def validate_color_data(colors_list: List[Dict[str, Any]]) -> bool:
    """
    Validate that colors list has proper structure.
    
    Args:
        colors_list (List[Dict[str, Any]]): Colors list to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not isinstance(colors_list, list):
        print("[WARN] Colors must be a list")
        return False
    
    for color in colors_list:
        if not isinstance(color, dict):
            print("[WARN] Each color must be a dictionary")
            return False
        
        # Check for hex field (required)
        if 'hex' not in color:
            print("[WARN] Each color must have a 'hex' field")
            return False
        
        if not isinstance(color['hex'], str) or not color['hex'].startswith('#'):
            print("[WARN] Color hex must be a string starting with '#'")
            return False
        
        # Validate hex format (#XXXXXX or #XXX)
        hex_val = color['hex'][1:]  # Remove #
        if not (len(hex_val) == 3 or len(hex_val) == 6) or not all(c in '0123456789ABCDEFabcdef' for c in hex_val):
            print(f"[WARN] Invalid hex color format: {color['hex']}")
            return False
    
    return True


def validate_scene_structure(scene_dict: Dict[str, Any]) -> bool:
    """
    Validate complete scene structure.
    
    Args:
        scene_dict (Dict[str, Any]): Scene dictionary to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not isinstance(scene_dict, dict):
        print("[WARN] Scene must be a dictionary")
        return False
    
    # Check required fields
    required_fields = ['image_id', 'overall_description', 'scene_type', 'entities']
    for field in required_fields:
        if field not in scene_dict:
            print(f"[WARN] Missing required field '{field}' in scene")
            return False
    
    # Validate image_id and scene_type are strings
    if not isinstance(scene_dict['image_id'], str) or not scene_dict['image_id']:
        print("[WARN] Scene image_id must be a non-empty string")
        return False
    
    if not isinstance(scene_dict['scene_type'], str):
        print("[WARN] Scene scene_type must be a string")
        return False
    
    # Validate overall_description is string
    if not isinstance(scene_dict['overall_description'], str):
        print("[WARN] Scene overall_description must be a string")
        return False
    
    # Validate entities is a list
    if not isinstance(scene_dict['entities'], list):
        print("[WARN] Scene entities must be a list")
        return False
    
    # Validate each entity
    for entity in scene_dict['entities']:
        if isinstance(entity, dict) and not validate_entity_structure(entity):
            return False
    
    return True


def validate_json_structure(json_str: str) -> bool:
    """
    Validate that a string contains valid JSON.
    
    Args:
        json_str (str): String to validate as JSON
        
    Returns:
        bool: True if valid JSON, False otherwise
    """
    try:
        json.loads(json_str)
        return True
    except json.JSONDecodeError as e:
        print(f"[WARN] Invalid JSON: {e}")
        return False


def validate_hex_color(hex_color: str) -> bool:
    """
    Validate that a string is a valid hex color.
    
    Args:
        hex_color (str): Hex color string to validate
        
    Returns:
        bool: True if valid hex color, False otherwise
    """
    if not isinstance(hex_color, str):
        return False
    
    if not hex_color.startswith('#'):
        return False
    
    hex_val = hex_color[1:]
    return (len(hex_val) == 3 or len(hex_val) == 6) and all(c in '0123456789ABCDEFabcdef' for c in hex_val)


def sanitize_entity_name(name: str) -> str:
    """
    Sanitize an entity name to ensure it's valid.
    
    Args:
        name (str): Entity name to sanitize
        
    Returns:
        str: Sanitized entity name
    """
    if not isinstance(name, str):
        return "Unknown"
    
    # Remove problematic characters
    sanitized = ''.join(c for c in name if c.isprintable() and c not in '<>:"/\\|?*')
    
    # Trim whitespace
    sanitized = sanitized.strip()
    
    # Ensure non-empty
    if not sanitized:
        return "Unknown"
    
    return sanitized


def sanitize_scene_description(description: str) -> str:
    """
    Sanitize a scene description to ensure it's valid.
    
    Args:
        description (str): Scene description to sanitize
        
    Returns:
        str: Sanitized scene description
    """
    if not isinstance(description, str):
        return "No description available"
    
    # Trim whitespace
    sanitized = description.strip()
    
    # Ensure non-empty
    if not sanitized:
        return "No description available"
    
    return sanitized