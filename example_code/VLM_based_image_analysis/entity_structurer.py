"""
DSPy modules specifically for structuring VLM output into hierarchical entities.
"""

import dspy
import json
import logging
from typing import List, Dict, Any


# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class ExtractSceneOverview(dspy.Signature):
    """
    Extract overall scene information from VLM description.
    """
    vlm_description = dspy.InputField(desc="Raw description from vision model")
    
    overall_description = dspy.OutputField(desc="High-level scene description")
    scene_type = dspy.OutputField(desc="Type of scene (landscape, urban, portrait, etc)")


class IdentifyMainEntities(dspy.Signature):
    """
    Identify main entities in the image scene from VLM description.
    """
    vlm_description = dspy.InputField(desc="Raw description from vision model")
    
    entity_list = dspy.OutputField(
        desc="JSON array of main entities with name and category. "
             "Example: [{\"name\": \"Mountain\", \"category\": \"natural\"}, "
             "{\"name\": \"Building\", \"category\": \"architecture\"}]"
    )


class ExtractEntityDetails(dspy.Signature):
    """
    Extract detailed information for a specific entity.
    """
    vlm_description = dspy.InputField(desc="Raw description from vision model")
    entity_name = dspy.InputField(desc="Name of the entity to extract details for")
    entity_category = dspy.InputField(desc="Category of the entity")
    
    entity_details = dspy.OutputField(
        desc="Complete entity in JSON format with name, category, position (with description, x_percent, y_percent, depth_layer), "
             "colors (array of objects with hex and name), description, quantity, dimensions, children (array), and attributes. "
             "Example: {\"name\":\"Mountain\",\"category\":\"natural\",\"position\":{\"description\":\"far background\",\"x_percent\":50,\"y_percent\":20,\"depth_layer\":\"background\"},"
             "\"colors\":[{\"hex\":\"#FFFFFF\",\"name\":\"white\"}],\"description\":\"Snow-capped peak\",\"quantity\":null,\"dimensions\":null,\"children\":[],\"attributes\":{}}"
    )


class OrganizeEntityHierarchy(dspy.Signature):
    """
    Organize entities into a hierarchical structure.
    """
    entities_data = dspy.InputField(desc="JSON array of entity data")
    
    organized_entities = dspy.OutputField(
        desc="JSON array of entities organized hierarchically with parent-child relationships. "
             "Entities in background/midground/foreground layers should be grouped appropriately."
    )


class RefineEntityStructure(dspy.Signature):
    """
    Refine entity structure for better organization and completeness.
    """
    raw_entities = dspy.InputField(desc="JSON array of raw entity data")
    scene_context = dspy.InputField(desc="Overall scene context for better enhancement")
    
    refined_entities = dspy.OutputField(
        desc="JSON array of refined entities with improved hierarchy, complete hex codes, "
             "better organized depth layers, and accessibility-focused descriptions"
    )


class ValidateEntityStructure(dspy.Signature):
    """
    Validate and fix entity structure to ensure consistency.
    """
    entities_data = dspy.InputField(desc="JSON array of entity data to validate")
    
    validated_entities = dspy.OutputField(
        desc="JSON array of validated and fixed entities. "
             "Ensures all required fields are present and correctly formatted."
    )


def create_scene_overview_module():
    """
    Create a DSPy module for extracting scene overview.
    
    Returns:
        dspy.Predict: Module for scene overview extraction
    """
    return dspy.ChainOfThought(ExtractSceneOverview)


def identify_main_entities_module():
    """
    Create a DSPy module for identifying main entities.
    
    Returns:
        dspy.Predict: Module for entity identification
    """
    return dspy.ChainOfThought(IdentifyMainEntities)


def extract_entity_details_module():
    """
    Create a DSPy module for extracting entity details.
    
    Returns:
        dspy.Predict: Module for entity detail extraction
    """
    return dspy.ChainOfThought(ExtractEntityDetails)


def organize_entity_hierarchy_module():
    """
    Create a DSPy module for organizing entity hierarchy.
    
    Returns:
        dspy.Predict: Module for entity hierarchy organization
    """
    return dspy.ChainOfThought(OrganizeEntityHierarchy)


def refine_entity_structure_module():
    """
    Create a DSPy module for refining entity structure.
    
    Returns:
        dspy.Predict: Module for entity structure refinement
    """
    return dspy.ChainOfThought(RefineEntityStructure)


def validate_entity_structure_module():
    """
    Create a DSPy module for validating entity structure.
    
    Returns:
        dspy.Predict: Module for entity structure validation
    """
    return dspy.ChainOfThought(ValidateEntityStructure)


class EntityStructuringPipeline(dspy.Module):
    """
    Complete pipeline for structuring VLM output into hierarchical entities.
    """
    
    def __init__(self):
        super().__init__()
        logger.debug("Initializing EntityStructuringPipeline")
        # Initialize modules without configuring them yet
        self.scene_overview = create_scene_overview_module()
        self.entity_identifier = identify_main_entities_module()
        self.entity_detail_extractor = extract_entity_details_module()
        self.hierarchy_organizer = organize_entity_hierarchy_module()
        self.structure_refiner = refine_entity_structure_module()
        self.structure_validator = validate_entity_structure_module()
        logger.debug("EntityStructuringPipeline initialized successfully")
    
    def forward(self, vlm_description: str) -> Dict[str, Any]:
        """
        Process VLM description into structured entity hierarchy.
        
        Args:
            vlm_description (str): Raw description from vision model
            
        Returns:
            Dict[str, Any]: Structured scene with entities
        """
        logger.debug(f"Starting EntityStructuringPipeline forward pass with VLM description length: {len(vlm_description) if vlm_description else 0}")
        
        # Step 1: Extract scene overview
        try:
            logger.debug("Step 1: Extracting scene overview")
            scene_info = self.scene_overview(vlm_description=vlm_description)
            overall_description = getattr(scene_info, 'overall_description', 'Unknown scene')
            scene_type = getattr(scene_info, 'scene_type', 'unknown')
            logger.debug(f"Scene overview extracted - Description: {overall_description[:100]}... Type: {scene_type}")
        except Exception as e:
            logger.warning(f"Failed to extract scene overview: {e}")
            overall_description = "Unknown scene"
            scene_type = "unknown"
        
        # Step 2: Identify main entities
        try:
            logger.debug("Step 2: Identifying main entities")
            entity_identification = self.entity_identifier(vlm_description=vlm_description)
            entity_list_str = getattr(entity_identification, 'entity_list', '[]')
            logger.debug(f"Entity list string: {entity_list_str[:200] if entity_list_str else 'None'}")
            
            # Parse entity list
            try:
                entities_info = json.loads(entity_list_str)
                if not isinstance(entities_info, list):
                    logger.warning(f"Parsed entity info is not a list: {type(entities_info)}")
                    entities_info = []
                logger.debug(f"Parsed {len(entities_info)} entities from entity list")
            except json.JSONDecodeError as je:
                logger.error(f"Failed to parse entity list JSON: {je}")
                logger.debug(f"Problematic JSON string: {entity_list_str[:500]}")
                entities_info = []
        except Exception as e:
            logger.error(f"Failed to identify entities: {e}")
            entities_info = []
        
        # Step 3: Extract details for each entity
        entities = []
        for i, entity_info in enumerate(entities_info[:10]):  # Limit to first 10 entities
            if not isinstance(entity_info, dict) or 'name' not in entity_info:
                logger.warning(f"Skipping invalid entity at index {i}: {entity_info}")
                continue
                
            entity_name = entity_info.get('name', 'Unknown')
            entity_category = entity_info.get('category', 'unknown')
            logger.debug(f"Step 3: Extracting details for entity {i}: {entity_name} ({entity_category})")
            
            try:
                entity_details = self.entity_detail_extractor(
                    vlm_description=vlm_description,
                    entity_name=entity_name,
                    entity_category=entity_category
                )
                
                entity_details_str = getattr(entity_details, 'entity_details', '{}')
                logger.debug(f"Entity details string for {entity_name}: {entity_details_str[:200] if entity_details_str else 'None'}")
                
                # Parse entity details
                try:
                    entity_data = json.loads(entity_details_str)
                    entities.append(entity_data)
                    logger.debug(f"Successfully parsed entity details for {entity_name}")
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse entity details for {entity_name}")
                    logger.debug(f"Problematic JSON: {entity_details_str[:500]}")
            except Exception as e:
                logger.warning(f"Failed to extract details for {entity_name}: {e}")
        
        logger.debug(f"Collected {len(entities)} entities with details")
        
        # Step 4: Organize hierarchy
        try:
            logger.debug("Step 4: Organizing entity hierarchy")
            if entities:
                hierarchy_result = self.hierarchy_organizer(entities_data=json.dumps(entities))
                organized_entities_str = getattr(hierarchy_result, 'organized_entities', '[]')
                logger.debug(f"Organized entities string: {organized_entities_str[:200] if organized_entities_str else 'None'}")
                
                try:
                    organized_entities = json.loads(organized_entities_str)
                    logger.debug(f"Successfully organized {len(organized_entities)} entities")
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse organized entities")
                    logger.debug(f"Problematic JSON: {organized_entities_str[:500]}")
                    organized_entities = entities
            else:
                organized_entities = []
                logger.debug("No entities to organize")
        except Exception as e:
            logger.warning(f"Failed to organize entity hierarchy: {e}")
            organized_entities = entities
        
        # Step 5: Refine structure
        try:
            logger.debug("Step 5: Refining entity structure")
            if organized_entities:
                refinement_result = self.structure_refiner(
                    raw_entities=json.dumps(organized_entities),
                    scene_context=overall_description
                )
                refined_entities_str = getattr(refinement_result, 'refined_entities', '[]')
                logger.debug(f"Refined entities string: {refined_entities_str[:200] if refined_entities_str else 'None'}")
                
                try:
                    refined_entities = json.loads(refined_entities_str)
                    logger.debug(f"Successfully refined {len(refined_entities)} entities")
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse refined entities")
                    logger.debug(f"Problematic JSON: {refined_entities_str[:500]}")
                    refined_entities = organized_entities
            else:
                refined_entities = []
                logger.debug("No entities to refine")
        except Exception as e:
            logger.warning(f"Failed to refine entity structure: {e}")
            refined_entities = organized_entities
        
        # Step 6: Validate structure
        try:
            logger.debug("Step 6: Validating entity structure")
            if refined_entities:
                validation_result = self.structure_validator(entities_data=json.dumps(refined_entities))
                validated_entities_str = getattr(validation_result, 'validated_entities', '[]')
                logger.debug(f"Validated entities string: {validated_entities_str[:200] if validated_entities_str else 'None'}")
                
                try:
                    validated_entities = json.loads(validated_entities_str)
                    logger.debug(f"Successfully validated {len(validated_entities)} entities")
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse validated entities")
                    logger.debug(f"Problematic JSON: {validated_entities_str[:500]}")
                    validated_entities = refined_entities
            else:
                validated_entities = []
                logger.debug("No entities to validate")
        except Exception as e:
            logger.warning(f"Failed to validate entity structure: {e}")
            validated_entities = refined_entities
        
        # Return structured result
        result = {
            "overall_description": overall_description,
            "scene_type": scene_type,
            "entities": validated_entities
        }
        
        logger.debug(f"EntityStructuringPipeline completed with {len(validated_entities)} final entities")
        return result


# Utility functions for working with entity structures
def fix_entity_structure(entity_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively fix entity structure to match expected format.
    
    Args:
        entity_dict (Dict[str, Any]): Entity dictionary to fix
        
    Returns:
        Dict[str, Any]: Fixed entity dictionary
    """
    logger.debug(f"Fixing entity structure for entity: {entity_dict.get('name', 'Unknown') if isinstance(entity_dict, dict) else 'Non-dict'}")
    
    # Handle non-dict inputs
    if not isinstance(entity_dict, dict):
        logger.warning(f"Entity is not a dict, returning empty dict. Type: {type(entity_dict)}")
        return {}
    
    # Fix position if it's just a string
    if isinstance(entity_dict.get('position'), str):
        logger.debug("Fixing position from string to dict")
        entity_dict['position'] = {
            'description': entity_dict['position'],
            'x_percent': None,
            'y_percent': None,
            'depth_layer': entity_dict['position'] if entity_dict['position'] in ['foreground', 'midground', 'background'] else 'midground'
        }
    elif 'position' not in entity_dict or not isinstance(entity_dict.get('position'), dict):
        # Default position if not provided
        logger.debug("Setting default position")
        entity_dict['position'] = {
            'description': 'unknown position',
            'x_percent': None,
            'y_percent': None,
            'depth_layer': 'midground'
        }
    else:
        # Ensure position has required fields
        pos = entity_dict['position']
        if 'description' not in pos:
            pos['description'] = pos.get('depth_layer', 'unknown position')
        if 'depth_layer' not in pos:
            pos['depth_layer'] = 'midground'
        if 'x_percent' not in pos:
            pos['x_percent'] = None
        if 'y_percent' not in pos:
            pos['y_percent'] = None
    
    # Fix colors if they use 'code' instead of 'hex'
    if 'colors' in entity_dict and isinstance(entity_dict['colors'], list):
        logger.debug(f"Fixing {len(entity_dict['colors'])} colors")
        for color in entity_dict['colors']:
            if isinstance(color, dict) and 'code' in color and 'hex' not in color:
                color['hex'] = color.pop('code')
            if isinstance(color, dict) and 'hex' not in color:
                color['hex'] = '#000000'  # Default
    
    # Ensure required fields exist
    if 'children' not in entity_dict:
        entity_dict['children'] = []
    if 'colors' not in entity_dict:
        entity_dict['colors'] = []
    if 'attributes' not in entity_dict:
        entity_dict['attributes'] = {}
    
    # Fix quantity and dimensions if missing
    if 'quantity' not in entity_dict:
        entity_dict['quantity'] = None
    if 'dimensions' not in entity_dict:
        entity_dict['dimensions'] = None
    
    # Recursively fix children
    for child in entity_dict.get('children', []):
        if isinstance(child, dict):
            fix_entity_structure(child)
    
    logger.debug(f"Entity structure fixed successfully: {entity_dict.get('name', 'Unknown')}")
    return entity_dict


def normalize_entity_categories(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize entity categories to a standard set.
    
    Args:
        entities (List[Dict[str, Any]]): List of entity dictionaries
        
    Returns:
        List[Dict[str, Any]]: List with normalized categories
    """
    logger.debug(f"Normalizing categories for {len(entities)} entities")
    
    # Standard category mappings
    category_mappings = {
        'person': 'person',
        'people': 'person',
        'human': 'person',
        'man': 'person',
        'woman': 'person',
        'child': 'person',
        'building': 'building',
        'structure': 'building',
        'house': 'building',
        'home': 'building',
        'mountain': 'natural',
        'hill': 'natural',
        'tree': 'natural',
        'plant': 'natural',
        'forest': 'natural',
        'water': 'natural',
        'lake': 'natural',
        'river': 'natural',
        'sky': 'natural',
        'cloud': 'natural',
        'road': 'infrastructure',
        'path': 'infrastructure',
        'street': 'infrastructure',
        'sidewalk': 'infrastructure',
        'vehicle': 'vehicle',
        'car': 'vehicle',
        'truck': 'vehicle',
        'bus': 'vehicle',
        'bicycle': 'vehicle',
        'animal': 'animal',
        'dog': 'animal',
        'cat': 'animal',
        'bird': 'animal'
    }
    
    normalized_entities = []
    for entity in entities:
        if isinstance(entity, dict):
            category = entity.get('category', 'unknown').lower()
            # Try to map to standard category
            for key, standard_cat in category_mappings.items():
                if key in category:
                    entity['category'] = standard_cat
                    break
            normalized_entities.append(entity)
    
    logger.debug(f"Normalized {len(normalized_entities)} entities")
    return normalized_entities