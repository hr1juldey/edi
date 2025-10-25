import dspy
import asyncio
from ollama import AsyncClient # type: ignore
from PIL import Image
import base64
import io
from typing import List, Union, Optional
from pydantic import BaseModel, Field
import json
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ============================================================================
# DATA STRUCTURES - Hierarchical Entity Tree
# ============================================================================

class Position(BaseModel):
    """Position information for an entity"""
    description: str = Field(description="Textual description of position")
    x_percent: Optional[float] = Field(None, description="Horizontal position as percentage")
    y_percent: Optional[float] = Field(None, description="Vertical position as percentage")
    depth_layer: str = Field(description="Depth layer: 'foreground', 'midground', or 'background'")


class Color(BaseModel):
    """Color information"""
    hex: str = Field(description="Hex color code (e.g., #FF5733)")
    name: Optional[str] = Field(None, description="Common color name")
    variations: Optional[List[str]] = Field(default_factory=list, description="Color variations")


class Entity(BaseModel):
    """A single entity/component in the image"""
    name: str = Field(description="Entity name or type")
    category: str = Field(description="Category (e.g., 'building', 'natural', 'infrastructure')")
    position: Position
    colors: List[Color] = Field(default_factory=list)
    description: str = Field(description="Detailed textual description")
    quantity: Optional[int] = Field(None, description="Number if multiple similar entities")
    dimensions: Optional[str] = Field(None, description="Size description")
    children: List["Entity"] = Field(default_factory=list, description="Sub-entities/components")
    attributes: dict = Field(default_factory=dict, description="Additional attributes")


class ImageScene(BaseModel):
    """Complete hierarchical scene description"""
    image_id: str = Field(description="Unique identifier for the image")
    image_path: Optional[str] = Field(None, description="Path to the image file")
    overall_description: str = Field(description="High-level scene description")
    scene_type: str = Field(description="Type of scene")
    entities: List[Entity] = Field(default_factory=list, description="Top-level entities")
    metadata: dict = Field(default_factory=dict, description="Additional scene metadata")


class MultiImageScene(BaseModel):
    """Container for multiple image scenes"""
    scenes: List[ImageScene] = Field(description="List of image scenes")
    relationship: Optional[str] = Field(None, description="Relationship between images")


# ============================================================================
# NEW SIMPLIFIED DSPY SIGNATURES (Breaking down the monolithic approach)
# ============================================================================

class IdentifyEntities(dspy.Signature):
    """Identify all main entities in the image scene"""
    vlm_description: str = dspy.InputField(desc="Raw description from vision model")
    image_id: str = dspy.InputField(desc="Unique identifier for the image")
    
    entity_list: str = dspy.OutputField(desc="JSON array of entity names and categories [{'name': 'entity_name', 'category': 'category'}]")


class ExtractSceneTypeAndDescription(dspy.Signature):
    """Extract basic scene type and overall description"""
    vlm_description: str = dspy.InputField(desc="Raw description from vision model")
    image_id: str = dspy.InputField(desc="Unique identifier for the image")
    
    overall_description: str = dspy.OutputField(desc="High-level scene description")
    scene_type: str = dspy.OutputField(desc="Type of scene (landscape, urban, village, etc)")


class ExtractSingleEntityDetails(dspy.Signature):
    """Extract complete details for a single entity"""
    vlm_description: str = dspy.InputField(desc="Raw description from vision model")
    entity_name: str = dspy.InputField(desc="Name of the entity to extract details for")
    entity_category: str = dspy.InputField(desc="Category of the entity")
    
    entity_json: str = dspy.OutputField(
        desc="Complete entity in JSON format with name, category, position (with description, x_percent, y_percent, depth_layer), "
             "colors (array of objects with hex and name), description, children (array), and attributes. "
             "Example: {\"name\":\"Mountain\",\"category\":\"natural\",\"position\":{\"description\":\"far background\",\"x_percent\":50,\"y_percent\":20,\"depth_layer\":\"background\"},"
             "\"colors\":[{\"hex\":\"#FFFFFF\",\"name\":\"white\"}],\"description\":\"Snow-capped peak\",\"children\":[],\"attributes\":{}}"
    )


class ExtractEntityChildren(dspy.Signature):
    """Extract child entities for a specific parent entity"""
    vlm_description: str = dspy.InputField(desc="Raw description from vision model")
    parent_entity_description: str = dspy.InputField(desc="Description of the parent entity")
    
    child_entities_json: str = dspy.OutputField(desc="JSON array of child entities for the parent")


# ============================================================================
# DSPY MODULE (Using decomposed approach for better stability)
# ============================================================================

class ImageToEntityTree(dspy.Module):
    """Module to convert VLM output to hierarchical entity tree using smaller, focused steps"""
    
    def __init__(self):
        super().__init__()
        # Use focused DSPy predictors for each aspect
        self.identify_entities = dspy.ChainOfThought(IdentifyEntities)
        self.extract_scene_info = dspy.ChainOfThought(ExtractSceneTypeAndDescription)
        self.extract_entity_details = dspy.ChainOfThought(ExtractSingleEntityDetails)
        self.extract_children = dspy.ChainOfThought(ExtractEntityChildren)
    
    def forward(self, vlm_output: str, image_id: str) -> ImageScene:
        """Process VLM output into structured entity tree using decomposed approach"""
        
        # Step 1: Extract overall scene information
        try:
            scene_info = self.extract_scene_info(
                vlm_description=vlm_output,
                image_id=image_id
            )
            
            overall_description = scene_info.overall_description if hasattr(scene_info, 'overall_description') and scene_info.overall_description else "Unknown scene"
            scene_type = scene_info.scene_type if hasattr(scene_info, 'scene_type') and scene_info.scene_type else "unknown"
        except Exception as e:
            logger.error(f"DSPy scene info extraction failed: {e}")
            return ImageScene(
                image_id=image_id,
                overall_description=vlm_output[:200] if vlm_output else "Unknown scene",
                scene_type="unknown",
                entities=[]
            )
        
        # Step 2: Identify entities in the scene
        try:
            entity_identification = self.identify_entities(
                vlm_description=vlm_output,
                image_id=image_id
            )
            
            entity_list_str = entity_identification.entity_list if hasattr(entity_identification, 'entity_list') and entity_identification.entity_list else "[]"
            
            # Parse the entity list JSON
            try:
                entities_info = json.loads(entity_list_str)
                if not isinstance(entities_info, list):
                    entities_info = []
            except json.JSONDecodeError:
                logger.warning(f"Could not parse entity list JSON: {entity_list_str}")
                entities_info = []
                
        except Exception as e:
            logger.error(f"DSPy entity identification failed: {e}")
            # If entity identification fails, return a scene with no entities
            return ImageScene(
                image_id=image_id,
                overall_description=overall_description,
                scene_type=scene_type,
                entities=[]
            )
        
        # Step 3: Process each identified entity to extract detailed information
        entities = []
        
        for i, entity_info in enumerate(entities_info[:5]):  # Limit to first 5 entities to avoid too many calls
            if not isinstance(entity_info, dict) or 'name' not in entity_info:
                continue
                
            entity_name = entity_info.get('name', 'unknown')
            entity_category = entity_info.get('category', 'unknown')
            
            try:
                # Extract detailed information for this entity
                entity_details = self.extract_entity_details(
                    vlm_description=vlm_output,
                    entity_name=entity_name,
                    entity_category=entity_category
                )
                
                # Parse the entity JSON
                entity_json_str = entity_details.entity_json if hasattr(entity_details, 'entity_json') and entity_details.entity_json else None
                
                if entity_json_str:
                    # Clean up the JSON string to handle possible issues
                    entity_json_str = entity_json_str.strip()
                    
                    # If it starts with a code block marker, try to extract the content
                    if entity_json_str.startswith("```"):
                        import re
                        json_match = re.search(r'\{.*\}', entity_json_str, re.DOTALL)
                        if json_match:
                            entity_json_str = json_match.group(0)
                        else:
                            # If no object found, try to find an array
                            json_match = re.search(r'\[.*\]', entity_json_str, re.DOTALL)
                            if json_match:
                                entity_json_str = json_match.group(0)
                    
                    try:
                        entity_data = json.loads(entity_json_str)
                        
                        # Validate and fix entity structure
                        fixed_entity = self._fix_entity_structure(entity_data)
                        
                        # Create Entity object
                        entity_obj = Entity(**fixed_entity)
                        
                        # Extract and process children if they exist
                        child_entities = []
                        if hasattr(entity_details, 'child_entities_json') and entity_details.child_entities_json:
                            try:
                                child_data = json.loads(entity_details.child_entities_json)
                                if isinstance(child_data, list):
                                    for child_item in child_data[:3]:  # Limit to 3 children
                                        if isinstance(child_item, dict) and 'name' in child_item:
                                            child_fixed = self._fix_entity_structure(child_item)
                                            child_entities.append(Entity(**child_fixed))
                            except json.JSONDecodeError:
                                logger.warning("Could not parse child entities JSON")
                        
                        # Update the entity with processed children
                        entity_obj.children = child_entities
                        
                        entities.append(entity_obj)
                        
                    except json.JSONDecodeError as e:
                        logger.error(f"Could not parse entity JSON for {entity_name}: {e}")
                        logger.debug(f"Problematic JSON: {entity_json_str[:500]}")
                else:
                    logger.warning(f"No entity JSON returned for {entity_name}")
                    
            except Exception as e:
                logger.error(f"Error processing entity {entity_name}: {e}")
                # Continue with other entities instead of failing completely
                continue
        
        # Create ImageScene with extracted data
        scene = ImageScene(
            image_id=image_id,
            overall_description=overall_description,
            scene_type=scene_type,
            entities=entities,
            metadata={"vlm_output_length": len(vlm_output), "num_entities_identified": len(entities_info)}
        )
        
        return scene
    
    def _fix_entity_structure(self, entity_dict):
        """Recursively fix entity structure to match Pydantic model"""
        # Fix position if it's just a string
        if isinstance(entity_dict.get('position'), str):
            entity_dict['position'] = {
                'description': entity_dict['position'],
                'x_percent': None,
                'y_percent': None,
                'depth_layer': entity_dict['position'] if entity_dict['position'] in ['foreground', 'midground', 'background'] else 'midground'
            }
        elif 'position' not in entity_dict or not isinstance(entity_dict.get('position'), dict):
            # Default position if not provided
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
        if 'colors' in entity_dict:
            for color in entity_dict['colors']:
                if 'code' in color and 'hex' not in color:
                    color['hex'] = color.pop('code')
                if 'hex' not in color:
                    color['hex'] = '#000000'  # Default
        else:
            # Default colors if not provided
            entity_dict['colors'] = [{'hex': '#808080', 'name': 'gray', 'variations': []}]
        
        # Ensure required fields exist
        if 'children' not in entity_dict:
            entity_dict['children'] = []
        if 'description' not in entity_dict:
            entity_dict['description'] = 'No description available'
        if 'category' not in entity_dict:
            entity_dict['category'] = 'unknown'
        if 'attributes' not in entity_dict:
            entity_dict['attributes'] = {}
        
        # Recursively fix children
        for child in entity_dict.get('children', []):
            self._fix_entity_structure(child)
        
        return entity_dict


# ============================================================================
# IMAGE ENCODING & VLM INTERACTION
# ============================================================================

def encode_images_base64(paths: Union[str, List[str]]) -> List[str]:
    """Encode one or more image files into base64 strings"""
    if isinstance(paths, str):
        paths = [paths]
    
    encoded_images = []
    for path in paths:
        try:
            with Image.open(path) as img:
                buffer = io.BytesIO()
                img.save(buffer, format=img.format or "PNG")
                byte_data = buffer.getvalue()
                base64_str = base64.b64encode(byte_data).decode("utf-8")
                encoded_images.append(base64_str)
        except Exception as e:
            logger.warning(f"Skipping '{path}' - {e}")
    
    return encoded_images


async def get_vlm_description(
    prompt: str, 
    image_paths: Union[str, List[str]], 
    vision_model: str = "gemma3:4b"
) -> str:
    """Get raw VLM description for image(s)"""
    image_b64_list = encode_images_base64(image_paths)
    
    message = {
        "role": "user",
        "content": prompt,
        "images": image_b64_list
    }
    
    response = await AsyncClient().chat(model=vision_model, messages=[message])
    return response['message']['content']


# ============================================================================
# STABLE ENTITY EXTRACTION WITH MULTIPLE RUNS
# ============================================================================

async def extract_entities_stable(
    image_paths: Union[str, List[str]],
    vision_model: str = "gemma3:4b",
    brain_model: str = "ollama/qwen3:8b",
    num_runs: int = 3
) -> Union[ImageScene, MultiImageScene]:
    """
    Extract entities with improved stability by running multiple times and selecting best result
    
    Args:
        image_paths: Single path or list of paths to images
        model: VLM model for image understanding
        brain_model: LLM model for structure extraction
        num_runs: Number of times to run the extraction (default: 3)
    
    Returns:
        ImageScene for single image, MultiImageScene for multiple images
    """
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    
    # Configure DSPy with the brain model
    dspy.configure(lm=dspy.LM(model=brain_model))
    
    # Detailed prompt for VLM
    vlm_prompt = """Systematically analyze this image and provide a comprehensive description:

                1. SCENE OVERVIEW: Describe the overall scene type and context
                2. BACKGROUND LAYER: All distant elements (mountains, sky, horizon)
                3. MIDGROUND LAYER: Middle-distance elements (buildings, terrain, paths)
                4. FOREGROUND LAYER: Close elements (objects, details, ground)

                For each entity/component, specify:
                - Name and category
                - Position (location description and depth layer)
                - Colors (with hex codes like #FFFFFF)
                - Detailed description
                - Quantity if multiple
                - Sub-components (e.g., a building has walls, roof, windows)

                Format: Organize by depth layers. Be extremely detailed for accessibility."""
    
    # Run multiple times and collect results
    results = []
    for run_idx in range(num_runs):
        logger.info(f"Running extraction attempt {run_idx + 1}/{num_runs}")
        
        scenes = []
        for idx, img_path in enumerate(image_paths):
            image_id = f"image_{idx}_{img_path.split('/')[-1]}"
            
            # Get VLM description
            logger.info(f"Processing {img_path} with VLM...")
            try:
                vlm_output = await get_vlm_description(vlm_prompt, img_path, vision_model)
            except Exception as e:
                logger.error(f"Failed to get VLM description: {e}")
                # Create a minimal scene from the error
                scene = ImageScene(
                    image_id=image_id,
                    image_path=img_path,
                    overall_description=f"Error processing image: {e}",
                    scene_type="error",
                    entities=[]
                )
                scenes.append(scene)
                continue
                
            logger.debug(f"VLM output length: {len(vlm_output)} chars")
            
            # Initialize the entity tree extractor
            extractor = ImageToEntityTree()
            
            # Extract structured entity tree using DSPy
            logger.info("Extracting entity tree with DSPy...")
            scene = extractor(vlm_output=vlm_output, image_id=image_id)
            scene.image_path = img_path
            
            scenes.append(scene)
        
        # Store the result for this run
        if len(scenes) == 1:
            results.append(scenes[0])
        else:
            results.append(MultiImageScene(scenes=scenes))
    
    # Select the best result based on number of entities (more is generally better)
    # Also prioritize results that have valid scene_type and overall_description
    best_result = None
    best_score = -1
    
    for result in results:
        score = 0
        # Score based on number of entities
        if isinstance(result, ImageScene):
            score = len(result.entities)
            # Bonus for having good descriptions
            if result.scene_type and result.scene_type != "unknown":
                score += 1
            if result.overall_description and len(result.overall_description) > 10:
                score += 1
        else:  # MultiImageScene
            score = sum(len(scene.entities) for scene in result.scenes)
            # Bonus for having good descriptions
            for scene in result.scenes:
                if scene.scene_type and scene.scene_type != "unknown":
                    score += 1
                if scene.overall_description and len(scene.overall_description) > 10:
                    score += 1
        
        if score > best_score:
            best_score = score
            best_result = result
    
    if best_result is None:
        # Fallback if all runs failed
        if len(scenes) == 1:
            return ImageScene(
                image_id="fallback", 
                overall_description="Failed to process image",
                scene_type="error",
                entities=[]
            )
        else:
            return MultiImageScene(scenes=[
                ImageScene(
                    image_id="fallback", 
                    overall_description="Failed to process image",
                    scene_type="error",
                    entities=[]
                )
            ])
    
    logger.info(f"Selected best result with {best_score} points from {num_runs} attempts")
    return best_result


# ============================================================================
# HIGH-LEVEL API
# ============================================================================

async def process_images_to_entity_tree(
    image_paths: Union[str, List[str]],
    vision_model: str = "gemma3:4b",
    brain_model: str = "ollama/qwen3:8b"
) -> Union[ImageScene, MultiImageScene]:
    """
    Process single or multiple images into hierarchical entity trees
    
    Args:
        image_paths: Single path or list of paths to images
        model: VLM model for image understanding
        brain_model: LLM model for structure extraction
        
    Returns:
        ImageScene for single image, MultiImageScene for multiple images
    """
    return await extract_entities_stable(
        image_paths=image_paths,
        vision_model=vision_model,
        brain_model=brain_model,
        num_runs=3
    )


def save_entity_tree(entity_tree: Union[ImageScene, MultiImageScene], output_path: str):
    """Save entity tree to JSON file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(entity_tree.model_dump(), f, indent=2, ensure_ascii=False)
    logger.info(f"Saved entity tree to {output_path}")


def print_entity_tree_summary(scene: ImageScene, indent: int = 0):
    """Print a human-readable summary of the entity tree"""
    prefix = "  " * indent
    print(f"\n{prefix}{'='*50}")
    print(f"{prefix}SCENE SUMMARY")
    print(f"{prefix}{'='*50}")
    print(f"{prefix}Type: {scene.scene_type}")
    print(f"{prefix}Description: {scene.overall_description}")
    print(f"{prefix}Total Entities: {len(scene.entities)}")
    print(f"{prefix}{'='*50}\n")
    
    def print_entity(entity: Entity, level: int = 0):
        """Recursively print entity tree"""
        indent_str = "  " * level
        print(f"{indent_str}▸ {entity.name} ({entity.category})")
        print(f"{indent_str}  Position: {entity.position.description} [{entity.position.depth_layer}]")
        if entity.colors:
            color_str = ", ".join([f"{c.name or 'color'}:{c.hex}" for c in entity.colors[:3]])
            print(f"{indent_str}  Colors: {color_str}")
        print(f"{indent_str}  Description: {entity.description[:80]}...")
        if entity.quantity:
            print(f"{indent_str}  Quantity: {entity.quantity}")
        if entity.children:
            print(f"{indent_str}  ↳ {len(entity.children)} sub-entities:")
            for child in entity.children:
                print_entity(child, level + 2)
        print()
    
    for entity in scene.entities:
        print_entity(entity)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

async def main():
    # Example 1: Single image
    print("="*60)
    print("SINGLE IMAGE PROCESSING")
    print("="*60)
    
    single_image = "/home/riju279/Documents/Code/Zonko/Interpreter/interpreter/IP.jpeg"
    
    result = await process_images_to_entity_tree(
        image_paths=single_image,
        vision_model="gemma3:4b",
        brain_model="ollama/qwen3:8b"
    )
    
    # Display summary
    print_entity_tree_summary(result)
    
    # Save to JSON
    save_entity_tree(result, "stable_single_image_entities.json")
    
    print("\n" + "="*60)
    print("SAMPLE ENTITY (JSON):")
    print("="*60)
    if result.entities:
        print(json.dumps(result.entities[0].model_dump(), indent=2))


if __name__ == "__main__":
    asyncio.run(main())