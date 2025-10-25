"""
Main orchestrator for the complete image entity extraction pipeline.
"""

import dspy
import json
import asyncio
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

from config import setup_llm_models
from image_encoder import encode_image_to_base64, validate_image_path
from vlm_interface import (
    analyze_image_with_vlm, 
    format_vlm_prompt_for_entity_extraction
)
from entity_structurer import EntityStructuringPipeline, fix_entity_structure
from utils.validators import validate_scene_structure
from utils.formatters import format_scene_as_json


# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class ImageEntityExtractor:
    """
    Main orchestrator for extracting structured entities from images.
    """
    
    def __init__(self, vlm_model: str = "gemma3:4b"):
        """
        Initialize the extractor with models.
        
        Args:
            vlm_model (str): Vision Language Model to use for image analysis
        """
        logger.debug("Initializing ImageEntityExtractor")
        # Setup models
        self.eye_model, self.brain_model = setup_llm_models()
        self.vlm_model = vlm_model
        
        # Initialize DSPy pipeline (without configuring - will be done in context)
        self.structuring_pipeline = EntityStructuringPipeline()
        logger.debug("ImageEntityExtractor initialized successfully")
    
    async def process_single_image(self, image_path: str) -> Dict[str, Any]:
        """
        Process a single image through the complete pipeline.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            Dict[str, Any]: Structured scene with entities
        """
        logger.debug(f"Processing single image: {image_path}")
        
        # Validate image path
        if not validate_image_path(image_path):
            raise ValueError(f"Invalid image path: {image_path}")
        
        # Encode image
        try:
            logger.debug("Encoding image to base64")
            base64_image = encode_image_to_base64(image_path)
            logger.debug(f"Image encoded successfully, length: {len(base64_image) if base64_image else 0}")
        except Exception as e:
            logger.error(f"Failed to encode image: {e}")
            raise RuntimeError(f"Failed to encode image: {e}")
        
        # Create VLM prompt
        logger.debug("Creating VLM prompt")
        vlm_prompt = format_vlm_prompt_for_entity_extraction()
        logger.debug(f"VLM prompt created, length: {len(vlm_prompt)}")
        
        # Analyze image with VLM
        try:
            logger.debug("Analyzing image with VLM")
            vlm_description = await analyze_image_with_vlm(
                base64_image, vlm_prompt, self.vlm_model
            )
            logger.debug(f"VLM analysis completed, response length: {len(vlm_description) if vlm_description else 0}")
        except Exception as e:
            logger.error(f"Failed to analyze image with VLM: {e}")
            raise RuntimeError(f"Failed to analyze image with VLM: {e}")
        
        # Structure entities with DSPy (using context)
        try:
            logger.debug("Structuring entities with DSPy")
            # Use DSPy context to ensure proper model configuration
            with dspy.context(lm=self.brain_model):
                structured_scene = self.structuring_pipeline(vlm_description=vlm_description)
            logger.debug("DSPy structuring completed successfully")
        except Exception as e:
            logger.error(f"Failed to structure entities with DSPy: {e}")
            logger.debug(f"VLM description that failed: {vlm_description[:500] if vlm_description else 'None'}")
            raise RuntimeError(f"Failed to structure entities with DSPy: {e}")
        
        # Add image metadata
        logger.debug("Adding image metadata")
        structured_scene["image_id"] = f"image_{Path(image_path).stem}"
        structured_scene["image_path"] = image_path
        
        # Fix entity structure
        logger.debug("Fixing entity structure")
        fixed_entities = []
        entities = structured_scene.get("entities", [])
        logger.debug(f"Found {len(entities)} entities to fix")
        
        for i, entity in enumerate(entities):
            if isinstance(entity, dict):
                try:
                    fixed_entity = fix_entity_structure(entity)
                    fixed_entities.append(fixed_entity)
                    logger.debug(f"Fixed entity {i}: {fixed_entity.get('name', 'Unknown')}")
                except Exception as e:
                    logger.warning(f"Failed to fix entity {i}: {e}")
                    # Still add the entity even if fixing failed
                    fixed_entities.append(entity)
            else:
                logger.warning(f"Entity {i} is not a dict: {type(entity)}")
                # Add non-dict entities as-is
                fixed_entities.append(entity)
        
        structured_scene["entities"] = fixed_entities
        
        # Validate structure
        logger.debug("Validating scene structure")
        try:
            is_valid = validate_scene_structure(structured_scene)
            if not is_valid:
                logger.warning("Scene structure validation failed")
            else:
                logger.debug("Scene structure validation passed")
        except Exception as e:
            logger.warning(f"Scene structure validation error: {e}")
        
        logger.debug(f"Completed processing image: {image_path}")
        return structured_scene
    
    async def process_multiple_images(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple images through the pipeline.
        
        Args:
            image_paths (List[str]): List of paths to image files
            
        Returns:
            List[Dict[str, Any]]: List of structured scenes
        """
        tasks = [self.process_single_image(path) for path in image_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"[WARN] Failed to process image {image_paths[i]}: {result}")
                # Create a minimal scene for failed processing
                processed_results.append({
                    "image_id": f"image_{i}_failed",
                    "image_path": image_paths[i],
                    "overall_description": "Failed to process image",
                    "scene_type": "error",
                    "entities": []
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    def save_scene_to_file(self, scene: Dict[str, Any], output_path: str) -> None:
        """
        Save a structured scene to a JSON file.
        
        Args:
            scene (Dict[str, Any]): Structured scene to save
            output_path (str): Path to save the JSON file
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(scene, f, indent=2, ensure_ascii=False)
            print(f"[INFO] Saved scene to {output_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to save scene to {output_path}: {e}")
    
    def print_scene_summary(self, scene: Dict[str, Any]) -> None:
        """
        Print a human-readable summary of the scene.
        
        Args:
            scene (Dict[str, Any]): Structured scene to summarize
        """
        print(f"\n{'='*50}")
        print(f"SCENE SUMMARY")
        print(f"{'='*50}")
        print(f"Type: {scene.get('scene_type', 'unknown')}")
        print(f"Description: {scene.get('overall_description', 'No description')}")
        print(f"Entities: {len(scene.get('entities', []))}")
        print(f"{'='*50}\n")
        
        entities = scene.get('entities', [])
        for i, entity in enumerate(entities[:5]):  # Show first 5 entities
            if isinstance(entity, dict):
                print(f"▸ {entity.get('name', 'Unknown')} ({entity.get('category', 'unknown')})")
                position = entity.get('position', {})
                if isinstance(position, dict):
                    print(f"  Position: {position.get('description', 'unknown position')}")
                colors = entity.get('colors', [])
                if colors:
                    color_str = ", ".join([f"{c.get('name', 'color')}:{c.get('hex', '#000000')}" 
                                          for c in colors[:3] if isinstance(c, dict)])
                    print(f"  Colors: {color_str}")
                print(f"  Description: {entity.get('description', 'No description')[:80]}...")
                if entity.get('children'):
                    print(f"  Children: {len(entity.get('children', []))}")
                print()


async def main():
    """
    Main function for testing the entity extractor.
    """
    print("=" * 60)
    print("SINGLE IMAGE PROCESSING")
    print("=" * 60)
    
    # Initialize extractor
    extractor = ImageEntityExtractor()
    
    # Process a single image
    single_image = "/home/riju279/Documents/Code/Zonko/Interpreter/interpreter/IP.jpeg"
    
    try:
        print(f"[INFO] Processing {single_image}...")
        scene = await extractor.process_single_image(single_image)
        
        # Print summary
        extractor.print_scene_summary(scene)
        
        # Save to JSON
        extractor.save_scene_to_file(scene, "test_single_scene.json")
        
        print("\n" + "=" * 60)
        print("SAMPLE ENTITY (JSON):")
        print("=" * 60)
        if scene.get('entities'):
            print(json.dumps(scene['entities'][0], indent=2))
        
        print("[SUCCESS] Single image processing completed!")
        
    except Exception as e:
        print(f"[ERROR] Failed to process single image: {e}")
        return False
    
    # Process multiple images
    print("\n" + "=" * 60)
    print("MULTIPLE IMAGE PROCESSING")
    print("=" * 60)
    
    multi_images = [
        "/home/riju279/Documents/Code/Zonko/Interpreter/interpreter/IP.jpeg",
        "/home/riju279/Documents/Code/Zonko/Interpreter/interpreter/WP.jpg"
    ]
    
    # Filter to only existing files
    from pathlib import Path
    existing_images = [path for path in multi_images if Path(path).exists()]
    
    if not existing_images:
        print("[WARN] No test images found")
        return True
    
    try:
        print(f"[INFO] Processing {len(existing_images)} images...")
        scenes = await extractor.process_multiple_images(existing_images)
        
        # Print summaries for each scene
        for scene in scenes:
            extractor.print_scene_summary(scene)
        
        # Save combined output
        extractor.save_scenes_to_file(scenes, "test_multiple_scenes.json")
        
        print("[SUCCESS] Multiple image processing completed!")
        
    except Exception as e:
        print(f"[ERROR] Failed to process multiple images: {e}")
        return False
    
    return True


if __name__ == "__main__":
    asyncio.run(main())