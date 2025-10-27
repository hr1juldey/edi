import cv2
import numpy as np
import torch
from PIL import Image
import open_clip
from typing import List, Tuple, Dict
import logging
from functools import lru_cache
import json
import dspy

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("clip_entity_detector")

@lru_cache(maxsize=1)
def _load_clip(device):
    """
    Load CLIP model + transform once and cache it.
    Tries multiple pretrained tags and logs each attempt.
    """
    model_name = "ViT-B-32"
    candidates = ("openai", "laion2b_s34b_b79k", "laion400m_e32")
    errors = {}

    for pretrained in candidates:
        try:
            logger.info(f"Attempting to load CLIP model '{model_name}' with pretrained='{pretrained}'...")
            clip_model, clip_transform, _ = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained
            )
            clip_model.to(device)
            clip_model.eval()
            logger.info(f"Successfully loaded CLIP model with '{pretrained}'.")
            return clip_model, clip_transform

        except FileNotFoundError as e:
            errors[pretrained] = f"File not found: {e}"
            logger.warning(f"Pretrained weights '{pretrained}' not found.")
        except (RuntimeError, OSError) as e:
            errors[pretrained] = str(e)
            logger.warning(f"Runtime error loading '{pretrained}': {e}")
        except Exception as e:
            errors[pretrained] = f"Unexpected error: {e.__class__.__name__}: {e}"
            logger.error(f"Unexpected error with '{pretrained}': {e}", exc_info=True)

    # If none succeeded
    error_summary = "\n".join([f" - {k}: {v}" for k, v in errors.items()])
    raise RuntimeError(
        f"Failed to load any CLIP model from {candidates}. Errors:\n{error_summary}"
    )

class ExtractEntities(dspy.Signature):
    """
    Extract entities from a user prompt for image editing.
    Focus on objects that need to be identified in the image.
    """
    prompt = dspy.InputField(desc="Original user prompt for image editing")
    
    entities = dspy.OutputField(desc="JSON list of entities to identify in the image")

class ImprovedEntityExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extract_entities = dspy.ChainOfThought(ExtractEntities)
    
    def forward(self, prompt):
        # Extract entities from prompt
        entities_result = self.extract_entities(prompt=prompt)
        
        # Parse the entities from the JSON output
        try:
            import json
            entities = json.loads(entities_result.entities)
        except:
            # Fallback: extract basic entities from prompt
            entities = []
            if "blue" in prompt.lower():
                entities.append("blue")
            if "red" in prompt.lower():
                entities.append("red")
            if "shed" in prompt.lower() or "building" in prompt.lower():
                entities.append("building")
            if "tin" in prompt.lower():
                entities.append("tin")
            if "roof" in prompt.lower():
                entities.append("roof")
            
        # Return a dspy.Prediction object as per DSPy patterns
        return dspy.Prediction(
            entities=entities
        )

# Define a DSPy signature for improved keyword extraction (this was missing)
class ExtractKeywords(dspy.Signature):
    """
    Extract relevant keywords from a user prompt for image editing.
    Focus on colors, objects, and specific entities that need to be identified in the image.
    """
    prompt = dspy.InputField(desc="Original user prompt for image editing")
    
    keywords = dspy.OutputField(desc="JSON list of relevant keywords for image search (colors, objects, entities)")


def get_entity_bounding_boxes(image, prompt, device=None):
    """
    Use CLIP to detect entities from user prompt and return vague bounding boxes.
    
    Args:
        image: Input image as numpy array
        prompt: User prompt describing what to edit
        device: Device to run on ('cuda' or 'cpu')
    
    Returns:
        dict: Dictionary with entity names as keys and bounding box lists as values
    """
    # Define the DSPy signature inside the function to avoid import issues
    def create_entity_extractor():
        import dspy

        class ExtractKeywords(dspy.Signature):
            """
            Extract relevant keywords from a user prompt for image editing.
            Focus on colors, objects, and specific entities that need to be identified in the image.
            """
            prompt = dspy.InputField(desc="Original user prompt for image editing")
            
            keywords = dspy.OutputField(desc="JSON list of relevant keywords for image search (colors, objects, entities)")

        class ImprovedEntityExtractor(dspy.Module):
            def __init__(self):
                super().__init__()
                self.extract_keywords = dspy.ChainOfThought(ExtractKeywords)
            
            def forward(self, prompt):
                # Extract entities from prompt using the new signature
                keywords_result = self.extract_keywords(prompt=prompt)
                
                # Parse the entities from the JSON output
                try:
                    import json
                    entities = json.loads(keywords_result.keywords)
                except:
                    # Fallback: extract basic entities from prompt
                    entities = []
                    if "blue" in prompt.lower():
                        entities.append("blue")
                    if "red" in prompt.lower():
                        entities.append("red")
                    if "shed" in prompt.lower() or "building" in prompt.lower():
                        entities.append("building")
                    if "tin" in prompt.lower():
                        entities.append("tin")
                    if "roof" in prompt.lower():
                        entities.append("roof")
                    
                # Return a dspy.Prediction object as per DSPy patterns
                return dspy.Prediction(
                    entities=entities
                )
        
        return ImprovedEntityExtractor()

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    clip_model, clip_transform = _load_clip(device)
    
    # Initialize DSPy extractor
    dspy_extractor = None
    try:
        import dspy
        # Try to configure DSPy with Ollama as the LLM
        try:
            # Use one of the available models
            lm = dspy.LM('ollama_chat/qwen3:8b', api_base='http://localhost:11434', api_key='')
            dspy.settings.configure(lm=lm)
            dspy_extractor = create_entity_extractor()
        except Exception as e:
            logger.warning(f"Ollama configuration failed: {e}, using fallback extraction")
            dspy_extractor = create_entity_extractor()
    except ImportError:
        logger.warning("DSPy not available, using fallback extraction")
        dspy_extractor = None
    
    # Extract entities from prompt
    entities = []
    if dspy_extractor is not None:
        try:
            entities_prediction = dspy_extractor(prompt=prompt)
            entities = entities_prediction.entities
        except Exception as e:
            logger.warning(f"DSPy extraction failed: {e}, using fallback extraction")
        
        # Fallback if no entities extracted
        if not entities or not isinstance(entities, list):
            entities = []
    
    # Add fallback if no entities were extracted via DSPy
    if not entities:
        # Simple regex-based entity extraction with better handling of combined terms
        prompt_lower = prompt.lower()
        
        # Extract color + object combinations first
        import re
        # Look for patterns like "blue shed", "red building", etc.
        color_obj_pattern = r'(blue|red|green|yellow|orange|purple|pink|brown|black|white|gray|grey)\s+(tin\s+)?(shed|building|roof|hut|cabin|structure)'
        color_obj_matches = re.findall(color_obj_pattern, prompt_lower)
        
        for match in color_obj_matches:
            color = match[0]
            tin_part = match[1] if match[1] else ""
            obj_part = match[2]
            combined_entity = f"{color} {tin_part}{obj_part}".strip()
            if combined_entity not in entities:
                entities.append(combined_entity)
        
        # Add individual components if not already added
        if "blue" in prompt_lower and not any("blue" in ent for ent in entities):
            entities.append("blue")
        if "red" in prompt_lower and not any("red" in ent for ent in entities):
            entities.append("red")
        if "shed" in prompt_lower and not any("shed" in ent for ent in entities):
            entities.append("shed")
        if "tin" in prompt_lower and not any("tin" in ent for ent in entities):
            entities.append("tin")
        if "building" in prompt_lower and not any("building" in ent for ent in entities):
            entities.append("building")
        if "village" in prompt_lower and not any("village" in ent for ent in entities):
            entities.append("village")
        if "roof" in prompt_lower and not any("roof" in ent for ent in entities):
            entities.append("roof")
    
    logger.info(f"Extracted entities: {entities}")
    
    # Create a list of prompts for each entity
    prompts = [f"{entity}" for entity in entities]
    
    # Tokenize all prompts
    text_tokens = open_clip.tokenize(prompts).to(device)
    
    # Get text embeddings
    with torch.no_grad():
        text_emb = clip_model.encode_text(text_tokens)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
    
    # Create a grid of potential regions to check
    height, width = image.shape[:2]
    region_size = min(height // 4, width // 4)  # Use quarter-sized regions for efficiency
    
    detections = {}
    
    # Check overlapping regions across the image
    for y in range(0, height - region_size, region_size // 2):
        for x in range(0, width - region_size, region_size // 2):
            # Crop the region
            crop = image[y:y+region_size, x:x+region_size]
            pil = Image.fromarray(crop)
            inp = clip_transform(pil).unsqueeze(0).to(device)
            
            # Get image embedding
            with torch.no_grad():
                img_emb = clip_model.encode_image(inp)
                img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            
            # Calculate similarities with all text embeddings
            sims = (text_emb @ img_emb.T).squeeze().cpu().numpy()
            
            # For each entity with similarity above threshold, add detection
            for i, entity in enumerate(entities):
                if sims[i] > 0.2:  # Threshold for detection
                    if entity not in detections:
                        detections[entity] = []
                    
                    # Add bounding box (convert to original image coordinates)
                    bbox = (x, y, x + region_size, y + region_size)
                    detections[entity].append({
                        'bbox': bbox,
                        'score': float(sims[i]),
                        'source': 'clip'
                    })
    
    logger.info(f"Found {len(detections)} entities in image")
    
    return detections