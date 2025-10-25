"""
Interface with Vision Language Models for image analysis.
"""

import asyncio
from ollama import AsyncClient # type: ignore
from typing import List, Optional


async def analyze_image_with_vlm(
    base64_image: str, 
    prompt: str, 
    model: str = "gemma3:4b"
) -> str:
    """
    Send an image to a VLM and retrieve its description.
    
    Args:
        base64_image (str): Base64 encoded image
        prompt (str): Prompt to send to the VLM
        model (str): VLM model to use
        
    Returns:
        str: VLM's description of the image
    """
    message = {
        "role": "user",
        "content": prompt,
        "images": [base64_image]
    }
    
    try:
        response = await AsyncClient().chat(model=model, messages=[message])
        return response['message']['content']
    except Exception as e:
        raise RuntimeError(f"Failed to analyze image with VLM: {e}")


async def batch_analyze_images(
    base64_images: List[str], 
    prompt: str, 
    model: str = "gemma3:4b"
) -> List[str]:
    """
    Batch process multiple images with a VLM.
    
    Args:
        base64_images (List[str]): List of base64 encoded images
        prompt (str): Prompt to send to the VLM
        model (str): VLM model to use
        
    Returns:
        List[str]: List of VLM descriptions for each image
    """
    tasks = [
        analyze_image_with_vlm(img, prompt, model) 
        for img in base64_images 
        if img is not None
    ]
    
    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions in results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"[WARN] Failed to analyze image {i}: {result}")
                processed_results.append("")
            else:
                processed_results.append(result)
                
        return processed_results
    except Exception as e:
        raise RuntimeError(f"Failed to batch analyze images: {e}")


async def analyze_image_with_retry(
    base64_image: str, 
    prompt: str, 
    model: str = "gemma3:4b",
    max_retries: int = 3
) -> str:
    """
    Send an image to a VLM with retry logic.
    
    Args:
        base64_image (str): Base64 encoded image
        prompt (str): Prompt to send to the VLM
        model (str): VLM model to use
        max_retries (int): Maximum number of retry attempts
        
    Returns:
        str: VLM's description of the image
    """
    for attempt in range(max_retries):
        try:
            return await analyze_image_with_vlm(base64_image, prompt, model)
        except Exception as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Failed to analyze image after {max_retries} attempts: {e}")
            print(f"[WARN] Attempt {attempt + 1} failed, retrying... ({e})")
            await asyncio.sleep(2 ** attempt)  # Exponential backoff


def format_vlm_prompt_for_entity_extraction(image_description: Optional[str] = None) -> str:
    """
    Format a prompt specifically for entity extraction from images.
    
    Args:
        image_description (Optional[str]): Additional context about the image
        
    Returns:
        str: Formatted prompt for entity extraction
    """
    base_prompt = """Systematically analyze this image and provide a comprehensive description:

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

    if image_description:
        base_prompt += f"\n\nAdditional context: {image_description}"
    
    return base_prompt


def format_vlm_prompt_for_scene_understanding() -> str:
    """
    Format a prompt for general scene understanding.
    
    Returns:
        str: Formatted prompt for scene understanding
    """
    return """Provide a detailed description of this image including:
- Overall scene type and context
- Main subjects and objects
- Color palette and lighting
- Composition and perspective
- Mood and atmosphere
- Any text or annotations visible in the image"""


async def compare_images_with_vlm(
    base64_image1: str,
    base64_image2: str,
    prompt: str,
    model: str = "gemma3:4b"
) -> str:
    """
    Compare two images using a VLM.
    
    Args:
        base64_image1 (str): Base64 encoded first image
        base64_image2 (str): Base64 encoded second image
        prompt (str): Comparison prompt
        model (str): VLM model to use
        
    Returns:
        str: VLM's comparison of the two images
    """
    message = {
        "role": "user",
        "content": prompt,
        "images": [base64_image1, base64_image2]
    }
    
    try:
        response = await AsyncClient().chat(model=model, messages=[message])
        return response['message']['content']
    except Exception as e:
        raise RuntimeError(f"Failed to compare images with VLM: {e}")