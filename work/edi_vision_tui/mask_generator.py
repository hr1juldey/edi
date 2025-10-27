"""
Mask generation functionality for EDI vision subsystem.
Based on advanced_mask_generator.py from example_code/Image_analysis/
"""

import cv2
import numpy as np
import torch
from PIL import Image
import open_clip
from ultralytics import SAM, YOLO
from typing import List, Tuple, Dict
import re
import logging
from functools import lru_cache
import json
import dspy
import math

# Import our new pipeline modules
from pipeline.stage1_clip.clip_entity_detector import get_entity_bounding_boxes
from pipeline.stage2_yolo.yolo_refiner import YOLORefiner
from pipeline.stage3_sam.sam_integration import SAMIntegration


def load_image(path):
    """Load image from path, converting BGR to RGB"""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


@lru_cache(maxsize=1)
def _load_clip(device):
    """
    Load CLIP model + transform once and cache it.
    Tries multiple pretrained tags and logs each attempt.
    """
    logger = logging.getLogger("_load_clip")
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

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


def validate_building_region(image, x1, y1, x2, y2, min_height=50, max_height_ratio=0.3):
    """
    Validate if a region is likely to contain a building based on its spatial properties.
    
    Args:
        image: Input image as numpy array
        x1, y1, x2, y2: Bounding box coordinates
        min_height: Minimum height for a valid building region
        max_height_ratio: Maximum ratio of region height to image height
    """
    # Check if bounding box has reasonable dimensions
    width = x2 - x1
    height = y2 - y1
    
    if height < min_height or width < min_height:
        return False, "Region too small to be a building"
    
    # Check if region is at ground level (not in sky)
    # Buildings should be below 80% of image height
    image_height = image.shape[0]
    if y1 < image_height * 0.2:  # Top 20% of image is likely sky
        return False, "Region is in sky area"
    
    # Check if region is not too high in the image
    if y1 > image_height * 0.8:  # Bottom 20% is likely ground
        return False, "Region is too low to be a building"
    
    # Check if region height is not too large compared to image
    if height > image_height * max_height_ratio:
        return False, "Region is too large to be a single building"
    
    # Additional validation: check if region contains some texture (not just sky)
    # Calculate mean color and variance in the region
    region = image[y1:y2, x1:x2]
    mean_color = np.mean(region, axis=(0,1))
    std_color = np.std(region, axis=(0,1))
    
    # If all colors are very similar (low variance), it might be sky or water
    if np.mean(std_color) < 10:  # Low variance indicates uniform color
        return False, "Region has low color variance, likely sky"
    
    return True, "Valid building region"


class ExtractTargetColor(dspy.Signature):
    """
    Extract the target color from a user prompt for image editing.
    """
    prompt = dspy.InputField(desc="Original user prompt for image editing")
    
    target_color = dspy.OutputField(desc="The target color mentioned in the prompt (e.g., 'red', 'blue', 'green')")
    

class AnalyzeImageForObjectDetection(dspy.Signature):
    """
    Analyze an image and identify specific objects that match the user prompt.
    This signature is used to guide the VLM in finding relevant objects in the image.
    """
    image_description = dspy.InputField(desc="Description of the image content")
    user_prompt = dspy.InputField(desc="User's editing prompt")
    
    objects_to_find = dspy.OutputField(desc="JSON list of specific objects to identify in the image")
    search_strategy = dspy.OutputField(desc="Strategy for finding these objects in the image")


class ExtractKeywords(dspy.Signature):
    """
    Extract relevant keywords from a user prompt for image editing.
    Focus on colors, objects, and specific entities that need to be identified in the image.
    """
    prompt = dspy.InputField(desc="Original user prompt for image editing")
    
    keywords = dspy.OutputField(desc="JSON list of relevant keywords for image search (colors, objects, entities)")


class ImprovedKeywordExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extract_keywords = dspy.ChainOfThought(ExtractKeywords)
        self.extract_target_color = dspy.ChainOfThought(ExtractTargetColor)
        self.analyze_for_detection = dspy.ChainOfThought(AnalyzeImageForObjectDetection)
    
    def forward(self, prompt, image_context=""):
        # Extract general keywords
        keywords_result = self.extract_keywords(prompt=prompt)
        # Extract target color for color change operations
        color_result = self.extract_target_color(prompt=prompt)
        # Analyze what objects to look for in the image
        detection_result = self.analyze_for_detection(
            image_description=image_context or "An image that may contain buildings, structures, or other objects",
            user_prompt=prompt
        )
        
        # Parse the keywords from the JSON output
        try:
            import json
            keywords = json.loads(keywords_result.keywords)
        except:
            # Fallback: extract basic keywords
            keywords = [color_result.target_color] if color_result.target_color else []
        
        # Add the target color to keywords if not already present
        if color_result.target_color and color_result.target_color not in keywords:
            keywords.append(color_result.target_color)
        
        # Return a dspy.Prediction object as per DSPy patterns
        return dspy.Prediction(
            keywords=keywords, 
            target_color=color_result.target_color,
            objects_to_find=detection_result.objects_to_find,
            search_strategy=detection_result.search_strategy
        )


# Global instance of the DSPy extractor
_dspy_extractor = None


def decompose_prompt(prompt: str) -> List[str]:
    """
    Break complex prompts into atomic components using DSPy for improved accuracy.
    
    "blue roof and clouds" -> ["blue roof", "clouds"]
    "change red door to green" -> ["door", "green"]
    """
    global _dspy_extractor
    
    # Initialize DSPy extractor if not already done
    if _dspy_extractor is None:
        try:
            import dspy
            # Try to configure DSPy with Ollama as the LLM
            
            # Use a model that is available according to 'ollama list'
            try:
                # Use one of the available models
                lm = dspy.LM('ollama_chat/qwen3:8b', api_base='http://localhost:11434', api_key='')
                dspy.settings.configure(lm=lm)
                _dspy_extractor = ImprovedKeywordExtractor()
            except Exception as e:
                print(f"[!] Ollama configuration failed: {e}, trying alternative model")
                try:
                    # Try another available model
                    lm = dspy.LM('ollama_chat/qwen2.5vl:7b', api_base='http://localhost:11434', api_key='')
                    dspy.settings.configure(lm=lm)
                    _dspy_extractor = ImprovedKeywordExtractor()
                except Exception as e2:
                    print(f"[!] Alternative Ollama configuration failed: {e2}, using fallback extraction")
                    return _fallback_decompose_prompt(prompt)
        except ImportError:
            print("[!] DSPy not available, using fallback extraction")
            return _fallback_decompose_prompt(prompt)
    
    if _dspy_extractor is not None:
        try:
            # Use DSPy to extract keywords - call the module directly, not forward
            keywords_prediction = _dspy_extractor(prompt=prompt)
            # The result should be a Prediction object with attributes
            if hasattr(keywords_prediction, 'keywords'):
                return keywords_prediction.keywords
            else:
                # If the attribute doesn't exist, try to access it differently
                return keywords_prediction
        except Exception as e:
            print(f"[!] DSPy extraction failed: {e}, using fallback extraction")
    else:
        print("[!] DSPy extractor not initialized, using fallback extraction")
    
    # Use fallback if DSPy fails
    return _fallback_decompose_prompt(prompt)


def _fallback_decompose_prompt(prompt: str) -> List[str]:
    """
    Fallback prompt decomposition using regex patterns when DSPy is unavailable.
    """
    # Common edit instruction patterns to remove
    edit_patterns = [
        r'change\s+.*?\s+to\s+',
        r'make\s+.*?\s+',
        r'turn\s+.*?\s+into\s+',
        r'convert\s+.*?\s+to\s+',
        r'edit\s+',
        r'modify\s+',
        r'update\s+',
        r'alter\s+',
        r'transform\s+',
        r'recolor\s+',
        r'repaint\s+',
        r'switch\s+'  # Added more edit verbs
    ]
    
    cleaned = prompt.lower()
    for pattern in edit_patterns:
        cleaned = re.sub(pattern, '', cleaned)
    
    # First, look for color + object combinations (like "blue roof", "green house")
    color_pattern = r'(?:red|blue|green|yellow|orange|purple|pink|brown|black|white|gray|grey|light|dark)\s+(?:roof|house|building|shed|structure|object|element)'
    color_objects = re.findall(color_pattern, cleaned)
    
    # Look for specific objects
    object_pattern = r'(?:roof|house|building|shed|structure|object|element|tile|tin|metal)'  # Added more specific terms
    objects = re.findall(object_pattern, cleaned)
    
    # Create a list of all extracted parts
    parts = []
    
    # Add color-object combinations first
    for color_obj in color_objects:
        parts.append(color_obj.strip())
    
    # Add other objects
    for obj in objects:
        # Only add if not already in parts
        if obj not in parts:
            parts.append(obj)
    
    # Also try splitting on common conjunctions as fallback
    if not parts:
        # Split on common conjunctions
        split_parts = re.split(r'\s+and\s+|\s+,\s+|\s+\+\s+', cleaned)
        
        # Filter out very short parts (noise)
        for p in split_parts:
            p_clean = p.strip()
            if len(p_clean) > 2:
                # Check if it contains any color or object terms
                has_color = any(color in p_clean for color in ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'brown', 'black', 'white', 'gray', 'grey'])
                has_object = any(obj in p_clean for obj in ['roof', 'house', 'building', 'shed', 'structure', 'object', 'element', 'tile', 'tin', 'metal'])
                if has_color or has_object:
                    parts.append(p_clean)
    
    # If still no parts found, use the original prompt
    if not parts:
        parts = [prompt]
    
    return parts



def get_topk_masks_by_clip(image, prompt, masks, k=5, device=None):
    """
    Return top-k masks ranked by CLIP similarity.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    clip_model, clip_transform = _load_clip(device)
    
    print(f"[DEBUG] Processing prompt: '{prompt}'")
    text_tokens = open_clip.tokenize([prompt]).to(device)
    with torch.no_grad():
        text_emb = clip_model.encode_text(text_tokens)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        print(f"[DEBUG] Text embedding shape: {text_emb.shape}")
    
    scores = []
    for i in range(masks.shape[0]):
        m = (masks[i] > 0.5).astype(np.uint8)
        ys, xs = np.where(m > 0)
        
        if ys.size == 0 or (ys.max() - ys.min()) < 8 or (xs.max() - xs.min()) < 8:
            scores.append((i, -99.0, m))
            continue
        
        y1, y2 = int(ys.min()), int(ys.max())
        x1, x2 = int(xs.min()), int(xs.max())
        
        print(f"[DEBUG] Processing mask {i} with bbox ({x1}, {y1}, {x2}, {y2})")
        
        crop = image[y1:y2+1, x1:x2+1]
        pil = Image.fromarray(crop)
        inp = clip_transform(pil).unsqueeze(0).to(device)
        
        with torch.no_grad():
            img_emb = clip_model.encode_image(inp)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            sim = float((text_emb @ img_emb.T).cpu().item())
            
        scores.append((i, sim, m))
        
        print(f"[DEBUG] Mask {i} similarity score: {sim:.4f}")
    
    # Sort by similarity, take top-k
    scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)
    topk = scores_sorted[:min(k, len(scores_sorted))]
    
    # Filter out masks with low scores (below 0.2 threshold)
    topk_filtered = [(idx, sim, mask) for idx, sim, mask in topk if sim > 0.2]
    
    print(f"[DEBUG] Top {len(topk)} masks selected with scores: {[f'{s:.4f}' for _, s, _ in topk]}")
    print(f"[DEBUG] After threshold filtering: {len(topk_filtered)} masks remain")
    
    return topk_filtered  # Return filtered results


def merge_masks_with_threshold(masks_list: List[np.ndarray], 
                                scores_list: List[float],
                                threshold=0.3) -> np.ndarray:
    """
    Merge multiple masks using UNION operation (not weighted average).
    Only include masks with score > threshold.
    
    Key change: Use OR operation instead of weighted average to preserve all regions.
    """
    if not masks_list:
        return None
    
    # Normalize scores
    scores_array = np.array(scores_list)
    if scores_array.max() > 0:
        scores_array = scores_array / scores_array.max()
    
    # FIXED: Use union (OR) operation instead of weighted average
    # This preserves all mask regions instead of blending them
    combined = np.zeros_like(masks_list[0], dtype=np.uint8)
    
    for mask, score in zip(masks_list, scores_array):
        if score > threshold:
            # Union: any pixel that's 1 in any high-scoring mask becomes 1
            combined = np.logical_or(combined, mask).astype(np.uint8)
    
    print(f"    [DEBUG] After union: {combined.sum()} pixels ({combined.sum()/combined.size:.4%} coverage)")
    
    return combined


def get_advanced_mask(image, prompt, sam_checkpoint=None, 
                     topk=5, merge_threshold=0.3, device=None):
    """
    Advanced mask generation with multi-stage pipeline:
    1. Use CLIP to get entity bounding boxes
    2. Refine with YOLO for precise detection
    3. Generate masks with SAM 2.1 using YOLO's precise boxes
    4. Post-process (morphological operations)
    
    Args:
        image: Input image as numpy array
        prompt: User prompt describing what to mask
        sam_checkpoint: Path to SAM model checkpoint (default: auto-download)
        topk: Number of top masks to consider per prompt part
        merge_threshold: Threshold for including masks in merge
        device: Device to run on ('cuda' or 'cpu')
    """
    # Use CPU to avoid memory issues
    device = "cpu"  # Force CPU to prevent CUDA memory errors
    
    print("[+] Stage 1: Using CLIP to detect entities...")
    # Get entity bounding boxes from CLIP
    entity_boxes = get_entity_bounding_boxes(image, prompt, device=device)
    print(f"[+] CLIP detected {len(entity_boxes)} entities: {list(entity_boxes.keys())}")
    
    # If no entities detected by CLIP, fallback to full inference
    if not entity_boxes:
        print("[!] No entities detected by CLIP, using fallback method...")
        return get_advanced_mask_fallback(image, prompt, sam_checkpoint, topk, merge_threshold, device)
    
    print("[+] Stage 2: Refining bounding boxes with YOLO...")
    # Initialize YOLO refiner
    yolo_refiner = YOLORefiner()
    # Refine the bounding boxes using YOLO
    refined_bounding_boxes = yolo_refiner.refine_bounding_boxes(image, entity_boxes)
    print(f"[+] YOLO refined bounding boxes for {len(refined_bounding_boxes)} entities")
    
    print("[+] Stage 3: Generating masks with SAM using precise YOLO boxes...")
    # Initialize SAM integration
    sam_integration = SAMIntegration(sam_checkpoint)
    # Generate masks using SAM with YOLO's precise bounding boxes
    masks_results = sam_integration.generate_masks(image, refined_bounding_boxes)
    print(f"[+] SAM generated masks for {len(masks_results)} entities")
    
    if not masks_results:
        print("[!] No masks generated by SAM, using fallback method...")
        return get_advanced_mask_fallback(image, prompt, sam_checkpoint, topk, merge_threshold, device)
    
    # Combine masks from all entities
    print("[+] Combining masks from all entities...")
    combined_mask = sam_integration.combine_masks(masks_results, merge_threshold)
    
    # Step 4: Post-processing (morphological operations)
    print("[+] Post-processing mask...")
    final_mask = post_process_mask(combined_mask)
    
    coverage = final_mask.sum() / final_mask.size
    print(f"[+] Final mask covers {coverage:.2%} of image")
    
    return final_mask


def get_advanced_mask_fallback(image, prompt, sam_checkpoint=None, 
                              topk=5, merge_threshold=0.3, device=None):
    """
    Fallback method for mask generation using the original approach if the multi-stage pipeline fails.
    """
    print("[+] Running fallback SAM inference...")
    
    # Use SAM 2.1 for better precision (larger model)
    print("[+] Running SAM 2.1...")
    try:
        # Use the larger SAM 2.1 model for better precision
        sam = SAM("sam2.1_l.pt")  # Use the larger model as requested
    except Exception as e:
        print(f"[*] SAM 2.1_l.pt not available: {e}, trying SAM 2.1 base model...")
        try:
            sam = SAM("sam2.1_b.pt")
        except Exception as e2:
            print(f"[*] SAM 2.1 models not available: {e2}, falling back to original SAM...")
            sam = SAM('mobile_sam.pt')
    
    results = sam(image, verbose=False)
    masks_t = results[0].masks.data
    
    if masks_t is None or masks_t.shape[0] == 0:
        raise RuntimeError("SAM returned no masks")
    
    masks = masks_t.cpu().numpy()
    print(f"[+] SAM 2.1 generated {masks.shape[0]} candidate masks")
    
    # Step 2: Decompose prompt
    prompt_parts = decompose_prompt(prompt)
    print(f"[+] Prompt decomposed into: {prompt_parts}")
    
    # Step 3: For each prompt part, find top-k masks
    all_relevant_masks = []
    all_scores = []
    all_bboxes = []
    
    for part in prompt_parts:
        print(f"[+] Finding masks for: '{part}'")
        topk_results = get_topk_masks_by_clip(image, part, masks, k=topk, device=device)
        
        if topk_results:
            print(f"    Found {len(topk_results)} relevant masks (scores: {[f'{s:.3f}' for _, s, _ in topk_results[:3]]})")
            for idx, sim, mask in topk_results:
                all_relevant_masks.append(mask)
                all_scores.append(sim)
                
                # Get the bounding box for this mask
                ys, xs = np.where(mask > 0)
                if len(ys) > 0 and len(xs) > 0:
                    y1, y2 = int(ys.min()), int(ys.max())
                    x1, x2 = int(xs.min()), int(xs.max())
                    all_bboxes.append((x1, y1, x2, y2))
                    print(f"      Mask region: ({x1}, {y1}) to ({x2}, {y2}), size: {(x2-x1)*(y2-y1)} pixels")
    
    if not all_relevant_masks:
        print("[!] Warning: No relevant masks found, using best single mask")
        # Fallback to original single-mask approach
        topk_results = get_topk_masks_by_clip(image, prompt, masks, k=1, device=device)
        if topk_results:
            return topk_results[0][2]
        else:
            return (masks[0] > 0.5).astype(np.uint8)
    
    # Step 4: Merge masks
    print(f"[+] Merging {len(all_relevant_masks)} masks...")
    merged_mask = merge_masks_with_threshold(all_relevant_masks, all_scores, 
                                            threshold=merge_threshold)
    
    # Step 5: Post-processing (morphological operations)
    print("[+] Post-processing mask...")
    final_mask = post_process_mask(merged_mask)
    
    coverage = final_mask.sum() / final_mask.size
    print(f"[+] Final mask covers {coverage:.2%} of image")
    
    return final_mask


def analyze_image_with_ollama_vlm(image, prompt):
    """
    Analyze the image using Ollama-based VLM to identify objects relevant to the prompt.
    This function sends the image to the Ollama VLM and gets a JSON response.
    """
    try:
        import json
        import requests
        
        # Since the direct dspy.OllamaLocal approach failed, use the correct Ollama API approach
        # Prepare the image for API call
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3:  # Color image
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:  # Grayscale
                pil_image = Image.fromarray(image)
        
        # Convert image to base64 for API
        import base64
        from io import BytesIO
        
        buffer = BytesIO()
        pil_image.save(buffer, format='JPEG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Prepare the request to Ollama API
        ollama_url = "http://localhost:11434/api/generate"
        
        # Use models that are actually available based on 'ollama list' command
        models_to_try = ["qwen2.5vl:7b", "qwen3:8b", "qwen3:30b", "gemma3:4b", "mistral:7b", "gemma3n:e4b"]
        
        # Get better context from DSPy if available
        objects_to_find = []
        search_strategy = "Look for buildings, structures, and objects that match the user's request"
        
        # Try to get better object detection guidance from DSPy
        try:
            dspy_keywords = decompose_prompt(prompt)
            if dspy_keywords:
                objects_to_find = dspy_keywords
                search_strategy = f"Focus on finding objects related to: {', '.join(objects_to_find)}"
        except:
            pass
        
        for model in models_to_try:
            # Create a more detailed prompt asking for specific object locations
            vlm_prompt = f"""Analyze this image and identify the locations of objects that match this request: "{prompt}".

Context:
- Focus on objects related to: {', '.join(objects_to_find) if objects_to_find else 'the main subject'}
- Search strategy: {search_strategy}
- Important: Buildings and structures should be at ground level or on hillsides, not in the sky
- Only return objects that appear to be actual physical structures (buildings, sheds, houses, etc.)
- Reject any detections that appear to be sky regions or clouds

Respond in JSON format with 'instances' as a list of objects, each with:
- 'name': descriptive name of the object
- 'bbox': [x1, y1, x2, y2] bounding box coordinates
- 'confidence': confidence score between 0.0 and 1.0
- 'type': classification of the object ('building', 'structure', 'other')

Respond ONLY with the JSON, no other text. Example format:
{{"instances": [{{"name": "blue tin shed", "bbox": [100, 200, 300, 400], "confidence": 0.95, "type": "building"}}]}}"""
            
            data = {
                "model": model,
                "prompt": vlm_prompt,
                "images": [img_base64],
                "format": "json",
                "stream": False
            }
            
            response = requests.post(ollama_url, json=data)
            
            if response.status_code == 200:
                result = response.json()
                if 'response' in result:
                    # Parse the response which should be in JSON format
                    try:
                        analysis = json.loads(result['response'])
                        print(f"[+] VLM successfully analyzed image using model {model}")
                        return analysis
                    except json.JSONDecodeError:
                        # If JSON parsing fails, try the next model
                        continue
            elif response.status_code == 404:
                print(f"[!] Model {model} not found, trying next model...")
                continue
            else:
                print(f"[!] VLM analysis failed with status {response.status_code} for model {model}: {response.text}")
        
        print("[!] All VLM models failed, returning empty analysis")
        
    except Exception as e:
        print(f"[!] VLM analysis failed: {e}, returning empty analysis")
    
    # Return a structure with empty instances list if API fails
    return {"instances": []}


def generate_multiple_instance_masks(image, prompt, sam_checkpoint=None, device=None):
    """
    Generate multiple smaller masks around individual target instances with names and confidence.
    This function uses SAM 2.1 for instance segmentation and VLM for object identification.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    # Use Ollama VLM to identify objects in the image
    print("[+] Analyzing image with Ollama VLM for instance detection...")
    image_analysis = analyze_image_with_ollama_vlm(image, prompt)
    
    # Use SAM 2.1 model for precise segmentation
    print("[+] Running SAM 2.1 for instance segmentation...")
    try:
        sam = SAM("sam2.1_l.pt")  # Use the larger model for better precision
    except Exception as e:
        print(f"[*] SAM 2.1_l.pt not available: {e}, trying base model...")
        try:
            sam = SAM("sam2.1_b.pt")
        except Exception as e2:
            print(f"[*] SAM 2.1 models not available: {e2}, falling back...")
            sam = SAM('mobile_sam.pt')
    
    instance_masks = []
    
    # If VLM provided specific instances
    if image_analysis and "instances" in image_analysis:
        for i, instance in enumerate(image_analysis["instances"]):
            if "bbox" in instance:
                x1, y1, x2, y2 = instance["bbox"]
                
                # Get mask for this specific bounding box
                results = sam(image, bboxes=[[x1, y1, x2, y2]], verbose=False)
                mask_data = results[0].masks.data
                
                if mask_data is not None and mask_data.shape[0] > 0:
                    # Convert to numpy and get the first mask
                    mask = mask_data[0].cpu().numpy()
                    
                    # Validate the mask
                    mask_binary = (mask > 0.5).astype(np.uint8)
                    if mask_binary.sum() > 100:  # At least 100 pixels
                        instance_info = {
                            'id': f'instance_{i}',
                            'object_name': instance.get('name', 'unknown'),
                            'confidence': instance.get('confidence', 0.5),
                            'mask': mask_binary,
                            'bbox': (x1, y1, x2, y2)
                        }
                        instance_masks.append(instance_info)
    
    return instance_masks


def post_process_mask(mask, min_area=50):
    """
    Clean up mask with morphological operations.
    - Remove small isolated regions
    - Fill small holes
    - Smooth boundaries
    """
    # Remove small components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    
    for i in range(1, num_labels):  # skip background (0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == i] = 1
    
    # Morphological closing (fill small holes)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    
    # Slight dilation to ensure coverage
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.dilate(cleaned, kernel_dilate, iterations=1)
    
    return cleaned.astype(np.uint8)


def generate_mask_for_prompt(image_path: str, prompt: str) -> dict:
    """
    Main function to generate a mask for a given image and prompt.
    Returns a dictionary with mask information.
    """
    try:
        # Load image
        image = load_image(image_path)
        
        # Generate mask using advanced mask generator
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mask = get_advanced_mask(image, prompt, device=device)
        
        # Calculate bounding box for the mask
        ys, xs = np.where(mask > 0)
        if len(ys) > 0 and len(xs) > 0:
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            
            # Add some padding to the bounding box to ensure full coverage
            padding = 10
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(image.shape[1], x2 + padding)
            y2 = min(image.shape[0], y2 + padding)
            
            bbox = (x1, y1, x2, y2)
        else:
            bbox = (0, 0, 0, 0)  # Default if no mask found
        
        # Calculate coverage percentage
        coverage = mask.sum() / mask.size if mask.size > 0 else 0.0
        
        return {
            'mask': mask,
            'bbox': bbox,
            'coverage': coverage,
            'success': True,
            'error': None
        }
        
    except Exception as e:
        return {
            'mask': None,
            'bbox': (0, 0, 0, 0),
            'coverage': 0.0,
            'success': False,
            'error': str(e)
        }