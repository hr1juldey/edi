"""
Mask generation functionality for EDI vision subsystem.
Based on advanced_mask_generator.py from example_code/Image_analysis/
"""

import cv2
import numpy as np
import torch
from PIL import Image
import open_clip
from ultralytics import SAM
from typing import List
import re
import logging
from functools import lru_cache
import dspy


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


# Define a DSPy signature for improved keyword extraction
class ExtractKeywords(dspy.Signature):
    """
    Extract relevant keywords from a user prompt for image editing.
    Focus on colors, objects, and specific entities that need to be identified in the image.
    """
    prompt = dspy.InputField(desc="Original user prompt for image editing")
    
    keywords = dspy.OutputField(desc="JSON list of relevant keywords for image search (colors, objects, entities)")


class ExtractTargetColor(dspy.Signature):
    """
    Extract the target color from a user prompt for image editing.
    """
    prompt = dspy.InputField(desc="Original user prompt for image editing")
    
    target_color = dspy.OutputField(desc="The target color mentioned in the prompt (e.g., 'red', 'blue', 'green')")
    

class ImprovedKeywordExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extract_keywords = dspy.ChainOfThought(ExtractKeywords)
        self.extract_target_color = dspy.ChainOfThought(ExtractTargetColor)
    
    def forward(self, prompt):
        # Extract general keywords
        keywords_result = self.extract_keywords(prompt=prompt)
        # Extract target color for color change operations
        color_result = self.extract_target_color(prompt=prompt)
        
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
        
        return keywords


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
            
            # Correct DSPy 3.0 Ollama integration
            try:
                lm = dspy.LM('ollama_chat/qwen2:7b-instruct-q4_K_M', api_base='http://localhost:11434', api_key='')
                dspy.settings.configure(lm=lm)
                _dspy_extractor = ImprovedKeywordExtractor()
            except Exception as e:
                print(f"[!] Ollama configuration failed: {e}, using fallback extraction")
                return _fallback_decompose_prompt(prompt)
        except ImportError:
            print("[!] DSPy not available, using fallback extraction")
            return _fallback_decompose_prompt(prompt)
    
    if _dspy_extractor is not None:
        try:
            # Use DSPy to extract keywords
            keywords = _dspy_extractor.forward(prompt)
            return keywords
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
    Advanced mask generation with multi-region support.
    
    Strategy:
    1. Decompose prompt into atomic parts
    2. For each part, get top-k SAM masks via CLIP
    3. Merge all relevant masks
    4. Post-process (morphological operations)
    
    Args:
        image: Input image as numpy array
        prompt: User prompt describing what to mask
        sam_checkpoint: Path to SAM model checkpoint (default: auto-download)
        topk: Number of top masks to consider per prompt part
        merge_threshold: Threshold for including masks in merge
        device: Device to run on ('cuda' or 'cpu')
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    # Step 1: Run SAM - try mobile_sam first, fallback to sam_b.pt
    print("[+] Running SAM...")
    try:
        # Use the ultralytics SAM model without specifying a checkpoint
        # This will use the model's built-in default checkpoint
        sam = SAM('mobile_sam.pt')  # Try using mobile SAM which is smaller and more reliable
        print("[+] Using mobile_sam.pt")
    except Exception as e:
        # Fallback to the default SAM model
        print(f"[*] Mobile SAM not available: {e}, trying default...")
        sam = SAM('sam_b.pt')
        print("[+] Using sam_b.pt")
    
    results = sam(image, verbose=False)
    masks_t = results[0].masks.data
    
    if masks_t is None or masks_t.shape[0] == 0:
        raise RuntimeError("SAM returned no masks")
    
    masks = masks_t.cpu().numpy()
    print(f"[+] SAM generated {masks.shape[0]} candidate masks")
    
    # Step 2: Decompose prompt
    prompt_parts = decompose_prompt(prompt)
    print(f"[+] Prompt decomposed into: {prompt_parts}")
    
    # Step 3: For each prompt part, find top-k masks
    all_relevant_masks = []
    all_scores = []
    
    for part in prompt_parts:
        print(f"[+] Finding masks for: '{part}'")
        topk_results = get_topk_masks_by_clip(image, part, masks, k=topk, device=device)
        
        if topk_results:
            print(f"    Found {len(topk_results)} relevant masks (scores: {[f'{s:.3f}' for _, s, _ in topk_results[:3]]})")
            for idx, sim, mask in topk_results:
                all_relevant_masks.append(mask)
                all_scores.append(sim)
                # Debug: Print details about the mask region
                ys, xs = np.where(mask > 0)
                if len(ys) > 0 and len(xs) > 0:
                    y1, y2 = int(ys.min()), int(ys.max())
                    x1, x2 = int(xs.min()), int(xs.max())
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
    merged_mask = post_process_mask(merged_mask)
    
    coverage = merged_mask.sum() / merged_mask.size
    print(f"[+] Final mask covers {coverage:.2%} of image")
    
    return merged_mask


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