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


def decompose_prompt(prompt: str) -> List[str]:
    """
    Break complex prompts into atomic components.
    
    "blue roof and clouds" -> ["blue roof", "clouds"]
    "change red door to green" -> ["door"]
    """
    # Common edit instruction patterns to remove
    edit_patterns = [
        r'change\s+.*?\s+to\s+',
        r'make\s+.*?\s+',
        r'turn\s+.*?\s+into\s+',
        r'convert\s+.*?\s+to\s+',
        r'edit\s+',
        r'modify\s+',
        r'update\s+'
    ]
    
    cleaned = prompt.lower()
    for pattern in edit_patterns:
        cleaned = re.sub(pattern, '', cleaned)
    
    # Split on common conjunctions
    parts = re.split(r'\s+and\s+|\s+,\s+|\s+\+\s+', cleaned)
    
    # Filter out very short parts (noise)
    parts = [p.strip() for p in parts if len(p.strip()) > 2]
    
    # If no split happened, return original
    if not parts:
        parts = [prompt]
    
    return parts


def get_topk_masks_by_clip(image, prompt, masks, k=5, device=None):
    """
    Return top-k masks ranked by CLIP similarity.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    clip_model, clip_transform = _load_clip(device)
    
    text_tokens = open_clip.tokenize([prompt]).to(device)
    with torch.no_grad():
        text_emb = clip_model.encode_text(text_tokens)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
    
    scores = []
    for i in range(masks.shape[0]):
        m = (masks[i] > 0.5).astype(np.uint8)
        ys, xs = np.where(m > 0)
        
        if ys.size == 0 or (ys.max() - ys.min()) < 8 or (xs.max() - xs.min()) < 8:
            scores.append((i, -99.0, m))
            continue
        
        y1, y2 = int(ys.min()), int(ys.max())
        x1, x2 = int(xs.min()), int(xs.max())
        
        crop = image[y1:y2+1, x1:x2+1]
        pil = Image.fromarray(crop)
        inp = clip_transform(pil).unsqueeze(0).to(device)
        
        with torch.no_grad():
            img_emb = clip_model.encode_image(inp)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            sim = float((text_emb @ img_emb.T).cpu().item())
        
        scores.append((i, sim, m))
    
    # Sort by similarity, take top-k
    scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)
    topk = scores_sorted[:min(k, len(scores_sorted))]
    
    return [(idx, sim, mask) for idx, sim, mask in topk if sim > 0.2]  # threshold


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
    
    # Use default SAM model if none provided
    if sam_checkpoint is None:
        sam_checkpoint = "sam_b.pt"  # Use the base model that was downloaded
    
    # Step 1: Run SAM
    print("[+] Running SAM...")
    sam = SAM(sam_checkpoint)
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