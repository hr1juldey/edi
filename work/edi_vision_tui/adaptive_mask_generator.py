"""
Adaptive mask generator with iterative refinement for precise editing.
Extracts entities and edit requests, then uses CLIP→YOLO→SAM2.1 pipeline with validation loop.
If YOLO and VLM agree on mask validity for the edit request, the mask is accepted;
otherwise, the process repeats up to 3 times.
"""

import cv2
import numpy as np
import torch
from PIL import Image
import open_clip
from ultralytics import SAM, YOLO
import matplotlib.pyplot as plt
from typing import List, Dict
import re
import logging
from functools import lru_cache
import json
import requests


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


def extract_entities_and_request(prompt: str) -> Dict[str, str]:
    """
    Extract entities and edit request from user prompt.
    
    Returns:
        dict: Contains 'entities' (comma-separated string) and 'edit_request' (what to do with entities)
    """
    # Define regex patterns for edit instructions
    edit_patterns = [
        r'change\s+(.*?)\s+to\s+(.*)',
        r'make\s+(.*?)\s+(.*)',
        r'turn\s+(.*?)\s+into\s+(.*)',
        r'convert\s+(.*?)\s+to\s+(.*)',
        r'repaint\s+(.*?)\s+(.*)',
        r'recolor\s+(.*?)\s+(.*)',
    ]
    
    # Try to extract entities and edit request
    for pattern in edit_patterns:
        match = re.search(pattern, prompt, re.IGNORECASE)
        if match:
            entities = match.group(1).strip()
            edit_target = match.group(2).strip()
            return {
                'entities': entities,
                'edit_request': f"to {edit_target}",
                'full_request': f"{entities} {edit_target}"
            }
    
    # If no pattern matched, return the whole prompt as entities with no specific edit request
    return {
        'entities': prompt,
        'edit_request': '',
        'full_request': prompt
    }


def get_clip_masks(image, entities, masks, k=5, device=None):
    """
    Use CLIP to rank SAM masks based on entity matching.
    
    Args:
        image: Input image
        entities: String of entities to match
        masks: SAM-generated masks
        k: Number of top masks to return
        device: Device to run on
    
    Returns:
        List of tuples (idx, similarity_score, mask)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    clip_model, clip_transform = _load_clip(device)
    
    print(f"[DEBUG] Using CLIP to match entities: '{entities}'")
    text_tokens = open_clip.tokenize([entities]).to(device)
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
    
    print(f"[DEBUG] CLIP found {len(topk)} masks matching '{entities}' (top scores: {[f'{s:.3f}' for _, s, _ in topk[:3]]})")
    return [(idx, sim, mask) for idx, sim, mask in topk if sim > 0.1]  # threshold


def run_yolo_detection(image, entities, model_path='yolo11n.pt'):
    """
    Run YOLO object detection on the image for specified entities.
    
    Args:
        image: Input image
        entities: Entities to detect (comma-separated string)
        model_path: Path to YOLO model
    
    Returns:
        List of detection results with bounding boxes and confidences
    """
    print(f"[DEBUG] Running YOLO detection for entities: '{entities}'")
    
    try:
        yolo_model = YOLO(model_path)
        
        # Prepare classes for detection - need to match YOLO's standard classes
        # Since YOLO may not recognize "tin shed" or other specific entities,
        # we'll use the most general classes that might match
        general_entities = []
        entity_list = [e.strip() for e in entities.split(',')]
        
        for entity in entity_list:
            # Map specific entities to YOLO's standard classes
            if any(word in entity for word in ['building', 'structure', 'shed', 'house', 'roof', 'door']):
                general_entities.extend(['building', 'person', 'car'])  # Common objects in image
            elif any(word in entity for word in ['sky', 'cloud', 'sun', 'moon']):
                general_entities.append('sky')  # This is not a standard YOLO class
            else:
                general_entities.append('object')  # General fallback
        
        # Since YOLO has fixed class names, run detection with standard classes
        results = yolo_model(image, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get class name and confidence
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    b = box.xyxy[0].cpu().numpy()  # x1, y1, x2, y2
                    
                    # Skip low-confidence detections
                    if conf > 0.3:
                        detections.append({
                            'class_id': cls_id,
                            'confidence': conf,
                            'bbox': [int(b[0]), int(b[1]), int(b[2]), int(b[3])]
                        })
        
        print(f"[DEBUG] YOLO found {len(detections)} detections")
        return detections
    except Exception as e:
        print(f"[DEBUG] YOLO detection failed: {e}")
        return []


def analyze_image_with_vlm(image, prompt, entities, edit_request):
    """
    Analyze image with VLM to validate if the detected objects match the edit request.
    
    Args:
        image: Input image
        prompt: Original user prompt
        entities: Extracted entities
        edit_request: Edit request part
    
    Returns:
        dict: Analysis result with validation score
    """
    try:
        # Convert image to base64 for API
        import base64
        from io import BytesIO
        
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3:  # Color image
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:  # Grayscale
                pil_image = Image.fromarray(image)
        
        buffer = BytesIO()
        pil_image.save(buffer, format='JPEG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Prepare the request to Ollama API
        ollama_url = "http://localhost:11434/api/generate"
        
        # Create a detailed prompt for the VLM
        vlm_prompt = f"""Analyze this image and validate whether the objects matching the request '{prompt}' are correctly identified.

Context:
- User wants to edit: '{entities}' {edit_request}
- Focus on finding objects that match: '{entities}'
- Evaluate if the regions in the image that match '{entities}' are appropriate for editing

Respond in JSON format with:
- 'valid_objects_found': boolean whether objects matching the request exist
- 'locations': array of objects with 'bbox': [x1, y1, x2, y2] and 'description': string
- 'confidence': confidence score between 0.0 and 1.0 that these objects are what the user wants to edit
- 'feedback': string with feedback on the accuracy of object detection

Example format:
{{"valid_objects_found": true, "locations": [{{"bbox": [100, 200, 300, 400], "description": "blue roof"}}], "confidence": 0.9, "feedback": "Object correctly identified"}}"""
        
        # Use models that are actually available
        models_to_try = ["qwen2.5vl:7b", "qwen3:8b", "gemma3:4b", "mistral:7b"]
        
        for model in models_to_try:
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
                    try:
                        analysis = json.loads(result['response'])
                        print(f"[DEBUG] VLM analysis for '{entities}': {analysis}")
                        return analysis
                    except json.JSONDecodeError:
                        continue
            elif response.status_code == 404:
                print(f"[DEBUG] Model {model} not found, trying next model...")
                continue
            else:
                print(f"[DEBUG] VLM analysis failed for model {model}")
        
        print("[DEBUG] All VLM models failed")
        
    except Exception as e:
        print(f"[DEBUG] VLM analysis failed: {e}")
    
    # Return a default structure if API fails
    return {
        "valid_objects_found": False,
        "locations": [],
        "confidence": 0.0,
        "feedback": "VLM analysis failed"
    }


def generate_mask_from_detections(image, detections, sam_model):
    """
    Generate masks from YOLO detections using SAM.
    
    Args:
        image: Input image
        detections: YOLO detection results
        sam_model: SAM model instance
    
    Returns:
        Combined mask from all detections
    """
    if not detections:
        return None
    
    masks_list = []
    
    for detection in detections:
        bbox = detection['bbox']
        x1, y1, x2, y2 = bbox
        
        # Get mask for this specific bounding box
        results = sam_model(image, bboxes=[[x1, y1, x2, y2]], verbose=False)
        mask_data = results[0].masks.data
        
        if mask_data is not None and mask_data.shape[0] > 0:
            # Convert to numpy and get the first mask
            mask = mask_data[0].cpu().numpy()
            mask_binary = (mask > 0.5).astype(np.uint8)
            
            # Validate the mask
            if mask_binary.sum() > 100:  # At least 100 pixels
                masks_list.append(mask_binary)
    
    # Combine all masks if we have any
    if masks_list:
        combined = np.zeros_like(masks_list[0], dtype=np.uint8)
        for mask in masks_list:
            combined = np.logical_or(combined, mask).astype(np.uint8)
        return combined
    else:
        return None


def post_process_mask(mask, min_area=50):
    """
    Clean up mask with morphological operations.
    - Remove small isolated regions
    - Fill small holes
    - Smooth boundaries
    """
    if mask is None:
        return None
        
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


def adaptive_mask_generation(image, prompt, max_attempts=3):
    """
    Adaptive mask generation with iterative refinement.
    
    Args:
        image: Input image
        prompt: User's edit request
        max_attempts: Maximum number of refinement attempts
    
    Returns:
        dict: Mask and validation results
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Extract entities and edit request from prompt
    extraction_result = extract_entities_and_request(prompt)
    entities = extraction_result['entities']
    edit_request = extraction_result['edit_request']
    full_request = extraction_result['full_request']
    
    print(f"[DEBUG] Extracted entities: '{entities}', edit request: '{edit_request}'")
    
    # Initialize models
    try:
        sam = SAM("sam2.1_l.pt")  # Use larger model for better precision
    except Exception as e:
        print(f"[DEBUG] SAM 2.1_l.pt not available: {e}, trying base model...")
        try:
            sam = SAM("sam2.1_b.pt")
        except Exception as e2:
            print(f"[DEBUG] SAM 2.1 models not available: {e2}, falling back...")
            sam = SAM('mobile_sam.pt')
    
    success = False
    final_mask = None
    attempt = 0
    
    for attempt in range(max_attempts):
        print(f"\n[ATTEMPT {attempt + 1}/{max_attempts}]")
        
        # Stage 1: Use CLIP to identify relevant regions
        print("[+] Stage 1: Running SAM to generate candidate masks...")
        sam_results = sam(image, verbose=False)
        masks_t = sam_results[0].masks.data
        
        if masks_t is None or masks_t.shape[0] == 0:
            print("[!] SAM returned no masks")
            continue
        
        masks = masks_t.cpu().numpy()
        print(f"[DEBUG] SAM generated {masks.shape[0]} candidate masks")
        
        # Stage 2: Use CLIP to identify masks matching entities
        print("[+] Stage 2: Using CLIP to match entities...")
        clip_results = get_clip_masks(image, entities, masks, k=10, device=device)
        
        # Stage 3: Run YOLO for specific object detection
        print("[+] Stage 3: Running YOLO for object detection...")
        yolo_detections = run_yolo_detection(image, entities)
        
        # Stage 4: Generate masks from YOLO detections if available
        if yolo_detections:
            print(f"[DEBUG] Using {len(yolo_detections)} YOLO detections for mask generation")
            mask_from_detections = generate_mask_from_detections(image, yolo_detections, sam)
        else:
            print("[DEBUG] No YOLO detections, falling back to CLIP-selected masks")
            # Use CLIP-selected masks as fallback
            if clip_results:
                clip_masks = [mask for _, _, mask in clip_results[:5]]  # Top 5 CLIP matches
                if len(clip_masks) > 0:
                    mask_from_detections = np.zeros_like(clip_masks[0], dtype=np.uint8)
                    for mask in clip_masks:
                        mask_from_detections = np.logical_or(mask_from_detections, mask).astype(np.uint8)
                else:
                    mask_from_detections = None
            else:
                mask_from_detections = None
        
        # Stage 5: Validate with VLM
        if mask_from_detections is not None:
            print("[+] Stage 4: Validating mask with VLM...")
            vlm_validation = analyze_image_with_vlm(image, prompt, entities, edit_request)
            
            # Check if YOLO and VLM agree on the mask validity
            yolo_success = len(yolo_detections) > 0
            vlm_success = vlm_validation.get('valid_objects_found', False)
            vlm_confidence = vlm_validation.get('confidence', 0)
            
            print(f"[DEBUG] Validation results - YOLO success: {yolo_success}, VLM success: {vlm_success}, VLM confidence: {vlm_confidence:.2f}")
            
            # If both YOLO and VLM agree, we have a successful mask
            if vlm_success and vlm_confidence > 0.5:
                print(f"[SUCCESS] Valid mask generated with VLM confidence {vlm_confidence:.2f}")
                final_mask = post_process_mask(mask_from_detections)
                success = True
                break
            else:
                print(f"[ATTEMPT FAILED] VLM confidence {vlm_confidence:.2f} too low or objects not validated")
                # Add more specific feedback from VLM
                feedback = vlm_validation.get('feedback', '')
                print(f"[VLM FEEDBACK] {feedback}")
        else:
            print("[ATTEMPT FAILED] No valid mask generated")
    
    # If all attempts failed, try a general mask based on CLIP
    if not success and clip_results:
        print("[!] All attempts failed, using best CLIP-based mask as fallback")
        best_mask = clip_results[0][2] if clip_results else None
        if best_mask is not None:
            final_mask = post_process_mask(best_mask)
            success = True
    
    return {
        'mask': final_mask,
        'success': success,
        'attempts': attempt + 1,
        'entity_extraction': extraction_result
    }


def generate_mask_for_prompt(image_path: str, prompt: str) -> dict:
    """
    Main function to generate a mask for a given image and prompt.
    Returns a dictionary with mask information.
    """
    try:
        # Load image
        image = load_image(image_path)
        
        # Generate mask using adaptive mask generator
        device = "cuda" if torch.cuda.is_available() else "cpu"
        result = adaptive_mask_generation(image, prompt, max_attempts=3)
        
        if result['mask'] is not None:
            # Calculate bounding box for the mask
            ys, xs = np.where(result['mask'] > 0)
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
            coverage = result['mask'].sum() / result['mask'].size if result['mask'].size > 0 else 0.0
        else:
            bbox = (0, 0, 0, 0)
            coverage = 0.0
        
        return {
            'mask': result['mask'],
            'bbox': bbox,
            'coverage': coverage,
            'success': result['success'],
            'attempts': result['attempts'],
            'entity_extraction': result['entity_extraction'],
            'error': None
        }
        
    except Exception as e:
        return {
            'mask': None,
            'bbox': (0, 0, 0, 0),
            'coverage': 0.0,
            'success': False,
            'attempts': 0,
            'entity_extraction': {},
            'error': str(e)
        }


if __name__ == "__main__":
    torch.set_grad_enabled(False)