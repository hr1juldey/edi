"""
Change detection functionality for EDI vision subsystem.
Compares input and output images to detect changes inside and outside masks.
"""

import cv2
import numpy as np
from PIL import Image
from skimage.color import rgb2lab, deltaE_ciede2000
from skimage.metrics import structural_similarity as ssim
from skimage.filters import gaussian
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import os


def load_image(path):
    """Load image from path, converting BGR to RGB"""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def delta_map_raw(imgA, imgB, blur_sigma=0.8):
    """Return raw CIEDE2000 ΔE map with noise reduction."""
    A_f = imgA.astype(np.float32) / 255.0
    B_f = imgB.astype(np.float32) / 255.0

    A_blur = gaussian(A_f, sigma=blur_sigma, channel_axis=-1)
    B_blur = gaussian(B_f, sigma=blur_sigma, channel_axis=-1)

    A_lab = rgb2lab((A_blur * 255).astype(np.uint8))
    B_lab = rgb2lab((B_blur * 255).astype(np.uint8))

    delta = deltaE_ciede2000(A_lab, B_lab)
    delta = np.nan_to_num(delta)
    return delta


def compute_adaptive_thresholds(delta, mask):
    """
    Compute adaptive thresholds based on statistical analysis of the delta map.
    
    Strategy:
    1. Analyze distribution of changes inside and outside mask
    2. Use Otsu-like method to find natural separation point
    3. Apply robust statistics (percentiles) to handle outliers
    """
    mask_bool = (mask == 1)
    inside = delta[mask_bool]
    outside = delta[~mask_bool]
    
    # Compute percentile-based thresholds (more robust than mean/std)
    # Use 95th percentile for "significant change" detection
    inside_95th = np.percentile(inside, 95) if inside.size > 0 else 0
    outside_95th = np.percentile(outside, 95) if outside.size > 0 else 0
    
    # Median for typical change level
    inside_median = np.median(inside) if inside.size > 0 else 0
    outside_median = np.median(outside) if outside.size > 0 else 0
    
    # Adaptive threshold: changes should be "significantly different"
    # Outside should be much lower than inside
    delta_threshold = max(
        6.0,  # minimum perceptual threshold (JND)
        outside_median + 2 * (outside_95th - outside_median)  # 2 sigma equivalent
    )
    
    # Dynamic tolerance based on actual distribution
    # Allow more outside change if editing is very subtle
    if inside_median < 10:  # subtle edit
        max_outside_ratio = 0.15  # allow 15%
    else:  # significant edit
        max_outside_ratio = 0.05  # allow 5%
    
    return {
        'delta_threshold': delta_threshold,
        'max_outside_ratio': max_outside_ratio,
        'inside_median': inside_median,
        'inside_95th': inside_95th,
        'outside_median': outside_median,
        'outside_95th': outside_95th
    }


def evaluate_with_adaptive_threshold(delta, mask):
    """
    Adaptive evaluation that automatically determines appropriate thresholds.
    """
    mask_bool = (mask == 1)
    inside = delta[mask_bool]
    outside = delta[~mask_bool]
    
    # Compute adaptive thresholds
    thresholds = compute_adaptive_thresholds(delta, mask)
    delta_thresh = thresholds['delta_threshold']
    max_outside_ratio = thresholds['max_outside_ratio']
    
    # Statistics
    mean_in = float(inside.mean()) if inside.size else 0.0
    mean_out = float(outside.mean()) if outside.size else 0.0
    median_in = float(np.median(inside)) if inside.size else 0.0
    median_out = float(np.median(outside)) if outside.size else 0.0
    
    # Changed pixel counts
    in_changed_count = float((inside > delta_thresh).sum()) if inside.size else 0.0
    out_changed_count = float((outside > delta_thresh).sum()) if outside.size else 0.0
    pct_in_changed = in_changed_count / (inside.size + 1e-8)
    pct_out_changed = out_changed_count / (outside.size + 1e-8)
    
    # Compute signal-to-noise ratio
    # Desired: high change inside, low change outside
    if mean_out > 0:
        snr = mean_in / mean_out
    else:
        snr = mean_in / 0.01  # avoid division by zero
    
    # Multi-criteria evaluation
    criteria = {
        'sufficient_inside_change': mean_in > delta_thresh / 2,
        'low_outside_change': pct_out_changed <= max_outside_ratio,
        'good_snr': snr > 2.0,  # inside changes should be 2x outside
        'median_separation': median_in > median_out * 1.5
    }
    
    # Count passing criteria
    passed = sum(criteria.values())
    total = len(criteria)
    
    # Verdict logic: need at least 3/4 criteria
    verdict = "PASS" if passed >= 3 else "FAIL"
    
    # Generate detailed explanation
    expl = (
        f"Adaptive: delta_thr={delta_thresh:.1f}, "
        f"mean_in={mean_in:.2f}, mean_out={mean_out:.2f}, "
        f"pct_out={pct_out_changed:.4f} (max={max_outside_ratio:.2f}), "
        f"SNR={snr:.2f}, criteria={passed}/{total}"
    )
    
    return {
        "verdict": verdict,
        "mean_in": mean_in,
        "mean_out": mean_out,
        "median_in": median_in,
        "median_out": median_out,
        "pct_in_changed": pct_in_changed,
        "pct_out_changed": pct_out_changed,
        "snr": snr,
        "adaptive_threshold": delta_thresh,
        "max_outside_ratio": max_outside_ratio,
        "criteria_passed": passed,
        "criteria_total": total,
        "criteria": criteria,
        "explanation": expl,
        "thresholds": thresholds
    }


def detect_changes_in_out_masks(image_path_before: str, 
                               image_path_after: str, 
                               masks: List[Dict]) -> Dict:
    """
    Detect changes between input and output images inside and outside masks.
    
    Args:
        image_path_before: Path to the original image
        image_path_after: Path to the edited image
        masks: List of mask dictionaries with 'bbox' and other properties
    
    Returns:
        Dictionary with change detection results
    """
    # Load both images
    img_before = load_image(image_path_before)
    img_after = load_image(image_path_after)
    
    # Resize after image to match before image dimensions if needed
    if img_before.shape != img_after.shape:
        img_after = cv2.resize(img_after, (img_before.shape[1], img_before.shape[0]))
    
    # Calculate perceptual delta map
    delta = delta_map_raw(img_before, img_after)
    
    # Create a combined mask from all individual masks
    combined_mask = np.zeros((img_before.shape[0], img_before.shape[1]), dtype=np.uint8)
    
    for mask_info in masks:
        x1, y1, x2, y2 = mask_info['bbox']
        # Ensure coordinates are within image bounds
        y1 = max(0, min(y1, img_before.shape[0] - 1))
        y2 = max(0, min(y2, img_before.shape[0] - 1))
        x1 = max(0, min(x1, img_before.shape[1] - 1))
        x2 = max(0, min(x2, img_before.shape[1] - 1))
        
        if y1 < y2 and x1 < x2:  # Valid bbox
            combined_mask[y1:y2, x1:x2] = 1
    
    # Evaluate changes with the combined mask
    evaluation = evaluate_with_adaptive_threshold(delta, combined_mask)
    
    # Additional analysis: count actual changed pixels inside/outside masks
    inside_pixels = combined_mask.sum()
    outside_pixels = (1 - combined_mask).sum()
    
    # Calculate alignment score
    # Higher score means changes are well-located where they should be (inside masks)
    if inside_pixels + outside_pixels > 0:
        inside_ratio = evaluation['pct_in_changed']
        outside_ratio = evaluation['pct_out_changed']
        
        # Alignment score: 1.0 if all changes are inside masks, 0.0 if all changes are outside
        alignment_score = max(0.0, min(1.0, (inside_ratio - outside_ratio + 1.0) / 2.0))
    else:
        alignment_score = 0.0
    
    # Count detected entities (for testing purposes)
    detected_entities = []
    for i, mask_info in enumerate(masks):
        # For each mask, check if there are significant changes in that region
        x1, y1, x2, y2 = mask_info['bbox']
        region_delta = delta[y1:y2, x1:x2]
        
        if region_delta.size > 0:
            avg_region_change = float(region_delta.mean())
            if avg_region_change > evaluation['adaptive_threshold']:
                detected_entities.append(f"entity_{i}")
    
    return {
        'alignment_score': alignment_score,
        'changes_inside': int(evaluation['pct_in_changed'] * inside_pixels),
        'changes_outside': int(evaluation['pct_out_changed'] * outside_pixels),
        'detected_entities': detected_entities,
        'preserved_entities': [],  # For now, we're not tracking preserved entities separately
        'unintended_changes': [] if evaluation['pct_out_changed'] <= evaluation['max_outside_ratio'] else ['unintended_outside_changes'],
        'evaluation_details': evaluation,
        'total_pixels_inside': int(inside_pixels),
        'total_pixels_outside': int(outside_pixels)
    }


def compare_output(image_path_before: str, 
                  image_path_after: str, 
                  expected_keywords: List[str], 
                  masks: List[Dict]) -> Dict:
    """
    Compare input and output images to detect changes inside/outside masks.
    
    Args:
        image_path_before: Path to original image
        image_path_after: Path to edited image
        expected_keywords: List of expected entities to be modified
        masks: List of mask dictionaries with 'bbox'
    
    Returns:
        Dictionary with comparison results
    """
    try:
        # Perform change detection
        change_results = detect_changes_in_out_masks(image_path_before, image_path_after, masks)
        
        # Add expected keywords to results
        change_results['expected_keywords'] = expected_keywords
        change_results['masks_used'] = len(masks)
        
        return change_results
    
    except Exception as e:
        return {
            'error': f"Change detection failed: {str(e)}",
            'alignment_score': 0.0,
            'changes_inside': 0,
            'changes_outside': 0,
            'detected_entities': [],
            'preserved_entities': [],
            'unintended_changes': ['processing_error']
        }