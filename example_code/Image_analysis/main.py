#!/usr/bin/env python3
"""
Adaptive validation for diffusion-based image editing.
Uses statistical methods to automatically determine thresholds and LMM-based scoring.

Usage:
    python adaptive_validator.py --before before.jpg --after after.jpg --prompt "roof"
"""

import argparse
from functools import lru_cache
import cv2
import numpy as np
import torch
from PIL import Image
import open_clip
from ultralytics import SAM
from skimage.color import rgb2lab, deltaE_ciede2000
from skimage.metrics import structural_similarity as ssim
from skimage.filters import gaussian
import matplotlib.pyplot as plt


# -------------------------
# Utilities
# -------------------------
def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


@lru_cache(maxsize=1)
def _load_clip(device):
    """Load CLIP model + transform once and cache it."""
    model_name = "ViT-B-32"
    tried = []
    for pretrained in ("openai", "laion2b_s34b_b79k", "laion400m_e32"):
        try:
            clip_model, clip_transform, _ = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained
            )
            clip_model.to(device)
            clip_model.eval()
            return clip_model, clip_transform
        except Exception as e:
            tried.append((pretrained, str(e)))
            continue
    msg = "open_clip.create_model_and_transforms failed"
    raise RuntimeError(msg)


def get_sam_mask(image, prompt, sam_checkpoint="sam2.1_b.pt", device=None):
    """Generate mask using SAM + CLIP similarity scoring."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    sam = SAM(sam_checkpoint)
    results = sam(image, verbose=False)
    masks_t = results[0].masks.data
    if masks_t is None or masks_t.shape[0] == 0:
        raise RuntimeError("SAM returned no masks.")

    masks = masks_t.cpu().numpy()
    clip_model, clip_transform = _load_clip(device)

    text_tokens = open_clip.tokenize([prompt]).to(device)
    with torch.no_grad():
        text_emb = clip_model.encode_text(text_tokens)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

    sims = []
    for i in range(masks.shape[0]):
        m = (masks[i] > 0.5).astype(np.uint8)
        ys, xs = np.where(m > 0)
        if ys.size == 0:
            sims.append((i, -99.0))
            continue
        y1, y2 = int(ys.min()), int(ys.max())
        x1, x2 = int(xs.min()), int(xs.max())
        if (y2 - y1) < 8 or (x2 - x1) < 8:
            sims.append((i, -99.0))
            continue
        crop = image[y1:y2 + 1, x1:x2 + 1]
        pil = Image.fromarray(crop)
        inp = clip_transform(pil).unsqueeze(0).to(device)
        with torch.no_grad():
            img_emb = clip_model.encode_image(inp)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            sim = float((text_emb @ img_emb.T).cpu().squeeze().item())
        sims.append((i, sim))

    sims_sorted = sorted(sims, key=lambda x: x[1], reverse=True)
    best_idx = sims_sorted[0][0]
    chosen_mask = (masks[best_idx] > 0.5).astype(np.uint8)
    
    return chosen_mask


# -------------------------
# Adaptive evaluation methods
# -------------------------
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


def compute_confidence_score(stats):
    """
    Compute a confidence score (0-100) for the verdict.
    Higher score = more confident in the result.
    """
    # Factors that increase confidence:
    # 1. High SNR (clear separation)
    # 2. Low outside change
    # 3. Sufficient inside change
    
    snr_score = min(100, stats['snr'] / 5.0 * 100)  # normalize to 0-100
    outside_score = max(0, 100 * (1 - stats['pct_out_changed'] / stats['max_outside_ratio']))
    inside_score = min(100, stats['mean_in'] / 20.0 * 100)  # assuming 20 is "very visible"
    
    # Weighted average
    confidence = 0.4 * snr_score + 0.4 * outside_score + 0.2 * inside_score
    
    return confidence


def visualize_adaptive(before, after, delta, mask, stats, out_path="result_adaptive.png"):
    """Enhanced visualization with adaptive threshold information."""
    delta_thresh = stats['adaptive_threshold']
    change_mask = (delta > delta_thresh).astype(np.uint8)

    in_change = (change_mask & mask).astype(np.uint8)
    out_change = (change_mask & (1 - mask)).astype(np.uint8)

    overlay = before.copy().astype(np.uint8)
    overlay[in_change.astype(bool)] = (255, 0, 0)   # red = inside change
    overlay[out_change.astype(bool)] = (0, 0, 255)  # blue = outside change

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Top row
    axes[0, 0].imshow(before)
    axes[0, 0].set_title("Before")
    axes[0, 0].axis("off")
    
    axes[0, 1].imshow(after)
    axes[0, 1].set_title("After")
    axes[0, 1].axis("off")
    
    axes[0, 2].imshow(delta, cmap="inferno", vmax=30)
    axes[0, 2].set_title(f"ΔE Map (adaptive thr={delta_thresh:.1f})")
    axes[0, 2].axis("off")
    
    # Bottom row
    axes[1, 0].imshow(mask * 255, cmap='gray')
    axes[1, 0].set_title("Editing Mask")
    axes[1, 0].axis("off")
    
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title(f"Change Overlay\n{stats['verdict']} (SNR={stats['snr']:.2f})")
    axes[1, 1].axis("off")
    
    # Statistics panel
    axes[1, 2].axis("off")
    confidence = compute_confidence_score(stats)
    
    criteria_text = "\n".join([
        f"{'✓' if v else '✗'} {k.replace('_', ' ').title()}"
        for k, v in stats['criteria'].items()
    ])
    
    stats_text = (
        f"VERDICT: {stats['verdict']}\n"
        f"Confidence: {confidence:.1f}%\n\n"
        f"Inside Region:\n"
        f"  Mean ΔE: {stats['mean_in']:.2f}\n"
        f"  Median ΔE: {stats['median_in']:.2f}\n"
        f"  Changed: {stats['pct_in_changed']:.1%}\n\n"
        f"Outside Region:\n"
        f"  Mean ΔE: {stats['mean_out']:.2f}\n"
        f"  Median ΔE: {stats['median_out']:.2f}\n"
        f"  Changed: {stats['pct_out_changed']:.2%}\n"
        f"  Allowed: {stats['max_outside_ratio']:.1%}\n\n"
        f"Signal-to-Noise: {stats['snr']:.2f}\n\n"
        f"Criteria ({stats['criteria_passed']}/{stats['criteria_total']}):\n"
        f"{criteria_text}"
    )
    
    axes[1, 2].text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
                    fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(f"Adaptive Validation Results\n{stats['explanation']}", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    print(f"[+] Saved visualization to {out_path}")


# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--sam-checkpoint", default="sam2.1_b.pt")
    parser.add_argument("--output", default="result_adaptive.png")
    args = parser.parse_args()

    print("[+] Loading images...")
    A = load_image(args.before)
    B = load_image(args.after)
    B = cv2.resize(B, (A.shape[1], A.shape[0]))

    print(f"[+] Generating SAM mask for: {args.prompt}")
    mask = get_sam_mask(A, args.prompt, sam_checkpoint=args.sam_checkpoint)

    print("[+] Computing perceptual delta map...")
    delta = delta_map_raw(A, B)

    print("[+] Performing adaptive evaluation...")
    stats = evaluate_with_adaptive_threshold(delta, mask)
    
    confidence = compute_confidence_score(stats)
    ssim_score = ssim(A, B, channel_axis=2)

    print("\n" + "="*60)
    print("ADAPTIVE VALIDATION RESULTS")
    print("="*60)
    print(f"Verdict       : {stats['verdict']}")
    print(f"Confidence    : {confidence:.1f}%")
    print(f"Criteria      : {stats['criteria_passed']}/{stats['criteria_total']} passed")
    print("\nInside Mask:")
    print(f"  Mean ΔE     : {stats['mean_in']:.4f}")
    print(f"  Median ΔE   : {stats['median_in']:.4f}")
    print(f"  % Changed   : {stats['pct_in_changed']:.4f}")
    print("\nOutside Mask:")
    print(f"  Mean ΔE     : {stats['mean_out']:.4f}")
    print(f"  Median ΔE   : {stats['median_out']:.4f}")
    print(f"  % Changed   : {stats['pct_out_changed']:.6f}")
    print(f"  Max Allowed : {stats['max_outside_ratio']:.4f}")
    print(f"\nSignal-to-Noise: {stats['snr']:.4f}")
    print(f"Adaptive Δ Thr : {stats['adaptive_threshold']:.4f}")
    print(f"SSIM Overall   : {ssim_score:.4f}")
    print("\nCriteria Breakdown:")
    for criterion, passed in stats['criteria'].items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {criterion.replace('_', ' ').title()}")
    print("="*60)

    visualize_adaptive(A, B, delta, mask, stats, out_path=args.output)
    
    # Return exit code based on verdict
    return 0 if stats['verdict'] == "PASS" else 1


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    exit(main())