#!/usr/bin/env python3
"""
Compare statistical adaptive approach vs fixed thresholds across multiple images.
Helps understand when statistical methods provide better discrimination.

Usage:
    python comparison_analyzer.py --before before.jpg --after after.jpg --prompt "roof"
"""

import argparse
import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from scipy.stats import mannwhitneyu, ks_2samp, iqr
import open_clip
from ultralytics import SAM
from skimage.color import rgb2lab, deltaE_ciede2000
from skimage.filters import gaussian
from functools import lru_cache

sns.set_style("whitegrid")

# -------------------------
# Reuse core functions
# -------------------------
def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

@lru_cache(maxsize=1)
def _load_clip(device):
    model_name = "ViT-B-32"
    for pretrained in ("openai", "laion2b_s34b_b79k", "laion400m_e32"):
        try:
            clip_model, clip_transform, _ = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained
            )
            clip_model.to(device)
            clip_model.eval()
            return clip_model, clip_transform
        except:
            continue
    raise RuntimeError("Cannot load CLIP")

def get_sam_mask(image, prompt, sam_checkpoint="sam2.1_b.pt", device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    sam = SAM(sam_checkpoint)
    results = sam(image, verbose=False)
    masks_t = results[0].masks.data
    if masks_t is None or masks_t.shape[0] == 0:
        raise RuntimeError("SAM returned no masks")
    
    masks = masks_t.cpu().numpy()
    clip_model, clip_transform = _load_clip(device)
    
    text_tokens = open_clip.tokenize([prompt]).to(device)
    with torch.no_grad():
        text_emb = clip_model.encode_text(text_tokens)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
    
    best_sim = -99
    best_mask = None
    for i in range(masks.shape[0]):
        m = (masks[i] > 0.5).astype(np.uint8)
        ys, xs = np.where(m > 0)
        if ys.size == 0 or (ys.max() - ys.min()) < 8:
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
        if sim > best_sim:
            best_sim = sim
            best_mask = (masks[i] > 0.5).astype(np.uint8)
    
    return best_mask

def delta_map_raw(imgA, imgB, blur_sigma=0.8):
    A_f = imgA.astype(np.float32) / 255.0
    B_f = imgB.astype(np.float32) / 255.0
    A_blur = gaussian(A_f, sigma=blur_sigma, channel_axis=-1)
    B_blur = gaussian(B_f, sigma=blur_sigma, channel_axis=-1)
    A_lab = rgb2lab((A_blur * 255).astype(np.uint8))
    B_lab = rgb2lab((B_blur * 255).astype(np.uint8))
    delta = deltaE_ciede2000(A_lab, B_lab)
    return np.nan_to_num(delta)

# -------------------------
# Fixed threshold approach (original)
# -------------------------
def evaluate_fixed_threshold(delta, mask):
    """Original fixed threshold approach."""
    delta_thresh = 6.0
    max_allowed_outside_frac = 0.01
    min_required_inside_frac = 0.02
    
    mask_bool = (mask == 1)
    inside = delta[mask_bool]
    outside = delta[~mask_bool]
    
    mean_in = float(inside.mean()) if inside.size else 0.0
    mean_out = float(outside.mean()) if outside.size else 0.0
    
    in_changed = float((inside > delta_thresh).sum()) if inside.size else 0.0
    out_changed = float((outside > delta_thresh).sum()) if outside.size else 0.0
    pct_in = in_changed / (inside.size + 1e-8)
    pct_out = out_changed / (outside.size + 1e-8)
    
    ok_inside = pct_in >= min_required_inside_frac or mean_in >= (delta_thresh / 2.0)
    ok_outside = pct_out <= max_allowed_outside_frac and mean_out < (delta_thresh / 2.0)
    
    verdict = "PASS" if (ok_inside and ok_outside) else "FAIL"
    
    return {
        "method": "Fixed",
        "verdict": verdict,
        "mean_in": mean_in,
        "mean_out": mean_out,
        "pct_out": pct_out,
        "threshold": delta_thresh,
        "max_out_ratio": max_allowed_outside_frac
    }

# -------------------------
# Statistical approach
# -------------------------
def evaluate_statistical(delta, mask):
    """Statistical adaptive approach."""
    mask_bool = (mask == 1)
    inside = delta[mask_bool]
    outside = delta[~mask_bool]
    
    # Statistics
    inside_median = np.median(inside) if inside.size else 0
    outside_median = np.median(outside) if outside.size else 0
    inside_iqr = iqr(inside) if inside.size else 0
    outside_iqr = iqr(outside) if outside.size else 0
    
    # Statistical tests
    if inside.size > 10 and outside.size > 10:
        u_stat, u_pval = mannwhitneyu(inside, outside, alternative='greater')
        ks_stat, ks_pval = ks_2samp(inside, outside)
    else:
        u_pval, ks_pval = 1.0, 1.0
    
    # Cohen's d
    inside_mean = np.mean(inside) if inside.size else 0
    outside_mean = np.mean(outside) if outside.size else 0
    inside_std = np.std(inside) if inside.size else 1
    outside_std = np.std(outside) if outside.size else 1
    pooled_std = np.sqrt((inside_std**2 + outside_std**2) / 2)
    cohens_d = (inside_mean - outside_mean) / (pooled_std + 1e-8)
    
    # Adaptive threshold
    if cohens_d > 1.0:
        max_out_ratio = 0.03
    elif cohens_d > 0.5:
        max_out_ratio = 0.08
    else:
        max_out_ratio = 0.15
    
    outside_mad = np.median(np.abs(outside - outside_median)) if outside.size else 0
    threshold = max(6.0, outside_median + 2.5 * outside_mad * 1.4826)
    
    # Changed pixels
    out_changed = float((outside > threshold).sum()) if outside.size else 0.0
    pct_out = out_changed / (outside.size + 1e-8)
    
    # Criteria
    criteria = {
        'sufficient_change': inside_median > threshold * 0.5,
        'low_outside': pct_out <= max_out_ratio,
        'effect_size': cohens_d > 0.3,
        'significant': u_pval < 0.01
    }
    
    score = sum(criteria.values()) / len(criteria)
    verdict = "PASS" if score >= 0.70 else "FAIL"
    
    return {
        "method": "Statistical",
        "verdict": verdict,
        "mean_in": inside_mean,
        "mean_out": outside_mean,
        "pct_out": pct_out,
        "threshold": threshold,
        "max_out_ratio": max_out_ratio,
        "cohens_d": cohens_d,
        "pval": u_pval,
        "score": score
    }

# -------------------------
# Visualization
# -------------------------
def create_comparison_plot(before, after, delta, mask, fixed_stats, stat_stats, out_path="comparison.png"):
    """Create comprehensive comparison visualization."""
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Row 1: Images
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(before)
    ax1.set_title("Before Image", fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(after)
    ax2.set_title("After Image", fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(delta, cmap='inferno', vmax=30)
    ax3.set_title("ΔE Color Difference Map", fontsize=12, fontweight='bold')
    ax3.axis('off')
    cbar = plt.colorbar(ax3.images[0], ax=ax3, fraction=0.046)
    cbar.set_label('ΔE (CIEDE2000)', rotation=270, labelpad=20)
    
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(mask * 255, cmap='gray')
    ax4.set_title("Editing Mask (SAM+CLIP)", fontsize=12, fontweight='bold')
    ax4.axis('off')
    
    # Row 2: Distribution analysis
    mask_bool = (mask == 1)
    inside = delta[mask_bool]
    outside = delta[~mask_bool]
    
    ax5 = fig.add_subplot(gs[1, :2])
    ax5.hist(outside, bins=50, alpha=0.6, label='Outside Mask', color='blue', density=True)
    ax5.hist(inside, bins=50, alpha=0.6, label='Inside Mask', color='red', density=True)
    ax5.axvline(fixed_stats['threshold'], color='green', linestyle='--', 
                label=f"Fixed Thr={fixed_stats['threshold']:.1f}", linewidth=2)
    ax5.axvline(stat_stats['threshold'], color='orange', linestyle='--',
                label=f"Adaptive Thr={stat_stats['threshold']:.1f}", linewidth=2)
    ax5.set_xlabel('ΔE Value', fontsize=11)
    ax5.set_ylabel('Density', fontsize=11)
    ax5.set_title('Distribution of Changes: Inside vs Outside', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Box plot
    ax6 = fig.add_subplot(gs[1, 2:])
    box_data = [outside, inside]
    bp = ax6.boxplot(box_data, labels=['Outside', 'Inside'], patch_artist=True,
                      medianprops=dict(color='red', linewidth=2),
                      boxprops=dict(facecolor='lightblue', alpha=0.7))
    ax6.set_ylabel('ΔE Value', fontsize=11)
    ax6.set_title('Box Plot: Change Distribution', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Add statistical annotations
    ax6.text(0.02, 0.98, 
             f"Cohen's d = {stat_stats['cohens_d']:.3f}\n"
             f"p-value = {stat_stats['pval']:.6f}",
             transform=ax6.transAxes, fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Row 3: Comparison table
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')
    
    # Create comparison table
    table_data = [
        ["Metric", "Fixed Threshold", "Statistical Adaptive", "Winner"],
        ["Verdict", 
         f"{fixed_stats['verdict']}", 
         f"{stat_stats['verdict']}", 
         ""],
        ["Mean Inside ΔE", 
         f"{fixed_stats['mean_in']:.2f}", 
         f"{stat_stats['mean_in']:.2f}", 
         ""],
        ["Mean Outside ΔE", 
         f"{fixed_stats['mean_out']:.2f}", 
         f"{stat_stats['mean_out']:.2f}", 
         ""],
        ["% Outside Changed", 
         f"{fixed_stats['pct_out']:.4f}", 
         f"{stat_stats['pct_out']:.4f}", 
         ""],
        ["Threshold Used", 
         f"{fixed_stats['threshold']:.2f}", 
         f"{stat_stats['threshold']:.2f} (adaptive)", 
         "✓" if stat_stats['threshold'] > fixed_stats['threshold'] else ""],
        ["Max Outside Allowed", 
         f"{fixed_stats['max_out_ratio']:.3f}", 
         f"{stat_stats['max_out_ratio']:.3f} (adaptive)", 
         "✓"],
        ["Statistical Tests", 
         "None", 
         f"Cohen's d={stat_stats['cohens_d']:.2f}, p={stat_stats['pval']:.4f}", 
         "✓"],
        ["Adaptivity", 
         "No", 
         "Yes", 
         "✓"],
    ]
    
    # Color coding for verdicts
    cell_colors = []
    for i, row in enumerate(table_data):
        if i == 0:  # Header
            cell_colors.append(['lightgray'] * 4)
        elif i == 1:  # Verdict row
            colors = []
            for j, cell in enumerate(row):
                if j == 0:
                    colors.append('white')
                elif cell == "PASS":
                    colors.append('lightgreen')
                elif cell == "FAIL":
                    colors.append('lightcoral')
                else:
                    colors.append('white')
            cell_colors.append(colors)
        else:
            cell_colors.append(['white'] * 4)
    
    table = ax7.table(cellText=table_data, cellLoc='left', loc='center',
                      cellColours=cell_colors, colWidths=[0.25, 0.25, 0.35, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('darkgray')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Add summary text
    summary = (
        f"SUMMARY\n"
        f"{'='*60}\n"
        f"Fixed Method: {fixed_stats['verdict']} "
        f"(Rigid thresholds, no context adaptation)\n"
        f"Statistical Method: {stat_stats['verdict']} "
        f"(Adaptive, uses Mann-Whitney U, Cohen's d)\n\n"
        f"The statistical method adapts to image characteristics:\n"
        f"• Effect size (Cohen's d={stat_stats['cohens_d']:.2f}) determines strictness\n"
        f"• Large effect → strict (max out = {stat_stats['max_out_ratio']:.1%})\n"
        f"• Small effect → lenient (max out up to 15%)\n"
        f"• Uses robust statistics (median, IQR, MAD) instead of mean/std"
    )
    
    fig.text(0.5, 0.02, summary, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
             family='monospace')
    
    plt.suptitle("Fixed vs Statistical Adaptive Evaluation Comparison", 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"[+] Saved comparison to {out_path}")

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Compare evaluation methods")
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--sam-checkpoint", default="sam2.1_b.pt")
    parser.add_argument("--output", default="comparison.png")
    args = parser.parse_args()
    
    print("[+] Loading images...")
    A = load_image(args.before)
    B = load_image(args.after)
    B = cv2.resize(B, (A.shape[1], A.shape[0]))
    
    print(f"[+] Generating mask for: '{args.prompt}'")
    mask = get_sam_mask(A, args.prompt, sam_checkpoint=args.sam_checkpoint)
    
    print("[+] Computing delta map...")
    delta = delta_map_raw(A, B)
    
    print("[+] Evaluating with fixed threshold method...")
    fixed_stats = evaluate_fixed_threshold(delta, mask)
    
    print("[+] Evaluating with statistical adaptive method...")
    stat_stats = evaluate_statistical(delta, mask)
    
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    print(f"\n{'Method':<25} {'Verdict':<10} {'Mean Out':<12} {'% Out':<12} {'Threshold':<10}")
    print("-"*70)
    print(f"{'Fixed Threshold':<25} {fixed_stats['verdict']:<10} "
          f"{fixed_stats['mean_out']:<12.4f} {fixed_stats['pct_out']:<12.6f} {fixed_stats['threshold']:<10.2f}")
    print(f"{'Statistical Adaptive':<25} {stat_stats['verdict']:<10} "
          f"{stat_stats['mean_out']:<12.4f} {stat_stats['pct_out']:<12.6f} {stat_stats['threshold']:<10.2f}")
    print("="*70)
    
    print(f"\nStatistical Metrics:")
    print(f"  Cohen's d: {stat_stats['cohens_d']:.4f} ({'large' if abs(stat_stats['cohens_d']) > 0.8 else 'medium' if abs(stat_stats['cohens_d']) > 0.5 else 'small'} effect)")
    print(f"  p-value  : {stat_stats['pval']:.6f} ({'significant' if stat_stats['pval'] < 0.01 else 'not significant'})")
    print(f"  Score    : {stat_stats['score']:.2%}")
    
    if fixed_stats['verdict'] != stat_stats['verdict']:
        print(f"\n⚠️  METHODS DISAGREE!")
        print(f"   Fixed says: {fixed_stats['verdict']}")
        print(f"   Statistical says: {stat_stats['verdict']}")
        print(f"   Reason: Statistical method adapted threshold to {stat_stats['threshold']:.2f} "
              f"and max outside ratio to {stat_stats['max_out_ratio']:.2%}")
    else:
        print(f"\n✓ Both methods agree: {fixed_stats['verdict']}")
    
    print("\n[+] Creating visualization...")
    create_comparison_plot(A, B, delta, mask, fixed_stats, stat_stats, out_path=args.output)
    
    return 0 if stat_stats['verdict'] == "PASS" else 1

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    exit(main())
