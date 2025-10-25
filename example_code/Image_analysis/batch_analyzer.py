#!/usr/bin/env python3
"""
Batch processing tool for validating multiple image edits.
Generates aggregate statistics and identifies patterns.

Usage:
    python batch_analyzer.py --config edits.json --output-dir results/
    
Config JSON format:
{
    "edits": [
        {
            "name": "edit1",
            "before": "path/to/before1.jpg",
            "after": "path/to/after1.jpg",
            "prompt": "blue roof"
        },
        ...
    ]
}
"""

import argparse
import json
import os
from pathlib import Path
import cv2
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from functools import lru_cache
from scipy.stats import mannwhitneyu, iqr
import open_clip
from ultralytics import SAM
from skimage.color import rgb2lab, deltaE_ciede2000
from skimage.filters import gaussian
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
sns.set_palette("husl")

# -------------------------
# Core functions (reused)
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
        return None
    
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

def evaluate_edit(delta, mask):
    """Statistical evaluation returning comprehensive metrics."""
    mask_bool = (mask == 1)
    inside = delta[mask_bool]
    outside = delta[~mask_bool]
    
    if inside.size == 0 or outside.size == 0:
        return None
    
    # Statistics
    inside_median = float(np.median(inside))
    outside_median = float(np.median(outside))
    inside_mean = float(np.mean(inside))
    outside_mean = float(np.mean(outside))
    inside_iqr = float(iqr(inside))
    outside_iqr = float(iqr(outside))
    
    # Statistical tests
    if inside.size > 10 and outside.size > 10:
        u_stat, u_pval = mannwhitneyu(inside, outside, alternative='greater')
    else:
        u_pval = 1.0
    
    # Cohen's d
    inside_std = np.std(inside)
    outside_std = np.std(outside)
    pooled_std = np.sqrt((inside_std**2 + outside_std**2) / 2)
    cohens_d = (inside_mean - outside_mean) / (pooled_std + 1e-8)
    
    # Adaptive threshold
    if cohens_d > 1.0:
        max_out_ratio = 0.03
    elif cohens_d > 0.5:
        max_out_ratio = 0.08
    else:
        max_out_ratio = 0.15
    
    outside_mad = np.median(np.abs(outside - outside_median))
    threshold = max(6.0, outside_median + 2.5 * outside_mad * 1.4826)
    
    # Changed pixels
    in_changed = float((inside > threshold).sum()) / inside.size
    out_changed = float((outside > threshold).sum()) / outside.size
    
    # Criteria
    criteria_scores = {
        'sufficient_change': 1 if inside_median > threshold * 0.5 else 0,
        'low_outside': 1 if out_changed <= max_out_ratio else 0,
        'effect_size': 1 if cohens_d > 0.3 else 0,
        'significant': 1 if u_pval < 0.01 else 0
    }
    
    score = sum(criteria_scores.values()) / len(criteria_scores)
    verdict = "PASS" if score >= 0.70 else "FAIL"
    
    # Special case: very clean edit
    if out_changed < 0.01 and inside_median > 5.0:
        verdict = "PASS"
        score = 1.0
    
    return {
        "verdict": verdict,
        "score": score,
        "inside_median": inside_median,
        "outside_median": outside_median,
        "inside_mean": inside_mean,
        "outside_mean": outside_mean,
        "inside_iqr": inside_iqr,
        "outside_iqr": outside_iqr,
        "pct_in_changed": in_changed,
        "pct_out_changed": out_changed,
        "cohens_d": cohens_d,
        "pval": u_pval,
        "threshold": threshold,
        "max_out_ratio": max_out_ratio,
        **criteria_scores
    }

# -------------------------
# Batch processing
# -------------------------
def process_batch(config_path, sam_checkpoint="sam2.1_b.pt"):
    """Process multiple image pairs and collect statistics."""
    with open(config_path) as f:
        config = json.load(f)
    
    results = []
    
    for edit in tqdm(config['edits'], desc="Processing edits"):
        try:
            name = edit['name']
            before_path = edit['before']
            after_path = edit['after']
            prompt = edit['prompt']
            
            # Load images
            A = load_image(before_path)
            B = load_image(after_path)
            B = cv2.resize(B, (A.shape[1], A.shape[0]))
            
            # Generate mask
            mask = get_sam_mask(A, prompt, sam_checkpoint=sam_checkpoint)
            if mask is None:
                print(f"[!] Failed to generate mask for {name}")
                continue
            
            # Compute delta
            delta = delta_map_raw(A, B)
            
            # Evaluate
            stats = evaluate_edit(delta, mask)
            if stats is None:
                print(f"[!] Failed to evaluate {name}")
                continue
            
            # Add metadata
            stats['name'] = name
            stats['prompt'] = prompt
            stats['image_size'] = f"{A.shape[1]}x{A.shape[0]}"
            stats['mask_coverage'] = float(mask.sum()) / mask.size
            
            results.append(stats)
            
        except Exception as e:
            print(f"[!] Error processing {edit.get('name', 'unknown')}: {e}")
            continue
    
    return pd.DataFrame(results)

# -------------------------
# Analysis and visualization
# -------------------------
def generate_summary_report(df, output_dir):
    """Generate comprehensive analysis report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Summary statistics
    print("\n" + "="*70)
    print("BATCH ANALYSIS SUMMARY")
    print("="*70)
    print(f"Total edits processed: {len(df)}")
    print(f"Passed: {(df['verdict'] == 'PASS').sum()} ({(df['verdict'] == 'PASS').mean():.1%})")
    print(f"Failed: {(df['verdict'] == 'FAIL').sum()} ({(df['verdict'] == 'FAIL').mean():.1%})")
    print("\nAverage metrics:")
    print(f"  Cohen's d       : {df['cohens_d'].mean():.3f} (std: {df['cohens_d'].std():.3f})")
    print(f"  Score           : {df['score'].mean():.3f} (std: {df['score'].std():.3f})")
    print(f"  Inside median ΔE: {df['inside_median'].mean():.2f} (std: {df['inside_median'].std():.2f})")
    print(f"  Outside median ΔE: {df['outside_median'].mean():.2f} (std: {df['outside_median'].std():.2f})")
    print(f"  % Outside changed: {df['pct_out_changed'].mean():.4f} (std: {df['pct_out_changed'].std():.4f})")
    print("="*70)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Verdict distribution
    verdict_counts = df['verdict'].value_counts()
    colors = ['lightgreen' if v == 'PASS' else 'lightcoral' for v in verdict_counts.index]
    axes[0, 0].bar(verdict_counts.index, verdict_counts.values, color=colors, edgecolor='black')
    axes[0, 0].set_title('Verdict Distribution', fontweight='bold', fontsize=12)
    axes[0, 0].set_ylabel('Count')
    for i, v in enumerate(verdict_counts.values):
        axes[0, 0].text(i, v + 0.5, str(v), ha='center', fontweight='bold')
    
    # 2. Score distribution
    axes[0, 1].hist(df['score'], bins=20, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(0.70, color='red', linestyle='--', label='Pass threshold (0.70)', linewidth=2)
    axes[0, 1].set_xlabel('Score')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Score Distribution', fontweight='bold', fontsize=12)
    axes[0, 1].legend()
    
    # 3. Cohen's d vs Score
    colors_scatter = ['green' if v == 'PASS' else 'red' for v in df['verdict']]
    axes[0, 2].scatter(df['cohens_d'], df['score'], c=colors_scatter, alpha=0.6, s=100, edgecolors='black')
    axes[0, 2].axhline(0.70, color='gray', linestyle='--', alpha=0.5)
    axes[0, 2].axvline(0.3, color='gray', linestyle='--', alpha=0.5)
    axes[0, 2].set_xlabel("Cohen's d (Effect Size)")
    axes[0, 2].set_ylabel('Score')
    axes[0, 2].set_title('Effect Size vs Score', fontweight='bold', fontsize=12)
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Inside vs Outside median comparison
    axes[1, 0].scatter(df['outside_median'], df['inside_median'], 
                       c=colors_scatter, alpha=0.6, s=100, edgecolors='black')
    # Diagonal line (equal change)
    max_val = max(df['outside_median'].max(), df['inside_median'].max())
    axes[1, 0].plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='Equal change')
    axes[1, 0].set_xlabel('Outside Median ΔE')
    axes[1, 0].set_ylabel('Inside Median ΔE')
    axes[1, 0].set_title('Inside vs Outside Change', fontweight='bold', fontsize=12)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Outside change percentage
    axes[1, 1].scatter(df['max_out_ratio'], df['pct_out_changed'], 
                       c=colors_scatter, alpha=0.6, s=100, edgecolors='black')
    # Add diagonal line showing threshold
    axes[1, 1].plot([0, 0.15], [0, 0.15], 'k--', alpha=0.3, label='Threshold line')
    axes[1, 1].set_xlabel('Max Allowed Outside Change (Adaptive)')
    axes[1, 1].set_ylabel('Actual Outside Change')
    axes[1, 1].set_title('Outside Change vs Threshold', fontweight='bold', fontsize=12)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Criteria breakdown
    criteria_cols = ['sufficient_change', 'low_outside', 'effect_size', 'significant']
    criteria_pass_rates = df[criteria_cols].mean()
    axes[1, 2].barh(criteria_cols, criteria_pass_rates, color='skyblue', edgecolor='black')
    axes[1, 2].set_xlabel('Pass Rate')
    axes[1, 2].set_title('Criteria Pass Rates', fontweight='bold', fontsize=12)
    axes[1, 2].set_xlim([0, 1])
    for i, v in enumerate(criteria_pass_rates):
        axes[1, 2].text(v + 0.02, i, f'{v:.1%}', va='center', fontweight='bold')
    
    plt.suptitle('Batch Image Edit Validation Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    summary_path = output_dir / 'batch_summary.png'
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    print(f"\n[+] Saved summary plot to {summary_path}")
    
    # Save detailed CSV
    csv_path = output_dir / 'results.csv'
    df.to_csv(csv_path, index=False)
    print(f"[+] Saved detailed results to {csv_path}")
    
    # Generate insights
    generate_insights(df, output_dir)

def generate_insights(df, output_dir):
    """Generate actionable insights from the batch analysis."""
    insights_path = output_dir / 'insights.txt'
    
    with open(insights_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("ACTIONABLE INSIGHTS\n")
        f.write("="*70 + "\n\n")
        
        # 1. Pass rate analysis
        pass_rate = (df['verdict'] == 'PASS').mean()
        f.write(f"1. OVERALL QUALITY\n")
        f.write(f"   Pass rate: {pass_rate:.1%}\n")
        if pass_rate < 0.5:
            f.write(f"   ⚠️  CONCERN: Less than 50% of edits passed validation.\n")
            f.write(f"   Action: Review editing pipeline for systematic issues.\n")
        elif pass_rate < 0.7:
            f.write(f"   ⚡ MODERATE: Majority pass but significant failures remain.\n")
            f.write(f"   Action: Investigate failed cases for patterns.\n")
        else:
            f.write(f"   ✓ GOOD: High pass rate indicates quality editing.\n")
        f.write("\n")
        
        # 2. Common failure modes
        failed_df = df[df['verdict'] == 'FAIL']
        if len(failed_df) > 0:
            f.write(f"2. FAILURE ANALYSIS ({len(failed_df)} failures)\n")
            
            # Check which criteria fail most
            criteria_cols = ['sufficient_change', 'low_outside', 'effect_size', 'significant']
            failed_criteria = failed_df[criteria_cols].mean()
            worst_criterion = failed_criteria.idxmin()
            worst_rate = failed_criteria.min()
            
            f.write(f"   Most common failure: {worst_criterion} ({worst_rate:.1%} pass rate)\n")
            
            if worst_criterion == 'low_outside':
                f.write(f"   Issue: Excessive unintended changes outside edit region.\n")
                f.write(f"   Average outside change in failures: {failed_df['pct_out_changed'].mean():.2%}\n")
                f.write(f"   Action: Improve masking or use more controlled editing methods.\n")
            elif worst_criterion == 'sufficient_change':
                f.write(f"   Issue: Insufficient change in target region.\n")
                f.write(f"   Action: Increase edit strength or verify prompts match targets.\n")
            elif worst_criterion == 'effect_size':
                f.write(f"   Issue: Poor separation between edited and unchanged regions.\n")
                f.write(f"   Action: Make edits more distinctive or improve mask accuracy.\n")
            elif worst_criterion == 'significant':
                f.write(f"   Issue: Changes not statistically significant.\n")
                f.write(f"   Action: Verify edits are actually occurring.\n")
            f.write("\n")
        
        # 3. Effect size distribution
        f.write(f"3. EDIT STRENGTH ANALYSIS\n")
        avg_cohens = df['cohens_d'].mean()
        f.write(f"   Average Cohen's d: {avg_cohens:.3f}\n")
        
        if avg_cohens < 0.5:
            f.write(f"   Status: Small effect sizes (subtle edits)\n")
            f.write(f"   Note: System automatically increases tolerance for subtle edits.\n")
        elif avg_cohens < 1.0:
            f.write(f"   Status: Medium effect sizes (moderate edits)\n")
        else:
            f.write(f"   Status: Large effect sizes (strong edits)\n")
            f.write(f"   Note: System applies stricter validation for obvious edits.\n")
        f.write("\n")
        
        # 4. Outlier detection
        f.write(f"4. OUTLIER DETECTION\n")
        
        # Find edits with unusually high outside change
        high_outside = df[df['pct_out_changed'] > df['pct_out_changed'].quantile(0.90)]
        if len(high_outside) > 0:
            f.write(f"   {len(high_outside)} edits with high outside change (>90th percentile):\n")
            for _, row in high_outside.iterrows():
                f.write(f"   - {row['name']}: {row['pct_out_changed']:.2%} outside changed\n")
            f.write(f"   Action: Manual review recommended for these cases.\n")
        f.write("\n")
        
        # Find edits with low effect size
        low_effect = df[df['cohens_d'] < 0.2]
        if len(low_effect) > 0:
            f.write(f"   {len(low_effect)} edits with very small effect (Cohen's d < 0.2):\n")
            for _, row in low_effect.iterrows():
                f.write(f"   - {row['name']}: d={row['cohens_d']:.3f}\n")
            f.write(f"   Action: Verify these edits are intentional or successful.\n")
        f.write("\n")
        
        # 5. Recommendations
        f.write(f"5. RECOMMENDATIONS\n")
        
        # Statistical significance
        sig_rate = df['significant'].mean()
        if sig_rate < 0.8:
            f.write(f"   • Only {sig_rate:.1%} of edits are statistically significant (p<0.01)\n")
            f.write(f"     Consider: Stronger edits or better targeting\n")
        
        # Outside change budget
        avg_outside_pct = df['pct_out_changed'].mean()
        if avg_outside_pct > 0.05:
            f.write(f"   • Average {avg_outside_pct:.2%} outside change (target: <5%)\n")
            f.write(f"     Consider: Better mask generation or attention mechanisms\n")
        
        # Adaptive threshold usage
        adaptive_variance = df['max_out_ratio'].std()
        f.write(f"   • Adaptive threshold variance: {adaptive_variance:.3f}\n")
        f.write(f"     Good: System adapts to different edit magnitudes\n")
        
        f.write("\n" + "="*70 + "\n")
    
    print(f"[+] Saved insights to {insights_path}")
    
    # Print insights to console too
    with open(insights_path, 'r') as f:
        print("\n" + f.read())

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Batch validation of image edits")
    parser.add_argument("--config", required=True, help="JSON config file with edit definitions")
    parser.add_argument("--output-dir", default="batch_results", help="Output directory")
    parser.add_argument("--sam-checkpoint", default="sam2.1_b.pt")
    args = parser.parse_args()
    
    print("[+] Starting batch processing...")
    df = process_batch(args.config, sam_checkpoint=args.sam_checkpoint)
    
    if len(df) == 0:
        print("[!] No edits processed successfully. Check your config file.")
        return 1
    
    print(f"\n[+] Successfully processed {len(df)} edits")
    print("[+] Generating analysis report...")
    generate_summary_report(df, args.output_dir)
    
    # Return exit code based on pass rate
    pass_rate = (df['verdict'] == 'PASS').mean()
    if pass_rate < 0.5:
        return 1  # Majority failed
    return 0

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    exit(main())
