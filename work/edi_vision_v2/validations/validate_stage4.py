#!/usr/bin/env python3
"""Visual validation for Stage 4: CLIP Semantic Filtering"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pipeline.stage2_color_filter import color_prefilter
from pipeline.stage3_sam_segmentation import segment_regions
from pipeline.stage4_clip_filter import clip_filter_masks

# Load test image
test_img = cv2.imread("test_image.jpeg")
test_img_rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

print("Stage 2: Generating color mask...")
# Stage 2: Color filter
color_mask = color_prefilter(test_img_rgb, "blue")
print(f"Color mask coverage: {np.sum(color_mask) / color_mask.size * 100:.2f}%")

print("\nStage 3: Generating SAM masks...")
# Stage 3: SAM segmentation
sam_masks = segment_regions(test_img_rgb, color_mask, min_area=500)
print(f"Generated {len(sam_masks)} SAM masks")

print("\nStage 4: Applying CLIP filtering...")
# Stage 4: CLIP filtering
filtered_masks = clip_filter_masks(test_img_rgb, sam_masks, "tin roof",
                                  similarity_threshold=0.22)
print(f"Filtered to {len(filtered_masks)} masks")

# Create comparison visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Before CLIP (all SAM masks)
overlay_before = test_img_rgb.copy()
for mask in sam_masks:
    overlay_before[mask > 0] = overlay_before[mask > 0] * 0.5 + np.array([255, 0, 0]) * 0.5

axes[0].imshow(overlay_before)
axes[0].set_title(f"Before CLIP: {len(sam_masks)} masks\n(includes sky, buildings, etc.)")
axes[0].axis('off')

# After CLIP (filtered masks)
overlay_after = test_img_rgb.copy()
for mask, score in filtered_masks:
    overlay_after[mask > 0] = overlay_after[mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5

axes[1].imshow(overlay_after)
axes[1].set_title(f"After CLIP: {len(filtered_masks)} masks\n(roofs only)")
axes[1].axis('off')

plt.tight_layout()
plt.savefig("logs/stage4_clip_filtering.png", dpi=150, bbox_inches='tight')
print("\nSaved: logs/stage4_clip_filtering.png")

# Print top similarity scores
print("\nTop similarity scores:")
for i, (_, score) in enumerate(filtered_masks[:10]):
    print(f"  Mask {i+1}: {score:.3f}")

print("\nSummary:")
print(f"- Stage 2: Color mask with {np.sum(color_mask) / color_mask.size * 100:.2f}% coverage")
print(f"- Stage 3: {len(sam_masks)} SAM masks")
print(f"- Stage 4: {len(filtered_masks)} CLIP-filtered masks")
print(f"- Filtered out: {len(sam_masks) - len(filtered_masks)} masks ({(len(sam_masks) - len(filtered_masks)) / len(sam_masks) * 100:.1f}%)")
