#!/usr/bin/env python3
"""Visual validation for Stage 3: SAM Segmentation"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pipeline.stage2_color_filter import color_prefilter
from pipeline.stage3_sam_segmentation import segment_regions

# Load test image
test_img = cv2.imread("test_image.jpeg")
test_img_rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

print("Stage 2: Generating color mask...")
# Stage 2: Color filter
color_mask = color_prefilter(test_img_rgb, "blue")
print(f"Color mask coverage: {np.sum(color_mask) / color_mask.size * 100:.2f}%")

print("\nStage 3: Generating individual SAM masks...")
# Stage 3: SAM segmentation
individual_masks = segment_regions(test_img_rgb, color_mask, min_area=500)
print(f"Generated {len(individual_masks)} individual masks")

# Create visualization grid
num_masks = len(individual_masks)
cols = min(5, num_masks)
rows = (num_masks + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))

# Handle case of single row or column
if rows == 1 and cols == 1:
    axes = [[axes]]
elif rows == 1:
    axes = [axes]
elif cols == 1:
    axes = [[ax] for ax in axes]

for idx, mask in enumerate(individual_masks):
    row = idx // cols
    col = idx % cols

    # Create overlay
    overlay = test_img_rgb.copy()
    overlay[mask > 0] = overlay[mask > 0] * 0.5 + np.array([255, 0, 0]) * 0.5

    axes[row][col].imshow(overlay)
    axes[row][col].set_title(f"Mask {idx+1} ({np.sum(mask)} px)")
    axes[row][col].axis('off')

# Hide unused subplots
for idx in range(num_masks, rows * cols):
    row = idx // cols
    col = idx % cols
    axes[row][col].axis('off')

plt.tight_layout()
plt.savefig("logs/stage3_individual_masks.png", dpi=150, bbox_inches='tight')
print("\nSaved: logs/stage3_individual_masks.png")
print("\nSummary:")
print(f"- Input image: {test_img.shape}")
print(f"- Color mask coverage: {np.sum(color_mask) / color_mask.size * 100:.2f}%")
print(f"- Individual masks generated: {len(individual_masks)}")
print(f"- Mask sizes (pixels): {[np.sum(m) for m in individual_masks[:5]]}...")
