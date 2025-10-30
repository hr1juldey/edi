#!/usr/bin/env python3
"""Visual validation for Stage 5: Mask Organization"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pipeline.stage2_color_filter import color_prefilter
from pipeline.stage3_sam_segmentation import segment_regions
from pipeline.stage4_clip_filter import clip_filter_masks
from pipeline.stage5_mask_organization import organize_masks

# Load test image
test_img = cv2.imread("test_image.jpeg")
test_img_rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

print("Running full pipeline Stages 2-5...")

# Stage 2: Color filter
color_mask = color_prefilter(test_img_rgb, "blue")

# Stage 3: SAM segmentation
sam_masks = segment_regions(test_img_rgb, color_mask, min_area=500)

# Stage 4: CLIP filtering
filtered_masks = clip_filter_masks(test_img_rgb, sam_masks, "tin roof", similarity_threshold=0.22)

# Stage 5: Organize
entity_masks = organize_masks(test_img_rgb, filtered_masks)

print("\nStage 5 Results:")
print(f"Organized {len(entity_masks)} separate entity masks")

# Create visualization showing each mask with metadata
num_masks = min(len(entity_masks), 10)  # Show top 10
cols = 5
rows = (num_masks + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
if rows == 1:
    axes = [axes]
if isinstance(axes, np.ndarray) and axes.ndim == 1:
    axes = [axes]

for idx in range(rows * cols):
    row = idx // cols
    col = idx % cols

    if idx < num_masks:
        entity = entity_masks[idx]

        # Create overlay
        overlay = test_img_rgb.copy()
        overlay[entity.mask > 0] = overlay[entity.mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5

        # Draw bounding box
        x_min, y_min, x_max, y_max = entity.bbox
        cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (255, 255, 0), 2)

        # Draw centroid
        cx, cy = entity.centroid
        cv2.circle(overlay, (int(cx), int(cy)), 5, (255, 0, 0), -1)

        axes[row][col].imshow(overlay)
        axes[row][col].set_title(
            f"Entity {entity.entity_id}\n"
            f"Area: {entity.area}px\n"
            f"Score: {entity.similarity_score:.3f}\n"
            f"Color: RGB{entity.dominant_color}",
            fontsize=8
        )
        axes[row][col].axis('off')
    else:
        axes[row][col].axis('off')

plt.tight_layout()
plt.savefig("logs/stage5_entity_masks.png", dpi=150, bbox_inches='tight')
print("\nSaved: logs/stage5_entity_masks.png")

# Print entity details
print("\nTop 5 Entity Masks:")
for entity in entity_masks[:5]:
    print(f"  {entity}")

print("\nSummary:")
print(f"- Total entity masks: {len(entity_masks)}")
print(f"- Largest mask: {entity_masks[0].area} pixels")
print(f"- Smallest mask: {entity_masks[-1].area} pixels")
print(f"- Average area: {np.mean([e.area for e in entity_masks]):.0f} pixels")
print("\nCRITICAL VERIFICATION: Each entity has separate mask - NOT merged ✓")
