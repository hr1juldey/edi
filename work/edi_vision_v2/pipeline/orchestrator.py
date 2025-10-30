"""Stage 7: Pipeline Orchestrator

This module orchestrates the complete 6-stage vision pipeline.
"""

import logging
import time
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt

from pipeline.stage1_entity_extraction import parse_intent
from pipeline.stage2_color_filter import color_prefilter
from pipeline.stage3_sam_segmentation import segment_regions
from pipeline.stage4_clip_filter import clip_filter_masks
from pipeline.stage5_mask_organization import organize_masks, EntityMask
from pipeline.stage6_vlm_validation import validate_with_vlm, create_validation_overlay


class VisionPipeline:
    """Orchestrates the complete 6-stage vision pipeline.

    Stages:
        1. DSpy Entity Extraction - Parse user intent
        2. Color Pre-Filtering - HSV-based color detection
        3. SAM Segmentation - Pixel-perfect masks
        4. CLIP Filtering - Semantic filtering
        5. Mask Organization - Structure with metadata
        6. VLM Validation - Quality assurance (optional)
    """

    def __init__(self,
                 enable_validation: bool = True,
                 save_intermediate: bool = False,
                 output_dir: str = "logs"):
        """
        Initialize the vision pipeline.

        Args:
            enable_validation: Whether to run Stage 6 VLM validation
            save_intermediate: Save intermediate results to disk
            output_dir: Directory for logs and intermediate files
        """
        self.enable_validation = enable_validation
        self.save_intermediate = save_intermediate
        self.output_dir = output_dir

        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        logging.info(f"VisionPipeline initialized (validation={enable_validation}, "
                    f"save_intermediate={save_intermediate})")

    def process(self, image_path: str, user_prompt: str) -> Dict[str, Any]:
        """
        Process an image with the complete vision pipeline.

        Args:
            image_path: Path to input image
            user_prompt: User's editing request (e.g., "change blue roofs to green")

        Returns:
            Dictionary containing:
            {
                'success': bool,
                'entity_masks': List[EntityMask],  # Stage 5 output
                'validation': Dict,                 # Stage 6 output (if enabled)
                'stage_timings': Dict[str, float],  # Execution times
                'metadata': Dict,                   # Image info, detected entities
                'error': str or None                # Error message if failed
            }
        """
        # Initialize result structure
        result = {
            'success': False,
            'entity_masks': [],
            'validation': None,
            'stage_timings': {},
            'metadata': {},
            'error': None
        }

        try:
            # Load and validate image
            image = self._load_image(image_path)
            result['metadata']['image_shape'] = image.shape
            result['metadata']['image_path'] = image_path

            # Stage 1: Entity Extraction
            logging.info("="*60)
            logging.info("STAGE 1: Entity Extraction (DSpy)")
            logging.info("="*60)
            t_start = time.time()

            intent = parse_intent(user_prompt)

            result['stage_timings']['stage1_entity_extraction'] = time.time() - t_start
            result['metadata']['intent'] = intent

            logging.info(f"Extracted entities: {intent['target_entities']}")
            logging.info(f"Edit type: {intent['edit_type']}")
            logging.info(f"Confidence: {intent['confidence']}")

            if intent['confidence'] < 0.5:
                raise ValueError(f"Low confidence in intent parsing: {intent['confidence']}")

            # Extract color from entities
            target_color = self._extract_color(intent['target_entities'])
            if not target_color:
                raise ValueError("Could not extract target color from entities")

            # Stage 2: Color Pre-Filter
            logging.info("="*60)
            logging.info(f"STAGE 2: Color Pre-Filter (HSV - {target_color})")
            logging.info("="*60)
            t_start = time.time()

            color_mask = color_prefilter(image, target_color)

            result['stage_timings']['stage2_color_filter'] = time.time() - t_start
            result['metadata']['color_coverage'] = np.sum(color_mask) / color_mask.size * 100

            logging.info(f"Color coverage: {result['metadata']['color_coverage']:.2f}%")

            if self.save_intermediate:
                self._save_mask(color_mask, "stage2_color_mask.png")

            if np.sum(color_mask) == 0:
                raise ValueError(f"No {target_color} regions detected")

            # Stage 3: SAM Segmentation
            logging.info("="*60)
            logging.info("STAGE 3: SAM Segmentation")
            logging.info("="*60)
            t_start = time.time()

            sam_masks = segment_regions(image, color_mask, min_area=500)

            result['stage_timings']['stage3_sam_segmentation'] = time.time() - t_start
            result['metadata']['sam_masks_count'] = len(sam_masks)

            logging.info(f"SAM generated {len(sam_masks)} masks")

            if len(sam_masks) == 0:
                raise ValueError("SAM failed to generate any masks")

            if self.save_intermediate:
                self._save_masks_grid(image, sam_masks, "stage3_sam_masks.png")

            # Stage 4: CLIP Filtering
            logging.info("="*60)
            logging.info("STAGE 4: CLIP Semantic Filtering")
            logging.info("="*60)
            t_start = time.time()

            # Extract target entity description for CLIP
            target_description = self._get_target_description(intent['target_entities'])

            filtered_masks = clip_filter_masks(image, sam_masks, target_description,
                                              similarity_threshold=0.22)

            result['stage_timings']['stage4_clip_filter'] = time.time() - t_start
            result['metadata']['clip_filtered_count'] = len(filtered_masks)
            result['metadata']['filter_rate'] = (len(sam_masks) - len(filtered_masks)) / len(sam_masks) * 100

            logging.info(f"CLIP filtered to {len(filtered_masks)} masks ({result['metadata']['filter_rate']:.1f}% removed)")

            if len(filtered_masks) == 0:
                raise ValueError(f"All masks filtered out - no '{target_description}' found")

            if self.save_intermediate:
                self._save_filtered_comparison(image, sam_masks, filtered_masks,
                                              "stage4_clip_filtering.png")

            # Stage 5: Mask Organization
            logging.info("="*60)
            logging.info("STAGE 5: Mask Organization")
            logging.info("="*60)
            t_start = time.time()

            entity_masks = organize_masks(image, filtered_masks)

            result['stage_timings']['stage5_organization'] = time.time() - t_start
            result['metadata']['final_entity_count'] = len(entity_masks)

            logging.info(f"Organized {len(entity_masks)} entity masks")

            if self.save_intermediate:
                self._save_entity_masks(image, entity_masks, "stage5_entity_masks.png")

            # Store results
            result['entity_masks'] = entity_masks

            # Stage 6: VLM Validation (optional)
            if self.enable_validation:
                logging.info("="*60)
                logging.info("STAGE 6: VLM Validation")
                logging.info("="*60)
                t_start = time.time()

                try:
                    # Extract mask arrays from EntityMask objects
                    mask_arrays = [entity.mask for entity in entity_masks]

                    validation_result = validate_with_vlm(image, mask_arrays, user_prompt)

                    result['stage_timings']['stage6_validation'] = time.time() - t_start
                    result['validation'] = validation_result

                    logging.info(f"VLM Validation: {validation_result.feedback}")
                    logging.info(f"Confidence: {validation_result.confidence:.2f}")

                    if self.save_intermediate:
                        validation_overlay = create_validation_overlay(image, mask_arrays)
                        self._save_image(validation_overlay, "stage6_validation.png")

                except Exception as e:
                    logging.warning(f"VLM validation failed: {e}")
                    result['validation'] = {'error': str(e), 'confidence': 0.0}

            # Pipeline complete
            result['success'] = True
            result['metadata']['total_time'] = sum(result['stage_timings'].values())

            logging.info("="*60)
            logging.info("PIPELINE COMPLETE")
            logging.info("="*60)
            logging.info(f"Total time: {result['metadata']['total_time']:.2f}s")
            logging.info(f"Final entity count: {len(entity_masks)}")

            return result

        except Exception as e:
            logging.error(f"Pipeline failed: {e}")
            result['error'] = str(e)
            return result

    def _load_image(self, image_path: str) -> np.ndarray:
        """Load and validate image."""
        import cv2
        from pathlib import Path

        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        logging.info(f"Loaded image: {image_path} ({image_rgb.shape})")

        return image_rgb

    def _extract_color(self, target_entities: List[Dict]) -> str:
        """Extract color from entity descriptions."""
        for entity in target_entities:
            if 'color' in entity and entity['color']:
                return entity['color'].lower()

        # Fallback: return first entity label if it contains a color word
        color_words = ['blue', 'red', 'green', 'yellow', 'orange', 'white', 'black']
        for entity in target_entities:
            label = entity.get('label', '').lower()
            for color in color_words:
                if color in label:
                    return color

        return None

    def _get_target_description(self, target_entities: List[Dict]) -> str:
        """Get target entity description for CLIP."""
        if not target_entities:
            return "object"

        # Use first entity's label
        entity = target_entities[0]
        label = entity.get('label', 'object')

        # Optionally prepend color
        color = entity.get('color', '')
        if color:
            return f"{color} {label}"

        return label

    def _save_mask(self, mask: np.ndarray, filename: str):
        """Save binary mask to output directory."""
        from pathlib import Path
        import cv2

        output_path = Path(self.output_dir) / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to 0-255 range for saving
        mask_img = (mask * 255).astype(np.uint8)
        cv2.imwrite(str(output_path), mask_img)

        logging.debug(f"Saved mask: {output_path}")

    def _save_image(self, image: np.ndarray, filename: str):
        """Save RGB image to output directory."""
        from pathlib import Path
        import cv2

        output_path = Path(self.output_dir) / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert RGB to BGR for cv2
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), image_bgr)

        logging.debug(f"Saved image: {output_path}")

    def _save_masks_grid(self, image: np.ndarray, masks: List[np.ndarray],
                        filename: str, max_display: int = 20):
        """Save grid visualization of masks."""

        num_masks = min(len(masks), max_display)
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
                overlay = image.copy()
                overlay[masks[idx] > 0] = overlay[masks[idx] > 0] * 0.5 + np.array([255, 0, 0]) * 0.5
                axes[row][col].imshow(overlay)
                axes[row][col].set_title(f"Mask {idx+1}")
                axes[row][col].axis('off')
            else:
                axes[row][col].axis('off')

        plt.tight_layout()
        output_path = Path(self.output_dir) / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        logging.debug(f"Saved masks grid: {output_path}")

    def _save_filtered_comparison(self, image: np.ndarray,
                                  before_masks: List[np.ndarray],
                                  after_masks: List[Tuple[np.ndarray, float]],
                                  filename: str):
        """Save before/after CLIP filtering comparison."""

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Before CLIP
        overlay_before = image.copy()
        for mask in before_masks:
            overlay_before[mask > 0] = overlay_before[mask > 0] * 0.5 + np.array([255, 0, 0]) * 0.5
        axes[0].imshow(overlay_before)
        axes[0].set_title(f"Before CLIP: {len(before_masks)} masks")
        axes[0].axis('off')

        # After CLIP
        overlay_after = image.copy()
        for mask, score in after_masks:
            overlay_after[mask > 0] = overlay_after[mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
        axes[1].imshow(overlay_after)
        axes[1].set_title(f"After CLIP: {len(after_masks)} masks")
        axes[1].axis('off')

        plt.tight_layout()
        output_path = Path(self.output_dir) / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        logging.debug(f"Saved filtering comparison: {output_path}")

    def _save_entity_masks(self, image: np.ndarray, entity_masks: List[EntityMask],
                          filename: str, max_display: int = 10):
        """Save entity masks with metadata."""

        num_masks = min(len(entity_masks), max_display)
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
                overlay = image.copy()
                overlay[entity.mask > 0] = overlay[entity.mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5

                # Draw bbox
                x_min, y_min, x_max, y_max = entity.bbox
                cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (255, 255, 0), 2)

                # Draw centroid
                cx, cy = entity.centroid
                cv2.circle(overlay, (int(cx), int(cy)), 5, (255, 0, 0), -1)

                axes[row][col].imshow(overlay)
                axes[row][col].set_title(
                    f"Entity {entity.entity_id}\n"
                    f"Area: {entity.area}px\n"
                    f"Score: {entity.similarity_score:.3f}",
                    fontsize=8
                )
                axes[row][col].axis('off')
            else:
                axes[row][col].axis('off')

        plt.tight_layout()
        output_path = Path(self.output_dir) / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        logging.debug(f"Saved entity masks: {output_path}")