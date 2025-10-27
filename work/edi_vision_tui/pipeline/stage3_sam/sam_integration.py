import cv2
import numpy as np
import torch
from ultralytics import SAM
from typing import List, Dict, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("sam_integration")

class SAMIntegration:
    """
    Class to integrate SAM 2.1 with precise bounding boxes from YOLO.
    """
    def __init__(self, model_path=None):
        """
        Initialize the SAM integration with a model.
        
        Args:
            model_path: Path to SAM model checkpoint. If None, uses default model.
        """
        if model_path is None:
            # Use default SAM model
            logger.info("Loading default SAM 2.1 model...")
            self.model = SAM("sam2.1_l.pt")  # Use the larger model as requested
        else:
            logger.info(f"Loading custom SAM model from {model_path}...")
            self.model = SAM(model_path)
        
        logger.info("SAM model loaded successfully.")
    
    def generate_masks(self, image, refined_bounding_boxes, mask_threshold=0.5):
        """
        Generate high-quality masks using SAM 2.1 with precise bounding boxes from YOLO.
        
        Args:
            image: Input image as numpy array
            refined_bounding_boxes: Dictionary of entities with their refined bounding boxes from YOLO
            mask_threshold: Threshold for mask binarization
        
        Returns:
            dict: Dictionary with masks for each entity
        """
        logger.info("Starting SAM mask generation process...")
        
        # Dictionary to store masks
        masks_results = {}
        
        # If no refined bounding boxes were found, try general SAM segmentation
        if not refined_bounding_boxes:
            logger.info("No refined bounding boxes found. Running general SAM segmentation...")
            return self._generate_general_masks(image, mask_threshold)
        
        # Process each entity separately
        for entity, boxes in refined_bounding_boxes.items():
            logger.info(f"Generating masks for entity: {entity}")
            
            entity_masks = []
            
            for box_info in boxes:
                # Extract the bounding box coordinates
                x1, y1, x2, y2 = box_info['bbox']
                
                # Ensure the bounding box is within image bounds
                x1 = max(0, int(x1))
                y1 = max(0, int(y1))
                x2 = min(image.shape[1], int(x2))
                y2 = min(image.shape[0], int(y2))
                
                # Create bounding box in the format expected by SAM
                bbox = [[x1, y1, x2, y2]]
                
                # Run SAM with the precise bounding box from YOLO
                try:
                    results = self.model(image, bboxes=bbox, verbose=False)
                    masks_t = results[0].masks.data
                    
                    if masks_t is not None and masks_t.shape[0] > 0:
                        # Process the resulting mask
                        masks = masks_t.cpu().numpy()
                        
                        # Process each mask (there might be multiple masks per bbox)
                        for i in range(masks.shape[0]):
                            mask = masks[i]
                            
                            # Apply threshold to create binary mask
                            binary_mask = (mask > mask_threshold).astype(np.uint8)
                            
                            # Store the mask with its associated information
                            mask_info = {
                                'mask': binary_mask,
                                'bbox': box_info['bbox'],
                                'confidence': box_info['confidence'],
                                'class': box_info['class'],
                                'original_bbox': box_info['original_bbox'],
                                'original_score': box_info['original_score']
                            }
                            
                            entity_masks.append(mask_info)
                            logger.debug(f"Generated mask for {entity} with size {binary_mask.shape}")
                    else:
                        logger.warning(f"SAM returned no masks for entity {entity} at bbox {bbox}")
                except Exception as e:
                    logger.error(f"Error processing SAM for entity {entity}: {e}")
                    continue
            
            masks_results[entity] = entity_masks
            logger.info(f"Generated {len(entity_masks)} masks for entity: {entity}")
        
        logger.info("SAM mask generation process completed.")
        return masks_results
    
    def _generate_general_masks(self, image, mask_threshold=0.5):
        """
        Generate masks using general SAM segmentation when no specific bounding boxes are available.
        
        Args:
            image: Input image as numpy array
            mask_threshold: Threshold for mask binarization
        
        Returns:
            dict: Dictionary with general masks
        """
        logger.info("Generating general masks using full-image SAM segmentation...")
        
        try:
            # Run SAM on the entire image
            results = self.model(image, verbose=False)
            masks_t = results[0].masks.data
            
            if masks_t is not None and masks_t.shape[0] > 0:
                # Process the resulting masks
                masks = masks_t.cpu().numpy()
                
                # Store all masks under a 'general' key
                general_masks = []
                for i in range(masks.shape[0]):
                    mask = masks[i]
                    
                    # Apply threshold to create binary mask
                    binary_mask = (mask > mask_threshold).astype(np.uint8)
                    
                    # Store the mask with basic information
                    mask_info = {
                        'mask': binary_mask,
                        'bbox': (0, 0, image.shape[1], image.shape[0]),  # Full image bbox
                        'confidence': 0.5,  # Default confidence for general masks
                        'class': 'general',
                        'original_bbox': (0, 0, image.shape[1], image.shape[0]),
                        'original_score': 0.5
                    }
                    
                    general_masks.append(mask_info)
                
                logger.info(f"Generated {len(general_masks)} general masks")
                return {'general': general_masks}
            else:
                logger.warning("SAM returned no masks during general segmentation")
                return {}
        except Exception as e:
            logger.error(f"Error during general SAM segmentation: {e}")
            return {}
    
    def combine_masks(self, masks_results: Dict, merge_threshold=0.3):
        """
        Combine masks for multiple entities into a final mask.
        
        Args:
            masks_results: Dictionary of masks for each entity from SAM
            merge_threshold: Threshold for including masks in merge
        
        Returns:
            numpy.ndarray: Combined mask
        """
        logger.info("Starting mask combination process...")
        
        # Create a base mask to combine all entity masks
        if not masks_results:
            logger.warning("No masks to combine")
            return None
        
        # Get the size of the first mask to create a base
        first_entity = next(iter(masks_results))
        if not masks_results[first_entity]:
            logger.warning("No masks available for combination")
            return None
        
        base_mask = np.zeros_like(masks_results[first_entity][0]['mask'], dtype=np.float32)
        
        # Combine all masks from all entities
        for entity, masks in masks_results.items():
            for mask_info in masks:
                mask = mask_info['mask'].astype(np.float32)
                confidence = mask_info['confidence']
                
                # Weight the mask by confidence
                weighted_mask = mask * confidence
                
                # Add to the base mask (union operation)
                base_mask = np.where(base_mask + weighted_mask > 0, 1, 0).astype(np.uint8)
        
        logger.info("Mask combination process completed.")
        return base_mask