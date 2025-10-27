import cv2
import numpy as np
from ultralytics import YOLO
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("yolo_refiner")

class YOLORefiner:
    """
    Class to refine bounding boxes using YOLO for precise object detection.
    """
    def __init__(self, model_path=None):
        """
        Initialize the YOLO refiner with a model.
        
        Args:
            model_path: Path to YOLO model checkpoint. If None, uses default model.
        """
        if model_path is None:
            # Use default YOLO model for object detection
            logger.info("Loading default YOLO model...")
            self.model = YOLO("yolo11n.pt")  # Load official YOLOv8 nano model
        else:
            logger.info(f"Loading custom YOLO model from {model_path}...")
            self.model = YOLO(model_path)
        
        logger.info("YOLO model loaded successfully.")
    
    def refine_bounding_boxes(self, image, entity_boxes, confidence_threshold=0.3):
        """
        Refine CLIP's vague bounding boxes into precise detections using YOLO.
        
        Args:
            image: Input image as numpy array
            entity_boxes: Dictionary of entities with their vague bounding boxes from CLIP
            confidence_threshold: Minimum confidence score for YOLO detections
        
        Returns:
            dict: Dictionary with refined bounding boxes for each entity
        """
        logger.info("Starting YOLO refinement process...")
        
        # Dictionary to store refined results
        refined_results = {}
        
        # Process each entity separately
        for entity, boxes in entity_boxes.items():
            logger.info(f"Refining bounding boxes for entity: {entity}")
            
            # For each vague bounding box from CLIP, run YOLO on that region
            refined_boxes = []
            
            for box_info in boxes:
                # Extract the bounding box coordinates
                x1, y1, x2, y2 = box_info['bbox']
                
                # Ensure the bounding box is within image bounds
                x1 = max(0, int(x1))
                y1 = max(0, int(y1))
                x2 = min(image.shape[1], int(x2))
                y2 = min(image.shape[0], int(y2))
                
                # Crop the region for YOLO processing
                cropped_region = image[y1:y2, x1:x2]
                
                if cropped_region.size == 0:
                    logger.warning(f"Empty region detected for entity {entity}, skipping...")
                    continue
                
                # Run YOLO on the cropped region
                try:
                    results = self.model(cropped_region, verbose=False)
                    
                    # Process YOLO results
                    for result in results:
                        # Get bounding boxes in xyxy format (top-left, bottom-right)
                        if result.boxes is not None:
                            xyxy = result.boxes.xyxy  # top-left-x, top-left-y, bottom-right-x, bottom-right-y
                            confs = result.boxes.conf  # confidence scores
                            names = [result.names[cls.item()] for cls in result.boxes.cls.int()] if result.boxes.cls is not None else []
                        
                            # Filter results based on confidence threshold and class matching
                            for i, (box, conf, name) in enumerate(zip(xyxy, confs, names)):
                                # Lower the confidence threshold for initial detection, 
                                # since we'll be filtering by relevance anyway
                                if conf >= min(confidence_threshold, 0.2):  # At least 0.2 confidence
                                    # Convert to original image coordinates
                                    orig_x1 = int(box[0]) + x1
                                    orig_y1 = int(box[1]) + y1
                                    orig_x2 = int(box[2]) + x1
                                    orig_y2 = int(box[3]) + y1  # Fixed: was y2, should be + y1
                                    
                                    # Calculate center point
                                    center_x = (orig_x1 + orig_x2) / 2
                                    center_y = (orig_y1 + orig_y2) / 2
                                    
                                    # Check if this detection is relevant to the entity
                                    if self._is_entity_relevant(entity, name):
                                        refined_box = {
                                            'bbox': (orig_x1, orig_y1, orig_x2, orig_y2),
                                            'center': (center_x, center_y),
                                            'width': orig_x2 - orig_x1,
                                            'height': orig_y2 - orig_y1,
                                            'confidence': float(conf),
                                            'class': name,
                                            'source': 'yolo',
                                            'original_bbox': box_info['bbox'],
                                            'original_score': box_info['score']
                                        }
                                        refined_boxes.append(refined_box)
                                        
                                        logger.debug(f"Detected {name} with confidence {conf:.3f} at ({orig_x1}, {orig_y1}) to ({orig_x2}, {orig_y2})")
                        else:
                            logger.warning(f"No boxes detected in result for entity {entity}")
                except Exception as e:
                    logger.error(f"Error processing region for entity {entity}: {e}")
                    continue
            
            # Sort refined boxes by confidence (descending)
            refined_boxes.sort(key=lambda x: x['confidence'], reverse=True)
            
            refined_results[entity] = refined_boxes
            
            logger.info(f"Found {len(refined_boxes)} refined bounding boxes for entity: {entity}")
        
        # If no entities were refined, try a general detection of buildings/structures across the entire image
        if not any(refined_results[ent] for ent in refined_results):
            logger.info("No entities refined. Running general YOLO detection for buildings/structures...")
            refined_results = self._general_building_detection(image, confidence_threshold)
        
        logger.info("YOLO refinement process completed.")
        return refined_results
    
    def _general_building_detection(self, image, confidence_threshold=0.3):
        """
        Run general YOLO detection to find buildings/structures across the entire image.
        
        Args:
            image: Input image as numpy array
            confidence_threshold: Minimum confidence score for YOLO detections
        
        Returns:
            dict: Dictionary with building/structure detections
        """
        logger.info("Running general building detection...")
        refined_results = {}
        
        try:
            results = self.model(image, verbose=False)
            
            for result in results:
                if result.boxes is not None:
                    xyxy = result.boxes.xyxy
                    confs = result.boxes.conf
                    names = [result.names[cls.item()] for cls in result.boxes.cls.int()] if result.boxes.cls is not None else []
                    
                    # Look for building-related objects
                    building_classes = ['building', 'structure', 'house', 'construction', 'edifice', 'hut', 'cabin', 'shack', 'shed']
                    
                    for i, (box, conf, name) in enumerate(zip(xyxy, confs, names)):
                        if conf >= min(confidence_threshold, 0.2):
                            # Check if it's a building-related class
                            if any(bc in name.lower() for bc in building_classes) or 'building' in name.lower():
                                # Convert to image coordinates (no offset needed since full image)
                                orig_x1 = int(box[0])
                                orig_y1 = int(box[1])
                                orig_x2 = int(box[2])
                                orig_y2 = int(box[3])
                                
                                # Calculate center point
                                center_x = (orig_x1 + orig_x2) / 2
                                center_y = (orig_y1 + orig_y2) / 2
                                
                                refined_box = {
                                    'bbox': (orig_x1, orig_y1, orig_x2, orig_y2),
                                    'center': (center_x, center_y),
                                    'width': orig_x2 - orig_x1,
                                    'height': orig_y2 - orig_y1,
                                    'confidence': float(conf),
                                    'class': name,
                                    'source': 'yolo',
                                    'original_bbox': (orig_x1, orig_y1, orig_x2, orig_y2),
                                    'original_score': float(conf)
                                }
                                
                                # Add to 'building' category
                                if 'building' not in refined_results:
                                    refined_results['building'] = []
                                refined_results['building'].append(refined_box)
                                
                                logger.debug(f"Detected building {name} with confidence {conf:.3f} at ({orig_x1}, {orig_y1}) to ({orig_x2}, {orig_y2})")
        
        except Exception as e:
            logger.error(f"Error during general building detection: {e}")
        
        logger.info(f"Found {len(refined_results.get('building', []))} buildings/structures in general detection")
        return refined_results
    
    def _is_entity_relevant(self, entity, detected_class):
        """
        Check if a detected class is relevant to the requested entity.
        
        Args:
            entity: The entity name from the user prompt
            detected_class: The class name detected by YOLO
        
        Returns:
            bool: True if the detected class is relevant to the entity
        """
        # Simple relevance check - can be enhanced with more sophisticated logic
        entity_lower = entity.lower()
        detected_class_lower = detected_class.lower()
        
        # Handle multi-word entities like "tin shed"
        if ' ' in entity_lower:
            entity_parts = entity_lower.split()
            for part in entity_parts:
                if part in detected_class_lower or detected_class_lower in part:
                    return True
        
        # Check if detected class contains any part of the entity name
        if detected_class_lower in entity_lower or entity_lower in detected_class_lower:
            return True
        
        # Check for synonyms or related terms
        synonyms = {
            'building': ['house', 'structure', 'shed', 'hut', 'cabin', 'apartment', 'mansion', 'construction'],
            'roof': ['top', 'ceiling', 'cover', 'tile'],
            'shed': ['outbuilding', 'garage', 'barn', 'hut', 'cabin', 'structure', 'building'],
            'tin': ['metal', 'aluminum', 'steel', 'construction', 'roofing'],
            'village': ['building', 'structure', 'house', 'home', 'cabin', 'hut', 'shed']
        }
        
        for key, values in synonyms.items():
            if (entity_lower == key and detected_class_lower in values) or \
               (detected_class_lower == key and entity_lower in values):
                return True
        
        # Additional check: if it's a building-related term, it's likely relevant for village scenes
        building_related = ['building', 'structure', 'house', 'construction', 'edifice', 'hut', 'cabin', 'shack', 'shed']
        if entity_lower in ['village', 'building', 'structure'] and detected_class_lower in building_related:
            return True
        
        return False
    
    def get_precise_bounding_boxes(self, image, entities_and_attributes, confidence_threshold=0.5):
        """
        Get precise bounding boxes for entities using YOLO based on entities and attributes.
        
        Args:
            image: Input image as numpy array
            entities_and_attributes: Dictionary with entities and attributes (from CLIP detection)
            confidence_threshold: Minimum confidence score for YOLO detections
        
        Returns:
            dict: Dictionary with precise bounding boxes for each entity
        """
        logger.info("Getting precise bounding boxes for entities and attributes...")
        
        # In this version, entities_and_attributes contains both entities and attributes
        # We'll process the entities from the CLIP detection results
        refined_results = self.refine_bounding_boxes(image, entities_and_attributes, confidence_threshold)
        
        logger.info("Precise bounding boxes obtained.")
        return refined_results