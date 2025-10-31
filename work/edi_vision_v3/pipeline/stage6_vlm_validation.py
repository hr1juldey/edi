"""Stage 6: VLM Validation

This module uses a Vision Language Model (VLM) to validate that the detected
entity masks match the user's intent from the original request.
"""

import logging
import numpy as np
import cv2
import json
import requests
from typing import List
from dataclasses import dataclass
import base64


@dataclass
class ValidationResult:
    """Container for VLM validation results.

    Attributes:
        covers_all_targets: Whether masks cover all target entities in image
        confidence: Overall confidence in validation (0.0-1.0)
        feedback: Textual feedback about the validation
        target_coverage: Fraction of target objects covered by masks (0.0-1.0)
        false_positive_ratio: Fraction of mask area on non-target objects
        missing_targets: Description of missed target objects
        suggestions: Suggestions for improving mask quality
    """
    covers_all_targets: bool
    confidence: float
    feedback: str
    target_coverage: float
    false_positive_ratio: float
    missing_targets: str
    suggestions: List[str]


def create_validation_overlay(image: np.ndarray, masks: List[np.ndarray]) -> np.ndarray:
    """Create an overlay image highlighting the masks on the original image.

    Args:
        image: Original RGB image (H x W x 3)
        masks: List of binary masks

    Returns:
        Overlay image with masks highlighted in red
    """
    # Create overlay by making the image slightly transparent
    overlay = image.copy().astype(np.float32)
    
    # For each mask, highlight it in red
    for mask in masks:
        # Where mask is 1, apply semi-transparent red overlay
        alpha = 0.3  # Transparency for the base image
        mask_overlay = np.zeros_like(overlay)
        mask_overlay[mask > 0] = [255, 0, 0]  # Red color
        
        # Blend the mask with the original image
        overlay[mask > 0] = overlay[mask > 0] * (1 - alpha) + mask_overlay[mask > 0] * alpha

    return overlay.astype(np.uint8)


def validate_with_vlm(image: np.ndarray,
                      entity_masks: List[np.ndarray],
                      user_prompt: str,
                      ollama_url: str = "http://localhost:11434/api/generate") -> ValidationResult:
    """Validate entity masks using a Vision Language Model.

    Args:
        image: Original RGB image (H x W x 3)
        entity_masks: List of binary masks corresponding to detected entities
        user_prompt: Original user request (e.g., "turn blue roofs green")
        ollama_url: URL for Ollama VLM API endpoint

    Returns:
        ValidationResult with validation metrics
    """
    logging.info(f"Starting VLM validation for prompt: '{user_prompt}'")
    logging.info(f"Validating {len(entity_masks)} entity masks")

    if len(entity_masks) == 0:
        return ValidationResult(
            covers_all_targets=False,
            confidence=0.0,
            feedback="No masks provided for validation",
            target_coverage=0.0,
            false_positive_ratio=0.0,
            missing_targets="No entities detected",
            suggestions=["Re-run detection pipeline", "Check if target entities exist in image"]
        )

    # Create overlay image showing the masks
    overlay_image = create_validation_overlay(image, entity_masks)

    # Convert overlay to base64 for API request
    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR))
    overlay_base64 = base64.b64encode(buffer).decode('utf-8')

    # Prepare the VLM prompt
    vlm_prompt = (
        f"Please analyze this image where red areas indicate detected regions. "
        f"The user requested: '{user_prompt}'. "
        f"Validate if the red masked areas correctly identify all target entities and nothing else. "
        f"Respond in JSON format with the following structure: "
        f"{{"
        f"  \"covers_all_targets\": <bool>,"
        f"  \"confidence\": <float 0.0-1.0>,"
        f"  \"feedback\": \"<str>\","
        f"  \"target_coverage\": <float 0.0-1.0>,"
        f"  \"false_positive_ratio\": <float 0.0-1.0>,"
        f"  \"missing_targets\": \"<str>\","
        f"  \"suggestions\": [\"<str>\"...]"
        f"}}"
    )

    # Prepare the API request
    payload = {
        "model": "qwen2.5vl:7b",
        "prompt": vlm_prompt,
        "images": [overlay_base64],
        "stream": False
    }

    try:
        # Call Ollama VLM API
        response = requests.post(ollama_url, json=payload, timeout=60)
        response.raise_for_status()
        
        # Parse the response
        result = response.json()
        response_text = result.get('response', '')
        
        logging.info(f"VLM response: {response_text}")
        
        # Extract JSON from response
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start != -1 and json_end != 0:
            json_str = response_text[json_start:json_end]
            vlm_result = json.loads(json_str)
            
            # Create ValidationResult from VLM output
            validation_result = ValidationResult(
                covers_all_targets=vlm_result.get('covers_all_targets', False),
                confidence=vlm_result.get('confidence', 0.0),
                feedback=vlm_result.get('feedback', 'No feedback provided'),
                target_coverage=vlm_result.get('target_coverage', 0.0),
                false_positive_ratio=vlm_result.get('false_positive_ratio', 0.0),
                missing_targets=vlm_result.get('missing_targets', 'No missing targets identified'),
                suggestions=vlm_result.get('suggestions', [])
            )
            
            logging.info(f"Validation complete. Confidence: {validation_result.confidence}, "
                        f"Coverage: {validation_result.target_coverage}, "
                        f"False positive ratio: {validation_result.false_positive_ratio}")
            
            return validation_result
        
        else:
            logging.error(f"Could not extract JSON from VLM response: {response_text}")
            return ValidationResult(
                covers_all_targets=False,
                confidence=0.3,
                feedback="Could not parse VLM response",
                target_coverage=0.0,
                false_positive_ratio=0.5,
                missing_targets="Unable to determine from VLM response",
                suggestions=["Try again with clearer prompt"]
            )
    
    except requests.exceptions.RequestException as e:
        logging.error(f"Error calling Ollama VLM API: {e}")
        return ValidationResult(
            covers_all_targets=False,
            confidence=0.0,
            feedback=f"API request failed: {str(e)}",
            target_coverage=0.0,
            false_positive_ratio=0.0,
            missing_targets="API request failed",
            suggestions=["Check Ollama service is running", "Verify model name is correct"]
        )
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing VLM JSON response: {e}")
        return ValidationResult(
            covers_all_targets=False,
            confidence=0.2,
            feedback="Could not parse VLM response JSON",
            target_coverage=0.0,
            false_positive_ratio=0.5,
            missing_targets="Could not parse VLM response",
            suggestions=["Try again with the same request"]
        )
    except Exception as e:
        logging.error(f"Unexpected error in VLM validation: {e}")
        return ValidationResult(
            covers_all_targets=False,
            confidence=0.1,
            feedback=f"Unexpected error: {str(e)}",
            target_coverage=0.0,
            false_positive_ratio=0.0,
            missing_targets="Processing error",
            suggestions=["Retry validation", "Check system resources"]
        )