import json
import cv2
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple
import dspy


class EditLogGenerator:
    """
    Generate JSON edit log by detecting changes in edited image.
    """
    def __init__(self):
        self.edit_log = {
            'original_prompt': '',
            'detected_changes': [],
            'unchanged_regions': [],
            'confidence_scores': {}
        }
    
    def generate_edit_log(self, original_image: np.ndarray, edited_image: np.ndarray, original_prompt: str) -> Dict:
        """
        Generate JSON log of what went in, what was required to be done, and what came out.
        
        Args:
            original_image: Original image as numpy array
            edited_image: Edited image as numpy array
            original_prompt: Original user prompt
        
        Returns:
            dict: JSON edit log
        """
        self.edit_log['original_prompt'] = original_prompt
        
        # Detect changes in edited image using image difference
        changes = self._detect_changes(original_image, edited_image)
        
        self.edit_log['detected_changes'] = changes
        
        # Generate confidence scores for each change
        self.edit_log['confidence_scores'] = self._calculate_confidence_scores(changes, original_image)
        
        return self.edit_log
    
    def _detect_changes(self, original_image: np.ndarray, edited_image: np.ndarray) -> List[Dict]:
        """
        Detect changes between original and edited images.
        
        Args:
            original_image: Original image as numpy array
            edited_image: Edited image as numpy array
        
        Returns:
            list: List of detected changes
        """
        # Calculate difference between images
        if original_image.shape != edited_image.shape:
            raise ValueError("Original and edited images must have the same shape")
        
        # Calculate absolute difference
        diff = cv2.absdiff(original_image, edited_image)
        
        # Convert to grayscale if needed for thresholding
        if len(diff.shape) == 3:
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        else:
            diff_gray = diff
        
        # Apply adaptive threshold to identify significant changes
        # Using a lower threshold to be more sensitive to changes
        _, thresh = cv2.threshold(diff_gray, 15, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to clean up the threshold result
        kernel = np.ones((3,3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Find contours of changed regions
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        changes = []
        for i, contour in enumerate(contours):
            # Get bounding box of changed region
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate area of changed region
            area = cv2.contourArea(contour)
            
            # Only include changes that are above a minimum size
            if area > 100:  # Minimum area threshold
                # Calculate more detailed metrics for the change
                mask = np.zeros_like(diff_gray)
                cv2.fillPoly(mask, [contour], 255)
                
                # Get the mean difference within the changed region
                mean_diff = cv2.mean(diff_gray, mask=mask)[0]
                
                # Add to changes list
                changes.append({
                    'id': f'change_{i}',
                    'bbox': [x, y, x + w, y + h],
                    'area': area,
                    'mean_difference': float(mean_diff),
                    'description': f'Region with significant changes (area: {area}, mean_diff: {mean_diff:.2f})'
                })
        
        return changes
    
    def _calculate_confidence_scores(self, changes: List[Dict], original_image: np.ndarray) -> Dict:
        """
        Calculate confidence scores for each change based on multiple factors.
        
        Args:
            changes: List of detected changes
            original_image: Original image for spatial context
        
        Returns:
            dict: Confidence scores for each change
        """
        confidence_scores = {}
        
        # Calculate image dimensions for spatial analysis
        img_height = original_image.shape[0]
        
        for change in changes:
            # Calculate confidence score based on multiple factors
            area = change['area']
            bbox = change['bbox']
            mean_diff = change.get('mean_difference', 0)
            
            # Start with area-based confidence (normalized to 0-1 range)
            # Use image size to normalize area confidence
            total_pixels = original_image.shape[0] * original_image.shape[1]
            area_confidence = min(1.0, area / (total_pixels * 0.1))  # Max 10% of image size gets 1.0 area confidence
            
            # Spatial confidence - based on position in image
            # Weights for different image regions (ground level changes more likely to be intended)
            y_center = (bbox[1] + bbox[3]) / 2
            
            if y_center > img_height * 0.7:  # Bottom 30% of image
                spatial_confidence = 1.0  # Ground level changes are most likely intended
            elif y_center < img_height * 0.3:  # Top 30% of image
                spatial_confidence = 0.3  # Sky changes are less likely intended
            else:
                spatial_confidence = 0.7  # Middle area changes are moderately likely intended
            
            # Difference magnitude confidence - higher differences get higher confidence
            diff_confidence = min(1.0, mean_diff / 50.0)  # Normalize to 0-1 based on max expected difference
            
            # Combine confidences with weights
            # Area: 30%, Spatial: 40%, Difference: 30%
            confidence = (0.3 * area_confidence + 0.4 * spatial_confidence + 0.3 * diff_confidence)
            
            confidence_scores[change['id']] = min(1.0, confidence)
        
        return confidence_scores


class DSPyRefiner:
    """
    Use DSPy to refine prompts based on validation results.
    """
    def __init__(self):
        # Initialize DSPy settings if not already done
        try:
            import dspy
            # Try to configure DSPy with Ollama as the LLM
            try:
                lm = dspy.LM('ollama_chat/qwen3:8b', api_base='http://localhost:11434', api_key='')
                dspy.settings.configure(lm=lm)
            except Exception as e:
                print(f"[!] Ollama configuration failed: {e}, trying alternative model")
                try:
                    lm = dspy.LM('ollama_chat/qwen2.5vl:7b', api_base='http://localhost:11434', api_key='')
                    dspy.settings.configure(lm=lm)
                except Exception as e2:
                    print(f"[!] Alternative Ollama configuration failed: {e2}, using fallback")
        except ImportError:
            print("[!] DSPy not available, using fallback refinement")
    
    def refine_prompt(self, original_prompt: str, edit_log: Dict, validation_metrics: Dict) -> str:
        """
        Refine prompt based on validation results.
        
        Args:
            original_prompt: Original user prompt
            edit_log: JSON edit log
            validation_metrics: Validation metrics
        
        Returns:
            str: Refined prompt
        """
        # Create a refined prompt based on the edit log and validation metrics
        refined_prompt = self._create_refined_prompt(original_prompt, edit_log, validation_metrics)
        
        return refined_prompt
    
    def _create_refined_prompt(self, original_prompt: str, edit_log: Dict, validation_metrics: Dict) -> str:
        """
        Create a refined prompt based on the edit log and validation results.
        
        Args:
            original_prompt: Original user prompt
            edit_log: JSON edit log
            validation_metrics: Validation metrics
        
        Returns:
            str: Refined prompt
        """
        # Start with the original prompt
        refined_prompt = original_prompt
        
        # Add context about what was changed and what wasn't
        if edit_log.get('detected_changes'):
            if len(edit_log['detected_changes']) > 0:
                changes_str = ', '.join([f"{change.get('description', 'region')}" for change in edit_log['detected_changes'][:3]])  # Only show first 3 changes
                if len(edit_log['detected_changes']) > 3:
                    changes_str += f", and {len(edit_log['detected_changes']) - 3} more changes"
                refined_prompt += f"\nNote: {len(edit_log['detected_changes'])} changes were detected in the edited image: {changes_str}"
            else:
                refined_prompt += f"\nNote: No significant changes were detected in the edited image."
        else:
            refined_prompt += f"\nNote: Change detection was not performed."
        
        # Add validation metrics
        if validation_metrics and validation_metrics.get('success', False):
            psnr = validation_metrics.get('psnr', 'N/A')
            delta_e = validation_metrics.get('delta_e', 'N/A')
            
            if psnr != 'N/A':
                refined_prompt += f"\nPSNR (Peak Signal-to-Noise Ratio): {psnr:.2f} dB"
                if psnr < 20:
                    refined_prompt += " (Low quality, improve edit quality)"
                elif psnr < 30:
                    refined_prompt += " (Moderate quality)"
                else:
                    refined_prompt += " (High quality)"
            
            if delta_e != 'N/A':
                refined_prompt += f"\nDelta E (Color difference): {delta_e:.2f}"
                if delta_e > 15:
                    refined_prompt += " (High color difference, adjust for accuracy)"
                elif delta_e > 5:
                    refined_prompt += " (Moderate color difference)"
                else:
                    refined_prompt += " (Low color difference, colors match well)"
        else:
            refined_prompt += f"\nValidation metrics: Not available"
        
        # Analyze detected changes for spatial issues
        if edit_log.get('detected_changes'):
            # Get image dimensions from one of the changes
            if edit_log['detected_changes']:
                # Calculate a reasonable image height based on the changes
                # We'll use the 1000px assumption only as fallback
                sample_change = edit_log['detected_changes'][0]
                bbox = sample_change.get('bbox', [0, 0, 100, 100])
                img_height = max(1000, bbox[3])  # Use max of 1000 or the change's bottom coordinate
                
                # Check for sky-region changes (top 25% of image)
                sky_changes = [change for change in edit_log['detected_changes'] 
                              if change.get('bbox', [0,0,0,0])[1] < img_height * 0.25]
                
                if sky_changes:
                    refined_prompt += f"\nIMPORTANT: {len(sky_changes)} changes detected in sky region (top 25% of image). Avoid modifying sky unless specifically requested. Focus edits on ground-level structures."
                else:
                    refined_prompt += f"\nGood: No unwanted changes detected in sky region."
        
        # Add feedback based on the original prompt intent
        original_lower = original_prompt.lower()
        has_color_request = any(color in original_lower for color in ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'brown', 'black', 'white', 'gray', 'grey'])
        
        if has_color_request and validation_metrics.get('delta_e', 0) > 15:
            refined_prompt += f"\nSince your prompt specified a color change, consider adjusting the edit to better match the target color."
        
        return refined_prompt