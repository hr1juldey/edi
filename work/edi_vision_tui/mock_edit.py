"""
Mock editing functionality for EDI vision subsystem.
Simulates the edit request process without requiring external services.
"""

import time
import random
from pathlib import Path
import shutil
from typing import Dict, Any
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageOps


class MockEditProcessor:
    """Class to handle mock image editing operations"""
    
    def __init__(self):
        self.edit_history = []
        self.current_session_id = None
    
    def process_edit_request(self, 
                           image_path: str, 
                           prompt: str, 
                           masks: list = None) -> Dict[str, Any]:
        """
        Process a mock edit request and return the result.
        """
        start_time = time.time()
        
        # Create output path based on input
        input_path = Path(image_path)
        session_id = f"session_{int(time.time())}"
        output_path = input_path.parent / f"edited_{session_id}_{input_path.name}"
        
        # Process the edit based on the prompt
        success, error_msg = self._apply_edit_to_image(image_path, str(output_path), prompt, masks)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        result = {
            'success': success,
            'input_path': image_path,
            'output_path': str(output_path) if output_path.exists() else None,
            'prompt': prompt,
            'session_id': session_id,
            'processing_time': processing_time,
            'error': error_msg,
            'edit_metadata': {
                'prompt_parts': self._parse_prompt(prompt),
                'masks_applied': len(masks) if masks else 0
            }
        }
        
        if success:
            self.edit_history.append(result)
        
        return result
    
    def _parse_prompt(self, prompt: str) -> list:
        """Parse the prompt to extract key components"""
        # This is a simplified prompt parsing - in reality this would be more sophisticated
        import re
        
        # Extract color changes: "to red", "to green", etc.
        color_changes = re.findall(r'to\s+(red|blue|green|yellow|orange|purple|pink|brown|black|white|gray|grey)', prompt, re.IGNORECASE)
        
        # Extract entities: "tin sheds", "car", etc.
        entities = re.findall(r'(\w+\s+\w+|\w+)\s+(?:in\s+the\s+image|to)', prompt, re.IGNORECASE)
        
        return {
            'color_changes': color_changes,
            'entities': entities,
            'raw_prompt': prompt
        }
    
    def _apply_edit_to_image(self, input_path: str, output_path: str, prompt: str, masks: list = None) -> tuple:
        """
        Apply mock edits to the image based on the prompt.
        """
        try:
            # Load the input image
            img = Image.open(input_path).convert('RGB')
            img_array = np.array(img)
            
            # Determine what kind of edit to apply based on the prompt
            edit_type = self._determine_edit_type(prompt)
            
            # Apply the edit
            if edit_type == "color_change":
                processed_img = self._apply_color_change(img_array, prompt, masks)
            elif edit_type == "style_change":
                processed_img = self._apply_style_change(img_array, prompt, masks)
            elif edit_type == "enhancement":
                processed_img = self._apply_enhancement(img_array, prompt, masks)
            else:
                # Default: apply a random edit
                processed_img = self._apply_random_edit(img_array, prompt, masks)
            
            # Save the processed image
            processed_pil = Image.fromarray(processed_img.astype('uint8'))
            processed_pil.save(output_path)
            
            return True, None
            
        except Exception as e:
            return False, str(e)
    
    def _determine_edit_type(self, prompt: str) -> str:
        """Determine the type of edit based on the prompt"""
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ["color", "red", "blue", "green", "yellow", "change to", "make", "turn"]):
            return "color_change"
        elif any(word in prompt_lower for word in ["style", "artistic", "painting", "sketch", "cartoon", "realistic"]):
            return "style_change"
        elif any(word in prompt_lower for word in ["enhance", "sharpen", "brighten", "contrast", "vivid"]):
            return "enhancement"
        else:
            return "random"  # Default to random edit
    
    def _apply_color_change(self, img_array: np.ndarray, prompt: str, masks: list = None) -> np.ndarray:
        """Apply color change based on the prompt"""
        # Extract target color from prompt
        target_color = self._extract_color_from_prompt(prompt)
        
        if masks and len(masks) > 0:
            # Apply color change to masked regions
            result_img = img_array.copy()
            
            for mask_info in masks:
                # For mock purposes, we'll use the bbox to apply changes
                x1, y1, x2, y2 = mask_info['bbox']
                
                # Ensure coordinates are within image bounds
                y1 = max(0, min(y1, img_array.shape[0] - 1))
                y2 = max(0, min(y2, img_array.shape[0] - 1))
                x1 = max(0, min(x1, img_array.shape[1] - 1))
                x2 = max(0, min(x2, img_array.shape[1] - 1))
                
                if y1 < y2 and x1 < x2:  # Valid bbox
                    # Extract the region
                    region = result_img[y1:y2, x1:x2]
                    
                    # Apply color transformation
                    if target_color == "red":
                        region[:, :, 0] = np.clip(region[:, :, 0] * 1.3, 0, 255)
                        region[:, :, 1] = np.clip(region[:, :, 1] * 0.7, 0, 255)
                        region[:, :, 2] = np.clip(region[:, :, 2] * 0.7, 0, 255)
                    elif target_color == "blue":
                        region[:, :, 0] = np.clip(region[:, :, 0] * 0.7, 0, 255)
                        region[:, :, 1] = np.clip(region[:, :, 1] * 0.7, 0, 255)
                        region[:, :, 2] = np.clip(region[:, :, 2] * 1.3, 0, 255)
                    elif target_color == "green":
                        region[:, :, 0] = np.clip(region[:, :, 0] * 0.7, 0, 255)
                        region[:, :, 1] = np.clip(region[:, :, 1] * 1.3, 0, 255)
                        region[:, :, 2] = np.clip(region[:, :, 2] * 0.7, 0, 255)
                    elif target_color == "yellow":
                        region[:, :, 0] = np.clip(region[:, :, 0] * 1.2, 0, 255)
                        region[:, :, 1] = np.clip(region[:, :, 1] * 1.2, 0, 255)
                        region[:, :, 2] = np.clip(region[:, :, 2] * 0.6, 0, 255)
                    elif target_color == "purple":
                        region[:, :, 0] = np.clip(region[:, :, 0] * 1.1, 0, 255)
                        region[:, :, 1] = np.clip(region[:, :, 1] * 0.6, 0, 255)
                        region[:, :, 2] = np.clip(region[:, :, 2] * 1.1, 0, 255)
                    
                    # Place the modified region back
                    result_img[y1:y2, x1:x2] = region
        else:
            # Apply to entire image if no masks provided
            result_img = img_array.copy()
            
            if target_color == "red":
                result_img[:, :, 0] = np.clip(result_img[:, :, 0] * 1.3, 0, 255)
                result_img[:, :, 1] = np.clip(result_img[:, :, 1] * 0.7, 0, 255)
                result_img[:, :, 2] = np.clip(result_img[:, :, 2] * 0.7, 0, 255)
            elif target_color == "blue":
                result_img[:, :, 0] = np.clip(result_img[:, :, 0] * 0.7, 0, 255)
                result_img[:, :, 1] = np.clip(result_img[:, :, 1] * 0.7, 0, 255)
                result_img[:, :, 2] = np.clip(result_img[:, :, 2] * 1.3, 0, 255)
            elif target_color == "green":
                result_img[:, :, 0] = np.clip(result_img[:, :, 0] * 0.7, 0, 255)
                result_img[:, :, 1] = np.clip(result_img[:, :, 1] * 1.3, 0, 255)
                result_img[:, :, 2] = np.clip(result_img[:, :, 2] * 0.7, 0, 255)
            elif target_color == "yellow":
                result_img[:, :, 0] = np.clip(result_img[:, :, 0] * 1.2, 0, 255)
                result_img[:, :, 1] = np.clip(result_img[:, :, 1] * 1.2, 0, 255)
                result_img[:, :, 2] = np.clip(result_img[:, :, 2] * 0.6, 0, 255)
        
        return result_img
    
    def _apply_style_change(self, img_array: np.ndarray, prompt: str, masks: list = None) -> np.ndarray:
        """Apply style change based on the prompt"""
        # For mock purposes, apply a random style transformation
        pil_img = Image.fromarray(img_array.astype('uint8'), 'RGB')
        
        if "cartoon" in prompt.lower():
            # Simulate a cartoon effect by reducing colors and increasing contrast
            pil_img = pil_img.quantize(colors=32).convert('RGB')
            enhancer = ImageEnhance.Contrast(pil_img)
            pil_img = enhancer.enhance(1.5)
        elif "sketch" in prompt.lower() or "pencil" in prompt.lower():
            # Convert to grayscale and invert to simulate a pencil sketch
            pil_img = pil_img.convert('L')
            pil_img = ImageOps.invert(pil_img)
            pil_img = pil_img.convert('RGB')
        elif "painting" in prompt.lower() or "oil" in prompt.lower():
            # Simulate painting by increasing saturation
            enhancer = ImageEnhance.Color(pil_img)
            pil_img = enhancer.enhance(1.8)
        elif "vintage" in prompt.lower() or "old" in prompt.lower():
            # Apply vintage effect (sepia tone)
            img_array = np.array(pil_img)
            img_array = img_array.astype(np.float32)
            img_array[:, :, 0] = np.clip(img_array[:, :, 0] * 0.393 + img_array[:, :, 1] * 0.769 + img_array[:, :, 2] * 0.189, 0, 255)  # R
            img_array[:, :, 1] = np.clip(img_array[:, :, 0] * 0.349 + img_array[:, :, 1] * 0.686 + img_array[:, :, 2] * 0.168, 0, 255)  # G
            img_array[:, :, 2] = np.clip(img_array[:, :, 0] * 0.272 + img_array[:, :, 1] * 0.534 + img_array[:, :, 2] * 0.131, 0, 255)  # B
            return img_array.astype(np.uint8)
        
        return np.array(pil_img)
    
    def _apply_enhancement(self, img_array: np.ndarray, prompt: str, masks: list = None) -> np.ndarray:
        """Apply enhancement based on the prompt"""
        pil_img = Image.fromarray(img_array.astype('uint8'), 'RGB')
        
        if "sharpen" in prompt.lower():
            from PIL import ImageFilter
            pil_img = pil_img.filter(ImageFilter.SHARPEN)
        elif "brighten" in prompt.lower() or "lighten" in prompt.lower():
            enhancer = ImageEnhance.Brightness(pil_img)
            pil_img = enhancer.enhance(1.3)
        elif "contrast" in prompt.lower():
            enhancer = ImageEnhance.Contrast(pil_img)
            pil_img = enhancer.enhance(1.4)
        elif "vivid" in prompt.lower() or "saturation" in prompt.lower():
            enhancer = ImageEnhance.Color(pil_img)
            pil_img = enhancer.enhance(1.5)
        
        return np.array(pil_img)
    
    def _apply_random_edit(self, img_array: np.ndarray, prompt: str, masks: list = None) -> np.ndarray:
        """Apply a random edit as fallback"""
        # Randomly choose an edit type
        edit_types = ["color_change", "style_change", "enhancement"]
        chosen_type = random.choice(edit_types)
        
        if chosen_type == "color_change":
            # Apply a random color change
            colors = ["red", "blue", "green", "yellow", "purple"]
            random_color = random.choice(colors)
            # Create a mock prompt with the random color
            mock_prompt = f"change some parts to {random_color}"
            return self._apply_color_change(img_array, mock_prompt, masks)
        elif chosen_type == "style_change":
            styles = ["cartoon", "sketch", "painting", "vintage"]
            random_style = random.choice(styles)
            mock_prompt = f"make it look like {random_style}"
            return self._apply_style_change(img_array, mock_prompt, masks)
        else:  # enhancement
            enhancements = ["sharpen", "brighten", "contrast", "vivid"]
            random_enhancement = random.choice(enhancements)
            mock_prompt = f"apply {random_enhancement} effect"
            return self._apply_enhancement(img_array, mock_prompt, masks)
    
    def _extract_color_from_prompt(self, prompt: str) -> str:
        """Extract target color from the prompt"""
        prompt_lower = prompt.lower()
        
        color_map = {
            "red": ["red", "crimson", "scarlet", "ruby"],
            "blue": ["blue", "navy", "sapphire", "azure"],
            "green": ["green", "emerald", "jade", "olive"],
            "yellow": ["yellow", "gold", "amber", "lemon"],
            "purple": ["purple", "violet", "magenta", "lavender"],
            "pink": ["pink", "rose", "coral", "fuchsia"],
            "orange": ["orange", "amber", "peach", "apricot"],
            "brown": ["brown", "tan", "chestnut", "copper"],
            "black": ["black", "ebony", "charcoal", "obsidian"],
            "white": ["white", "ivory", "alabaster", "snow"],
            "gray": ["gray", "grey", "silver", "slate"]
        }
        
        for color, variations in color_map.items():
            for variation in variations:
                if variation in prompt_lower:
                    return color
        
        # Default to green if no color found
        return "green"


# Singleton instance for mock edit processing
mock_processor = MockEditProcessor()


def send_mock_edit_request(image_path: str, prompt: str, masks: list = None) -> Dict[str, Any]:
    """
    Public function to send a mock edit request.
    """
    return mock_processor.process_edit_request(image_path, prompt, masks)