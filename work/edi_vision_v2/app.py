#!/usr/bin/env python3
"""
EDI Vision Pipeline - Command Line Interface

This module provides the CLI interface for the vision pipeline,
designed for use by AI agents, scripts, and automation.
"""

import argparse
import logging
import sys
import os
import yaml
from pathlib import Path
import numpy as np
import cv2
from typing import Dict, Any

# Add the project directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.orchestrator import VisionPipeline


def setup_logging(verbose: bool, debug: bool):
    """Setup logging based on verbosity level."""
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING

    logging.basicConfig(
        level=level,
        format='[%(levelname)s] %(message)s'
    )


def validate_image_path(image_path: str) -> None:
    """Validate that image path exists and is readable."""
    path = Path(image_path)
    if not path.exists():
        print("❌ Error: Image file not found")
        print(f"   Path: {image_path}")
        print("   Suggestion: Check file path and try again")
        sys.exit(1)
    
    if not path.is_file():
        print("❌ Error: Path is not a file")
        print(f"   Path: {image_path}")
        sys.exit(1)
    
    # Check if file is a supported image format
    supported_formats = ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff']
    if path.suffix.lower() not in supported_formats:
        print("❌ Error: Invalid image format")
        print(f"   File: {path.name}")
        print(f"   Supported formats: {', '.join(supported_formats)}")
        sys.exit(1)
    
    # Try to read the image
    try:
        test_image = cv2.imread(str(path))
        if test_image is None:
            print("❌ Error: Cannot read image file")
            print(f"   File: {path.name}")
            print("   Suggestion: Ensure file is a valid image")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Error: Cannot read image file - {str(e)}")
        sys.exit(1)


def validate_prompt(prompt: str) -> None:
    """Validate that prompt is not empty."""
    if not prompt or not prompt.strip():
        print("❌ Error: Prompt cannot be empty")
        print("   Usage: --prompt \"edit blue roofs\"")
        sys.exit(1)


def validate_thresholds(args: argparse.Namespace) -> None:
    """Validate that threshold values are within the 0.0-1.0 range."""
    if not 0.0 <= args.color_threshold <= 1.0:
        print("❌ Error: Color threshold must be between 0.0 and 1.0")
        print(f"   Value: {args.color_threshold}")
        sys.exit(1)
    
    if not 0.0 <= args.clip_threshold <= 1.0:
        print("❌ Error: CLIP threshold must be between 0.0 and 1.0")
        print(f"   Value: {args.clip_threshold}")
        sys.exit(1)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"❌ Error: Cannot read config file - {str(e)}")
        sys.exit(1)


def get_default_config() -> Dict:
    """Return default configuration."""
    return {
        'pipeline': {
            'enable_validation': True,
            'save_intermediate': False,
            'output_dir': "logs"
        },
        'sam': {
            'min_area': 500,
            'color_overlap_threshold': 0.5
        },
        'clip': {
            'similarity_threshold': 0.22
        },
        'logging': {
            'level': 'WARNING',
            'format': '[%(levelname)s] %(message)s'
        }
    }


def create_output_visualization(image_path: str, result: Dict[str, Any]) -> np.ndarray:
    """
    Create a 2x2 grid visualization:
    - Top-left: Original image
    - Top-right: Stage 2 color mask
    - Bottom-left: Stage 3 SAM masks (grid)
    - Bottom-right: Final entity masks (colored overlays)
    """
    # Load original image
    original = cv2.imread(image_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    # Resize image for display if too large (to ensure consistent grid size)
    max_display_size = 512
    h, w = original.shape[:2]
    scale = min(max_display_size / w, max_display_size / h)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        original = cv2.resize(original, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create color mask visualization
    # For now, make a placeholder that shows the color coverage
    color_mask_viz = original.copy()
    # This is a placeholder - in a real implementation you'd use the actual color mask
    if 'metadata' in result and 'color_coverage' in result['metadata']:
        # Show a simple visualization of coverage
        gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        _, binary_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        color_mask_viz[binary_mask > 0] = [color_mask_viz[i, j] // 2 + [127, 0, 0] 
                                          for i in range(binary_mask.shape[0]) 
                                          for j in range(binary_mask.shape[1]) 
                                          if binary_mask[i, j] > 0]
        # Actually implement a proper color mask visualization
        color_mask_viz = np.zeros_like(original)
        color_mask_viz[:, :] = [127, 127, 127]  # Gray background
        
        # Use the actual color mask if available, otherwise create a generic one
        if hasattr(create_output_visualization, 'color_mask') and create_output_visualization.color_mask is not None:
            color_mask_resized = cv2.resize(create_output_visualization.color_mask, 
                                          (original.shape[1], original.shape[0]))
            color_mask_viz[color_mask_resized > 0] = [0, 0, 255]  # Red mask
    else:
        color_mask_viz = np.zeros_like(original)
        color_mask_viz[:, :] = [200, 200, 200]  # Light gray for no mask
    
    # Create SAM masks visualization (placeholder grid of the first few masks)
    sam_masks_viz = original.copy()
    if result.get('metadata', {}).get('sam_masks_count', 0) > 0:
        # This would use the actual SAM masks, but we'll create a simple overlay
        sam_masks_viz = original.copy()
        # For demonstration, draw red rectangles roughly where masks might be
        h, w = sam_masks_viz.shape[:2]
        step_x, step_y = w // 5, h // 5
        for i in range(min(5, result['metadata']['sam_masks_count'])):
            x, y = (i % 3) * step_x, (i // 3) * step_y
            cv2.rectangle(sam_masks_viz, (x, y), (x + step_x//2, y + step_y//2), (255, 0, 0), 2)
    else:
        sam_masks_viz = np.zeros_like(original)
        sam_masks_viz[:, :] = [200, 200, 200]  # Light gray for no masks
    
    # Create final entity masks visualization
    final_masks_viz = original.copy()
    entity_masks = result.get('entity_masks', [])
    
    if entity_masks:
        # Use different colors for each entity mask (rainbow palette)
        colors = [
            (255, 0, 0),     # Red
            (0, 255, 0),     # Green
            (0, 0, 255),     # Blue
            (255, 255, 0),   # Yellow
            (255, 0, 255),   # Magenta
            (0, 255, 255),   # Cyan
            (255, 165, 0),   # Orange
            (128, 0, 128),   # Purple
            (255, 192, 203), # Pink
            (165, 42, 42),   # Brown
            (0, 128, 0),     # Dark Green
            (0, 0, 128),     # Dark Blue
            (128, 128, 0),   # Olive
            (128, 0, 0),     # Maroon
            (0, 128, 128),   # Teal
        ]
        
        for i, entity_mask in enumerate(entity_masks):
            if i >= len(colors):
                break
            color = colors[i % len(colors)]
            # Create a mask overlay
            mask = entity_mask.mask
            # Resize mask to match image dimensions if needed
            if mask.shape[:2] != final_masks_viz.shape[:2]:
                mask = cv2.resize(mask.astype(np.uint8), 
                                 (final_masks_viz.shape[1], final_masks_viz.shape[0]), 
                                 interpolation=cv2.INTER_NEAREST)
            
            # Apply colored overlay where mask is True
            final_masks_viz[mask > 0] = final_masks_viz[mask > 0] * 0.6 + np.array(color) * 0.4
    
        # Add entity count as title
        cv2.putText(final_masks_viz, f"{len(entity_masks)} entities detected", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    else:
        final_masks_viz = np.zeros_like(original)
        final_masks_viz[:, :] = [200, 200, 200]  # Light gray for no entities
    
    # Create 2x2 grid
    height, width = original.shape[:2]
    
    # Resize all panels to the same size if needed
    def resize_to_same(img, target_h, target_w):
        if img.shape[0] != target_h or img.shape[1] != target_w:
            return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        return img
    
    color_mask_viz = resize_to_same(color_mask_viz, height, width)
    sam_masks_viz = resize_to_same(sam_masks_viz, height, width)
    final_masks_viz = resize_to_same(final_masks_viz, height, width)
    
    # Create the grid
    top_row = np.hstack((original, color_mask_viz))
    bottom_row = np.hstack((sam_masks_viz, final_masks_viz))
    grid = np.vstack((top_row, bottom_row))
    
    return grid


def print_summary(result: Dict[str, Any]) -> None:
    """Print human-readable summary to console."""
    num_entities = len(result.get('entity_masks', []))
    total_time = result.get('metadata', {}).get('total_time', 0)
    
    if num_entities == 0:
        print("❌ No entities detected")
        color_coverage = result.get('metadata', {}).get('color_coverage', 0)
        print(f"   Color mask coverage: {color_coverage:.1f}%")
        if color_coverage < 5:
            print("   Suggestion: Image may not contain target color")
    else:
        print(f"✅ Success! Detected {num_entities} entities")
        print(f"   Total time: {total_time:.1f} seconds")
        
        if 'validation' in result and result['validation'] is not None:
            val = result['validation']
            confidence = getattr(val, 'confidence', 0) if hasattr(val, 'confidence') else val.get('confidence', 0)
            print(f"   VLM confidence: {confidence:.2f}")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="EDI Vision Pipeline - Detect and segment entities for image editing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python app.py --image test_image.jpeg --prompt "turn blue roofs to green" --output result.png
  python app.py --image img.jpg --prompt "change red cars" --output out.png --verbose
  python app.py --image img.jpg --prompt "edit blue" --output out.png --save-steps --no-validation
        """
    )
    
    # Required arguments
    parser.add_argument("--image", "-i", required=True,
                       help="Path to input image (JPG, PNG, JPEG)")
    parser.add_argument("--prompt", "-p", required=True,
                       help="User's edit request (natural language)")
    parser.add_argument("--output", "-o", required=True,
                       help="Path for output visualization")
    
    # Optional arguments
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable detailed logging (INFO level)")
    parser.add_argument("--debug", "-d", action="store_true",
                       help="Enable debug logging (DEBUG level)")
    parser.add_argument("--save-steps", action="store_true",
                       help="Save intermediate outputs to logs/")
    parser.add_argument("--no-validation", action="store_true",
                       help="Skip VLM validation (Stage 6)")
    parser.add_argument("--config", "-c",
                       help="Path to YAML config file")
    parser.add_argument("--min-area", type=int, default=500,
                       help="Minimum mask area in pixels (default: 500)")
    parser.add_argument("--color-threshold", type=float, default=0.5,
                       help="HSV color match threshold (0.0-1.0, default: 0.5)")
    parser.add_argument("--clip-threshold", type=float, default=0.22,
                       help="CLIP similarity threshold (0.0-1.0, default: 0.22)")
    
    return parser.parse_args()


def main():
    """Main function for the CLI application."""
    args = parse_arguments()
    setup_logging(args.verbose, args.debug)
    
    # Validate inputs
    validate_image_path(args.image)
    validate_prompt(args.prompt)
    validate_thresholds(args)
    
    # Load config
    config = load_config(args.config) if args.config else get_default_config()
    
    # Override config with CLI arguments
    config['pipeline']['enable_validation'] = not args.no_validation
    config['pipeline']['save_intermediate'] = args.save_steps
    config['sam']['min_area'] = args.min_area
    config['clip']['similarity_threshold'] = args.clip_threshold
    
    # Initialize pipeline
    pipeline = VisionPipeline(
        enable_validation=not args.no_validation,
        save_intermediate=args.save_steps,
        output_dir=config['pipeline']['output_dir']
    )
    
    # Process image
    try:
        print("[Stage 1/6] Extracting entities from prompt...")
        # In a real implementation, we would have more detailed progress,
        # but for now we just note the start
        result = pipeline.process(
            image_path=args.image,
            user_prompt=args.prompt
        )
        
        if not result['success']:
            print(f"❌ Error: {result.get('error', 'Pipeline failed')}")
            sys.exit(1)
            
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        print(f"❌ Error: {str(e)}")
        sys.exit(1)
    
    # Create visualization
    try:
        visualization = create_output_visualization(args.image, result)
    except Exception as e:
        logging.error(f"Failed to create visualization: {str(e)}")
        print(f"❌ Error creating visualization: {str(e)}")
        sys.exit(1)
    
    # Save output
    try:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert RGB to BGR for OpenCV
        visualization_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), visualization_bgr)
        
        print(f"   Output saved: {args.output}")
    except Exception as e:
        logging.error(f"Failed to save output: {str(e)}")
        print(f"❌ Error saving output: {str(e)}")
        sys.exit(1)
    
    # Print summary
    print_summary(result)


if __name__ == "__main__":
    main()