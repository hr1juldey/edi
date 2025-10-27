import cv2
import numpy as np
import math
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.color import rgb2ycbcr, ycbcr2rgb
import matplotlib.pyplot as plt
import os

def calculate_psnr(original_image, edited_image):
    """
    Calculate PSNR (Peak Signal-to-Noise Ratio) between original and edited images.
    
    Args:
        original_image: Original image as numpy array
        edited_image: Edited image as numpy array
    
    Returns:
        float: PSNR value in dB
    """
    # Ensure images are in the same format
    if original_image.shape != edited_image.shape:
        raise ValueError("Original and edited images must have the same dimensions")
    
    # Convert to float32 for calculation
    original_float = original_image.astype(np.float32)
    edited_float = edited_image.astype(np.float32)
    
    # Calculate MSE (Mean Squared Error)
    mse = np.mean((original_float - edited_float) ** 2)
    
    # Calculate PSNR
    if mse == 0:
        return float('inf')
    
    max_pixel_value = 255.0  # For 8-bit images
    psnr = 20 * math.log10(max_pixel_value / math.sqrt(mse))
    
    return psnr

def calculate_delta_e_ycbcr(original_image, edited_image):
    """
    Calculate Delta E in YCbCr color space between original and edited images.
    
    Args:
        original_image: Original image as numpy array
        edited_image: Edited image as numpy array
    
    Returns:
        float: Average Delta E value
    """
    # Ensure images are in the same format
    if original_image.shape != edited_image.shape:
        raise ValueError("Original and edited images must have the same dimensions")
    
    # Convert to YCbCr color space
    original_ycbcr = rgb2ycbcr(original_image)
    edited_ycbcr = rgb2ycbcr(edited_image)
    
    # Calculate Delta E for each pixel
    delta_y = original_ycbcr[:, :, 0] - edited_ycbcr[:, :, 0]
    delta_cb = original_ycbcr[:, :, 1] - edited_ycbcr[:, :, 1]
    delta_cr = original_ycbcr[:, :, 2] - edited_ycbcr[:, :, 2]
    
    # Calculate Delta E using Euclidean distance in YCbCr space
    delta_e = np.sqrt(delta_y**2 + delta_cb**2 + delta_cr**2)
    
    # Return average Delta E
    return np.mean(delta_e)

def create_heatmap(original_image, edited_image, output_path=None):
    """
    Create a heatmap visualization showing differences between original and edited images.
    
    Args:
        original_image: Original image as numpy array
        edited_image: Edited image as numpy array
        output_path: Path to save the heatmap (optional)
    
    Returns:
        numpy.ndarray: Heatmap image
    """
    # Ensure images are in the same format
    if original_image.shape != edited_image.shape:
        raise ValueError("Original and edited images must have the same dimensions")
    
    # Calculate difference between images
    diff = np.abs(original_image.astype(np.float32) - edited_image.astype(np.float32))
    
    # Normalize difference to [0, 255]
    diff_normalized = (diff / np.max(diff)) * 255
    
    # Create heatmap (using red for differences)
    heatmap = np.zeros_like(original_image, dtype=np.uint8)
    heatmap[:, :, 0] = diff_normalized[:, :, 0]  # Red channel
    
    # Optionally save the heatmap
    if output_path:
        cv2.imwrite(output_path, cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
    
    return heatmap

def validate_edit(original_image, edited_image, mask=None):
    """
    Validate edit quality using PSNR, delta E, and create heatmap visualization.
    
    Args:
        original_image: Original image as numpy array
        edited_image: Edited image as numpy array
        mask: Optional mask to focus validation on specific regions
    
    Returns:
        dict: Validation metrics including PSNR, delta E, and heatmap path
    """
    # Initialize validation results
    validation_results = {
        'psnr': None,
        'delta_e': None,
        'heatmap_path': None,
        'mask_coverage': None,
        'success': False,
        'error': None
    }
    
    try:
        # Calculate PSNR
        validation_results['psnr'] = calculate_psnr(original_image, edited_image)
        
        # Calculate Delta E in YCbCr color space
        validation_results['delta_e'] = calculate_delta_e_ycbcr(original_image, edited_image)
        
        # Create heatmap
        heatmap_output_path = "result_adaptive.png"  # Default output path
        heatmap = create_heatmap(original_image, edited_image, heatmap_output_path)
        validation_results['heatmap_path'] = heatmap_output_path
        
        # If mask is provided, calculate mask coverage
        if mask is not None:
            # Calculate percentage of mask coverage
            mask_area = np.sum(mask > 0)
            total_area = mask.size
            validation_results['mask_coverage'] = mask_area / total_area
        
        validation_results['success'] = True
        
    except Exception as e:
        validation_results['error'] = str(e)
    
    return validation_results

def plot_validation_metrics(psnr, delta_e, heatmap_path=None):
    """
    Plot validation metrics for visual inspection.
    
    Args:
        psnr: PSNR value
        delta_e: Delta E value
        heatmap_path: Path to heatmap image (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot PSNR and Delta E
    axes[0].bar(['PSNR', 'Delta E'], [psnr, delta_e], color=['blue', 'red'])
    axes[0].set_title('Validation Metrics')
    axes[0].set_ylabel('Value')
    
    # Add text annotations
    axes[0].text(0, psnr, f'{psnr:.2f} dB', ha='center', va='bottom')
    axes[0].text(1, delta_e, f'{delta_e:.2f}', ha='center', va='bottom')
    
    # Display heatmap if available
    if heatmap_path and os.path.exists(heatmap_path):
        heatmap = cv2.imread(heatmap_path)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        axes[1].imshow(heatmap)
        axes[1].set_title('Heatmap Visualization')
        axes[1].axis('off')
    else:
        axes[1].text(0.5, 0.5, 'No heatmap available', ha='center', va='center', fontsize=12)
        axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()