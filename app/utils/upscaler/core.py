"""
VectraHub GPU Server - Upscaler Core

Smart upscaling strategy using Real-ESRGAN to reach target resolution.
"""
import os
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional, List
import torch

from app.config import config
from app.utils.upscaler.upsampler import load_realesrgan_upsampler
from app.utils.upscaler.palette import extract_palette, correct_palette


def calculate_upscale_strategy(width: int, height: int, target_mp: float = 12.0) -> Tuple[int, int]:
    """
    Calculate the upscale strategy based on input dimensions.
    
    Strategy:
    - < 0.5 MP → x2, x2 (two passes = x4)
    - 0.5-3 MP → x2, shrink to 3MP, x2 (final 12MP)
    - > 3 MP → shrink to 3MP, x2 (final 12MP)
    
    Args:
        width: Input image width
        height: Input image height
        target_mp: Target megapixels (default 12.0)
        
    Returns:
        Tuple of (scale_factor, num_passes)
    """
    current_mp = (width * height) / 1_000_000
    
    if current_mp < 0.5:
        # Very small image - upscale twice
        return 2, 2
    elif current_mp < 3.0:
        # Medium image - upscale once
        return 2, 1
    else:
        # Large image - just shrink and upscale
        return 2, 1


def resize_to_target(image: np.ndarray, target_mp: float) -> np.ndarray:
    """
    Resize image to target megapixels while maintaining aspect ratio.
    
    Args:
        image: Input image (numpy array)
        target_mp: Target megapixels
        
    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    current_mp = (w * h) / 1_000_000
    
    if abs(current_mp - target_mp) < 0.5:
        return image
    
    # Calculate new dimensions
    ratio = (target_mp / current_mp) ** 0.5
    new_w = int(w * ratio)
    new_h = int(h * ratio)
    
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)


def apply_median_blur(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    """
    Apply GPU median blur for edge cleanup.
    
    Args:
        image: Input image
        ksize: Kernel size (must be odd)
        
    Returns:
        Blurred image
    """
    if ksize % 2 == 0:
        ksize += 1
    
    return cv2.medianBlur(image, ksize)


def upscale_image(
    input_path: str,
    output_path: str,
    target_mp: float = 12.0,
    apply_palette_correction: bool = True
) -> Tuple[bool, Optional[str], Optional[List[str]], bool]:
    """
    Upscale image using Real-ESRGAN to target resolution.
    
    Args:
        input_path: Path to input image
        output_path: Path for upscaled output
        target_mp: Target megapixels (default 12MP)
        apply_palette_correction: Whether to correct colors to original palette
        
    Returns:
        Tuple of (success, background_color, palette, has_gradients)
    """
    try:
        # Load image
        img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Failed to load image: {input_path}")
            return False, None, None, False
        
        original_img = img.copy()
        height, width = img.shape[:2]
        print(f"Input image: {width}x{height} ({(width*height)/1_000_000:.2f} MP)")
        
        # Extract palette before upscaling
        if apply_palette_correction:
            palette, bg_color, has_gradients = extract_palette(input_path)
        else:
            palette, bg_color, has_gradients = None, None, False
        
        # Determine upscale strategy
        scale_factor, num_passes = calculate_upscale_strategy(width, height, target_mp)
        print(f"Upscale strategy: {scale_factor}x, {num_passes} pass(es)")
        
        # Load upsampler
        upsampler = load_realesrgan_upsampler(
            scale=scale_factor,
            half=config.upscaler_half_precision
        )
        
        if upsampler is None:
            print("Failed to load Real-ESRGAN upsampler")
            return False, None, None, False
        
        # Perform upscaling
        for pass_num in range(num_passes):
            print(f"Upscaling pass {pass_num + 1}/{num_passes}...")
            
            # Convert BGR to RGB if needed
            if len(img.shape) == 3 and img.shape[2] == 3:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = img
            
            # Upscale
            try:
                img_upscaled, _ = upsampler.enhance(img_rgb, outscale=scale_factor)
            except RuntimeError as error:
                print(f"Error during upscaling: {error}")
                # Try with tiling for large images
                print("Retrying with tiling...")
                upsampler = load_realesrgan_upsampler(
                    scale=scale_factor,
                    half=config.upscaler_half_precision,
                    tile=400
                )
                img_upscaled, _ = upsampler.enhance(img_rgb, outscale=scale_factor)
            
            img = cv2.cvtColor(img_upscaled, cv2.COLOR_RGB2BGR)
            
            # If intermediate step, resize to target
            if pass_num < num_passes - 1:
                img = resize_to_target(img, 3.0)  # Shrink to 3MP between passes
        
        # Final resize to exact target
        img = resize_to_target(img, target_mp)
        
        # Apply median blur for edge cleanup
        img = apply_median_blur(img, ksize=5)
        
        # Apply palette correction
        if apply_palette_correction and palette:
            img = correct_palette(img, palette)
        
        # Save output
        cv2.imwrite(output_path, img)
        print(f"Upscaled image saved: {output_path}")
        
        return True, bg_color, palette, has_gradients
        
    except Exception as e:
        print(f"Upscaling error: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None, False
