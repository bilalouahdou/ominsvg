"""
VectraHub GPU Server - Image Type Detection

Advanced image type detection for VTracer parameter selection.
"""
import cv2
import numpy as np
from PIL import Image
from typing import Literal

ImageType = Literal["logo", "illustration", "line_art"]


def detect_image_type(image_path: str) -> ImageType:
    """
    Detect image type based on visual properties.
    
    Used as a secondary check or when CLIP is unavailable.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Detected image type
    """
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return "illustration"
        
        # Convert to various color spaces
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Calculate metrics
        
        # 1. Color variance (how many distinct colors)
        color_variance = np.var(img, axis=(0, 1)).mean()
        
        # 2. Edge density (how many edges)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # 3. Saturation variance (colorfulness)
        saturation = hsv[:, :, 1]
        sat_variance = np.var(saturation)
        
        # 4. Number of distinct colors
        pixels = img.reshape(-1, 3)
        unique_colors = len(np.unique(pixels, axis=0))
        color_ratio = unique_colors / len(pixels)
        
        # Heuristics
        
        # Line art: Low color variance, high edges, low saturation variance
        if color_variance < 2000 and edge_density > 0.05 and sat_variance < 1000:
            return "line_art"
        
        # Logo: Moderate colors, high edge density, transparency often present
        if 1000 < color_variance < 8000 and edge_density > 0.03:
            # Check for transparency
            pil_img = Image.open(image_path)
            if pil_img.mode == "RGBA":
                alpha = np.array(pil_img)[:, :, 3]
                transparency_ratio = np.sum(alpha < 255) / alpha.size
                if transparency_ratio > 0.1:  # More than 10% transparent
                    return "logo"
        
        # Illustration: High color variance, many unique colors
        if color_variance > 5000 or unique_colors > 1000:
            return "illustration"
        
        # Default
        return "illustration"
        
    except Exception as e:
        print(f"Image type detection error: {e}")
        return "illustration"
