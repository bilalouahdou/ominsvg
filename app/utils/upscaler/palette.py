"""
VectraHub GPU Server - LAB Palette Correction

Color palette extraction and correction for upscaled images.
"""
import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional
from sklearn.cluster import KMeans
from collections import Counter


def rgb_to_lab(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
    """Convert RGB to LAB color space."""
    # Create a small image with the color
    color = np.uint8([[rgb]])
    lab = cv2.cvtColor(color, cv2.COLOR_RGB2LAB)
    return tuple(lab[0, 0].astype(float))


def lab_to_rgb(lab: Tuple[float, float, float]) -> Tuple[int, int, int]:
    """Convert LAB to RGB color space."""
    lab_array = np.uint8([[lab]])
    rgb = cv2.cvtColor(lab_array, cv2.COLOR_LAB2RGB)
    return tuple(rgb[0, 0].astype(int))


def color_distance_lab(c1: Tuple[float, float, float], c2: Tuple[float, float, float]) -> float:
    """Calculate distance between two LAB colors (simplified Delta E)."""
    return sum((a - b) ** 2 for a, b in zip(c1, c2)) ** 0.5


def extract_palette(image_path: str, num_colors: int = 16) -> Tuple[List[str], Optional[str], bool]:
    """
    Extract dominant color palette from an image.
    
    Args:
        image_path: Path to image
        num_colors: Number of colors to extract
        
    Returns:
        Tuple of (palette_hex_list, background_color, has_gradients)
    """
    try:
        # Load image
        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img)
        
        # Reshape for clustering
        pixels = img_array.reshape(-1, 3)
        
        # Use K-means to find dominant colors
        kmeans = KMeans(n_clusters=min(num_colors, len(pixels)), random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Get colors
        colors = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_
        
        # Count occurrences
        counts = Counter(labels)
        
        # Sort by frequency
        sorted_colors = sorted(
            [(counts[i], colors[i]) for i in range(len(colors))],
            reverse=True
        )
        
        # Convert to hex
        palette_hex = []
        for _, color in sorted_colors:
            hex_color = "#{:02x}{:02x}{:02x}".format(*color)
            palette_hex.append(hex_color)
        
        # Detect background (most common color, usually in corners)
        bg_color = palette_hex[0] if palette_hex else None
        
        # Detect gradients (high color variance)
        color_variance = np.var(pixels, axis=0).mean()
        has_gradients = color_variance > 1000
        
        return palette_hex, bg_color, has_gradients
        
    except Exception as e:
        print(f"Palette extraction error: {e}")
        return [], None, False


def correct_palette(image: np.ndarray, palette: List[str]) -> np.ndarray:
    """
    Correct image colors to match original palette using LAB color space.
    
    Args:
        image: Input image (BGR format from OpenCV)
        palette: List of hex colors from original
        
    Returns:
        Color-corrected image
    """
    try:
        if not palette:
            return image
        
        # Convert to RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert palette to LAB
        palette_rgb = []
        palette_lab = []
        for hex_color in palette:
            # Parse hex
            hex_color = hex_color.lstrip("#")
            if len(hex_color) == 3:
                hex_color = "".join([c*2 for c in hex_color])
            rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            palette_rgb.append(rgb)
            palette_lab.append(rgb_to_lab(rgb))
        
        # Convert image to LAB
        img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        
        # Create lookup table for faster processing
        # For each pixel, find nearest palette color in LAB space
        h, w = img_lab.shape[:2]
        img_lab_flat = img_lab.reshape(-1, 3)
        
        # Vectorized distance calculation
        corrected = np.zeros_like(img_rgb)
        corrected_flat = corrected.reshape(-1, 3)
        
        for i, pixel_lab in enumerate(img_lab_flat):
            # Find nearest palette color
            distances = [color_distance_lab(tuple(pixel_lab), p_lab) for p_lab in palette_lab]
            nearest_idx = np.argmin(distances)
            corrected_flat[i] = palette_rgb[nearest_idx]
        
        corrected = corrected_flat.reshape(h, w, 3)
        
        # Blend with original for smoothness (optional)
        alpha = 0.7  # How much to snap to palette
        result = cv2.addWeighted(img_rgb, 1 - alpha, corrected, alpha, 0)
        
        # Convert back to BGR
        return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        
    except Exception as e:
        print(f"Palette correction error: {e}")
        return image
