"""
VectraHub GPU Server - Color Verification

Verify SVG colors match original image palette.
"""
import cv2
import numpy as np
from xml.etree import ElementTree as ET
from typing import List, Tuple, Optional


def extract_dominant_colors(image_path: str, n_colors: int = 16) -> List[Tuple[int, int, int]]:
    """
    Extract dominant colors from an image.
    
    Args:
        image_path: Path to image
        n_colors: Number of colors to extract
        
    Returns:
        List of RGB tuples
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return []
        
        # Convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Reshape for k-means
        pixels = img.reshape(-1, 3).astype(np.float32)
        
        # K-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(
            pixels, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )
        
        # Convert to integers
        colors = [tuple(map(int, color)) for color in centers]
        
        return colors
        
    except Exception as e:
        print(f"Color extraction error: {e}")
        return []


def hex_to_rgb(hex_color: str) -> Optional[Tuple[int, int, int]]:
    """Convert hex color to RGB tuple."""
    try:
        hex_color = hex_color.lstrip("#")
        if len(hex_color) == 3:
            hex_color = "".join([c*2 for c in hex_color])
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    except:
        return None


def color_distance(c1: Tuple[int, int, int], c2: Tuple[int, int, int]) -> float:
    """Calculate Euclidean distance between two colors."""
    return sum((a - b) ** 2 for a, b in zip(c1, c2)) ** 0.5


def find_nearest_color(color: Tuple[int, int, int], palette: List[Tuple[int, int, int]]) -> Tuple[int, int, int]:
    """Find the nearest color in palette."""
    if not palette:
        return color
    
    distances = [(color_distance(color, p), p) for p in palette]
    return min(distances, key=lambda x: x[0])[1]


def verify_colors(svg_path: str, original_image_path: str, threshold: int = 50):
    """
    Verify and fix SVG colors to match original image palette.
    
    Args:
        svg_path: Path to SVG file
        original_image_path: Path to original raster image
        threshold: Maximum color distance for snapping
    """
    try:
        # Extract colors from original
        original_colors = extract_dominant_colors(original_image_path)
        
        if not original_colors:
            return
        
        # Parse SVG
        ET.register_namespace("", "http://www.w3.org/2000/svg")
        tree = ET.parse(svg_path)
        root = tree.getroot()
        
        def process_element(elem):
            """Process element color attributes."""
            color_attrs = ["fill", "stroke", "stop-color"]
            
            for attr in color_attrs:
                if attr in elem.attrib:
                    value = elem.attrib[attr]
                    
                    # Skip non-color values
                    if value in ("none", "transparent") or value.startswith("url"):
                        continue
                    
                    # Convert to RGB
                    rgb = hex_to_rgb(value)
                    if rgb:
                        # Find nearest original color
                        nearest = find_nearest_color(rgb, original_colors)
                        
                        # Check if within threshold
                        if color_distance(rgb, nearest) <= threshold:
                            # Convert back to hex
                            new_hex = "#{:02x}{:02x}{:02x}".format(*nearest)
                            elem.attrib[attr] = new_hex
            
            # Process children
            for child in elem:
                process_element(child)
        
        # Process all elements
        process_element(root)
        
        # Save
        tree.write(svg_path, encoding="utf-8", xml_declaration=True)
        
    except Exception as e:
        print(f"Color verification error: {e}")
