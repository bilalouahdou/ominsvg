"""
VectraHub GPU Server - SVG Color Fixer

Post-processing utilities to fix SVG color issues and artifacts.
"""
import re
from xml.etree import ElementTree as ET
from typing import List, Tuple, Optional


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = "".join([c*2 for c in hex_color])
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    """Convert RGB tuple to hex color."""
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def color_distance(c1: Tuple[int, int, int], c2: Tuple[int, int, int]) -> float:
    """Calculate Euclidean distance between two RGB colors."""
    return sum((a - b) ** 2 for a, b in zip(c1, c2)) ** 0.5


def snap_color_to_palette(color: str, palette: List[str], threshold: int = 30) -> str:
    """
    Snap a color to the nearest color in a palette if within threshold.
    
    Args:
        color: Hex color string
        palette: List of hex colors in the original palette
        threshold: Maximum distance to snap (default 30)
        
    Returns:
        Snapped color or original color
    """
    try:
        rgb = hex_to_rgb(color)
        palette_rgbs = [hex_to_rgb(c) for c in palette]
        
        # Find nearest palette color
        distances = [(color_distance(rgb, p), p) for p in palette_rgbs]
        min_dist, nearest = min(distances, key=lambda x: x[0])
        
        if min_dist <= threshold:
            return rgb_to_hex(nearest)
        
        return color
    except:
        return color


def fix_svg_colors(svg_path: str, palette: Optional[List[str]] = None):
    """
    Fix SVG color issues including:
    - Snap colors to original palette
    - Remove black/white artifacts
    - Fix opacity issues
    
    Args:
        svg_path: Path to SVG file
        palette: Optional list of hex colors from original image
    """
    try:
        # Register SVG namespace
        ET.register_namespace("", "http://www.w3.org/2000/svg")
        ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
        
        # Parse SVG
        tree = ET.parse(svg_path)
        root = tree.getroot()
        
        # Color attributes to fix
        color_attrs = ["fill", "stroke", "stop-color"]
        
        def process_element(elem):
            """Process an element's color attributes."""
            for attr in color_attrs:
                if attr in elem.attrib:
                    color = elem.attrib[attr]
                    
                    # Skip "none", "transparent", url references
                    if color in ("none", "transparent") or color.startswith("url"):
                        continue
                    
                    # Convert to hex if rgb()
                    if color.startswith("rgb("):
                        match = re.match(r"rgb\((\d+),\s*(\d+),\s*(\d+)\)", color)
                        if match:
                            r, g, b = map(int, match.groups())
                            color = rgb_to_hex((r, g, b))
                    
                    # Snap to palette if provided
                    if palette and color.startswith("#"):
                        color = snap_color_to_palette(color, palette)
                    
                    elem.attrib[attr] = color
            
            # Process children
            for child in elem:
                process_element(child)
        
        # Process all elements
        process_element(root)
        
        # Save fixed SVG
        tree.write(svg_path, encoding="utf-8", xml_declaration=True)
        
    except Exception as e:
        print(f"SVG color fix error: {e}")


def remove_black_artifacts(svg_path: str, threshold: int = 30):
    """
    Remove paths that are nearly pure black (likely artifacts).
    
    Args:
        svg_path: Path to SVG file
        threshold: RGB values below this are considered "black"
    """
    try:
        ET.register_namespace("", "http://www.w3.org/2000/svg")
        tree = ET.parse(svg_path)
        root = tree.getroot()
        
        def is_near_black(elem):
            """Check if element has near-black fill/stroke."""
            for attr in ["fill", "stroke"]:
                color = elem.get(attr, "")
                if color.startswith("#"):
                    try:
                        rgb = hex_to_rgb(color)
                        if all(c < threshold for c in rgb):
                            return True
                    except:
                        pass
            return False
        
        def remove_black_elements(elem):
            """Recursively remove black elements."""
            to_remove = []
            for child in elem:
                if child.tag.endswith("path") or child.tag.endswith("rect") or child.tag.endswith("circle"):
                    if is_near_black(child):
                        to_remove.append(child)
                else:
                    remove_black_elements(child)
            
            for child in to_remove:
                elem.remove(child)
        
        remove_black_elements(root)
        tree.write(svg_path, encoding="utf-8", xml_declaration=True)
        
    except Exception as e:
        print(f"Remove artifacts error: {e}")
