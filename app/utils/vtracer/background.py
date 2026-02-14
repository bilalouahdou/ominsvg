"""
VectraHub GPU Server - Background Removal

Remove background paths from SVG files.
"""
import re
from xml.etree import ElementTree as ET


def remove_background_path(svg_path: str):
    """
    Remove background/fill paths from SVG.
    
    Background paths are typically:
    - Large rectangles covering most of the viewBox
    - White/transparent fills that serve as canvas
    
    Args:
        svg_path: Path to SVG file to modify in place
    """
    try:
        # Register namespaces
        ET.register_namespace("", "http://www.w3.org/2000/svg")
        ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
        
        # Parse SVG
        tree = ET.parse(svg_path)
        root = tree.getroot()
        
        # Get viewBox for size reference
        viewBox = root.get("viewBox", "")
        if viewBox:
            parts = viewBox.split()
            if len(parts) == 4:
                vb_width = float(parts[2])
                vb_height = float(parts[3])
            else:
                vb_width = float(root.get("width", "1000").replace("px", ""))
                vb_height = float(root.get("height", "1000").replace("px", ""))
        else:
            vb_width = float(root.get("width", "1000").replace("px", ""))
            vb_height = float(root.get("height", "1000").replace("px", ""))
        
        def is_background_element(elem):
            """Check if element is likely a background."""
            tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
            
            if tag not in ("rect", "path", "polygon"):
                return False
            
            # Check if it covers most of the viewBox
            width = elem.get("width", "0").replace("px", "").replace("%", "")
            height = elem.get("height", "0").replace("px", "").replace("%", "")
            
            try:
                if width and height:
                    w = float(width)
                    h = float(height)
                    # If it covers >80% of viewBox, likely background
                    if w > vb_width * 0.8 and h > vb_height * 0.8:
                        return True
            except:
                pass
            
            # Check for white/transparent fill
            fill = elem.get("fill", "").lower()
            if fill in ("white", "#ffffff", "#fff", "none", "transparent"):
                # If it has no stroke, likely background
                if not elem.get("stroke"):
                    return True
            
            return False
        
        def remove_backgrounds(elem):
            """Recursively remove background elements."""
            to_remove = []
            for child in elem:
                if is_background_element(child):
                    to_remove.append(child)
                else:
                    remove_backgrounds(child)
            
            for child in to_remove:
                elem.remove(child)
        
        # Process root
        remove_backgrounds(root)
        
        # Save modified SVG
        tree.write(svg_path, encoding="utf-8", xml_declaration=True)
        
    except Exception as e:
        print(f"Background removal error: {e}")
