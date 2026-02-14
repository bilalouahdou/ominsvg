"""
VectraHub GPU Server - VTracer Core

VTracer-based vectorization with presets for different image types.
"""
import os
import subprocess
from typing import Dict, Any, Optional

from app.config import config
from app.utils.vtracer.image_type import detect_image_type
from app.utils.vtracer.background import remove_background_path
from app.utils.vtracer.color_verify import verify_colors
from app.utils.vtracer.gradients import detect_gradients


def get_vtracer_preset(image_type: str) -> Dict[str, Any]:
    """
    Get VTracer parameters preset based on image type.
    
    Args:
        image_type: One of "logo", "illustration", "line_art"
        
    Returns:
        Dictionary of VTracer parameters
    """
    return config.vtracer_presets.get(image_type, config.vtracer_presets["illustration"])


def vectorize_image(
    input_path: str,
    output_path: str,
    image_type: str = "illustration",
    custom_params: Optional[Dict[str, Any]] = None
) -> str:
    """
    Vectorize an image using VTracer.
    
    Args:
        input_path: Path to input raster image
        output_path: Path for output SVG
        image_type: Type of image for preset selection
        custom_params: Override preset parameters
        
    Returns:
        Path to generated SVG file
    """
    try:
        # Get preset parameters
        params = get_vtracer_preset(image_type)
        
        # Apply custom overrides
        if custom_params:
            params.update(custom_params)
        
        # Build VTracer command
        cmd = [
            "vtracer",
            "--input", input_path,
            "--output", output_path,
            "--color_mode", "color",
            "--color_precision", str(params["color_precision"]),
            "--layer_difference", str(params["layer_difference"]),
            "--mode", "spline",
            "--filter_speckle", "4",
            "--corner_threshold", "60",
            "--length_threshold", "4.0",
            "--max_iterations", str(params["max_iterations"]),
            "--splice_threshold", "45",
        ]
        
        print(f"Running VTracer: {' '.join(cmd)}")
        
        # Execute VTracer
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"VTracer failed: {result.stderr}")
        
        print(f"VTracer completed: {output_path}")
        
        # Post-processing
        # 1. Remove background paths if detected
        remove_background_path(output_path)
        
        # 2. Verify and fix colors
        verify_colors(output_path, input_path)
        
        return output_path
        
    except subprocess.CalledProcessError as e:
        print(f"VTracer error: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        raise RuntimeError(f"VTracer execution failed: {e}")
        
    except Exception as e:
        print(f"Vectorization error: {e}")
        raise
