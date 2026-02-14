"""
VectraHub GPU Server - Task Definitions

Celery/RQ task definitions for async processing (optional).
For RunPod serverless, tasks are handled synchronously in handler.py.
"""
from typing import Dict, Any
import os
import uuid

from app.config import config
from app.utils.image_classifier import classify_image
from app.utils.upscaler import upscale_image
from app.utils.vtracer import vectorize_image as vtracer_vectorize
from app.utils.omnisvg import vectorize_logo_with_omnisvg


def process_image_task(
    image_path: str,
    upscale: int = 4,
    color_mode: str = "color",
    vtracer_params: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Process an image through the full pipeline.
    
    This is the main processing task that can be called:
    - Synchronously from handler.py (RunPod serverless)
    - Asynchronously via Celery/RQ (optional scaling)
    
    Args:
        image_path: Path to input image
        upscale: Upscale factor
        color_mode: Color mode preference
        vtracer_params: Custom VTracer parameters
    
    Returns:
        Dict with processing results
    """
    result = {
        "success": False,
        "svg_path": None,
        "image_type": None,
        "error": None
    }
    
    try:
        # Step 1: Classify image
        result["image_type"] = classify_image(image_path)
        
        # Step 2: Upscale
        upscaled_path = os.path.join(config.temp_dir, f"{uuid.uuid4()}_upscaled.png")
        success, bg_color, palette, has_gradients = upscale_image(image_path, upscaled_path)
        
        if not success:
            result["error"] = "Upscaling failed"
            return result
        
        result["background_color"] = bg_color
        result["has_gradients"] = has_gradients
        
        # Step 3: Vectorize based on type
        output_path = os.path.join(config.outputs_dir, f"{uuid.uuid4()}.svg")
        
        if result["image_type"] == "logo" and config.use_omnisvg_for_logos:
            svg_path = _vectorize_with_omnisvg_or_fallback(
                upscaled_path, output_path, vtracer_params
            )
        else:
            svg_path = vtracer_vectorize(
                upscaled_path, 
                output_path,
                result["image_type"],
                vtracer_params
            )
        
        result["svg_path"] = svg_path
        result["success"] = True
        
        # Cleanup
        if os.path.exists(upscaled_path):
            os.remove(upscaled_path)
            
    except Exception as e:
        result["error"] = str(e)
    
    return result


def _vectorize_with_omnisvg_or_fallback(
    input_path: str,
    output_path: str,
    vtracer_params: Dict[str, Any] = None
) -> str:
    """
    Vectorize using OmniSVG with VTracer fallback.
    
    Args:
        input_path: Path to upscaled image
        output_path: Desired output SVG path
        vtracer_params: Fallback VTracer parameters
    
    Returns:
        Path to generated SVG
    """
    try:
        # Try OmniSVG
        svg_path = vectorize_logo_with_omnisvg(input_path, output_path)
        
        # Quality check
        if config.enable_quality_check:
            from app.utils.omnisvg.core import quality_check
            if quality_check(svg_path):
                return svg_path
        else:
            return svg_path
            
    except Exception as e:
        print(f"OmniSVG failed: {e}")
    
    # Fallback to VTracer
    if config.enable_vtracer_fallback:
        print("Falling back to VTracer")
        return vtracer_vectorize(input_path, output_path, "logo", vtracer_params)
    
    raise RuntimeError("OmniSVG failed and fallback is disabled")
