"""
VectraHub GPU Server - RunPod Serverless Handler

Main entry point for RunPod serverless GPU workers.
Handles image download, classification, upscaling, and vectorization.
"""
import os
import sys
import json
import base64
import tempfile
import uuid
import requests
from typing import Dict, Any, Optional

import runpod

# Add app to path
sys.path.insert(0, "/app")

from app.config import config
from app.utils.image_classifier import classify_image
from app.utils.upscaler import upscale_image
from app.utils.vtracer import vectorize_image as vtracer_vectorize
from app.utils.cleanup import cleanup_temp_files
from app.utils.svg_color_fixer import fix_svg_colors

# Try to import OmniSVG, fallback gracefully if not available
try:
    from app.utils.omnisvg import vectorize_logo_with_omnisvg
    OMNISVG_AVAILABLE = True
except ImportError as e:
    print(f"OmniSVG not available: {e}")
    OMNISVG_AVAILABLE = False

# Ensure directories exist
config.ensure_directories()


def download_image(image_url: str, output_path: str) -> bool:
    """Download image from URL to local path."""
    try:
        headers = {
            "User-Agent": "VectraHub-GPU-Server/1.0"
        }
        response = requests.get(image_url, headers=headers, timeout=60)
        response.raise_for_status()
        
        with open(output_path, "wb") as f:
            f.write(response.content)
        
        return True
    except Exception as e:
        print(f"Failed to download image: {e}")
        return False


def decode_base64_image(base64_data: str, output_path: str) -> bool:
    """Decode base64 image data to file."""
    try:
        # Remove data URI prefix if present
        if "," in base64_data:
            base64_data = base64_data.split(",")[1]
        
        image_data = base64.b64decode(base64_data)
        with open(output_path, "wb") as f:
            f.write(image_data)
        
        return True
    except Exception as e:
        print(f"Failed to decode base64 image: {e}")
        return False


def upload_svg(svg_path: str, original_filename: str) -> Optional[str]:
    """
    Upload SVG to pixel2vector.com server.
    
    Returns:
        Public URL of uploaded SVG, or None if upload failed
    """
    try:
        upload_url = f"{config.app_base_url}/php/api/svg_upload.php"
        
        with open(svg_path, "rb") as f:
            files = {"svg_file": (os.path.basename(svg_path), f, "image/svg+xml")}
            data = {
                "secret": config.svg_upload_secret,
                "original_filename": original_filename
            }
            
            response = requests.post(upload_url, files=files, data=data, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            if result.get("success"):
                return result.get("svg_url")
            else:
                print(f"Upload failed: {result.get('error', 'Unknown error')}")
                return None
                
    except Exception as e:
        print(f"Failed to upload SVG: {e}")
        return None


def quality_check(svg_path: str) -> bool:
    """
    Check if SVG output is usable.
    
    Fail conditions:
    - File too small (likely empty)
    - No paths generated
    - Mostly black/white (optional check)
    """
    try:
        # Check file size
        file_size = os.path.getsize(svg_path)
        if file_size < 500:
            print(f"Quality check failed: File too small ({file_size} bytes)")
            return False
        
        # Check for paths
        with open(svg_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        if "<path" not in content and "<circle" not in content and "<rect" not in content:
            print("Quality check failed: No paths/shapes found")
            return False
        
        return True
        
    except Exception as e:
        print(f"Quality check error: {e}")
        return False


def process_job(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main job processing logic.
    
    Pipeline:
    1. Download/decode input image
    2. Classify image type (logo/illustration/line_art)
    3. Upscale with Real-ESRGAN to 12MP
    4. Vectorize with appropriate method
    5. Upload SVG and return URL
    """
    result = {
        "success": False,
        "svg_url": None,
        "svg_size": 0,
        "image_type": None,
        "error": None
    }
    
    temp_files = []
    
    try:
        # Parse input
        image_source = job_input.get("image")
        if not image_source:
            result["error"] = "No image provided"
            return result
        
        upscale_factor = job_input.get("upscale", 4)
        color_mode = job_input.get("color_mode", "color")
        vtracer_params = job_input.get("vtracer_params", {})
        
        # Step 1: Get input image
        input_path = os.path.join(config.temp_dir, f"{uuid.uuid4()}_input.png")
        temp_files.append(input_path)
        
        if image_source.startswith("http://") or image_source.startswith("https://"):
            print(f"Downloading image from: {image_source}")
            if not download_image(image_source, input_path):
                result["error"] = "Failed to download image"
                return result
        else:
            print("Decoding base64 image")
            if not decode_base64_image(image_source, input_path):
                result["error"] = "Failed to decode image"
                return result
        
        print(f"Input image saved: {input_path}")
        
        # Step 2: Classify image
        print("Classifying image...")
        image_type = classify_image(input_path)
        result["image_type"] = image_type
        print(f"Image type: {image_type}")
        
        # Step 3: Upscale
        print("Upscaling image...")
        upscaled_path = os.path.join(config.temp_dir, f"{uuid.uuid4()}_upscaled.png")
        temp_files.append(upscaled_path)
        
        success, bg_color, palette, has_gradients = upscale_image(
            input_path, 
            upscaled_path,
            target_mp=config.target_megapixels
        )
        
        if not success:
            result["error"] = "Upscaling failed"
            return result
        
        print(f"Upscaled image saved: {upscaled_path}")
        print(f"Background: {bg_color}, Has gradients: {has_gradients}")
        
        # Step 4: Vectorize
        output_path = os.path.join(config.outputs_dir, f"{uuid.uuid4()}.svg")
        temp_files.append(output_path)
        
        use_omnisvg = (
            image_type == "logo" 
            and config.use_omnisvg_for_logos 
            and OMNISVG_AVAILABLE
        )
        
        if use_omnisvg:
            print("Using OmniSVG for vectorization...")
            try:
                svg_path = vectorize_logo_with_omnisvg(upscaled_path, output_path)
                
                # Quality check
                if config.enable_quality_check and not quality_check(svg_path):
                    print("OmniSVG quality check failed, falling back to VTracer")
                    svg_path = vtracer_vectorize(upscaled_path, output_path, image_type, vtracer_params)
                    
            except Exception as e:
                print(f"OmniSVG error: {e}")
                if config.enable_vtracer_fallback:
                    print("Falling back to VTracer")
                    svg_path = vtracer_vectorize(upscaled_path, output_path, image_type, vtracer_params)
                else:
                    raise
        else:
            print(f"Using VTracer for {image_type} vectorization...")
            svg_path = vtracer_vectorize(upscaled_path, output_path, image_type, vtracer_params)
        
        # Post-process SVG colors
        print("Post-processing SVG...")
        fix_svg_colors(svg_path, palette)
        
        # Step 5: Upload SVG
        print("Uploading SVG...")
        original_filename = job_input.get("filename", "image.png")
        svg_url = upload_svg(svg_path, original_filename)
        
        if svg_url:
            result["svg_url"] = svg_url
            result["svg_size"] = os.path.getsize(svg_path)
            result["success"] = True
            print(f"Success! SVG URL: {svg_url}")
        else:
            # Return local path if upload fails
            result["svg_url"] = svg_path
            result["svg_size"] = os.path.getsize(svg_path)
            result["success"] = True
            result["warning"] = "Upload failed, returning local path"
        
    except Exception as e:
        print(f"Job processing error: {e}")
        import traceback
        traceback.print_exc()
        result["error"] = str(e)
    
    finally:
        # Cleanup temp files
        cleanup_temp_files(temp_files)
    
    return result


def handler(event):
    """
    RunPod serverless handler function.
    
    Args:
        event: RunPod job event containing input data
    
    Returns:
        Job result dict
    """
    print(f"Received job: {event.get('id', 'unknown')}")
    
    job_input = event.get("input", {})
    result = process_job(job_input)
    
    return result


# Start RunPod serverless handler
if __name__ == "__main__":
    print("Starting VectraHub GPU Server...")
    print(f"CUDA available: {os.system('nvidia-smi') == 0}")
    print(f"OmniSVG available: {OMNISVG_AVAILABLE}")
    
    runpod.serverless.start({"handler": handler})
