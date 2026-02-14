"""
VectraHub GPU Server - Main FastAPI Application

This module provides a FastAPI application for local testing.
In production (RunPod serverless), handler.py is used instead.
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from contextlib import asynccontextmanager
import os
import tempfile
import uuid

from app.config import config
from app.utils.image_classifier import classify_image
from app.utils.upscaler import upscale_image
from app.utils.vtracer import vectorize_image as vtracer_vectorize
from app.utils.omnisvg import vectorize_logo_with_omnisvg


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    config.ensure_directories()
    yield
    # Shutdown


app = FastAPI(
    title="VectraHub GPU Server",
    description="Image upscaling and vectorization API",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "VectraHub GPU Server"}


@app.get("/health")
async def health():
    """Detailed health check."""
    import torch
    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_memory": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
    }


@app.post("/process")
async def process_image(
    file: UploadFile = File(...),
    upscale: int = 4,
    color_mode: str = "color"
):
    """
    Process an image: classify, upscale, and vectorize.
    
    Args:
        file: Input image file
        upscale: Upscale factor (2, 4)
        color_mode: "color", "grayscale", or "bw"
    
    Returns:
        SVG file path and metadata
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")
    
    # Save uploaded file
    input_path = os.path.join(config.temp_dir, f"{uuid.uuid4()}.png")
    with open(input_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    try:
        # Step 1: Classify image
        image_type = classify_image(input_path)
        
        # Step 2: Upscale
        upscaled_path = os.path.join(config.temp_dir, f"{uuid.uuid4()}_upscaled.png")
        success, bg_color, palette, has_gradients = upscale_image(input_path, upscaled_path)
        
        if not success:
            raise HTTPException(500, "Upscaling failed")
        
        # Step 3: Vectorize
        output_path = os.path.join(config.outputs_dir, f"{uuid.uuid4()}.svg")
        
        if image_type == "logo" and config.use_omnisvg_for_logos:
            # Try OmniSVG first
            try:
                svg_path = vectorize_logo_with_omnisvg(upscaled_path, output_path)
                
                # Quality check
                if config.enable_quality_check:
                    from app.utils.omnisvg.core import quality_check
                    if not quality_check(svg_path):
                        # Fallback to VTracer
                        if config.enable_vtracer_fallback:
                            svg_path = vtracer_vectorize(upscaled_path, output_path, image_type)
            except Exception as e:
                if config.enable_vtracer_fallback:
                    svg_path = vtracer_vectorize(upscaled_path, output_path, image_type)
                else:
                    raise
        else:
            # Use VTracer for illustrations and line art
            svg_path = vtracer_vectorize(upscaled_path, output_path, image_type)
        
        return {
            "success": True,
            "image_type": image_type,
            "svg_path": svg_path,
            "has_gradients": has_gradients,
            "background_color": bg_color
        }
        
    finally:
        # Cleanup temp files
        if os.path.exists(input_path):
            os.remove(input_path)
        if os.path.exists(upscaled_path):
            os.remove(upscaled_path)


@app.get("/output/{filename}")
async def get_output(filename: str):
    """Serve output SVG files."""
    file_path = os.path.join(config.outputs_dir, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="image/svg+xml")
    raise HTTPException(404, "File not found")
