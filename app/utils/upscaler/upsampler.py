"""
VectraHub GPU Server - Real-ESRGAN Upsampler

Model loading and inference for Real-ESRGAN upscaling.
"""
import os
import torch
from realesrgan import RealESRGANer

from app.config import config

# Model cache
_upsampler_cache = {}

# Model URLs
MODEL_URLS = {
    "RealESRGAN_x2plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    "RealESRGAN_x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
}


def download_model(model_name: str, model_path: str):
    """Download model if not exists."""
    if os.path.exists(model_path):
        return
    
    import urllib.request
    url = MODEL_URLS.get(model_name)
    if not url:
        raise ValueError(f"Unknown model: {model_name}")
    
    print(f"Downloading {model_name}...")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    urllib.request.urlretrieve(url, model_path)
    print(f"Downloaded to {model_path}")


def load_realesrgan_upsampler(
    scale: int = 2,
    model_name: str = None,
    tile: int = 0,
    tile_pad: int = 10,
    pre_pad: int = 0,
    half: bool = True,
    gpu_id: int = None
) -> RealESRGANer:
    """
    Load or retrieve cached Real-ESRGAN upsampler.
    
    Args:
        scale: Upscaling factor (2 or 4)
        model_name: Model name (auto-detected from scale if not specified)
        tile: Tile size for splitting large images (0 = no tiling)
        tile_pad: Padding for tiles
        pre_pad: Pre-padding
        half: Use half precision (FP16)
        gpu_id: GPU device ID (None = auto)
        
    Returns:
        RealESRGANer instance
    """
    cache_key = f"{scale}_{tile}_{half}"
    
    if cache_key in _upsampler_cache:
        return _upsampler_cache[cache_key]
    
    # Determine model
    if model_name is None:
        model_name = f"RealESRGAN_x{scale}plus"
    
    # Model path
    model_path = os.path.join(config.models_dir, f"{model_name}.pth")
    
    # Download if needed
    if not os.path.exists(model_path):
        download_model(model_name, model_path)
    
    # Create model architecture - lazy import to avoid torchvision compatibility issues
    from basicsr.archs.rrdbnet_arch import RRDBNet
    
    if scale == 2:
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=2
        )
    elif scale == 4:
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4
        )
    else:
        raise ValueError(f"Unsupported scale: {scale}")
    
    # Create upsampler
    upsampler = RealESRGANer(
        scale=scale,
        model_path=model_path,
        model=model,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        half=half,
        gpu_id=gpu_id
    )
    
    # Cache and return
    _upsampler_cache[cache_key] = upsampler
    return upsampler
