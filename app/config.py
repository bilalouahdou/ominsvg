"""
VectraHub GPU Server - Configuration

Server configuration and environment variable management.
"""
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class ServerConfig:
    """Server configuration settings."""
    
    # Base URL for uploads
    app_base_url: str = "https://pixel2vector.com"
    
    # Secrets
    svg_upload_secret: str = ""
    file_proxy_secret: str = ""
    
    # Paths
    models_dir: str = "/app/models"
    volume_dir: str = "/runpod-volume"
    outputs_dir: str = "/app/outputs"
    temp_dir: str = "/app/temp"
    
    # Feature flags
    use_omnisvg_for_logos: bool = True
    enable_vtracer_fallback: bool = True
    enable_quality_check: bool = True
    
    # Image processing
    target_megapixels: float = 12.0  # Target 12MP output
    max_image_size: int = 50_000_000  # 50MB max input
    
    # Upscaler settings
    upscaler_model: str = "RealESRGAN_x2plus"
    upscaler_tile: int = 0  # 0 = no tiling
    upscaler_half_precision: bool = True
    
    # Vectorizer settings
    vtracer_presets = {
        "logo": {
            "color_precision": 8,
            "layer_difference": 16,
            "max_iterations": 15
        },
        "illustration": {
            "color_precision": 6,
            "layer_difference": 32,
            "max_iterations": 10
        },
        "line_art": {
            "color_precision": 4,
            "layer_difference": 48,
            "max_iterations": 10
        }
    }
    
    # OmniSVG settings
    omnisvg_model_path: str = "/runpod-volume/OmniSVG"
    omnisvg_image_size: int = 448
    omnisvg_temperature: float = 0.3
    omnisvg_max_tokens: int = 2048
    
    @classmethod
    def from_env(cls) -> "ServerConfig":
        """Load configuration from environment variables."""
        config = cls()
        
        # Override with environment variables
        config.app_base_url = os.getenv("APP_BASE_URL", config.app_base_url)
        config.svg_upload_secret = os.getenv("SVG_UPLOAD_SECRET", "")
        config.file_proxy_secret = os.getenv("FILE_PROXY_SECRET", "")
        
        # Paths
        config.models_dir = os.getenv("MODELS_DIR", config.models_dir)
        config.volume_dir = os.getenv("VOLUME_DIR", config.volume_dir)
        config.outputs_dir = os.getenv("OUTPUTS_DIR", config.outputs_dir)
        config.temp_dir = os.getenv("TEMP_DIR", config.temp_dir)
        
        # Feature flags
        config.use_omnisvg_for_logos = os.getenv("USE_OMNISVG", "true").lower() == "true"
        config.enable_vtracer_fallback = os.getenv("ENABLE_FALLBACK", "true").lower() == "true"
        config.enable_quality_check = os.getenv("ENABLE_QUALITY_CHECK", "true").lower() == "true"
        
        # Set HuggingFace cache to volume at runtime
        if os.path.exists(config.volume_dir):
            os.environ["HF_HOME"] = config.volume_dir
            os.environ["TRANSFORMERS_CACHE"] = config.volume_dir
        
        return config
    
    def ensure_directories(self):
        """Ensure all required directories exist."""
        for path in [self.models_dir, self.outputs_dir, self.temp_dir]:
            os.makedirs(path, exist_ok=True)
        
        # Create volume subdirectories if volume is mounted
        if os.path.exists(self.volume_dir):
            os.makedirs(os.path.join(self.volume_dir, "hub"), exist_ok=True)


# Global config instance
config = ServerConfig.from_env()
