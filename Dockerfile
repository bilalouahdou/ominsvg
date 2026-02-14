# VectraHub GPU Server - RunPod Serverless Dockerfile
# Base: PyTorch 2.2.0 with CUDA 12.1.1 on Ubuntu 22.04

FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME=/app/models
ENV TORCH_HOME=/app/models
ENV TRANSFORMERS_CACHE=/app/models
ENV HF_DATASETS_CACHE=/app/models
ENV XDG_CACHE_HOME=/app/models

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    libcairo2-dev \
    pkg-config \
    python3-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements-base.txt requirements.txt ./

# Install base dependencies (numpy first to avoid version conflicts)
RUN pip install --no-cache-dir -r requirements-base.txt

# Install main dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create directories
RUN mkdir -p /app/models /app/outputs /app/temp /app/omnisvg_repo

# Pre-download CLIP model (~600MB) - baked into image
RUN python3 -c "from transformers import CLIPProcessor, CLIPModel; CLIPModel.from_pretrained('openai/clip-vit-base-patch32'); CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')" || true

# Pre-download Real-ESRGAN model (~65MB) - baked into image
RUN python3 -c "
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
upsampler = RealESRGANer(scale=2, model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth', model=model, tile=0, pre_pad=0, half=True)
" || true

# Clone OmniSVG repository (code only, ~50MB) - baked into image
# Models will be downloaded at runtime to Network Volume
RUN git clone https://github.com/OmniSVG/OmniSVG.git /app/omnisvg_repo/OmniSVG || \
    (cd /app/omnisvg_repo/OmniSVG && git pull origin main) || true

# Copy application code
COPY . /app/

# Ensure proper permissions
RUN chmod -R 755 /app

# Expose port (for local testing, RunPod uses its own port management)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import sys; sys.exit(0)"

# RunPod serverless entry point
CMD ["python3", "-u", "handler.py"]
