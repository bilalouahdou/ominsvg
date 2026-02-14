# VectraHub GPU Server

RunPod serverless GPU server for image upscaling and vectorization.

## Overview

This server receives images from pixel2vector.com, upscales them with Real-ESRGAN, 
vectorizes them (OmniSVG for logos, VTracer for illustrations), and returns the SVG result.

## Architecture

```
Image Upload → CLIP Classification → Real-ESRGAN Upscale → Vectorization → SVG Output
                    ↓                                              ↓
            (logo/illustration/line_art)              OmniSVG or VTracer
```

## File Structure

```
omnisvg-server/
├── Dockerfile                    # Docker build configuration
├── handler.py                    # RunPod serverless entry point
├── requirements.txt              # Python dependencies
├── requirements-base.txt         # Base deps (numpy)
├── README.md                     # This file
└── app/
    ├── __init__.py
    ├── config.py                 # Server configuration
    ├── main.py                   # FastAPI app (local testing)
    ├── tasks.py                  # Task definitions
    └── utils/
        ├── __init__.py
        ├── image_classifier.py   # CLIP-based image classification
        ├── svg_color_fixer.py    # SVG post-processing
        ├── cleanup.py            # Temp file cleanup
        │
        ├── upscaler/             # Real-ESRGAN module
        │   ├── __init__.py
        │   ├── core.py           # Smart upscale logic
        │   ├── upsampler.py      # Model loading
        │   └── palette.py        # LAB color correction
        │
        ├── vtracer/              # VTracer vectorization
        │   ├── __init__.py
        │   ├── core.py           # Main vectorization
        │   ├── image_type.py     # Image type detection
        │   ├── background.py     # Background removal
        │   ├── color_verify.py   # Color verification
        │   └── gradients.py      # Gradient detection
        │
        └── omnisvg/              # OmniSVG deep learning
            ├── __init__.py
            └── core.py           # Model loading & inference
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_BASE_URL` | `https://pixel2vector.com` | SVG upload destination |
| `SVG_UPLOAD_SECRET` | - | Auth for SVG upload API |
| `FILE_PROXY_SECRET` | - | Auth for file proxy API |
| `USE_OMNISVG` | `true` | Enable OmniSVG for logos |
| `ENABLE_FALLBACK` | `true` | Enable VTracer fallback |
| `HF_HOME` | `/runpod-volume` | HuggingFace model cache |

## GPU Memory Budget (A40 48GB)

| Component | VRAM |
|-----------|------|
| CLIP | ~1 GB |
| Real-ESRGAN | ~2 GB |
| OmniSVG (Qwen + weights) | ~17 GB |
| **Peak** | **~20 GB** |
| **Available** | **~28 GB** |

## Deployment

1. Build Docker image:
   ```bash
   docker build -t vectrahub-gpu-server .
   ```

2. Push to RunPod:
   - Connect GitHub repo
   - Auto-build on push

3. First run will download models to Network Volume (~5-10 min)

## API Format

### Input
```json
{
  "input": {
    "image": "https://pixel2vector.com/uploads/abc123.png",
    "upscale": 4,
    "color_mode": "color",
    "vtracer_params": {}
  }
}
```

### Output
```json
{
  "svg_url": "https://pixel2vector.com/php/api/file_proxy.php?...",
  "svg_size": 45230,
  "image_type": "logo",
  "success": true
}
```

## License

Proprietary - VectraHub
