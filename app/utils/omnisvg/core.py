"""
VectraHub GPU Server - OmniSVG Core

Deep learning SVG generation for logos using Qwen2.5-VL.
Based on OmniSVG architecture with Qwen2.5-VL encoder and SketchDecoder.
"""
import os
import sys
import torch
import numpy as np
from PIL import Image
from typing import Optional, Tuple, List
import warnings

from app.config import config

# Suppress warnings
warnings.filterwarnings("ignore")

# Model cache
_model_cache = {
    "loaded": False,
    "model": None,
    "processor": None,
    "tokenizer": None,
    "device": None
}

# OmniSVG repo path
OMNISVG_REPO_PATH = "/app/omnisvg_repo/OmniSVGf"


def load_omnisvg_model():
    """
    Load OmniSVG model and dependencies.
    
    Architecture:
    - Qwen2.5-VL-7B-Instruct (base model)
    - OmniSVG1.1_8B (fine-tuned weights)
    - SVGTokenizer for output decoding
    
    Models are cached in /runpod-volume for persistence.
    """
    global _model_cache
    
    if _model_cache["loaded"]:
        return _model_cache
    
    print("Loading OmniSVG model...")
    
    # Set HuggingFace cache to volume for persistence
    volume_cache = os.path.join(config.volume_dir, "hub")
    os.makedirs(volume_cache, exist_ok=True)
    os.environ["HF_HOME"] = config.volume_dir
    os.environ["TRANSFORMERS_CACHE"] = config.volume_dir
    os.environ["HF_DATASETS_CACHE"] = config.volume_dir
    
    try:
        # Import transformers
        from transformers import (
            Qwen2_5_VLForConditionalGeneration,
            AutoProcessor,
            AutoTokenizer
        )
        
        # Model configuration
        model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading Qwen2.5-VL model on {device}...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=volume_cache,
            trust_remote_code=True
        )
        
        # Load processor for vision inputs
        processor = AutoProcessor.from_pretrained(
            model_name,
            cache_dir=volume_cache,
            trust_remote_code=True
        )
        
        # Load model with optimizations for inference
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            cache_dir=volume_cache,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True
        )
        
        if device == "cpu":
            model = model.to(device)
        
        # Load OmniSVG fine-tuned weights if available
        omnisvg_weights_path = os.path.join(config.volume_dir, "omnisvg_weights")
        if os.path.exists(omnisvg_weights_path):
            print("Loading OmniSVG fine-tuned weights...")
            # Load adapter weights (LoRA or full)
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, omnisvg_weights_path)
        else:
            print("Warning: OmniSVG weights not found, using base Qwen model")
            print(f"Expected weights at: {omnisvg_weights_path}")
        
        # Cache models
        _model_cache["model"] = model
        _model_cache["processor"] = processor
        _model_cache["tokenizer"] = tokenizer
        _model_cache["device"] = device
        _model_cache["loaded"] = True
        
        print("OmniSVG model loaded successfully!")
        
    except Exception as e:
        print(f"Error loading OmniSVG model: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    return _model_cache


def preprocess_image(image_path: str, target_size: int = 448) -> Image.Image:
    """
    Preprocess image for OmniSVG input.
    
    Args:
        image_path: Path to input image
        target_size: Target size for model input (default 448x448)
        
    Returns:
        Preprocessed PIL Image
    """
    img = Image.open(image_path).convert("RGB")
    
    # Resize to target size
    img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
    
    # Add white background if needed (for logos on transparent backgrounds)
    # This is already handled by convert("RGB")
    
    return img


def generate_svg_tokens(
    image: Image.Image,
    model,
    processor,
    tokenizer,
    device: str,
    temperature: float = 0.3,
    max_tokens: int = 2048
) -> str:
    """
    Generate SVG tokens from image using the model.
    
    Args:
        image: Preprocessed PIL Image
        model: Loaded model
        processor: Vision processor
        tokenizer: Text tokenizer
        device: Device string
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        
    Returns:
        Generated SVG token string
    """
    # Prepare system prompt for SVG generation
    system_prompt = """You are an expert SVG vector graphics generator. 
Given an image of a logo or icon, generate clean, optimized SVG code.
The SVG should be simple, well-formed, and maintain the visual appearance of the original.
Output only valid SVG code without markdown formatting."""
    
    # Create conversation format
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image
                },
                {
                    "type": "text",
                    "text": "Convert this image to SVG format. Generate clean, scalable vector paths."
                }
            ]
        }
    ]
    
    # Process inputs
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True
    )
    
    # Move to device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
              for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    generated_ids = outputs[:, inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    return response


def decode_svg_tokens(token_string: str) -> str:
    """
    Decode generated tokens to valid SVG.
    
    The model output might be wrapped in markdown code blocks or contain
    extra text. This function extracts clean SVG.
    
    Args:
        token_string: Raw model output
        
    Returns:
        Clean SVG string
    """
    import re
    
    # Remove markdown code blocks
    svg_match = re.search(r'```(?:svg|xml)?\s*(.*?)```', token_string, re.DOTALL)
    if svg_match:
        svg_content = svg_match.group(1).strip()
    else:
        svg_content = token_string.strip()
    
    # Ensure it starts with XML declaration or SVG tag
    if not svg_content.startswith('<?xml') and not svg_content.startswith('<svg'):
        # Try to find SVG tag
        svg_start = svg_content.find('<svg')
        if svg_start != -1:
            svg_content = svg_content[svg_start:]
    
    # Ensure valid SVG structure
    if '<svg' not in svg_content:
        # Wrap in SVG tag if missing
        svg_content = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">{svg_content}</svg>'
    
    return svg_content


def fill_missing_colors(svg_content: str) -> str:
    """
    Fill missing colors in SVG paths.
    
    Ensures all paths have a fill or stroke attribute.
    
    Args:
        svg_content: SVG string
        
    Returns:
        SVG with filled colors
    """
    import re
    from xml.etree import ElementTree as ET
    
    try:
        # Parse SVG
        root = ET.fromstring(svg_content)
        
        # Find all path elements
        for elem in root.iter():
            tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
            
            if tag in ('path', 'rect', 'circle', 'ellipse', 'polygon', 'polyline'):
                # Ensure fill or stroke
                if 'fill' not in elem.attrib and 'stroke' not in elem.attrib:
                    elem.set('fill', 'black')
        
        # Convert back to string
        return ET.tostring(root, encoding='unicode')
        
    except Exception as e:
        print(f"Color filling error: {e}")
        return svg_content


def set_viewbox(svg_content: str, width: int = 448, height: int = 448) -> str:
    """
    Ensure SVG has proper viewBox.
    
    Args:
        svg_content: SVG string
        width: ViewBox width
        height: ViewBox height
        
    Returns:
        SVG with viewBox
    """
    import re
    
    # Check if viewBox exists
    if 'viewBox' not in svg_content:
        # Add viewBox after svg tag opening
        svg_content = re.sub(
            r'(<svg[^>]*)>',
            rf'\1 viewBox="0 0 {width} {height}">',
            svg_content
        )
    
    return svg_content


def vectorize_logo_with_omnisvg(input_path: str, output_path: str) -> str:
    """
    Vectorize a logo image using OmniSVG.
    
    Args:
        input_path: Path to input image
        output_path: Path for output SVG
        
    Returns:
        Path to generated SVG file
    """
    print(f"Vectorizing with OmniSVG: {input_path}")
    
    # Load model
    cache = load_omnisvg_model()
    model = cache["model"]
    processor = cache["processor"]
    tokenizer = cache["tokenizer"]
    device = cache["device"]
    
    # Preprocess image
    image = preprocess_image(input_path, target_size=config.omnisvg_image_size)
    
    # Generate SVG tokens
    print("Generating SVG...")
    token_string = generate_svg_tokens(
        image,
        model,
        processor,
        tokenizer,
        device,
        temperature=config.omnisvg_temperature,
        max_tokens=config.omnisvg_max_tokens
    )
    
    # Decode to SVG
    svg_content = decode_svg_tokens(token_string)
    
    # Post-process
    svg_content = fill_missing_colors(svg_content)
    svg_content = set_viewbox(svg_content, config.omnisvg_image_size, config.omnisvg_image_size)
    
    # Save
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(svg_content)
    
    print(f"OmniSVG output saved: {output_path}")
    
    return output_path


def quality_check(svg_path: str) -> bool:
    """
    Check if OmniSVG output is usable.
    
    Fail conditions:
    - File too small (likely empty)
    - No paths/shapes generated
    - Invalid SVG structure
    
    Args:
        svg_path: Path to SVG file
        
    Returns:
        True if quality is acceptable
    """
    try:
        # Check file size
        file_size = os.path.getsize(svg_path)
        if file_size < 500:
            print(f"Quality check failed: File too small ({file_size} bytes)")
            return False
        
        # Read content
        with open(svg_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for SVG content
        if '<svg' not in content.lower():
            print("Quality check failed: No SVG tag found")
            return False
        
        # Check for paths or shapes
        shape_tags = ['<path', '<rect', '<circle', '<ellipse', '<polygon', '<polyline', '<line']
        has_shapes = any(tag in content.lower() for tag in shape_tags)
        
        if not has_shapes:
            print("Quality check failed: No vector shapes found")
            return False
        
        # Try to parse as XML
        from xml.etree import ElementTree as ET
        try:
            ET.fromstring(content)
        except ET.ParseError as e:
            print(f"Quality check failed: Invalid XML - {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"Quality check error: {e}")
        return False
