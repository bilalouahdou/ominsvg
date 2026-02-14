"""
VectraHub GPU Server - Image Classifier

Uses OpenAI CLIP for zero-shot classification of images into:
- logo (company logos, brand icons)
- illustration (artwork with gradients)
- line_art (line drawings, sketches)
"""
import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from typing import Literal

from app.config import config

# Image type literal
ImageType = Literal["logo", "illustration", "line_art"]

# Classification prompts
CLASS_PROMPTS = {
    "logo": [
        "a company logo or brand icon",
        "a business logo design",
        "a brand mark or emblem"
    ],
    "illustration": [
        "an illustration or artwork with gradients",
        "a digital illustration with colors",
        "artistic image with shading and colors"
    ],
    "line_art": [
        "a line art drawing or sketch",
        "a black and white line drawing",
        "a simple outline drawing"
    ]
}

# Singleton model cache
_model_cache = {
    "model": None,
    "processor": None
}


def get_clip_model():
    """Load or retrieve cached CLIP model."""
    if _model_cache["model"] is None:
        model_name = "openai/clip-vit-base-patch32"
        
        print(f"Loading CLIP model: {model_name}")
        _model_cache["model"] = CLIPModel.from_pretrained(
            model_name,
            cache_dir=config.models_dir
        )
        _model_cache["processor"] = CLIPProcessor.from_pretrained(
            model_name,
            cache_dir=config.models_dir
        )
        
        # Move to GPU if available
        if torch.cuda.is_available():
            _model_cache["model"] = _model_cache["model"].cuda()
            print("CLIP model moved to CUDA")
    
    return _model_cache["model"], _model_cache["processor"]


def classify_image(image_path: str) -> ImageType:
    """
    Classify an image as logo, illustration, or line_art using CLIP.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        One of: "logo", "illustration", "line_art"
    """
    try:
        # Load model
        model, processor = get_clip_model()
        
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        
        # Prepare text inputs (flatten all prompts)
        all_prompts = []
        class_indices = []
        for class_name, prompts in CLASS_PROMPTS.items():
            for prompt in prompts:
                all_prompts.append(prompt)
                class_indices.append(class_name)
        
        # Process inputs
        inputs = processor(
            text=all_prompts,
            images=image,
            return_tensors="pt",
            padding=True
        )
        
        # Move to GPU
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
        
        # Aggregate scores by class
        class_scores = {"logo": 0.0, "illustration": 0.0, "line_art": 0.0}
        for idx, class_name in enumerate(class_indices):
            class_scores[class_name] += probs[0][idx].item()
        
        # Normalize scores
        total = sum(class_scores.values())
        if total > 0:
            class_scores = {k: v/total for k, v in class_scores.items()}
        
        # Get best class
        best_class = max(class_scores, key=class_scores.get)
        
        print(f"Classification scores: {class_scores}")
        print(f"Selected class: {best_class}")
        
        return best_class
        
    except Exception as e:
        print(f"Classification error: {e}")
        # Default to illustration as safest choice
        return "illustration"


def classify_image_simple(image_path: str) -> ImageType:
    """
    Simple fallback classifier using image properties.
    Used when CLIP is unavailable.
    """
    try:
        from PIL import Image
        import numpy as np
        
        img = Image.open(image_path).convert("RGB")
        arr = np.array(img)
        
        # Calculate color variance
        color_variance = np.var(arr, axis=(0, 1)).mean()
        
        # Check for transparency (would be RGBA)
        has_alpha = img.mode == "RGBA"
        
        # Simple heuristics
        if color_variance < 1000:
            # Low variance = likely line art
            return "line_art"
        elif has_alpha or color_variance < 5000:
            # Medium variance with transparency = likely logo
            return "logo"
        else:
            # High variance = illustration
            return "illustration"
            
    except Exception as e:
        print(f"Simple classification error: {e}")
        return "illustration"
