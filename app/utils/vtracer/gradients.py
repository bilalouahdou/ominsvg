"""
VectraHub GPU Server - Gradient Detection

Detect if an image has gradients and mark for appropriate processing.
"""
import cv2
import numpy as np


def detect_gradients(image_path: str) -> bool:
    """
    Detect if an image contains gradients.
    
    Gradients are detected by:
    1. High local color variance
    2. Smooth transitions between colors
    
    Args:
        image_path: Path to image
        
    Returns:
        True if gradients detected
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False
        
        # Convert to HSV for better gradient detection
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Calculate gradients in each channel
        h_grad = cv2.Sobel(hsv[:, :, 0], cv2.CV_64F, 1, 1, ksize=3)
        s_grad = cv2.Sobel(hsv[:, :, 1], cv2.CV_64F, 1, 1, ksize=3)
        v_grad = cv2.Sobel(hsv[:, :, 2], cv2.CV_64F, 1, 1, ksize=3)
        
        # Combine gradients
        total_grad = np.abs(h_grad) + np.abs(s_grad) + np.abs(v_grad)
        
        # Calculate statistics
        mean_grad = np.mean(total_grad)
        std_grad = np.std(total_grad)
        
        # High mean indicates many edges
        # High std indicates mix of flat and gradient areas
        # We look for moderate values that suggest smooth gradients
        gradient_score = mean_grad * std_grad
        
        # Threshold for gradient detection (tuned empirically)
        return gradient_score > 10000
        
    except Exception as e:
        print(f"Gradient detection error: {e}")
        return False
