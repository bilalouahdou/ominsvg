"""
VectraHub GPU Server - Cleanup Utilities

Temporary file management and cleanup.
"""
import os
import glob
import time
from typing import List, Optional

from app.config import config


def cleanup_temp_files(file_paths: Optional[List[str]] = None, max_age_hours: int = 1):
    """
    Clean up temporary files.
    
    Args:
        file_paths: Specific files to delete, or None to clean by age
        max_age_hours: Maximum age for auto-cleanup (default 1 hour)
    """
    if file_paths:
        # Delete specific files
        for path in file_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
                    print(f"Cleaned up: {path}")
            except Exception as e:
                print(f"Failed to delete {path}: {e}")
    else:
        # Clean old files in temp directory
        cleanup_old_files(config.temp_dir, max_age_hours)


def cleanup_old_files(directory: str, max_age_hours: int = 1):
    """
    Remove files older than specified hours.
    
    Args:
        directory: Directory to clean
        max_age_hours: Maximum file age in hours
    """
    try:
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        for pattern in ["*.png", "*.jpg", "*.jpeg", "*.svg", "*.tmp"]:
            for file_path in glob.glob(os.path.join(directory, pattern)):
                try:
                    file_age = current_time - os.path.getmtime(file_path)
                    if file_age > max_age_seconds:
                        os.remove(file_path)
                        print(f"Removed old file: {file_path}")
                except Exception as e:
                    print(f"Failed to remove {file_path}: {e}")
                    
    except Exception as e:
        print(f"Cleanup error: {e}")


def cleanup_outputs(max_age_hours: int = 24):
    """
    Clean up old output files.
    
    Args:
        max_age_hours: Maximum age for output files (default 24 hours)
    """
    cleanup_old_files(config.outputs_dir, max_age_hours)


def get_temp_path(suffix: str = ".tmp") -> str:
    """
    Generate a unique temporary file path.
    
    Args:
        suffix: File extension
        
    Returns:
        Unique temporary file path
    """
    import uuid
    return os.path.join(config.temp_dir, f"{uuid.uuid4()}{suffix}")
