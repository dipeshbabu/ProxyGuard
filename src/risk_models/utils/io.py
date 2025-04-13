"""Input/output utilities for saving and loading models."""

import joblib
import os
from pathlib import Path

from risk_models.config import MODEL_SAVE_PATH


def save_model(model, path=None):
    """Save trained model to disk.
    
    Args:
        model: Trained model to save
        path: Path to save the model, defaults to MODEL_SAVE_PATH
        
    Returns:
        Path where model was saved
    """
    if path is None:
        path = MODEL_SAVE_PATH
        
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    joblib.dump(model, path)
    return path


def load_model(path=None):
    """Load model from disk.
    
    Args:
        path: Path to load the model from, defaults to MODEL_SAVE_PATH
        
    Returns:
        Loaded model
    """
    if path is None:
        path = MODEL_SAVE_PATH
        
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")
        
    return joblib.load(path) 