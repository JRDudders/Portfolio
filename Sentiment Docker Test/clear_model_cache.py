#!/usr/bin/env python3
"""
Clear HuggingFace model cache to resolve corrupted downloads.
Run this if you're getting model loading errors after SSL certificate issues.
"""

import os
import shutil
from pathlib import Path

def get_hf_cache_dir():
    """Get HuggingFace cache directory for current platform."""
    # Check environment variable first
    cache_dir = os.getenv("HF_HOME") or os.getenv("HUGGINGFACE_HUB_CACHE")

    if cache_dir:
        return Path(cache_dir)

    # Default locations
    if os.name == 'nt':  # Windows
        base = Path(os.getenv("USERPROFILE", "~"))
        cache_dir = base / ".cache" / "huggingface"
    else:  # Unix-like
        cache_dir = Path.home() / ".cache" / "huggingface"

    return cache_dir.expanduser()

def clear_model_cache(model_id: str = None):
    """Clear HuggingFace model cache."""
    cache_dir = get_hf_cache_dir()
    hub_dir = cache_dir / "hub"

    print(f"HuggingFace cache location: {cache_dir}")

    if not hub_dir.exists():
        print("No cache directory found. Nothing to clear.")
        return

    if model_id:
        # Clear specific model
        model_cache_name = f"models--{model_id.replace('/', '--')}"
        model_path = hub_dir / model_cache_name

        if model_path.exists():
            print(f"Clearing cache for model: {model_id}")
            print(f"Removing: {model_path}")
            try:
                shutil.rmtree(model_path)
                print(f"✓ Successfully cleared cache for {model_id}")
            except Exception as e:
                print(f"✗ Error clearing cache: {e}")
        else:
            print(f"No cache found for model: {model_id}")
    else:
        # Clear all models
        print("Clearing ALL model caches...")
        try:
            shutil.rmtree(hub_dir)
            print("✓ Successfully cleared all model caches")
        except Exception as e:
            print(f"✗ Error clearing cache: {e}")

if __name__ == "__main__":
    import sys

    # Default model to clear
    model = "cardiffnlp/twitter-roberta-base-sentiment-latest"

    if len(sys.argv) > 1:
        if sys.argv[1] == "--all":
            model = None
        else:
            model = sys.argv[1]

    clear_model_cache(model)

    print("\nNow restart run_local.py to re-download the model.")
