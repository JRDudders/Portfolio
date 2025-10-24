#!/usr/bin/env python3
"""
Helper script to download the anti-spoofing model.

The model needs to be manually downloaded from Google Drive due to size restrictions.
This script will guide you through the process.
"""

import os
from pathlib import Path

MODEL_DIR = Path(__file__).parent / "models" / "audio_antispoofing"
MODEL_PATH = MODEL_DIR / "best_model.pth"

GOOGLE_DRIVE_URL = "https://drive.google.com/drive/folders/1c4ywztEVlYVijfwbGLl9OEa1SNtFKppB"

def main():
    print("=" * 70)
    print("Audio Anti-Spoofing Model Download")
    print("=" * 70)
    print()

    # Create directory
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created model directory: {MODEL_DIR}")
    print()

    # Check if model exists
    if MODEL_PATH.exists():
        size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
        print(f"✓ Model already exists: {MODEL_PATH}")
        print(f"  Size: {size_mb:.2f} MB")
        print()
        print("If you want to re-download, delete the file and run this script again.")
        return

    print("Model not found. Please download it manually:")
    print()
    print("STEP 1: Open this URL in your browser:")
    print(f"  {GOOGLE_DRIVE_URL}")
    print()
    print("STEP 2: Look for one of these files:")
    print("  - best_SSL_model_LA.pth")
    print("  - best_model.pth")
    print("  - Any .pth file for the LA (Logical Access) track")
    print()
    print("STEP 3: Download the file")
    print()
    print("STEP 4: Save it to this location:")
    print(f"  {MODEL_PATH.absolute()}")
    print()
    print("STEP 5: Rename it to 'best_model.pth' if needed")
    print()
    print("=" * 70)
    print()
    print("Alternative: If you have the file already, you can:")
    print(f"  cp /path/to/your/model.pth {MODEL_PATH}")
    print()
    print("After downloading, run your application normally.")
    print("The model will be loaded automatically.")
    print()

if __name__ == "__main__":
    main()
