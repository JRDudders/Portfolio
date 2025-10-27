#!/usr/bin/env python3
"""
Diagnostic script to test audio deepfake detection model.
Run this to verify your model is working correctly.
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_model():
    print("=" * 70)
    print("Audio Deepfake Detection - Model Diagnostic")
    print("=" * 70)
    print()

    # Check imports
    print("1. Checking dependencies...")
    try:
        import librosa
        print("   ✓ librosa installed")
    except ImportError:
        print("   ✗ librosa NOT installed - run: pip install librosa")
        return False

    try:
        from transformers import Wav2Vec2Model
        print("   ✓ transformers installed")
    except ImportError:
        print("   ✗ transformers NOT installed - run: pip install transformers")
        return False

    try:
        import audio_antispoofing
        print("   ✓ audio_antispoofing module loaded")
    except Exception as e:
        print(f"   ✗ Failed to import audio_antispoofing: {e}")
        return False

    print()

    # Check model files
    print("2. Checking model files...")
    model_path = audio_antispoofing.get_antispoofing_model_path()
    if model_path:
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"   ✓ Found model: {model_path.name}")
        print(f"   ✓ Size: {size_mb:.2f} MB")
    else:
        print(f"   ✗ Model not found in {audio_antispoofing.MODEL_DIR}")
        print(f"   Expected filenames: {audio_antispoofing.ANTISPOOFING_MODEL_NAMES}")
        return False

    print()

    # Try loading the model
    print("3. Loading model...")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   Using device: {device}")

        model = audio_antispoofing.AntispoofingModel(device)
        print("   ✓ Model architecture created")

        checkpoint = torch.load(model_path, map_location=device)
        print(f"   ✓ Checkpoint loaded ({len(checkpoint)} parameters)")

        # Check if keys match
        model_keys = set(model.state_dict().keys())
        checkpoint_keys = set(checkpoint.keys())

        if model_keys == checkpoint_keys:
            print("   ✓ All model keys match checkpoint")
        else:
            missing = model_keys - checkpoint_keys
            unexpected = checkpoint_keys - model_keys
            if missing:
                print(f"   ⚠ Missing keys in checkpoint: {len(missing)}")
                if len(missing) <= 5:
                    for key in list(missing)[:5]:
                        print(f"      - {key}")
            if unexpected:
                print(f"   ⚠ Unexpected keys in checkpoint: {len(unexpected)}")
                if len(unexpected) <= 5:
                    for key in list(unexpected)[:5]:
                        print(f"      - {key}")

        model.load_state_dict(checkpoint, strict=False)
        model.to(device)
        model.eval()
        print("   ✓ Model loaded and set to eval mode")

    except Exception as e:
        print(f"   ✗ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return False

    print()

    # Test with synthetic audio
    print("4. Testing with synthetic audio...")
    try:
        # Create 4 seconds of random noise
        sample_rate = 16000
        duration = 4
        audio1 = np.random.randn(sample_rate * duration).astype(np.float32)
        audio2 = np.random.randn(sample_rate * duration).astype(np.float32)

        print("   Testing audio 1...")
        audio1_padded = audio_antispoofing.pad_audio(audio1)
        audio1_tensor = torch.FloatTensor(audio1_padded).unsqueeze(0).to(device)

        with torch.no_grad():
            output1 = model(audio1_tensor)
            scores1 = torch.softmax(output1, dim=1)
            pred1 = torch.argmax(output1, dim=1).item()
            conf1 = scores1[0][pred1].item()

        print(f"   Result 1: {'bonafide' if pred1 == 1 else 'spoofed'} (confidence: {conf1:.4f})")

        print("   Testing audio 2...")
        audio2_padded = audio_antispoofing.pad_audio(audio2)
        audio2_tensor = torch.FloatTensor(audio2_padded).unsqueeze(0).to(device)

        with torch.no_grad():
            output2 = model(audio2_tensor)
            scores2 = torch.softmax(output2, dim=1)
            pred2 = torch.argmax(output2, dim=1).item()
            conf2 = scores2[0][pred2].item()

        print(f"   Result 2: {'bonafide' if pred2 == 1 else 'spoofed'} (confidence: {conf2:.4f})")

        # Test determinism
        print("   Testing determinism (running audio 1 again)...")
        with torch.no_grad():
            output1_again = model(audio1_tensor)
            scores1_again = torch.softmax(output1_again, dim=1)
            pred1_again = torch.argmax(output1_again, dim=1).item()
            conf1_again = scores1_again[0][pred1_again].item()

        print(f"   Result 1 (repeat): {'bonafide' if pred1_again == 1 else 'spoofed'} (confidence: {conf1_again:.4f})")

        if abs(conf1 - conf1_again) < 0.0001:
            print("   ✓ Results are deterministic!")
        else:
            print(f"   ✗ Results are NOT deterministic! Difference: {abs(conf1 - conf1_again)}")
            return False

        if abs(conf1 - conf2) < 0.0001:
            print(f"   ⚠ WARNING: Different audio samples produce identical scores!")
            print(f"      This suggests the model is not working properly.")
            return False
        else:
            print(f"   ✓ Different audio produces different scores (diff: {abs(conf1 - conf2):.4f})")

    except Exception as e:
        print(f"   ✗ Testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print()
    print("=" * 70)
    print("✓ All tests passed! Model appears to be working correctly.")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = test_model()
    sys.exit(0 if success else 1)
