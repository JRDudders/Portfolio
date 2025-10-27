# audio_antispoofing.py
"""
Audio deepfake detection using HuggingFace Inference API.
No local model downloads required - all inference happens via API calls.
"""

import os
from pathlib import Path
from typing import Tuple, Dict
import requests

import numpy as np
import librosa


# HuggingFace Inference API
INFERENCE_API_URL = "https://api-inference.huggingface.co/models/nii-yamagishilab/wav2vec2-large-960h-lv60-self"
# Alternative models to try if primary fails
ALTERNATIVE_MODELS = [
    "facebook/wav2vec2-base",
    "facebook/wav2vec2-large-960h"
]

# Get API key from environment (optional - works without it but may have rate limits)
HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")


def load_audio(file_path: str, sr: int = 16000) -> np.ndarray:
    """Load audio file and resample to target sample rate"""
    audio, _ = librosa.load(file_path, sr=sr)
    return audio


def check_models_available() -> Dict[str, bool]:
    """Check if API is reachable"""
    try:
        response = requests.head(INFERENCE_API_URL, timeout=5)
        return {"api_available": response.status_code in [200, 401, 403, 503]}
    except:
        return {"api_available": False}


def download_models():
    """No downloads needed - using API"""
    print("Audio deepfake detection uses HuggingFace Inference API.")
    print("No model downloads required!")
    if not HF_API_KEY:
        print("\nNote: For better performance and higher rate limits, set HUGGINGFACE_API_KEY environment variable.")
        print("Get your API key from: https://huggingface.co/settings/tokens")


def query_api(audio_bytes: bytes, max_retries: int = 3) -> Dict:
    """
    Query HuggingFace Inference API for audio classification

    Returns:
        dict with 'label' and 'score' keys, or error info
    """
    headers = {}
    if HF_API_KEY:
        headers["Authorization"] = f"Bearer {HF_API_KEY}"

    # Try primary model
    for attempt in range(max_retries):
        try:
            response = requests.post(
                INFERENCE_API_URL,
                headers=headers,
                data=audio_bytes,
                timeout=30
            )

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 503:
                # Model is loading, wait and retry
                if attempt < max_retries - 1:
                    print(f"Model loading, retrying in 10 seconds... (attempt {attempt + 1}/{max_retries})")
                    import time
                    time.sleep(10)
                    continue

            # If we get here, there was an error
            return {"error": f"API returned status {response.status_code}: {response.text}"}

        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                print(f"Request timeout, retrying... (attempt {attempt + 1}/{max_retries})")
                continue
            return {"error": "API request timed out after multiple retries"}
        except Exception as e:
            return {"error": f"API request failed: {str(e)}"}

    return {"error": "Max retries exceeded"}


def predict_audio(audio_path: str, device: str = None) -> Tuple[str, float, float]:
    """
    Predict if audio is bonafide or spoofed using HuggingFace Inference API

    Note: device parameter is ignored (kept for API compatibility)

    Returns:
        prediction: "bonafide" or "spoofed"
        confidence: confidence score (0-1)
        score: raw score
    """
    print("Loading audio file...")
    audio = load_audio(audio_path)

    # Convert to bytes for API
    import io
    import soundfile as sf

    buffer = io.BytesIO()
    sf.write(buffer, audio, 16000, format='WAV')
    audio_bytes = buffer.getvalue()

    print(f"Sending audio to HuggingFace Inference API...")
    print(f"Audio size: {len(audio_bytes) / 1024:.2f} KB")

    # Query the API
    result = query_api(audio_bytes)

    if "error" in result:
        raise Exception(result["error"])

    # For now, use a simple heuristic since the base model isn't trained for deepfake detection
    # This is a placeholder - in production you'd use a proper deepfake detection API
    print("\nWARNING: Using base wav2vec2 model as placeholder.")
    print("For production use, consider:")
    print("  1. HuggingFace Pro account with access to specialized models")
    print("  2. Commercial deepfake detection API (e.g., Deepgram, AssemblyAI)")
    print("  3. Self-hosted model with Python 3.10 + fairseq")

    # Placeholder logic - analyze audio features as a heuristic
    # Real deepfake detection would use a trained model
    duration = len(audio) / 16000

    # Simple heuristics (NOT RELIABLE - just for demonstration)
    # Real deepfakes often have:
    # - Very consistent volume
    # - Lack of natural breath sounds
    # - Unusual spectral characteristics

    volume_std = np.std(np.abs(audio))
    mean_energy = np.mean(audio ** 2)

    # Very rough heuristic score (DO NOT USE IN PRODUCTION)
    # This is just to show the UI works
    spoof_score = 0.5  # Default: uncertain

    if volume_std < 0.05:  # Very consistent volume
        spoof_score += 0.2
    if mean_energy > 0.01:  # High energy
        spoof_score -= 0.1

    spoof_score = max(0.0, min(1.0, spoof_score))

    prediction = "spoofed" if spoof_score > 0.5 else "bonafide"
    confidence = abs(spoof_score - 0.5) * 2  # Convert to 0-1 confidence

    print(f"\nPlaceholder analysis:")
    print(f"  Duration: {duration:.2f}s")
    print(f"  Volume std: {volume_std:.4f}")
    print(f"  Mean energy: {mean_energy:.6f}")
    print(f"  Prediction: {prediction} (confidence: {confidence:.2%})")
    print(f"  Spoof score: {spoof_score:.4f}")

    return prediction, confidence, spoof_score
