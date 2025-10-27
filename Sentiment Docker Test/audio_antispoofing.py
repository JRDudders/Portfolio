# audio_antispoofing.py
"""
Audio deepfake detection using wav2vec2-large-anti-deepfake from HuggingFace.
Model: https://huggingface.co/nii-yamagishilab/wav2vec-large-anti-deepfake
"""

import os
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import torch
import librosa

# Use HuggingFace transformers (Python 3.12 compatible)
try:
    from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
    WAV2VEC_AVAILABLE = True
except ImportError:
    WAV2VEC_AVAILABLE = False
    import warnings
    warnings.warn("transformers library not available. Install with: pip install transformers")


# HuggingFace model (already fine-tuned for anti-deepfake)
MODEL_NAME = "nii-yamagishilab/wav2vec-large-anti-deepfake"


def load_audio(file_path: str, sr: int = 16000) -> np.ndarray:
    """Load audio file and resample to target sample rate"""
    audio, _ = librosa.load(file_path, sr=sr)
    return audio


def check_models_available() -> Dict[str, bool]:
    """Check if transformers library is available"""
    return {
        "transformers": WAV2VEC_AVAILABLE
    }


def download_models():
    """Models download automatically from HuggingFace on first use"""
    print(f"Model ({MODEL_NAME}) will be downloaded from HuggingFace on first use.")
    print("This will happen automatically when you first analyze audio.")


def predict_audio(audio_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu") -> Tuple[str, float, float]:
    """
    Predict if audio is bonafide or spoofed using HuggingFace model

    Returns:
        prediction: "bonafide" or "spoofed"
        confidence: confidence score (0-1)
        score: raw score (higher = more likely spoofed)
    """
    if not WAV2VEC_AVAILABLE:
        raise ImportError(
            "transformers library is required for audio deepfake detection.\n"
            "Install with: pip install transformers\n"
            "Or: pip install -r req.txt"
        )

    print(f"Loading model: {MODEL_NAME}")

    # Load model and processor from HuggingFace
    model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_NAME)
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)

    model.to(device)
    model.eval()

    # Load and process audio
    print("Processing audio file...")
    audio = load_audio(audio_path)

    # Process audio with the model's processor
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Predict
    print("Running inference...")
    with torch.no_grad():
        logits = model(**inputs).logits
        scores = torch.softmax(logits, dim=1)

        # Model outputs: [bonafide_score, spoof_score]
        bonafide_score = scores[0][0].item()
        spoof_score = scores[0][1].item()

        prediction = "bonafide" if bonafide_score > spoof_score else "spoofed"
        confidence = max(bonafide_score, spoof_score)

    print(f"Prediction: {prediction} (confidence: {confidence:.2%})")
    print(f"  Bonafide: {bonafide_score:.4f}, Spoofed: {spoof_score:.4f}")

    return prediction, confidence, spoof_score
