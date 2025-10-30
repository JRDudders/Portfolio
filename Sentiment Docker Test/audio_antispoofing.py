# audio_antispoofing.py
"""
Audio deepfake detection using wav2vec-large-anti-deepfake model.
Model: https://huggingface.co/nii-yamagishilab/wav2vec-large-anti-deepfake-nda

Supports two modes:
1. Local inference (requires fairseq - Python 3.10 recommended)
2. API inference (works with any Python version, no fairseq needed)
"""

import os
from pathlib import Path
from typing import Tuple, Dict
import io

import torch
import numpy as np
import librosa
import requests

# Check if fairseq is available for local inference
try:
    from fairseq.models.wav2vec import Wav2Vec2Model, Wav2Vec2Config
    from huggingface_hub import PyTorchModelHubMixin
    FAIRSEQ_AVAILABLE = True
except ImportError:
    FAIRSEQ_AVAILABLE = False
    print("fairseq not available - will use API inference mode")


# Model name on HuggingFace
MODEL_NAME = "nii-yamagishilab/wav2vec-large-anti-deepfake-nda"
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"

# Get API key from environment (optional but recommended for rate limits)
HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Global model instance (loaded once, only used if fairseq available)
_model = None
_device = None


# === Local Inference Mode (requires fairseq) ===

if FAIRSEQ_AVAILABLE:
    # === Wrapper for the SSL model ===
    class SSLModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Model config used to build SSL architecture
            cfg = Wav2Vec2Config(
                quantize_targets=True,
                extractor_mode="layer_norm",
                layer_norm_first=True,
                final_dim=768,
                latent_temp=(2.0, 0.1, 0.999995),
                encoder_layerdrop=0.0,
                dropout_input=0.0,
                dropout_features=0.0,
                dropout=0.0,
                attention_dropout=0.0,
                conv_bias=True,
                encoder_layers=24,
                encoder_embed_dim=1024,
                encoder_ffn_embed_dim=4096,
                encoder_attention_heads=16,
                feature_grad_mult=1.0
            )
            # Initialize SSL model with random weights
            self.model = Wav2Vec2Model(cfg)

        def extract_feat(self, input_data):
            # If input has shape (B, T, 1), squeeze the last dim
            if input_data.ndim == 3:
                input_data = input_data[:, :, 0]
            # Extract features
            with torch.no_grad():
                features = self.model(input_data, mask=False, features_only=True)['x']
            return features


    # === The actual deepfake detection model using SSL frontend + FC backend ===
    class DeepfakeDetector(torch.nn.Module, PyTorchModelHubMixin):
        def __init__(self):
            super().__init__()
            self.ssl_orig_output_dim = 1024
            self.num_classes = 2

            # Frontend: SSL model
            self.m_ssl = SSLModel()

            # Backend: Pooling + Classification
            self.adap_pool1d = torch.nn.AdaptiveAvgPool1d(output_size=1)
            self.proj_fc = torch.nn.Linear(
                in_features=self.ssl_orig_output_dim,
                out_features=self.num_classes,
            )

        def forward(self, wav):
            emb = self.m_ssl.extract_feat(wav)  # [B, T, D]
            emb = emb.transpose(1, 2)           # [B, D, T]
            pooled_emb = self.adap_pool1d(emb)  # [B, D, 1]
            pooled_emb = pooled_emb.squeeze(-1) # [B, D]
            logits = self.proj_fc(pooled_emb)   # [B, 2]
            return logits


def load_wav_and_preprocess(wav_path: str, target_sr: int = 16000) -> torch.Tensor:
    """
    Load audio file and preprocess for model input

    Args:
        wav_path: Path to audio file
        target_sr: Target sampling rate (default 16kHz)

    Returns:
        Preprocessed waveform tensor ready for model input
    """
    # Load audio file using librosa (handles all formats, no codec issues)
    wav, sr = librosa.load(wav_path, sr=target_sr, mono=True)

    # Convert to torch tensor
    wav = torch.from_numpy(wav).float()

    # Normalize waveform
    with torch.no_grad():
        wav = torch.nn.functional.layer_norm(wav, wav.shape)

    # Add batch dimension and return
    return wav.unsqueeze(0)


def check_models_available() -> Dict[str, bool]:
    """
    Check if required libraries are available
    """
    return {
        "fairseq": FAIRSEQ_AVAILABLE,
        "torch": True,
        "librosa": True,
        "api_mode": not FAIRSEQ_AVAILABLE
    }


def download_models():
    """
    Download and initialize the deepfake detection model (local mode only)
    """
    global _model, _device

    if not FAIRSEQ_AVAILABLE:
        print("=" * 70)
        print("API INFERENCE MODE")
        print("=" * 70)
        print("fairseq not available - using HuggingFace Inference API instead")
        print("No model downloads needed!")
        if not HF_API_KEY:
            print("\nNote: For better performance and no rate limits,")
            print("set HUGGINGFACE_API_KEY environment variable:")
            print("  export HUGGINGFACE_API_KEY='your_token_here'")
            print("  Get token from: https://huggingface.co/settings/tokens")
        print("=" * 70)
        return

    print(f"Loading model: {MODEL_NAME}")
    print("This will download the model from HuggingFace on first use (~1.2GB)")

    # Set device
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {_device}")

    # Load model from HuggingFace
    try:
        _model = DeepfakeDetector.from_pretrained(MODEL_NAME)
        _model.to(_device)
        _model.eval()
        print("Model loaded successfully! Using local inference mode.")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


def query_api(audio_bytes: bytes, max_retries: int = 3) -> Dict:
    """
    Query HuggingFace Inference API for deepfake detection

    Args:
        audio_bytes: Audio file as bytes
        max_retries: Number of retry attempts

    Returns:
        dict with prediction results or error
    """
    headers = {}
    if HF_API_KEY:
        headers["Authorization"] = f"Bearer {HF_API_KEY}"

    for attempt in range(max_retries):
        try:
            response = requests.post(
                API_URL,
                headers=headers,
                data=audio_bytes,
                timeout=3600
            )

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 503:
                # Model is loading, wait and retry
                if attempt < max_retries - 1:
                    wait_time = 10 * (attempt + 1)
                    print(f"Model loading on server, waiting {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                    import time
                    time.sleep(wait_time)
                    continue
            else:
                return {"error": f"API error {response.status_code}: {response.text[:200]}"}

        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                print(f"Request timeout, retrying... (attempt {attempt + 1}/{max_retries})")
                continue
            return {"error": "API request timed out"}
        except Exception as e:
            return {"error": f"API request failed: {str(e)}"}

    return {"error": "Max retries exceeded"}


def predict_audio_api(audio_path: str) -> Tuple[str, float, float]:
    """
    Predict using HuggingFace Inference API (no fairseq needed)

    Note: The wav2vec-large-anti-deepfake-nda model is not available via free API.
    This falls back to audio analysis heuristics.

    Args:
        audio_path: Path to audio file

    Returns:
        prediction, confidence, spoof_score
    """
    print("Loading audio file...")
    wav = load_wav_and_preprocess(audio_path)

    # Convert to WAV bytes for API
    buffer = io.BytesIO()
    # Convert back to numpy for soundfile
    audio_np = wav.squeeze(0).numpy()

    import soundfile as sf
    sf.write(buffer, audio_np, 16000, format='WAV')
    audio_bytes = buffer.getvalue()

    print(f"Attempting HuggingFace API... ({len(audio_bytes)/1024:.1f} KB)")
    result = query_api(audio_bytes)

    # If API fails (which it likely will for this model), use heuristics
    if "error" in result:
        error_msg = result['error']
        print(f"\nAPI Error: {error_msg[:100]}...")

        if "401" in error_msg or "404" in error_msg:
            print("\n" + "=" * 70)
            print("MODEL NOT AVAILABLE VIA FREE API")
            print("=" * 70)
            print("The wav2vec-large-anti-deepfake-nda model requires either:")
            print("  1. Local inference (fairseq + Python 3.10)")
            print("  2. HuggingFace Pro account with inference endpoints")
            print("\nFalling back to audio analysis heuristics...")
            print("=" * 70)

            # Use improved heuristics as fallback
            return predict_audio_heuristics(audio_path)
        else:
            raise Exception(f"API error: {error_msg}")

    # Parse API response (if it ever works)
    if isinstance(result, list) and len(result) >= 2:
        fake_prob = None
        real_prob = None

        for item in result:
            if item.get("label") == "LABEL_0":
                fake_prob = item.get("score", 0.0)
            elif item.get("label") == "LABEL_1":
                real_prob = item.get("score", 0.0)

        if fake_prob is None or real_prob is None:
            return predict_audio_heuristics(audio_path)

        prediction = "bonafide" if real_prob > fake_prob else "spoofed"
        confidence = max(real_prob, fake_prob)
        spoof_score = fake_prob

        print("\n" + "=" * 70)
        print("DEEPFAKE DETECTION RESULTS (via API)")
        print("=" * 70)
        print(f"Real probability:   {real_prob:.4f}")
        print(f"Fake probability:   {fake_prob:.4f}")
        print("-" * 70)
        print(f"Prediction:         {prediction.upper()}")
        print(f"Confidence:         {confidence:.1%}")
        print("=" * 70)

        return prediction, confidence, spoof_score
    else:
        return predict_audio_heuristics(audio_path)


def predict_audio_heuristics(audio_path: str) -> Tuple[str, float, float]:
    """
    Fallback: Analyze audio using signal processing heuristics

    This is NOT machine learning-based deepfake detection.
    For production, you need fairseq + Python 3.10 for local inference.

    Args:
        audio_path: Path to audio file

    Returns:
        prediction, confidence, spoof_score
    """
    print("\nAnalyzing audio features...")

    # Load audio using librosa
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    duration = len(audio) / sr

    # Extract audio features
    # These are real audio features, but NOT trained for deepfake detection
    volume_std = np.std(np.abs(audio))
    mean_energy = np.mean(audio ** 2)
    zero_crossing_rate = np.mean(librosa.zero_crossings(audio))

    # Spectral features
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))

    # MFCC features (often used in speech analysis)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfccs)
    mfcc_std = np.std(mfccs)

    # Heuristic scoring (NOT scientifically valid for deepfake detection)
    spoof_score = 0.5  # Start neutral

    # Very basic checks (these are NOT reliable deepfake indicators!)
    if volume_std < 0.05:  # Very consistent volume
        spoof_score += 0.1
    if volume_std > 0.15:  # Very varied volume
        spoof_score -= 0.05
    if zero_crossing_rate > 0.5:
        spoof_score += 0.08
    if spectral_centroid < 1000:  # Very low frequencies
        spoof_score += 0.07
    if mfcc_std < 10:  # Very consistent MFCCs
        spoof_score += 0.1

    # Clamp to valid range
    spoof_score = max(0.0, min(1.0, spoof_score))

    prediction = "spoofed" if spoof_score > 0.5 else "bonafide"
    confidence = abs(spoof_score - 0.5) * 2

    print("\n" + "=" * 70)
    print("AUDIO ANALYSIS RESULTS (HEURISTICS - NOT AI)")
    print("=" * 70)
    print(f"Duration:              {duration:.2f}s")
    print(f"Volume std dev:        {volume_std:.4f}")
    print(f"Mean energy:           {mean_energy:.6f}")
    print(f"Zero crossing rate:    {zero_crossing_rate:.4f}")
    print(f"Spectral centroid:     {spectral_centroid:.2f} Hz")
    print(f"Spectral rolloff:      {spectral_rolloff:.2f} Hz")
    print(f"Spectral bandwidth:    {spectral_bandwidth:.2f} Hz")
    print(f"MFCC mean:             {mfcc_mean:.4f}")
    print(f"MFCC std dev:          {mfcc_std:.4f}")
    print("-" * 70)
    print(f"Prediction:            {prediction.upper()}")
    print(f"Confidence:            {confidence:.1%}")
    print(f"Spoof score:           {spoof_score:.4f}")
    print("=" * 70)
    print("\nWARNING: This uses basic audio features, NOT trained AI.")
    print("For real deepfake detection:")
    print("  - Install fairseq with Python 3.10 for local inference")
    print("  - Or use a commercial deepfake detection API")
    print("=" * 70)

    return prediction, confidence, spoof_score


def predict_audio_local(audio_path: str) -> Tuple[str, float, float]:
    """
    Predict using local model (requires fairseq)

    Args:
        audio_path: Path to audio file

    Returns:
        prediction, confidence, spoof_score
    """
    global _model, _device

    # Load model if not already loaded
    if _model is None:
        download_models()

    print("Loading and preprocessing audio file...")
    wav = load_wav_and_preprocess(audio_path)
    wav = wav.to(_device)

    print("Running inference...")
    with torch.no_grad():
        logits = _model(wav)
        probs = torch.nn.functional.softmax(logits, dim=1)
        probs = probs.cpu().numpy()[0]

    # Model outputs: [fake_prob, real_prob]
    fake_prob = probs[0]
    real_prob = probs[1]

    # Determine prediction
    prediction = "bonafide" if real_prob > fake_prob else "spoofed"
    confidence = max(real_prob, fake_prob)
    spoof_score = fake_prob

    print("\n" + "=" * 70)
    print("DEEPFAKE DETECTION RESULTS (local model)")
    print("=" * 70)
    print(f"Real probability:   {real_prob:.4f}")
    print(f"Fake probability:   {fake_prob:.4f}")
    print("-" * 70)
    print(f"Prediction:         {prediction.upper()}")
    print(f"Confidence:         {confidence:.1%}")
    print("=" * 70)

    return prediction, confidence, spoof_score


def predict_audio(audio_path: str, device: str = None) -> Tuple[str, float, float]:
    """
    Predict if audio is bonafide or spoofed

    Automatically uses:
    - Local inference if fairseq is available
    - API inference if fairseq is not available

    Args:
        audio_path: Path to audio file (WAV or FLAC)
        device: Device to use (cuda/cpu), auto-detected if None (local mode only)

    Returns:
        prediction: "bonafide" or "spoofed"
        confidence: confidence score (0-1)
        score: spoof score (higher = more likely spoofed)
    """
    if FAIRSEQ_AVAILABLE:
        print("Using local inference mode (fairseq available)")
        return predict_audio_local(audio_path)
    else:
        print("Using API inference mode (fairseq not available)")
        return predict_audio_api(audio_path)


# For backwards compatibility
def load_audio(file_path: str, sr: int = 16000) -> np.ndarray:
    """
    Load audio file and return as numpy array (for compatibility)
    """
    # Use librosa - handles all formats, no codec issues
    wav, _ = librosa.load(file_path, sr=sr, mono=True)
    return wav
