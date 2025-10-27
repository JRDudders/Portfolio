# audio_antispoofing.py
"""
Audio deepfake detection using wav2vec-large-anti-deepfake model.
Model: https://huggingface.co/nii-yamagishilab/wav2vec-large-anti-deepfake-nda

Based on the official inference code from the model card.
"""

import os
from pathlib import Path
from typing import Tuple, Dict

import torch
import torchaudio
import numpy as np

# Check if fairseq is available
try:
    from fairseq.models.wav2vec import Wav2Vec2Model, Wav2Vec2Config
    from huggingface_hub import PyTorchModelHubMixin
    FAIRSEQ_AVAILABLE = True
except ImportError:
    FAIRSEQ_AVAILABLE = False


# Model name on HuggingFace
MODEL_NAME = "nii-yamagishilab/wav2vec-large-anti-deepfake-nda"

# Global model instance (loaded once)
_model = None
_device = None


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
    # Load audio file
    wav, sr = torchaudio.load(wav_path)

    # Convert to mono if stereo
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0)
    else:
        wav = wav.squeeze(0)

    # Resample to target sampling rate if needed
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, new_freq=target_sr)

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
        "torchaudio": True
    }


def download_models():
    """
    Download and initialize the deepfake detection model
    """
    global _model, _device

    if not FAIRSEQ_AVAILABLE:
        raise ImportError(
            "fairseq is required for audio deepfake detection.\n"
            "Install with: pip install fairseq\n\n"
            "Note: fairseq may require Python 3.10. If you have Python 3.12:\n"
            "  1. Try installing anyway (pip install fairseq)\n"
            "  2. If it fails, you'll need to use Python 3.10 or use the API-based approach"
        )

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
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


def predict_audio(audio_path: str, device: str = None) -> Tuple[str, float, float]:
    """
    Predict if audio is bonafide or spoofed using trained deepfake detection model

    Args:
        audio_path: Path to audio file (WAV or FLAC)
        device: Device to use (cuda/cpu), auto-detected if None

    Returns:
        prediction: "bonafide" or "spoofed"
        confidence: confidence score (0-1)
        score: spoof score (higher = more likely spoofed)
    """
    global _model, _device

    if not FAIRSEQ_AVAILABLE:
        raise ImportError(
            "fairseq is required. Install with: pip install fairseq\n"
            "Note: fairseq may require Python 3.10"
        )

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
    spoof_score = fake_prob  # Use fake probability as spoof score

    print("\n" + "=" * 70)
    print("DEEPFAKE DETECTION RESULTS")
    print("=" * 70)
    print(f"Real probability:   {real_prob:.4f}")
    print(f"Fake probability:   {fake_prob:.4f}")
    print("-" * 70)
    print(f"Prediction:         {prediction.upper()}")
    print(f"Confidence:         {confidence:.1%}")
    print("=" * 70)

    return prediction, confidence, spoof_score


# For backwards compatibility with old code
def load_audio(file_path: str, sr: int = 16000) -> np.ndarray:
    """
    Load audio file and return as numpy array (for compatibility)
    """
    wav, _ = torchaudio.load(file_path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0)
    else:
        wav = wav.squeeze(0)
    if _ != sr:
        wav = torchaudio.functional.resample(wav, _, new_freq=sr)
    return wav.numpy()
