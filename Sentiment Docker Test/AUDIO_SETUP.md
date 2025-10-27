# Audio Deepfake Detection Setup

## Overview

This feature uses **wav2vec-large-anti-deepfake-nda**, a production-ready deepfake detection model:
- **Model**: [nii-yamagishilab/wav2vec-large-anti-deepfake-nda](https://huggingface.co/nii-yamagishilab/wav2vec-large-anti-deepfake-nda)
- SSL (Self-Supervised Learning) with wav2vec 2.0 Large (1024-dim)
- Trained on anti-spoofing datasets for real deepfake detection
- Outputs: [fake_probability, real_probability]

## Installation

### Step 1: Install Dependencies

```bash
# Install all dependencies including fairseq
pip install fairseq huggingface_hub librosa soundfile torchaudio

# Or install from requirements
pip install -r req.txt
```

### Step 2: fairseq Compatibility

**Important**: fairseq officially supports Python 3.10. If you're using Python 3.12:

1. **Try installing anyway** - it may work:
   ```bash
   pip install fairseq
   ```

2. **If installation fails**, you have two options:

   **Option A: Use Python 3.10** (Recommended for production)
   ```bash
   # Using pyenv
   pyenv install 3.10.13
   pyenv local 3.10.13
   pip install -r req.txt
   ```

   **Option B: Build fairseq from source** (Advanced)
   ```bash
   pip install git+https://github.com/facebookresearch/fairseq.git
   ```

### Step 3: Model Download

The model (~1.2GB) downloads automatically from HuggingFace on first use. No manual download needed!

## Usage

1. Start the application:
   ```bash
   cd "Sentiment Docker Test"
   python run_local.py
   ```

2. Open browser to `http://localhost:8080`

3. Click the "Audio Deepfake Detection" tab

4. Upload FLAC or WAV audio files

5. Click "Analyze Audio" to detect deepfakes

## How It Works

The model uses a two-stage architecture:

1. **SSL Frontend** (wav2vec 2.0 Large):
   - Extracts rich audio representations
   - 24 transformer layers, 1024-dim embeddings
   - Trained on massive amounts of unlabeled audio

2. **Classification Backend**:
   - Adaptive average pooling over time
   - Linear classifier (1024 → 2 classes)
   - Outputs: [fake_prob, real_prob]

## Model Performance

This model is trained to detect:
- ✅ Text-to-speech (TTS) synthesis
- ✅ Voice conversion
- ✅ AI-generated audio (GPT, Tacotron, etc.)
- ✅ Audio deepfakes
- ✅ Spoofing attacks

Trained on datasets like ASVspoof for production-level accuracy.

## Troubleshooting

### "No module named 'fairseq'"
```bash
pip install fairseq
```

If that fails (Python 3.12 compatibility):
```bash
# Try building from source
pip install git+https://github.com/facebookresearch/fairseq.git

# Or use Python 3.10
pyenv install 3.10.13
pyenv local 3.10.13
pip install fairseq
```

### "No module named 'huggingface_hub'"
```bash
pip install huggingface_hub
```

### Model download is slow
The first time you use the audio feature, HuggingFace will download ~1.2GB. This is normal and only happens once. The model is cached in `~/.cache/huggingface/`.

### CUDA/GPU Issues
The model works on both CPU and GPU. If you have CUDA issues:
```bash
# Force CPU mode by setting environment variable
export CUDA_VISIBLE_DEVICES=""
python run_local.py
```

### Audio file format errors
- Ensure your file is FLAC or WAV format
- Sample rate will be automatically resampled to 16kHz
- Stereo audio will be automatically converted to mono

## Test Datasets

For testing the model:

- **ASVspoof 2019**: https://datashare.is.ed.ac.uk/handle/10283/3336
  - Large dataset of bonafide and spoofed audio
  - Industry standard for anti-spoofing research
  - Multiple attack types (TTS, VC, etc.)

- **In-the-Wild Audio Deepfake**: https://www.kaggle.com/datasets/abdallamohamed312/in-the-wild-audio-deepfake
  - Real-world deepfake examples
  - Good for testing robustness

## Model Details

**Architecture:**
- SSL Model: wav2vec 2.0 Large (fairseq)
  - 24 transformer encoder layers
  - 1024-dim embeddings
  - 16 attention heads
  - 4096-dim FFN

- Classification Head:
  - Adaptive average pooling
  - Linear layer: 1024 → 2
  - Softmax for probabilities

**Training:**
- Frozen SSL weights (pretrained on unlabeled audio)
- Fine-tuned classification head on anti-spoofing data
- Binary classification: bonafide vs. spoofed

## Python Version Compatibility

| Python Version | fairseq Support | Status |
|----------------|-----------------|--------|
| 3.8 | ✅ Official | Supported |
| 3.9 | ✅ Official | Supported |
| 3.10 | ✅ Official | **Recommended** |
| 3.11 | ⚠️ Unofficial | May work |
| 3.12 | ⚠️ Unofficial | May work, or use source install |

If you have compatibility issues, the safest option is Python 3.10.

## Alternative: Heuristics-Only Mode

If you can't install fairseq, there's a fallback mode using basic audio heuristics (not reliable for production). See git history for the heuristics-only version.

## Summary

**Current Status**: ✅ Production-ready deepfake detection
**Model**: wav2vec-large-anti-deepfake-nda from NII Yamagishilab
**Requirements**: fairseq (Python 3.10 recommended), torch, torchaudio
**Performance**: Trained on anti-spoofing datasets, production-level accuracy
