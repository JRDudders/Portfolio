# Audio Deepfake Detection Setup

## Python Version Requirement

**Important:** The audio deepfake detection feature requires **Python 3.10 or lower** due to fairseq compatibility limitations.

If you're using Python 3.11+ for other features, you have two options:

### Option 1: Create a Separate Environment (Recommended)

Create a dedicated Python 3.10 environment for audio detection:

```bash
# Create new environment
conda create -n audio-detect python=3.10

# Activate it
conda activate audio-detect

# Install dependencies
conda install -c conda-forge fairseq librosa
pip install -r req.txt
```

### Option 2: Downgrade Your Main Environment

Only do this if you don't need Python 3.11+ features:

```bash
# Create new environment with Python 3.10
conda create -n myenv python=3.10
conda activate myenv
pip install -r req.txt
```

## Installation Steps

1. **Install audio dependencies:**
   ```bash
   # If using conda (recommended for fairseq)
   conda install -c conda-forge fairseq librosa torchaudio tensorboardX

   # Or using pip (may have issues on Python 3.11+)
   pip install librosa torchaudio fairseq tensorboardX
   ```

2. **Download pre-trained models:**
   - The XLSR wav2vec 2.0 model (~1.2GB) will be downloaded automatically on first use
   - Or download manually from: https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt
   - Save to: `models/audio_antispoofing/xlsr2_300m.pt`

3. **Download anti-spoofing model:**
   - Download from: https://drive.google.com/drive/folders/1c4ywztEVlYVijfwbGLl9OEa1SNtFKppB
   - Save to: `models/audio_antispoofing/best_model.pth`

## Testing

Upload FLAC or WAV audio files through the Audio tab in the web UI to detect if they are genuine or AI-generated/spoofed.

## Test Datasets

- **ASVspoof 2019**: https://datashare.is.ed.ac.uk/handle/10283/3336
- **In-the-Wild Audio Deepfake**: https://www.kaggle.com/datasets/abdallamohamed312/in-the-wild-audio-deepfake

## Troubleshooting

### fairseq won't install
- Make sure you're using Python 3.10 or lower
- Try: `conda install -c conda-forge fairseq` instead of pip
- On Windows, you may need Visual Studio C++ Build Tools

### "No module named 'fairseq'" error
- Verify Python version: `python --version` (should be 3.10 or lower)
- Activate the correct environment
- Reinstall fairseq: `conda install -c conda-forge fairseq`

### Model download fails
- Check your internet connection
- Try downloading manually from the Google Drive link above
- Ensure the `models/audio_antispoofing/` directory exists

## Architecture

The audio deepfake detection uses:
- **wav2vec 2.0 XLSR** (300M parameters) for SSL feature extraction
- **AASIST** (Audio Anti-Spoofing using Integrated Spectro-Temporal graph attention networks) backend
- Trained on ASVspoof 2019 LA dataset

Based on: https://github.com/TakHemlata/SSL_Anti-spoofing
