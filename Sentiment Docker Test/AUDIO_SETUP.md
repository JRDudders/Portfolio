# Audio Deepfake Detection Setup

## Overview

The audio deepfake detection feature uses:
- **wav2vec 2.0 XLS-R** (300M parameters) from HuggingFace for feature extraction
- **AASIST** backend for anti-spoofing classification
- Works with **Python 3.8+** including Python 3.12

## Installation

### 1. Install Dependencies

```bash
# Install audio processing libraries
pip install librosa soundfile torchaudio

# transformers is already in requirements
pip install -r req.txt
```

That's it! No special Python version needed.

### 2. Model Downloads

The wav2vec 2.0 model will be downloaded automatically from HuggingFace on first use (~1.2GB).

For better accuracy, optionally download the fine-tuned anti-spoofing model:
- Download from: https://drive.google.com/drive/folders/1c4ywztEVlYVijfwbGLl9OEa1SNtFKppB
- Save to: `models/audio_antispoofing/best_model.pth`

The system will work without the fine-tuned model, but accuracy may be lower.

## Usage

1. Start the application:
   ```bash
   cd "Sentiment Docker Test"
   python run_local.py
   ```

2. Open browser to `http://localhost:8080`

3. Click the "Audio Deepfake Detection" tab

4. Upload FLAC or WAV audio files

5. Click "Analyze Audio" to detect if the audio is genuine or AI-generated

## Test Datasets

- **ASVspoof 2019**: https://datashare.is.ed.ac.uk/handle/10283/3336
- **In-the-Wild Audio Deepfake**: https://www.kaggle.com/datasets/abdallamohamed312/in-the-wild-audio-deepfake

## Troubleshooting

### "No module named 'transformers'"
```bash
pip install transformers
```

### "No module named 'librosa'"
```bash
pip install librosa soundfile
```

### Model download is slow
The first time you use the audio feature, HuggingFace will download the wav2vec2 model (~1.2GB). This is normal and only happens once. Subsequent uses will be much faster.

### Audio file format errors
- Ensure your file is FLAC or WAV format
- Sample rate will be automatically resampled to 16kHz

## Architecture

Based on: https://github.com/TakHemlata/SSL_Anti-spoofing

Changes from original:
- Replaced fairseq with HuggingFace transformers for Python 3.12 compatibility
- Automatic model downloading from HuggingFace Hub
- Simplified installation process
