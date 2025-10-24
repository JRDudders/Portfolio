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

**IMPORTANT**: The fine-tuned anti-spoofing model is **REQUIRED**. Without it, the detector will not work.

#### Option 1: Using the helper script

```bash
cd "Sentiment Docker Test"
python download_audio_model.py
```

This will show you exactly where to download the model and where to save it.

#### Option 2: Manual download

1. **Go to Google Drive**: https://drive.google.com/drive/folders/1c4ywztEVlYVijfwbGLl9OEa1SNtFKppB

2. **Download** one of these files:
   - `best_SSL_model_LA.pth` (recommended)
   - Any `.pth` file for the LA (Logical Access) track

3. **Save it to**:
   ```
   Sentiment Docker Test/models/audio_antispoofing/best_model.pth
   ```

4. **Create the directory** if it doesn't exist:
   ```bash
   mkdir -p "Sentiment Docker Test/models/audio_antispoofing"
   ```

5. **Verify the file** is in the right place:
   ```bash
   ls -lh "Sentiment Docker Test/models/audio_antispoofing/best_model.pth"
   ```

The wav2vec 2.0 base model (~1.2GB) will be downloaded automatically from HuggingFace on first use.

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
