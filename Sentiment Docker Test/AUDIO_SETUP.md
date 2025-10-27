# Audio Deepfake Detection Setup

## Overview

The audio deepfake detection feature uses **wav2vec2-large-anti-deepfake** from HuggingFace:
- Model: [nii-yamagishilab/wav2vec-large-anti-deepfake](https://huggingface.co/nii-yamagishilab/wav2vec-large-anti-deepfake)
- Pre-trained wav2vec 2.0 Large model fine-tuned for anti-spoofing
- Works with Python 3.8+ including Python 3.12
- No manual model downloads required!

## Installation

```bash
# Install audio dependencies
pip install librosa soundfile

# transformers is already in requirements
pip install -r req.txt
```

That's it! The model downloads automatically from HuggingFace on first use (~1.2GB).

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
The first time you use the audio feature, HuggingFace will download the model (~1.2GB). This is normal and only happens once. Subsequent uses will be much faster.

### Audio file format errors
- Ensure your file is FLAC or WAV format
- Sample rate will be automatically resampled to 16kHz

## Model Information

**Model**: nii-yamagishilab/wav2vec2-large-anti-deepfake
**Paper**: "Utterance-level Aggregation For Speaker Recognition In The Wild" (ICASSP 2019)
**Trained on**: Multiple anti-spoofing datasets
**Architecture**: wav2vec 2.0 Large (300M parameters) + classification head

This model is specifically trained to detect:
- Text-to-speech (TTS) synthesis
- Voice conversion
- Audio deepfakes
- Other spoofing attacks

## Performance

The model is trained on ASVspoof and other datasets, achieving strong performance on:
- Logical Access (LA) attacks
- Physical Access (PA) attacks
- Audio deepfake detection

For exact metrics, see the [model card](https://huggingface.co/nii-yamagishilab/wav2vec-large-anti-deepfake).
