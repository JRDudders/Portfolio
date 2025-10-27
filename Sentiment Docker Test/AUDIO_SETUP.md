# Audio Deepfake Detection Setup

## Overview

The audio deepfake detection feature uses the **HuggingFace Inference API**:
- **No model downloads required** - all processing happens via API calls
- Works with Python 3.8+ including Python 3.12
- Optional: Set `HUGGINGFACE_API_KEY` environment variable for better rate limits
- Get your API key from: https://huggingface.co/settings/tokens

## Installation

```bash
# Install audio dependencies only
pip install librosa soundfile

# No model downloads needed!
```

That's it! The feature uses cloud-based inference, so no large model downloads are required.

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

### "No module named 'librosa'"
```bash
pip install librosa soundfile
```

### API request timeouts
- The API has built-in retry logic with exponential backoff
- If models are loading on HuggingFace servers, it will wait and retry automatically
- Check your internet connection if persistent issues occur

### Audio file format errors
- Ensure your file is FLAC or WAV format
- Sample rate will be automatically resampled to 16kHz

### Rate limiting
- Without an API key, HuggingFace Inference API has rate limits
- Set `HUGGINGFACE_API_KEY` environment variable for higher limits:
  ```bash
  export HUGGINGFACE_API_KEY="your_token_here"
  ```
- Get your token from: https://huggingface.co/settings/tokens

## Current Implementation Status

**IMPORTANT**: This is currently a proof-of-concept implementation using placeholder heuristics.

The current implementation:
- ✅ Successfully loads and processes audio files
- ✅ Connects to HuggingFace Inference API
- ✅ Provides UI integration with three-tab interface
- ⚠️ Uses **placeholder heuristics** for detection (NOT production-ready)

### Why Placeholder Heuristics?

The wav2vec2 base models available via free Inference API are NOT trained for deepfake detection. They're general-purpose audio models designed for speech recognition tasks.

Current heuristics analyze:
- Volume consistency (std deviation)
- Mean energy levels

**These are NOT reliable deepfake indicators** - they're just demonstrating the UI/API pipeline works.

## Production Solutions

For reliable deepfake detection, consider:

1. **HuggingFace Pro Account**
   - Access to specialized anti-spoofing models
   - Higher API rate limits
   - https://huggingface.co/pricing

2. **Commercial Deepfake Detection APIs**
   - Deepgram: https://deepgram.com/
   - AssemblyAI: https://www.assemblyai.com/
   - Purpose-built for audio verification

3. **Self-Hosted Model** (requires Python 3.10)
   - Use fairseq-based models like AASIST
   - Full control but environment management complexity
   - Models: SSL_Anti-spoofing, wav2vec2-large-anti-deepfake

## Test Datasets

For testing once production solution is implemented:
- **ASVspoof 2019**: https://datashare.is.ed.ac.uk/handle/10283/3336
- **In-the-Wild Audio Deepfake**: https://www.kaggle.com/datasets/abdallamohamed312/in-the-wild-audio-deepfake
