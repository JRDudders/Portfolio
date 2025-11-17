# Portfolio - JRDudders

This portfolio showcases NLP coursework and practical solutions I've developed, including a full-featured web application for NLP, graph analytics, and audio deepfake detection.

---

## 🎯 Featured Project: CiceroWatch

**CiceroWatch** is a comprehensive web application combining three powerful analysis tools with production-ready capabilities.

### Features

#### 1. NLP & Text Analysis
- **Sentiment Analysis**: Twitter (RoBERTa), SST-2 (DistilBERT)
- **Zero-Shot Classification**: Custom categories without training (BART-MNLI, mDeBERTa)
- **Named Entity Recognition (NER)**: Extract entities using HuggingFace, spaCy, or Stanza
- **Topic Modeling**: Discover topics using BERTopic, NMF, or K-means clustering
- **Text Embeddings**: Generate sentence embeddings using Sentence-BERT
- **POS Tagging & Dependency Parsing**: Linguistic analysis with spaCy and Stanza
- **Stance Detection**: NLI-based stance detection with claim support

#### 2. Graph Analytics
- **Network Analysis**: PageRank, betweenness, eigenvector centrality
- **Graph Algorithms**: BFS, triangle counting, degree analysis
- **Interactive Visualization**: Explore networks with vis.js
- **Ego Networks**: Analyze social network circles and features
- **GPU Acceleration**: Optional RAPIDS cuGraph support for large graphs (10-100x speedup)

#### 3. Audio Deepfake Detection
- **AI-Generated Audio Detection**: Identify spoofed or AI-generated audio using wav2vec 2.0
- **SSL Anti-Spoofing**: State-of-the-art model combining wav2vec 2.0 XLS-R (300M) with AASIST backend
- **Model**: [nii-yamagishilab/wav2vec-large-anti-deepfake-nda](https://huggingface.co/nii-yamagishilab/wav2vec-large-anti-deepfake-nda)
- **Supported Formats**: FLAC and WAV audio files
- **Python 3.12 Compatible**: Uses HuggingFace transformers (no fairseq required)
- **Two Modes**: API mode (default, any Python version) or Local mode (Python 3.10, faster)

---

## 🚀 Quick Start - Local Installation

### Prerequisites

- Python 3.10-3.12
- pip package manager
- (Optional) NVIDIA GPU with CUDA 12.1 for GPU acceleration

### Installation

**Option 1: Automated Setup (Linux/Mac)**
```bash
cd "Sentiment Docker Test"

# For CPU-only
bash setup_local.sh cpu

# For CUDA 12.1 GPU
bash setup_local.sh cuda121

# For CUDA 11.8 GPU
bash setup_local.sh cuda118
```

**Option 2: Manual Setup**
```bash
cd "Sentiment Docker Test"

# Step 1: Install PyTorch (REQUIRED - do this first!)
# For CPU:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# For CUDA 12.1 GPU:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8 GPU:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Step 2: Install other dependencies
pip install -r requirements-local.txt

# Step 3: (Optional) Download spaCy model
python -m spacy download en_core_web_sm

# Step 4: Run the server
python run_local.py
```

### Start the Application

```bash
python run_local.py
```

Open your browser to: **http://localhost:8080**

The application will automatically:
- Detect GPU availability (CUDA)
- Use cuGraph if available (GPU graph acceleration)
- Fall back to CPU-based processing if needed

---

## 📦 GPU Acceleration (Optional)

### Graph Analytics GPU Support

For massive speedups on large graphs (10-100x faster):

**Requirements:**
- NVIDIA GPU with CUDA support
- Python 3.10-3.12
- CUDA 11.8 or 12.1 installed

**Installation:**
```bash
# For CUDA 12.x:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install cudf-cu12 cugraph-cu12 --extra-index-url=https://pypi.nvidia.com

# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install cudf-cu11 cugraph-cu11 --extra-index-url=https://pypi.nvidia.com
```

**Verify GPU Support:**
```bash
# Check NVIDIA drivers
nvidia-smi

# Run the app and check startup logs
python run_local.py
```

### Performance Comparison

| Dataset Size | CPU (NetworkX) | GPU (cuGraph) | Speedup |
|--------------|----------------|---------------|---------|
| 1K nodes     | 1 sec          | 0.1 sec       | 10x     |
| 10K nodes    | 30 sec         | 0.5 sec       | 60x     |
| 100K nodes   | 20 min         | 10 sec        | **120x** |

---

## 🎤 Audio Deepfake Detection

The audio detection feature uses a real, trained deepfake detection model (not heuristics).

### Two Operating Modes

#### 1. API Mode (Default - Recommended)
- ✅ Works with any Python version (3.8+)
- ✅ No large model downloads
- ✅ Same trained model via HuggingFace API
- ✅ Easy setup
- ⚠️ Requires internet connection
- ⚠️ Rate limits: 60 requests/hour (free)

#### 2. Local Mode (Optional - For Performance)
- ✅ Faster inference (no network latency)
- ✅ Works offline
- ✅ No rate limits
- ⚠️ Requires fairseq (Python 3.10 recommended)
- ⚠️ Large model download (~1.2GB first time)

### Setup for Local Mode

If you want local inference for best performance:

```bash
# Option A: Try with your current Python version
pip install fairseq huggingface_hub

# Option B: Use Python 3.10 (safest for fairseq)
pyenv install 3.10.13
pyenv local 3.10.13
pip install fairseq huggingface_hub

# Option C: Build fairseq from source
pip install git+https://github.com/facebookresearch/fairseq.git
```

**Note:** If fairseq installation fails, the system automatically falls back to API mode.

### Optional: HuggingFace API Key

To remove rate limits in API mode:

1. Get free token: https://huggingface.co/settings/tokens
2. Set environment variable:
   ```bash
   export HUGGINGFACE_API_KEY="hf_..."
   # Or add to .env file
   echo "HUGGINGFACE_API_KEY=hf_..." >> .env
   ```

### Model Information

- **Model**: wav2vec2-large-anti-deepfake-nda (NII Yamagishi Lab)
- **Architecture**: wav2vec 2.0 Large (24 layers, 1024-dim)
- **Training**: ASVspoof and other anti-spoofing datasets
- **Detects**: TTS synthesis, voice conversion, AI-generated audio, deepfakes

---

## 📋 Dependencies

### Core Requirements
- Python 3.10-3.12
- FastAPI >= 0.111.0
- Uvicorn >= 0.29.0
- PyTorch (see installation instructions above)
- Transformers >= 4.40.0

### NLP Features
- spacy >= 3.7.0
- nltk >= 3.8.0
- sentence-transformers
- scikit-learn >= 1.3.0

### Graph Analytics
- networkx >= 3.0 (CPU)
- cudf, cugraph (optional, GPU)

### Audio Detection
- librosa >= 0.10.0
- soundfile >= 0.12.1
- huggingface_hub >= 0.20.0
- fairseq >= 0.12.0 (optional, for local mode)

### Web Scraping
- beautifulsoup4 >= 4.12.0
- trafilatura >= 1.6.0
- requests >= 2.31.0

Full dependency lists available in:
- `requirements-local.txt` - Minimal dependencies for local dev
- `req.txt` - Full feature set

---

## 🛠️ Troubleshooting

### PyTorch Installation Error

**Error:** `No matching distribution found for torch`

**Solution:** PyTorch must be installed from pytorch.org index URLs, not PyPI:
```bash
# Install PyTorch FIRST, before other requirements
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Port Already in Use

```bash
# Find process using port 8080
lsof -i :8080  # Linux/Mac
netstat -ano | findstr :8080  # Windows

# Kill the process or change port in run_local.py
```

### GPU Not Detected

```bash
# Check NVIDIA drivers
nvidia-smi

# Reinstall PyTorch with CUDA support
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Audio Detection Not Working

The system automatically handles this:
- If fairseq is available → Uses local mode
- If fairseq is not available → Uses API mode (requires internet)

Both modes use the same trained model and produce identical results.

### "fairseq not available - will use API inference mode"

This is **normal**! Your audio detection will work via the HuggingFace API. No action needed unless you want faster local inference.

---

## 📁 Project Structure

```
Portfolio/
├── README.md                          # This file
├── Sentiment Docker Test/             # CiceroWatch application
│   ├── app.py                         # Main FastAPI application
│   ├── run_local.py                   # Local development server
│   ├── setup_local.sh                 # Automated setup script
│   ├── requirements-local.txt         # Minimal dependencies
│   ├── req.txt                        # Full dependencies
│   ├── requirements-gpu.txt           # GPU acceleration
│   ├── nlp.py                         # NLP core (presets, models)
│   ├── nlp_processor.py               # NLP business logic
│   ├── topics.py                      # Topic modeling
│   ├── graph_processor.py             # Graph analytics
│   ├── audio_antispoofing.py          # Audio deepfake detection
│   └── index.html                     # Frontend UI
├── naive-bayes-JRDudders/             # Naive Bayes classifier
├── Dudley_Viterbi_Tagger.py           # HMM POS tagger
├── Dudley_ngrams.py                   # N-gram language models
└── Dudley_POS_tags.py                 # POS tagging utilities
```

---

## 🎓 Academic Projects

This portfolio also includes coursework from my NLP studies:

- **Naive Bayes**: Multinomial Naive Bayes document classifier (Advanced NLP)
- **N-Grams, POS Tagger, Viterbi Tagger**: Modified assignments from Intro to NLP
- **Folder Walker, Extension Finder, Pandas Practice, Folder Mapper**: Practical solutions from previous positions

---

## 📚 Learning & Development

### Current Reading List

- *Hands-On Machine Learning with Scikit-Learn, Keras, and Tensorflow* - Géron
- *Python Machine Learning* - Raschka & Mirjalili
- *Statistics for Linguists: an Introduction Using R* - Winter
- *Machine Learning for Algorithmic Trading* - Jansen (includes NLP chapter)

### Useful Resources

**BERT & Transformers:**
- https://jalammar.github.io/illustrated-transformer/
- https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270

**General NLP:**
- https://www.tensorflow.org/tutorials/text/word_embeddings
- https://github.com/mhagiwara/100-nlp-papers
- https://rare-technologies.com/word2vec-tutorial/
- https://web.stanford.edu/~jurafsky/slp3/
- https://github.com/flairNLP/flair

---

## 🔧 API Reference

### Health Check
```bash
curl http://localhost:8080/health
```

### Sentiment Analysis
```bash
curl -X POST http://localhost:8080/analyze \
  -F "file=@tweets.csv" \
  -F "task=sentiment"
```

### Named Entity Recognition
```bash
curl -X POST http://localhost:8080/analyze \
  -F "file=@article.txt" \
  -F "task=ner"
```

### Topic Modeling
```bash
curl -X POST http://localhost:8080/analyze \
  -F "file=@documents.json" \
  -F "task=topics"
```

### Graph Analysis
```bash
curl -X POST http://localhost:8080/graph/pagerank \
  -F "file=@network.csv"
```

### Audio Deepfake Detection
```bash
curl -X POST http://localhost:8080/audio/analyze \
  -F "file=@recording.wav"
```

---

## 📝 File Formats

### NLP Input

**JSON (list of strings):**
```json
["First text", "Second text", "Third text"]
```

**JSON (list of objects):**
```json
[{"text": "First text"}, {"text": "Second text"}]
```

**CSV (with 'text' column):**
```csv
text
"First text"
"Second text"
```

### Graph Input

**CSV edge list:**
```csv
src,dst
Alice,Bob
Bob,Charlie
```

### Audio Input

- **Formats**: WAV, FLAC
- **Sample rate**: Automatically resampled to 16kHz
- **Channels**: Automatically converted to mono

---

## 🤝 Contributing

This is a personal portfolio project, but suggestions and feedback are welcome!

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🙏 Credits

- **NLP Models**: HuggingFace Transformers
- **Graph Analytics**: NVIDIA RAPIDS, NetworkX
- **Audio Detection**: NII Yamagishi Lab wav2vec2-large-anti-deepfake
- **Frontend**: Vis.js for network visualization

---

## 💬 Contact

For questions or collaboration opportunities, feel free to reach out via GitHub.

**Happy analyzing! 🚀**
