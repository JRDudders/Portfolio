# CiceroWatch - Multi-Service NLP & Analytics Platform

Production-ready microservices platform for NLP analysis, GPU-accelerated graph analytics, and audio deepfake detection.

## Architecture

**Microservices Design:**
- **Frontend** (nginx) - Serves UI and reverse proxy
- **NLP Service** (Python 3.12 + GPU) - Sentiment, entities, topics, classification
- **Graph Service** (Python 3.11 + RAPIDS) - Network analysis with GPU acceleration
- **Audio Service** (Python 3.10 + fairseq) - Audio deepfake detection

All services communicate via REST APIs and support live code editing without rebuilds.

---

## Features

### NLP & Text Analysis
- ✅ **8 Production Presets**:
  - `sentiment-twitter` - Twitter sentiment (RoBERTa)
  - `sentiment-sst2` - SST-2 sentiment (DistilBERT)
  - `zeroshot-bart` - Zero-shot classification (BART-MNLI)
  - `zeroshot-mdeberta` - Multilingual zero-shot (mDeBERTa)
  - `ner-conll` - Named entity recognition (BERT-NER)
  - `ner-bertbase` - Named entity recognition (BERT-base)
  - `topics-nmf` - Topic modeling (NMF with TF-IDF)
  - `topics-kmeans` - Topic clustering (K-means)

- 📄 **Smart File Processing**:
  - **JSON**: `["text1", "text2"]` or `[{"text": "text1"}]`
  - **CSV**: with `text` column
  - **HTML**: Automatic chunking for books/long articles
    - Topic modeling: chunks into paragraphs (finds themes across chapters)
    - Other tasks: treats as single document

### Graph Analytics
- ⚡ **GPU Acceleration**: RAPIDS cuGraph (10-100x faster than CPU)
- 📊 **Algorithms**: PageRank, Betweenness, Eigenvector, Degree Centrality, BFS, Triangle Counting
- 🎨 **Interactive Visualization**: Vis.js with ego network support
- 📁 **Formats**: SNAP ego networks (.edges, .circles, .feat, .featnames)
- 💻 **Cross-Platform**: NetworkX CPU fallback

### Audio Deepfake Detection
- 🎤 **Model**: wav2vec2-large-anti-deepfake (NII Yamagishi Lab)
- 🔍 **Detection**: Identifies AI-generated voice (deepfakes)
- 📊 **Output**: Prediction (bonafide/spoofed), confidence, spoof score
- 🎧 **Formats**: WAV, FLAC, MP3

---

## Quick Start

### Production Deployment (Stable Build)

**Frozen code, no live editing - recommended for production use:**

```powershell
# GPU Version
docker-compose -f docker-compose.prod.gpu.yml up --build -d

# CPU Version
docker-compose -f docker-compose.prod.yml up --build -d

# Access at http://localhost:80
```

**Production features:**
- ✅ Code baked into images (frozen and stable)
- ✅ Auto-restart on failure (`restart: always`)
- ✅ Health checks enabled
- ✅ Named volumes for persistence
- ✅ Resource limits configured
- ❌ No live editing (requires rebuild for code changes)

### Development Setup (Live Editing)

**Code mounted for instant changes - recommended for development:**

```powershell
# GPU Version (Requires NVIDIA GPU)
# Prerequisites: Docker Desktop + NVIDIA Container Toolkit
# See "GPU Setup" section below

cd "Sentiment Docker Test"
docker-compose -f docker-compose.gpu.yml up --build

# CPU Version (Any Platform)
docker-compose up --build

# Access at http://localhost:80
```

**Development Workflow** (No Rebuilds!):
```bash
# 1. Edit Python files (service_nlp.py, graph_processor.py, etc.)
# 2. Restart service (takes 2 seconds)
docker-compose -f docker-compose.gpu.yml restart nlp
# 3. Changes are live!

# Edit HTML
docker-compose -f docker-compose.gpu.yml restart frontend
```

**Only rebuild when:**
- Changing `Dockerfile`
- Changing `requirements-*.txt`
- Installing new system packages

---

## GPU Setup (Windows/WSL2)

### Prerequisites
- NVIDIA GPU (RTX 2000+ series)
- Docker Desktop with WSL2 enabled
- NVIDIA drivers (latest)

### Installation

```powershell
# 1. Install WSL2 with Ubuntu 24.04
wsl --install -d Ubuntu-24.04

# 2. Install NVIDIA Container Toolkit in WSL2
wsl
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
exit

# 3. Test GPU access
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

**Expected output:** Should show your GPU info!

### Performance Comparison

| Dataset | CPU (NetworkX) | GPU (cuGraph) | Speedup |
|---------|----------------|---------------|---------|
| 1K nodes | 1 sec | 0.1 sec | 10x |
| 10K nodes | 30 sec | 0.5 sec | 60x |
| 100K nodes | 20 min | 10 sec | **120x** |

---

## API Reference

### NLP Service (Port 8001)

**File Analysis:**
```bash
# Sentiment analysis on CSV
curl -X POST http://localhost:80/api/nlp/analyze/file \
  -F "file=@tweets.csv" \
  -F "preset=sentiment-twitter"

# Topic modeling on book (HTML)
curl -X POST http://localhost:80/api/nlp/analyze/file \
  -F "file=@pride-and-prejudice.html" \
  -F "preset=topics-nmf"

# Named entity recognition
curl -X POST http://localhost:80/api/nlp/analyze/file \
  -F "file=@article.json" \
  -F "preset=ner-conll"
```

**URL Analysis:**
```bash
curl -X POST http://localhost:80/api/nlp/analyze/url \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/article",
    "preset": "sentiment-sst2"
  }'
```

**Zero-Shot Classification:**
```bash
curl -X POST http://localhost:80/api/nlp/analyze/url \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://news.example.com",
    "preset": "zeroshot-bart",
    "labels": ["politics", "technology", "sports", "health"]
  }'
```

### Graph Service (Port 8002)

**Ego Network Visualization:**
```bash
# Upload SNAP ego network (80 files: .edges, .circles, .feat, etc.)
curl -X POST http://localhost:80/api/graph/ego-network \
  -F "files=@0.edges" \
  -F "files=@0.circles" \
  -F "files=@0.feat" \
  -F "files=@0.featnames" \
  -F "ego_id=0"
```

**Graph Metrics:**
```bash
# PageRank
curl -X POST http://localhost:80/api/graph/pagerank \
  -F "edges_file=@edges.csv" \
  -F "alpha=0.85"

# Betweenness Centrality
curl -X POST http://localhost:80/api/graph/betweenness \
  -F "edges_file=@edges.csv"

# Triangle Counting
curl -X POST http://localhost:80/api/graph/triangles \
  -F "edges_file=@edges.csv"
```

### Audio Service (Port 8003)

**Deepfake Detection:**
```bash
# Analyze audio file
curl -X POST http://localhost:80/api/audio/analyze \
  -F "file=@recording.wav"

# Response:
{
  "success": true,
  "prediction": "bonafide",  # or "spoofed"
  "confidence": 0.92,
  "spoof_score": 0.08,  # higher = more likely fake
  "filename": "recording.wav"
}
```

**Model Status:**
```bash
curl http://localhost:80/api/audio/model-status
```

---

## File Formats

### NLP Input Formats

**JSON - List of strings:**
```json
["First tweet text", "Second tweet text", "Third tweet text"]
```

**JSON - List of objects:**
```json
[
  {"text": "First tweet text"},
  {"text": "Second tweet text"}
]
```

**CSV - With 'text' column:**
```csv
text
"First tweet text"
"Second tweet text"
```

**HTML - Books/Articles:**
```html
<!-- Automatically chunked for topic modeling -->
<div class="chapter">
  <h1>Chapter 1</h1>
  <p>First paragraph...</p>
  <p>Second paragraph...</p>
</div>
```

### Graph Input Formats

**Edges CSV:**
```csv
src,dst
Alice,Bob
Bob,Charlie
Charlie,Alice
```

**SNAP Ego Network:**
```
0.edges       # Edge list: node1 node2
0.circles     # Social circles: circle_name node1 node2 ...
0.feat        # Node features: binary feature vector
0.featnames   # Feature names: feature_id name anonymized_status
```

**Nodes CSV (optional):**
```csv
id,name,group
Alice,Alice Smith,1
Bob,Bob Jones,2
```

---

## Configuration

### NLP Presets

Modify `nlp.py` to add custom presets:

```python
PRESETS: Dict[str, Tuple[str, Optional[str], Dict[str, Any]]] = {
    "your-preset": ("text-classification", "org/model-name", {}),
}
```

### Topic Modeling Parameters

**In `topics.py`:**
- `min_df: int = 2` - Minimum document frequency (increase for noise reduction)
- `n_topics: int = 10` - Number of topics to extract
- `stop_words='english'` - Filters common stopwords
- `ngram_range: Tuple = (1, 2)` - Unigrams and bigrams

### Graph Processing

**Volume Mounts:**
All Python files are volume-mounted for live editing:
- `service_graph.py` - API endpoints
- `graph_processor.py` - Business logic
- `graph_tasks.py` - Core algorithms

**GPU Detection:**
- Automatically uses cuGraph if GPU available
- Falls back to NetworkX on CPU
- Check logs for: `"Using cuGraph (GPU)"` or `"Using NetworkX (CPU)"`

---

## Technology Stack

### NLP Service
- **Framework**: FastAPI + Uvicorn
- **NLP**: Transformers 4.40-4.47, PyTorch 2.5.1
- **Models**: HuggingFace Hub (RoBERTa, BERT, BART, mDeBERTa)
- **Processing**: BeautifulSoup4, Trafilatura (HTML extraction)
- **Topic Modeling**: scikit-learn (NMF, K-means, TF-IDF)

### Graph Service
- **GPU**: RAPIDS cuGraph 24.08a (CUDA 12.5, Python 3.11, Ubuntu 24.04)
- **CPU Fallback**: NetworkX
- **Format**: Pandas DataFrames, cuDF (GPU)
- **Visualization**: Vis.js network graphs

### Audio Service
- **Model**: wav2vec2-large-anti-deepfake (fairseq)
- **Processing**: librosa, soundfile
- **Backend**: PyTorch 2.5.1 (CPU)

### Frontend
- **Server**: nginx (Alpine)
- **Routing**: Reverse proxy to backend services
- **Limits**: 500MB upload, 120s timeout

---

## Development

### Project Structure
```
Sentiment Docker Test/
├── docker-compose.yml              # DEV: CPU services (live editing)
├── docker-compose.gpu.yml          # DEV: GPU services (live editing)
├── docker-compose.prod.yml         # PROD: CPU services (frozen code)
├── docker-compose.prod.gpu.yml     # PROD: GPU services (frozen code)
├── Dockerfile.frontend             # nginx frontend
├── Dockerfile.nlp.gpu              # NLP service (GPU)
├── Dockerfile.graph.gpu            # Graph service (RAPIDS)
├── Dockerfile.audio                # Audio service
├── service_nlp.py                  # NLP API endpoints
├── service_graph.py            # Graph API endpoints
├── audio_service.py            # Audio API endpoints
├── nlp.py                      # NLP core (presets, models)
├── nlp_processor.py            # NLP business logic
├── graph_processor.py          # Graph business logic
├── graph_tasks.py              # Graph algorithms (GPU/CPU)
├── audio_antispoofing.py       # Audio deepfake detection
├── topics.py                   # Topic modeling (NMF, K-means)
├── index.html                  # Frontend UI
├── nginx.conf                  # nginx configuration
└── requirements-*.txt          # Python dependencies
```

### Adding New Features

**New NLP Preset:**
1. Add to `nlp.py` PRESETS dict
2. Add to `index.html` PRESETS array
3. Restart NLP service (no rebuild!)

**New Graph Algorithm:**
1. Add function to `graph_tasks.py` (GPU + CPU versions)
2. Add endpoint to `service_graph.py`
3. Add wrapper to `graph_processor.py`
4. Restart graph service (no rebuild!)

### Debugging

**View logs:**
```bash
docker-compose -f docker-compose.gpu.yml logs -f nlp
docker-compose -f docker-compose.gpu.yml logs -f graph
docker-compose -f docker-compose.gpu.yml logs -f audio
```

**Check service health:**
```bash
curl http://localhost:80/api/nlp/health
curl http://localhost:80/api/graph/health
curl http://localhost:80/api/audio/health
```

**Restart individual service:**
```bash
docker-compose -f docker-compose.gpu.yml restart nlp
```

---

## Troubleshooting

### GPU Not Detected

```bash
# Check NVIDIA drivers
nvidia-smi

# Check Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi

# Check RAPIDS container
docker-compose -f docker-compose.gpu.yml logs graph | grep -i gpu
```

### Port Already in Use

```bash
# Find process using port 80
sudo lsof -i :80  # Linux/Mac
netstat -ano | findstr :80  # Windows

# Change port in docker-compose.gpu.yml:
ports:
  - "8080:80"  # Use 8080 instead
```

### Out of Memory

```bash
# Reduce batch size in topics.py:
max_features: int = 5000  # Down from 20000

# Limit Docker memory in docker-compose.gpu.yml:
services:
  nlp:
    deploy:
      resources:
        limits:
          memory: 4G  # Down from 8G
```

### Topic Modeling Errors

**"No terms remain after filtering"**
- Dataset too small or repetitive
- Need at least 2 documents with meaningful content
- Reduce `min_df` in `topics.py` for small datasets

**"Requires at least 2 documents"**
- For HTML books: Use topic presets (auto-chunks)
- For other formats: Upload multiple documents

---

## Known Limitations

1. **Topic Modeling**: Requires ≥2 documents (or HTML chunking)
2. **GPU Memory**: Large graphs (>1M nodes) may need ≥16GB VRAM
3. **Audio Model**: Downloaded on first use (~2GB, takes 2-3 minutes)
4. **PyTorch Version**: Currently 2.5.1 (2.6 not yet released, required by transformers ≥4.48)

---

## Version History

### Current Version
- **Architecture**: Microservices (NLP, Graph, Audio)
- **PyTorch**: 2.5.1
- **Transformers**: 4.40-4.47
- **RAPIDS**: 24.08a (CUDA 12.5, Ubuntu 24.04)
- **NLP Presets**: 8 working presets
- **File Support**: JSON, CSV, HTML (with smart chunking)
- **Live Editing**: Volume mounts for all Python files

### Recent Changes
- ✅ Fixed PyTorch compatibility (2.5.1 for transformers <4.48)
- ✅ Added HTML chunking for book topic modeling
- ✅ Added stopword filtering to topic modeling
- ✅ Fixed all NLP preset configurations
- ✅ Added audio deepfake detection service
- ✅ Implemented volume mounts for live editing
- ✅ Fixed nginx upload limits (500MB)
- ✅ Added parameter validation for topic modeling
- ✅ Removed 7 broken presets (missing dependencies)

---

## License

MIT License - See LICENSE file for details

## Credits

- **NLP Models**: HuggingFace Transformers
- **Graph Analytics**: NVIDIA RAPIDS, NetworkX
- **Audio Detection**: NII Yamagishi Lab wav2vec2-large-anti-deepfake
- **Frontend**: Vis.js, nginx

---

## Support

For issues or questions:
1. Check logs: `docker-compose logs -f <service>`
2. Verify health: `curl http://localhost:80/api/<service>/health`
3. Restart service: `docker-compose restart <service>`
4. Rebuild if needed: `docker-compose up --build <service>`

**Happy analyzing! 🚀**
