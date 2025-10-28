# CiceroWatch NLP Tasks

Production-ready FastAPI service for sentiment analysis, NLP tasks, and GPU-accelerated graph analytics.

## Features

### NLP & Sentiment Analysis
- **Sentiment Analysis**: Twitter-RoBERTa, SST-2, and custom models
- **Zero-Shot Classification**: Classify text with custom labels
- **Named Entity Recognition**: Extract people, organizations, locations
- **Topic Modeling**: BERTopic, NMF, K-means clustering
- **Embeddings**: Sentence-BERT for semantic similarity
- **Multi-format**: JSON, CSV, HTML, URLs with smart preprocessing

### Graph Analytics
- **GPU Acceleration**: 10-100x faster with NVIDIA GPUs (RAPIDS cuGraph)
- **Centrality Measures**: PageRank, Betweenness, Eigenvector, Degree
- **Graph Algorithms**: BFS, triangle counting, community detection
- **Interactive Visualization**: Vis.js network graphs with hover labels
- **Cross-Platform**: NetworkX CPU fallback (Windows/Mac/Linux compatible)
- **Formats**: SNAP ego networks (.edges, .circles, .feat, .node)

### Web Processing
- **URL Scraping**: BeautifulSoup4, Trafilatura, Readability
- **JavaScript Rendering**: Playwright & Selenium for SPAs
- **Batch Processing**: Multi-text and multi-URL analysis

---

## Quick Start

### WSL2 Native (Recommended - Fastest)

**Best for:** Active development with GPU acceleration on Windows

```bash
# 1. Install WSL2 with Ubuntu 24.04 (matches RAPIDS CUDA 12.5+ images)
wsl --install -d Ubuntu-24.04
wsl

# 2. Navigate to project
cd /mnt/c/Users/YourUser/Path/To/Sentiment\ Docker\ Test

# 3. Setup (one-time, ~10 minutes)
bash setup_wsl2_gpu.sh

# 4. Run
conda activate base
python run_local.py
```

**Access:** http://localhost:8080

**Development workflow:**
1. Edit `.py` files in Windows (VS Code)
2. Save
3. `Ctrl+C` in WSL2 terminal
4. `↑` + `Enter` to restart (2 seconds)
5. Changes live!

### Docker GPU (Windows)

**Best for:** Production-like environment with GPU

```powershell
# 1. Install Docker Desktop + NVIDIA Container Toolkit
# See "GPU Setup" section below

# 2. Build (one-time, ~10 minutes)
docker build -f Dockerfile.gpu -t sentiment-gpu .

# 3. Run
.\run_docker_gpu.ps1

# Or manually:
docker run --rm --gpus all -p 8080:8080 -v "${PWD}:/app" sentiment-gpu
```

**Code changes:** Edit `.py` files → Changes apply immediately (no rebuild!)

### Local CPU (Any Platform)

**Best for:** Quick testing without GPU

```bash
# Install dependencies
pip install -r req.txt
python -m spacy download en_core_web_sm

# Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# Run
python run_local.py
```

**Access:** http://localhost:8080

---

## GPU Setup

### Windows (Docker + WSL2)

**Prerequisites:**
- NVIDIA GPU (RTX series recommended)
- Docker Desktop with WSL2 enabled
- NVIDIA drivers installed

**Setup:**

```powershell
# 1. Enable WSL2 in Docker Desktop
# Settings → General → Enable "Use the WSL 2 based engine"

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

**Expected output:** Should show your GPU!

### Linux

```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Run with GPU
./run_docker_gpu.sh
```

### Performance

| Graph Size | CPU (NetworkX) | GPU (cuGraph) | Speedup |
|------------|----------------|---------------|---------|
| 1K nodes | 1 sec | 0.1 sec | 10x |
| 10K nodes | 30 sec | 0.5 sec | 60x |
| 100K nodes | 20 min | 10 sec | **120x** |

---

## API Usage

### Sentiment Analysis

**File Upload:**
```bash
curl -X POST http://localhost:8080/process/file \
  -F "file=@tweets.csv" \
  -F "preset=sentiment-twitter"
```

**URL Analysis:**
```bash
curl -X POST "http://localhost:8080/process/url?url=https://example.com&preset=sentiment-sst2"
```

**Zero-Shot Classification:**
```bash
curl -X POST "http://localhost:8080/process/url" \
  -F "url=https://news.site" \
  -F "preset=zeroshot-bart" \
  -F "labels=politics,technology,sports"
```

### Graph Analytics

**Load and Visualize:**
```bash
curl -X POST http://localhost:8080/graph/load \
  -F "edges_file=@edges.csv" \
  -F "nodes_file=@nodes.csv"
```

Response includes:
- Node/edge counts
- Centrality measures (PageRank, Betweenness, Eigenvector, Degree)
- Visualization data (nodes/edges for vis.js)

**PageRank:**
```bash
curl -X POST http://localhost:8080/graph/pagerank \
  -F "edges_file=@edges.csv" \
  -F "alpha=0.85"
```

**Ego Networks:**
```bash
curl -X POST http://localhost:8080/graph/ego \
  -F "files=@user.edges" \
  -F "files=@user.circles" \
  -F "files=@user.feat"
```

---

## Interactive UI

Open http://localhost:8080 in your browser for the web interface.

### Features

**File Upload Section:**
- Upload JSON/CSV/HTML files
- Select NLP preset (sentiment, NER, topics, embeddings)
- Custom zero-shot labels
- Download results

**URL Processing:**
- Fetch and analyze web pages
- JavaScript rendering (Playwright/Selenium)
- Custom headers, cookies, selectors

**Graph Analytics:**
1. Upload edge list + optional node attributes
2. Select metric: load, pagerank, bfs, triangles
3. View interactive graph with drag/zoom
4. **Color by centrality:** Select PageRank/Betweenness/Eigenvector/Degree
5. **Hover to show labels**, click to pin
6. Toggle physics for force-directed layout

**Ego Networks:**
- Upload multiple files at once (.edges, .circles, .feat, etc.)
- Automatic format detection
- Community visualization with colors

---

## Configuration

### Environment Variables

```bash
# GPU Detection
VERBOSE_GPU=1           # Show GPU detection messages (default: 1)

# API Settings
HOST=0.0.0.0           # Bind address
PORT=8080              # Port number
```

### Presets

**Sentiment:**
- `sentiment-twitter` - Twitter RoBERTa (3-way)
- `sentiment-sst2` - SST-2 (2-way: positive/negative)

**Zero-Shot:**
- `zeroshot-bart` - Facebook BART
- `zeroshot-mdeberta` - Microsoft DeBERTa

**NER:**
- `ner-conll` - CoNLL-2003 trained
- `ner-bertbase` - BERT base

**Other:**
- `spacy-ner`, `spacy-posdep`, `spacy-sents` - spaCy models
- `stanza-posdep`, `stanza-sents` - Stanza models
- `sbert-embed` - Sentence embeddings
- `bertopic` - Topic modeling with BERT
- `topics-nmf`, `topics-kmeans` - Classical topic models

---

## Development

### File Structure

```
.
├── app.py                  # FastAPI application
├── nlp.py                  # NLP task implementations
├── graph_tasks.py          # Graph algorithms (GPU + CPU)
├── fetch.py                # URL fetching and scraping
├── render.py               # JavaScript rendering
├── topics.py               # Topic modeling
├── adapters.py             # spaCy/Stanza adapters
├── index.html              # Web UI
├── req.txt                 # Python dependencies
├── Dockerfile.gpu          # GPU-enabled Docker image
├── Dockerfile.cpu          # CPU-only Docker image
├── run_local.py            # Local development server
├── run_docker_gpu.ps1      # Windows Docker GPU launcher
├── run_docker_gpu.sh       # Linux Docker GPU launcher
└── setup_wsl2_gpu.sh       # WSL2 GPU setup script
```

### Adding New NLP Models

Edit `nlp.py` and add to `PRESETS` dict:

```python
PRESETS = {
    "your-preset": {
        "model_name": "org/model-name",
        "task": "sentiment",  # or "ner", "zeroshot", etc.
        "labels": ["POSITIVE", "NEGATIVE"],  # if applicable
    }
}
```

### Modifying Graph Algorithms

Edit `graph_tasks.py`:
- GPU functions: `*_gpu()` (uses cuGraph)
- CPU functions: `*_cpu()` (uses NetworkX)
- Wrapper functions automatically choose GPU/CPU

---

## Troubleshooting

### GPU Not Detected

**Check NVIDIA drivers:**
```bash
nvidia-smi
```

**Check PyTorch CUDA:**
```python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.get_device_name(0))
```

**Check cuGraph (WSL2/Linux only):**
```python
import cudf, cugraph
print(cugraph.__version__)
```

**Windows:** cuGraph/cuDF do NOT support Windows natively. Use Docker or WSL2.

### Docker Volume Mount Not Working

**PowerShell syntax:**
```powershell
docker run --rm --gpus all -p 8080:8080 -v "$(Get-Location):/app" sentiment-gpu
```

**Bash syntax:**
```bash
docker run --rm --gpus all -p 8080:8080 -v "$(pwd):/app" sentiment-gpu
```

### Line Ending Errors in WSL2

```bash
dos2unix setup_wsl2_gpu.sh
bash setup_wsl2_gpu.sh
```

Or just run commands directly (skip the script).

### GraphBLAS Errors

GraphBLAS is optional. If you see errors:

```bash
pip uninstall python-graphblas pygraphblas -y
```

App works fine without it using NetworkX.

### Out of Memory (GPU)

**Reduce graph size:**
- UI automatically limits to 1000 nodes for visualization
- For large graphs, use API instead of UI
- Process in batches

**Increase Docker memory:**
- Docker Desktop → Settings → Resources → Memory
- Allocate at least 8GB for large graphs

---

## Dependencies

### Core
- FastAPI >= 0.111
- uvicorn >= 0.30
- PyTorch >= 2.3
- transformers >= 4.44
- pandas >= 2.2
- numpy >= 1.26
- networkx >= 3.0

### NLP
- spacy >= 3.7
- stanza >= 1.8
- nltk >= 3.8
- sentence-transformers >= 3.0

### Topic Modeling
- bertopic >= 0.16
- umap-learn >= 0.5
- hdbscan >= 0.8

### Web Scraping
- beautifulsoup4 >= 4.12
- trafilatura >= 1.6
- playwright >= 1.47
- selenium >= 4.23

### GPU (Optional)
- cudf-cu12 (RAPIDS, Linux/WSL2/Docker only)
- cugraph-cu12 (RAPIDS, Linux/WSL2/Docker only)

---

## License

MIT

## Contributing

Pull requests welcome! Please ensure:
1. Code passes syntax validation
2. New features include API documentation
3. GPU features have CPU fallbacks

---

## Citation

If you use this in research:

```bibtex
@software{cicerowatch_nlp,
  title = {CiceroWatch NLP Tasks},
  year = {2024},
  author = {Your Name},
  description = {GPU-accelerated sentiment analysis and graph analytics API}
}
```

---

## Support

- **Issues:** Report bugs via GitHub Issues
- **GPU Setup:** See inline troubleshooting above
- **API Docs:** http://localhost:8080/docs (when running)

---

## Quick Reference

**Start server (WSL2):**
```bash
cd /mnt/c/.../Sentiment\ Docker\ Test
conda activate base
python run_local.py
```

**Start server (Docker GPU):**
```powershell
.\run_docker_gpu.ps1
```

**Start server (CPU):**
```bash
python run_local.py
```

**Access UI:** http://localhost:8080
**Access API docs:** http://localhost:8080/docs
**Monitor GPU:** `nvidia-smi -l 1`

**Dev workflow:** Edit code → Save → Restart server (2 sec) → Changes live!
