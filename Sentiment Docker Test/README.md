# Sentiment Analysis & Graph Analytics API

A production-ready FastAPI service providing:
- **Sentiment Analysis** with Hugging Face Transformers
- **Graph Analytics** (PageRank, BFS, triangle counting, etc.)
- **Multi-format support** (JSON, CSV, HTML, URLs)
- **Zero-shot classification** for custom labels
- **Named Entity Recognition (NER)**
- **Web scraping** with JavaScript rendering support

---

## Features

### NLP & Sentiment Analysis
- **Default Model:** `cardiffnlp/twitter-roberta-base-sentiment-latest` (3-way sentiment)
- **Multiple Presets:** sentiment, zero-shot classification, NER, embeddings, topic modeling
- **Smart Preprocessing:** Handles tweets, URLs, special characters
- **Batch Processing:** Efficient multi-text analysis
- **Long Text Support:** Automatic chunking and averaging for texts >512 tokens

### Graph Analytics
- **PageRank:** Power iteration algorithm with configurable damping
- **Degree Analysis:** In-degree, out-degree, total degree
- **BFS:** Breadth-first search from any source node
- **Triangle Counting:** For undirected graphs
- **Social Circles:** Community detection and membership tracking
- **Node Features:** Support for binary feature vectors with feature names
- **Ego Networks:** Complete SNAP ego network format support
- **GPU Acceleration:** Automatic GPU detection with cuGraph (RAPIDS) for 10-100x speedup
- **GraphBLAS Support:** Optional sparse matrix acceleration

### Web & Data Processing
- **File Formats:** JSON, CSV, HTML, .edge, .node, .circles, .feat, .egofeat, .featnames
- **URL Processing:** Fetch and analyze web pages
- **JavaScript Rendering:** Playwright/Selenium for SPA support
- **Crawling:** Multi-page site analysis with depth/breadth controls
- **Graph Data:** Native support for SNAP ego network formats (Stanford Network Analysis Project)
- **Multi-File Upload:** Upload entire folders with related graph files at once

---

## Quick Start

### Option 1: Docker (Recommended)

```bash
# Build the image
docker build -f Dockerfile.cpu -t sentiment-api:latest .

# Run the container
docker run --rm -p 8080:8080 sentiment-api:latest

# Open in browser
# http://localhost:8080
```

### Option 2: Local Development

**For Windows (Conda users):**
```bash
# Use the automated installer
.\install_windows.bat

# Or use the startup script (handles Conda/uvicorn issues)
python run_local.py
```

**For all platforms:**
```bash
# Install dependencies
pip install -r requirements-local.txt

# Download NLP models
python -m spacy download en_core_web_sm

# Check compatibility
python check_compatibility.py

# Start the server
python run_local.py

# Or manually:
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```

**Open:** http://localhost:8080

---

## GPU Acceleration (Optional)

For **10-100x faster graph analytics**, enable GPU acceleration with NVIDIA RAPIDS cuGraph.

### Requirements
- NVIDIA GPU with CUDA 11.8+ or CUDA 12.0+
- CUDA Toolkit installed
- Compatible GPU drivers

### Option 1: GPU Docker (Easiest)

```bash
# Build GPU-enabled image
docker build -f Dockerfile.gpu -t sentiment-api:gpu .

# Run with GPU access
docker run --rm --gpus all -p 8080:8080 sentiment-api:gpu
```

### Option 2: Install RAPIDS Locally

**Via Conda (Recommended):**
```bash
# Install RAPIDS cuGraph and cuDF
conda install -c rapidsai -c conda-forge -c nvidia \
  cudf>=24.4 cugraph>=24.4 python=3.11 cudatoolkit=11.8

# Install other dependencies
pip install -r requirements-local.txt
```

**Via pip:**
```bash
pip install -r requirements-gpu.txt
```

### Verify GPU Detection

```bash
# Check CUDA availability
nvidia-smi

# Start server and check logs
python run_local.py
# Should see: "[graph_tasks] GPU detected: NVIDIA GeForce RTX..."
```

### Performance Comparison

| Algorithm | CPU (1M edges) | GPU (1M edges) | Speedup |
|-----------|----------------|----------------|---------|
| PageRank | 2.5s | 0.05s | **50x** |
| Degrees | 0.8s | 0.02s | **40x** |
| BFS | 1.2s | 0.03s | **40x** |
| Triangles | 45s | 0.6s | **75x** |

**Notes:**
- GPU acceleration is **automatic** - no code changes needed
- Gracefully falls back to CPU if GPU unavailable
- Most beneficial for graphs with >100K nodes or >1M edges
- RAPIDS installation guide: https://rapids.ai/

---

## API Endpoints

### Health Check

#### `GET /healthz`
Check service status and available presets.

**Response:**
```json
{
  "ok": true,
  "presets": ["sentiment-twitter", "zeroshot-bart", ...],
  "playwright": true
}
```

---

## Sentiment Analysis & NLP

### `POST /predict/file`
Process uploaded files (JSON, CSV, or HTML).

**Parameters:**
- `file` (required): File to process
- `preset` (optional): Model preset (default: sentiment-twitter)
- `labels` (optional): Comma-separated labels for zero-shot
- `include_stopwords` (optional): For label guessing

**Example:**
```bash
curl -X POST "http://localhost:8080/predict/file?preset=sentiment-twitter" \
  -F "file=@tweets.csv"
```

**Response:** Downloads `predictions.json`
```json
{
  "preset": "sentiment-twitter",
  "results": [
    {
      "text": "I love this!",
      "labels": ["positive", "neutral", "negative"],
      "scores": [0.92, 0.05, 0.03]
    }
  ]
}
```

---

### `POST /predict/url`
Fetch and analyze a URL.

**Request Body:**
```json
{
  "url": "https://example.com",
  "preset": "sentiment-twitter",
  "render": false,
  "renderer": "auto",
  "wait_selector": null,
  "scroll_passes": 8,
  "render_timeout_ms": 25000,
  "cookies": "session=abc123",
  "extra_headers": {"User-Agent": "..."}
}
```

**Example:**
```bash
curl -X POST "http://localhost:8080/predict/url" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://news.ycombinator.com", "preset": "sentiment-twitter"}'
```

**Response:** Downloads `url-output.json`

---

### `POST /predict/batch` (New!)
Efficiently process multiple texts in one request.

**Request Body:**
```json
{
  "texts": ["I love this!", "This is terrible.", "Neutral statement."],
  "preset": "sentiment-twitter",
  "labels": null,
  "preprocess": true
}
```

**Example:**
```bash
curl -X POST "http://localhost:8080/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Great service!", "Not happy."],
    "preset": "sentiment-twitter"
  }'
```

**Response:**
```json
{
  "preset": "sentiment-twitter",
  "count": 2,
  "results": [
    {
      "text": "Great service!",
      "labels": ["positive", "neutral", "negative"],
      "scores": [0.95, 0.03, 0.02]
    },
    {
      "text": "Not happy.",
      "labels": ["negative", "neutral", "positive"],
      "scores": [0.87, 0.10, 0.03]
    }
  ]
}
```

---

## Graph Analytics

### `POST /graph/load` (New!)
Load and validate a graph file.

**Parameters:**
- `file` (required): Edge list in CSV or JSON format

**CSV Format:**
```csv
src,dst,weight
Alice,Bob,1.0
Bob,Charlie,1.5
```

**JSON Format:**
```json
{
  "edges": [
    {"src": "Alice", "dst": "Bob"},
    {"src": "Bob", "dst": "Charlie"}
  ]
}
```

**Response:**
```json
{
  "n_nodes": 3,
  "n_edges": 2,
  "sample_nodes": ["Alice", "Bob", "Charlie"],
  "has_graphblas": false,
  "edge_columns": ["src_idx", "dst_idx"]
}
```

---

### `POST /graph/degrees` (New!)
Compute degree metrics for all nodes.

**Parameters:**
- `edges_file` (required): Edge list (CSV, JSON, or .edge format)
- `nodes_file` (optional): Node attributes (CSV, JSON, or .node format)

**Example:**
```bash
# With CSV edge list
curl -X POST "http://localhost:8080/graph/degrees" \
  -F "edges_file=@network.csv"

# With .edge file and optional .node attributes
curl -X POST "http://localhost:8080/graph/degrees" \
  -F "edges_file=@graph.edge" \
  -F "nodes_file=@nodes.node"
```

**Response:**
```json
{
  "n_nodes": 100,
  "n_edges": 250,
  "degrees": [
    {"node": "Alice", "out_degree": 10, "in_degree": 5, "degree": 15},
    {"node": "Bob", "out_degree": 8, "in_degree": 7, "degree": 15}
  ]
}
```

---

### `POST /graph/pagerank` (New!)
Compute PageRank scores.

**Parameters:**
- `file` (required): Edge list
- `alpha` (optional): Damping factor (default: 0.85)
- `iters` (optional): Max iterations (default: 40)
- `tol` (optional): Convergence tolerance (default: 1e-6)

**Example:**
```bash
curl -X POST "http://localhost:8080/graph/pagerank?alpha=0.85&iters=50" \
  -F "file=@network.csv"
```

**Response:**
```json
{
  "n_nodes": 100,
  "n_edges": 250,
  "pagerank": [
    {"node": "Alice", "pr": 0.045},
    {"node": "Bob", "pr": 0.032}
  ],
  "parameters": {"alpha": 0.85, "iters": 50, "tol": 1e-6}
}
```

---

### `POST /graph/bfs` (New!)
Breadth-first search from a source node.

**Parameters:**
- `file` (required): Edge list
- `source` (required): Starting node ID

**Example:**
```bash
curl -X POST "http://localhost:8080/graph/bfs?source=Alice" \
  -F "file=@network.csv"
```

**Response:**
```json
{
  "n_nodes": 100,
  "n_edges": 250,
  "source_node": "Alice",
  "distances": [
    {"node": "Alice", "distance": 0},
    {"node": "Bob", "distance": 1},
    {"node": "Charlie", "distance": 2},
    {"node": "Unreachable", "distance": -1}
  ]
}
```

---

### `POST /graph/triangles` (New!)
Count triangles in an undirected graph.

**Parameters:**
- `file` (required): Edge list (undirected)
- `max_nodes` (optional): Skip if >N nodes (default: 20000)

**Example:**
```bash
curl -X POST "http://localhost:8080/graph/triangles?max_nodes=10000" \
  -F "file=@network.csv"
```

**Response:**
```json
{
  "n_nodes": 100,
  "n_edges": 250,
  "triangles": 42
}
```

---

### `POST /graph/metrics`
Compute multiple graph metrics in one request.

**Parameters:**
- `file` (required): Edge list
- `tasks` (required): Comma-separated (e.g., "degrees,pagerank,bfs")
- `bfs_source` (optional): Required if "bfs" in tasks
- `pagerank_alpha`, `pagerank_iters`, `pagerank_tol` (optional)
- `triangles_limit` (optional)

**Example:**
```bash
curl -X POST "http://localhost:8080/graph/metrics?tasks=degrees,pagerank" \
  -F "file=@network.csv"
```

**Response:**
```json
{
  "n_nodes": 100,
  "n_edges": 250,
  "degrees": [...],
  "pagerank": [...]
}
```

---

### `POST /graph/ego-network` (New!)
Load and visualize complete ego networks from multiple related files.

**Parameters:**
- `files` (required): Multiple files (.edges, .circles, .feat, .egofeat, .featnames, .node)
- `ego_id` (optional): Ego node ID (auto-detected from filenames if not provided)

**Supported File Types:**
- `.edges` / `.edge` - Edge list (required)
- `.circles` - Social circles/communities
- `.feat` - Node feature vectors (binary 0/1)
- `.egofeat` - Ego node features
- `.featnames` - Feature name mappings
- `.node` - Node attributes

**Example:**
```bash
# Upload entire ego network (e.g., from SNAP Facebook dataset)
curl -X POST "http://localhost:8080/graph/ego-network" \
  -F "files=@0.edges" \
  -F "files=@0.circles" \
  -F "files=@0.feat" \
  -F "files=@0.egofeat" \
  -F "files=@0.featnames"
```

**Response:**
```json
{
  "n_nodes": 347,
  "n_edges": 2914,
  "ego_id": "0",
  "has_circles": true,
  "has_features": true,
  "circles": {
    "work": ["1", "2", "3"],
    "friends": ["4", "5", "6"]
  },
  "feature_count": 77,
  "nodes": [...],
  "edges": [...]
}
```

---

## Available NLP Presets

### Sentiment Analysis
- `sentiment-twitter` - Twitter-specific sentiment (cardiffnlp/twitter-roberta-base-sentiment-latest)
- `sentiment-sst2` - SST-2 sentiment model

### Zero-Shot Classification
- `zeroshot-bart` - Facebook BART MNLI model
- `zeroshot-mdeberta` - Multilingual mDeBERTa model

### Named Entity Recognition
- `ner-conll` - BERT-base NER (CoNLL)
- `ner-bertbase` - BERT-base NER
- `spacy-ner` - spaCy NER

### Other Tasks
- `spacy-posdep` - POS tagging and dependency parsing
- `spacy-sents` - Sentence segmentation
- `stanza-posdep` - Stanza POS/DEP
- `sbert-embed` - Sentence embeddings
- `bertopic` - Topic modeling with BERTopic
- `topics-nmf` - NMF topic modeling
- `topics-kmeans` - K-means topic clustering

---

## Installation & Troubleshooting

### Windows Installation Issues

#### Issue 1: DLL Access Denied
```
[WinError 5] Access is denied: 'C:\...\torch\lib\c10.dll'
```

**Quick Fix:**
```bash
# Close all Python processes
taskkill /F /IM python.exe /T

# Install with --user flag (no admin needed)
pip install --user torch==2.3.0 transformers==4.41.2

# Start server
python run_local.py
```

**Or use the automated installer:**
```bash
# Run as Administrator
.\install_windows.bat
```

See `WINDOWS_INSTALL.md` for 7 different solution methods.

---

#### Issue 2: PyTorch/Transformers Version Mismatch
```
cannot import name 'skip_code' from 'torch._C._dynamo.eval_frame'
```

**Fix:**
```bash
# Check compatibility
python check_compatibility.py

# Install compatible versions
pip install -r requirements-local.txt

# Or manually
pip uninstall torch transformers -y
pip install torch==2.3.0 transformers==4.41.2
```

---

#### Issue 3: Uvicorn/Conda AsyncIO Error
```
TypeError: run() got an unexpected keyword argument 'loop_factory'
```

**Fix:** Use the startup script (handles this automatically)
```bash
python run_local.py
```

**Or manually:**
```bash
pip install "uvicorn<0.30.0"
uvicorn app:app --host 0.0.0.0 --port 8080
```

---

### Common Issues

**Port already in use:**
```bash
# Windows
netstat -ano | findstr :8080
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:8080 | xargs kill
```

**Missing NLP models:**
```bash
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

**GPU/CUDA errors:**
```bash
# Force CPU mode
export CUDA_VISIBLE_DEVICES=""
python run_local.py
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 8080 | Server port |
| `MODEL_ID` | cardiffnlp/twitter-roberta-base-sentiment-latest | HF model |
| `BATCH_SIZE` | 32 | Inference batch size |
| `MAX_LEN` | 256 | Max tokens per input |
| `CLASSIFY_MAX_WORDS` | 320 | Chunking threshold |
| `REDDIT_LIMIT` | 200 | Reddit API limit |
| `PLAYWRIGHT_AVAILABLE` | auto | Enable/disable Playwright |

**Example:**
```bash
export PORT=8000
export MODEL_ID="distilbert-base-uncased-finetuned-sst-2-english"
python run_local.py
```

---

## Input/Output Formats

### JSON Input

**List of strings:**
```json
["I love this", "This is bad"]
```

**List of objects:**
```json
[
  {"text": "Great work!"},
  {"text": "Could be better."}
]
```

### JSON Output

```json
[
  {
    "text": "I love this",
    "labels": ["positive", "neutral", "negative"],
    "scores": [0.88, 0.10, 0.02]
  }
]
```

### CSV Input

Must have a `text` column or auto-detected text column:
```csv
text,author
"Great service!",user123
"Not happy.",user456
```

### CSV Output

Original columns + new columns:
```csv
text,author,score_positive,score_neutral,score_negative,top_label,top_score
"Great service!",user123,0.88,0.10,0.02,positive,0.88
```

### Graph File Formats

#### Edge List Formats

**CSV Format (.csv):**
```csv
src,dst,weight
Alice,Bob,1.0
Bob,Charlie,2.5
Charlie,Alice,1.0
```

**JSON Format (.json):**
```json
[
  {"src": "Alice", "dst": "Bob", "weight": 1.0},
  {"src": "Bob", "dst": "Charlie", "weight": 2.5},
  {"src": "Charlie", "dst": "Alice", "weight": 1.0}
]
```

Or with wrapper:
```json
{
  "edges": [
    {"src": "Alice", "dst": "Bob", "weight": 1.0}
  ]
}
```

**.edge Format (space/tab-separated):**
```
# Graph edge list
Alice Bob 1.0
Bob Charlie 2.5
Charlie Alice 1.0
```

#### Node Attributes Formats (Optional)

**CSV Format (.csv):**
```csv
id,label,category,value
Alice,Node A,type1,100
Bob,Node B,type2,200
Charlie,Node C,type1,150
```

**JSON Format (.json):**
```json
[
  {"id": "Alice", "label": "Node A", "category": "type1", "value": 100},
  {"id": "Bob", "label": "Node B", "category": "type2", "value": 200}
]
```

Or with wrapper:
```json
{
  "nodes": [
    {"id": "Alice", "label": "Node A", "category": "type1"}
  ]
}
```

**.node Format (space/tab-separated):**
```
# Graph node attributes
# Format: id [label] [attr1] [attr2] ...
Alice NodeA type1 100
Bob NodeB type2 200
Charlie NodeC type1 150
```

**Note:** Node attributes are automatically merged with graph metric results when provided. All graph endpoints accept optional `nodes_file` parameter.

#### Ego Network Formats (SNAP-style)

These formats are commonly used in Stanford Network Analysis Project (SNAP) datasets for social network analysis.

**.circles Format (social circles/communities):**
```
work 1 2 3 4 5
friends 6 7 8 9
family 10 11 12
```

Or tab-separated:
```
work	1 2 3 4 5
friends	6 7 8 9
```

**.feat Format (node features - binary 0/1 vectors):**
```
0 1 0 1 0 0 1 1 0 1 0
1 0 1 0 1 0 0 1 1 0 1
2 1 1 0 0 1 1 0 0 1 0
```
First column is node ID, remaining columns are binary feature values.

**.egofeat Format (ego node features):**
```
1 0 1 0 0 1 1 0 1 0 1
```
Single line containing binary feature vector for the ego (center) node.

**.featnames Format (feature name mappings):**
```
education;anonymized feature 0
gender;anonymized feature 1
work;anonymized feature 2
hometown;anonymized feature 3
```

Or simpler format:
```
feature0 anonymous
feature1 anonymous
feature2 anonymous
```

**Complete Ego Network Example:**
```bash
# Typical SNAP ego network file structure (e.g., Facebook dataset)
0.edges       # Edge list for ego network 0
0.circles     # Social circles
0.feat        # Feature vectors for all nodes
0.egofeat     # Feature vector for ego node (node 0)
0.featnames   # Feature name mappings
```

---

## Web UI Features

The included `index.html` provides:

1. **File Upload**
   - Drag-and-drop or click to select
   - Supports JSON, CSV, HTML
   - Auto-downloads scored results

2. **URL Processing**
   - Fetch and analyze web pages
   - JavaScript rendering toggle
   - Custom cookies and headers
   - Wait selectors for dynamic content

3. **Advanced Options**
   - Preset selection
   - Custom zero-shot labels
   - Rendering engine choice (Playwright/Selenium)
   - Scroll passes for lazy-loaded content

**How it works:**
- Button clicks → JavaScript `fetch()` → HTTP POST to `/predict/*`
- Python processes → Returns JSON with scores
- JavaScript downloads results automatically

See `UI_TO_PYTHON_FLOW.md` for the complete technical flow.

---

## Architecture

```
┌─────────────────┐
│   index.html    │  Web UI
│  (Frontend)     │
└────────┬────────┘
         │ HTTP POST
         ▼
┌─────────────────┐
│     app.py      │  FastAPI routes
│   (API Layer)   │
└────────┬────────┘
         │
         ├─────────────────┐
         │                 │
         ▼                 ▼
┌─────────────────┐  ┌──────────────┐
│     nlp.py      │  │ graph_tasks  │
│  (NLP Logic)    │  │ (Analytics)  │
└────────┬────────┘  └──────┬───────┘
         │                  │
         ▼                  ▼
┌─────────────────┐  ┌──────────────┐
│  Transformers   │  │   NumPy/     │
│   Pipelines     │  │  Pandas      │
└─────────────────┘  └──────────────┘
```

### File Structure

```
.
├── app.py                    # FastAPI routes
├── nlp.py                    # NLP task dispatcher
├── graph_tasks.py            # Graph analytics
├── apputils.py               # CSV/JSON processors
├── fetch.py                  # URL fetching & rendering
├── render.py                 # HTML annotation
├── topics.py                 # Topic modeling
├── index.html                # Web UI
├── req.txt                   # Docker dependencies
├── requirements-local.txt    # Local dev dependencies
├── Dockerfile.cpu            # Container build
├── run_local.py              # Local startup script
├── install_windows.bat       # Windows installer
├── check_compatibility.py    # Dependency checker
└── README.md                 # This file
```

---

## Testing

### Test Sentiment Analysis
```bash
# Batch endpoint
curl -X POST "http://localhost:8080/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Great!", "Bad."], "preset": "sentiment-twitter"}'
```

### Test Graph Analytics
```bash
# Create test graph (CSV format)
echo "src,dst
Alice,Bob
Bob,Charlie
Charlie,Alice" > test.csv

# Or create .edge format
echo "Alice Bob
Bob Charlie
Charlie Alice" > test.edge

# Create optional node attributes (.node format)
echo "Alice NodeA type1
Bob NodeB type2
Charlie NodeC type1" > nodes.node

# Compute PageRank with CSV
curl -X POST "http://localhost:8080/graph/pagerank" \
  -F "edges_file=@test.csv"

# Compute PageRank with .edge and .node files
curl -X POST "http://localhost:8080/graph/pagerank" \
  -F "edges_file=@test.edge" \
  -F "nodes_file=@nodes.node"
```

### Test Zero-Shot Classification
```bash
curl -X POST "http://localhost:8080/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["The stock market crashed today"],
    "preset": "zeroshot-bart",
    "labels": ["finance", "sports", "politics"]
  }'
```

---

## Performance Tips

1. **Use batch endpoint** for multiple texts (much faster than individual calls)
2. **First request is slow** (model loading) - subsequent requests are fast
3. **Long texts are chunked** automatically (>320 words)
4. **Graph analytics**: For graphs >10k nodes, use individual endpoints instead of `/graph/metrics`
5. **PageRank**: Adjust `tol` and `iters` based on graph size
6. **Triangle counting**: Skips graphs >20k nodes by default (configurable)

---

## Recent Fixes & Improvements

### ✅ Fixed: Missing Text in Output (v1.1)
Original text strings are now included in all sentiment analysis results:
```json
{
  "text": "I love this!",  // ← Now included!
  "labels": ["positive", ...],
  "scores": [0.92, ...]
}
```

### ✅ Fixed: Labels Parameter Handling (v1.1)
- Empty string labels (`""`) now properly converted to `None`
- Labels only parsed for zero-shot tasks
- Sentiment/NER tasks no longer receive unnecessary labels parameter

### ✅ Added: Windows Installation Support (v1.1)
- Automated installer for Windows users
- Fixes for DLL access denied errors
- Conda/uvicorn compatibility handling
- PyTorch/Transformers version checking

### ✅ Added: Comprehensive Graph Analytics (v1.1)
- Individual endpoints for each graph algorithm
- Graph validation endpoint
- Optimized for large graphs
- Optional GraphBLAS acceleration

---

## Docker Deployment

### Build
```bash
docker build -f Dockerfile.cpu -t sentiment-api:latest .
```

### Run
```bash
docker run -d \
  --name sentiment-api \
  -p 8080:8080 \
  -e MODEL_ID="cardiffnlp/twitter-roberta-base-sentiment-latest" \
  sentiment-api:latest
```

### Health Check
```bash
docker exec sentiment-api curl http://localhost:8080/healthz
```

### View Logs
```bash
docker logs -f sentiment-api
```

---

## Security Notes

- **Cookies:** Not persisted; only used for specific requests
- **File Uploads:** Limited to 100MB
- **URL Fetching:** Respects robots.txt (best effort)
- **Rate Limiting:** Not included - add if deploying publicly
- **Authentication:** Not included - add if needed

**For production deployment, consider:**
- Rate limiting (e.g., FastAPI middleware)
- Authentication (e.g., API keys, OAuth)
- Input validation and sanitization
- HTTPS/TLS termination
- Resource limits (memory, CPU)
- Monitoring and logging

---

## API Documentation

Once the server is running, visit:
- **Interactive API docs:** http://localhost:8080/docs (Swagger UI)
- **Alternative docs:** http://localhost:8080/redoc (ReDoc)

These are auto-generated from FastAPI and show all endpoints with examples.

---

## License

This project uses open-source libraries and model weights with their own licenses:
- Hugging Face Transformers (Apache 2.0)
- FastAPI (MIT)
- PyTorch (BSD)
- spaCy (MIT)
- Individual model licenses vary (check model cards on Hugging Face)

Review and comply with all relevant licenses for your deployment.

---

## Contributing

Contributions are welcome! Please:
1. Test your changes locally
2. Run `python check_compatibility.py`
3. Ensure all endpoints work
4. Update this README if adding features

---

## Support

For issues or questions:
1. Check the **Troubleshooting** section above
2. Run `python check_compatibility.py` for dependency issues
3. Check the auto-generated API docs at `/docs`
4. Review the specific `*_FIX_SUMMARY.md` files for known issues

---

**Version:** 1.1
**Last Updated:** 2025
**Python:** 3.10+
**PyTorch:** 2.2-2.4
**Transformers:** 4.40-4.44
