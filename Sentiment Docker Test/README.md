# CiceroWatch - NLP & Graph Analytics Platform

Production-ready platform for sentiment analysis, batch Excel processing, social media network analysis, GPU-accelerated graph analytics, and audio deepfake detection.

## 🚀 Features

### 📊 Batch Excel Processing (NEW!)
- **Multi-sheet processing** - Automatically process all sheets or select specific ones
- **Sentiment analysis** - Extract sentiment from thousands of texts efficiently
- **Theme extraction** - Zero-shot classification with custom or default labels
- **Auto-column detection** - Automatically finds text columns
- **Preserves data** - Adds new columns without modifying originals
- **Batch optimized** - Processes thousands of rows with intelligent batching

**Quick Example:**
```bash
# Process Excel with sentiment + themes
curl -X POST http://localhost:8080/batch/excel \
  -F "file=@data.xlsx" \
  -F "theme_labels=politics,economics,technology,health"

# Returns: data_analyzed.xlsx with sentiment and theme columns added
```

### 🌐 Social Media Network Analysis (NEW!)
- **Multi-tab Excel support** - Process social media data from Excel sheets
- **Automatic edge extraction** - Mentions, replies, retweets, quotes, URLs, hashtags
- **Column auto-detection** - Finds author, text, timestamp columns automatically
- **Network visualization** - Ready-to-use graph data for visualization

**Quick Example:**
```bash
# Extract social network from Excel
curl -X POST http://localhost:8080/graph/prepare \
  -F "file=@tweets.xlsx" \
  -F "sheet=0" \
  -F "extract_hashtags=true"

# Returns: mention_edges, reply_edges, retweet_edges, domain_edges, hashtag_edges, nodes
```

### 📝 NLP & Text Analysis
- ✅ **8 Production Presets**:
  - `sentiment-twitter` - Twitter sentiment (RoBERTa)
  - `sentiment-sst2` - SST-2 sentiment (DistilBERT)
  - `zeroshot-bart` - Zero-shot classification (BART-MNLI)
  - `zeroshot-mdeberta` - Multilingual zero-shot (mDeBERTa)
  - `ner-conll` - Named entity recognition (BERT-NER)
  - `ner-bertbase` - Named entity recognition (BERT-base)
  - `topics-nmf` - Topic modeling (NMF with TF-IDF)
  - `topics-kmeans` - Topic clustering (K-means)

- 📄 **File Formats**: JSON, CSV, HTML, Excel (.xlsx, .xlsm, .xls)
- 🔗 **URL Analysis**: Fetch and analyze web content
- ⚡ **Batch API**: Process multiple texts in parallel

### 📈 Graph Analytics
- ⚡ **GPU Acceleration**: RAPIDS cuGraph (10-100x faster than CPU)
- 📊 **Algorithms**: PageRank, Betweenness, Eigenvector, Degree Centrality, BFS, Triangle Counting
- 🎨 **Interactive Visualization**: Vis.js network graphs
- 📁 **Formats**: CSV edge lists, JSON, SNAP ego networks (.edges, .circles, .feat, .featnames)
- 💻 **Cross-Platform**: NetworkX CPU fallback

### 🎤 Audio Deepfake Detection
- 🎤 **Model**: wav2vec2-large-anti-deepfake (NII Yamagishi Lab)
- 🔍 **Detection**: Identifies AI-generated voice (deepfakes)
- 📊 **Output**: Prediction (bonafide/spoofed), confidence, spoof score
- 🎧 **Formats**: WAV, FLAC, MP3

---

## 🏃 Quick Start

### Option 1: Docker (Recommended for Production)

```bash
cd "Sentiment Docker Test"

# CPU Version (any platform)
docker-compose up --build

# GPU Version (requires NVIDIA GPU)
docker-compose -f docker-compose.gpu.yml up --build

# Access at http://localhost:80
```

### Option 2: Local Python (Development)

```bash
cd "Sentiment Docker Test"

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements-nlp.txt

# Run server
python run_local.py

# Access at http://localhost:8080/docs
```

### Option 3: Conda (Windows)

```bash
# Create environment
conda create -n cicerowatch python=3.11 -y
conda activate cicerowatch

# Install PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install dependencies
pip install -r requirements-conda.txt

# Run server
python run_local.py
```

---

## 📖 API Documentation

### Batch Excel Processing

#### `/batch/excel` - Process Excel File

Upload an Excel file with multiple sheets for automated sentiment and theme extraction.

**Request:**
```bash
curl -X POST http://localhost:8080/batch/excel \
  -F "file=@data.xlsx" \
  -F "extract_sentiment=true" \
  -F "extract_themes=true" \
  -F "theme_labels=politics,economics,social issues,environment,technology" \
  -F "top_n_themes=3" \
  -F "sheets_to_process=Sheet1,Sheet2"
```

**Parameters:**
- `file` (required): Excel file (.xlsx, .xlsm, .xls)
- `extract_sentiment` (default: true): Extract sentiment from texts
- `extract_themes` (default: true): Extract themes using zero-shot classification
- `theme_labels` (optional): Comma-separated custom theme labels
- `sentiment_preset` (default: sentiment-twitter): Sentiment model preset
- `theme_preset` (default: zeroshot-bart): Zero-shot model preset
- `text_column` (optional): Text column name (auto-detected if not provided)
- `sheets_to_process` (optional): Comma-separated sheet names (processes all if not provided)
- `top_n_themes` (default: 3): Number of top themes to extract (1-10)
- `add_confidence_scores` (default: true): Include confidence scores
- `batch_size` (default: 32): Batch size for processing (1-128)

**Output Columns Added:**
- `sentiment`: Sentiment label (positive/negative/neutral)
- `sentiment_confidence`: Confidence score
- `theme_1`, `theme_2`, `theme_3`: Top N themes
- `theme_1_score`, `theme_2_score`, `theme_3_score`: Theme confidence scores

**Response:**
Returns enhanced Excel file with original data preserved + new analysis columns.

**Default Theme Labels** (if not provided):
```
politics, economy, military, health, science, technology, sports,
entertainment, climate, crime, education, misinformation, opinion
```

**Example Use Cases:**

1. **Military Exercise Detection:**
```bash
curl -X POST http://localhost:8080/batch/excel \
  -F "file=@defense_news.xlsx" \
  -F "theme_labels=UNITAS exercises,military cooperation,naval operations,humanitarian assistance,maritime security"
```

2. **News Categorization:**
```bash
curl -X POST http://localhost:8080/batch/excel \
  -F "file=@news_articles.xlsx" \
  -F "theme_labels=breaking news,politics,business,technology,health,sports,entertainment"
```

3. **Sentiment Only (Skip Themes):**
```bash
curl -X POST http://localhost:8080/batch/excel \
  -F "file=@customer_feedback.xlsx" \
  -F "extract_themes=false"
```

#### `/batch/excel-from-url` - Process Excel from URL

Download and process Excel file directly from a URL.

```bash
curl -X POST "http://localhost:8080/batch/excel-from-url?file_url=https://example.com/data.xlsx&theme_labels=politics,economics,technology"
```

**Requirements:**
- URL must be publicly accessible
- Must be direct link to Excel file
- File size limit: 100MB

---

### Social Media Graph Preparation

#### `/graph/prepare` - Extract Social Network Edges

Convert social media data (CSV/Excel) into network edge lists.

**Request:**
```bash
curl -X POST http://localhost:8080/graph/prepare \
  -F "file=@tweets.xlsx" \
  -F "sheet=0" \
  -F "extract_hashtags=true"
```

**Parameters:**
- `file` (required): CSV or Excel file with social media data
- `sheet` (default: "0"): Sheet name or index for Excel files
- `extract_hashtags` (default: false): Extract hashtag edges (user→hashtag)

**Automatically Detects Columns** (case-insensitive):
- **Author**: author, username, user, screen_name, from, account
- **Text**: text, full_text, tweet, content, body
- **Mentions**: mentions, mentioned_users, entities_user_mentions
- **Reply**: in_reply_to_screen_name, in_reply_to_user, reply_to
- **Retweet**: retweeted_user, retweeted_username, rt_username
- **Quote**: quoted_user, quoted_username
- **URLs**: urls, entities_urls, links, expanded_urls
- **Hashtags**: hashtags, entities_hashtags

**Response:**
```json
{
  "success": true,
  "edges": {
    "mention_edges": [{"src": "user1", "dst": "user2"}, ...],
    "reply_edges": [...],
    "retweet_edges": [...],
    "quote_edges": [...],
    "domain_edges": [{"src": "user1", "dst": "example.com"}, ...],
    "hashtag_edges": [{"src": "user1", "dst": "#politics"}, ...]
  },
  "stats": {
    "mention_count": 150,
    "reply_count": 50,
    "retweet_count": 30,
    "node_count": 200
  },
  "original_data": [...],
  "author_column": "username"
}
```

---

### Text Analysis

#### `/predict/file` - Analyze File

Analyze CSV, JSON, HTML, or Excel files.

```bash
# Sentiment analysis
curl -X POST http://localhost:8080/predict/file \
  -F "file=@tweets.csv" \
  -F "preset=sentiment-twitter"

# Named entity recognition
curl -X POST http://localhost:8080/predict/file \
  -F "file=@article.json" \
  -F "preset=ner-conll"

# Zero-shot classification
curl -X POST http://localhost:8080/predict/file \
  -F "file=@news.csv" \
  -F "preset=zeroshot-bart" \
  -F "labels=politics,technology,sports,health"
```

#### `/predict/url` - Analyze URL

Fetch and analyze web content.

```bash
curl -X POST http://localhost:8080/predict/url \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/article",
    "preset": "sentiment-sst2"
  }'
```

#### `/predict/batch` - Batch Text Analysis

Process multiple texts efficiently.

```bash
curl -X POST http://localhost:8080/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["First text", "Second text", "Third text"],
    "preset": "sentiment-twitter"
  }'
```

---

### Graph Analytics

#### `/graph/load` - Load & Visualize Graph

Load graph from edge list with centrality measures.

```bash
curl -X POST http://localhost:8080/graph/load \
  -F "edges_file=@edges.csv" \
  -F "compute_centrality=true"
```

**Response includes:**
- PageRank scores
- Betweenness centrality
- Eigenvector centrality
- Degree centrality
- Node/edge lists for visualization

#### `/graph/metrics` - Compute Multiple Metrics

```bash
curl -X POST http://localhost:8080/graph/metrics \
  -F "edges_file=@edges.csv" \
  -F "tasks=degrees,pagerank,triangles"
```

#### `/graph/ego-network` - Visualize Ego Networks

Load SNAP ego networks (.edges, .circles, .feat files).

```bash
curl -X POST http://localhost:8080/graph/ego-network \
  -F "files=@0.edges" \
  -F "files=@0.circles" \
  -F "files=@0.feat" \
  -F "ego_id=0"
```

---

### Audio Deepfake Detection

#### `/audio/analyze` - Detect Deepfakes

```bash
curl -X POST http://localhost:8080/audio/analyze \
  -F "file=@recording.wav"
```

**Response:**
```json
{
  "prediction": "bonafide",  // or "spoofed"
  "confidence": 0.92,
  "spoof_score": 0.08,  // higher = more likely fake
  "filename": "recording.wav"
}
```

---

## 🛠️ Local Development Setup

### Requirements
- Python 3.11 or 3.12
- 4GB+ RAM
- Optional: NVIDIA GPU (for GPU acceleration)

### Common Issues & Fixes

**"No module named 'transformers'"**
```bash
# Ensure venv is activated
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Reinstall dependencies
pip install -r requirements-nlp.txt
```

**"AttributeError: __pydantic_core_schema__" (Conda)**

This is a Pydantic version conflict. Fix:
```bash
conda activate cicerowatch

# Run the automated fix
bash fix_pydantic.sh  # Linux/Mac
fix_pydantic.bat      # Windows

# Or manually
pip uninstall -y pydantic pydantic-core fastapi uvicorn starlette
pip cache purge
pip install --no-cache-dir pydantic-core==2.27.1 pydantic==2.10.3 fastapi==0.115.5 uvicorn==0.32.1
```

**Windows numpy Build Errors**

Use Conda:
```bash
conda create -n cicerowatch python=3.11
conda activate cicerowatch
conda install numpy  # Get pre-built binary
pip install -r requirements-local.txt
```

Or use specific version:
```bash
pip install numpy==1.26.3
pip install -r requirements-local.txt
```

**IDE Configuration:**
- **VS Code**: `Ctrl+Shift+P` → "Python: Select Interpreter" → Choose venv
- **PyCharm**: Settings → Project Interpreter → Add → Select venv/bin/python
- **Spyder**: Activate venv first, then `pip install spyder` and launch from venv

---

## 🐳 Docker Setup

### Development (Live Editing)

```bash
cd "Sentiment Docker Test"

# Start services
docker-compose up --build

# Edit Python files and restart service (2 seconds)
docker-compose restart nlp

# Changes are live! No rebuild needed
```

### Production (Stable)

```bash
# CPU version
docker-compose -f docker-compose.prod.yml up -d

# GPU version
docker-compose -f docker-compose.prod.gpu.yml up -d
```

### GPU Setup (Windows/WSL2)

**Prerequisites:**
- NVIDIA GPU (RTX 2000+ series)
- Docker Desktop with WSL2
- NVIDIA drivers

**Installation:**
```bash
# Install WSL2
wsl --install -d Ubuntu-24.04

# In WSL2, install NVIDIA Container Toolkit
wsl
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Test GPU access
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

**Performance:**
| Dataset | CPU | GPU | Speedup |
|---------|-----|-----|---------|
| 1K nodes | 1 sec | 0.1 sec | 10x |
| 10K nodes | 30 sec | 0.5 sec | 60x |
| 100K nodes | 20 min | 10 sec | **120x** |

---

## 📊 Example Use Cases

### 1. Military Exercise Detection (UNITAS)

Analyze defense news for military exercises and cooperation:

```bash
curl -X POST http://localhost:8080/batch/excel \
  -F "file=@latin_america_defense.xlsx" \
  -F "theme_labels=UNITAS naval exercises,PANAMAX exercises,SOUTHCOM operations,bilateral defense,multilateral cooperation,humanitarian assistance,disaster relief,maritime security,counter-narcotics,anti-submarine warfare,U.S. naval presence,partner nation capacity building" \
  -F "top_n_themes=5" \
  -F "extract_sentiment=true"
```

**Output:** Excel file with sentiment + top 5 detected themes per article.

### 2. Social Media Influence Network

Extract and visualize Twitter influence networks:

```bash
# Step 1: Extract network edges
curl -X POST http://localhost:8080/graph/prepare \
  -F "file=@tweets.xlsx" \
  -F "sheet=0" \
  -F "extract_hashtags=true" \
  > network.json

# Step 2: Load and compute centrality
curl -X POST http://localhost:8080/graph/load \
  -F "edges_file=@mention_edges.csv" \
  -F "compute_centrality=true" \
  > graph_viz.json
```

**Output:** Network visualization with PageRank, betweenness, and eigenvector centrality scores.

### 3. Customer Feedback Analysis

Batch process customer reviews:

```bash
curl -X POST http://localhost:8080/batch/excel \
  -F "file=@customer_reviews.xlsx" \
  -F "extract_themes=true" \
  -F "theme_labels=product quality,customer service,shipping,price,ease of use,features,reliability,support" \
  -F "top_n_themes=2"
```

**Output:** Each review tagged with sentiment + top 2 relevant themes.

### 4. News Article Categorization

Categorize news by topic:

```bash
curl -X POST http://localhost:8080/batch/excel \
  -F "file=@news_feed.xlsx" \
  -F "theme_labels=breaking news,politics,business,technology,health,science,sports,entertainment,world news,local news" \
  -F "extract_sentiment=false"
```

### 5. Misinformation Detection

Detect potential misinformation indicators:

```bash
curl -X POST http://localhost:8080/batch/excel \
  -F "file=@social_media_posts.xlsx" \
  -F "theme_labels=verified facts,unverified claims,conspiracy theories,emotional manipulation,clickbait,satire,opinion,misleading context"
```

---

## 📂 File Formats

### Excel Input Format

**Requirements:**
- At least one column with text data
- System auto-detects columns named: `text`, `comment`, `message`, `content`, `description`, `post`, `tweet`
- Or specify manually with `text_column` parameter

**Example:**
```
| author     | text                           | timestamp          |
|------------|--------------------------------|--------------------|
| user1      | This is a sample tweet         | 2024-11-18 10:00   |
| user2      | Another tweet with content     | 2024-11-18 10:05   |
```

### CSV Edge Lists

**Format:**
```csv
src,dst
Alice,Bob
Bob,Charlie
Charlie,Alice
```

Or with weights:
```csv
src,dst,weight
Alice,Bob,5
Bob,Charlie,3
```

### JSON Lists

**Simple:**
```json
["text1", "text2", "text3"]
```

**With metadata:**
```json
[
  {"text": "text1", "author": "user1"},
  {"text": "text2", "author": "user2"}
]
```

---

## 🏗️ Architecture

### Deployment Options

**Option 1: Consolidated (run_local.py)**
- Single FastAPI server
- All features in one process
- Port: 8080
- Best for: Local development

**Option 2: Microservices (Docker)**
- NLP Service: Port 8001
- Graph Service: Port 8002
- Audio Service: Port 8003
- Frontend: Port 80 (nginx reverse proxy)
- Best for: Production, scalability

### Technology Stack

**NLP:**
- FastAPI + Uvicorn
- Transformers 4.40-4.47
- PyTorch 2.5.1
- HuggingFace Models (RoBERTa, BERT, BART, mDeBERTa)
- scikit-learn (NMF, K-means, TF-IDF)

**Graph:**
- RAPIDS cuGraph 24.08a (GPU)
- NetworkX (CPU fallback)
- pandas/cuDF

**Audio:**
- wav2vec2-large-anti-deepfake (fairseq)
- librosa, soundfile
- PyTorch 2.5.1

**Frontend:**
- nginx (Alpine)
- Vanilla JavaScript
- Vis.js (network visualization)

---

## 🧪 Testing

**Health Checks:**
```bash
curl http://localhost:8080/healthz
```

**API Documentation:**
```
http://localhost:8080/docs
```

**View Logs:**
```bash
# Docker
docker-compose logs -f nlp

# Local
python run_local.py  # Logs to stdout
```

---

## 🔧 Troubleshooting

### Port Already in Use

```bash
# Find process
sudo lsof -i :8080  # Linux/Mac
netstat -ano | findstr :8080  # Windows

# Kill process or change port in run_local.py
```

### Out of Memory

Reduce batch sizes in batch_processor.py:
```python
batch_size = 16  # Down from 32
```

Or limit Docker memory:
```yaml
services:
  nlp:
    deploy:
      resources:
        limits:
          memory: 4G
```

### Models Not Loading

First run downloads models (~2GB each):
```bash
# Pre-download models
python -c "from transformers import AutoModelForSequenceClassification, AutoTokenizer; \
AutoModelForSequenceClassification.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest'); \
AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest')"
```

---

## 📚 Version History

### Current Version (2024-11)
- ✅ **Batch Excel Processing** - Multi-sheet sentiment + theme extraction
- ✅ **Social Media Graph Preparation** - Excel-based network edge extraction
- ✅ **Pydantic 2.x Compatibility** - Fixed TypedDict issues
- ✅ **Windows Setup** - Numpy build fixes, Conda support
- ✅ **Military Analysis** - UNITAS exercise detection support
- ✅ **8 NLP Presets** - All working and tested
- ✅ **Audio Deepfake Detection**
- ✅ **GPU Graph Analytics** - RAPIDS cuGraph 24.08a
- ✅ **Live Code Editing** - Volume mounts for development

### Recent Fixes
- Fixed Pydantic TypeAdapter errors (TypedDict → BaseModel)
- Added requirements-conda.txt for Conda environments
- Fixed numpy Windows build issues
- Added openpyxl for Excel support
- Updated transformers to 4.47.1

---

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Credits

- **NLP Models**: HuggingFace Transformers
- **Graph Analytics**: NVIDIA RAPIDS, NetworkX
- **Audio Detection**: NII Yamagishi Lab wav2vec2-large-anti-deepfake
- **Frontend**: Vis.js, nginx

---

## 💬 Support

**Health Checks:**
```bash
curl http://localhost:8080/healthz
```

**Logs:**
```bash
# Docker
docker-compose logs -f <service>

# Local
python run_local.py  # Logs to stdout
```

**Issues:**
- Check API docs: http://localhost:8080/docs
- Verify environment activation: `which python`
- Test imports: `python -c "import transformers; print(transformers.__version__)"`

---

**Happy analyzing! 🚀**
