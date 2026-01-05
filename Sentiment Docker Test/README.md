# CiceroWatch - Batch Excel NLP Processing Platform

Production-ready microservices platform for batch Excel annotation with NLP analysis, graph analytics, and audio deepfake detection.

## Architecture

**Microservices Design:**
- **Frontend** (nginx) - Serves UI and reverse proxy
- **NLP Service** (Python 3.12 + GPU) - Batch Excel processing with stance, themes, and narrative extraction
- **Graph Service** (Python 3.11) - Network analysis and centrality metrics
- **Audio Service** (Python 3.10 + fairseq) - Audio deepfake detection

All services communicate via REST APIs and support live code editing without rebuilds.

---

## Features

### Batch Excel Processing (Primary Feature)
- **Excel Input**: Upload Excel files with URL columns for batch annotation
- **Stance/Sentiment Classification**: Automatic stance detection (Support, Oppose, Neutral, Discuss)
- **Theme Extraction**: Zero-shot classification with customizable themes from Guidance sheet
- **Narrative Summarization**: AI-generated summaries for each article
- **Few-Shot Learning**: Train from human-annotated examples to improve accuracy
- **Checkpoint Saves**: Automatic progress saves during batch processing
- **Twitter/X Support**: Automatic fallback to mirror sites for Twitter URLs

### Guidance Sheet Parsing
- **Custom Themes**: Define priority themes in your Excel Guidance sheet
- **Numbered Lists**: Supports numbered priorities (1, 2, 3...) in any location on the sheet
- **Auto-Detection**: Finds numbered lists and extracts themes from adjacent column

### Annotation Comparison (QA/Validation)
- **Side-by-Side Comparison**: Upload automated vs human annotated files
- **URL-Based Matching**: Rows matched by URL for accurate comparison
- **Accuracy Metrics**: Stance accuracy, theme match rates, confusion matrix
- **Diff Output**: Excel file highlighting agreements and disagreements

### Graph Analytics
- **Centrality Measures**: PageRank, Betweenness, Eigenvector, Degree Centrality
- **Network Visualization**: Interactive Vis.js graphs
- **Ego Networks**: SNAP format support (.edges, .circles, .feat, .featnames)
- **CPU Processing**: Lightweight NetworkX-based processing

### Audio Deepfake Detection
- **Model**: wav2vec2-large-anti-deepfake (NII Yamagishi Lab)
- **Detection**: Identifies AI-generated voice (deepfakes)
- **Output**: Prediction (bonafide/spoofed), confidence, spoof score
- **Formats**: WAV, FLAC, MP3

---

## Quick Start

### Using Docker Profiles

The platform uses profile-based service selection:

```bash
cd "Sentiment Docker Test"

# Core services (Frontend + NLP + Graph) - Most common
docker compose -f docker-compose.minimal.yml --profile core up -d

# NLP only (Batch Excel processing)
docker compose -f docker-compose.minimal.yml --profile nlp up -d

# All services including Audio
docker compose -f docker-compose.minimal.yml --profile full up -d

# Access at http://localhost:8080
```

### Development Workflow (Live Editing)

All Python files are volume-mounted for instant changes:

```bash
# 1. Edit Python files (service_nlp.py, nlp_processor.py, etc.)
# 2. Restart service (takes ~2 seconds)
docker compose -f docker-compose.minimal.yml --profile core restart nlp

# 3. Changes are live!

# For HTML changes - just refresh the browser (no restart needed)
```

**Only rebuild when:**
- Changing `Dockerfile.*`
- Changing `requirements-*.txt`
- Installing new system packages

---

## Batch Excel Processing

### Input Format

Your Excel file should contain:
- **URL column**: Links to articles/tweets for processing
- **Optional Guidance sheet**: Custom themes as numbered list

### Guidance Sheet Format

To define custom themes, add a "Guidance" sheet with numbered priorities:

```
Column A    Column B
1           Economic Impact
2           Public Health
3           Political Analysis
4           Environmental Concerns
```

The system scans for consecutive numbered rows (1, 2, 3...) and extracts themes from the adjacent column.

### Output

Processed Excel includes:
- **Stance**: Support, Oppose, Neutral, or Discuss
- **Themes**: Matched themes from guidance (comma-separated)
- **Narrative**: AI-generated summary of the article
- **Confidence**: Model confidence scores

### Few-Shot Learning

Train the model with human-annotated examples:

1. Upload a training Excel file with URL, Stance, and Themes columns
2. System learns from these examples to improve accuracy
3. Training data persists between container restarts

---

## Annotation Comparison

Compare automated annotations against human annotations for QA:

1. **Upload two files**: Automated results and human-annotated ground truth
2. **URL matching**: Rows are matched by URL column
3. **Metrics**: View stance accuracy, theme match rates
4. **Diff output**: Download Excel with side-by-side comparison

### Comparison Output

- **Stance_Match**: MATCH, MISMATCH, or BOTH_EMPTY
- **Themes_Match**: MATCH, PARTIAL (N common), MISMATCH, or BOTH_EMPTY
- **Summary sheet**: Overall accuracy statistics

---

## API Reference

### NLP Service (Port 8001)

**Batch Excel Processing:**
```bash
curl -X POST http://localhost:8080/api/nlp/process-batch \
  -F "file=@articles.xlsx" \
  -F "url_column=URL" \
  -F "include_stance=true" \
  -F "include_themes=true" \
  -F "include_narrative=true"
```

**Annotation Comparison:**
```bash
curl -X POST http://localhost:8080/api/nlp/compare-annotations \
  -F "automated_file=@automated.xlsx" \
  -F "human_file=@human_annotated.xlsx" \
  -F "url_column=URL" \
  -F "compare_stance=true" \
  -F "compare_themes=true"
```

**Upload Training Data:**
```bash
curl -X POST http://localhost:8080/api/nlp/training/upload \
  -F "file=@training_examples.xlsx"
```

### Graph Service (Port 8002)

**Centrality Metrics:**
```bash
# PageRank
curl -X POST http://localhost:8080/api/graph/pagerank \
  -F "edges_file=@edges.csv"

# Betweenness Centrality
curl -X POST http://localhost:8080/api/graph/betweenness \
  -F "edges_file=@edges.csv"
```

**Ego Network Visualization:**
```bash
curl -X POST http://localhost:8080/api/graph/ego-network \
  -F "files=@0.edges" \
  -F "files=@0.circles" \
  -F "ego_id=0"
```

### Audio Service (Port 8003)

**Deepfake Detection:**
```bash
curl -X POST http://localhost:8080/api/audio/analyze \
  -F "file=@recording.wav"
```

---

## Configuration

### Service Ports
- **Frontend**: 8080 (nginx reverse proxy)
- **NLP**: 8001 (FastAPI)
- **Graph**: 8002 (FastAPI)
- **Audio**: 8003 (FastAPI)

### Environment Variables
- `HF_TOKEN`: HuggingFace token for gated models
- `TZ`: Timezone (default: America/New_York)
- `CUDA_VISIBLE_DEVICES`: GPU device selection

### Volume Mounts

Python files mounted for live editing:
- `service_nlp.py` - NLP API endpoints
- `nlp_processor.py` - NLP business logic
- `nlp.py` - Model presets
- `topics.py` - Theme extraction
- `training_store.py` - Few-shot learning storage
- `guidance_defaults.json` - Default theme list

---

## Project Structure

```
Sentiment Docker Test/
├── docker-compose.minimal.yml      # Main compose file (profile-based)
├── Dockerfile.frontend             # nginx frontend
├── Dockerfile.nlp.minimal          # NLP service
├── Dockerfile.graph                # Graph service
├── Dockerfile.audio                # Audio service
├── service_nlp.py                  # NLP API endpoints
├── service_graph.py                # Graph API endpoints
├── audio_service.py                # Audio API endpoints
├── nlp.py                          # NLP core (presets, models)
├── nlp_processor.py                # NLP business logic
├── graph_processor.py              # Graph business logic
├── graph_tasks.py                  # Graph algorithms
├── audio_antispoofing.py           # Audio deepfake detection
├── topics.py                       # Theme/topic extraction
├── training_store.py               # Few-shot learning storage
├── excel_utils.py                  # Excel file utilities
├── fetch.py                        # URL content fetching
├── render.py                       # Browser-based rendering
├── adapters.py                     # Platform adapters (Twitter, etc.)
├── guidance_defaults.json          # Default theme configuration
├── index.html                      # Frontend UI
├── nginx.conf                      # nginx configuration
├── requirements-minimal.txt        # NLP dependencies
├── requirements-graph.txt          # Graph dependencies
└── requirements-audio.txt          # Audio dependencies
```

---

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker compose -f docker-compose.minimal.yml --profile core logs -f nlp

# Verify GPU access (if using GPU)
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

### Port Already in Use

```bash
# Find process using port 8080
sudo lsof -i :8080  # Linux/Mac
netstat -ano | findstr :8080  # Windows

# Or change port in docker-compose.minimal.yml
```

### Out of Memory

Reduce memory limit in docker-compose.minimal.yml:
```yaml
deploy:
  resources:
    limits:
      memory: 8G  # Reduce from 16G
```

### Training Data Not Loading

Training data is stored in `./training_data/` volume. Ensure:
1. Directory exists and is writable
2. Training files have URL, Stance/Sentiment, and Themes columns

---

## Technology Stack

### NLP Service
- **Framework**: FastAPI + Uvicorn
- **NLP**: Transformers, PyTorch 2.5.1
- **Models**: HuggingFace Hub (RoBERTa, BART, mDeBERTa)
- **Processing**: BeautifulSoup4, Trafilatura, openpyxl

### Graph Service
- **Framework**: FastAPI + Uvicorn
- **Algorithms**: NetworkX
- **Visualization**: Vis.js

### Audio Service
- **Model**: wav2vec2-large-anti-deepfake (fairseq)
- **Processing**: librosa, soundfile
- **Backend**: PyTorch 2.5.1

### Frontend
- **Server**: nginx (Alpine)
- **Routing**: Reverse proxy to backend services
- **Limits**: 500MB upload, 120s timeout

---

## License

MIT License - See LICENSE file for details

## Credits

- **NLP Models**: HuggingFace Transformers
- **Graph Analytics**: NetworkX
- **Audio Detection**: NII Yamagishi Lab wav2vec2-large-anti-deepfake
- **Frontend**: Vis.js, nginx
