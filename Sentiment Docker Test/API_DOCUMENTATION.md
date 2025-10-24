# Sentiment Analysis & Graph Analytics API Documentation

This API provides sentiment analysis, NLP tasks, and graph analytics capabilities through a FastAPI service.

## Base URL
```
http://localhost:8080
```

## Health Check

### GET `/healthz`
Check service health and available presets.

**Response:**
```json
{
  "ok": true,
  "presets": ["sentiment-twitter", "sentiment-sst2", "zeroshot-bart", ...],
  "playwright": true
}
```

---

## Sentiment Analysis & NLP Endpoints

### POST `/predict/file`
Process a file (CSV, JSON, or HTML) for sentiment analysis or other NLP tasks.

**Parameters:**
- `file` (UploadFile): The file to process
- `preset` (str, optional): Preset name (e.g., "sentiment-twitter", "zeroshot-bart")
- `labels` (str, optional): Comma-separated labels for zero-shot classification
- `include_stopwords` (bool, optional): Default: false

**Example:**
```bash
curl -X POST "http://localhost:8080/predict/file?preset=sentiment-twitter" \
  -F "file=@tweets.csv"
```

**Response:** Downloads `predictions.json`

---

### POST `/predict/url`
Fetch a URL and analyze its content.

**Request Body:**
```json
{
  "url": "https://example.com",
  "preset": "sentiment-twitter",
  "render": false,
  "renderer": "auto",
  "wait_selector": null,
  "scroll_passes": 8,
  "render_timeout_ms": 25000
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

### POST `/predict/batch`
**NEW!** Efficiently process multiple texts in batch.

**Request Body:**
```json
{
  "texts": [
    "I love this product!",
    "This is terrible.",
    "Pretty good overall."
  ],
  "preset": "sentiment-twitter",
  "preprocess": true
}
```

**Example:**
```bash
curl -X POST "http://localhost:8080/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Great service!", "Not happy with this."],
    "preset": "sentiment-twitter"
  }'
```

**Response:**
```json
{
  "preset": "sentiment-twitter",
  "count": 2,
  "results": [
    {"labels": ["positive", "negative", "neutral"], "scores": [0.92, 0.05, 0.03]},
    {"labels": ["negative", "neutral", "positive"], "scores": [0.78, 0.15, 0.07]}
  ]
}
```

---

## Graph Analytics Endpoints

### POST `/graph/load`
**NEW!** Load and validate a graph file, returning basic statistics.

**Parameters:**
- `file` (UploadFile): Edge list in CSV or JSON format

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
    {"src": "Alice", "dst": "Bob", "weight": 1.0},
    {"src": "Bob", "dst": "Charlie", "weight": 1.5}
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

### POST `/graph/degrees`
**NEW!** Compute in-degree, out-degree, and total degree for all nodes.

**Parameters:**
- `file` (UploadFile): Edge list CSV/JSON

**Example:**
```bash
curl -X POST "http://localhost:8080/graph/degrees" \
  -F "file=@edges.csv"
```

**Response:** Downloads `degrees.json`
```json
{
  "n_nodes": 100,
  "n_edges": 250,
  "degrees": [
    {"node": "Alice", "out_degree": 10, "in_degree": 5, "degree": 15},
    {"node": "Bob", "out_degree": 8, "in_degree": 7, "degree": 15},
    ...
  ]
}
```

---

### POST `/graph/pagerank`
**NEW!** Compute PageRank scores for all nodes.

**Parameters:**
- `file` (UploadFile): Edge list CSV/JSON
- `alpha` (float, optional): Damping factor (default: 0.85)
- `iters` (int, optional): Maximum iterations (default: 40)
- `tol` (float, optional): Convergence tolerance (default: 1e-6)

**Example:**
```bash
curl -X POST "http://localhost:8080/graph/pagerank?alpha=0.85&iters=50" \
  -F "file=@edges.csv"
```

**Response:** Downloads `pagerank.json`
```json
{
  "n_nodes": 100,
  "n_edges": 250,
  "pagerank": [
    {"node": "Alice", "pr": 0.045},
    {"node": "Bob", "pr": 0.032},
    ...
  ],
  "parameters": {"alpha": 0.85, "iters": 50, "tol": 1e-6}
}
```

---

### POST `/graph/bfs`
**NEW!** Compute breadth-first search distances from a source node.

**Parameters:**
- `file` (UploadFile): Edge list CSV/JSON
- `source` (str, required): Source node ID

**Example:**
```bash
curl -X POST "http://localhost:8080/graph/bfs?source=Alice" \
  -F "file=@edges.csv"
```

**Response:** Downloads `bfs.json`
```json
{
  "n_nodes": 100,
  "n_edges": 250,
  "source_node": "Alice",
  "distances": [
    {"node": "Alice", "distance": 0},
    {"node": "Bob", "distance": 1},
    {"node": "Charlie", "distance": 2},
    {"node": "Disconnected", "distance": -1},
    ...
  ]
}
```

---

### POST `/graph/triangles`
**NEW!** Count triangles in an undirected graph.

**Parameters:**
- `file` (UploadFile): Edge list CSV/JSON (undirected graph)
- `max_nodes` (int, optional): Skip if graph has more nodes (default: 20000)

**Example:**
```bash
curl -X POST "http://localhost:8080/graph/triangles?max_nodes=10000" \
  -F "file=@edges.csv"
```

**Response:** Downloads `triangles.json`
```json
{
  "n_nodes": 100,
  "n_edges": 250,
  "triangles": 42
}
```

---

### POST `/graph/metrics`
Compute multiple graph metrics in one call (original endpoint, now enhanced).

**Parameters:**
- `file` (UploadFile): Edge list CSV/JSON
- `tasks` (str): Comma-separated list (e.g., "degrees,pagerank,bfs,triangles")
- `bfs_source` (str, optional): Required if "bfs" is in tasks
- `pagerank_alpha` (float, optional): Default: 0.85
- `pagerank_iters` (int, optional): Default: 40
- `pagerank_tol` (float, optional): Default: 1e-6
- `triangles_limit` (int, optional): Default: 20000

**Example:**
```bash
curl -X POST "http://localhost:8080/graph/metrics?tasks=degrees,pagerank&pagerank_alpha=0.9" \
  -F "file=@edges.csv"
```

**Response:** Downloads `graph-metrics.json`
```json
{
  "n_nodes": 100,
  "n_edges": 250,
  "degrees": [...],
  "pagerank": [...]
}
```

---

## Available NLP Presets

### Sentiment Analysis
- `sentiment-twitter`: Twitter-specific sentiment (cardiffnlp/twitter-roberta-base-sentiment-latest)
- `sentiment-sst2`: SST-2 sentiment model

### Zero-Shot Classification
- `zeroshot-bart`: Facebook BART MNLI model
- `zeroshot-mdeberta`: Multilingual mDeBERTa model

### Named Entity Recognition (NER)
- `ner-conll`: BERT-base NER (CoNLL)
- `ner-bertbase`: BERT-base NER
- `spacy-ner`: spaCy NER

### Other NLP Tasks
- `spacy-posdep`: spaCy POS tagging and dependency parsing
- `spacy-sents`: spaCy sentence segmentation
- `stanza-posdep`: Stanza POS/DEP
- `sbert-embed`: Sentence embeddings
- `bertopic`: Topic modeling with BERTopic
- `topics-nmf`: NMF topic modeling
- `topics-kmeans`: K-means topic clustering

---

## Optimizations

### What's New
1. **Removed duplicate model loading** in main app.py - uses global PIPE for efficiency
2. **Individual graph endpoints** for granular control (degrees, pagerank, bfs, triangles)
3. **Batch sentiment analysis** endpoint for processing multiple texts efficiently
4. **Graph validation** endpoint to check graph structure before analysis
5. **Comprehensive error handling** with detailed error messages

### Performance Tips
- Use `/predict/batch` for multiple texts instead of individual calls
- For large graphs (>10k nodes), use individual endpoints instead of `/graph/metrics`
- PageRank convergence: adjust `tol` and `iters` based on graph size
- Triangle counting skips graphs >20k nodes by default (configurable)

---

## Error Handling

All endpoints return structured error responses:

```json
{
  "detail": "Error description here"
}
```

Common status codes:
- `400`: Bad request (invalid input, file format error)
- `404`: Resource not found (e.g., BFS source node doesn't exist)
- `500`: Server error (model loading failure, computation error)

---

## Examples

### Complete Workflow: Sentiment Analysis

```bash
# 1. Check service health
curl http://localhost:8080/healthz

# 2. Batch analyze texts
curl -X POST "http://localhost:8080/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "This API is amazing!",
      "The documentation could be better.",
      "Neutral statement here."
    ],
    "preset": "sentiment-twitter"
  }'

# 3. Process CSV file
curl -X POST "http://localhost:8080/predict/file?preset=sentiment-twitter" \
  -F "file=@customer_reviews.csv" \
  -o results.json
```

### Complete Workflow: Graph Analytics

```bash
# 1. Validate graph
curl -X POST "http://localhost:8080/graph/load" \
  -F "file=@social_network.csv"

# 2. Compute degrees
curl -X POST "http://localhost:8080/graph/degrees" \
  -F "file=@social_network.csv" \
  -o degrees.json

# 3. Compute PageRank
curl -X POST "http://localhost:8080/graph/pagerank?alpha=0.85" \
  -F "file=@social_network.csv" \
  -o pagerank.json

# 4. BFS from specific node
curl -X POST "http://localhost:8080/graph/bfs?source=Alice" \
  -F "file=@social_network.csv" \
  -o bfs_distances.json

# 5. All metrics at once
curl -X POST "http://localhost:8080/graph/metrics?tasks=degrees,pagerank,triangles" \
  -F "file=@social_network.csv" \
  -o all_metrics.json
```

---

## Docker Deployment

```bash
# Build
docker build -f Dockerfile.cpu -t sentiment-api .

# Run
docker run -p 8080:8080 sentiment-api

# Test
curl http://localhost:8080/healthz
```

---

## License & Contact

For issues or questions, please contact the repository maintainer.
