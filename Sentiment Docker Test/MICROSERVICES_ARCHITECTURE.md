# CiceroWatch Microservices Architecture

## Overview

CiceroWatch is now built as a microservices architecture, with each UI tab powered by an independent, scalable service.

```
                    ┌─────────────────────────────┐
                    │   Frontend (Nginx)          │
                    │   Port: 80                  │
                    │   Static files + Routing    │
                    └────────────┬────────────────┘
                                 │
            ┌────────────────────┼────────────────────┐
            │                    │                    │
            ▼                    ▼                    ▼
┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐
│  NLP Service      │  │  Graph Service    │  │  Audio Service    │
│  Python 3.12      │  │  Python 3.12      │  │  Python 3.10      │
│  Port: 8001       │  │  Port: 8002       │  │  Port: 8003       │
│                   │  │                   │  │                   │
│  - Sentiment      │  │  - Similarity     │  │  - Deepfake       │
│  - Entities       │  │  - Clustering     │  │    Detection      │
│  - Topics         │  │  - Networks       │  │  - fairseq        │
│  - URL scraping   │  │  - Communities    │  │  - wav2vec        │
└───────────────────┘  └───────────────────┘  └───────────────────┘
```

## Services

### 1. Frontend (Nginx)

**Purpose**: Serves static UI and routes API requests to backend services

**Technology**: Nginx Alpine

**Port**: 80

**Routes**:
- `/` → Static HTML/CSS/JS
- `/api/nlp/*` → NLP Service (port 8001)
- `/api/graph/*` → Graph Service (port 8002)
- `/api/audio/*` → Audio Service (port 8003)

**Files**:
- `Dockerfile.frontend`
- `nginx.conf`
- `static/index.html`

### 2. NLP Service

**Purpose**: Natural language processing tasks

**Technology**: Python 3.12, FastAPI, transformers

**Port**: 8001

**Endpoints**:
- `GET /health` - Health check
- `POST /analyze/text` - Analyze text
- `POST /analyze/url` - Scrape and analyze URL
- `POST /analyze/file` - Analyze uploaded file
- `POST /sentiment` - Sentiment analysis only
- `POST /entities` - Entity extraction only
- `POST /topics` - Topic modeling only

**Features**:
- Sentiment analysis
- Named entity recognition
- Topic modeling with BERTopic
- URL scraping and analysis
- Multi-language support

**Files**:
- `Dockerfile.nlp`
- `service_nlp.py`
- `nlp_processor.py`
- `requirements-nlp.txt`

### 3. Graph Analytics Service

**Purpose**: Graph-based text analysis

**Technology**: Python 3.12, FastAPI, NetworkX

**Port**: 8002

**Endpoints**:
- `GET /health` - Health check
- `POST /similarity` - Calculate text similarity matrix
- `POST /cluster` - Cluster texts (KMeans, Hierarchical, DBSCAN)
- `POST /network` - Build network graph
- `POST /communities` - Detect communities

**Features**:
- Text similarity (cosine, jaccard)
- Clustering algorithms
- Network graph generation
- Community detection

**Files**:
- `Dockerfile.graph`
- `service_graph.py`
- `graph_processor.py`
- `requirements-graph.txt`

### 4. Audio Service

**Purpose**: Audio deepfake detection

**Technology**: Python 3.10, FastAPI, fairseq, wav2vec

**Port**: 8003

**Endpoints**:
- `GET /health` - Health check
- `POST /analyze` - Analyze audio file
- `POST /download-models` - Pre-download models

**Features**:
- Audio deepfake detection
- fairseq + wav2vec2 model
- Supports WAV, FLAC, MP3

**Files**:
- `Dockerfile.audio`
- `audio_service.py`
- `audio_antispoofing.py`
- `requirements-audio.txt`

## Quick Start

### Build and Run All Services

```bash
# Navigate to project directory
cd "Sentiment Docker Test"

# Build all services
docker-compose build

# Start all services
docker-compose up

# Or run in background
docker-compose up -d
```

### Access the Application

- **Main UI**: http://localhost
- **NLP API Docs**: http://localhost:8001/docs
- **Graph API Docs**: http://localhost:8002/docs
- **Audio API Docs**: http://localhost:8003/docs

## Development Workflow

### Work on Individual Services

Each service is independent and can be developed separately:

**NLP Service:**
```bash
cd "Sentiment Docker Test"
docker-compose up nlp
# Or rebuild after changes:
docker-compose up --build nlp
```

**Graph Service:**
```bash
docker-compose up graph
# Or rebuild:
docker-compose up --build graph
```

**Audio Service:**
```bash
docker-compose up audio
# Or rebuild:
docker-compose up --build audio
```

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f nlp
docker-compose logs -f graph
docker-compose logs -f audio
docker-compose logs -f frontend
```

### Restart Individual Service

```bash
# Restart NLP service
docker-compose restart nlp

# Rebuild and restart
docker-compose up --build -d nlp
```

### Scale Services

```bash
# Run 3 instances of NLP service
docker-compose up --scale nlp=3

# Scale graph service
docker-compose up --scale graph=2
```

## Service Communication

### From Frontend JavaScript

Services are accessed via `/api/{service}/` prefix:

```javascript
// NLP Service
fetch('/api/nlp/analyze/text', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        text: "Sample text",
        tasks: ["sentiment", "entities"]
    })
});

// Graph Service
fetch('/api/graph/similarity', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        texts: ["text1", "text2"],
        method: "cosine"
    })
});

// Audio Service
const formData = new FormData();
formData.append('file', audioFile);
fetch('/api/audio/analyze', {
    method: 'POST',
    body: formData
});
```

### Between Services (Optional)

Services can call each other using service names:

```python
import requests

# From NLP service, call Graph service
response = requests.post(
    "http://graph:8002/similarity",
    json={"texts": texts, "method": "cosine"}
)
```

## Configuration

### Environment Variables

Set in `docker-compose.yml` or `.env` file:

```bash
# Service ports
NLP_PORT=8001
GRAPH_PORT=8002
AUDIO_PORT=8003

# Audio service specific
CUDA_VISIBLE_DEVICES=0
HUGGINGFACE_API_KEY=hf_...

# Resource limits
NLP_MEMORY_LIMIT=4G
GRAPH_MEMORY_LIMIT=2G
AUDIO_MEMORY_LIMIT=6G
```

### Volume Mounts

```yaml
volumes:
  - ./temp:/app/temp          # Shared temporary files
  - ./models:/app/models      # Audio models cache
  - ./static:/usr/share/nginx/html  # Frontend files
```

## Adding New Services

### Step 1: Create Service File

```python
# service_newfeature.py
from fastapi import FastAPI
import uvicorn
import os

app = FastAPI(title="New Feature Service")

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/process")
async def process(data: dict):
    # Your logic here
    return {"result": "processed"}

if __name__ == "__main__":
    port = int(os.getenv("SERVICE_PORT", 8004))
    uvicorn.run(app, host="0.0.0.0", port=port)
```

### Step 2: Create Dockerfile

```dockerfile
# Dockerfile.newfeature
FROM python:3.12-slim

WORKDIR /app

COPY requirements-newfeature.txt .
RUN pip install -r requirements-newfeature.txt

COPY service_newfeature.py .

EXPOSE 8004

CMD ["python", "service_newfeature.py"]
```

### Step 3: Add to docker-compose.yml

```yaml
newfeature:
  build:
    context: .
    dockerfile: Dockerfile.newfeature
  ports:
    - "8004:8004"
  environment:
    - SERVICE_PORT=8004
  networks:
    - cicerowatch
```

### Step 4: Add Route to nginx.conf

```nginx
upstream newfeature_service {
    server newfeature:8004;
}

location /api/newfeature/ {
    proxy_pass http://newfeature_service/;
    # ... proxy settings
}
```

### Step 5: Update Frontend

Add new tab in `index.html` and JavaScript to call `/api/newfeature/`.

## Monitoring and Health Checks

### Health Check All Services

```bash
# Using curl
curl http://localhost/health        # Frontend
curl http://localhost:8001/health   # NLP
curl http://localhost:8002/health   # Graph
curl http://localhost:8003/health   # Audio

# Using docker-compose
docker-compose ps
```

### View Resource Usage

```bash
docker stats
```

### Check Service Logs

```bash
# Recent logs
docker-compose logs --tail=100 nlp

# Follow logs in real-time
docker-compose logs -f graph

# All services
docker-compose logs -f
```

## Production Deployment

### Build for Production

```bash
# Build with production settings
docker-compose -f docker-compose.yml -f docker-compose.prod.yml build

# Use BuildKit for faster builds
DOCKER_BUILDKIT=1 docker-compose build
```

### Security Best Practices

1. **Use secrets for API keys**:
   ```yaml
   secrets:
     - huggingface_token
   ```

2. **Limit resource usage** (already configured):
   ```yaml
   deploy:
     resources:
       limits:
         memory: 4G
         cpus: '2.0'
   ```

3. **Enable HTTPS** (add to nginx.conf):
   ```nginx
   server {
       listen 443 ssl;
       ssl_certificate /path/to/cert.pem;
       ssl_certificate_key /path/to/key.pem;
   }
   ```

4. **Add authentication** to sensitive endpoints

### Kubernetes Deployment (Optional)

For large-scale deployments, convert to Kubernetes:

```bash
# Generate Kubernetes manifests
kompose convert

# Deploy to cluster
kubectl apply -f *-deployment.yaml
kubectl apply -f *-service.yaml
```

## Troubleshooting

### Service Won't Start

```bash
# Check logs
docker-compose logs servicename

# Common issues:
# - Port already in use → Change port in docker-compose.yml
# - Dependency missing → Check requirements-*.txt
# - Build failed → Check Dockerfile
```

### Can't Connect to Service

```bash
# Check if service is running
docker-compose ps

# Test network connectivity
docker-compose exec frontend ping nlp
docker-compose exec nlp ping graph

# Check nginx routing
docker-compose logs frontend | grep error
```

### Service Crashes/Restarts

```bash
# Check resource limits
docker stats

# Increase memory limit in docker-compose.yml:
deploy:
  resources:
    limits:
      memory: 8G  # Increased from 4G
```

### Performance Issues

1. **Scale horizontally**:
   ```bash
   docker-compose up --scale nlp=3
   ```

2. **Add load balancer** to nginx.conf:
   ```nginx
   upstream nlp_service {
       least_conn;  # Load balancing method
       server nlp_1:8001;
       server nlp_2:8001;
       server nlp_3:8001;
   }
   ```

3. **Optimize models**:
   - Use smaller models
   - Enable model quantization
   - Cache results

## Benefits of Microservices Architecture

✅ **Independent Development**: Edit each service without affecting others
✅ **Independent Scaling**: Scale NLP, Graph, Audio services separately
✅ **Technology Freedom**: Use different Python versions per service
✅ **Fault Isolation**: If one service fails, others keep running
✅ **Easy Deployment**: Deploy updates to individual services
✅ **Clear Boundaries**: Each service has a single responsibility
✅ **Team Organization**: Different teams can own different services

## File Structure

```
Sentiment Docker Test/
├── docker-compose.yml           # Orchestration
├── nginx.conf                    # Frontend routing
│
├── Dockerfile.frontend           # Frontend container
├── Dockerfile.nlp                # NLP container
├── Dockerfile.graph              # Graph container
├── Dockerfile.audio              # Audio container
│
├── service_nlp.py                # NLP service
├── service_graph.py              # Graph service
├── audio_service.py              # Audio service
│
├── nlp_processor.py              # NLP business logic
├── graph_processor.py            # Graph business logic
├── audio_antispoofing.py         # Audio business logic
│
├── requirements-nlp.txt          # NLP dependencies
├── requirements-graph.txt        # Graph dependencies
├── requirements-audio.txt        # Audio dependencies
│
└── static/
    └── index.html                # Frontend UI
```

## Next Steps

1. **Implement processor modules**:
   - `nlp_processor.py` - NLP logic
   - `graph_processor.py` - Graph logic

2. **Update frontend UI**:
   - Modify `static/index.html`
   - Add JavaScript to call new API endpoints

3. **Add authentication**:
   - API keys
   - JWT tokens
   - OAuth2

4. **Add monitoring**:
   - Prometheus metrics
   - Grafana dashboards
   - Log aggregation

5. **Add CI/CD**:
   - Automated testing
   - Docker Hub / registry
   - Deployment pipelines

## Summary

**Architecture**: Microservices with Nginx reverse proxy
**Services**: Frontend (Nginx), NLP (3.12), Graph (3.12), Audio (3.10)
**Communication**: HTTP/REST via Nginx routing
**Deployment**: Docker Compose (dev) or Kubernetes (prod)
**Scalability**: Horizontal scaling per service
**Maintainability**: Independent services, clear boundaries
