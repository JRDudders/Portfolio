# GPU Setup Guide

## ⚠️ Important: Use the Microservices Architecture

The old `Dockerfile.gpu` is **legacy** and builds the monolithic app.

For the new microservices architecture with GPU support, use:
```bash
docker-compose -f docker-compose.gpu.yml up --build
```

## Prerequisites

1. **NVIDIA GPU** with CUDA support
2. **NVIDIA Drivers** installed (version 525+ for CUDA 12.2)
3. **Docker** with GPU support:
   - **Linux**: Install nvidia-docker2
   - **Windows**: Docker Desktop with WSL2 + NVIDIA GPU support

### Check GPU Availability

```bash
# Check NVIDIA drivers
nvidia-smi

# Should show your GPU and CUDA version
```

### Install Docker GPU Support

**On Linux:**
```bash
# Install nvidia-docker2
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

**On Windows:**
- Install Docker Desktop
- Enable WSL2 backend
- Install NVIDIA drivers for WSL2
- GPU support should work automatically

### Verify Docker GPU Access

```bash
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

Should show your GPU info.

---

## GPU-Accelerated Microservices

### Architecture

```
┌─────────────┐
│  Frontend   │  No GPU
└──────┬──────┘
       │
   ┌───┴───┬───────┬───────┐
   │       │       │       │
   ▼       ▼       ▼       ▼
┌──────┐ ┌──────┐ ┌──────┐
│ NLP  │ │Graph │ │Audio │
│ GPU  │ │ GPU  │ │ GPU  │
│CUDA  │ │RAPIDS│ │CUDA  │
└──────┘ └──────┘ └──────┘
```

### Services with GPU:

1. **NLP Service** (Python 3.12 + CUDA)
   - PyTorch with CUDA 12.1
   - GPU-accelerated transformers
   - Dockerfile: `Dockerfile.nlp.gpu`

2. **Graph Service** (Python 3.11 + RAPIDS)
   - cuGraph for GPU-accelerated graph algorithms
   - cuDF for GPU dataframes
   - 10-100x faster than CPU
   - Dockerfile: `Dockerfile.graph.gpu`

3. **Audio Service** (Python 3.10 + CUDA)
   - PyTorch with CUDA for fairseq
   - GPU-accelerated audio processing
   - Dockerfile: `Dockerfile.audio` (already supports GPU)

---

## Quick Start

### 1. Build and Run with GPU

```bash
# Navigate to project
cd "Sentiment Docker Test"

# Build all GPU-enabled services
docker-compose -f docker-compose.gpu.yml build

# Start all services with GPU support
docker-compose -f docker-compose.gpu.yml up

# Or run in background
docker-compose -f docker-compose.gpu.yml up -d
```

### 2. Verify GPU Usage

```bash
# Check GPU utilization
nvidia-smi

# Should show docker containers using GPU

# Check logs
docker-compose -f docker-compose.gpu.yml logs nlp
docker-compose -f docker-compose.gpu.yml logs graph
docker-compose -f docker-compose.gpu.yml logs audio
```

### 3. Access Services

- **UI**: http://localhost
- **NLP API**: http://localhost:8001/docs
- **Graph API**: http://localhost:8002/docs (with cuGraph!)
- **Audio API**: http://localhost:8003/docs

---

## Configuration

### Assign Different GPUs to Services

Edit `docker-compose.gpu.yml`:

```yaml
nlp:
  environment:
    - CUDA_VISIBLE_DEVICES=0  # Use GPU 0

graph:
  environment:
    - CUDA_VISIBLE_DEVICES=1  # Use GPU 1

audio:
  environment:
    - CUDA_VISIBLE_DEVICES=0  # Use GPU 0
```

### Limit GPU Memory

```yaml
nlp:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
      limits:
        memory: 8G  # Total memory (includes GPU memory)
```

### CPU-Only Mode

If you want to run without GPU:
```bash
# Use regular docker-compose
docker-compose up
```

---

## Development Workflows

### Test GPU Service Locally

**NLP Service:**
```bash
# Requires CUDA installed locally
export SERVICE_PORT=8001
export CUDA_VISIBLE_DEVICES=0
python service_nlp.py
```

**Graph Service:**
```bash
# Requires RAPIDS installed (tricky - use Docker instead)
docker-compose -f docker-compose.gpu.yml up graph
```

**Audio Service:**
```bash
# Requires fairseq + CUDA
export SERVICE_PORT=8003
export CUDA_VISIBLE_DEVICES=0
python audio_service.py
```

### Rebuild Individual Service

```bash
# Rebuild just NLP service
docker-compose -f docker-compose.gpu.yml build nlp

# Restart it
docker-compose -f docker-compose.gpu.yml up -d nlp

# View logs
docker-compose -f docker-compose.gpu.yml logs -f nlp
```

### Scale GPU Services

```bash
# Run 2 NLP instances (load balancing)
docker-compose -f docker-compose.gpu.yml up --scale nlp=2

# Run 3 Graph instances
docker-compose -f docker-compose.gpu.yml up --scale graph=3
```

---

## Performance Comparison

### CPU vs GPU (Approximate)

| Task | CPU Time | GPU Time | Speedup |
|------|----------|----------|---------|
| **NLP - Sentiment Analysis** | 500ms | 50ms | 10x |
| **NLP - Entity Extraction** | 800ms | 100ms | 8x |
| **Graph - Clustering (1000 docs)** | 60s | 2s | 30x |
| **Graph - PageRank** | 45s | 0.5s | 90x |
| **Audio - Deepfake Detection** | 3s | 0.3s | 10x |

### Memory Usage

| Service | CPU Mode | GPU Mode |
|---------|----------|----------|
| NLP | 2GB | 4GB (+ 2GB VRAM) |
| Graph | 2GB | 8GB (+ 4GB VRAM) |
| Audio | 4GB | 6GB (+ 2GB VRAM) |

---

## Troubleshooting

### "docker: Error response from daemon: could not select device driver"

**Solution**: Install nvidia-docker2 or enable GPU support in Docker Desktop

```bash
# Linux
sudo apt-get install nvidia-container-toolkit
sudo systemctl restart docker

# Windows
# Enable GPU support in Docker Desktop settings
```

### "RuntimeError: No CUDA GPUs are available"

**Solution**: Check nvidia-smi and CUDA_VISIBLE_DEVICES

```bash
# Check GPUs
nvidia-smi

# Run container with GPU
docker run --rm --gpus all your-image

# Or in docker-compose (should already have it)
```

### "RAPIDS image not found"

**Solution**: The image tag might not exist. Use a valid version:

```dockerfile
# In Dockerfile.graph.gpu, use an existing version:
FROM rapidsai/rapidsai:24.08-cuda12.2-runtime-ubuntu22.04-py3.11

# Or use latest:
FROM rapidsai/rapidsai:latest
```

Check available tags: https://hub.docker.com/r/rapidsai/rapidsai/tags

### Graph Service Slower Than Expected

**Check cuGraph is being used:**

```python
# In graph_processor.py
import importlib.util
if importlib.util.find_spec("cugraph"):
    import cugraph as nx  # Use GPU
    print("Using cuGraph (GPU)")
else:
    import networkx as nx  # Use CPU
    print("Using NetworkX (CPU)")
```

### Out of Memory

**Reduce batch sizes or scale down models:**

```python
# In service code
MAX_BATCH_SIZE = 8  # Reduce from 32
```

Or increase memory limits in docker-compose.gpu.yml:

```yaml
deploy:
  resources:
    limits:
      memory: 16G  # Increased
```

---

## RAPIDS cuGraph Features

The Graph service with GPU acceleration supports:

- **PageRank**: 50-100x faster
- **Community Detection** (Louvain): 20-40x faster
- **Shortest Path** (Dijkstra): 10-30x faster
- **Connected Components**: 40-80x faster
- **Clustering Coefficient**: 30-60x faster
- **Triangle Counting**: 50-100x faster

### Using cuGraph in Graph Service

```python
# In graph_processor.py
try:
    import cugraph
    import cudf
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

def build_network_graph(texts, threshold=0.5):
    if GPU_AVAILABLE:
        # Use cuGraph (GPU)
        df = cudf.DataFrame({
            'src': sources,
            'dst': destinations,
            'weight': weights
        })
        G = cugraph.Graph()
        G.from_cudf_edgelist(df, source='src', destination='dst', edge_attr='weight')
        return cugraph.pagerank(G)
    else:
        # Fall back to NetworkX (CPU)
        import networkx as nx
        G = nx.Graph()
        # ... standard NetworkX code
```

---

## Legacy Dockerfile.gpu

The old `Dockerfile.gpu` still works for the monolithic app:

```bash
# Build legacy monolithic GPU image
docker build -t cicerowatch-gpu -f Dockerfile.gpu .

# Run it
docker run --gpus all -p 8080:8080 cicerowatch-gpu
```

**But we recommend using the new microservices with `docker-compose.gpu.yml` instead!**

---

## Recommendations

### For Development
- Use regular `docker-compose.yml` (CPU mode)
- Faster builds, easier debugging
- GPU not needed for development

### For Testing
- Use `docker-compose.gpu.yml`
- Test GPU acceleration
- Verify performance gains

### For Production
- Use `docker-compose.gpu.yml`
- Deploy to GPU-enabled servers
- Configure auto-scaling based on GPU utilization
- Use Kubernetes with GPU node pools

---

## Summary

**Old Way** (monolithic):
```bash
docker build -t app -f Dockerfile.gpu .  # ❌ Wrong RAPIDS version
docker run --gpus all -p 8080:8080 app
```

**New Way** (microservices):
```bash
docker-compose -f docker-compose.gpu.yml up  # ✅ Correct
```

**Benefits**:
- ✅ Each service gets GPU independently
- ✅ Can assign different GPUs to services
- ✅ Scale GPU services separately
- ✅ Correct RAPIDS version (24.08)
- ✅ Python 3.12 for NLP, 3.11 for RAPIDS, 3.10 for Audio

**Access**: http://localhost (all services with GPU acceleration!)
