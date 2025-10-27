# Running Multiple Python Versions in Docker

This guide explains how to run the CiceroWatch application with multiple Python environments for audio deepfake detection.

## The Problem

- **Main app**: Works best with Python 3.12
- **fairseq (audio detection)**: Requires Python 3.10
- **Solution**: Run two separate services with docker-compose

## Architecture

```
┌─────────────────────────────────────┐
│  Main App (Python 3.12)             │
│  - FastAPI                          │
│  - NLP, Graph Analytics             │
│  - Web UI                           │
│  Port: 8080                         │
└────────────┬────────────────────────┘
             │ HTTP
             │
┌────────────▼────────────────────────┐
│  Audio Service (Python 3.10)        │
│  - fairseq + wav2vec                │
│  - Audio deepfake detection         │
│  Port: 8081                         │
└─────────────────────────────────────┘
```

## Quick Start

### 1. Build and Run with Docker Compose

```bash
# Build both services
docker-compose build

# Start both services
docker-compose up

# Or run in background
docker-compose up -d
```

### 2. Access the Application

- **Main App**: http://localhost:8080
- **Audio Service**: http://localhost:8081/docs (Swagger UI)

### 3. Test Audio Detection

Upload an audio file through the web UI, or test the service directly:

```bash
# Check audio service health
curl http://localhost:8081/health

# Test audio analysis
curl -X POST http://localhost:8081/analyze \
  -F "file=@your_audio.wav"
```

## How It Works

### Main App (app.py)

The main app uses `audio_client.py` to communicate with the audio service:

```python
from audio_client import analyze_audio

# This automatically uses remote service if available,
# falls back to local processing if not
prediction, confidence, score = analyze_audio(audio_path)
```

### Audio Service (audio_service.py)

Runs independently with Python 3.10 + fairseq:

```bash
# Inside the audio container
python audio_service.py
```

### Communication

- Main app sends HTTP POST to `http://audio:8081/analyze`
- Audio service processes with fairseq + wav2vec model
- Returns JSON with prediction results
- Falls back to local processing if service unavailable

## Development Workflow

### Local Development (Without Docker)

For local development, you can run the audio service separately:

**Terminal 1 - Audio Service (Python 3.10)**:
```bash
pyenv local 3.10.13
pip install -r requirements-audio.txt
python audio_service.py
```

**Terminal 2 - Main App (Python 3.12)**:
```bash
export AUDIO_SERVICE_URL=http://localhost:8081
python run_local.py
```

### Docker Compose Commands

```bash
# Build services
docker-compose build

# Start services
docker-compose up

# Start in background
docker-compose up -d

# View logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f audio

# Stop services
docker-compose down

# Rebuild and restart
docker-compose up --build

# Scale audio service (multiple workers)
docker-compose up --scale audio=3
```

## Configuration

### Environment Variables

Set in `docker-compose.yml` or `.env` file:

```bash
# Main App
AUDIO_SERVICE_URL=http://audio:8081

# Audio Service
CUDA_VISIBLE_DEVICES=0  # GPU device (if available)
HUGGINGFACE_API_KEY=hf_...  # For model downloads
```

### Volume Mounts

```yaml
volumes:
  - ./temp:/app/temp          # Shared temp files
  - ./models:/app/models      # Cache downloaded models
```

## Alternative: Single Image with pyenv

If you prefer a single Docker image with multiple Python versions:

```dockerfile
FROM ubuntu:22.04

# Install pyenv
RUN apt-get update && apt-get install -y \
    git curl build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget \
    llvm libncurses5-dev libncursesw5-dev \
    xz-utils tk-dev libffi-dev liblzma-dev

RUN curl https://pyenv.run | bash

# Install Python 3.10 and 3.12
RUN pyenv install 3.10.13
RUN pyenv install 3.12.0

# Set Python 3.12 as default
RUN pyenv global 3.12.0

# Install main app
RUN pip install -r requirements.txt

# Install audio deps in Python 3.10
RUN pyenv shell 3.10.13 && \
    pip install -r requirements-audio.txt

# Run audio service with Python 3.10 in background
CMD pyenv shell 3.10.13 && \
    python audio_service.py & \
    pyenv shell 3.12.0 && \
    python run_local.py
```

**Note**: The docker-compose approach is cleaner and more maintainable.

## Troubleshooting

### Audio Service Not Starting

Check logs:
```bash
docker-compose logs audio
```

Common issues:
- fairseq install failed → Check Python version is 3.10
- Out of memory → Reduce batch size or use CPU
- Model download failed → Check internet, set HUGGINGFACE_API_KEY

### Main App Can't Connect to Audio Service

Check network:
```bash
docker-compose exec app ping audio
```

Check audio service is running:
```bash
curl http://localhost:8081/health
```

### Port Conflicts

Change ports in `docker-compose.yml`:
```yaml
services:
  app:
    ports:
      - "9080:8080"  # Changed from 8080
  audio:
    ports:
      - "9081:8081"  # Changed from 8081
```

## Performance Optimization

### Use Pre-built Models

Download models once and share via volume:

```bash
# Download models locally
python -c "import audio_antispoofing; audio_antispoofing.download_models()"

# Models cached in ~/.cache/huggingface/
# Mount this in docker-compose.yml:
volumes:
  - ~/.cache/huggingface:/root/.cache/huggingface
```

### GPU Support

Enable GPU in docker-compose.yml:

```yaml
services:
  audio:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

Requires:
- NVIDIA GPU
- nvidia-docker installed
- CUDA-compatible PyTorch

## Production Deployment

For production, consider:

1. **Kubernetes**: Deploy as microservices with horizontal scaling
2. **Load Balancer**: Multiple audio service instances
3. **Model Caching**: Persistent volumes for model cache
4. **Monitoring**: Health checks, metrics, logging
5. **Security**: API authentication, rate limiting

## Summary

**Docker Compose Approach (Recommended)**:
- ✅ Clean separation of concerns
- ✅ Easy to scale services independently
- ✅ Standard microservices pattern
- ✅ Each service has its own Python version
- ✅ Simple to maintain and debug

**Single Image with pyenv**:
- ⚠️ More complex Dockerfile
- ⚠️ Harder to debug
- ⚠️ Can't scale services independently
- ✅ Single image to deploy
- ✅ No network communication overhead

Choose docker-compose for flexibility, pyenv for simplicity.
