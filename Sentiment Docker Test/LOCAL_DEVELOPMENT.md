# Local Development Guide

## Overview

You have **three ways** to run CiceroWatch:

1. **Docker Compose** (microservices) - Full production-like environment
2. **Local Microservices** (Python) - All services running locally
3. **Monolithic Mode** (Python) - Original app.py (legacy)

## Option 1: Docker Compose (Recommended)

**Best for**: Testing the full system, production-like environment

```bash
# Start all services
docker-compose up

# Or in background
docker-compose up -d

# Rebuild after changes
docker-compose up --build

# Work on specific service
docker-compose up nlp
```

**Access**:
- UI: http://localhost
- NLP API: http://localhost:8001/docs
- Graph API: http://localhost:8002/docs
- Audio API: http://localhost:8003/docs

**Pros**:
- ✅ Matches production environment
- ✅ All services isolated
- ✅ Easy to test full system
- ✅ Different Python versions work

**Cons**:
- ⚠️ Slower startup (container build)
- ⚠️ Requires Docker installed

---

## Option 2: Local Microservices (Fast Development)

**Best for**: Developing individual services, fast iteration

### Quick Start

```bash
# Install all dependencies
pip install -r requirements-nlp.txt
pip install -r requirements-graph.txt
pip install -r requirements-audio.txt  # Requires Python 3.10

# Run all services
python run_local_microservices.py
```

This starts:
- **NLP Service**: http://localhost:8001
- **Graph Service**: http://localhost:8002
- **Audio Service**: http://localhost:8003

### Serve Frontend Separately

```bash
# In another terminal
cd static
python -m http.server 8080
```

Then access: http://localhost:8080

**Update your frontend to call services**:
```javascript
// Instead of /api/nlp/..., use direct ports:
fetch('http://localhost:8001/analyze/text', {...})
fetch('http://localhost:8002/similarity', {...})
fetch('http://localhost:8003/analyze', {...})
```

### Run Services Individually

**NLP Service only**:
```bash
export SERVICE_PORT=8001
python service_nlp.py
```

**Graph Service only**:
```bash
export SERVICE_PORT=8002
python service_graph.py
```

**Audio Service only** (requires Python 3.10):
```bash
# Use pyenv or conda to switch to Python 3.10
pyenv local 3.10.13
export SERVICE_PORT=8003
python audio_service.py
```

**Pros**:
- ✅ Fast startup
- ✅ Easy debugging
- ✅ Direct code changes (no rebuild)
- ✅ Use your IDE/debugger

**Cons**:
- ⚠️ Need to manage multiple terminals
- ⚠️ Audio service needs Python 3.10
- ⚠️ No Nginx routing (direct port calls)

---

## Option 3: Monolithic Mode (Legacy)

**Best for**: Quick testing of NLP/Graph features only (no audio)

```bash
python run_local.py
```

This runs the original `app.py` on port 8080.

**Access**: http://localhost:8080

**Note**: This is the **old monolithic architecture**. It doesn't include the microservices features and Audio service requires fairseq which may not work with Python 3.12.

**Pros**:
- ✅ Simplest to run
- ✅ One command
- ✅ Good for quick NLP testing

**Cons**:
- ⚠️ Doesn't use microservices architecture
- ⚠️ Audio detection may not work (fairseq issues)
- ⚠️ Can't edit services independently

---

## Recommended Workflows

### For NLP Development

```bash
# Terminal 1: Run NLP service
export SERVICE_PORT=8001
python service_nlp.py

# Terminal 2: Test it
curl -X POST http://localhost:8001/sentiment \
  -H "Content-Type: application/json" \
  -d '{"text": "This is great!"}'
```

### For Graph Development

```bash
# Terminal 1: Run Graph service
export SERVICE_PORT=8002
python service_graph.py

# Terminal 2: Test it
curl -X POST http://localhost:8002/similarity \
  -H "Content-Type: application/json" \
  -d '{"texts": ["hello world", "hello there"], "method": "cosine"}'
```

### For Audio Development

```bash
# Terminal 1: Switch to Python 3.10 and run
pyenv local 3.10.13
export SERVICE_PORT=8003
python audio_service.py

# Terminal 2: Test it
curl -X POST http://localhost:8003/analyze \
  -F "file=@test_audio.wav"
```

### For Full System Testing

```bash
# Use Docker Compose
docker-compose up

# Or if you want to see changes without rebuild:
docker-compose up -d
docker-compose logs -f nlp
# Make changes to service_nlp.py
docker-compose restart nlp
```

---

## Environment Setup

### Install All Dependencies

```bash
# Core dependencies
pip install fastapi uvicorn python-multipart pydantic requests

# NLP dependencies
pip install -r requirements-nlp.txt

# Graph dependencies
pip install -r requirements-graph.txt

# Audio dependencies (Python 3.10 required)
# Switch to Python 3.10 first
pyenv install 3.10.13
pyenv local 3.10.13
pip install -r requirements-audio.txt
```

### Python Version Management

**Using pyenv**:
```bash
# Install Python versions
pyenv install 3.12.0
pyenv install 3.10.13

# For NLP and Graph (Python 3.12)
pyenv local 3.12.0
pip install -r requirements-nlp.txt
pip install -r requirements-graph.txt

# For Audio (Python 3.10)
pyenv local 3.10.13
pip install -r requirements-audio.txt
```

**Using conda**:
```bash
# Environment for NLP/Graph
conda create -n cicerowatch python=3.12
conda activate cicerowatch
pip install -r requirements-nlp.txt
pip install -r requirements-graph.txt

# Environment for Audio
conda create -n cicerowatch-audio python=3.10
conda activate cicerowatch-audio
pip install -r requirements-audio.txt
```

---

## Debugging

### Debug with IDE (VS Code)

**NLP Service** (`.vscode/launch.json`):
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "NLP Service",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/Sentiment Docker Test/service_nlp.py",
            "console": "integratedTerminal",
            "env": {
                "SERVICE_PORT": "8001"
            }
        }
    ]
}
```

### Debug with pdb

```python
# Add to service code
import pdb; pdb.set_trace()
```

### View Logs

**Docker Compose**:
```bash
docker-compose logs -f nlp
docker-compose logs -f graph
docker-compose logs -f audio
```

**Local**:
```bash
# Services print to stdout
python service_nlp.py  # Watch terminal
```

---

## Testing Services

### Using curl

```bash
# NLP Service
curl http://localhost:8001/health

curl -X POST http://localhost:8001/analyze/text \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a test", "tasks": ["sentiment", "entities"]}'

# Graph Service
curl http://localhost:8002/health

curl -X POST http://localhost:8002/similarity \
  -H "Content-Type: application/json" \
  -d '{"texts": ["hello world", "goodbye world"], "method": "cosine"}'

# Audio Service
curl http://localhost:8003/health

curl -X POST http://localhost:8003/analyze \
  -F "file=@test.wav"
```

### Using Python requests

```python
import requests

# NLP Service
response = requests.post(
    "http://localhost:8001/sentiment",
    json={"text": "This is amazing!"}
)
print(response.json())

# Graph Service
response = requests.post(
    "http://localhost:8002/cluster",
    json={
        "texts": ["text1", "text2", "text3"],
        "num_clusters": 2,
        "method": "kmeans"
    }
)
print(response.json())

# Audio Service
with open("test.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8003/analyze",
        files={"file": f}
    )
print(response.json())
```

### Using Swagger UI

Each service has interactive API docs:
- NLP: http://localhost:8001/docs
- Graph: http://localhost:8002/docs
- Audio: http://localhost:8003/docs

---

## Port Reference

| Service | Port | URL |
|---------|------|-----|
| Frontend (Nginx) | 80 | http://localhost |
| NLP Service | 8001 | http://localhost:8001 |
| Graph Service | 8002 | http://localhost:8002 |
| Audio Service | 8003 | http://localhost:8003 |
| Monolithic App | 8080 | http://localhost:8080 |

---

## Common Issues

### "Port already in use"

```bash
# Find process using port
lsof -i :8001  # On macOS/Linux
netstat -ano | findstr :8001  # On Windows

# Kill process
kill -9 <PID>  # macOS/Linux
taskkill /PID <PID> /F  # Windows
```

### "fairseq not available"

Audio service requires Python 3.10. Either:
1. Use Python 3.10 for audio service
2. Use Docker Compose (handles it automatically)
3. Audio service will fall back to heuristics

### "Module not found"

```bash
# Make sure you installed dependencies
pip install -r requirements-nlp.txt
pip install -r requirements-graph.txt

# For audio (Python 3.10)
pyenv local 3.10.13
pip install -r requirements-audio.txt
```

---

## Recommendation

**For development**: Use Option 2 (Local Microservices)
- Fast iteration
- Easy debugging
- Direct code changes

**For testing**: Use Option 1 (Docker Compose)
- Full integration
- Matches production
- All services work

**For quick NLP tests**: Use Option 3 (Monolithic)
- Simplest
- One command
- Legacy compatibility
