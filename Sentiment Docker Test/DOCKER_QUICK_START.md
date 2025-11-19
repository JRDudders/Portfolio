# Docker Quick Start Guide

This guide shows how to selectively run only the services you need.

## Prerequisites

1. **Add your corporate certificate** (if needed):
   ```bash
   # Place your .crt file in the certs/ directory
   copy "path\to\your\cert.crt" "certs\corporate.crt"
   ```

2. **Set your HuggingFace token** (if needed):
   ```bash
   # Create .env file with your token
   echo HF_TOKEN=your_huggingface_token_here > .env
   ```

---

## Option 1: Run NLP + Graph + Frontend (Recommended)

Build and run the core services with UI:

```bash
# Build all core services (NLP, Graph, Frontend)
docker-compose -f docker-compose.minimal.yml --profile core build

# Start all core services
docker-compose -f docker-compose.minimal.yml --profile core up

# Or combine build + start
docker-compose -f docker-compose.minimal.yml --profile core up --build
```

**Access:**
- Frontend UI: http://localhost:8080
- NLP API: http://localhost:8001
- Graph API: http://localhost:8002

---

## Option 2: Run Only NLP

Build and run just the NLP service:

```bash
# Build
docker-compose -f docker-compose.minimal.yml --profile nlp build

# Start
docker-compose -f docker-compose.minimal.yml --profile nlp up
```

**Access:**
- NLP API: http://localhost:8001
- Frontend: Open `index.html` directly

---

## Option 3: Run Only Graph

Build and run just the Graph service:

```bash
docker-compose -f docker-compose.minimal.yml --profile graph build
docker-compose -f docker-compose.minimal.yml --profile graph up
```

---

## Option 4: Run Everything (Full Stack)

Build and run all services including Audio:

```bash
docker-compose -f docker-compose.minimal.yml --profile full up --build
```

**Access:**
- Frontend: http://localhost:8080
- NLP API: http://localhost:8001
- Graph API: http://localhost:8002
- Audio API: http://localhost:8003

---

## Profile Reference

| Profile | Services Included | Use Case |
|---------|------------------|----------|
| `core` | Frontend + NLP + Graph | Most common - full UI with main features (no audio) |
| `nlp` | NLP only | Text analysis and sentiment |
| `graph` | Graph only | Network analysis |
| `audio` | Audio only | Speech-to-text |
| `frontend` | Frontend only | Nginx web server |
| `full` | All services | Complete deployment including audio |

---

## Common Commands

### Check running containers:
```bash
docker ps
```

### View logs:
```bash
# All services
docker-compose -f docker-compose.minimal.yml --profile core logs

# Specific service
docker-compose -f docker-compose.minimal.yml logs nlp
```

### Stop services:
```bash
docker-compose -f docker-compose.minimal.yml --profile core down
```

### Rebuild a single service:
```bash
docker-compose -f docker-compose.minimal.yml build nlp
```

### Remove all containers and volumes:
```bash
docker-compose -f docker-compose.minimal.yml --profile full down -v
```

---

## Troubleshooting

### Build stuck or hanging:
```bash
# Cancel with Ctrl+C, then try with no cache:
docker-compose -f docker-compose.minimal.yml --profile core build --no-cache
```

### Certificate errors during build:
```bash
# Verify certificate is in place:
dir certs\*.crt

# Make sure it's a valid .crt file (should start with -----BEGIN CERTIFICATE-----)
type certs\corporate.crt
```

### Service won't start:
```bash
# Check logs for errors:
docker-compose -f docker-compose.minimal.yml logs nlp

# Check if port is already in use:
netstat -ano | findstr :8001
```

### Out of memory errors:
```bash
# Increase Docker Desktop memory limit:
# Docker Desktop → Settings → Resources → Memory → 8GB+
```

---

## Simple Alternative: Run Without Docker

If Docker is giving you trouble, you can run the services directly:

```bash
# In conda environment
conda activate cicerowatch

# Run NLP service
python service_nlp.py

# In another terminal, run Graph service
python service_graph.py

# Open index.html in your browser
```

This is simpler and avoids Docker complexity while you're developing.
