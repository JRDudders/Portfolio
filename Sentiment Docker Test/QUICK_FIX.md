# Quick Fix: Nginx Error - Services Not Running

## 🔴 The Problem

You're getting an nginx error page because the **backend services crashed** during startup.

**Root cause:** You're running containers built BEFORE the fixes. The NLP service is crashing with:
```
ModuleNotFoundError: No module named 'adapters'
```

---

## ✅ The Solution: Rebuild Containers

### Option 1: Rebuild All Services (Recommended)

```bash
# Stop everything
docker-compose -f docker-compose.prod.yml down

# Rebuild with latest fixes
docker-compose -f docker-compose.prod.yml build

# Start
docker-compose -f docker-compose.prod.yml up
```

### Option 2: Rebuild Only NLP Service (Faster)

```bash
# Stop services
docker-compose -f docker-compose.prod.yml down

# Rebuild just NLP (has the critical fix)
docker-compose -f docker-compose.prod.yml build nlp

# Start everything
docker-compose -f docker-compose.prod.yml up
```

---

## 🔍 What the Fixes Include

The latest Dockerfiles now include:
- ✅ `adapters.py` - Adapter system for URL processing
- ✅ `fetch.py` - URL fetching and crawling
- ✅ `render.py` - HTML processing
- ✅ `apputils.py` - Utility functions
- ✅ `playwright` and `selenium` in requirements (for rendering)

---

## 🚀 Expected Startup Messages

After rebuild, you should see:

### Frontend (nginx)
```
Configuration complete; ready for start up
```

### NLP Service
```
CiceroWatch NLP Service
Listening on: http://0.0.0.0:8001
```

### Graph Service
```
CiceroWatch Graph Analytics Service
Listening on: http://0.0.0.0:8002
⚠ GPU graph acceleration disabled - using NetworkX (CPU)
```

### Audio Service
```
CiceroWatch Audio Service
Listening on: http://0.0.0.0:8003
```

---

## ⚠️ If Rebuild Fails

If the build crashes (especially audio service with fairseq):

### Try Building One at a Time

```bash
# Build in order (lightest to heaviest)
docker-compose -f docker-compose.prod.yml build frontend
docker-compose -f docker-compose.prod.yml build nlp
docker-compose -f docker-compose.prod.yml build graph
docker-compose -f docker-compose.prod.yml build audio

# Then start
docker-compose -f docker-compose.prod.yml up
```

### Check Docker Memory

Go to Docker Desktop → Settings → Resources
- Set Memory to at least **8GB** (12GB+ recommended)

### See DOCKER_TROUBLESHOOTING.md for more help

---

## 🔧 Verify Services are Running

After starting, check container status:

```bash
# See running containers
docker ps

# Should show:
# cicerowatch-frontend
# cicerowatch-nlp
# cicerowatch-graph
# cicerowatch-audio
```

Check service health:

```bash
# Test each service
curl http://localhost:8001/health  # NLP
curl http://localhost:8002/health  # Graph
curl http://localhost:8003/health  # Audio
curl http://localhost/             # Frontend
```

All should return success responses.

---

## 💡 Why This Happened

1. You pulled the repo with all the fixes
2. But Docker was still using OLD images (cached)
3. The old images were missing the new dependencies
4. Services crashed on startup → nginx couldn't proxy → error page

**Solution:** Rebuild to bake in the new code and dependencies!

---

## 🎯 Quick Commands Reference

```bash
# Full rebuild (clean)
docker-compose -f docker-compose.prod.yml down
docker-compose -f docker-compose.prod.yml build --no-cache
docker-compose -f docker-compose.prod.yml up

# Fast rebuild (uses cache)
docker-compose -f docker-compose.prod.yml build
docker-compose -f docker-compose.prod.yml up

# Check what's wrong
docker-compose -f docker-compose.prod.yml logs nlp
docker-compose -f docker-compose.prod.yml logs graph
docker-compose -f docker-compose.prod.yml logs audio

# Check running containers
docker ps

# Restart specific service
docker-compose -f docker-compose.prod.yml restart nlp
```

---

## ✅ After Rebuild

Once rebuilt and running, you'll be able to:
- ✅ Use zero-shot classification
- ✅ Analyze URLs with rendering
- ✅ Upload files for analysis
- ✅ Run unsupervised topic modeling (NMF/K-Means)
- ✅ Audio deepfake detection
- ✅ Graph analytics

All the bug fixes will be active! 🎉
