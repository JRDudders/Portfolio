# Docker Build Failure Troubleshooting

**Error:** `rpc error: code = Unavailable desc = error reading from server: EOF` + Segmentation fault

This indicates Docker daemon instability, likely on WSL2. Here are the solutions:

---

## 🔧 Quick Fix (Try These First)

### 1. Restart Docker Desktop
```bash
# In PowerShell (as Administrator):
wsl --shutdown
# Then restart Docker Desktop from Windows
```

### 2. Check Docker Status
```bash
docker info
docker ps
```

If these commands fail or hang, the daemon is crashed.

### 3. Clean Docker State
```bash
# Remove dangling images and build cache
docker system prune -a --volumes

# If that fails, force cleanup
docker system prune -f
```

---

## 🛠️ If Quick Fix Doesn't Work

### Option A: Restart WSL2 + Docker
```powershell
# In PowerShell (Admin)
wsl --shutdown
net stop com.docker.service
net start com.docker.service
# Restart Docker Desktop
```

### Option B: Reset Docker Desktop
1. Open Docker Desktop
2. Settings → Troubleshoot → "Clean / Purge data"
3. Restart Docker Desktop

### Option C: Rebuild Without Cache (Less Memory Intensive)
```bash
# Build one service at a time
docker-compose build --no-cache nlp
docker-compose build --no-cache graph
docker-compose build --no-cache audio
docker-compose build --no-cache frontend

# Then start
docker-compose up
```

---

## 🔍 Common Causes

### 1. Memory Exhaustion
Docker ran out of memory during build. The audio service build is particularly heavy (fairseq, PyTorch).

**Solution:** Increase Docker memory limit:
- Docker Desktop → Settings → Resources
- Set Memory to at least **8GB** (12GB+ recommended)

### 2. WSL2 Disk Space
Check WSL2 disk usage:
```bash
df -h
```

If disk is full:
```bash
docker system df  # Check Docker disk usage
docker system prune -a --volumes  # Free up space
```

### 3. Corrupted Build Cache
```bash
# Clear build cache
docker builder prune -a
```

---

## 🚀 Recommended Build Strategy

Given the memory-intensive builds (especially audio with fairseq), try this:

### Step 1: Build Frontend First (Lightest)
```bash
docker-compose build frontend
```

### Step 2: Build NLP Service
```bash
docker-compose build nlp
```

### Step 3: Build Graph Service
```bash
docker-compose build graph
```

### Step 4: Build Audio Last (Heaviest)
```bash
# This one takes the most memory/time
docker-compose build audio
```

### Step 5: Start Everything
```bash
docker-compose up
```

---

## 💡 Alternative: Use Pre-built Images

If builds keep failing, you can pull pre-built PyTorch images:

**Modify Dockerfile.audio:**
```dockerfile
# Instead of:
FROM python:3.10-slim

# Use:
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime
# Or for CPU:
FROM pytorch/pytorch:2.5.1-cpu
```

This skips the heavy PyTorch installation step.

---

## 🔄 If Segmentation Fault Persists

### Reset Docker Completely
```powershell
# PowerShell (Admin)
wsl --shutdown
wsl --unregister docker-desktop
wsl --unregister docker-desktop-data
# Then reinstall Docker Desktop
```

### Check System Resources
```bash
# In WSL
free -h  # Check available memory
df -h    # Check disk space
```

---

## ✅ Working Setup Checklist

- [ ] Docker Desktop is running
- [ ] WSL2 backend is enabled
- [ ] At least 8GB memory allocated to Docker
- [ ] At least 20GB free disk space
- [ ] Docker daemon responds to `docker info`
- [ ] No other heavy processes running

---

## 🎯 Fastest Path to Success

**If you just want to test the code changes without rebuilding:**

1. Use development mode (already running containers)
2. Restart services to pick up changes:
   ```bash
   docker-compose restart nlp audio frontend
   ```

3. Changes in these files apply immediately (volume-mounted):
   - service_nlp.py ✅
   - nlp.py ✅
   - topics.py ✅
   - audio_service.py ✅
   - index.html ✅

**You DON'T need to rebuild** to test the bug fixes in development mode!

---

## 📞 If Nothing Works

The issue is with Docker/WSL2 infrastructure, not the code. Consider:

1. **Restart your computer** - WSL2 can get into bad states
2. **Reinstall Docker Desktop** - Nuclear option but effective
3. **Use Docker on Linux VM** - More stable than WSL2
4. **Use cloud environment** - GitHub Codespaces, AWS, etc.

---

## 🆘 Quick Recovery Commands

```bash
# Full Docker reset
docker-compose down -v
docker system prune -a -f --volumes
wsl --shutdown
# Restart Docker Desktop

# Then rebuild
docker-compose build --no-cache
docker-compose up
```
