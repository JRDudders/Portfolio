# Development Workflow with GPU Acceleration

**Problem**: Don't want to rebuild Docker every time you change code
**Solution**: Code is mounted as volume - changes are instant!

## TL;DR

**Docker already mounts your code!** When you run:
```powershell
.\run_docker_gpu.ps1
```

Your code directory is mounted as a volume. **Edit Python files, refresh browser, changes appear immediately.** No rebuild needed!

**Only rebuild when**:
- You change `req.txt` (new dependencies)
- You change `Dockerfile.gpu`
- Never for `.py`, `.html`, or data files!

## Option 1: Docker with Live Code (Good for Testing)

### One-Time Setup (10 minutes)

```powershell
# 1. Setup Docker GPU (see WINDOWS_DOCKER_GPU_SETUP.md)
# 2. Build image ONCE
docker build -f Dockerfile.gpu -t sentiment-gpu .
```

### Daily Development (5 second startup)

```powershell
# Start container (mounts your code)
.\run_docker_gpu.ps1

# Or manually:
docker run --rm --gpus all -p 8080:8080 -v "${PWD}:/app" sentiment-gpu
```

**Your workflow:**
1. Start container once (5 seconds)
2. Edit `graph_tasks.py` in VS Code
3. Save file
4. Restart uvicorn inside container OR just refresh API call
5. Changes are live immediately!

**Container stays running while you edit code.**

### When to Rebuild

```powershell
# ONLY IF you changed req.txt or Dockerfile.gpu
docker build -f Dockerfile.gpu -t sentiment-gpu .
```

**99% of the time you don't need this!**

## Option 2: WSL2 Native (Best for Development)

**Even faster** - no Docker overhead, direct GPU access, instant restarts.

### One-Time Setup (10 minutes)

```powershell
# Install WSL2 if not already
wsl --install -d Ubuntu-22.04

# Enter WSL2
wsl

# Navigate to your project (Windows drives are at /mnt/c/)
cd /mnt/c/Users/jrdud/Documents/Portfolio/Sentiment\ Docker\ Test

# Create Python 3.12 environment
conda create -n rapids python=3.12 -y
conda activate rapids

# Install GPU dependencies (this is the one-time slow part)
pip install cudf-cu12 cugraph-cu12 --extra-index-url=https://pypi.nvidia.com

# Install app dependencies
pip install -r req.txt

# Install NLP models
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
playwright install chromium
```

### Daily Development (Instant)

```powershell
# Open WSL2
wsl

# Activate environment
conda activate rapids

# Navigate to project
cd /mnt/c/Users/jrdud/Documents/Portfolio/Sentiment\ Docker\ Test

# Run (starts in 2 seconds)
python run_local.py
```

**Your workflow:**
1. Edit `graph_tasks.py` in VS Code on Windows
2. Save file
3. In WSL2 terminal: `Ctrl+C` to stop, `↑` to re-run `python run_local.py`
4. Changes are live in 2 seconds!

**This is the fastest iteration cycle.**

## Option 3: Docker with Auto-Reload (Advanced)

Add `--reload` flag to uvicorn for automatic reloading on code changes:

```powershell
# Edit Dockerfile.gpu line 125:
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--reload", "--lifespan", "on"]
```

Then rebuild ONCE and run. Now code changes auto-reload without restarting!

## Comparison

| Method | Startup Time | Code Change Time | GPU Speed | Best For |
|--------|--------------|------------------|-----------|----------|
| **Docker (volume mount)** | 5 sec | Instant (no restart) | Full GPU | Production testing |
| **WSL2 Native** | 2 sec | 2 sec (restart) | Full GPU | **Fast development** |
| **Docker (auto-reload)** | 5 sec | Instant (auto) | Full GPU | Continuous dev |
| Windows CPU | Instant | Instant | No GPU | Not viable for large graphs |

## My Recommendation

**Use WSL2 Native for development:**
- Fastest iteration cycle
- Direct GPU access (no Docker overhead)
- Edit in Windows, run in WSL2
- VS Code has excellent WSL2 integration

**Use Docker for:**
- Testing production builds
- Sharing with team
- Deployment

## Fast WSL2 Setup Commands (Copy-Paste)

```powershell
# In PowerShell
wsl --install -d Ubuntu-22.04
wsl

# Now in WSL2 - copy-paste this entire block:
cd /mnt/c/Users/jrdud/Documents/Portfolio/Sentiment\ Docker\ Test
conda create -n rapids python=3.12 -y
conda activate rapids
pip install cudf-cu12 cugraph-cu12 --extra-index-url=https://pypi.nvidia.com
pip install -r req.txt
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
playwright install chromium
python run_local.py
```

**You'll see:**
```
[graph_tasks] ✓ GPU detected: NVIDIA GeForce RTX 3090
[graph_tasks] ✓ GPU GRAPH ACCELERATION ENABLED
🚀 Starting server...
INFO:     Application startup complete.
```

Access at: http://localhost:8080

## VS Code + WSL2 (Pro Tip)

Install "Remote - WSL" extension in VS Code:

1. Install extension: `ms-vscode-remote.remote-wsl`
2. In VS Code: `Ctrl+Shift+P` → "WSL: Connect to WSL"
3. Open folder: `/mnt/c/Users/jrdud/.../Sentiment Docker Test`
4. Terminal automatically opens in WSL2
5. Edit files, run commands, all in WSL2

**This is the smoothest development experience.**

## Troubleshooting

### "Code changes not appearing in Docker"

**Check volume mount:**
```powershell
docker run --rm --gpus all -p 8080:8080 -v "${PWD}:/app" sentiment-gpu
```

The `-v "${PWD}:/app"` mounts your current directory. If missing, add it!

### "Still need to rebuild for every change"

You're probably changing dependencies. Code changes (`*.py`, `*.html`) don't need rebuild.

**Only rebuild if you change:**
- `req.txt`
- `Dockerfile.gpu`
- System-level dependencies

### "WSL2 can't find files"

Windows drives are at `/mnt/c/`, `/mnt/d/`, etc.

```bash
# If your project is at C:\Users\jrdud\Documents\Portfolio
cd /mnt/c/Users/jrdud/Documents/Portfolio/Sentiment\ Docker\ Test
```

### "GPU not detected in WSL2"

Check NVIDIA drivers support WSL2:
```bash
nvidia-smi
```

Should show your GPU. If not, update NVIDIA drivers on Windows.

## Summary

**For 20+ minute graph wait times:**

1. **Fastest setup**: WSL2 Native (~10 min setup, instant iteration)
2. **Easier setup**: Docker with volume mount (~10 min setup, 5 sec startup)

**Both give you full RTX 3090 GPU acceleration.**

**Your 20 minute graphs → 10-30 seconds. Code changes → Instant.**

Choose WSL2 if you want **maximum development speed**.
Choose Docker if you prefer **easier setup** or need production testing.

**I recommend WSL2 Native for your use case** - fastest iteration for active development.
