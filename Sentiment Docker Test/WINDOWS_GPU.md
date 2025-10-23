# GPU Acceleration on Windows

## The Problem

**RAPIDS (cuGraph/cuDF) does NOT support Windows natively.** The Python packages only have Linux wheels (manylinux), not Windows wheels. This is why `pip install cudf-cu12 cugraph-cu12` fails on Windows.

Additionally, RAPIDS currently only supports Python 3.10, 3.11, and 3.12 (not 3.13).

## Your Options

### Option 1: Docker Desktop with WSL2 (Recommended)

This is the easiest way to use GPU acceleration on Windows.

**Prerequisites:**
1. Install [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/)
2. Enable WSL2 backend in Docker Desktop settings
3. Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-on-wsl2)

**Steps:**

```powershell
# 1. Navigate to project directory
cd "C:\Users\jrdud\...\Sentiment Docker Test"

# 2. Build GPU-enabled Docker image
docker build -f Dockerfile.gpu -t sentiment-gpu .

# 3. Run with GPU support
docker run --gpus all -p 8080:8080 sentiment-gpu
```

**Access the API:**
- UI: http://localhost:8080
- API Docs: http://localhost:8080/docs

### Option 2: WSL2 (Windows Subsystem for Linux)

Install and run everything directly in WSL2 Ubuntu.

**Prerequisites:**
1. Install WSL2: `wsl --install -d Ubuntu-22.04`
2. Install NVIDIA drivers for WSL2
3. Restart Windows

**Steps in WSL2:**

```bash
# In WSL2 Ubuntu terminal
cd /mnt/c/Users/jrdud/.../Sentiment\ Docker\ Test

# Create Python 3.12 environment
conda create -n rapids python=3.12
conda activate rapids

# Install RAPIDS (this will work in WSL2/Linux)
pip install cudf-cu12 cugraph-cu12 --extra-index-url=https://pypi.nvidia.com
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r req.txt

# Run
python run_local.py
```

### Option 3: CPU Fallback (Current Mode)

The application **will still work without GPU** - it just uses CPU instead, which is slower for large graphs.

**What works without GPU:**
- All sentiment analysis tasks
- All NLP tasks (NER, POS, topics, embeddings)
- Small graph analytics (<1000 nodes)
- Graph visualization

**What's slower without GPU:**
- Large graph analytics (>10,000 nodes)
- PageRank, Betweenness, Eigenvector centrality on large graphs

**To use CPU mode:**
```powershell
python run_local.py
```

You'll see this message:
```
⚠ cuGraph/cuDF not installed (graph GPU acceleration unavailable)
  ⚠ RAPIDS (cuGraph/cuDF) does NOT support Windows
  Solutions:
  1. Docker Desktop with WSL2 (recommended)
  2. Use WSL2 (Windows Subsystem for Linux)
  3. CPU fallback (current mode - functional but slower)
```

Everything will work fine, just slower on large graphs.

## Verifying GPU Usage

### In Docker:

```powershell
# Terminal 1: Run container
docker run --gpus all -p 8080:8080 sentiment-gpu

# Terminal 2: Monitor GPU
nvidia-smi -l 1
```

You should see GPU memory usage increase when running graph analytics.

### In WSL2:

```bash
# Terminal 1: Run app
python run_local.py

# Terminal 2: Monitor GPU
watch -n 1 nvidia-smi
```

## Why Does This Happen?

RAPIDS is built specifically for NVIDIA GPUs and high-performance computing environments, which are primarily Linux-based. The cuGraph/cuDF libraries:

- Use CUDA which requires Linux for optimal support
- Depend on many Linux-specific system libraries
- Are compiled as Linux-only wheels (manylinux)
- Have no official Windows builds

**This is a RAPIDS limitation, not a bug in this project.**

## Summary

| Method | GPU Support | Ease of Setup | Recommendation |
|--------|-------------|---------------|----------------|
| Docker Desktop + WSL2 | ✅ Full | ⭐⭐⭐⭐ Easy | **Recommended** |
| WSL2 Native | ✅ Full | ⭐⭐⭐ Medium | Good for dev |
| CPU Fallback (Windows) | ❌ CPU only | ⭐⭐⭐⭐⭐ Trivial | Works but slower |

**Recommendation:** Use Docker Desktop with WSL2 for the best balance of ease and performance.
