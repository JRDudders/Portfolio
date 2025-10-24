# Quick Start: GPU Acceleration on Windows

**Problem**: Graph analytics taking 20+ minutes on CPU
**Solution**: Use GPU with Docker (reduces to seconds)
**Hardware**: Your RTX 3090 → 50-100x speedup

## TL;DR

```powershell
# 1. One-time setup (~10 min)
#    Follow: WINDOWS_DOCKER_GPU_SETUP.md

# 2. Build image (one-time, ~10 min)
docker build -f Dockerfile.gpu -t sentiment-gpu .

# 3. Run with GPU (instant after build)
docker run --rm --gpus all -p 8080:8080 sentiment-gpu

# Or use the PowerShell script:
.\run_docker_gpu.ps1
```

## Performance Impact

| Your Experience | Graph Size | CPU Time | GPU Time | Speedup |
|-----------------|------------|----------|----------|---------|
| **Current (slow)** | Unknown | 20 min | 10-30 sec | **40-120x** |

## What You'll See

**Before** (CPU):
```
Computing PageRank... ████████░░░░ 60% [20 minutes elapsed]
```

**After** (GPU):
```
[graph_tasks] ✓ GPU detected: NVIDIA GeForce RTX 3090
[graph_tasks] ✓ GPU GRAPH ACCELERATION ENABLED
Computing PageRank... ████████████ 100% [12 seconds elapsed]
```

## Full Setup Instructions

See **WINDOWS_DOCKER_GPU_SETUP.md** for complete step-by-step guide.

## Quick Troubleshooting

**"docker: Error response from daemon: could not select device driver"**
→ NVIDIA Container Toolkit not installed. See WINDOWS_DOCKER_GPU_SETUP.md Step 2

**Build takes forever**
→ Normal for first build (downloading 5GB RAPIDS image). Subsequent builds are fast.

**GPU not detected inside container**
→ Docker Desktop not using WSL2 backend. Settings → General → Enable WSL2 engine

## Alternative: CPU with NetworkX (Current)

If you can't use Docker and need to keep testing:
- Small graphs (<1K nodes): Use NetworkX (fast enough)
- Large graphs (10K+ nodes): **You need Docker GPU** (no other option on Windows)

## Need Help?

1. Check WINDOWS_DOCKER_GPU_SETUP.md (detailed guide)
2. Verify GPU works: `docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi`
3. Check Docker logs: `docker logs sentiment-gpu-container`
