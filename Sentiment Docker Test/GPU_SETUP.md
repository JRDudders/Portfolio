# GPU Acceleration Setup Guide

This document explains GPU support for CiceroWatch services.

---

## 🎮 GPU Support Overview

| Service | GPU Support | Acceleration Library | Speed Improvement |
|---------|-------------|---------------------|-------------------|
| **Graph Analytics** | ✅ Yes | RAPIDS cuGraph/cuDF | 10-100x on large graphs |
| **NLP** | ✅ Yes | PyTorch (transformers) | 2-5x for text models |
| **Audio** | ✅ Yes | PyTorch (fairseq) | 3-10x for audio models |
| **Frontend** | ❌ No | N/A | N/A |

---

## 📋 Requirements

### Hardware
- **NVIDIA GPU** (GTX 1060+ or better)
- **8GB+ VRAM** (16GB+ recommended for large models)
- **Linux OS** (Ubuntu 20.04+ recommended)

### Software
- **Docker** with **nvidia-docker** runtime
- **NVIDIA drivers** (525+ for CUDA 12.x)
- **docker-compose** v2.0+

### ❌ NOT Supported
- **macOS** (Intel or Apple Silicon) - No NVIDIA GPU support
- **Windows native** - Use WSL2 instead
- **AMD GPUs** - RAPIDS only supports NVIDIA CUDA

---

## 🚀 Quick Start - GPU Mode

### 1. Verify GPU is Available

```bash
# Check NVIDIA driver
nvidia-smi

# Test Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### 2. Start GPU Services

```bash
# Production GPU mode
docker-compose -f docker-compose.prod.gpu.yml build
docker-compose -f docker-compose.prod.gpu.yml up

# Development GPU mode (with volume mounts)
docker-compose -f docker-compose.gpu.yml up
```

### 3. Verify GPU Acceleration

Check the logs - graph service should show:
- ✓ cuGraph available (version 24.08)
- ✓ cuDF available (version 24.08)

NLP/Audio services should detect CUDA.

---

## 🖥️ CPU Mode (No GPU Required)

### When to Use CPU Mode
- Running on macOS
- Running on Windows (without WSL2)
- No NVIDIA GPU available
- Development on laptop

### Start CPU Services

```bash
# Production CPU mode
docker-compose -f docker-compose.prod.yml up

# Development CPU mode
docker-compose up
```

---

## 📦 GPU Docker Configuration

### Graph Service (RAPIDS cuGraph)

**Dockerfile:** Dockerfile.graph.gpu
**Base Image:** rapidsai/base:24.08a-cuda12.5-py3.11
**Includes:** cuGraph, cuDF pre-installed

### NLP Service

**Dockerfile:** Dockerfile.nlp.gpu
**PyTorch:** 2.5.1 with CUDA 12.1

### Audio Service

**Dockerfile:** Dockerfile.audio
**PyTorch:** 2.5.1 (auto-detects GPU)

---

## 🐛 Troubleshooting

### "✗ cuGraph/cuDF not installed" Warning

**Using CPU mode?** This is NORMAL - CPU mode uses NetworkX.
**Using GPU mode?** Something is wrong - check nvidia-docker.

### Platform Support

- ✅ **Linux + NVIDIA GPU** - Full GPU support
- ✅ **Windows + WSL2 + NVIDIA GPU** - GPU support
- ❌ **macOS** - CPU only (no NVIDIA GPUs on Mac)

---

## 📚 Additional Resources

- [RAPIDS cuGraph Documentation](https://docs.rapids.ai/api/cugraph/stable/)
- [Docker GPU Guide](https://docs.docker.com/config/containers/resource_constraints/#gpu)
- Check DOCKER_TROUBLESHOOTING.md for Docker issues
