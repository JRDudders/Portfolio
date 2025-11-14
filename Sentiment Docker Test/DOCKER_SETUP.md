# Docker Setup Guide

## For Mac (Apple Silicon / Intel)

**Use the CPU version:**
```bash
docker-compose up --build
```

This uses `Dockerfile.nlp` which installs PyTorch CPU version - perfect for Mac.

## For Linux with NVIDIA GPU

**Use the GPU version:**
```bash
docker-compose -f docker-compose.gpu.yml up --build
```

This uses `Dockerfile.nlp.gpu` which requires:
- NVIDIA GPU
- NVIDIA Docker runtime
- CUDA 12.1+ support

## Quick Reference

| Platform | Command | Dockerfile Used |
|----------|---------|-----------------|
| **Mac Studio** | `docker-compose up --build` | `Dockerfile.nlp` (CPU) |
| **Mac (any)** | `docker-compose up --build` | `Dockerfile.nlp` (CPU) |
| **Linux (no GPU)** | `docker-compose up --build` | `Dockerfile.nlp` (CPU) |
| **Linux (NVIDIA GPU)** | `docker-compose -f docker-compose.gpu.yml up --build` | `Dockerfile.nlp.gpu` (CUDA 12.1) |

## Performance Notes

With batch processing enabled (recent updates):
- **CPU**: ~1000-2000 texts/min (stance detection)
- **GPU**: ~5000-10000 texts/min (stance detection)

Batch sizes can be configured via environment variables:
```bash
export STANCE_BATCH_SIZE=32      # Default: 32
export CLASSIFY_BATCH_SIZE=16    # Default: 16
export ZEROSHOT_BATCH_SIZE=8     # Default: 8
```

## Troubleshooting

### Error: "Could not find a version that satisfies the requirement torch==2.5.1"
- You're trying to use the GPU dockerfile on Mac
- Solution: Use `docker-compose up --build` (without `-f docker-compose.gpu.yml`)

### Error: "NVIDIA runtime not found"
- You're trying to use GPU on a system without NVIDIA Docker
- Solution: Use CPU version or install NVIDIA Docker runtime
