# Dockerfile.gpu Image Tag Reference

If the build fails with "not found", try these RAPIDS image tags in Dockerfile.gpu line 4:

## Recommended (Try in order)

```dockerfile
# Option 1: Latest with CUDA 12.5 (most recent)
FROM rapidsai/rapidsai:cuda12.5-runtime-ubuntu22.04-py3.11

# Option 2: RAPIDS 24.10 with CUDA 12.5 (October 2024)
FROM rapidsai/rapidsai:24.10-cuda12.5-runtime-ubuntu22.04-py3.11

# Option 3: RAPIDS 24.08 with CUDA 12.5 (August 2024)
FROM rapidsai/rapidsai:24.08-cuda12.5-runtime-ubuntu22.04-py3.11

# Option 4: RAPIDS 24.10 with CUDA 12.0 (if your GPU doesn't support 12.5)
FROM rapidsai/rapidsai:24.10-cuda12.0-runtime-ubuntu22.04-py3.11

# Option 5: Use base image instead of full rapidsai (smaller)
FROM rapidsai/base:24.10-cuda12.5-runtime-ubuntu22.04-py3.11
```

## Check Available Tags

Visit: https://hub.docker.com/r/rapidsai/rapidsai/tags

Or run:
```powershell
docker pull rapidsai/rapidsai:cuda12.5-runtime-ubuntu22.04-py3.11
```

## CUDA Version Check

Check your GPU's CUDA support:
```powershell
nvidia-smi
```

Look for "CUDA Version: 12.x" in the output. Use matching or lower CUDA version in Docker image.

For example:
- If you see "CUDA Version: 12.5" → Use cuda12.5 or cuda12.0 image
- If you see "CUDA Version: 12.0" → Use cuda12.0 image (not 12.5)

## Current Configuration

**Dockerfile.gpu line 4:**
```dockerfile
FROM rapidsai/rapidsai:24.10-cuda12.5-runtime-ubuntu22.04-py3.11
```

## Quick Fix

If build fails, edit Dockerfile.gpu line 4 and try the "Latest" option:

```dockerfile
FROM rapidsai/rapidsai:cuda12.5-runtime-ubuntu22.04-py3.11
```

This always pulls the most recent stable RAPIDS release.
