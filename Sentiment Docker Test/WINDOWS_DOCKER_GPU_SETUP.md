# Windows Docker GPU Setup (Fast Track)

**Goal**: Get GPU graph acceleration working on Windows in ~10 minutes

**Your Hardware**: NVIDIA RTX 3090 → Should give you 50-100x speedup over CPU

## Prerequisites Check

Open PowerShell and run these commands:

```powershell
# 1. Check NVIDIA drivers
nvidia-smi
```

**Expected**: Should show your RTX 3090 and driver version

```powershell
# 2. Check Docker Desktop
docker --version
```

**Expected**: `Docker version 24.x` or similar

**If Docker is not installed**: Download from https://www.docker.com/products/docker-desktop/

## Step 1: Enable WSL2 in Docker Desktop (2 minutes)

1. Open **Docker Desktop**
2. Click **Settings** (gear icon)
3. Go to **General**
4. ✅ Enable **"Use the WSL 2 based engine"**
5. Click **Apply & Restart**

## Step 2: Install NVIDIA Container Toolkit in WSL2 (5 minutes)

Open PowerShell and run:

```powershell
# Enter WSL2
wsl

# Now you're in Linux - run these commands:
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker to use NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Exit WSL2
exit
```

## Step 3: Test GPU Access in Docker (1 minute)

Back in PowerShell:

```powershell
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

**Expected output**: Should show your RTX 3090 inside Docker!

**If this fails**: Docker Desktop doesn't have GPU access yet. Try:
1. Restart Docker Desktop completely
2. Restart your computer
3. Try the test command again

## Step 4: Build GPU Image (5-10 minutes, one-time)

```powershell
cd "C:\Users\jrdud\...\Sentiment Docker Test"

# Build the GPU-enabled image (this takes a while first time)
docker build -f Dockerfile.gpu -t sentiment-gpu .
```

This downloads the RAPIDS base image (~5GB) with cuGraph pre-installed.

**Grab coffee while this runs** ☕

## Step 5: Run with GPU! (Instant startup after build)

```powershell
# Run the container with GPU access
docker run --rm --gpus all -p 8080:8080 sentiment-gpu
```

**Look for this in the output**:
```
[graph_tasks] ✓ GPU detected: NVIDIA GeForce RTX 3090
[graph_tasks] ✓ CUDA version: 12.0
[graph_tasks] ✓ GPU GRAPH ACCELERATION ENABLED
```

## Step 6: Test Your Graph Analytics

Open browser to: http://localhost:8080

1. Go to **Graph Analytics** section
2. Upload your edge list
3. Click **Load Graph**
4. Watch it compute in **seconds instead of 20 minutes**! 🚀

## Step 7: Monitor GPU Usage

Open a **second PowerShell window**:

```powershell
nvidia-smi -l 1
```

This updates every second. When you run graph analytics, you should see:
- **GPU Memory Usage**: Increase (graph data loaded on GPU)
- **GPU Utilization**: Spike to 50-100% during computation

## Troubleshooting

### "docker: Error response from daemon: could not select device driver"

**Fix**: NVIDIA Container Toolkit not installed or Docker not restarted

```powershell
wsl
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
exit
# Restart Docker Desktop too
```

### "docker: unknown server OS: windows"

**Fix**: Docker Desktop is in Windows containers mode, need Linux containers

1. Right-click Docker Desktop tray icon
2. Click **"Switch to Linux containers..."**
3. Try again

### GPU not detected inside container

**Fix**: WSL2 backend not enabled

1. Docker Desktop → Settings → General
2. Enable "Use the WSL 2 based engine"
3. Apply & Restart

### Build fails with "no space left on device"

**Fix**: Docker ran out of disk space

1. Docker Desktop → Settings → Resources → Disk image size
2. Increase to at least 80GB
3. Apply & Restart

## Quick Reference

**Start GPU container**:
```powershell
cd "C:\Users\jrdud\...\Sentiment Docker Test"
docker run --rm --gpus all -p 8080:8080 sentiment-gpu
```

**Rebuild after code changes**:
```powershell
docker build -f Dockerfile.gpu -t sentiment-gpu .
docker run --rm --gpus all -p 8080:8080 sentiment-gpu
```

**Run in background**:
```powershell
docker run -d --gpus all -p 8080:8080 --name sentiment-gpu-service sentiment-gpu
```

**Stop background container**:
```powershell
docker stop sentiment-gpu-service
```

**View logs**:
```powershell
docker logs sentiment-gpu-service
```

## Performance Expectations

With your RTX 3090:

| Graph Size | NetworkX CPU | cuGraph GPU | Speedup |
|------------|--------------|-------------|---------|
| 1K nodes | 1 sec | 0.1 sec | 10x |
| 10K nodes | 30 sec | 0.5 sec | 60x |
| 100K nodes | 20 min | 10 sec | **120x** |
| 1M nodes | Hours | 1-2 min | **100x+** |

**Your 20 minute graphs should take 10-30 seconds with GPU**. 🎉

## Next Steps

Once GPU is working:
1. Keep Docker container running during development
2. Access API at http://localhost:8080
3. All graph analytics automatically use GPU
4. Monitor with `nvidia-smi -l 1` to verify GPU usage

## Alternative: WSL2 Native (Advanced Users)

If you want to run directly in WSL2 without Docker:

```powershell
wsl --install -d Ubuntu-22.04
wsl

# Inside WSL2:
cd /mnt/c/Users/jrdud/.../Sentiment\ Docker\ Test
conda create -n rapids python=3.12
conda activate rapids
pip install cudf-cu12 cugraph-cu12 --extra-index-url=https://pypi.nvidia.com
pip install -r req.txt
python run_local.py
```

This gives you native performance but requires more setup.

**Recommendation**: Start with Docker (easier), then try WSL2 native if you want.
