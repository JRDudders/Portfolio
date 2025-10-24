# run_docker_gpu.ps1 - Run the Sentiment Analysis API with GPU acceleration on Windows
#
# Prerequisites:
# 1. NVIDIA GPU with CUDA support (RTX 3090 detected)
# 2. Docker Desktop with WSL2 backend enabled
# 3. NVIDIA Container Toolkit installed in WSL2
#
# Quick setup: See WINDOWS_DOCKER_GPU_SETUP.md
#
# Usage:
#   .\run_docker_gpu.ps1

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Sentiment Analysis API - GPU Docker Mode" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Check if nvidia-smi is available
try {
    $nvidiaInfo = nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ NVIDIA drivers detected" -ForegroundColor Green
        Write-Host "  $nvidiaInfo" -ForegroundColor Gray
        Write-Host ""
    } else {
        throw "nvidia-smi failed"
    }
} catch {
    Write-Host "❌ ERROR: nvidia-smi not found" -ForegroundColor Red
    Write-Host "   NVIDIA GPU drivers are not installed or not in PATH" -ForegroundColor Yellow
    Write-Host "   Install NVIDIA drivers first: https://www.nvidia.com/download/index.aspx" -ForegroundColor Yellow
    exit 1
}

# Check if Docker is available
try {
    $dockerVersion = docker --version 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Docker Desktop detected" -ForegroundColor Green
        Write-Host "  $dockerVersion" -ForegroundColor Gray
        Write-Host ""
    } else {
        throw "docker failed"
    }
} catch {
    Write-Host "❌ ERROR: Docker not found" -ForegroundColor Red
    Write-Host "   Install Docker Desktop: https://www.docker.com/products/docker-desktop/" -ForegroundColor Yellow
    Write-Host "   Enable WSL2 backend in Docker Desktop settings" -ForegroundColor Yellow
    exit 1
}

# Check if Docker supports GPU
Write-Host "Checking Docker GPU support..." -ForegroundColor Cyan
try {
    docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Docker GPU support detected" -ForegroundColor Green
        Write-Host ""
    } else {
        throw "docker gpu test failed"
    }
} catch {
    Write-Host "❌ ERROR: Docker GPU support not available" -ForegroundColor Red
    Write-Host ""
    Write-Host "Follow these steps:" -ForegroundColor Yellow
    Write-Host "1. Enable WSL2 in Docker Desktop (Settings → General)" -ForegroundColor Yellow
    Write-Host "2. Install NVIDIA Container Toolkit in WSL2:" -ForegroundColor Yellow
    Write-Host "   wsl" -ForegroundColor Gray
    Write-Host "   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg" -ForegroundColor Gray
    Write-Host "   distribution=`$(. /etc/os-release;echo `$ID`$VERSION_ID)" -ForegroundColor Gray
    Write-Host "   curl -s -L https://nvidia.github.io/libnvidia-container/`$distribution/libnvidia-container.list | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list" -ForegroundColor Gray
    Write-Host "   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit" -ForegroundColor Gray
    Write-Host "   sudo nvidia-ctk runtime configure --runtime=docker" -ForegroundColor Gray
    Write-Host "   sudo systemctl restart docker" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Full guide: WINDOWS_DOCKER_GPU_SETUP.md" -ForegroundColor Yellow
    exit 1
}

# Build GPU image if it doesn't exist
$imageName = "sentiment-gpu"
$imageExists = docker images -q $imageName 2>$null
if (-not $imageExists) {
    Write-Host "Building GPU-enabled Docker image..." -ForegroundColor Cyan
    Write-Host "This may take 5-10 minutes on first run (downloads RAPIDS base image ~5GB)" -ForegroundColor Yellow
    Write-Host ""
    docker build -f Dockerfile.gpu -t $imageName .
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "❌ Build failed!" -ForegroundColor Red
        exit 1
    }
    Write-Host ""
    Write-Host "✓ Build complete!" -ForegroundColor Green
    Write-Host ""
}

Write-Host "Starting container with GPU support..." -ForegroundColor Cyan
Write-Host ""
Write-Host "📍 API will be available at: http://localhost:8080" -ForegroundColor Green
Write-Host "📚 API docs: http://localhost:8080/docs" -ForegroundColor Green
Write-Host "📊 Interactive UI: http://localhost:8080" -ForegroundColor Green
Write-Host ""
Write-Host "💡 DEV MODE: Your code is mounted as a volume" -ForegroundColor Yellow
Write-Host "   Edit .py files → Changes apply immediately (no rebuild needed!)" -ForegroundColor Yellow
Write-Host "   Only rebuild if you change req.txt or Dockerfile.gpu" -ForegroundColor Yellow
Write-Host ""
Write-Host "Monitor GPU usage in another terminal: nvidia-smi -l 1" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Gray
Write-Host ""

# Run container with GPU support
# Convert Windows path to Linux path for Docker WSL2 backend
$currentPath = (Get-Location).Path
$volumeMount = "${currentPath}:/app"

# If Docker Desktop is using WSL2, paths should work as-is
# Docker Desktop automatically handles Windows path conversion
docker run --rm -it `
    --gpus all `
    -p 8080:8080 `
    -v "$volumeMount" `
    --name sentiment-gpu-container `
    $imageName

# Note: --gpus all enables all GPUs
# For specific GPU: --gpus '"device=0"'
# For multiple GPUs: --gpus '"device=0,1"'
