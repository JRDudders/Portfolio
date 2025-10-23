#!/bin/bash
# run_docker_gpu.sh - Run the Sentiment Analysis API with GPU acceleration using Docker
#
# Prerequisites:
# 1. NVIDIA GPU with CUDA support
# 2. NVIDIA Docker runtime installed (nvidia-docker2)
# 3. Docker with GPU support
#
# Installation (Ubuntu/Debian):
#   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
#   curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
#     sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
#     sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
#   sudo apt-get update
#   sudo apt-get install -y nvidia-container-toolkit
#   sudo systemctl restart docker
#
# Usage:
#   ./run_docker_gpu.sh

set -e

echo "=========================================="
echo "Sentiment Analysis API - GPU Docker Mode"
echo "=========================================="
echo ""

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ ERROR: nvidia-smi not found"
    echo "   NVIDIA GPU drivers are not installed or not in PATH"
    echo "   Install NVIDIA drivers first: https://www.nvidia.com/download/index.aspx"
    exit 1
fi

echo "✓ NVIDIA drivers detected"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo ""

# Check if Docker supports GPU
if ! docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo "❌ ERROR: Docker GPU support not available"
    echo "   Install nvidia-container-toolkit:"
    echo "   https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    exit 1
fi

echo "✓ Docker GPU support detected"
echo ""

# Build GPU image if it doesn't exist
IMAGE_NAME="sentiment-gpu"
if [[ "$(docker images -q $IMAGE_NAME 2> /dev/null)" == "" ]]; then
    echo "Building GPU-enabled Docker image..."
    echo "This may take several minutes on first run..."
    docker build -f Dockerfile.gpu -t $IMAGE_NAME .
    echo ""
fi

echo "Starting container with GPU support..."
echo ""
echo "📍 API will be available at: http://localhost:8080"
echo "📚 API docs: http://localhost:8080/docs"
echo "📊 Interactive UI: http://localhost:8080"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run container with GPU support
docker run --rm -it \
    --gpus all \
    -p 8080:8080 \
    -v "$(pwd):/app" \
    --name sentiment-gpu-container \
    $IMAGE_NAME

# Note: --gpus all enables all GPUs
# For specific GPU: --gpus '"device=0"'
# For multiple GPUs: --gpus '"device=0,1"'
