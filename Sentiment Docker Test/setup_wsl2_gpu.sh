#!/bin/bash
# setup_wsl2_gpu.sh - One-command WSL2 GPU setup for development
#
# Run this inside WSL2 after: wsl --install -d Ubuntu-24.04
#
# Usage:
#   cd /mnt/c/Users/YourUsername/.../Sentiment\ Docker\ Test
#   bash setup_wsl2_gpu.sh

set -e

echo "=========================================="
echo "WSL2 GPU Setup for Rapid Development"
echo "=========================================="
echo ""

# Check if we're in WSL2
if ! grep -q Microsoft /proc/version; then
    echo "❌ ERROR: This script must run inside WSL2"
    echo "   Run 'wsl' in PowerShell first"
    exit 1
fi

# Check for NVIDIA GPU
if ! nvidia-smi &> /dev/null; then
    echo "⚠ WARNING: nvidia-smi not found"
    echo "   Make sure you have NVIDIA drivers installed on Windows"
    echo "   WSL2 should automatically have GPU access"
    echo ""
    echo "   Continuing anyway..."
    echo ""
fi

# Check for conda
if ! command -v conda &> /dev/null; then
    echo "❌ ERROR: conda not found"
    echo "   Install Miniconda in WSL2:"
    echo "   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "   bash Miniconda3-latest-Linux-x86_64.sh"
    exit 1
fi

echo "✓ Running in WSL2"
echo "✓ conda detected"
echo ""

# Create conda environment
echo "Creating conda environment: rapids (Python 3.12)..."
conda create -n rapids python=3.12 -y

echo ""
echo "Activating environment..."
eval "$(conda shell.bash hook)"
conda activate rapids

echo ""
echo "Installing RAPIDS (cuGraph/cuDF) - this may take 5-10 minutes..."
pip install cudf-cu12 cugraph-cu12 --extra-index-url=https://pypi.nvidia.com

echo ""
echo "Installing application dependencies..."
pip install -r req.txt

echo ""
echo "Downloading NLP models..."
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('stopwords', quiet=True); nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

echo ""
echo "Installing Playwright browsers..."
playwright install chromium

echo ""
echo "=========================================="
echo "✓ Setup Complete!"
echo "=========================================="
echo ""
echo "To run the app:"
echo "  conda activate rapids"
echo "  python run_local.py"
echo ""
echo "Your GPU will be used for graph analytics automatically."
echo "Edit code on Windows, restart in WSL2 - changes apply in 2 seconds!"
echo ""
echo "Monitor GPU: watch -n 1 nvidia-smi"
echo ""

# Test GPU detection
echo "Testing GPU detection..."
python - <<'PYTHON'
import sys
import torch
if torch.cuda.is_available():
    print(f"✓ PyTorch GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠ PyTorch GPU not detected (graphs will use CPU)")

try:
    import cudf
    import cugraph
    print(f"✓ cuGraph {cugraph.__version__} and cuDF {cudf.__version__}")
    print("✓ GPU graph acceleration READY")
except ImportError:
    print("⚠ cuGraph/cuDF import failed (might need to restart WSL2)")
    sys.exit(0)
PYTHON

echo ""
echo "Ready for fast development! 🚀"
