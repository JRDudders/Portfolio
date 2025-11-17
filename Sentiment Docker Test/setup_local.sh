#!/bin/bash
# Setup script for local CiceroWatch development
# Run with: bash setup_local.sh [cpu|cuda118|cuda121]

set -e  # Exit on error

DEVICE="${1:-cpu}"

echo "=========================================="
echo "CiceroWatch Local Setup"
echo "=========================================="
echo ""
echo "Python version: $(python --version)"
echo "Platform: $(uname -s) $(uname -m)"
echo "Device target: $DEVICE"
echo ""

# Step 1: Install PyTorch based on device
echo "Step 1/3: Installing PyTorch..."
case "$DEVICE" in
    cpu)
        echo "Installing CPU version of PyTorch..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        ;;
    cuda118)
        echo "Installing CUDA 11.8 version of PyTorch..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        ;;
    cuda121)
        echo "Installing CUDA 12.1 version of PyTorch..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        ;;
    *)
        echo "ERROR: Invalid device '$DEVICE'. Use: cpu, cuda118, or cuda121"
        exit 1
        ;;
esac

# Step 2: Install main requirements
echo ""
echo "Step 2/3: Installing other requirements..."
if [ -f "requirements-local.txt" ]; then
    echo "Using requirements-local.txt (minimal dependencies)..."
    pip install -r requirements-local.txt
else
    echo "Using req.txt (full dependencies)..."
    pip install -r req.txt
fi

# Step 3: Download spaCy model (optional but recommended)
echo ""
echo "Step 3/3: Downloading spaCy model (optional)..."
python -m spacy download en_core_web_sm || echo "Warning: spaCy model download failed (non-critical)"

echo ""
echo "=========================================="
echo "✓ Setup complete!"
echo "=========================================="
echo ""
echo "To start the server:"
echo "  python run_local.py"
echo ""
echo "Then open: http://localhost:8080"
echo ""
