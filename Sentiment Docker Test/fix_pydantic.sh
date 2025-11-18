#!/bin/bash
# Aggressive fix for Pydantic version conflicts in Conda environments

echo "=========================================="
echo "Pydantic Conflict Fix Script"
echo "=========================================="
echo ""

# Check if in Conda environment
if [[ -z "$CONDA_DEFAULT_ENV" ]]; then
    echo "❌ ERROR: No Conda environment active"
    echo "Please run: conda activate cicerowatch"
    exit 1
fi

echo "✓ Conda environment: $CONDA_DEFAULT_ENV"
echo ""

# Step 1: Uninstall ALL potentially conflicting packages
echo "Step 1: Removing all potentially conflicting packages..."
pip uninstall -y \
    pydantic \
    pydantic-core \
    pydantic-settings \
    fastapi \
    uvicorn \
    starlette \
    typing-extensions \
    annotated-types 2>/dev/null

echo "✓ Old packages removed"
echo ""

# Step 2: Clear pip cache
echo "Step 2: Clearing pip cache..."
pip cache purge 2>/dev/null || echo "Cache already clear"
echo ""

# Step 3: Install core dependencies with exact versions
echo "Step 3: Installing core dependencies with pinned versions..."
pip install --no-cache-dir \
    "typing-extensions==4.12.2" \
    "annotated-types==0.7.0" \
    "pydantic-core==2.27.1" \
    "pydantic==2.10.3"

if [ $? -ne 0 ]; then
    echo "❌ Failed to install pydantic. Trying alternative..."
    pip install --no-cache-dir "pydantic==2.9.2" "pydantic-core==2.23.4"
fi
echo ""

# Step 4: Install FastAPI and Uvicorn
echo "Step 4: Installing FastAPI and Uvicorn..."
pip install --no-cache-dir \
    "starlette==0.41.3" \
    "fastapi==0.115.5" \
    "uvicorn==0.32.1" \
    "python-multipart==0.0.18"
echo ""

# Step 5: Verify installation
echo "Step 5: Verifying installation..."
echo ""

echo "--- Installed Versions ---"
python -c "import pydantic; print(f'Pydantic: {pydantic.__version__}')"
python -c "import pydantic_core; print(f'Pydantic Core: {pydantic_core.__version__}')"
python -c "import fastapi; print(f'FastAPI: {fastapi.__version__}')"
python -c "import uvicorn; print(f'Uvicorn: {uvicorn.__version__}')"
echo ""

echo "--- Testing Imports ---"
python -c "from pydantic import BaseModel; print('✓ Pydantic imports OK')" || {
    echo "❌ Pydantic import failed"
    exit 1
}

python -c "from fastapi import FastAPI; from pydantic import BaseModel; print('✓ FastAPI + Pydantic OK')" || {
    echo "❌ FastAPI import failed"
    exit 1
}

echo ""
echo "=========================================="
echo "✓ Fix completed successfully!"
echo "=========================================="
echo ""
echo "Now try running: python run_local.py"
