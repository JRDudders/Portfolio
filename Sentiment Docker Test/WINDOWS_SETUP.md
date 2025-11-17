# Windows Setup Guide for CiceroWatch

This guide helps you set up CiceroWatch for local development on Windows.

## Quick Start (Recommended)

### Option 1: Using Conda (Easiest)

```bash
# 1. Create conda environment
conda create -n cicerowatch python=3.12
conda activate cicerowatch

# 2. Install numpy via conda (avoids build issues)
conda install numpy

# 3. Install other dependencies
pip install -r requirements-local.txt

# 4. Run the application
python run_local.py
```

### Option 2: Using pip only

```bash
# 1. Make sure pip is up to date
pip install --upgrade pip

# 2. Install dependencies (requirements updated for Windows)
pip install -r requirements-local.txt

# 3. Run the application
python run_local.py
```

---

## Common Windows Issues

### Issue 1: numpy Build Failure

**Error Message**:
```
ERROR: Unknown compiler(s): [['icl'], ['cl'], ['cc'], ['gcc'], ['clang']...]
Failed to activate VS environment: Could not find vswhere.exe
× Preparing metadata (pyproject.toml) did not run successfully
```

**What's Happening**:
- numpy is trying to build from source (compile C code)
- Your system doesn't have a C compiler installed
- Building numpy from source requires Microsoft Visual Studio

**Solution A - Use Conda (Recommended)**:
```bash
conda create -n cicerowatch python=3.12
conda activate cicerowatch
conda install numpy  # Get pre-built binary
pip install -r requirements-local.txt
```

**Solution B - Use Fixed pip Requirements**:
The `requirements-local.txt` has been updated to use specific numpy versions with pre-built Windows wheels:
```bash
pip install --upgrade pip
pip install -r requirements-local.txt
```

**Solution C - Install Specific numpy Version**:
```bash
pip install numpy==1.26.3
pip install -r requirements-local.txt
```

**Solution D - Install Visual Studio Build Tools** (Last Resort):
1. Download [Microsoft Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/)
2. Install "Desktop development with C++"
3. Restart your terminal
4. Try `pip install -r requirements-local.txt` again

### Issue 2: PyTorch Installation (GPU Support)

If you want GPU acceleration for NLP models:

```bash
# Install PyTorch with CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Check GPU availability
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

**Requirements for GPU**:
- NVIDIA GPU with CUDA support
- [NVIDIA Drivers](https://www.nvidia.com/Download/index.aspx) installed
- Check with: `nvidia-smi` in command prompt

### Issue 3: Audio Service (fairseq)

The audio antispoofing service requires Python 3.10 (fairseq compatibility):

**Option A - Use Docker** (Recommended):
```bash
docker-compose up
```

**Option B - Use Python 3.10** (if you need local audio):
```bash
# Using conda
conda create -n cicerowatch-audio python=3.10
conda activate cicerowatch-audio
pip install -r requirements-audio.txt
```

---

## Step-by-Step Setup

### 1. Check Python Version

```bash
python --version
```

Recommended: Python 3.12 for NLP/Graph, Python 3.10 for Audio

### 2. Install Anaconda (Recommended for Windows)

Download from: https://www.anaconda.com/download

Benefits:
- Pre-built binaries for scientific packages
- No need for C compiler
- Easier package management
- Virtual environments included

### 3. Create Environment

**Using Conda**:
```bash
conda create -n cicerowatch python=3.12
conda activate cicerowatch
```

**Using venv** (alternative):
```bash
python -m venv cicerowatch-env
cicerowatch-env\Scripts\activate
```

### 4. Install Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install dependencies
pip install -r requirements-local.txt
```

### 5. Verify Installation

```bash
# Check numpy
python -c "import numpy; print('numpy version:', numpy.__version__)"

# Check PyTorch
python -c "import torch; print('PyTorch version:', torch.__version__)"

# Check CUDA (if GPU)
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### 6. Run the Application

```bash
# Run local server
python run_local.py

# Access at:
# http://localhost:8080
# http://localhost:8080/docs (API documentation)
```

---

## Alternative: Docker Desktop

If local setup is too complex, use Docker:

### 1. Install Docker Desktop

Download from: https://www.docker.com/products/docker-desktop/

### 2. Run with Docker Compose

```bash
# Start all services
docker-compose up

# Or in background
docker-compose up -d

# Access at http://localhost
```

### 3. GPU Support (WSL2)

For GPU support in Docker on Windows:

1. Install [WSL2](https://docs.microsoft.com/en-us/windows/wsl/install)
2. Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
3. Run GPU containers:
   ```bash
   docker-compose -f docker-compose.gpu.yml up
   ```

See [GPU_SETUP.md](GPU_SETUP.md) for detailed instructions.

---

## Troubleshooting

### "pip is not recognized"

Add Python to PATH:
1. Search for "Environment Variables" in Windows
2. Edit "Path" variable
3. Add Python installation directory (e.g., `C:\Users\YourName\anaconda3`)

### "conda is not recognized"

After installing Anaconda:
1. Close and reopen terminal/command prompt
2. Or use "Anaconda Prompt" from Start menu

### "Port 8080 already in use"

```bash
# Find process using port
netstat -ano | findstr :8080

# Kill process
taskkill /PID <PID> /F
```

### Dependencies take forever to install

This is normal on Windows. First-time installation can take 10-20 minutes.

To speed up:
1. Use conda for large packages (numpy, scipy, pytorch)
2. Use pip only for pure-Python packages
3. Consider using Docker instead

---

## Performance Tips

### 1. Use SSD for Environment

Install your conda/venv environment on an SSD (not HDD) for faster package loading.

### 2. Disable Antivirus for Development Folder

Windows Defender can slow down Python execution. Add your project folder to exclusions:
1. Windows Security → Virus & threat protection
2. Manage settings → Exclusions
3. Add your project folder

### 3. Use Windows Terminal

Better terminal experience than Command Prompt:
- Download from Microsoft Store: "Windows Terminal"
- Supports tabs, copy/paste, colors

---

## Recommended Tools

1. **Anaconda** - Package management
2. **VS Code** - Code editor with Python support
3. **Windows Terminal** - Better command line
4. **Docker Desktop** - Container runtime
5. **Git for Windows** - Version control

---

## Getting Help

If you're still having issues:

1. Check [LOCAL_DEVELOPMENT.md](LOCAL_DEVELOPMENT.md) for general setup
2. Check [DOCKER_SETUP.md](DOCKER_SETUP.md) for Docker alternative
3. See [Common Issues](#common-windows-issues) above
4. Open an issue on GitHub with:
   - Your Python version (`python --version`)
   - Your Windows version
   - Full error message
   - Output of `pip list`
