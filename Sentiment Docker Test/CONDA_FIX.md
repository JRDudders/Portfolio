# Conda Environment Fix - Pydantic Error

## Problem
```
AttributeError: __pydantic_core_schema__
```

This error occurs when `pydantic` and `pydantic-core` versions are incompatible. Common in Conda environments due to dependency conflicts.

---

## Quick Fix (Recommended)

### Option 1: Reinstall with Compatible Versions

```bash
# Activate your Conda environment
conda activate cicerowatch

# Uninstall conflicting packages
pip uninstall -y pydantic pydantic-core fastapi uvicorn

# Install with pinned compatible versions
pip install -r requirements-conda.txt

# Verify installation
python -c "from pydantic import BaseModel; print('Pydantic OK')"
python -c "import fastapi; print('FastAPI OK')"
python -c "import transformers; print('Transformers OK')"

# Run the server
python run_local.py
```

---

### Option 2: Fresh Conda Environment (Clean Slate)

```bash
# Deactivate current environment
conda deactivate

# Remove problematic environment
conda env remove -n cicerowatch

# Create fresh environment with Python 3.11
conda create -n cicerowatch python=3.11 -y

# Activate it
conda activate cicerowatch

# Install PyTorch first (Conda channel for better compatibility)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install other dependencies
pip install -r requirements-conda.txt

# Verify
python run_local.py
```

---

## Understanding the Error

**Root Cause:**
- `pydantic>=2.0` in requirements is too broad
- Conda may install incompatible `pydantic-core` version
- FastAPI depends on specific pydantic versions

**Why it happens in Conda:**
- Conda and pip mix dependencies differently
- `pip install -r requirements-nlp.txt` uses `>=` which grabs latest versions
- Latest versions may be incompatible with each other

**Solution:**
- Use `requirements-conda.txt` with exact pinned versions
- All versions tested together and confirmed working

---

## Verification Steps

After fixing, verify each component:

```bash
# 1. Check Pydantic
python -c "import pydantic; print(f'Pydantic: {pydantic.__version__}')"
# Expected: Pydantic: 2.10.3

# 2. Check Pydantic Core
python -c "import pydantic_core; print(f'Pydantic Core: {pydantic_core.__version__}')"
# Expected: Pydantic Core: 2.27.1

# 3. Check FastAPI
python -c "import fastapi; print(f'FastAPI: {fastapi.__version__}')"
# Expected: FastAPI: 0.115.5

# 4. Check Transformers
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
# Expected: Transformers: 4.47.1

# 5. Test FastAPI imports
python -c "from fastapi import FastAPI; from pydantic import BaseModel; print('All imports OK!')"
# Expected: All imports OK!
```

---

## If Still Having Issues

### Check for Multiple Python Environments

```bash
# Show all Conda environments
conda env list

# Check which Python is active
which python   # Linux/Mac
where python   # Windows

# Should show: .../anaconda3/envs/cicerowatch/...
```

### Check Installed Package Versions

```bash
pip list | grep -E "(pydantic|fastapi|uvicorn|transformers)"

# Should show:
# fastapi          0.115.5
# pydantic         2.10.3
# pydantic-core    2.27.1
# transformers     4.47.1
# uvicorn          0.32.1
```

### Nuclear Option: Complete Reinstall

```bash
# Backup your code first!

# Remove environment
conda deactivate
conda env remove -n cicerowatch

# Create minimal environment
conda create -n cicerowatch python=3.11 pip -y
conda activate cicerowatch

# Install ONLY from requirements-conda.txt
pip install -r requirements-conda.txt

# Test
python run_local.py
```

---

## Using Docker Instead (Recommended for Production)

If Conda continues causing issues, Docker is the most reliable option:

```bash
# Navigate to project
cd "Sentiment Docker Test"

# Run with Docker (CPU version)
docker-compose up --build

# Access at http://localhost:80
```

Docker advantages:
- ✅ No dependency conflicts
- ✅ Isolated environment
- ✅ GPU support available
- ✅ Same setup everywhere

See `README.md` for full Docker setup.

---

## IDE Configuration with Conda

### VS Code

```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": "C:/Users/633568/AppData/Local/anaconda3/envs/cicerowatch/python.exe",
    "python.terminal.activateEnvironment": true,
    "python.condaPath": "C:/Users/633568/AppData/Local/anaconda3/Scripts/conda.exe"
}
```

### PyCharm

1. Settings → Project Interpreter
2. Add Interpreter → Conda Environment
3. Select existing: `cicerowatch`

### Spyder

```bash
# Install Spyder IN the Conda environment
conda activate cicerowatch
conda install spyder -y
spyder
```

---

## Common Conda/Pip Conflicts

**Problem:** Mixing `conda install` and `pip install`

**Solution:** Use pip for everything AFTER creating Conda environment:
```bash
conda create -n cicerowatch python=3.11
conda activate cicerowatch
pip install -r requirements-conda.txt  # ✅ Good
conda install fastapi  # ❌ Can cause conflicts
```

**Exception:** PyTorch is better from Conda:
```bash
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
```

---

## Version Compatibility Table

| Package | Working Version | Notes |
|---------|----------------|-------|
| Python | 3.11.x | Recommended for all packages |
| pydantic | 2.10.3 | Must match pydantic-core |
| pydantic-core | 2.27.1 | Auto-installed with pydantic |
| fastapi | 0.115.5 | Requires pydantic 2.x |
| uvicorn | 0.32.1 | Latest stable |
| transformers | 4.47.1 | Compatible with torch 2.5.1 |
| torch | 2.5.1 | Latest before 2.6 (unreleased) |

---

## Summary Checklist

- [ ] Activated Conda environment: `conda activate cicerowatch`
- [ ] Uninstalled conflicting packages: `pip uninstall pydantic pydantic-core fastapi`
- [ ] Installed from requirements-conda.txt: `pip install -r requirements-conda.txt`
- [ ] Verified pydantic works: `python -c "from pydantic import BaseModel"`
- [ ] Verified FastAPI works: `python -c "import fastapi"`
- [ ] Ran server successfully: `python run_local.py`

---

## Need More Help?

**Check versions:**
```bash
pip list | grep -E "(pydantic|fastapi)"
```

**Check Python path:**
```bash
python -c "import sys; print(sys.executable)"
# Should be: .../anaconda3/envs/cicerowatch/python.exe
```

**Still broken?** Try Docker:
```bash
docker-compose up --build
```
