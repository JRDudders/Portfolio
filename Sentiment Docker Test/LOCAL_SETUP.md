# Local Development Setup Guide

## Problem
Getting error: `No module named 'transformers'` when running `run_local.py`

## Solution: Create Virtual Environment & Configure IDE

---

## Step 1: Create Virtual Environment

Navigate to the project directory in your terminal:

```bash
cd "Sentiment Docker Test"

# Create virtual environment
python -m venv venv

# Activate it:
# On Windows (PowerShell):
venv\Scripts\Activate.ps1

# On Windows (CMD):
venv\Scripts\activate.bat

# On Linux/Mac:
source venv/bin/activate
```

---

## Step 2: Install Dependencies

Once activated (you'll see `(venv)` in your prompt):

```bash
# Install NLP dependencies (includes transformers)
pip install -r requirements-nlp.txt

# Optional: Install other service dependencies
pip install -r requirements-local.txt  # If you need all services
pip install -r requirements-graph.txt  # For graph analytics
pip install -r requirements-audio.txt  # For audio processing
```

**Note:** For GPU support with PyTorch:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## Step 3: Verify Installation

```bash
# Check transformers is installed
python -c "import transformers; print(transformers.__version__)"

# Should print version number (e.g., 4.40.0)
```

---

## Step 4: Configure Your IDE

### 🔵 **VS Code**

1. **Open Command Palette**: `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (Mac)
2. **Select Interpreter**: Type "Python: Select Interpreter"
3. **Choose your venv**: Select the one that shows:
   - `./venv/Scripts/python.exe` (Windows)
   - `./venv/bin/python` (Linux/Mac)

**Verify in VS Code:**
- Bottom-right corner should show: `Python 3.x.x ('venv': venv)`
- Terminal should auto-activate venv when opened

**Alternative Method:**
```json
// Create .vscode/settings.json in project root:
{
    "python.defaultInterpreterPath": "${workspaceFolder}/venv/Scripts/python.exe",
    "python.terminal.activateEnvironment": true
}
```

---

### 🔴 **PyCharm**

1. **Open Settings**: `File` → `Settings` (Windows/Linux) or `PyCharm` → `Preferences` (Mac)
2. **Navigate**: `Project: Sentiment Docker Test` → `Python Interpreter`
3. **Add Interpreter**:
   - Click the gear icon ⚙️ → `Add`
   - Select `Existing environment`
   - Browse to: `Sentiment Docker Test/venv/Scripts/python.exe` (Windows) or `.../venv/bin/python` (Linux/Mac)
   - Click `OK`

**Verify in PyCharm:**
- Bottom-right corner shows: `Python 3.x (venv)`
- Run configuration uses correct interpreter

**Quick Test:**
- Right-click `run_local.py` → `Run 'run_local'`
- Should work without import errors

---

### 🟢 **Spyder**

**Option 1: Launch Spyder FROM the venv** (Recommended)

```bash
# Activate venv first
cd "Sentiment Docker Test"
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac

# Install Spyder in the venv
pip install spyder

# Launch Spyder from within venv
spyder
```

**Option 2: Change Interpreter in Spyder**

1. **Open Preferences**: `Tools` → `Preferences`
2. **Navigate**: `Python Interpreter`
3. **Select**: `Use the following Python interpreter:`
4. **Browse to**: `Sentiment Docker Test/venv/Scripts/python.exe`
5. **Restart Spyder**

**Verify in Spyder:**
```python
# In IPython console, run:
import sys
print(sys.executable)
# Should show path to your venv Python

import transformers
print(transformers.__version__)
# Should work without errors
```

---

### 🟡 **Terminal / Command Line**

You must **manually activate** the venv each time:

```bash
# Navigate to project
cd "Sentiment Docker Test"

# Activate venv:
# Windows PowerShell:
venv\Scripts\Activate.ps1

# Windows CMD:
venv\Scripts\activate.bat

# Linux/Mac:
source venv/bin/activate

# Verify activation (you should see "(venv)" prefix):
which python  # Linux/Mac
where python  # Windows

# Now run your script:
python run_local.py
```

**Deactivate when done:**
```bash
deactivate
```

---

## Step 5: Run the Local Server

Once your IDE/terminal is using the venv:

```bash
python run_local.py
```

You should see:
```
============================================================
Sentiment Analysis & Graph Analytics API - Local Server
============================================================

🔍 GPU Status Check:
⚠ No CUDA GPU detected (using CPU)
...

🚀 Starting server with Standard uvicorn
📍 http://localhost:8080
📚 API docs: http://localhost:8080/docs
```

Access the API at: **http://localhost:8080/docs**

---

## Troubleshooting

### ❌ "No module named 'transformers'" persists

**Check which Python is running:**
```bash
# In your IDE's terminal/console:
import sys
print(sys.executable)

# Should show path to venv Python, not system Python
# Good: /path/to/Sentiment Docker Test/venv/bin/python
# Bad:  /usr/bin/python or C:\Python3\python.exe
```

**Fix:** Restart your IDE after selecting the venv interpreter

---

### ❌ "venv\Scripts\activate.ps1 cannot be loaded"

PowerShell execution policy issue. Run as Administrator:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then try activating again.

---

### ❌ Still not working in Spyder

Spyder sometimes caches the interpreter. Try:
1. Close Spyder completely
2. Delete `.spyproject/` folder in your project directory
3. Activate venv in terminal
4. Launch Spyder from within the venv: `spyder`

---

### ❌ Out of Memory / Slow Performance

The NLP models are large. For local development:
- **Use CPU mode** (default in run_local.py)
- **Close other applications** to free RAM
- **Consider using Docker** instead for better resource management

---

## Quick Reference: Virtual Environment Commands

| Task | Windows (PowerShell) | Windows (CMD) | Linux/Mac |
|------|---------------------|---------------|-----------|
| **Create venv** | `python -m venv venv` | `python -m venv venv` | `python3 -m venv venv` |
| **Activate** | `venv\Scripts\Activate.ps1` | `venv\Scripts\activate.bat` | `source venv/bin/activate` |
| **Deactivate** | `deactivate` | `deactivate` | `deactivate` |
| **Check Python** | `where python` | `where python` | `which python` |
| **Check packages** | `pip list` | `pip list` | `pip list` |

---

## Why Docker is Recommended

This project was designed for Docker because:
- ✅ **Isolated environment** - No conflicts with system Python
- ✅ **GPU support** - CUDA/cuGraph pre-configured
- ✅ **All dependencies** - Everything pre-installed
- ✅ **Consistent** - Works the same everywhere

**To use Docker instead:**
```bash
docker-compose up --build
# Access at http://localhost:80
```

See main README.md for full Docker setup.

---

## Summary Checklist

- [ ] Created virtual environment: `python -m venv venv`
- [ ] Activated venv: `venv\Scripts\activate` or `source venv/bin/activate`
- [ ] Installed dependencies: `pip install -r requirements-nlp.txt`
- [ ] Verified installation: `python -c "import transformers"`
- [ ] Configured IDE to use venv interpreter
- [ ] Restarted IDE
- [ ] Ran `python run_local.py` successfully

---

**Need help?** Check which Python is running:
```python
import sys
print("Python:", sys.executable)
print("Version:", sys.version)
```

This should point to your venv, not system Python!
