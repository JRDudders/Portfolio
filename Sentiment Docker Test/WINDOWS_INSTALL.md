# Windows Installation Fix - DLL Access Denied

## The Problem

**Error:**
```
OSError: [WinError 5] Access is denied: 'C:\...\torch\lib\c10.dll'
```

**Cause:** PyTorch DLL files are locked because they're currently loaded in memory (by another Python process, Jupyter notebook, or the server itself).

---

## ✅ Solution 1: Close Everything & Reinstall (RECOMMENDED)

### Step 1: Close All Python Processes

```bash
# Open a NEW Command Prompt/PowerShell as Administrator
# Right-click Command Prompt -> "Run as administrator"

# Kill all Python processes
taskkill /F /IM python.exe /T
taskkill /F /IM pythonw.exe /T

# Wait 5 seconds
timeout /t 5

# Close Jupyter if running
jupyter notebook stop
taskkill /F /IM jupyter-notebook.exe /T
```

### Step 2: Fresh Installation

```bash
# In the SAME administrator terminal
cd "path\to\Sentiment Docker Test"

# Activate your Conda environment
conda activate main

# Uninstall old versions (ignore errors)
pip uninstall torch transformers tokenizers -y

# Install with --force-reinstall
pip install --force-reinstall --no-cache-dir torch==2.3.0 transformers==4.41.2
```

### Step 3: Verify & Run

```bash
# Check it worked
python check_compatibility.py

# Start the server
python run_local.py
```

---

## ✅ Solution 2: Install with --user Flag

If you can't run as administrator:

```bash
# Close all Python processes first
taskkill /F /IM python.exe /T

# Install to user directory (doesn't need admin)
pip install --user --force-reinstall torch==2.3.0 transformers==4.41.2

# Start server
python run_local.py
```

---

## ✅ Solution 3: Use a New Conda Environment (CLEANEST)

Create a fresh environment without conflicts:

```bash
# Close all Python/Jupyter processes
taskkill /F /IM python.exe /T
taskkill /F /IM jupyter-notebook.exe /T

# Create new environment
conda create -n sentiment python=3.11 -y
conda activate sentiment

# Navigate to project
cd "path\to\Sentiment Docker Test"

# Install dependencies (clean slate)
pip install -r requirements-local.txt

# Download models
python -m spacy download en_core_web_sm

# Run server
python run_local.py
```

---

## ✅ Solution 4: Install in Safe Mode

If DLLs are still locked, restart Windows and install immediately:

1. **Close ALL applications**
2. **Restart Windows**
3. **Immediately after restart**, open Command Prompt as Administrator
4. **Before opening anything else**, run:

```bash
conda activate main
cd "path\to\Sentiment Docker Test"
pip install --force-reinstall --no-cache-dir torch==2.3.0 transformers==4.41.2
```

---

## ✅ Solution 5: Check What's Locking the File

Find which process is using PyTorch:

### Method A: Task Manager
1. Open Task Manager (Ctrl+Shift+Esc)
2. Go to "Details" tab
3. Look for:
   - `python.exe`
   - `pythonw.exe`
   - `jupyter-notebook.exe`
   - `code.exe` (VS Code)
4. Right-click → "End task" on all Python processes

### Method B: Using PowerShell (as Admin)
```powershell
# Find what's using the DLL
Get-Process | Where-Object {$_.Modules.FileName -like "*torch*"} | Select-Object ProcessName, Id

# Kill those processes
Stop-Process -Name "python" -Force
Stop-Process -Name "pythonw" -Force
```

---

## ✅ Solution 6: Disable Antivirus Temporarily

Sometimes antivirus locks DLLs during installation:

1. **Temporarily disable** Windows Defender or your antivirus
2. **Uninstall and reinstall:**
   ```bash
   pip uninstall torch transformers -y
   pip install torch==2.3.0 transformers==4.41.2
   ```
3. **Re-enable** antivirus

---

## ✅ Solution 7: Install from Conda (Alternative)

Use Conda instead of pip for PyTorch:

```bash
# Close all Python processes
taskkill /F /IM python.exe /T

# Install PyTorch from Conda (might avoid DLL issues)
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# Then install transformers via pip
pip install transformers==4.41.2

# Install other deps
pip install fastapi uvicorn pandas numpy requests python-multipart
```

---

## 🔍 Verify What's Running

Before installation, check what's using Python:

```bash
# PowerShell
Get-Process python*, jupyter* | Format-Table ProcessName, Id, Path -AutoSize

# Command Prompt
tasklist | findstr /I python
wmic process where "name like '%python%'" get ProcessID,ExecutablePath
```

---

## 📋 Complete Fresh Install Checklist

Follow these steps in order:

```bash
# 1. Close EVERYTHING
taskkill /F /IM python.exe /T
taskkill /F /IM pythonw.exe /T
taskkill /F /IM jupyter-notebook.exe /T
taskkill /F /IM code.exe /T  # VS Code if using

# 2. Wait a moment
timeout /t 5

# 3. Open NEW Command Prompt as Administrator

# 4. Activate environment
conda activate main

# 5. Navigate to project
cd "C:\Users\jrdud\Portfolio\Sentiment Docker Test"

# 6. Clear pip cache
pip cache purge

# 7. Uninstall (ignore errors)
pip uninstall torch transformers tokenizers safetensors -y

# 8. Fresh install
pip install --force-reinstall --no-cache-dir torch==2.3.0 transformers==4.41.2

# 9. Install other requirements
pip install fastapi "uvicorn<0.30.0" pandas numpy requests python-multipart beautifulsoup4 lxml spacy

# 10. Download models
python -m spacy download en_core_web_sm

# 11. Check compatibility
python check_compatibility.py

# 12. Start server
python run_local.py
```

---

## 🆘 Still Getting Errors?

### If installation still fails:

1. **Restart your computer**
2. **Immediately open Command Prompt as Admin** (before any other programs)
3. Run the "Complete Fresh Install Checklist" above

### If you keep getting DLL errors:

**Use a Virtual Environment instead of Conda:**

```bash
# In Command Prompt as Admin
cd "C:\Users\jrdud\Portfolio\Sentiment Docker Test"

# Deactivate Conda
conda deactivate

# Create Python venv
python -m venv venv_sentiment

# Activate it
venv_sentiment\Scripts\activate

# Fresh install
pip install -r requirements-local.txt

# Run server
python run_local.py
```

---

## ⚡ Quick Test After Installation

```bash
# 1. Check versions
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import transformers; print('Transformers:', transformers.__version__)"

# 2. Run compatibility check
python check_compatibility.py

# 3. Start server
python run_local.py

# 4. In another terminal, test
curl http://localhost:8080/healthz
```

---

## 💡 Pro Tips

1. **Always close Jupyter** before pip installing PyTorch
2. **Use `--user` flag** if you don't have admin rights
3. **Create a new Conda environment** for clean slate
4. **Restart Windows** if DLLs are persistently locked
5. **Use Conda for PyTorch**, pip for everything else

---

## Common Mistakes

❌ **Don't do this:**
- Installing while Jupyter is running
- Installing while the server is running
- Installing in VS Code terminal while Python extension is active

✅ **Do this:**
- Close ALL Python processes first
- Use a fresh Command Prompt
- Run as Administrator if possible
