# Windows Installation Script for Sentiment Analysis API
# Run as Administrator: Right-click -> "Run with PowerShell"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Sentiment Analysis API - Windows Installation" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Check if running as Administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "WARNING: Not running as Administrator" -ForegroundColor Yellow
    Write-Host "Some operations may fail. Recommend:" -ForegroundColor Yellow
    Write-Host "  Right-click this script -> 'Run as Administrator'" -ForegroundColor Yellow
    Write-Host ""
    $continue = Read-Host "Continue anyway? (y/n)"
    if ($continue -ne "y") {
        exit
    }
}

Write-Host "This script will:" -ForegroundColor Green
Write-Host "  1. Kill all Python processes" -ForegroundColor Green
Write-Host "  2. Uninstall conflicting packages" -ForegroundColor Green
Write-Host "  3. Install compatible versions" -ForegroundColor Green
Write-Host "  4. Download required models" -ForegroundColor Green
Write-Host ""
$confirm = Read-Host "Continue? (y/n)"
if ($confirm -ne "y") {
    exit
}

# Step 1: Kill Python processes
Write-Host ""
Write-Host "[1/6] Killing all Python processes..." -ForegroundColor Cyan
try {
    Stop-Process -Name "python" -Force -ErrorAction SilentlyContinue
    Stop-Process -Name "pythonw" -Force -ErrorAction SilentlyContinue
    Stop-Process -Name "jupyter-notebook" -Force -ErrorAction SilentlyContinue
    Write-Host "  Done!" -ForegroundColor Green
} catch {
    Write-Host "  Warning: Some processes couldn't be killed" -ForegroundColor Yellow
}

# Step 2: Wait
Write-Host ""
Write-Host "[2/6] Waiting for processes to terminate..." -ForegroundColor Cyan
Start-Sleep -Seconds 3
Write-Host "  Done!" -ForegroundColor Green

# Step 3: Clear pip cache
Write-Host ""
Write-Host "[3/6] Clearing pip cache..." -ForegroundColor Cyan
pip cache purge
Write-Host "  Done!" -ForegroundColor Green

# Step 4: Uninstall old versions
Write-Host ""
Write-Host "[4/6] Uninstalling old versions..." -ForegroundColor Cyan
pip uninstall torch transformers tokenizers safetensors -y
Write-Host "  Done!" -ForegroundColor Green

# Step 5: Install compatible versions
Write-Host ""
Write-Host "[5/6] Installing PyTorch and Transformers..." -ForegroundColor Cyan
Write-Host "  (This may take several minutes)" -ForegroundColor Yellow

$installCmd = "pip install --force-reinstall --no-cache-dir torch==2.3.0 transformers==4.41.2"
Write-Host "  Running: $installCmd" -ForegroundColor Gray

try {
    Invoke-Expression $installCmd
    if ($LASTEXITCODE -ne 0) {
        throw "Installation failed with exit code $LASTEXITCODE"
    }
    Write-Host "  Done!" -ForegroundColor Green
} catch {
    Write-Host ""
    Write-Host "ERROR: Installation failed!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Try these alternatives:" -ForegroundColor Yellow
    Write-Host "  1. Run this script as Administrator" -ForegroundColor Yellow
    Write-Host "  2. Install with --user flag:" -ForegroundColor Yellow
    Write-Host "     pip install --user torch==2.3.0 transformers==4.41.2" -ForegroundColor Yellow
    Write-Host "  3. See WINDOWS_INSTALL.md for more solutions" -ForegroundColor Yellow
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

# Step 6: Install other dependencies
Write-Host ""
Write-Host "[6/6] Installing other dependencies..." -ForegroundColor Cyan
pip install fastapi "uvicorn<0.30.0" pandas numpy requests python-multipart beautifulsoup4 lxml spacy

# Download spaCy model
Write-Host ""
Write-Host "Downloading spaCy English model..." -ForegroundColor Cyan
python -m spacy download en_core_web_sm

# Success
Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "Installation Complete!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Run compatibility check:" -ForegroundColor White
Write-Host "     python check_compatibility.py" -ForegroundColor Gray
Write-Host "  2. Start the server:" -ForegroundColor White
Write-Host "     python run_local.py" -ForegroundColor Gray
Write-Host "  3. Test it works:" -ForegroundColor White
Write-Host "     curl http://localhost:8080/healthz" -ForegroundColor Gray
Write-Host ""
Read-Host "Press Enter to exit"
