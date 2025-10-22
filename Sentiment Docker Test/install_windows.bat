@echo off
REM Windows Installation Script for Sentiment Analysis API
REM Run this as Administrator: Right-click -> "Run as administrator"

echo ============================================================
echo Sentiment Analysis API - Windows Installation
echo ============================================================
echo.
echo This script will:
echo   1. Kill all Python processes
echo   2. Uninstall conflicting packages
echo   3. Install compatible versions
echo   4. Download required models
echo.
echo Press Ctrl+C to cancel, or
pause

echo.
echo [1/5] Killing all Python processes...
taskkill /F /IM python.exe /T >nul 2>&1
taskkill /F /IM pythonw.exe /T >nul 2>&1
taskkill /F /IM jupyter-notebook.exe /T >nul 2>&1
echo Done!

echo.
echo [2/5] Waiting for processes to fully terminate...
timeout /t 3 /nobreak >nul
echo Done!

echo.
echo [3/5] Uninstalling old versions...
pip uninstall torch transformers tokenizers safetensors -y
echo Done!

echo.
echo [4/5] Installing compatible versions...
echo (This may take a few minutes)
pip install --force-reinstall --no-cache-dir torch==2.3.0 transformers==4.41.2

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Installation failed!
    echo.
    echo Try these alternatives:
    echo   1. Run this script as Administrator
    echo   2. Install with --user flag: pip install --user torch==2.3.0 transformers==4.41.2
    echo   3. See WINDOWS_INSTALL.md for more solutions
    pause
    exit /b 1
)

echo.
echo [5/5] Installing other dependencies...
pip install fastapi "uvicorn<0.30.0" pandas numpy requests python-multipart beautifulsoup4 lxml spacy

echo.
echo [6/6] Downloading spaCy English model...
python -m spacy download en_core_web_sm

echo.
echo ============================================================
echo Installation Complete!
echo ============================================================
echo.
echo Next steps:
echo   1. Run compatibility check: python check_compatibility.py
echo   2. Start the server: python run_local.py
echo   3. Test: curl http://localhost:8080/healthz
echo.
pause
