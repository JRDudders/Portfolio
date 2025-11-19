@echo off
REM Conda installation script for Windows
REM Run this in Anaconda Prompt with cicerowatch environment activated

echo ============================================================
echo Installing CiceroWatch Dependencies (Conda Environment)
echo ============================================================
echo.

REM Check if conda environment is activated
if "%CONDA_DEFAULT_ENV%"=="cicerowatch" (
    echo [OK] cicerowatch environment is active
) else (
    echo [ERROR] cicerowatch environment is NOT active
    echo Please run: conda activate cicerowatch
    pause
    exit /b 1
)

echo.
echo Step 1: Installing PyTorch via conda (CPU version)...
echo --------------------------------------------------------
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

if errorlevel 1 (
    echo [ERROR] PyTorch installation failed
    pause
    exit /b 1
)

echo.
echo Step 2: Installing transformers and core ML libraries via pip...
echo --------------------------------------------------------
pip install transformers==4.47.1 tokenizers==0.21.0 safetensors==0.4.5 sentencepiece==0.2.0 sentence-transformers==3.3.1

if errorlevel 1 (
    echo [ERROR] Transformers installation failed
    pause
    exit /b 1
)

echo.
echo Step 3: Installing FastAPI and web frameworks...
echo --------------------------------------------------------
pip install fastapi==0.115.5 uvicorn==0.32.1 python-multipart==0.0.18 pydantic==2.10.3

if errorlevel 1 (
    echo [ERROR] FastAPI installation failed
    pause
    exit /b 1
)

echo.
echo Step 4: Installing data processing libraries...
echo --------------------------------------------------------
pip install pandas numpy scikit-learn openpyxl

if errorlevel 1 (
    echo [ERROR] Data processing libraries installation failed
    pause
    exit /b 1
)

echo.
echo Step 5: Installing NLP libraries...
echo --------------------------------------------------------
pip install nltk spacy stanza

if errorlevel 1 (
    echo [ERROR] NLP libraries installation failed
    pause
    exit /b 1
)

echo.
echo Step 6: Installing web scraping libraries...
echo --------------------------------------------------------
pip install beautifulsoup4 lxml trafilatura readability-lxml requests httpx

if errorlevel 1 (
    echo [ERROR] Web scraping libraries installation failed
    pause
    exit /b 1
)

echo.
echo Step 7: Verifying installation...
echo --------------------------------------------------------
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import transformers; print('Transformers:', transformers.__version__)"
python -c "import fastapi; print('FastAPI:', fastapi.__version__)"
python -c "import pandas; print('Pandas:', pandas.__version__)"

if errorlevel 1 (
    echo [ERROR] Verification failed
    pause
    exit /b 1
)

echo.
echo ============================================================
echo [SUCCESS] All dependencies installed successfully!
echo ============================================================
echo.
echo You can now run: python run_local.py
echo.
pause
