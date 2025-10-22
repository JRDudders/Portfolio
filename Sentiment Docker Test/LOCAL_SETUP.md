# Local Development Setup - Conda Environment Fix

## The Problem

When running the API locally in a Conda environment, you may encounter this error:

```
TypeError: _patch_asyncio.<locals>.run() got an unexpected keyword argument 'loop_factory'
```

This is due to Conda's asyncio patching conflicting with uvicorn's use of the `loop_factory` argument.

## Quick Fix - Use the Startup Script

**Recommended:** Use the provided startup script which automatically tries multiple solutions:

```bash
cd "Sentiment Docker Test"
python run_local.py
```

The script will automatically:
1. Try standard uvicorn
2. Patch asyncio if needed
3. Fall back to hypercorn (if installed)
4. Provide manual instructions if all fail

---

## Manual Solutions (Pick One)

### Solution 1: Downgrade uvicorn (Easiest)

```bash
pip install "uvicorn<0.30.0"
python -m uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```

### Solution 2: Install Hypercorn (Recommended for Conda)

Hypercorn is more compatible with Conda environments:

```bash
pip install hypercorn
hypercorn app:app --bind 0.0.0.0:8080 --reload
```

### Solution 3: Upgrade uvicorn (May work)

```bash
pip install --upgrade uvicorn
python -m uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```

### Solution 4: Use a Virtual Environment (Best long-term)

Exit Conda and use venv instead:

```bash
# Exit Conda
conda deactivate

# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r req.txt

# Run server
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```

---

## Verify Installation

Once the server starts, test it:

```bash
# Check health
curl http://localhost:8080/healthz

# Or open in browser:
# http://localhost:8080
# http://localhost:8080/docs (API documentation)
```

---

## Quick Test Commands

### Test Sentiment Analysis (Batch)

```bash
curl -X POST "http://localhost:8080/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["I love this!", "This is terrible."],
    "preset": "sentiment-twitter"
  }'
```

### Test Graph Analytics

First create a test graph file `test_edges.csv`:
```csv
src,dst
Alice,Bob
Bob,Charlie
Charlie,Alice
Alice,David
```

Then run:
```bash
curl -X POST "http://localhost:8080/graph/degrees" \
  -F "file=@test_edges.csv"
```

---

## Dependencies

Install all required packages:

```bash
# Core dependencies (minimal)
pip install fastapi uvicorn transformers torch pandas numpy requests python-multipart

# NLP dependencies
pip install beautifulsoup4 lxml trafilatura nltk spacy

# Graph dependencies
pip install scikit-learn

# Download models
python -m spacy download en_core_web_sm
```

For the full feature set, install from `req.txt`:

```bash
pip install -r req.txt
```

---

## Environment Variables

Optional configuration via environment variables:

```bash
# Set port (default: 8080)
export PORT=8000

# Disable Playwright (if not needed)
export PLAYWRIGHT_AVAILABLE=false

# Set Reddit API limit
export REDDIT_LIMIT=100
```

---

## Troubleshooting

### Issue: "No module named 'app'"

Make sure you're in the correct directory:
```bash
cd "Sentiment Docker Test"
```

### Issue: Import errors for NLP libraries

Some features require additional downloads:
```bash
# spaCy model
python -m spacy download en_core_web_sm

# NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

### Issue: Port already in use

Change the port:
```bash
python -m uvicorn app:app --host 0.0.0.0 --port 8081 --reload
```

Or find and kill the process using port 8080:
```bash
# Windows
netstat -ano | findstr :8080
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:8080 | xargs kill
```

### Issue: CUDA/GPU errors

The API works fine on CPU. If you see CUDA errors:
```bash
# Force CPU mode
export CUDA_VISIBLE_DEVICES=""
python run_local.py
```

---

## Performance Tips

- First request will be slow (model loading)
- Subsequent requests are fast (models cached in memory)
- Use `/predict/batch` for multiple texts (much faster than individual calls)
- Graph analytics on large graphs (>10k nodes) may take time

---

## Support

See `API_DOCUMENTATION.md` for full API documentation and examples.

For issues with dependencies or setup, check:
- Python version: 3.10+ recommended
- Conda environment activated
- All dependencies installed
