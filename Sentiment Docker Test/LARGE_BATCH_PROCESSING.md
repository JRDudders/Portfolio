# Large Batch Processing Guide (3000+ Rows)

Quick guide for processing large Excel/CSV files with 3000+ rows efficiently.

## 🚀 GPU Mode (Recommended for Large Batches)

### Prerequisites
- NVIDIA GPU with 8GB+ VRAM
- Docker with nvidia-docker runtime
- Linux or WSL2 (macOS not supported)

### Start GPU Services

```bash
# Production mode with GPU acceleration
docker-compose -f docker-compose.prod.gpu.yml up --build

# Or development mode with GPU
docker-compose -f docker-compose.gpu.yml up --build
```

### Environment Variables for Maximum Performance

Create a `.env` file or export these before starting Docker:

```bash
# Optimized batch sizes for GPU (adjust based on VRAM)
export STANCE_BATCH_SIZE=64          # Default: 32
export CLASSIFY_BATCH_SIZE=32        # Default: 16
export ZEROSHOT_BATCH_SIZE=16        # Default: 8

# Increase timeouts for large files
export REQUEST_TIMEOUT=7200          # 2 hours
```

### Expected Performance

| Task Type | 3000 Rows (GPU) | 3000 Rows (CPU) | Speedup |
|-----------|----------------|-----------------|---------|
| Sentiment Analysis | 3-6 minutes | 15-25 minutes | 3-5x |
| Stance Detection | 6-12 minutes | 30-45 minutes | 4-5x |
| Zero-shot Classification | 12-20 minutes | 45-90 minutes | 3-4x |

---

## 💻 CPU Mode (No GPU Available)

If you don't have a GPU, CPU mode still works but is slower:

```bash
# Production CPU mode
docker-compose -f docker-compose.prod.yml up --build
```

### Optimized CPU Batch Sizes

```bash
# Conservative batch sizes for CPU
export STANCE_BATCH_SIZE=16          # Default: 32
export CLASSIFY_BATCH_SIZE=8         # Default: 16
export ZEROSHOT_BATCH_SIZE=4         # Default: 8
```

**Note:** Smaller batches on CPU can sometimes be faster due to memory pressure.

---

## 📊 Processing Large Excel Files

### Via Web UI

1. Navigate to http://localhost:8080
2. Upload your Excel file (.xlsx)
3. Select the appropriate preset (sentiment, stance, etc.)
4. Wait for processing (progress shown in browser console)
5. Download results as JSON

### Via API (Recommended for Very Large Files)

```python
import requests
import pandas as pd

# Read your Excel file
df = pd.read_excel("large_dataset.xlsx")

# Extract text column
texts = df['text_column'].tolist()

# Process in batches via API
url = "http://localhost:8080/predict/batch"
response = requests.post(url, json={
    "texts": texts,
    "preset": "sentiment-twitter",
    "preprocess": True
})

results = response.json()["results"]

# Add results back to dataframe
df['sentiment'] = [r['topics'] for r in results]
df.to_excel("processed_output.xlsx", index=False)
```

---

## 🐛 Troubleshooting

### "Processing took 4+ hours then failed"

**Problem:** Server timeout or nginx error page received
**Solution:**
- ✅ Now fixed with HTML error detection (latest update)
- Increase REQUEST_TIMEOUT environment variable
- Check server logs for memory issues

### "Out of Memory" Error

**Problem:** GPU VRAM exhausted or system RAM full
**Solutions:**
- Reduce batch sizes (try halving them)
- Process file in chunks (split Excel into smaller files)
- Close other GPU applications

### GPU Not Detected

**Check Docker GPU access:**
```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

**Check service logs:**
```bash
docker-compose logs nlp | grep -i "gpu\|cuda"
```

Should see: `[nlp] Loading model on GPU (CUDA)`

---

## 📈 Monitoring Progress

### View real-time logs:

```bash
# NLP service logs
docker-compose logs -f nlp

# Graph service logs (for social media edge extraction)
docker-compose logs -f graph
```

Look for messages like:
```
[classify] Processing 3247 texts (4891 chunks) in batches
```

---

## 🎯 Best Practices

1. **Use GPU mode** for files >1000 rows
2. **Increase batch sizes** on GPU for maximum throughput
3. **Monitor VRAM usage** with `nvidia-smi` during processing
4. **Split very large files** (>10,000 rows) into chunks
5. **Save intermediate results** to avoid reprocessing

---

## 💡 Example: Processing 3000-Row Twitter Dataset

```bash
# 1. Start GPU services with optimized settings
export CLASSIFY_BATCH_SIZE=32
docker-compose -f docker-compose.prod.gpu.yml up -d

# 2. Upload via web UI at http://localhost:8080
#    - Select file: tweets.xlsx
#    - Preset: sentiment-twitter
#    - Click "Analyze"

# 3. Expected time: 4-8 minutes on GPU
# 4. Download results as tweets_predictions.json
```

---

For more information, see:
- [GPU_SETUP.md](GPU_SETUP.md) - Detailed GPU configuration
- [DOCKER_SETUP.md](DOCKER_SETUP.md) - Docker installation and setup
