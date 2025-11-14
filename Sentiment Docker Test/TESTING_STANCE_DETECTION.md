# Testing Stance Detection

## Prerequisites

Ensure your Docker containers are running:
```bash
docker-compose up -d
```

## Quick Test via UI

1. **Open the application:**
   - Navigate to `http://localhost` in your browser
   - Click on the **NLP** tab

2. **Find the Stance Detection card:**
   - Scroll down to find "Stance Detection (NLI-based)"

3. **Enter a claim:**
   ```
   Climate change is caused by human activity
   ```

4. **Enter texts to classify (one per line):**
   ```
   Climate change is the defining crisis of our time
   The climate has always changed naturally
   Scientists agree on human-caused warming
   Weather varies from year to year
   Fossil fuels are destroying our planet
   ```

5. **Select model:**
   - **DeBERTa-v3-base** (fast, ~50ms/text) - Recommended for testing
   - **DeBERTa-v3-large** (accurate, ~150ms/text) - For production

6. **Click "Detect Stance"**

7. **Expected results:**
   - Text 1: **SUPPORT** (high confidence)
   - Text 2: **OPPOSE** (moderate-high confidence)
   - Text 3: **SUPPORT** (high confidence)
   - Text 4: **NEUTRAL** (moderate confidence)
   - Text 5: **SUPPORT** (high confidence)

8. **Review the results table:**
   - Green = SUPPORT
   - Red = OPPOSE
   - Orange = NEUTRAL
   - Confidence scores shown as percentages

9. **Download results:**
   - Click "Download Results (JSON)" to save the full output

## Test via API (curl)

### Basic Test

```bash
curl -X POST http://localhost/api/nlp/stance \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Nuclear power is our cleanest energy source",
      "Nuclear waste is dangerous for thousands of years"
    ],
    "claim": "Nuclear energy should be expanded",
    "preset": "stance-deberta"
  }'
```

**Expected output:**
```json
{
  "success": true,
  "claim": "Nuclear energy should be expanded",
  "preset": "stance-deberta",
  "results": [
    {
      "stance": "SUPPORT",
      "scores": {
        "SUPPORT": 0.87,
        "OPPOSE": 0.06,
        "NEUTRAL": 0.07
      },
      "claim": "Nuclear energy should be expanded"
    },
    {
      "stance": "OPPOSE",
      "scores": {
        "SUPPORT": 0.12,
        "OPPOSE": 0.76,
        "NEUTRAL": 0.12
      },
      "claim": "Nuclear energy should be expanded"
    }
  ]
}
```

### Political Stance Test

```bash
curl -X POST http://localhost/api/nlp/stance \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Everyone deserves access to healthcare regardless of income",
      "Government healthcare leads to rationing and inefficiency",
      "Healthcare costs are crushing American families"
    ],
    "claim": "Universal healthcare is a human right",
    "preset": "stance-deberta-large"
  }'
```

### Misinformation Detection Test

```bash
curl -X POST http://localhost/api/nlp/stance \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Peer-reviewed studies found no link between vaccines and autism",
      "The original vaccine-autism study was retracted for fraud",
      "Many parents report autism symptoms after vaccination"
    ],
    "claim": "Vaccines cause autism",
    "preset": "stance-deberta"
  }'
```

## Test via Python

### Simple Test

```python
import requests

response = requests.post(
    "http://localhost/api/nlp/stance",
    json={
        "texts": [
            "Climate science is settled",
            "Climate has always changed"
        ],
        "claim": "Climate change is real",
        "preset": "stance-deberta"
    }
)

results = response.json()
print(f"Success: {results['success']}")
for i, result in enumerate(results['results']):
    print(f"\nText {i+1}: {result['stance']}")
    print(f"Confidence: {result['scores'][result['stance']]:.1%}")
```

### Batch Processing Test

```python
import requests
import pandas as pd

# Load your dataset
df = pd.read_csv("social_media_posts.csv")

# Process in batches
batch_size = 32
claim = "The new policy benefits citizens"

all_results = []
for i in range(0, len(df), batch_size):
    batch = df['text'].iloc[i:i+batch_size].tolist()

    response = requests.post(
        "http://localhost/api/nlp/stance",
        json={
            "texts": batch,
            "claim": claim,
            "preset": "stance-deberta"
        }
    )

    all_results.extend(response.json()['results'])
    print(f"Processed {min(i+batch_size, len(df))}/{len(df)}")

# Add to dataframe
df['stance'] = [r['stance'] for r in all_results]
df['stance_confidence'] = [r['scores'][r['stance']] for r in all_results]

# Analyze
print(df['stance'].value_counts())
```

## Verify Model Download

On first use, the DeBERTa models will download from Hugging Face:

```bash
# Check NLP container logs
docker logs sentiment_docker_test-nlp-cpu-1 2>&1 | tail -50

# You should see model download progress:
# Downloading (…)lve/main/config.json: 100%
# Downloading pytorch_model.bin: 100%
```

**Model sizes:**
- `stance-deberta` (microsoft/deberta-v3-base-mnli): ~420MB
- `stance-deberta-large` (microsoft/deberta-v3-large-mnli): ~1.3GB

**Download time:**
- First use: 1-2 minutes (one-time download)
- Subsequent uses: Instant (cached)

## Performance Benchmarks

Run this to test performance:

```python
import requests
import time

texts = ["This is a test statement"] * 10
claim = "Test claims are useful"

start = time.time()
response = requests.post(
    "http://localhost/api/nlp/stance",
    json={"texts": texts, "claim": claim, "preset": "stance-deberta"}
)
elapsed = time.time() - start

print(f"Processed {len(texts)} texts in {elapsed:.2f}s")
print(f"Average: {elapsed/len(texts)*1000:.0f}ms per text")
```

**Expected performance (CPU):**
- stance-deberta: ~50ms per text
- stance-deberta-large: ~150ms per text

**Expected performance (GPU):**
- stance-deberta: ~10ms per text
- stance-deberta-large: ~30ms per text

## Troubleshooting

### Model not downloading

```bash
# Check internet connectivity from container
docker exec sentiment_docker_test-nlp-cpu-1 curl -I https://huggingface.co

# Check Hugging Face Hub access
docker exec sentiment_docker_test-nlp-cpu-1 python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('microsoft/deberta-v3-base-mnli')"
```

### Out of memory errors

If you see OOM errors with the large model:

1. Use `stance-deberta` (base model) instead
2. Reduce batch size (process fewer texts at once)
3. Increase Docker memory limit in docker-compose.yml

### Slow performance

1. **Use GPU version** if available (service: nlp-gpu)
2. **Use base model** for faster inference
3. **Batch requests** instead of one-by-one
4. **Cache results** for repeated claim+text pairs

### API endpoint not found

```bash
# Verify NLP service is running
docker ps | grep nlp

# Check nginx routing
curl http://localhost/api/nlp/health

# Should return: {"service":"nlp","status":"healthy"}
```

## Integration with Graph Analytics

To analyze social media network stance:

1. **Extract network:**
   - Go to Graph Analytics tab
   - Upload CSV with social media posts
   - Extract edges and download nodes CSV

2. **Get author posts:**
   - Nodes CSV includes 'content' field with user posts

3. **Run stance detection:**
   - Use /stance endpoint with posts from specific users
   - Classify their stance on a topic

4. **Visualize:**
   - Color nodes by stance in graph visualization
   - Identify echo chambers (clusters with same stance)
   - Find bridge users (connecting different stances)

## Next Steps

1. Try the examples in `STANCE_DETECTION_EXAMPLES.md`
2. Review architecture details in `NLI_ARCHITECTURE.md`
3. Integrate with your existing workflows
4. Monitor performance metrics

## Support

If you encounter issues:
1. Check Docker logs: `docker logs sentiment_docker_test-nlp-cpu-1`
2. Verify API health: `curl http://localhost/api/nlp/health`
3. Review documentation: `NLI_ARCHITECTURE.md`
4. Test with simple examples first before complex datasets
