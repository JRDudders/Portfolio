# NLI Pipeline Architecture

## Overview

Natural Language Inference (NLI) is now integrated into the existing NLP service container. This document explains the architecture, capabilities, and considerations for scaling.

## Current Implementation

### Stance Detection (arxiv:2305.01723)

**What it does:**
- Classifies text stance towards a claim: SUPPORT, OPPOSE, or NEUTRAL
- Uses textual entailment to determine stance without task-specific training
- Based on the paper "Stance Detection: A Practical Guide to Classifying Political Beliefs in Text"

**How it works:**
1. Takes input texts and a claim/hypothesis
2. Uses pre-trained NLI model (DeBERTa-v3-base/large-mnli)
3. Classifies entailment → SUPPORT, contradiction → OPPOSE, neutral → NEUTRAL
4. Returns stance label with confidence scores

**API Endpoint:**
```bash
POST /stance
{
  "texts": ["Text to classify", "Another text"],
  "claim": "Climate change is real",
  "preset": "stance-deberta"  # or "stance-deberta-large"
}
```

**Response:**
```json
{
  "success": true,
  "claim": "Climate change is real",
  "results": [
    {
      "stance": "SUPPORT",
      "scores": {
        "SUPPORT": 0.89,
        "OPPOSE": 0.05,
        "NEUTRAL": 0.06
      },
      "claim": "Climate change is real"
    }
  ]
}
```

## Architecture Decision: Do We Need a Separate Container?

### **Answer: No, not currently. But here's when you would:**

### ✅ Current Setup (Integrated in NLP Container)

**Advantages:**
- **Zero overhead**: No inter-service communication latency
- **Shared resources**: Model caching works across all NLP tasks
- **Simpler deployment**: One less container to manage
- **Memory efficiency**: Transformer infrastructure already loaded
- **Good for**: Small to medium scale (< 1000 requests/day)

**Current Capacity:**
- NLP container has 8GB memory limit (GPU version)
- DeBERTa-v3-base: ~420MB model size
- DeBERTa-v3-large: ~1.3GB model size
- Can handle both models + other NLP tasks comfortably

### ⚠️ When to Split into Separate Container

You should create a dedicated NLI container when you experience:

1. **High Volume Stance Detection**
   - > 5,000 stance requests/day
   - Blocking other NLP tasks (sentiment, NER, topics)
   - Need for independent scaling

2. **Resource Contention**
   - OOM errors in NLP container
   - Slow response times for other tasks when stance is running
   - GPU memory exhaustion

3. **Different Scaling Requirements**
   - Stance detection needs auto-scaling but other tasks don't
   - Want to deploy stance on different hardware (e.g., larger GPU)
   - Need geographic distribution (stance in multiple regions)

4. **Production Isolation**
   - Stance detection is mission-critical and needs dedicated resources
   - Want to avoid stance failures affecting other NLP tasks
   - Need separate monitoring/alerting for stance

## Full NLI Pipeline Capabilities

### What NLI Can Do Beyond Stance

The same NLI infrastructure supports multiple use cases:

#### 1. **Textual Entailment Classification**
```python
# Does premise entail hypothesis?
premise = "The cat sat on the mat"
hypothesis = "An animal was on the mat"
# Result: ENTAILMENT
```

#### 2. **Zero-Shot Classification** (Already Implemented)
```python
# Already available via /analyze/file with zeroshot presets
# Uses same NLI models for classification without training
```

#### 3. **Claim Verification**
```python
# Is this claim supported by evidence?
claim = "Vaccines cause autism"
evidence = "Multiple peer-reviewed studies show no link between vaccines and autism"
# Result: CONTRADICTION
```

#### 4. **Semantic Similarity**
```python
# How similar are these texts semantically?
# Can be derived from entailment probabilities
```

#### 5. **Question Answering Validation**
```python
# Does this answer correctly respond to the question?
# Via entailment between question+answer and knowledge base
```

## Scaling Architecture Options

### Option 1: Keep Integrated (Current - Recommended)

```
┌─────────────────────────────────────┐
│     NLP Container (Python 3.12)     │
├─────────────────────────────────────┤
│ • Sentiment Analysis                │
│ • NER                               │
│ • Topics                            │
│ • Zero-Shot Classification          │
│ • Stance Detection (NLI)            │ ← New
├─────────────────────────────────────┤
│ Memory: 8GB                         │
│ Models: All cached in same process  │
└─────────────────────────────────────┘
```

**Use when:**
- < 1000 stance requests/day
- Resources not constrained
- Simplicity is priority

### Option 2: Separate NLI Container (Future)

```
┌──────────────────┐    ┌──────────────────┐
│  NLP Container   │    │  NLI Container   │
├──────────────────┤    ├──────────────────┤
│ • Sentiment      │    │ • Stance         │
│ • NER            │    │ • Entailment     │
│ • Topics         │    │ • Claim Verify   │
│ • Zero-Shot*     │    │ • Zero-Shot*     │
├──────────────────┤    ├──────────────────┤
│ Memory: 6GB      │    │ Memory: 4GB      │
│ Port: 8001       │    │ Port: 8004       │
└──────────────────┘    └──────────────────┘
         │                      │
         └──────────┬───────────┘
                    │
         ┌──────────▼───────────┐
         │   Frontend Proxy     │
         └──────────────────────┘
```

**Use when:**
- > 5000 stance requests/day
- Resource contention observed
- Need independent scaling

*Note: Zero-shot can use either container (both use NLI models)

### Option 3: Microservices per Task (Not Recommended)

```
┌─────────┐  ┌─────┐  ┌────────┐  ┌─────┐
│Sentiment│  │ NER │  │ Topics │  │ NLI │
└─────────┘  └─────┘  └────────┘  └─────┘
```

**Avoid unless:**
- > 100K requests/day per task
- Each task needs different infrastructure
- Enterprise-scale requirements

## Migration Path

### Phase 1: Current (Integrated)
- ✅ Implemented
- All NLP tasks in single container
- Suitable for development and medium production

### Phase 2: Separate NLI Container (If Needed)
1. Create `Dockerfile.nli` (similar to Dockerfile.nlp)
2. Create `service_nli.py` with stance/entailment endpoints
3. Add to docker-compose with port 8004
4. Update nginx routing in frontend
5. Migrate zero-shot to NLI container (optional)

**Files to create:**
```bash
Sentiment Docker Test/
├── Dockerfile.nli         # New: NLI-specific container
├── Dockerfile.nli.gpu     # New: GPU version
├── service_nli.py         # New: NLI API service
├── requirements-nli.txt   # New: NLI dependencies
└── docker-compose.*.yml   # Modified: Add nli service
```

### Phase 3: Load Balancing (High Scale)
- Multiple instances of NLI container
- nginx load balancing
- Redis caching for repeated requests

## Resource Requirements

### Current (Integrated)
| Component | CPU | Memory | GPU |
|-----------|-----|--------|-----|
| NLP (all tasks) | 4 cores | 8GB | Optional |

### Separated
| Component | CPU | Memory | GPU |
|-----------|-----|--------|-----|
| NLP (sentiment, NER, topics) | 2 cores | 6GB | Optional |
| NLI (stance, entailment) | 2 cores | 4GB | Optional |
| **Total** | **4 cores** | **10GB** | Optional |

## Performance Benchmarks

### DeBERTa-v3-base (stance-deberta)
- Model size: 420MB
- Inference time: ~50ms per text (CPU)
- Inference time: ~10ms per text (GPU)
- Throughput: ~20 req/s (CPU), ~100 req/s (GPU)

### DeBERTa-v3-large (stance-deberta-large)
- Model size: 1.3GB
- Inference time: ~150ms per text (CPU)
- Inference time: ~30ms per text (GPU)
- Throughput: ~7 req/s (CPU), ~30 req/s (GPU)

### Chunking Performance
- Long texts (> 320 words) are auto-chunked
- Each chunk processed independently
- Results averaged across chunks
- 1000-word document: ~100ms (base) or ~300ms (large)

## Monitoring Metrics

### Key Metrics to Track
```python
# If you decide to separate NLI container later, monitor:
metrics = {
    "stance_requests_per_minute": "...",
    "average_latency_ms": "...",
    "model_memory_usage_mb": "...",
    "gpu_utilization_percent": "...",  # If using GPU
    "error_rate": "...",
}
```

### Alert Thresholds (When to Split Container)
- Average latency > 500ms
- Memory usage > 90%
- Error rate > 1%
- Request queue depth > 100

## Recommendations

### For Your Current Use Case
**Keep integrated** (current implementation) because:
1. You're likely not doing 5000+ stance requests/day yet
2. Simpler to maintain and deploy
3. Shared model infrastructure is efficient
4. Easy to monitor as single service
5. Can split later if needed (data-driven decision)

### Future Enhancements
If you do need to scale, consider:
1. **Redis caching** for repeated claim+text pairs
2. **Batch processing** for large-scale analysis
3. **Async task queue** (Celery) for non-blocking requests
4. **Model quantization** to reduce memory (INT8, 4-bit)
5. **Distillation** to smaller, faster models

## Additional NLI Use Cases

### 1. Misinformation Detection Pipeline
```python
# Combine stance + claim verification
claim = "5G causes COVID"
evidence_texts = [
    "WHO confirms no link between 5G and COVID",
    "Viruses cannot travel on radio waves",
    # ... more evidence
]
# Classify each evidence's stance toward claim
# Aggregate to determine claim veracity
```

### 2. Debate Analysis
```python
# Classify debate statements toward propositions
proposition = "Nuclear energy is safe"
statements = [
    "Fukushima proved nuclear is dangerous",
    "Modern reactors have passive safety",
    # ... more statements
]
# Map debate flow by stance
```

### 3. Social Media Content Moderation
```python
# Detect if posts support harmful claims
harmful_claims = [
    "Vaccines are dangerous",
    "Election was stolen",
    # ... community guidelines
]
user_posts = [...]
# Flag posts with high SUPPORT scores
```

### 4. News Bias Detection
```python
# Measure article stance toward political positions
positions = ["Liberal agenda", "Conservative values"]
article_text = "..."
# Detect bias by relative stance scores
```

## Conclusion

**Current Answer: No separate container needed.**

The integrated approach is optimal for your current scale. The infrastructure is designed to support splitting later if monitoring data shows resource contention or high volume. Make the decision based on data, not speculation.

When to revisit this decision:
- Monitor metrics for 1 month
- If stance latency > 500ms consistently → consider separation
- If OOM errors occur → consider separation
- If stance becomes > 30% of NLP traffic → consider separation

The current implementation gives you all the flexibility you need while keeping architecture simple.
