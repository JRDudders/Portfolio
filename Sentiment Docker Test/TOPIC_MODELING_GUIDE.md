# Topic Modeling Guide: Supervised vs Unsupervised

**Question:** Can zero-shot tasks run without user-provided labels for unsupervised topic modeling?

**Answer:** Zero-shot can use default labels, but for **TRUE unsupervised topic modeling**, use `topics-nmf` or `topics-kmeans` instead.

---

## 🎯 Three Approaches to Topic Modeling

### 1. Zero-Shot Classification (Semi-Supervised)

**Presets:** `zeroshot-bart`, `zeroshot-mdeberta`

**How it works:**
- Classifies text against predefined topic labels
- Uses neural language models (BART/DeBERTa with MNLI training)
- Can work without user labels (uses defaults)

**Default labels if none provided:**
```python
["politics", "economy", "health", "science", "technology",
 "sports", "entertainment", "climate", "crime", "education",
 "misinformation", "opinion"]
```

**Output example:**
```json
{
  "labels": ["politics", "economy", "health"],
  "scores": [0.85, 0.12, 0.03]
}
```

**Best for:**
- ✅ Classifying documents into known categories
- ✅ When you have predefined topics in mind
- ✅ Cross-lingual classification (with mdeberta)
- ✅ Single document classification

**Usage:**
```bash
# With default labels
curl -X POST http://localhost:8001/analyze/file \
  -F "file=@article.txt" \
  -F "preset=zeroshot-bart"

# With custom labels
curl -X POST http://localhost:8001/analyze/file \
  -F "file=@article.txt" \
  -F "preset=zeroshot-bart" \
  -F "labels=finance,crypto,stocks,markets"
```

---

### 2. NMF Topic Modeling (Unsupervised)

**Preset:** `topics-nmf`

**How it works:**
- Non-negative Matrix Factorization over TF-IDF features
- Discovers latent topics as weighted term combinations
- Fully unsupervised - no predefined labels needed

**Output example:**
```json
{
  "topics": [
    {
      "topic_id": 0,
      "top_terms": [
        {"term": "election", "weight": 0.82},
        {"term": "vote", "weight": 0.65},
        {"term": "candidate", "weight": 0.58},
        {"term": "campaign", "weight": 0.51}
      ]
    },
    {
      "topic_id": 1,
      "top_terms": [
        {"term": "market", "weight": 0.73},
        {"term": "stock", "weight": 0.69},
        {"term": "trading", "weight": 0.54}
      ]
    }
  ],
  "doc_topics": [
    {"doc_id": 0, "topic_id": 0, "score": 0.75},
    {"doc_id": 1, "topic_id": 1, "score": 0.82}
  ]
}
```

**Best for:**
- ✅ **TRUE unsupervised discovery**
- ✅ Exploratory analysis
- ✅ Finding hidden themes in corpus
- ✅ When you don't know topics in advance
- ✅ Interpretable results (see actual words)

**Usage:**
```bash
# Default: 10 topics
curl -X POST http://localhost:8001/analyze/file \
  -F "file=@documents.csv" \
  -F "preset=topics-nmf"

# Custom parameters (via API if supported):
# - n_topics: Number of topics to extract
# - max_features: Max vocabulary size
# - min_df: Min document frequency
```

**Requirements:**
- Needs **multiple documents** (minimum 2)
- Works best with 50+ documents
- More documents = better topic quality

---

### 3. K-Means Topic Modeling (Unsupervised)

**Preset:** `topics-kmeans`

**How it works:**
- Clusters documents in TF-IDF space
- Each cluster represents a topic
- Hard assignment (one topic per document)

**Output example:**
```json
{
  "topics": [
    {
      "topic_id": 0,
      "top_terms": [
        {"term": "climate", "weight": 0.91},
        {"term": "global", "weight": 0.83},
        {"term": "warming", "weight": 0.79}
      ]
    }
  ],
  "doc_topics": [
    {"doc_id": 0, "topic_id": 0, "score": 1.0},  // Hard assignment
    {"doc_id": 1, "topic_id": 2, "score": 1.0}
  ]
}
```

**Best for:**
- ✅ **TRUE unsupervised discovery**
- ✅ Document clustering
- ✅ When documents belong to single topic
- ✅ Faster than NMF on large datasets

**Differences from NMF:**
- NMF: Documents can belong to multiple topics (soft assignment)
- K-Means: Each document assigned to one topic (hard assignment)

**Usage:**
```bash
# Default: 10 clusters
curl -X POST http://localhost:8001/analyze/file \
  -F "file=@documents.csv" \
  -F "preset=topics-kmeans"
```

---

## 📊 Comparison Table

| Feature | Zero-Shot | Topics-NMF | Topics-KMeans |
|---------|-----------|------------|---------------|
| **Supervision** | Semi (needs labels) | None | None |
| **Min docs required** | 1 | 2+ (50+ ideal) | 2+ (50+ ideal) |
| **Output** | Label scores | Weighted terms | Weighted terms |
| **Interpretability** | Predefined names | Term weights | Term weights |
| **Multi-topic docs** | ✅ Yes | ✅ Yes (soft) | ❌ No (hard) |
| **Speed** | Medium | Fast | Very fast |
| **Cross-lingual** | ✅ Yes (mdeberta) | ❌ No | ❌ No |
| **Discovery** | ❌ No (classifies) | ✅ Yes | ✅ Yes |

---

## 🎯 Which One Should You Use?

### Use **Zero-Shot** when:
- You know the topics you're looking for
- You want to categorize into predefined buckets
- You have single documents to classify
- You need multilingual support

### Use **Topics-NMF** when:
- You want to discover unknown topics
- Documents can belong to multiple topics
- You want interpretable term-based topics
- You have a corpus of 50+ documents

### Use **Topics-KMeans** when:
- You want to discover unknown topics
- Documents belong to single topics
- You need fast clustering
- You have a large corpus (1000+ documents)

---

## 💡 Recommendation for Unsupervised Topic Modeling

**For your use case (unsupervised topic modeling), use:**

```bash
# Best option: NMF with soft topic assignment
curl -X POST http://localhost:8001/analyze/file \
  -F "file=@my_documents.csv" \
  -F "preset=topics-nmf"
```

**Why NMF?**
- ✅ Fully unsupervised (no labels needed)
- ✅ Discovers topics from data
- ✅ Returns interpretable weighted terms
- ✅ Documents can have multiple topics
- ✅ Industry-standard approach

**File format:**
```csv
text
"First document about elections and voting..."
"Second document about stock market trends..."
"Third document about climate change..."
```

Or JSON:
```json
[
  {"text": "Document 1..."},
  {"text": "Document 2..."},
  {"text": "Document 3..."}
]
```

---

## 🔧 Frontend Usage

The frontend already supports all three methods:

**Zero-shot (with or without custom labels):**
1. Upload file
2. Select preset: `zeroshot-bart` or `zeroshot-mdeberta`
3. Optionally add custom labels (comma-separated)
4. Click analyze

**Unsupervised (no labels needed):**
1. Upload CSV/JSON with multiple documents
2. Select preset: `topics-nmf` or `topics-kmeans`
3. Click analyze
4. Get discovered topics with weighted terms

---

## 📝 Examples

### Zero-Shot with Default Labels
```bash
curl -X POST http://localhost:8001/analyze/file \
  -F "file=@news_article.txt" \
  -F "preset=zeroshot-bart"

# Uses default: politics, economy, health, etc.
```

### Zero-Shot with Custom Labels
```bash
curl -X POST http://localhost:8001/analyze/file \
  -F "file=@customer_reviews.csv" \
  -F "preset=zeroshot-bart" \
  -F "labels=pricing,quality,service,delivery,support"

# Classifies each review into these categories
```

### True Unsupervised Discovery
```bash
curl -X POST http://localhost:8001/analyze/file \
  -F "file=@reddit_comments.csv" \
  -F "preset=topics-nmf"

# Discovers topics like:
# Topic 0: [gpu: 0.8, nvidia: 0.7, gaming: 0.6]
# Topic 1: [cpu: 0.9, intel: 0.7, benchmark: 0.5]
# (discovered from data, not predefined)
```

---

## 🚨 Important Notes

1. **Zero-shot ≠ Unsupervised**
   - Zero-shot still needs labels (defaults or custom)
   - It classifies against those labels
   - Not discovering topics, just matching to known ones

2. **NMF/K-Means = True Unsupervised**
   - No labels needed at all
   - Discovers topics from the data
   - Returns weighted terms, not labels

3. **Document Count Requirements**
   - Zero-shot: Works on 1 document
   - NMF/K-Means: Needs 2+ (ideally 50+)

4. **Current Frontend Limitation**
   - The old `/analyze/text` endpoint (used for "topics" task) uses zero-shot
   - Use `/analyze/file` endpoint with preset for true unsupervised

---

## ✅ Summary

**For unsupervised topic modeling without labels:**

✅ **Use:** `topics-nmf` or `topics-kmeans`
❌ **Don't use:** `zeroshot-bart` (it's semi-supervised)

**Zero-shot can work without USER-provided labels** (uses defaults), but it's still classifying against predefined categories, not discovering topics.

For true topic discovery, use NMF or K-Means! 🎯
