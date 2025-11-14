# Stance Detection Examples

## Quick Start

### Basic Usage

**API Endpoint:** `POST http://localhost:8001/stance`

```bash
curl -X POST http://localhost:8001/stance \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Climate change is the defining crisis of our time",
      "The climate has always changed naturally"
    ],
    "claim": "Climate change is caused by human activity",
    "preset": "stance-deberta"
  }'
```

**Response:**
```json
{
  "success": true,
  "claim": "Climate change is caused by human activity",
  "preset": "stance-deberta",
  "results": [
    {
      "stance": "SUPPORT",
      "scores": {
        "SUPPORT": 0.87,
        "OPPOSE": 0.06,
        "NEUTRAL": 0.07
      },
      "claim": "Climate change is caused by human activity"
    },
    {
      "stance": "OPPOSE",
      "scores": {
        "SUPPORT": 0.12,
        "OPPOSE": 0.73,
        "NEUTRAL": 0.15
      },
      "claim": "Climate change is caused by human activity"
    }
  ]
}
```

## Use Cases

### 1. Political Stance Detection

**Claim:** "Universal healthcare is a right"

```python
import requests

texts = [
    "Everyone deserves access to medical care regardless of ability to pay",
    "Government-run healthcare leads to inefficiency and rationing",
    "Healthcare costs are bankrupting American families",
    "The free market provides the best healthcare outcomes"
]

response = requests.post(
    "http://localhost:8001/stance",
    json={
        "texts": texts,
        "claim": "Universal healthcare is a right",
        "preset": "stance-deberta-large"  # Use larger model for higher accuracy
    }
)

results = response.json()["results"]
for text, result in zip(texts, results):
    print(f"Text: {text[:50]}...")
    print(f"Stance: {result['stance']} ({result['scores'][result['stance']]:.2f})")
    print()
```

### 2. Misinformation Detection

**Claim:** "Vaccines cause autism"

```python
evidence_texts = [
    "Multiple peer-reviewed studies found no link between vaccines and autism",
    "The original study claiming a vaccine-autism link was retracted for fraud",
    "Many parents report autism symptoms after vaccination",
    "Autism rates increased while vaccine rates increased"
]

response = requests.post(
    "http://localhost:8001/stance",
    json={
        "texts": evidence_texts,
        "claim": "Vaccines cause autism",
        "preset": "stance-deberta"
    }
)

# Count OPPOSE vs SUPPORT to assess claim credibility
results = response.json()["results"]
oppose_count = sum(1 for r in results if r["stance"] == "OPPOSE")
support_count = sum(1 for r in results if r["stance"] == "SUPPORT")

print(f"Evidence against claim: {oppose_count}")
print(f"Evidence for claim: {support_count}")
print(f"Claim likely: {'FALSE' if oppose_count > support_count else 'TRUE'}")
```

### 3. Social Media Analysis

**Analyze Twitter conversation about a controversial topic:**

```python
import pandas as pd

# Load tweets from CSV
df = pd.read_csv("tweets.csv")

claim = "The 2020 election was stolen"

response = requests.post(
    "http://localhost:8001/stance",
    json={
        "texts": df["text"].tolist()[:100],  # First 100 tweets
        "claim": claim,
        "preset": "stance-deberta"
    }
)

results = response.json()["results"]

# Add stance to dataframe
df["stance"] = [r["stance"] for r in results]
df["stance_confidence"] = [r["scores"][r["stance"]] for r in results]

# Analyze distribution
print(df["stance"].value_counts())
print(f"\nAverage confidence: {df['stance_confidence'].mean():.2f}")

# Find most supportive/opposing tweets
top_support = df[df["stance"] == "SUPPORT"].nlargest(5, "stance_confidence")
top_oppose = df[df["stance"] == "OPPOSE"].nlargest(5, "stance_confidence")
```

### 4. News Article Bias Detection

**Detect if articles support political positions:**

```python
# Analyze multiple articles on the same topic
articles = [
    {
        "source": "Source A",
        "text": "The new bill will provide tax relief to working families..."
    },
    {
        "source": "Source B",
        "text": "Critics warn the bill will balloon the deficit and benefit the wealthy..."
    },
    {
        "source": "Source C",
        "text": "Economists are divided on the bill's potential impact..."
    }
]

claim = "The new tax bill benefits working families"

response = requests.post(
    "http://localhost:8001/stance",
    json={
        "texts": [a["text"] for a in articles],
        "claim": claim,
        "preset": "stance-deberta-large"
    }
)

results = response.json()["results"]

for article, result in zip(articles, results):
    print(f"Source: {article['source']}")
    print(f"Stance: {result['stance']}")
    print(f"Confidence: {result['scores'][result['stance']]:.2%}")
    print(f"Bias: {'Pro' if result['stance'] == 'SUPPORT' else 'Anti' if result['stance'] == 'OPPOSE' else 'Neutral'}")
    print()
```

### 5. Debate Transcript Analysis

**Map argument flow in a debate:**

```python
proposition = "Nuclear energy should be expanded"

statements = [
    {"speaker": "Pro", "text": "Nuclear power is the only carbon-free baseload energy source"},
    {"speaker": "Con", "text": "Renewable energy with storage can provide reliable clean power"},
    {"speaker": "Pro", "text": "Renewables require vast land areas and resource extraction"},
    {"speaker": "Con", "text": "Nuclear waste remains hazardous for thousands of years"},
    {"speaker": "Moderator", "text": "Both sides raise important considerations for energy policy"}
]

response = requests.post(
    "http://localhost:8001/stance",
    json={
        "texts": [s["text"] for s in statements],
        "claim": proposition,
        "preset": "stance-deberta"
    }
)

results = response.json()["results"]

# Create debate flow visualization
for statement, result in zip(statements, results):
    emoji = "✅" if result["stance"] == "SUPPORT" else "❌" if result["stance"] == "OPPOSE" else "⚖️"
    print(f"{emoji} {statement['speaker']}: {statement['text'][:60]}...")
    print(f"   Stance: {result['stance']} ({result['scores'][result['stance']]:.0%})")
    print()
```

### 6. Content Moderation Pipeline

**Flag content supporting harmful claims:**

```python
# Community guidelines
harmful_claims = [
    "Violence is justified against political opponents",
    "Vaccines are harmful to children",
    "Election results cannot be trusted"
]

user_posts = [
    "We need to fight back against these corrupt politicians",
    "My child developed autism right after vaccination",
    "The voting machines were hacked by foreign actors",
    "Remember to vote in the upcoming election"
]

# Check each post against each harmful claim
flagged_posts = []

for post in user_posts:
    post_flags = []

    for claim in harmful_claims:
        response = requests.post(
            "http://localhost:8001/stance",
            json={
                "texts": [post],
                "claim": claim,
                "preset": "stance-deberta"
            }
        )

        result = response.json()["results"][0]

        # Flag if strongly supports harmful claim
        if result["stance"] == "SUPPORT" and result["scores"]["SUPPORT"] > 0.7:
            post_flags.append({
                "claim": claim,
                "confidence": result["scores"]["SUPPORT"]
            })

    if post_flags:
        flagged_posts.append({
            "post": post,
            "flags": post_flags
        })

# Review flagged content
for flagged in flagged_posts:
    print(f"⚠️  Flagged: {flagged['post']}")
    for flag in flagged["flags"]:
        print(f"   - Supports: '{flag['claim']}' ({flag['confidence']:.0%})")
    print()
```

## Model Selection

### stance-deberta (Default)
- **Best for:** Fast inference, high throughput
- **Model:** microsoft/deberta-v3-base-mnli (420MB)
- **Accuracy:** Good (F1 ~0.70 on typical stance tasks)
- **Speed:** ~50ms per text (CPU), ~10ms (GPU)

### stance-deberta-large
- **Best for:** Maximum accuracy, research
- **Model:** microsoft/deberta-v3-large-mnli (1.3GB)
- **Accuracy:** Excellent (F1 ~0.75 on typical stance tasks)
- **Speed:** ~150ms per text (CPU), ~30ms (GPU)

## Advanced Features

### Custom Hypothesis Templates

By default, the hypothesis template is `"{}"` (just the claim text). You can customize this:

```python
# Default
response = requests.post(
    "http://localhost:8001/stance",
    json={
        "texts": ["Nuclear power is safe"],
        "claim": "nuclear energy",
        "preset": "stance-deberta"
    }
)
# Hypothesis: "nuclear energy"

# Custom template (via nlp.run_task directly in Python)
from nlp import run_task

results = run_task(
    texts=["Nuclear power is safe"],
    preset="stance-deberta",
    claim="nuclear energy",
    hypothesis_template="This text discusses {}."
)
# Hypothesis: "This text discusses nuclear energy."
```

### Batch Processing

For large-scale analysis:

```python
# Process in batches to avoid memory issues
import requests
from typing import List

def batch_stance_detection(texts: List[str], claim: str, batch_size: int = 32):
    all_results = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        response = requests.post(
            "http://localhost:8001/stance",
            json={
                "texts": batch,
                "claim": claim,
                "preset": "stance-deberta"
            }
        )

        all_results.extend(response.json()["results"])

    return all_results

# Process 10,000 texts
large_dataset = pd.read_csv("large_corpus.csv")
results = batch_stance_detection(
    texts=large_dataset["text"].tolist(),
    claim="Climate change requires immediate action",
    batch_size=32
)
```

## Integration with Graph Analytics

**Combine with social network analysis:**

```python
# 1. Extract social media network
response = requests.post(
    "http://localhost:8002/prepare",
    files={"file": open("tweets.csv", "rb")}
)

original_data = response.json()["original_data"]

# 2. Run stance detection on tweets
texts = [tweet["text"] for tweet in original_data]
claim = "The new policy will benefit citizens"

stance_response = requests.post(
    "http://localhost:8001/stance",
    json={
        "texts": texts,
        "claim": claim,
        "preset": "stance-deberta"
    }
)

# 3. Add stance to network nodes
results = stance_response.json()["results"]
for tweet, result in zip(original_data, results):
    tweet["stance"] = result["stance"]
    tweet["stance_confidence"] = result["scores"][result["stance"]]

# 4. Analyze echo chambers
# - Do users primarily interact with same-stance users?
# - Are SUPPORT/OPPOSE users in separate clusters?
# - Which users are stance "bridges"?
```

## Python Client

**Reusable client class:**

```python
class StanceDetector:
    def __init__(self, base_url="http://localhost:8001", preset="stance-deberta"):
        self.base_url = base_url
        self.preset = preset

    def detect(self, texts, claim):
        """Detect stance of texts toward claim"""
        response = requests.post(
            f"{self.base_url}/stance",
            json={
                "texts": texts if isinstance(texts, list) else [texts],
                "claim": claim,
                "preset": self.preset
            }
        )
        response.raise_for_status()
        return response.json()["results"]

    def detect_single(self, text, claim):
        """Detect stance of single text"""
        results = self.detect([text], claim)
        return results[0]

    def get_stance(self, text, claim):
        """Get just the stance label"""
        result = self.detect_single(text, claim)
        return result["stance"]

    def get_confidence(self, text, claim):
        """Get confidence score for predicted stance"""
        result = self.detect_single(text, claim)
        stance = result["stance"]
        return result["scores"][stance]

# Usage
detector = StanceDetector(preset="stance-deberta-large")

stance = detector.get_stance(
    "Fossil fuels are destroying our planet",
    "Climate change is real"
)
print(f"Stance: {stance}")  # "SUPPORT"

confidence = detector.get_confidence(
    "Fossil fuels are destroying our planet",
    "Climate change is real"
)
print(f"Confidence: {confidence:.1%}")  # "89.3%"
```

## Performance Tips

1. **Use batching:** Process multiple texts in one request
2. **Use base model for speed:** stance-deberta is 3x faster than large
3. **Cache results:** Same text+claim pairs return same results
4. **Use GPU:** 5x speedup for large batches
5. **Keep texts focused:** Long texts are auto-chunked, which is slower

## Error Handling

```python
import requests

try:
    response = requests.post(
        "http://localhost:8001/stance",
        json={
            "texts": ["Sample text"],
            "claim": "Sample claim",
            "preset": "stance-deberta"
        },
        timeout=30  # 30 second timeout
    )
    response.raise_for_status()
    results = response.json()["results"]
except requests.exceptions.Timeout:
    print("Request timed out - try smaller batch or shorter texts")
except requests.exceptions.HTTPError as e:
    if e.response.status_code == 500:
        print(f"Server error: {e.response.json().get('detail')}")
    else:
        print(f"HTTP error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Next Steps

- See `NLI_ARCHITECTURE.md` for infrastructure details
- See `nlp.py` for implementation details
- Check FastAPI docs at `http://localhost:8001/docs` for interactive API testing
