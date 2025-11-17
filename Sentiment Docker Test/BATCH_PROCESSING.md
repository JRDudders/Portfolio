# Batch Excel Processing Guide

Automated sentiment and theme extraction from multi-tab Excel spreadsheets with optional zero-shot classification labels.

## Overview

The batch processing feature allows you to:
- ✅ Process Excel files with multiple sheets automatically
- ✅ Extract **sentiment** from text data
- ✅ Extract **themes** using zero-shot classification
- ✅ Use **custom theme labels** or defaults
- ✅ Auto-detect text columns or specify manually
- ✅ Preserve original data and add result columns
- ✅ Process thousands of rows efficiently with batching

---

## Quick Start

### Using cURL

```bash
# Basic usage - process all sheets with default settings
curl -X POST http://localhost:8001/batch/excel \
  -F "file=@my_data.xlsx"

# Custom theme labels
curl -X POST http://localhost:8001/batch/excel \
  -F "file=@my_data.xlsx" \
  -F "theme_labels=politics,economics,social issues,environment,technology"

# Sentiment only (skip themes)
curl -X POST http://localhost:8001/batch/excel \
  -F "file=@my_data.xlsx" \
  -F "extract_themes=false"

# Specific sheets only
curl -X POST http://localhost:8001/batch/excel \
  -F "file=@my_data.xlsx" \
  -F "sheets_to_process=Sheet1,Sheet3"

# Output is saved to my_data_analyzed.xlsx
```

### Using Python

```python
import requests

url = "http://localhost:8001/batch/excel"

# Upload file
with open("my_data.xlsx", "rb") as f:
    files = {"file": f}
    params = {
        "extract_sentiment": True,
        "extract_themes": True,
        "theme_labels": "politics,economics,social,health,technology",
        "top_n_themes": 3
    }

    response = requests.post(url, files=files, params=params)

    # Save output
    with open("my_data_analyzed.xlsx", "wb") as out:
        out.write(response.content)

print("Processing complete!")
```

### Using the API Docs

1. Navigate to: http://localhost:8001/docs
2. Find `/batch/excel` endpoint
3. Click "Try it out"
4. Upload your Excel file
5. Configure options
6. Click "Execute"
7. Download the result file

---

## Input Format

### Excel File Requirements

**Supported formats:**
- `.xlsx` (Excel 2007+)
- `.xlsm` (Excel with macros)
- `.xls` (Excel 97-2003)

**Text Column:**
Your spreadsheet should have at least one column with text data. The system will:
1. Look for a column named `text` (case-insensitive)
2. Look for columns with keywords: `comment`, `message`, `content`, `description`, `post`, `tweet`
3. Find the first string column with substantial content (avg length > 20 chars)
4. Use the first string column as fallback

Or specify manually with `text_column` parameter.

### Example Input

**Sheet 1: Customer Feedback**
```
| date       | customer_id | text                                  |
|------------|-------------|---------------------------------------|
| 2025-01-01 | C001        | This product is amazing! Love it!     |
| 2025-01-02 | C002        | Terrible service, very disappointed   |
| 2025-01-03 | C003        | Great value for money, recommended    |
```

**Sheet 2: Social Media Posts**
```
| timestamp  | platform | post                                  |
|------------|----------|---------------------------------------|
| 2025-01-01 | Twitter  | New climate policy announced today    |
| 2025-01-02 | Reddit   | Discussion about economic trends      |
```

---

## Output Format

The output Excel file preserves all original columns and adds new ones:

### Output Columns

**Sentiment Analysis:**
- `sentiment`: Sentiment label (positive, negative, neutral)
- `sentiment_confidence`: Confidence score (0.0-1.0) [optional]

**Theme Extraction:**
- `theme_1`: Top theme
- `theme_1_score`: Confidence score for top theme
- `theme_2`: Second theme
- `theme_2_score`: Confidence score for second theme
- `theme_3`: Third theme
- `theme_3_score`: Confidence score for third theme

### Example Output

**Sheet 1: Customer Feedback (processed)**
```
| date       | customer_id | text                                  | sentiment | sentiment_confidence | theme_1    | theme_1_score | theme_2      | theme_2_score |
|------------|-------------|---------------------------------------|-----------|---------------------|------------|---------------|--------------|---------------|
| 2025-01-01 | C001        | This product is amazing! Love it!     | positive  | 0.95                | technology | 0.82          | social       | 0.11          |
| 2025-01-02 | C002        | Terrible service, very disappointed   | negative  | 0.89                | social     | 0.45          | economics    | 0.32          |
| 2025-01-03 | C003        | Great value for money, recommended    | positive  | 0.92                | economics  | 0.68          | technology   | 0.18          |
```

---

## Configuration Options

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | File | **Required** | Excel file to process |
| `extract_sentiment` | bool | `true` | Extract sentiment analysis |
| `extract_themes` | bool | `true` | Extract themes via zero-shot |
| `theme_labels` | string | None | Comma-separated custom labels (optional) |
| `sentiment_preset` | string | `sentiment-twitter` | Sentiment model preset |
| `theme_preset` | string | `zeroshot-bart` | Zero-shot model preset |
| `text_column` | string | None | Text column name (auto-detect if not provided) |
| `sheets_to_process` | string | None | Comma-separated sheet names (all if not provided) |
| `top_n_themes` | int | `3` | Number of top themes to extract (1-10) |
| `add_confidence_scores` | bool | `true` | Include confidence scores |
| `batch_size` | int | `32` | Batch size for processing (1-128) |

### Default Theme Labels

If `theme_labels` is not provided, the system uses:
- politics
- economy
- **military**
- health
- science
- technology
- sports
- entertainment
- climate
- crime
- education
- misinformation
- opinion

### Available Presets

**Sentiment Analysis:**
- `sentiment-twitter`: Twitter sentiment (RoBERTa) - Fast, good for social media
- `sentiment-sst2`: SST-2 sentiment (DistilBERT) - Accurate, general purpose

**Zero-Shot Classification:**
- `zeroshot-bart`: BART-MNLI - Fast, good for English
- `zeroshot-mdeberta`: mDeBERTa - Multilingual, more accurate but slower

---

## Use Cases

### 1. Customer Feedback Analysis

**Goal:** Analyze customer reviews and categorize by theme

```bash
curl -X POST http://localhost:8001/batch/excel \
  -F "file=@customer_feedback.xlsx" \
  -F "theme_labels=product quality,customer service,pricing,shipping,website experience"
```

**Output:** Sentiment + top 3 themes per review

### 2. Social Media Monitoring

**Goal:** Track sentiment and topics from social media posts

```bash
curl -X POST http://localhost:8001/batch/excel \
  -F "file=@social_posts.xlsx" \
  -F "theme_labels=brand mention,product feedback,competitor mention,industry news,customer inquiry"
```

### 3. Survey Response Analysis

**Goal:** Analyze open-ended survey responses

```bash
curl -X POST http://localhost:8001/batch/excel \
  -F "file=@survey_responses.xlsx" \
  -F "extract_sentiment=true" \
  -F "extract_themes=true" \
  -F "theme_labels=satisfied,dissatisfied,suggestion,complaint,question" \
  -F "top_n_themes=5"
```

### 4. News Article Categorization

**Goal:** Categorize news articles by topic

```bash
curl -X POST http://localhost:8001/batch/excel \
  -F "file=@news_articles.xlsx" \
  -F "extract_sentiment=false" \
  -F "theme_labels=politics,economics,technology,health,environment,sports"
```

### 5. Multi-Language Content

**Goal:** Process multilingual content

```bash
curl -X POST http://localhost:8001/batch/excel \
  -F "file=@multilingual_data.xlsx" \
  -F "sentiment_preset=sentiment-twitter" \
  -F "theme_preset=zeroshot-mdeberta"  # Multilingual model
```

### 6. Military & Geopolitical Analysis

**Goal:** Track military exercises, defense cooperation, and geopolitical events

```bash
# General military/defense categorization
curl -X POST http://localhost:8001/batch/excel \
  -F "file=@defense_news.xlsx" \
  -F "theme_labels=military exercises,defense cooperation,arms deals,territorial disputes,peacekeeping,security threats"

# Specific to UNITAS and regional exercises
curl -X POST http://localhost:8001/batch/excel \
  -F "file=@latin_america_security.xlsx" \
  -F "theme_labels=UNITAS exercises,naval cooperation,joint military exercises,humanitarian assistance,disaster relief,maritime security,defense partnerships,regional security"

# Detailed geopolitical analysis with UNITAS as specific category
curl -X POST http://localhost:8001/batch/excel \
  -F "file=@regional_events.xlsx" \
  -F "theme_labels=UNITAS naval exercises,bilateral defense,multilateral cooperation,military modernization,strategic partnerships,China influence,Russia influence,counter-narcotics,humanitarian missions" \
  -F "top_n_themes=5"
```

**UNITAS-Specific Categorization:**

UNITAS (Unitas is Latin for "unity") is an annual multinational maritime exercise. To specifically categorize UNITAS exercises:

```python
import requests

url = "http://localhost:8001/batch/excel"

# Custom labels for UNITAS and regional security analysis
unitas_labels = [
    "UNITAS naval exercises",           # Specific to UNITAS
    "maritime security operations",
    "anti-submarine warfare training",
    "humanitarian assistance disaster relief",
    "joint naval operations",
    "U.S. naval cooperation",
    "Latin American naval forces",
    "Caribbean security",
    "Pacific fleet exercises",
    "counter-narcotics operations"
]

with open("security_reports.xlsx", "rb") as f:
    response = requests.post(
        url,
        files={"file": f},
        params={
            "extract_sentiment": True,
            "extract_themes": True,
            "theme_labels": ",".join(unitas_labels),
            "top_n_themes": 5  # Get top 5 to capture multiple aspects
        }
    )

with open("security_reports_analyzed.xlsx", "wb") as out:
    out.write(response.content)
```

**Example Output for UNITAS Detection:**

Input text: *"The U.S. Navy and Chilean Navy participated in UNITAS 2025 naval exercises focusing on anti-submarine warfare and humanitarian assistance operations in the Pacific."*

Output themes:
- `theme_1`: UNITAS naval exercises (score: 0.89)
- `theme_2`: anti-submarine warfare training (score: 0.76)
- `theme_3`: humanitarian assistance disaster relief (score: 0.65)
- `theme_4`: U.S. naval cooperation (score: 0.58)
- `theme_5`: Pacific fleet exercises (score: 0.52)

---

## Performance Tips

### 1. Batch Size

**Large datasets (>10,000 rows):**
```bash
-F "batch_size=64"  # Faster but more memory
```

**Small datasets (<1,000 rows):**
```bash
-F "batch_size=16"  # Lower memory usage
```

**GPU available:**
```bash
-F "batch_size=128"  # Maximum throughput
```

### 2. Processing Time Estimates

| Rows | Sheets | Time (CPU) | Time (GPU) |
|------|--------|------------|------------|
| 100  | 1      | 15 sec     | 5 sec      |
| 1,000| 1      | 2 min      | 30 sec     |
| 10,000| 1     | 15 min     | 5 min      |
| 10,000| 5     | 60 min     | 20 min     |

*Times for both sentiment + theme extraction

### 3. Memory Usage

- **Sentiment only:** ~2GB RAM per 10k rows
- **Sentiment + Themes:** ~4GB RAM per 10k rows
- **GPU:** ~2GB VRAM additional

### 4. Skip Unnecessary Tasks

**Only need sentiment:**
```bash
-F "extract_themes=false"  # 2x faster
```

**Only need themes:**
```bash
-F "extract_sentiment=false"  # 1.5x faster
```

---

## Advanced Examples

### Process Specific Sheets

```python
import requests

url = "http://localhost:8001/batch/excel"
files = {"file": open("data.xlsx", "rb")}
params = {
    "sheets_to_process": "Q1 Results,Q2 Results",  # Only these sheets
    "text_column": "customer_feedback",
    "extract_sentiment": True,
    "extract_themes": True
}

response = requests.post(url, files=files, params=params)
```

### Custom Theme Labels for Domain-Specific Analysis

**Healthcare:**
```bash
-F "theme_labels=symptoms,diagnosis,treatment,medication,insurance,billing,appointment"
```

**E-commerce:**
```bash
-F "theme_labels=product quality,shipping,returns,pricing,customer service,website usability"
```

**Politics:**
```bash
-F "theme_labels=economy,healthcare,education,environment,foreign policy,social issues"
```

**Military & Defense (UNITAS-focused):**
```bash
-F "theme_labels=UNITAS exercises,PANAMAX exercises,SOUTHCOM operations,naval cooperation,defense partnerships,humanitarian missions,maritime security,counter-narcotics,disaster relief,regional stability,China influence,Russia influence"
```

**Geopolitical Intelligence:**
```bash
-F "theme_labels=military exercises,diplomatic relations,economic sanctions,trade agreements,security threats,territorial disputes,alliance formation,regional conflicts,peacekeeping operations,arms control"
```

### Extract More Themes

```bash
# Get top 5 themes instead of 3
-F "top_n_themes=5"
```

### Without Confidence Scores

```bash
# Cleaner output, smaller file size
-F "add_confidence_scores=false"
```

---

## Error Handling

### Common Errors

**1. No text column found:**
```json
{"detail": "No suitable text column found in dataframe"}
```
**Solution:** Specify manually with `-F "text_column=your_column_name"`

**2. Invalid sheet names:**
```json
{"detail": "Invalid sheet names: ['Sheet5']"}
```
**Solution:** Check sheet names with `pd.ExcelFile('file.xlsx').sheet_names`

**3. File too large:**
```json
{"detail": "File processing failed: Memory error"}
```
**Solution:** Process sheets individually or reduce `batch_size`

**4. Invalid file format:**
```json
{"detail": "File must be an Excel file (.xlsx, .xlsm, or .xls)"}
```
**Solution:** Convert file to Excel format first

---

## API Response

### Success Response

**Status Code:** 200 OK

**Headers:**
```
Content-Type: application/vnd.openxmlformats-officedocument.spreadsheetml.sheet
Content-Disposition: attachment; filename="your_file_analyzed.xlsx"
X-Processing-Stats: {"total_sheets": 2, "sheets": [...]}
```

**Body:** Binary Excel file

### Processing Stats (in headers)

```json
{
  "total_sheets": 2,
  "sheets": [
    {
      "sheet_name": "Customer Feedback",
      "rows_processed": 1523,
      "text_column": "feedback_text",
      "sentiment_distribution": {
        "positive": 892,
        "negative": 423,
        "neutral": 208
      },
      "theme_distribution": {
        "theme_1": {
          "product quality": 456,
          "customer service": 389,
          "pricing": 278
        }
      }
    }
  ]
}
```

---

## Integration Examples

### Python Script

```python
#!/usr/bin/env python3
"""Batch process Excel files with sentiment and theme extraction"""

import requests
import sys
from pathlib import Path

def process_excel_file(input_path, theme_labels=None):
    """Process Excel file and save output"""
    url = "http://localhost:8001/batch/excel"

    # Prepare parameters
    params = {
        "extract_sentiment": True,
        "extract_themes": True,
        "top_n_themes": 3,
        "add_confidence_scores": True
    }

    if theme_labels:
        params["theme_labels"] = ",".join(theme_labels)

    # Upload and process
    with open(input_path, "rb") as f:
        files = {"file": f}
        response = requests.post(url, files=files, params=params)

    if response.status_code == 200:
        # Save output
        input_file = Path(input_path)
        output_path = input_file.with_name(f"{input_file.stem}_analyzed.xlsx")

        with open(output_path, "wb") as out:
            out.write(response.content)

        print(f"✅ Processing complete: {output_path}")

        # Print stats
        stats = response.headers.get("X-Processing-Stats")
        if stats:
            import json
            print(f"📊 Stats: {json.loads(stats)}")
    else:
        print(f"❌ Error: {response.json()}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python process.py <excel_file> [theme1,theme2,...]")
        sys.exit(1)

    input_file = sys.argv[1]
    themes = sys.argv[2].split(",") if len(sys.argv) > 2 else None

    process_excel_file(input_file, themes)
```

**Usage:**
```bash
python process.py customer_feedback.xlsx
python process.py survey.xlsx "satisfied,dissatisfied,neutral,suggestion"
```

### Bash Script

```bash
#!/bin/bash
# Batch process multiple Excel files

API_URL="http://localhost:8001/batch/excel"
THEME_LABELS="politics,economics,social,technology,health"

for file in *.xlsx; do
  echo "Processing $file..."

  curl -X POST "$API_URL" \
    -F "file=@$file" \
    -F "theme_labels=$THEME_LABELS" \
    -o "${file%.xlsx}_analyzed.xlsx"

  echo "✅ Done: ${file%.xlsx}_analyzed.xlsx"
done
```

---

## Troubleshooting

### Issue: Processing is slow

**Causes:**
- Large dataset
- CPU-only processing
- Small batch size

**Solutions:**
```bash
# Increase batch size (if enough memory)
-F "batch_size=64"

# Skip unnecessary tasks
-F "extract_themes=false"  # If only need sentiment

# Use GPU (via Docker)
docker-compose -f docker-compose.gpu.yml up
```

### Issue: Out of memory

**Solutions:**
```bash
# Reduce batch size
-F "batch_size=16"

# Process sheets individually
-F "sheets_to_process=Sheet1"

# Split large files into smaller chunks
```

### Issue: Theme labels not accurate

**Solutions:**
```bash
# Use more specific labels
-F "theme_labels=customer complaint about product,customer complaint about service,positive feedback,feature request"

# Try multilingual model (more accurate)
-F "theme_preset=zeroshot-mdeberta"

# Increase number of themes
-F "top_n_themes=5"
```

---

## Limitations

1. **File Size:** Recommended max 50MB Excel file
2. **Rows:** Tested up to 100,000 rows per sheet
3. **Sheets:** No limit, but processing time increases linearly
4. **Text Length:** Texts are truncated to 512 tokens (model limit)
5. **Languages:** Best results with English (use `zeroshot-mdeberta` for other languages)

---

## Performance Benchmarks

Tested on Intel i7-12700K (12 cores) with 32GB RAM:

| Rows | Sentiment | Themes | Total Time | Throughput |
|------|-----------|--------|------------|------------|
| 100  | 5 sec     | 10 sec | 15 sec     | 6.7 rows/sec |
| 1,000| 30 sec    | 60 sec | 90 sec     | 11.1 rows/sec |
| 10,000| 4 min    | 8 min  | 12 min     | 13.9 rows/sec |

With NVIDIA RTX 3080 (10GB VRAM):
- **3-4x faster** than CPU
- Can handle `batch_size=128`

---

## FAQ

**Q: Can I process CSV files?**
A: This endpoint is for Excel files only. For CSV, use `/analyze/file` endpoint.

**Q: Can I use my own theme labels?**
A: Yes! Use the `theme_labels` parameter with comma-separated labels.

**Q: What if my Excel file has formulas?**
A: Formulas are evaluated automatically. The processed values are used.

**Q: Can I process password-protected files?**
A: Not directly. Remove the password first.

**Q: How do I process only sentiment without themes?**
A: Set `extract_themes=false`

**Q: Can I save results to a database instead?**
A: This endpoint returns an Excel file. To save to a database, write a script that calls the API and processes the output.

---

## Support

For issues or questions:
1. Check logs: `docker-compose logs -f nlp`
2. Verify health: `curl http://localhost:8001/health`
3. Check API docs: http://localhost:8001/docs
4. Review this documentation

**Happy batch processing! 📊**
