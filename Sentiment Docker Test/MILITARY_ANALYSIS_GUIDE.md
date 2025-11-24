# Military & Geopolitical Analysis Guide

Specialized guide for analyzing defense, military exercises, and geopolitical events with focus on UNITAS and regional security operations.

## Overview

This guide shows how to use CiceroWatch's batch processing capabilities for:
- 🎖️ **Military Exercise Detection** (UNITAS, PANAMAX, RIMPAC, etc.)
- 🌍 **Geopolitical Event Categorization**
- 🤝 **Defense Partnership Analysis**
- 🚢 **Maritime Security Monitoring**
- 📊 **Sentiment Analysis of Defense Coverage**

---

## Quick Start: UNITAS Exercise Detection

### Basic UNITAS Detection

```bash
curl -X POST http://localhost:8001/batch/excel \
  -F "file=@latin_america_news.xlsx" \
  -F "theme_labels=UNITAS exercises,military cooperation,naval operations"
```

### Comprehensive UNITAS Analysis

```bash
curl -X POST http://localhost:8001/batch/excel \
  -F "file=@defense_intelligence.xlsx" \
  -F "theme_labels=UNITAS naval exercises,PANAMAX exercises,SOUTHCOM operations,bilateral defense,multilateral cooperation,humanitarian assistance,disaster relief,maritime security,counter-narcotics,anti-submarine warfare,U.S. naval presence,partner nation capacity" \
  -F "top_n_themes=5" \
  -F "extract_sentiment=true"
```

---

## UNITAS Background

**UNITAS** (Latin: "Unity") is the world's longest-running annual multinational maritime exercise.

**Key Facts:**
- **Organizer:** U.S. Southern Command (SOUTHCOM) and U.S. Naval Forces Southern Command
- **Participants:** 20+ nations from South America, Central America, Caribbean, and global partners
- **Focus Areas:**
  - Anti-submarine warfare
  - Surface warfare
  - Maritime interdiction operations
  - Humanitarian assistance/disaster relief
  - Search and rescue
  - Counter-narcotics operations
  - Maritime security cooperation

**Why Track UNITAS:**
- Indicator of U.S.-Latin America defense relations
- Regional security cooperation trends
- Partner nation capabilities development
- Humanitarian assistance readiness

---

## Theme Label Strategies

### Strategy 1: Exercise-Specific Labels

**Best for:** Tracking specific named exercises

```python
exercise_labels = [
    "UNITAS exercises",
    "PANAMAX exercises",
    "RIMPAC exercises",
    "Trident Juncture",
    "Malabar exercises",
    "Talisman Sabre",
    "Cobra Gold",
    "Balikatan exercises"
]
```

### Strategy 2: Activity-Type Labels

**Best for:** Understanding types of military activities

```python
activity_labels = [
    "joint military exercises",
    "bilateral defense cooperation",
    "multilateral operations",
    "naval exercises",
    "air force operations",
    "ground force training",
    "cyber exercises",
    "special operations training"
]
```

### Strategy 3: Regional Security Labels

**Best for:** Regional analysis (Latin America/Caribbean)

```python
regional_labels = [
    "UNITAS naval exercises",
    "SOUTHCOM operations",
    "partner nation capacity building",
    "maritime security cooperation",
    "counter-narcotics operations",
    "humanitarian assistance",
    "disaster relief operations",
    "Caribbean security",
    "Pacific fleet operations",
    "Atlantic fleet operations"
]
```

### Strategy 4: Geopolitical Context Labels

**Best for:** Understanding strategic implications

```python
geopolitical_labels = [
    "UNITAS exercises",
    "U.S. military presence",
    "China influence",
    "Russia military cooperation",
    "regional security architecture",
    "defense partnerships",
    "strategic competition",
    "security cooperation",
    "military modernization",
    "arms sales"
]
```

### Strategy 5: Comprehensive Defense Analysis

**Best for:** Full-spectrum military intelligence

```python
comprehensive_labels = [
    # Specific Exercises
    "UNITAS naval exercises",
    "PANAMAX exercises",

    # Operations
    "joint operations",
    "humanitarian assistance disaster relief",
    "counter-narcotics missions",
    "anti-submarine warfare",
    "maritime interdiction",

    # Partnerships
    "U.S. naval cooperation",
    "bilateral defense agreements",
    "multilateral partnerships",
    "partner nation training",

    # Regional Security
    "maritime security",
    "regional stability",
    "security threats",
    "territorial disputes",

    # Strategic Context
    "China influence",
    "Russia military presence",
    "great power competition"
]
```

---

## Complete Examples

### Example 1: Annual UNITAS Exercise Monitoring

**Scenario:** Track UNITAS exercises over time to analyze participation trends, focus areas, and regional engagement.

```python
#!/usr/bin/env python3
"""Monitor UNITAS exercises from news and intelligence reports"""

import requests
import pandas as pd
from datetime import datetime

# API endpoint
url = "http://localhost:8001/batch/excel"

# UNITAS-specific labels
unitas_labels = [
    "UNITAS naval exercises",
    "anti-submarine warfare training",
    "surface warfare exercises",
    "maritime interdiction operations",
    "humanitarian assistance disaster relief",
    "search and rescue operations",
    "counter-narcotics operations",
    "U.S. Southern Command",
    "partner nation participation",
    "regional maritime security",
    "bilateral naval cooperation",
    "multilateral exercises"
]

# Process intelligence reports
with open("unitas_2025_coverage.xlsx", "rb") as f:
    response = requests.post(
        url,
        files={"file": f},
        params={
            "extract_sentiment": True,
            "extract_themes": True,
            "theme_labels": ",".join(unitas_labels),
            "top_n_themes": 5,
            "text_column": "article_text"
        }
    )

# Save analyzed data
output_file = f"unitas_analysis_{datetime.now().strftime('%Y%m%d')}.xlsx"
with open(output_file, "wb") as out:
    out.write(response.content)

print(f"✅ UNITAS analysis complete: {output_file}")

# Load and summarize
df = pd.read_excel(output_file)
print("\n📊 Theme Distribution:")
print(df['theme_1'].value_counts().head(10))

print("\n💭 Sentiment Breakdown:")
print(df['sentiment'].value_counts())
```

### Example 2: Multi-Exercise Regional Analysis

**Scenario:** Compare multiple exercises (UNITAS, PANAMAX, etc.) in Latin America region.

```bash
#!/bin/bash
# Analyze multiple regional exercises

curl -X POST http://localhost:8001/batch/excel \
  -F "file=@regional_exercises_2025.xlsx" \
  -F "theme_labels=UNITAS exercises,PANAMAX exercises,Tradewinds exercises,Resolute Sentinel,Southern Partnership Station,Continuing Promise,naval cooperation,disaster response,counter-narcotics,humanitarian assistance,partner capacity building,regional security" \
  -F "top_n_themes=5" \
  -F "extract_sentiment=true" \
  -F "add_confidence_scores=true" \
  -o "regional_exercises_analyzed.xlsx"

echo "✅ Regional exercise analysis complete"
```

### Example 3: UNITAS + Geopolitical Context

**Scenario:** Analyze UNITAS coverage with broader geopolitical implications.

```python
import requests

url = "http://localhost:8001/batch/excel"

# Labels combining exercise specifics with strategic context
strategic_labels = [
    # Exercise-specific
    "UNITAS naval exercises",
    "exercise participants",
    "exercise objectives",

    # Operational aspects
    "anti-submarine warfare",
    "humanitarian assistance",
    "maritime security",
    "counter-narcotics operations",

    # Strategic context
    "U.S. strategic presence",
    "China influence in Latin America",
    "Russia military cooperation",
    "regional security architecture",

    # Partnerships
    "bilateral defense cooperation",
    "multilateral partnerships",
    "partner nation capacity building",

    # Regional dynamics
    "Latin American security",
    "Caribbean stability",
    "Pacific partnerships",
    "Atlantic cooperation"
]

with open("strategic_analysis.xlsx", "rb") as f:
    response = requests.post(
        url,
        files={"file": f},
        params={
            "extract_sentiment": True,
            "extract_themes": True,
            "theme_labels": ",".join(strategic_labels),
            "top_n_themes": 7,  # More themes for complex analysis
            "sentiment_preset": "sentiment-twitter",
            "theme_preset": "zeroshot-mdeberta",  # More accurate for geopolitical
            "batch_size": 32
        }
    )

with open("strategic_analysis_output.xlsx", "wb") as out:
    out.write(response.content)
```

### Example 4: Time Series Analysis

**Scenario:** Track sentiment and theme trends over multiple years of UNITAS exercises.

```python
import requests
import pandas as pd
import matplotlib.pyplot as plt

url = "http://localhost:8001/batch/excel"

# Process historical data
years = ["2023", "2024", "2025"]
all_results = []

for year in years:
    print(f"Processing UNITAS {year}...")

    with open(f"unitas_{year}_news.xlsx", "rb") as f:
        response = requests.post(
            url,
            files={"file": f},
            params={
                "theme_labels": "UNITAS exercises,humanitarian assistance,anti-submarine warfare,regional cooperation,partner capacity,maritime security",
                "extract_sentiment": True,
                "top_n_themes": 3
            }
        )

    # Load results
    df = pd.read_excel(response.content)
    df['year'] = year
    all_results.append(df)

# Combine and analyze
combined_df = pd.concat(all_results, ignore_index=True)

# Sentiment trends
sentiment_trends = combined_df.groupby(['year', 'sentiment']).size().unstack(fill_value=0)
sentiment_trends.plot(kind='bar', stacked=True)
plt.title('UNITAS Exercise Sentiment Over Time')
plt.savefig('unitas_sentiment_trends.png')

# Top themes by year
for year in years:
    print(f"\n📊 Top Themes - UNITAS {year}:")
    year_df = combined_df[combined_df['year'] == year]
    print(year_df['theme_1'].value_counts().head(5))

# Save combined dataset
combined_df.to_excel('unitas_historical_analysis.xlsx', index=False)
print("\n✅ Historical analysis complete!")
```

---

## Integration with Other Data Sources

### Combining with ACLED Conflict Data

Once ACLED integration is added (see main discussion), you can correlate military exercises with conflict events:

```python
# Future implementation
import requests

# Process UNITAS data
unitas_response = requests.post(
    "http://localhost:8001/batch/excel",
    files={"file": open("unitas_news.xlsx", "rb")},
    params={"theme_labels": "UNITAS exercises,regional security"}
)

# Get ACLED conflict events in same region/timeframe
acled_response = requests.get(
    "http://localhost:8004/acled/events",
    params={
        "country": "Colombia,Peru,Chile",
        "start_date": "2025-08-01",
        "end_date": "2025-09-30"
    }
)

# Correlate exercise timing with regional conflict levels
# ... analysis code
```

### Social Media Monitoring

```bash
# Analyze social media discussions about UNITAS
curl -X POST http://localhost:8001/batch/excel \
  -F "file=@twitter_unitas_mentions.xlsx" \
  -F "theme_labels=UNITAS exercises,military cooperation,public opinion,transparency,effectiveness,regional security,U.S. influence" \
  -F "sentiment_preset=sentiment-twitter" \
  -F "extract_sentiment=true"
```

---

## Analysis Outputs

### What You Get

For each text/row in your spreadsheet, the system adds:

**Sentiment Columns:**
- `sentiment`: positive, negative, or neutral
- `sentiment_confidence`: 0.0 to 1.0

**Theme Columns (configurable 1-10, default 3):**
- `theme_1`, `theme_2`, `theme_3`: Top detected themes
- `theme_1_score`, `theme_2_score`, `theme_3_score`: Confidence scores

### Example Output for UNITAS Article

**Input:**
```
| date       | source | article_text                                                |
|------------|--------|-------------------------------------------------------------|
| 2025-09-15 | Navy   | The U.S. Navy and 15 partner nations kicked off UNITAS... |
```

**Output:**
```
| date       | source | article_text        | sentiment | confidence | theme_1              | theme_1_score | theme_2                      | theme_2_score | theme_3           | theme_3_score |
|------------|--------|---------------------|-----------|------------|----------------------|---------------|------------------------------|---------------|-------------------|---------------|
| 2025-09-15 | Navy   | The U.S. Navy...    | positive  | 0.87       | UNITAS exercises     | 0.94          | multilateral partnerships    | 0.78          | maritime security | 0.65          |
```

---

## Advanced Analysis Techniques

### 1. Participation Trend Analysis

Track which countries are mentioned in UNITAS coverage:

```python
# Custom labels for country participation
country_labels = [
    "UNITAS exercises",
    "U.S. Navy participation",
    "Brazil Navy participation",
    "Chile Navy participation",
    "Colombia Navy participation",
    "Peru Navy participation",
    "Argentina Navy participation",
    "Ecuador Navy participation",
    "Mexico Navy participation",
    "Caribbean participation",
    "European partner participation"
]
```

### 2. Focus Area Evolution

Track how UNITAS focus areas change over time:

```python
focus_labels = [
    "UNITAS exercises",
    "anti-submarine warfare priority",
    "humanitarian assistance focus",
    "disaster relief training",
    "counter-narcotics emphasis",
    "maritime security operations",
    "cyber warfare training",
    "electronic warfare exercises"
]
```

### 3. Media Framing Analysis

Understand how different sources frame UNITAS:

```python
framing_labels = [
    "UNITAS exercises",
    "positive regional cooperation",
    "U.S. military dominance concerns",
    "sovereignty concerns",
    "capacity building benefits",
    "humanitarian mission focus",
    "military competition context",
    "democratic partnerships"
]
```

---

## Performance Optimization

### For Large Datasets (10,000+ articles)

```python
# Optimize for speed and memory
params = {
    "extract_sentiment": True,
    "extract_themes": True,
    "theme_labels": "UNITAS exercises,maritime security,regional cooperation",
    "top_n_themes": 3,  # Fewer themes = faster
    "batch_size": 64,   # Larger batches (if enough memory)
    "add_confidence_scores": False  # Skip if not needed
}
```

### Processing Multiple Sheets

```bash
# Process specific sheets only
curl -X POST http://localhost:8001/batch/excel \
  -F "file=@multi_year_data.xlsx" \
  -F "sheets_to_process=2023,2024,2025" \
  -F "theme_labels=UNITAS exercises,regional security"
```

---

## Validation & Quality Checks

### Check Theme Accuracy

```python
import pandas as pd

# Load analyzed data
df = pd.read_excel('unitas_analyzed.xlsx')

# Check confidence scores
low_confidence = df[df['theme_1_score'] < 0.5]
print(f"⚠️  {len(low_confidence)} rows with low confidence (<0.5)")

# Review ambiguous classifications
print("\nAmbiguous classifications:")
print(low_confidence[['article_text', 'theme_1', 'theme_1_score']].head())
```

### Refine Labels Based on Results

If themes aren't accurate, refine your labels:

```python
# Initial labels (too broad)
initial_labels = ["military", "cooperation", "training"]

# Refined labels (more specific)
refined_labels = [
    "UNITAS naval exercises",
    "U.S.-Latin America military cooperation",
    "anti-submarine warfare training"
]
```

---

## Best Practices

### 1. Label Specificity

**Too Broad:**
```python
["military", "exercises", "cooperation"]
```

**Better:**
```python
["UNITAS exercises", "PANAMAX exercises", "bilateral naval cooperation"]
```

**Best:**
```python
["UNITAS 2025 exercises", "UNITAS anti-submarine warfare component", "UNITAS humanitarian assistance missions"]
```

### 2. Number of Labels

- **5-10 labels:** Good for focused analysis
- **10-15 labels:** Balanced coverage
- **15-20 labels:** Comprehensive but may dilute scores

### 3. Top N Themes

- **top_n_themes=3:** Standard, good for most cases
- **top_n_themes=5:** Better for multi-faceted events
- **top_n_themes=7-10:** Use when you need full context

### 4. Preset Selection

**For military/geopolitical analysis:**
- `theme_preset=zeroshot-mdeberta`: More accurate for complex topics
- `sentiment_preset=sentiment-twitter`: Good for news and social media
- `batch_size=32`: Good balance for most datasets

---

## Common Issues & Solutions

### Issue: UNITAS not detected

**Cause:** Label too generic or competing labels

**Solution:**
```python
# Instead of:
labels = ["military exercises", "cooperation"]

# Use:
labels = ["UNITAS naval exercises", "UNITAS 2025", "U.S. Southern Command exercises"]
```

### Issue: Too many themes detected

**Cause:** Labels too similar or overlapping

**Solution:**
```python
# Instead of:
labels = ["naval exercises", "military exercises", "joint exercises", "UNITAS exercises"]

# Use:
labels = ["UNITAS exercises", "humanitarian missions", "anti-submarine warfare", "maritime security"]
```

### Issue: Low confidence scores

**Cause:** Ambiguous text or poor label match

**Solution:**
1. Use `theme_preset=zeroshot-mdeberta` (more accurate)
2. Make labels more specific
3. Increase `top_n_themes` to see alternative matches

---

## Output Analysis Scripts

### Generate Summary Report

```python
import pandas as pd
from collections import Counter

df = pd.read_excel('unitas_analyzed.xlsx')

print("=" * 60)
print("UNITAS EXERCISE ANALYSIS SUMMARY")
print("=" * 60)

print(f"\n📊 Total Articles: {len(df)}")
print(f"\n💭 Sentiment Distribution:")
print(df['sentiment'].value_counts())

print(f"\n🎯 Top 10 Themes:")
all_themes = []
for col in ['theme_1', 'theme_2', 'theme_3']:
    if col in df.columns:
        all_themes.extend(df[col].dropna().tolist())

theme_counts = Counter(all_themes)
for theme, count in theme_counts.most_common(10):
    print(f"  {theme}: {count}")

print(f"\n⭐ High Confidence UNITAS Mentions:")
unitas_df = df[df['theme_1'].str.contains('UNITAS', case=False, na=False)]
unitas_high = unitas_df[unitas_df['theme_1_score'] > 0.8]
print(f"  {len(unitas_high)} articles with high confidence (>0.8)")

print("\n✅ Analysis complete!")
```

---

## Next Steps

1. **Start with defaults:** Test without custom labels to see baseline
2. **Refine labels:** Based on initial results, create custom labels
3. **Validate:** Check a sample of results manually
4. **Iterate:** Adjust labels and parameters as needed
5. **Automate:** Set up regular processing for ongoing monitoring

---

## Support

For questions or issues:
- Check main documentation: `BATCH_PROCESSING.md`
- API docs: http://localhost:8001/docs
- Test endpoint: `POST /batch/excel`

**Happy military intelligence analysis! 🎖️🌍**
