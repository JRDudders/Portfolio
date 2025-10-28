"""
NLP Processing Module

Contains business logic for NLP tasks by wrapping the existing nlp.py module.
"""

from typing import List, Dict, Any, Optional
import requests
from bs4 import BeautifulSoup

# Import the actual NLP implementation
from nlp import run_task, PRESETS, DEFAULT_ZS_LABELS


def analyze_sentiment(text: str) -> Dict[str, Any]:
    """
    Analyze sentiment of text using transformers model

    Args:
        text: Input text

    Returns:
        Dictionary with sentiment results
    """
    results = run_task([text], preset="sentiment-twitter")

    if results and len(results) > 0:
        result = results[0]

        # Extract labels and scores
        labels = result.get("labels", [])
        scores = result.get("scores", [])

        # Determine sentiment (highest score)
        if labels and scores:
            max_idx = scores.index(max(scores))
            sentiment = labels[max_idx]
            confidence = scores[max_idx]

            # Create scores dict
            scores_dict = {labels[i]: scores[i] for i in range(len(labels))}

            return {
                "sentiment": sentiment,
                "confidence": float(confidence),
                "scores": scores_dict
            }

    # Fallback
    return {
        "sentiment": "neutral",
        "confidence": 0.0,
        "scores": {}
    }


def extract_entities(text: str) -> Dict[str, Any]:
    """
    Extract named entities from text using NER model

    Args:
        text: Input text

    Returns:
        Dictionary with extracted entities
    """
    results = run_task([text], preset="ner-conll")

    if results and len(results) > 0:
        entities = results[0]  # List of entity dicts

        # Group by entity type
        entity_types: Dict[str, List[str]] = {}
        all_entities = []

        for ent in entities:
            entity_type = ent.get("entity_group", ent.get("entity", "OTHER"))
            word = ent.get("word", "")
            score = ent.get("score", 0.0)

            if entity_type not in entity_types:
                entity_types[entity_type] = []

            entity_types[entity_type].append(word)
            all_entities.append({
                "text": word,
                "type": entity_type,
                "score": float(score)
            })

        return {
            "entities": all_entities,
            "entity_types": entity_types,
            "num_entities": len(all_entities)
        }

    return {
        "entities": [],
        "entity_types": {},
        "num_entities": 0
    }


def analyze_topics(text: str) -> Dict[str, Any]:
    """
    Perform topic modeling on text using zero-shot classification

    Args:
        text: Input text

    Returns:
        Dictionary with topic results
    """
    # Use zero-shot classification with default topic labels
    results = run_task([text], preset="zeroshot-bart", labels=DEFAULT_ZS_LABELS)

    if results and len(results) > 0:
        result = results[0]
        labels = result.get("labels", [])
        scores = result.get("scores", [])

        # Get top topics (score > threshold)
        topics = []
        for i, (label, score) in enumerate(zip(labels, scores)):
            if score > 0.1:  # threshold
                topics.append({
                    "topic": label,
                    "score": float(score),
                    "rank": i + 1
                })

        return {
            "topics": topics,
            "num_topics": len(topics),
            "all_labels": labels,
            "all_scores": [float(s) for s in scores]
        }

    return {
        "topics": [],
        "num_topics": 0,
        "all_labels": [],
        "all_scores": []
    }


def scrape_and_analyze_url(url: str, tasks: List[str]) -> Dict[str, Any]:
    """
    Scrape URL and analyze content

    Args:
        url: URL to scrape
        tasks: List of analysis tasks to perform

    Returns:
        Dictionary with scraping and analysis results
    """
    try:
        # Fetch URL
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract text
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)

        # Limit text length for analysis
        max_length = 10000
        if len(text) > max_length:
            text = text[:max_length]

        # Analyze based on requested tasks
        results = {
            "url": url,
            "status_code": response.status_code,
            "text_length": len(text),
            "title": soup.title.string if soup.title else None,
        }

        if "sentiment" in tasks:
            results["sentiment"] = analyze_sentiment(text)

        if "entities" in tasks:
            results["entities"] = extract_entities(text)

        if "topics" in tasks:
            results["topics"] = analyze_topics(text)

        return results

    except Exception as e:
        return {
            "url": url,
            "error": str(e),
            "success": False
        }
