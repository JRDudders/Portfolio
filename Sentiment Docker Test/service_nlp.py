"""
NLP Microservice

Handles all NLP-related tasks:
- Sentiment analysis
- Entity extraction
- Topic modeling
- Text classification
- URL scraping and analysis
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, HttpUrl
from typing import List, Optional, Dict, Any
import uvicorn
import os
import io
import json
import pandas as pd
from bs4 import BeautifulSoup
import trafilatura

# Import NLP processing modules
from nlp_processor import (
    analyze_sentiment,
    extract_entities,
    analyze_topics,
    scrape_and_analyze_url
)
from nlp import run_task, PRESETS, DEFAULT_ZS_LABELS, preprocess_for_task

app = FastAPI(
    title="CiceroWatch NLP Service",
    description="Natural Language Processing microservice",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Helper Functions
# ============================================================

def _as_json_bytes(obj: Any) -> bytes:
    """Convert object to JSON bytes"""
    return json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")


def _parse_labels_csv(s: Optional[str]) -> Optional[List[str]]:
    """Parse comma-separated labels"""
    if not s or not s.strip():
        return None
    labels = [x.strip() for x in s.split(",") if x.strip()]
    return labels if labels else None


def _texts_from_json_bytes(b: bytes) -> List[str]:
    """Extract texts from JSON bytes"""
    data = json.loads(b.decode("utf-8"))
    if isinstance(data, list):
        if all(isinstance(x, str) for x in data):
            return data
        if all(isinstance(x, dict) for x in data):
            out = []
            for obj in data:
                if "text" in obj and isinstance(obj["text"], str):
                    out.append(obj["text"])
            if out:
                return out
    raise ValueError("JSON must be a list of strings or list of {'text': ...} objects")


def _texts_from_csv_bytes(b: bytes) -> List[str]:
    """Extract texts from CSV bytes"""
    df = pd.read_csv(io.BytesIO(b))
    # Prefer 'text' column; otherwise take the first object dtype column
    if "text" in df.columns:
        col = "text"
    else:
        obj_cols = [c for c in df.columns if df[c].dtype == object]
        if not obj_cols:
            raise ValueError("CSV must contain a 'text' column or at least one string column")
        col = obj_cols[0]
    vals = df[col].astype(str).tolist()
    return vals


def _make_download(name: str, payload: bytes, mime: str = "application/json") -> StreamingResponse:
    """Create downloadable file response"""
    resp = StreamingResponse(io.BytesIO(payload), media_type=mime)
    resp.headers["Content-Disposition"] = f'attachment; filename="{name}"'
    return resp


def _extract_text_from_html(html: str) -> str:
    """Extract text from HTML using trafilatura, fallback to BeautifulSoup"""
    # Try trafilatura first (better at extracting main content)
    try:
        extracted = trafilatura.extract(html, include_comments=False, favor_recall=False)
        if extracted and extracted.strip():
            return extracted.strip()
    except Exception:
        pass

    # Fallback: BeautifulSoup get_text
    soup = BeautifulSoup(html, "html.parser")
    # Drop script/style tags
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()
    text = soup.get_text("\n", strip=True)
    return text


# ============================================================
# Request Models
# ============================================================

class TextAnalysisRequest(BaseModel):
    text: str
    tasks: List[str] = ["sentiment", "entities"]


class URLAnalysisRequest(BaseModel):
    url: HttpUrl
    tasks: List[str] = ["sentiment", "entities", "topics"]


# ============================================================
# Health Check
# ============================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "service": "nlp",
        "status": "healthy",
        "version": "1.0.0"
    }


# ============================================================
# NLP Endpoints
# ============================================================

@app.post("/analyze/text")
async def analyze_text(request: TextAnalysisRequest):
    """
    Analyze text with requested NLP tasks

    Tasks:
    - sentiment: Sentiment analysis
    - entities: Named entity recognition
    - topics: Topic modeling
    """
    try:
        results = {}

        if "sentiment" in request.tasks:
            results["sentiment"] = analyze_sentiment(request.text)

        if "entities" in request.tasks:
            results["entities"] = extract_entities(request.text)

        if "topics" in request.tasks:
            results["topics"] = analyze_topics(request.text)

        return {
            "success": True,
            "text_length": len(request.text),
            "results": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/url")
async def analyze_url(request: URLAnalysisRequest):
    """
    Scrape URL and analyze content
    """
    try:
        result = scrape_and_analyze_url(
            url=str(request.url),
            tasks=request.tasks
        )
        return {
            "success": True,
            "url": str(request.url),
            "results": result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/file")
async def analyze_file(
    file: UploadFile = File(...),
    preset: Optional[str] = Query(None, description="Preset name from nlp.PRESETS"),
    labels: Optional[str] = Query(None, description="Comma-separated labels for zero-shot"),
    include_stopwords: Optional[bool] = Query(False),
):
    """
    Analyze text file and return annotated results as downloadable file

    Supports:
    - JSON: list of strings or objects with 'text' field
    - CSV: file with 'text' column or first string column
    - HTML/HTM: extracts text content from HTML file

    Returns annotated JSON file with predictions
    """
    try:
        b = await file.read()
        name = (file.filename or "").lower()

        # Parse file based on type
        if name.endswith(".json"):
            texts = _texts_from_json_bytes(b)
        elif name.endswith(".csv"):
            texts = _texts_from_csv_bytes(b)
        elif name.endswith((".html", ".htm")):
            # Extract text from HTML
            html = b.decode("utf-8", errors="ignore")
            text = _extract_text_from_html(html)
            texts = [text]
        else:
            # Try JSON first, then CSV
            try:
                texts = _texts_from_json_bytes(b)
            except Exception:
                texts = _texts_from_csv_bytes(b)

        # Determine task from preset
        task = PRESETS.get(preset, (None, None, {}))[0] if preset else None

        # Keep original texts for output, preprocess separately
        original_texts = texts
        processed_texts = [preprocess_for_task(t, task or "") for t in texts]

        # Parse labels for zero-shot tasks
        lbls = None
        if preset and "zeroshot" in preset:
            lbls = _parse_labels_csv(labels)
            if not lbls:
                lbls = DEFAULT_ZS_LABELS

        # Run NLP task
        predictions = run_task(processed_texts, preset=preset, labels=lbls)

        # Merge original texts with predictions
        if (task == "token-classification") or (preset and "ner" in preset):
            # NER: keep entities format
            output = [{"text": t, "entities": p} for t, p in zip(original_texts, predictions)]
        else:
            # Classification: merge text with scores
            output = []
            for t, p in zip(original_texts, predictions):
                result_dict = {"text": t}
                if isinstance(p, dict):
                    result_dict.update(p)  # Add labels, scores, etc.
                output.append(result_dict)

        # Return as downloadable JSON
        payload = _as_json_bytes({"preset": preset, "results": output})
        return _make_download("predictions.json", payload, "application/json")

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"File processing failed: {str(e)}")


# ============================================================
# Specific Task Endpoints
# ============================================================

@app.post("/sentiment")
async def sentiment_only(request: TextAnalysisRequest):
    """Sentiment analysis only"""
    try:
        result = analyze_sentiment(request.text)
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/entities")
async def entities_only(request: TextAnalysisRequest):
    """Entity extraction only"""
    try:
        result = extract_entities(request.text)
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/topics")
async def topics_only(request: TextAnalysisRequest):
    """Topic modeling only"""
    try:
        result = analyze_topics(request.text)
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    port = int(os.getenv("SERVICE_PORT", 8001))

    print("=" * 70)
    print("CiceroWatch NLP Service")
    print("=" * 70)
    print(f"Listening on: http://0.0.0.0:{port}")
    print("Features: Sentiment, Entities, Topics, URL Analysis")
    print("=" * 70)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
