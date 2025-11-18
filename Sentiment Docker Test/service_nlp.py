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
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Import URL fetch utility
from url_fetch import fetch_url, guess_file_extension

# Import NLP processing modules
from nlp_processor import (
    analyze_sentiment,
    extract_entities,
    analyze_topics
)
from nlp import run_task, PRESETS, DEFAULT_ZS_LABELS, preprocess_for_task
from adapters import process_url
from batch_processor import (
    process_excel_file,
    BatchProcessingConfig
)

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


def _truncate_text(text: str, max_length: int = 1000) -> str:
    """Truncate text to max_length characters for display, adding ellipsis if truncated"""
    if len(text) <= max_length:
        return text
    return text[:max_length].rstrip() + "..."


def _get_predictions_filename(input_filename: str) -> str:
    """Generate output filename with _predictions appended before extension"""
    from pathlib import Path
    p = Path(input_filename)
    stem = p.stem or "output"
    suffix = p.suffix or ".json"
    return f"{stem}_predictions{suffix}"


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


def _chunk_html_for_topics(html: str, min_chunk_length: int = 100) -> List[str]:
    """
    Split HTML into chunks (paragraphs/sections) for topic modeling.
    Useful for books, long articles, etc.

    Args:
        html: HTML content
        min_chunk_length: Minimum character length for a chunk to be included

    Returns:
        List of text chunks
    """
    soup = BeautifulSoup(html, "html.parser")

    # Drop script/style tags
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()

    chunks = []

    # Try to extract by structural elements (chapters, sections, divs, paragraphs)
    # Priority: div.chapter, div.section, h1-h6 + following content, then paragraphs

    # First try to find chapter/section divs
    for div in soup.find_all(['div', 'section', 'article']):
        if div.get('class') and any(c in str(div.get('class')).lower() for c in ['chapter', 'section', 'content']):
            text = div.get_text("\n", strip=True)
            if len(text) >= min_chunk_length:
                chunks.append(text)

    # If we got good chunks from structural elements, use those
    if chunks:
        return chunks

    # Otherwise, fall back to paragraphs
    for p in soup.find_all(['p', 'div']):
        text = p.get_text(strip=True)
        if len(text) >= min_chunk_length:
            chunks.append(text)

    # If still no chunks, split by double newlines
    if not chunks:
        full_text = soup.get_text("\n", strip=True)
        paragraphs = full_text.split('\n\n')
        chunks = [p.strip() for p in paragraphs if len(p.strip()) >= min_chunk_length]

    # Ensure we have at least some chunks
    if not chunks:
        # Last resort: split by single newlines
        full_text = soup.get_text("\n", strip=True)
        lines = full_text.split('\n')
        chunks = [line.strip() for line in lines if len(line.strip()) >= min_chunk_length]

    return chunks if chunks else [_extract_text_from_html(html)]


# ============================================================
# Request Models
# ============================================================

class TextAnalysisRequest(BaseModel):
    text: str
    tasks: List[str] = ["sentiment", "entities"]


class StanceDetectionRequest(BaseModel):
    """Request model for stance detection"""
    texts: List[str]
    claim: str
    preset: Optional[str] = "stance-deberta"
    hypothesis_template: Optional[str] = "{}"


class URLAnalysisRequest(BaseModel):
    """Request model for URL analysis with full adapter support"""
    url: HttpUrl
    # Rendering options
    render: bool = False
    renderer: str = "playwright"
    cookies: Optional[str] = None
    wait_selector: Optional[str] = None
    scroll_passes: int = 0
    render_timeout_ms: int = 3600000  # 1 hour
    extra_headers: Optional[Dict[str, Any]] = None
    # Crawling options
    crawl: bool = False
    max_pages: int = 10
    max_depth: int = 2
    same_host_only: bool = True
    delay_ms: int = 1000
    # Analysis options
    preset: Optional[str] = None
    labels: Optional[List[str]] = None
    claim: Optional[str] = None  # Claim/hypothesis for stance detection
    include_stopwords: bool = False


# ============================================================
# Health Check
# ============================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    logger.info("Health check requested")
    return {
        "service": "nlp",
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
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
    Scrape URL and analyze content with full adapter support

    Supports:
    - Zero-shot classification with custom labels
    - Sentiment analysis, NER, topic modeling
    - JavaScript rendering (Playwright/Selenium)
    - Site crawling with multi-page analysis
    - Custom presets from nlp.PRESETS

    Returns annotated HTML file with predictions
    """
    try:
        # Use the adapter system for full-featured URL processing
        result = await process_url(
            url=str(request.url),
            app=app,
            # Rendering
            render=request.render,
            renderer=request.renderer,
            cookies=request.cookies,
            wait_selector=request.wait_selector,
            scroll_passes=request.scroll_passes,
            render_timeout_ms=request.render_timeout_ms,
            extra_headers=request.extra_headers,
            # Crawling
            crawl=request.crawl,
            max_pages=request.max_pages,
            max_depth=request.max_depth,
            same_host_only=request.same_host_only,
            delay_ms=request.delay_ms,
            # Analysis
            task=None,  # Will be determined from preset
            preset=request.preset,
            labels=request.labels,
            claim=request.claim,
            include_stopwords=request.include_stopwords,
        )

        # Return the processed content as a downloadable file
        return _make_download(result.filename, result.content, result.media_type)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/file")
async def analyze_file(
    file: UploadFile = File(...),
    preset: Optional[str] = Query(None, description="Preset name from nlp.PRESETS"),
    labels: Optional[str] = Query(None, description="Comma-separated labels for zero-shot"),
    claim: Optional[str] = Query(None, description="Claim/hypothesis for stance detection"),
    include_stopwords: Optional[bool] = Query(False),
):
    """
    Analyze text file and return annotated results as downloadable file

    Supports:
    - JSON: list of strings or objects with 'text' field
    - CSV: file with 'text' column or first string column
    - HTML/HTM: extracts text content from HTML file
      * For topic modeling: automatically chunks into paragraphs/sections (great for books!)
      * For other tasks: treats as single document

    Returns annotated JSON file with predictions
    """
    logger.info(f"File analysis started: {file.filename}, preset={preset}, claim={claim}")
    try:
        b = await file.read()
        original_filename = file.filename or "input.json"
        name = original_filename.lower()

        # Determine task from preset early (needed for HTML chunking decision)
        task = PRESETS.get(preset, (None, None, {}))[0] if preset else None

        # Parse file based on type
        if name.endswith(".json"):
            texts = _texts_from_json_bytes(b)
        elif name.endswith(".csv"):
            texts = _texts_from_csv_bytes(b)
        elif name.endswith((".html", ".htm")):
            # Extract text from HTML
            html = b.decode("utf-8", errors="ignore")

            # For topic modeling, chunk the HTML into paragraphs/sections
            # For other tasks (sentiment, NER), treat as single document
            is_topic_task = task and ("topics" in task or "topic" in task)

            if is_topic_task:
                # Split into chunks for topic modeling (books, long articles)
                texts = _chunk_html_for_topics(html, min_chunk_length=100)
            else:
                # Single document for sentiment, NER, etc.
                text = _extract_text_from_html(html)
                texts = [text]
        else:
            # Try JSON first, then CSV
            try:
                texts = _texts_from_json_bytes(b)
            except Exception:
                texts = _texts_from_csv_bytes(b)

        # Keep original texts for output, preprocess separately
        original_texts = texts
        processed_texts = [preprocess_for_task(t, task or "") for t in texts]

        # Parse labels for zero-shot tasks
        lbls = None
        if preset and "zeroshot" in preset:
            lbls = _parse_labels_csv(labels)
            if not lbls:
                lbls = DEFAULT_ZS_LABELS

        # Validate claim for stance detection
        if preset and "stance" in preset:
            if not claim:
                raise ValueError("Stance detection requires a 'claim' parameter")

        # Run NLP task
        predictions = run_task(processed_texts, preset=preset, labels=lbls, claim=claim)

        logger.info(f"File analysis completed: {len(original_texts)} texts processed")

        # Merge original texts with predictions
        if (task == "token-classification") or (preset and "ner" in preset):
            # NER: keep entities format
            output = [{"text (analyzed in full, truncated for display)": _truncate_text(t), "entities": p} for t, p in zip(original_texts, predictions)]
        else:
            # Classification: merge text with scores
            output = []
            for t, p in zip(original_texts, predictions):
                result_dict = {"text (analyzed in full, truncated for display)": _truncate_text(t)}
                if isinstance(p, dict):
                    result_dict.update(p)  # Add labels, scores, etc.
                output.append(result_dict)

        # Return as downloadable JSON
        payload = _as_json_bytes({"preset": preset, "results": output})
        output_filename = _get_predictions_filename(original_filename)
        return _make_download(output_filename, payload, "application/json")

    except Exception as e:
        logger.error(f"File processing failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"File processing failed: {str(e)}")


class FileURLRequest(BaseModel):
    """Request model for analyzing file from URL"""
    url: HttpUrl
    preset: Optional[str] = None
    labels: Optional[str] = None  # Comma-separated for zero-shot
    claim: Optional[str] = None  # Claim/hypothesis for stance detection
    include_stopwords: Optional[bool] = False


@app.post("/analyze/file-from-url")
async def analyze_file_from_url(request: FileURLRequest):
    """
    Download and analyze a file from URL

    Supports same file types as /analyze/file:
    - JSON: list of strings or objects with 'text' field
    - CSV: file with 'text' column or first string column
    - HTML/HTM: extracts text content (with smart chunking for topics)

    Returns annotated JSON file with predictions
    """
    try:
        # Fetch file from URL
        file_bytes, content_type = await fetch_url(str(request.url))

        # Extract filename from URL for output naming
        from pathlib import Path
        from urllib.parse import urlparse
        url_path = urlparse(str(request.url)).path
        original_filename = Path(url_path).name or "downloaded_file"

        # Guess extension
        ext = guess_file_extension(str(request.url), content_type)

        # Ensure filename has proper extension
        if not original_filename.lower().endswith(ext):
            original_filename = f"{Path(original_filename).stem}{ext}"

        # Determine task from preset early (needed for HTML chunking decision)
        task = PRESETS.get(request.preset, (None, None, {}))[0] if request.preset else None

        # Parse file based on type
        if ext in ['.json']:
            texts = _texts_from_json_bytes(file_bytes)
        elif ext in ['.csv']:
            texts = _texts_from_csv_bytes(file_bytes)
        elif ext in ['.html', '.htm']:
            # Extract text from HTML
            html = file_bytes.decode("utf-8", errors="ignore")

            # For topic modeling, chunk the HTML into paragraphs/sections
            # For other tasks (sentiment, NER), treat as single document
            is_topic_task = task and ("topics" in task or "topic" in task)

            if is_topic_task:
                # Split into chunks for topic modeling (books, long articles)
                texts = _chunk_html_for_topics(html, min_chunk_length=100)
            else:
                # Single document for sentiment, NER, etc.
                text = _extract_text_from_html(html)
                texts = [text]
        else:
            # Try JSON first, then CSV
            try:
                texts = _texts_from_json_bytes(file_bytes)
            except Exception:
                texts = _texts_from_csv_bytes(file_bytes)

        # Keep original texts for output, preprocess separately
        original_texts = texts
        processed_texts = [preprocess_for_task(t, task or "") for t in texts]

        # Parse labels if provided
        labels_list = _parse_labels_csv(request.labels) if request.labels else None

        # Validate claim for stance detection
        if request.preset and "stance" in request.preset:
            if not request.claim:
                raise ValueError("Stance detection requires a 'claim' parameter")

        # Run analysis
        if request.preset:
            # Use preset (note: include_stopwords is handled in adapters, not in run_task)
            predictions = run_task(
                processed_texts,
                preset=request.preset,
                labels=labels_list,
                claim=request.claim
            )
        else:
            raise ValueError("preset parameter is required")

        # Format output
        if isinstance(predictions, dict) and "topics" in predictions:
            # Topic modeling result
            output = predictions
        else:
            # Per-document results
            output = []
            for t, p in zip(original_texts, predictions):
                result_dict = {"text (analyzed in full, truncated for display)": _truncate_text(t)}
                if isinstance(p, dict):
                    result_dict.update(p)  # Add labels, scores, etc.
                output.append(result_dict)

        # Return as downloadable JSON
        payload = _as_json_bytes({"preset": request.preset, "url": str(request.url), "results": output})
        output_filename = _get_predictions_filename(original_filename)
        return _make_download(output_filename, payload, "application/json")

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"File processing from URL failed: {str(e)}")


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


@app.post("/stance")
async def stance_detection(request: StanceDetectionRequest):
    """
    NLI-based stance detection (arxiv:2305.01723)

    Classifies each text's stance towards a claim as:
    - SUPPORT: Text entails/agrees with the claim
    - OPPOSE: Text contradicts the claim
    - NEUTRAL: No clear stance

    Uses pre-trained NLI models (DeBERTa-v3-base/large-mnli) to classify
    textual entailment without requiring task-specific training data.

    Example:
        {
            "texts": ["Climate change is real and urgent", "Weather changes naturally"],
            "claim": "Climate change is caused by humans",
            "preset": "stance-deberta"
        }
    """
    logger.info(f"Stance detection started: {len(request.texts)} texts, claim='{request.claim}', preset={request.preset}")
    try:
        # Preprocess texts
        processed_texts = [preprocess_for_task(t, "stance-detection") for t in request.texts]

        # Run stance detection
        results = run_task(
            processed_texts,
            preset=request.preset,
            claim=request.claim
        )

        logger.info(f"Stance detection completed: {len(results)} texts processed")

        return {
            "success": True,
            "claim": request.claim,
            "preset": request.preset,
            "results": results
        }
    except Exception as e:
        logger.error(f"Stance detection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch/excel")
async def batch_process_excel(
    file: UploadFile = File(...),
    extract_sentiment: bool = Query(True, description="Extract sentiment from texts"),
    extract_themes: bool = Query(True, description="Extract themes using zero-shot classification"),
    theme_labels: Optional[str] = Query(None, description="Comma-separated theme labels (optional, uses defaults if not provided)"),
    sentiment_preset: str = Query("sentiment-twitter", description="Sentiment analysis preset"),
    theme_preset: str = Query("zeroshot-bart", description="Zero-shot classification preset"),
    text_column: Optional[str] = Query(None, description="Text column name (auto-detect if not provided)"),
    sheets_to_process: Optional[str] = Query(None, description="Comma-separated sheet names (process all if not provided)"),
    top_n_themes: int = Query(3, description="Number of top themes to extract", ge=1, le=10),
    add_confidence_scores: bool = Query(True, description="Include confidence scores in output"),
    batch_size: int = Query(32, description="Batch size for processing", ge=1, le=128),
):
    """
    Batch process Excel file with multiple sheets for sentiment and theme extraction

    **Features:**
    - Processes all sheets (or specify which ones)
    - Extracts sentiment and/or themes (zero-shot classification)
    - Optional custom labels for theme detection
    - Auto-detects text column or specify manually
    - Preserves original data and adds result columns
    - Returns enhanced Excel file

    **Example:**
    ```bash
    curl -X POST http://localhost:8001/batch/excel \\
      -F "file=@data.xlsx" \\
      -F "extract_sentiment=true" \\
      -F "extract_themes=true" \\
      -F "theme_labels=politics,economics,social issues,environment,technology"
    ```

    **Output Columns:**
    - Original columns (preserved)
    - `sentiment`: Sentiment label (positive/negative/neutral)
    - `sentiment_confidence`: Confidence score (if enabled)
    - `theme_1`, `theme_2`, `theme_3`: Top themes
    - `theme_1_score`, `theme_2_score`, `theme_3_score`: Theme confidence scores

    **Custom Theme Labels:**
    If not provided, uses defaults: politics, economics, social, technology, health, environment, education, culture, sports, entertainment

    **Supported Presets:**
    - Sentiment: sentiment-twitter, sentiment-sst2
    - Themes: zeroshot-bart, zeroshot-mdeberta
    """
    logger.info(f"Batch Excel processing started: {file.filename}")
    logger.info(f"Config: sentiment={extract_sentiment}, themes={extract_themes}, custom_labels={theme_labels is not None}")

    try:
        # Validate file type
        if not file.filename or not file.filename.lower().endswith(('.xlsx', '.xlsm', '.xls')):
            raise HTTPException(
                status_code=400,
                detail="File must be an Excel file (.xlsx, .xlsm, or .xls)"
            )

        # Read file bytes
        file_bytes = await file.read()

        # Parse optional parameters
        parsed_theme_labels = None
        if theme_labels:
            parsed_theme_labels = [label.strip() for label in theme_labels.split(',') if label.strip()]
            logger.info(f"Using custom theme labels: {parsed_theme_labels}")

        parsed_sheets = None
        if sheets_to_process:
            parsed_sheets = [sheet.strip() for sheet in sheets_to_process.split(',') if sheet.strip()]
            logger.info(f"Processing specific sheets: {parsed_sheets}")

        # Create configuration
        config = BatchProcessingConfig(
            extract_sentiment=extract_sentiment,
            extract_themes=extract_themes,
            theme_labels=parsed_theme_labels,
            sentiment_preset=sentiment_preset,
            theme_preset=theme_preset,
            text_column=text_column,
            sheets_to_process=parsed_sheets,
            add_confidence_scores=add_confidence_scores,
            top_n_themes=top_n_themes,
            batch_size=batch_size
        )

        # Process Excel file
        output_bytes, stats = process_excel_file(file_bytes, config)

        logger.info(f"Batch processing complete: {stats['total_sheets']} sheets processed")

        # Return enhanced Excel file
        output_filename = file.filename.replace('.xlsx', '_analyzed.xlsx').replace('.xls', '_analyzed.xlsx')

        return StreamingResponse(
            io.BytesIO(output_bytes),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "Content-Disposition": f'attachment; filename="{output_filename}"',
                "X-Processing-Stats": json.dumps(stats)
            }
        )

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Batch processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")


@app.post("/batch/excel-from-url")
async def batch_process_excel_from_url(
    file_url: str = Query(..., description="URL of Excel file to download and process"),
    extract_sentiment: bool = Query(True, description="Extract sentiment from texts"),
    extract_themes: bool = Query(True, description="Extract themes using zero-shot classification"),
    theme_labels: Optional[str] = Query(None, description="Comma-separated theme labels (optional, uses defaults if not provided)"),
    sentiment_preset: str = Query("sentiment-twitter", description="Sentiment analysis preset"),
    theme_preset: str = Query("zeroshot-bart", description="Zero-shot classification preset"),
    text_column: Optional[str] = Query(None, description="Text column name (auto-detect if not provided)"),
    sheets_to_process: Optional[str] = Query(None, description="Comma-separated sheet names (process all if not provided)"),
    top_n_themes: int = Query(3, description="Number of top themes to extract", ge=1, le=10),
    add_confidence_scores: bool = Query(True, description="Include confidence scores in output"),
    batch_size: int = Query(32, description="Batch size for processing", ge=1, le=128),
):
    """
    Download Excel file from URL and batch process for sentiment and theme extraction

    **Features:**
    - Downloads Excel file from any publicly accessible URL
    - Processes all sheets (or specify which ones)
    - Extracts sentiment and/or themes (zero-shot classification)
    - Optional custom labels for theme detection
    - Auto-detects text column or specify manually
    - Preserves original data and adds result columns
    - Returns enhanced Excel file

    **Example:**
    ```bash
    curl -X POST "http://localhost:8001/batch/excel-from-url?file_url=https://example.com/data.xlsx&theme_labels=UNITAS%20exercises,military%20cooperation"
    ```

    **URL Requirements:**
    - Must be publicly accessible (no authentication required)
    - Must be a direct link to Excel file (.xlsx, .xlsm, .xls)
    - File size limit: 100MB

    **Output:**
    Same as /batch/excel - enhanced Excel file with sentiment and theme columns
    """
    logger.info(f"Batch Excel processing from URL started: {file_url}")
    logger.info(f"Config: sentiment={extract_sentiment}, themes={extract_themes}, custom_labels={theme_labels is not None}")

    try:
        # Fetch file from URL
        logger.info(f"Downloading file from URL: {file_url}")

        # Use fetch_url from url_fetch module
        file_bytes, content_type = await fetch_url(file_url)

        # Validate it's an Excel file
        if not file_url.lower().endswith(('.xlsx', '.xlsm', '.xls')):
            logger.warning(f"URL may not be an Excel file: {file_url}")

        if len(file_bytes) > 100 * 1024 * 1024:  # 100MB limit
            raise HTTPException(status_code=400, detail="File too large (>100MB)")

        logger.info(f"Downloaded {len(file_bytes)} bytes")

        # Parse optional parameters
        parsed_theme_labels = None
        if theme_labels:
            parsed_theme_labels = [label.strip() for label in theme_labels.split(',') if label.strip()]
            logger.info(f"Using custom theme labels: {parsed_theme_labels}")

        parsed_sheets = None
        if sheets_to_process:
            parsed_sheets = [sheet.strip() for sheet in sheets_to_process.split(',') if sheet.strip()]
            logger.info(f"Processing specific sheets: {parsed_sheets}")

        # Create configuration
        config = BatchProcessingConfig(
            extract_sentiment=extract_sentiment,
            extract_themes=extract_themes,
            theme_labels=parsed_theme_labels,
            sentiment_preset=sentiment_preset,
            theme_preset=theme_preset,
            text_column=text_column,
            sheets_to_process=parsed_sheets,
            add_confidence_scores=add_confidence_scores,
            top_n_themes=top_n_themes,
            batch_size=batch_size
        )

        # Process Excel file
        output_bytes, stats = process_excel_file(file_bytes, config)

        logger.info(f"Batch processing complete: {stats['total_sheets']} sheets processed")

        # Generate output filename from URL
        from pathlib import Path
        from urllib.parse import urlparse
        url_path = urlparse(file_url).path
        original_filename = Path(url_path).name or "downloaded_file.xlsx"
        output_filename = original_filename.replace('.xlsx', '_analyzed.xlsx').replace('.xls', '_analyzed.xlsx')

        return StreamingResponse(
            io.BytesIO(output_bytes),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "Content-Disposition": f'attachment; filename="{output_filename}"',
                "X-Processing-Stats": json.dumps(stats),
                "X-Source-URL": file_url
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch processing from URL failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")


if __name__ == "__main__":
    port = int(os.getenv("SERVICE_PORT", 8001))

    logger.info("=" * 70)
    logger.info("CiceroWatch NLP Service")
    logger.info("=" * 70)
    logger.info(f"Listening on: http://0.0.0.0:{port}")
    logger.info("Features: Sentiment, Entities, Topics, URL Analysis, Stance Detection")
    logger.info("Batch Excel Processing: /batch/excel (multi-tab with optional zero-shot labels)")
    logger.info("Batch processing enabled: CLASSIFY_BATCH_SIZE=16, ZEROSHOT_BATCH_SIZE=8, STANCE_BATCH_SIZE=32")
    logger.info("Health checks: every 10 minutes")
    logger.info("=" * 70)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
