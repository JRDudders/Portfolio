"""
NLP Microservice

Handles all NLP-related tasks:
- Sentiment analysis
- Entity extraction
- Topic modeling
- Text classification
- URL scraping and analysis
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Query, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel, HttpUrl
from typing import List, Optional, Dict, Any
import uvicorn
import os
import io
import json
import asyncio
import random
import pandas as pd
from bs4 import BeautifulSoup
import trafilatura
import logging
from datetime import datetime
from fetch import fetch_url_bytes_sync, fetch_url_bytes_rendered, ensure_browser

# Path to default guidance config (mounted in Docker, or local)
GUIDANCE_DEFAULTS_PATH = os.getenv("GUIDANCE_DEFAULTS_PATH", "/app/guidance_defaults.json")

def load_guidance_defaults():
    """Load default hypothesis and themes from guidance_defaults.json"""
    defaults = {
        "hypothesis": "U.S. strategic priorities and interests are viewed favorably in this content.",
        "themes": []
    }

    # Try multiple paths (Docker mount, local dev)
    paths_to_try = [
        GUIDANCE_DEFAULTS_PATH,
        "./guidance_defaults.json",
        os.path.join(os.path.dirname(__file__), "guidance_defaults.json")
    ]

    for path in paths_to_try:
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                    if "hypothesis" in loaded:
                        defaults["hypothesis"] = loaded["hypothesis"]
                    if "themes" in loaded:
                        defaults["themes"] = loaded["themes"]
                    print(f"[nlp] Loaded guidance defaults from {path}: {len(defaults['themes'])} themes")
                    return defaults
            except Exception as e:
                print(f"[nlp] Warning: Failed to load {path}: {e}")

    print("[nlp] No guidance_defaults.json found, using built-in defaults")
    return defaults

# Load defaults at startup
_guidance_defaults = load_guidance_defaults()

# Configure logging with local time
import time
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
# Use local time instead of UTC
logging.Formatter.converter = time.localtime
logger = logging.getLogger(__name__)

# Import URL fetch utility
from url_fetch import fetch_url, guess_file_extension, get_wayback_snapshot

# Import NLP processing modules
from nlp_processor import (
    analyze_sentiment,
    extract_entities,
    analyze_topics
)
from nlp import run_task, PRESETS, DEFAULT_ZS_LABELS, preprocess_for_task, generate_narratives_batch, translate_batch
from adapters import process_url
from excel_utils import (
    preview_excel_structure,
    extract_texts_from_excel
)
from file_store import FileStore
from training_store import (
    get_training_store,
    parse_excel_training_data,
    TrainingExample
)

app = FastAPI(
    title="CiceroWatch NLP Service",
    description="Natural Language Processing microservice",
    version="1.0.0"
)

# Initialize file store
file_store = FileStore()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- File Persistence (Optional "Save for Later") ------------------------- #

@app.post("/files/save")
async def save_file_for_later(
    file: UploadFile = File(...),
    retention_days: int = Form(..., le=90, ge=1, description="Days to retain file (max 90)")
):
    """
    Save file to persistent storage with expiration date.
    Only called when user checks "Save for later use".
    """
    try:
        file_bytes = await file.read()
        file_id = file_store.save_file(file_bytes, file.filename, retention_days)
        metadata = file_store.get_metadata(file_id)
        
        return {
            "file_id": file_id,
            "filename": file.filename,
            "upload_date": metadata["upload_date"],
            "expiry_date": metadata["expiry_date"],
            "retention_days": retention_days,
            "size_bytes": metadata["size_bytes"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")


@app.get("/files/list")
async def list_saved_files():
    """List all active (non-expired) saved files"""
    try:
        files = file_store.list_active_files()
        stats = file_store.get_stats()
        
        return {
            "files": files,
            "stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list files: {str(e)}")


@app.get("/files/{file_id}")
async def get_saved_file(file_id: str):
    """Retrieve a saved file by ID"""
    try:
        file_bytes = file_store.get_file(file_id)
        metadata = file_store.get_metadata(file_id)
        
        return Response(
            content=file_bytes,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f'attachment; filename="{metadata["original_filename"]}"'
            }
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found or expired")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve file: {str(e)}")


@app.get("/files/{file_id}/metadata")
async def get_file_metadata(file_id: str):
    """Retrieve metadata for a saved file"""
    try:
        return file_store.get_metadata(file_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found or expired")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metadata: {str(e)}")


@app.delete("/files/{file_id}")
async def delete_saved_file(file_id: str):
    """Delete a saved file before its expiry"""
    try:
        if file_store.delete_file(file_id):
            return {"status": "success", "message": "File deleted"}
        else:
            raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")


@app.get("/files/stats")
async def get_storage_stats():
    """Get overall storage statistics"""
    return file_store.get_stats()


# ============================================================
# Training Data Management (Few-Shot Learning)
# ============================================================

@app.post("/training-data/upload")
async def upload_training_data(
    file: UploadFile = File(...),
    text_column: Optional[str] = Query(None, description="Column containing text (auto-detected if not specified)"),
    stance_column: Optional[str] = Query(None, description="Column containing stance labels"),
    themes_column: Optional[str] = Query(None, description="Column containing theme labels"),
    narrative_column: Optional[str] = Query(None, description="Column containing narratives"),
    append: bool = Query(True, description="Append to existing data (False = replace all)"),
    generate_embeddings: bool = Query(True, description="Generate embeddings after upload"),
):
    """
    Upload a human-labeled Excel file to use as training data for few-shot learning.

    The system will:
    1. Parse the Excel file and extract labeled examples
    2. Generate embeddings for similarity-based retrieval
    3. Use these examples as few-shot context when processing new files

    Expected columns (auto-detected, case-insensitive):
    - Text: Body, Text, Content, Post, Tweet, Message
    - Stance: Stance (SUPPORT/OPPOSE/NEUTRAL)
    - Themes: Themes (comma-separated)
    - Narrative: Narrative, Summary

    Returns statistics about the uploaded training data.
    """
    logger.info(f"Uploading training data: {file.filename}")

    # Validate file type
    filename = file.filename or "upload.xlsx"
    if not filename.lower().endswith(('.xlsx', '.xlsm', '.xls')):
        raise HTTPException(status_code=400, detail="File must be an Excel file (.xlsx, .xlsm, .xls)")

    # Read file content
    content = await file.read()

    # Parse training examples
    examples = parse_excel_training_data(
        excel_bytes=content,
        text_column=text_column or "Body",
        stance_column=stance_column or "Stance",
        themes_column=themes_column or "Themes",
        narrative_column=narrative_column or "Narrative",
        source_filename=filename
    )

    if not examples:
        raise HTTPException(status_code=400, detail="No valid training examples found in file. Ensure there are rows with text and at least one label (stance, themes, or narrative).")

    # Get training store
    store = get_training_store()

    # Clear existing data if not appending
    if not append:
        store.clear()
        logger.info("Cleared existing training data")

    # Add examples
    added = store.add_examples(examples)
    logger.info(f"Added {added} training examples")

    # Generate embeddings
    if generate_embeddings:
        logger.info("Generating embeddings...")
        store.generate_embeddings()

    # Save to disk
    store._save()

    # Return stats
    stats = store.get_stats()
    stats["examples_added"] = added
    stats["source_file"] = filename

    return stats


@app.get("/training-data/stats")
async def get_training_data_stats():
    """Get statistics about the current training data"""
    store = get_training_store()
    return store.get_stats()


@app.delete("/training-data/clear")
async def clear_training_data():
    """Clear all training data"""
    store = get_training_store()
    store.clear()
    return {"status": "success", "message": "All training data cleared"}


@app.post("/training-data/generate-embeddings")
async def regenerate_embeddings():
    """Regenerate embeddings for all training examples"""
    store = get_training_store()
    count = store.generate_embeddings()
    return {"status": "success", "examples_processed": count}


@app.get("/training-data/similar")
async def find_similar_examples(
    text: str = Query(..., description="Text to find similar examples for"),
    task: Optional[str] = Query(None, description="Filter by task type: stance, themes, narrative"),
    top_k: int = Query(5, description="Number of similar examples to return"),
):
    """
    Find training examples most similar to the given text.

    Useful for debugging and understanding what few-shot examples will be used.
    """
    store = get_training_store()
    similar = store.find_similar(text, top_k=top_k, task=task)

    results = []
    for example, score in similar:
        results.append({
            "similarity": round(score, 4),
            "text": example.text[:500] + "..." if len(example.text) > 500 else example.text,
            "stance": example.stance,
            "themes": example.themes,
            "narrative": example.narrative[:200] + "..." if example.narrative and len(example.narrative) > 200 else example.narrative,
            "source": example.source_file
        })

    return {
        "query": text[:200] + "..." if len(text) > 200 else text,
        "task_filter": task,
        "results": results
    }


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
    # Try UTF-8 first (standard), then common alternatives
    for encoding in ['utf-8', 'utf-8-sig', 'cp1252', 'latin-1']:
        try:
            text = b.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        # Last resort: UTF-8 with replacement for invalid chars
        text = b.decode('utf-8', errors='replace')

    data = json.loads(text)
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
    # Try UTF-8 first, then fall back to common encodings (Windows, Latin-1)
    for encoding in ['utf-8', 'utf-8-sig', 'cp1252', 'latin-1']:
        try:
            df = pd.read_csv(io.BytesIO(b), encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        # Last resort: UTF-8 with replacement for invalid chars
        df = pd.read_csv(io.BytesIO(b), encoding='utf-8', errors='replace')

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


def _texts_from_excel_bytes(b: bytes, sheet_names: Optional[str] = None, text_column: Optional[str] = None) -> List[str]:
    """Extract texts from Excel bytes

    Args:
        b: Excel file bytes
        sheet_names: Comma-separated sheet names to process, or None for all sheets
        text_column: Column name containing text, or None to auto-detect

    Returns:
        List of text strings from all specified sheets
    """
    from excel_utils import detect_text_column

    excel_file = pd.ExcelFile(io.BytesIO(b))

    # Determine which sheets to process
    if sheet_names:
        # Parse comma-separated sheet names
        requested_sheets = [s.strip() for s in sheet_names.split(',') if s.strip()]
        sheets_to_process = []
        for sheet in requested_sheets:
            if sheet not in excel_file.sheet_names:
                raise ValueError(f"Sheet '{sheet}' not found. Available sheets: {', '.join(excel_file.sheet_names)}")
            sheets_to_process.append(sheet)
    else:
        # Process all sheets
        sheets_to_process = excel_file.sheet_names

    logger.info(f"Processing {len(sheets_to_process)} sheet(s): {sheets_to_process}")

    all_texts = []

    for sheet in sheets_to_process:
        df = pd.read_excel(excel_file, sheet_name=sheet)

        if df.empty:
            logger.info(f"Sheet '{sheet}' is empty, skipping")
            continue

        # Detect or validate text column for this sheet
        if text_column:
            if text_column not in df.columns:
                logger.warning(f"Column '{text_column}' not found in sheet '{sheet}', skipping. Available: {list(df.columns)}")
                continue
            col = text_column
        else:
            try:
                col = detect_text_column(df)
                logger.info(f"Sheet '{sheet}': auto-detected text column '{col}'")
            except ValueError:
                logger.warning(f"Sheet '{sheet}': no suitable text column found, skipping")
                continue

        # Extract texts from this sheet
        texts = df[col].dropna().astype(str).tolist()
        logger.info(f"Sheet '{sheet}': extracted {len(texts)} texts from column '{col}'")
        all_texts.extend(texts)

    if not all_texts:
        raise ValueError(f"No text data found in any of the processed sheets")

    logger.info(f"Total: extracted {len(all_texts)} texts from {len(sheets_to_process)} sheet(s)")
    return all_texts


def _process_excel_with_predictions(
    b: bytes,
    predictions: List[Dict],
    sheet_names: Optional[str] = None,
    text_column: Optional[str] = None,
    preset: Optional[str] = None
) -> bytes:
    """Process Excel file and add prediction columns, returning annotated Excel bytes

    Args:
        b: Original Excel file bytes
        predictions: List of prediction dicts from run_task
        sheet_names: Comma-separated sheet names that were processed
        text_column: Column name containing text
        preset: Preset used for analysis (for column naming)

    Returns:
        Annotated Excel file bytes with prediction columns added
    """
    from excel_utils import detect_text_column

    excel_file = pd.ExcelFile(io.BytesIO(b))

    # Determine which sheets to process
    if sheet_names:
        requested_sheets = [s.strip() for s in sheet_names.split(',') if s.strip()]
        sheets_to_process = [s for s in requested_sheets if s in excel_file.sheet_names]
    else:
        sheets_to_process = list(excel_file.sheet_names)

    # Track prediction index across sheets
    pred_idx = 0
    output_sheets = {}

    def flatten_prediction(pred: Dict, prefix: str = "") -> Dict[str, any]:
        """Flatten nested prediction dict into separate columns"""
        flat = {}
        for key, value in pred.items():
            col_name = f"{prefix}{key}" if prefix else key
            if isinstance(value, dict):
                # Nested dict - flatten with key as prefix
                for sub_key, sub_value in value.items():
                    flat[f"{col_name}_{sub_key}"] = sub_value
            elif isinstance(value, list):
                # List - join as string or take first
                if value and isinstance(value[0], dict):
                    # List of dicts (like top themes)
                    flat[col_name] = ", ".join(str(v.get('label', v)) for v in value[:5])
                else:
                    flat[col_name] = ", ".join(str(v) for v in value[:5])
            else:
                flat[col_name] = value
        return flat

    for sheet in excel_file.sheet_names:
        df = pd.read_excel(excel_file, sheet_name=sheet)

        if sheet in sheets_to_process and not df.empty:
            # Detect text column for this sheet
            if text_column and text_column in df.columns:
                col = text_column
            else:
                try:
                    col = detect_text_column(df)
                except ValueError:
                    # No text column found, keep original
                    output_sheets[sheet] = df
                    continue

            # Get indices of non-null text rows
            valid_mask = df[col].notna()
            valid_indices = df.index[valid_mask].tolist()

            # Add prediction columns for valid rows
            num_valid = len(valid_indices)
            sheet_predictions = predictions[pred_idx:pred_idx + num_valid]
            pred_idx += num_valid

            # Flatten all predictions and collect column names
            if sheet_predictions:
                # Flatten first prediction to get column structure
                flat_sample = flatten_prediction(sheet_predictions[0])

                # Initialize new columns with None
                for col_name in flat_sample.keys():
                    if col_name not in df.columns:
                        df[col_name] = None

                # Fill in predictions for valid rows
                for i, idx in enumerate(valid_indices):
                    if i < len(sheet_predictions):
                        flat_pred = flatten_prediction(sheet_predictions[i])
                        for col_name, value in flat_pred.items():
                            df.at[idx, col_name] = value

        output_sheets[sheet] = df

    # Write to Excel bytes
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for sheet_name, df in output_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    output.seek(0)
    return output.read()


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


@app.post("/preview-excel")
async def preview_excel(file: UploadFile = File(...)):
    """
    Preview Excel file structure before processing.

    Returns sheet names, columns, row counts, and detected text columns.
    """
    try:
        logger.info(f"Previewing Excel file: {file.filename}")
        result = await preview_excel_structure(file)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error previewing Excel file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
    text_column: Optional[str] = Query(None, description="Column name containing text (auto-detect if not specified)"),
    sheets: Optional[str] = Query(None, description="Comma-separated sheet names to process (all sheets if not specified)"),
):
    """
    Analyze text file and return annotated results as downloadable file

    Supports:
    - JSON: list of strings or objects with 'text' field
    - CSV: file with 'text' column or first string column
    - Excel (.xlsx, .xlsm, .xls): auto-detects text column
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
        is_excel = False
        if name.endswith(".json"):
            texts = _texts_from_json_bytes(b)
        elif name.endswith(".csv"):
            texts = _texts_from_csv_bytes(b)
        elif name.endswith((".xlsx", ".xlsm", ".xls")):
            # Excel file - extract text from specified or auto-detected column
            is_excel = True
            texts = _texts_from_excel_bytes(b, sheet_names=sheets, text_column=text_column)
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
            # Try JSON first, then CSV (but not Excel since it's binary)
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

        # For Excel files, return annotated Excel with predictions added as columns
        if is_excel:
            # predictions is a list of dicts, add them to original Excel
            excel_bytes = _process_excel_with_predictions(
                b,
                predictions,
                sheet_names=sheets,
                text_column=text_column,
                preset=preset
            )
            output_filename = original_filename.rsplit('.', 1)[0] + '_analyzed.xlsx'
            return _make_download(
                output_filename,
                excel_bytes,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        # For other file types, return JSON
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


@app.post("/batch-excel")
async def batch_excel(
    file: UploadFile = File(...),
    stance_preset: Optional[str] = Query("stance-deberta", description="Stance detection preset"),
    theme_preset: Optional[str] = Query("zeroshot-bart", description="Theme/zero-shot preset"),
    labels: Optional[str] = Query(None, description="Comma-separated theme labels"),
    hypothesis: Optional[str] = Query(None, description="Custom hypothesis for stance detection (auto-generated from guidance if not provided)"),
    text_column: Optional[str] = Query(None, description="Column name containing text"),
    sheets: Optional[str] = Query(None, description="Comma-separated sheet names to process"),
    extract_stance: bool = Query(True, description="Extract stance relative to hypothesis"),
    extract_themes: bool = Query(True, description="Extract themes"),
    top_themes: int = Query(3, description="Number of top themes to return"),
    generate_narrative: bool = Query(True, description="Generate narrative explaining theme relevance"),
    use_training_data: bool = Query(True, description="Use uploaded training data for few-shot learning"),
):
    """
    Batch process Excel file with stance detection AND theme extraction.

    If training data has been uploaded via /training-data/upload, it will be used for:
    - Theme labels (extracted from training examples)
    - Few-shot narrative generation (similar examples inform the LLM)

    Returns Excel file with columns:
    - Stance (SUPPORT/OPPOSE/NEUTRAL) - relative to guidance hypothesis
    - Stance_Confidence (0-1 score)
    - Themes (top N predicted themes)
    - Themes_Confidence (0-1 scores)
    - Narrative (LLM-generated explanation)
    """
    import time
    start_time = time.time()

    logger.info(f"Batch Excel processing: {file.filename}, stance={extract_stance}, themes={extract_themes}")

    # Helper to save checkpoint files
    def save_checkpoint(excel_bytes: bytes, stage: str, filename: str, is_partial: bool = False):
        """Save checkpoint file to temp directory.

        Args:
            excel_bytes: The Excel file bytes to save
            stage: Stage name (e.g., '3_stance', '4_themes')
            filename: Original filename
            is_partial: If True, overwrites a single 'working' file instead of creating stage-specific files
        """
        import os
        from pathlib import Path
        checkpoint_dir = Path("/app/temp/checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        base_name = Path(filename).stem

        if is_partial:
            # Use single working file that gets overwritten
            checkpoint_path = checkpoint_dir / f"{base_name}_checkpoint_working.xlsx"
        else:
            # Stage complete - save with stage name
            checkpoint_path = checkpoint_dir / f"{base_name}_checkpoint_{stage}.xlsx"

        with open(checkpoint_path, 'wb') as f:
            f.write(excel_bytes)
        logger.info(f"CHECKPOINT SAVED: {checkpoint_path} ({len(excel_bytes):,} bytes)")
        return str(checkpoint_path)

    try:
        b = await file.read()
        original_filename = file.filename or "input.xlsx"
        name = original_filename.lower()

        if not name.endswith((".xlsx", ".xlsm", ".xls")):
            raise HTTPException(status_code=400, detail="File must be an Excel file (.xlsx, .xlsm, .xls)")

        # Check for Guidance sheet/column to extract theme labels
        guidance_labels = None
        guidance_hypothesis = None  # May be extracted from Guidance sheet
        excel_file = pd.ExcelFile(io.BytesIO(b))

        # Look for Guidance tab (case-insensitive)
        guidance_sheet = None
        for sheet in excel_file.sheet_names:
            if sheet.lower() == 'guidance':
                guidance_sheet = sheet
                break

        if guidance_sheet:
            # Read without headers to handle various formats (numbered lists, no headers, etc.)
            guidance_df = pd.read_excel(excel_file, sheet_name=guidance_sheet, header=None)
            logger.info(f"Guidance sheet '{guidance_sheet}' has {len(guidance_df)} rows, {len(guidance_df.columns)} columns")

            # Debug: show first few rows of each column
            for col_idx in range(min(len(guidance_df.columns), 5)):
                col_data = guidance_df.iloc[:, col_idx].dropna().tolist()[:10]
                logger.info(f"  Column {col_idx} sample: {col_data}")

            all_values = []  # Use list to preserve order
            seen = set()  # Track duplicates

            # Helper to strip leading numbers from text like "1. Theme" or "1) Theme" or "1 - Theme"
            import re
            def strip_leading_number(text):
                """Remove leading number patterns like '1. ', '1) ', '1 - ', '1: ' from text"""
                text = str(text).strip()
                # Pattern: optional number(s), then separator (. ) - : etc), then the actual text
                match = re.match(r'^\d+[\.\)\-:\s]+\s*(.+)$', text)
                if match:
                    return match.group(1).strip()
                return text

            def is_integer(val):
                """Check if value is an integer (1, 2, 3, not 1.5)"""
                if pd.isna(val):
                    return False, None
                try:
                    num = int(float(val))
                    if float(val) == num:  # Make sure it's not a decimal
                        return True, num
                except (ValueError, TypeError):
                    pass
                return False, None

            # STRATEGY: Find where numbered list (1, 2, 3...) starts by scanning cells
            # Look for a cell with "1" followed by "2" in the next row of the same column
            ranking_col = None
            ranking_start_row = None
            label_col = None

            for col_idx in range(len(guidance_df.columns)):
                for row_idx in range(len(guidance_df) - 1):  # -1 because we check row+1
                    val = guidance_df.iloc[row_idx, col_idx]
                    is_int, num = is_integer(val)
                    if is_int and num == 1:
                        # Check if next row has 2
                        next_val = guidance_df.iloc[row_idx + 1, col_idx]
                        is_next_int, next_num = is_integer(next_val)
                        if is_next_int and next_num == 2:
                            # Found the start of a numbered list!
                            ranking_col = col_idx
                            ranking_start_row = row_idx
                            if col_idx + 1 < len(guidance_df.columns):
                                label_col = col_idx + 1
                            logger.info(f"Found numbered list starting at row {row_idx}, col {col_idx} (1, 2, ...). Using col {label_col} for labels.")
                            break
                if ranking_col is not None:
                    break

            # If we found a numbered list, extract labels from adjacent column
            if ranking_col is not None and label_col is not None:
                # Follow the numbered sequence and extract corresponding labels
                current_row = ranking_start_row
                expected_num = 1
                while current_row < len(guidance_df):
                    val = guidance_df.iloc[current_row, ranking_col]
                    is_int, num = is_integer(val)

                    if is_int and num == expected_num:
                        # Get the label from the adjacent column
                        label_val = guidance_df.iloc[current_row, label_col]
                        if pd.notna(label_val):
                            label_str = str(label_val).strip()
                            label_str = strip_leading_number(label_str)  # In case there's also a number prefix
                            if label_str and len(label_str) > 1 and label_str not in seen:
                                all_values.append(label_str)
                                seen.add(label_str)
                                logger.info(f"  Priority {expected_num}: {label_str}")
                        expected_num += 1
                        current_row += 1
                    else:
                        # Sequence broken, stop
                        break
            else:
                # Fallback: No numbered list found, try to find text column
                logger.info("No numbered list found, falling back to text column detection")
                label_col = None
                for col_idx in range(len(guidance_df.columns)):
                    col_data = guidance_df.iloc[:, col_idx].dropna()
                    text_count = sum(1 for v in col_data if not str(v).strip().replace('.', '').isdigit() and len(str(v).strip()) > 2)
                    if text_count >= 3:
                        label_col = col_idx
                        logger.info(f"Using column {col_idx} for labels (text content detected)")
                        break

                if label_col is not None:
                    for val in guidance_df.iloc[:, label_col].dropna():
                        val_str = str(val).strip()
                        val_str = strip_leading_number(val_str)
                        val_lower = val_str.lower()
                        if any(skip in val_lower for skip in ['hierarchy', 'informal instruction', 'theme:', 'priority', 'rank']):
                            continue
                        if val_str.replace('.', '').isdigit():
                            continue
                        if not val_str or len(val_str) <= 1:
                            continue
                        if val_str not in seen:
                            all_values.append(val_str)
                            seen.add(val_str)

            guidance_labels = all_values
            logger.info(f"Extracted {len(guidance_labels)} theme labels from Guidance sheet (ordered): {guidance_labels}")

            # Also try to extract hypothesis from Guidance sheet
            # Look for cells containing "hypothesis", "claim", or "research question"
            guidance_hypothesis = None
            for col_idx in range(len(guidance_df.columns)):
                for row_idx in range(len(guidance_df)):
                    cell_val = guidance_df.iloc[row_idx, col_idx]
                    if pd.isna(cell_val):
                        continue
                    cell_str = str(cell_val).strip().lower()
                    # Check if this cell is a label for hypothesis
                    if any(kw in cell_str for kw in ['hypothesis:', 'claim:', 'research question:', 'stance question:']):
                        # The hypothesis might be in the same cell after the colon, or in the next column
                        if ':' in cell_str:
                            parts = str(cell_val).split(':', 1)
                            if len(parts) > 1 and len(parts[1].strip()) > 10:
                                guidance_hypothesis = parts[1].strip()
                                logger.info(f"Extracted hypothesis from Guidance (same cell): {guidance_hypothesis}")
                                break
                        # Or check next column
                        if col_idx + 1 < len(guidance_df.columns):
                            next_val = guidance_df.iloc[row_idx, col_idx + 1]
                            if pd.notna(next_val) and len(str(next_val).strip()) > 10:
                                guidance_hypothesis = str(next_val).strip()
                                logger.info(f"Extracted hypothesis from Guidance (adjacent cell): {guidance_hypothesis}")
                                break
                if guidance_hypothesis:
                    break

        # If no guidance sheet, try to get labels from training data or defaults
        training_store_instance = None
        if use_training_data:
            training_store_instance = get_training_store()
            stats = training_store_instance.get_stats()

            if stats["total_examples"] > 0:
                logger.info(f"Training data available: {stats['total_examples']} examples")

                # Use training data labels if no guidance labels
                if not guidance_labels and stats["unique_themes"]:
                    guidance_labels = stats["unique_themes"]
                    logger.info(f"Using {len(guidance_labels)} theme labels from training data: {guidance_labels}")
            else:
                logger.info("No training data uploaded")
                training_store_instance = None  # Don't use if empty

        # If still no labels, use defaults from guidance_defaults.json
        if not guidance_labels and _guidance_defaults["themes"]:
            guidance_labels = _guidance_defaults["themes"]
            logger.info(f"Using {len(guidance_labels)} theme labels from guidance_defaults.json: {guidance_labels}")

        # Generate hypothesis from guidance if not provided
        generated_hypothesis = hypothesis
        if not generated_hypothesis:
            # First try hypothesis extracted from Guidance sheet
            if guidance_hypothesis:
                generated_hypothesis = guidance_hypothesis
                logger.info(f"Using hypothesis from Guidance sheet: {generated_hypothesis}")
            elif guidance_labels:
                # Create hypothesis from guidance themes
                themes_summary = ", ".join(guidance_labels[:5])
                if len(guidance_labels) > 5:
                    themes_summary += f" (and {len(guidance_labels) - 5} more)"
                generated_hypothesis = f"U.S. priorities as outlined in the guidance ({themes_summary}) are viewed favorably and having a positive influence."
                logger.info(f"Generated hypothesis from themes: {generated_hypothesis}")
            else:
                # Use default hypothesis from guidance_defaults.json
                generated_hypothesis = _guidance_defaults["hypothesis"]
                logger.info(f"Using hypothesis from guidance_defaults.json: {generated_hypothesis}")

        # Default text_column to "Body" if not specified
        if not text_column:
            text_column = "Body"

        # Helper to check if sheet should be excluded
        def _should_exclude_sheet(sheet_name):
            name_lower = sheet_name.lower()
            if guidance_sheet and sheet_name == guidance_sheet:
                return True
            if 'pivot' in name_lower:
                return True
            return False

        # Extract texts from Excel (excluding Guidance and Pivot sheets)
        process_sheets = sheets
        if not process_sheets:
            # Exclude Guidance and Pivot sheets from processing if no sheets specified
            process_sheets = ','.join([s for s in excel_file.sheet_names if not _should_exclude_sheet(s)])

        # Fetch URLs and populate Body column for each sheet
        sheets_to_process = [s.strip() for s in process_sheets.split(',')] if process_sheets else [s for s in excel_file.sheet_names if not _should_exclude_sheet(s)]
        # Filter out excluded sheets even if explicitly specified
        sheets_to_process = [s for s in sheets_to_process if not _should_exclude_sheet(s)]

        if sheets_to_process:
            excluded = [s for s in excel_file.sheet_names if _should_exclude_sheet(s)]
            if excluded:
                logger.info(f"Excluding sheets: {excluded}")

        # Note: Each stage now handles partial completion at the row level
        # (skipping rows that already have data, processing only empty rows)

        modified_dfs = {}
        url_fetch_count = 0

        for sheet in sheets_to_process:
            if sheet not in excel_file.sheet_names:
                continue
            df = pd.read_excel(excel_file, sheet_name=sheet)

            # Find URL column (case-insensitive)
            url_col = None
            for col in df.columns:
                if 'url' in str(col).lower():
                    url_col = col
                    break

            # Check if we should fetch URLs for this sheet
            # Handle partial completion: if Body column exists, only fetch for empty rows
            has_body_col = text_column in df.columns
            if url_col:
                if has_body_col:
                    # Body column exists - check for unfetched rows (null/empty Body with valid URL)
                    needs_fetch_mask = df[text_column].isna() | (df[text_column].astype(str).str.strip() == '')
                    needs_fetch_mask &= df[url_col].notna() & (df[url_col].astype(str).str.strip() != '')
                    unfetched_count = needs_fetch_mask.sum()
                    already_fetched = (~needs_fetch_mask & df[url_col].notna()).sum()
                    if unfetched_count > 0:
                        logger.info(f"Sheet '{sheet}': {already_fetched} rows already fetched, {unfetched_count} remaining")
                    elif already_fetched > 0:
                        logger.info(f"Sheet '{sheet}': SKIPPING URL fetch (all {already_fetched} rows already populated)")
                else:
                    # No Body column yet - need to fetch all URLs
                    needs_fetch_mask = df[url_col].notna() & (df[url_col].astype(str).str.strip() != '')
                    unfetched_count = needs_fetch_mask.sum()
                    df[text_column] = None  # Initialize Body column
            else:
                unfetched_count = 0
                needs_fetch_mask = pd.Series([False] * len(df))

            if url_col and unfetched_count > 0:
                logger.info(f"Sheet '{sheet}': Found URL column '{url_col}', fetching content for Body column")

                # Get browser for JS-rendered content
                browser = await ensure_browser(app)

                # Parallel URL fetching with semaphore to limit concurrent requests
                # Twitter/X rate limits: ~50-100 req/15min unauthenticated
                # Reduced concurrency and increased delays to avoid connection refusals
                MAX_CONCURRENT_FETCHES = 10
                MIN_DELAY_MS = 1000  # 1 second minimum
                MAX_DELAY_MS = 3000  # Randomize up to 3s to look natural
                semaphore = asyncio.Semaphore(MAX_CONCURRENT_FETCHES)

                # Track if we need emergency checkpoint save on interrupt
                url_fetch_interrupted = False

                def is_blocked_response(text: str, url: str) -> bool:
                    """Check if the response looks like a blocked/login page"""
                    if not text or len(text.strip()) < 50:
                        return True

                    text_lower = text.lower()
                    # Twitter/X specific blocked indicators
                    twitter_blocked_indicators = [
                        "sign in to x",
                        "log in to x",
                        "sign in to twitter",
                        "log in to twitter",
                        "something went wrong",
                        "try again",
                        "this page isn't available",
                        "hmm...this page doesn't exist",
                        "caution: this profile may include potentially sensitive content",
                        "age-restricted adult content",
                        "create your account",
                        "don't miss what's happening",
                        "join x today",
                        "see what's happening",
                    ]

                    # Generic blocked indicators
                    generic_blocked = [
                        "access denied",
                        "403 forbidden",
                        "please enable javascript",
                        "enable cookies",
                        "verify you are human",
                        "captcha",
                        "checking your browser",
                        "just a moment",  # Cloudflare
                        "attention required",  # Cloudflare
                    ]

                    # Check Twitter/X URLs specifically
                    is_twitter = any(d in url.lower() for d in ['twitter.com', 'x.com', 't.co'])

                    if is_twitter:
                        for indicator in twitter_blocked_indicators:
                            if indicator in text_lower:
                                return True
                        # Twitter pages should have substantial content
                        if len(text.strip()) < 200:
                            return True

                    for indicator in generic_blocked:
                        if indicator in text_lower:
                            return True

                    return False

                # List of working Nitter/Twitter mirrors (some may go down, so we try multiple)
                NITTER_INSTANCES = [
                    "twiiit.com",  # Search/redirect service - finds working mirrors
                    "xcancel.com",  # Most reliable direct mirror
                    "nitter.poast.org",
                    "nitter.privacyredirect.com",
                    "lightbrd.com",
                    "nitter.space",
                    "nitter.tiekoetter.com",
                    "nuku.trabun.org",
                    "nitter.catsarch.com",
                ]

                def get_nitter_urls(twitter_url: str) -> list[tuple[str, str]]:
                    """
                    Convert Twitter/X URL to multiple Nitter URLs for fallback attempts.

                    Returns list of (url, mirror_name) tuples.
                    All mirrors (including twiiit.com) accept the full path.
                    twiiit.com auto-redirects to a working mirror.
                    """
                    import re
                    # Match twitter.com or x.com URLs
                    match = re.match(r'https?://(www\.)?(twitter\.com|x\.com)/(.+)', twitter_url)
                    if not match:
                        return []

                    path = match.group(3)  # e.g., "username/status/123456"
                    return [(f"https://{instance}/{path}", instance) for instance in NITTER_INSTANCES]

                async def fetch_single_url(idx: int, url) -> tuple:
                    """Fetch a single URL and return (index, text)"""
                    if pd.isna(url) or not str(url).strip():
                        return (idx, '')

                    url_str = str(url).strip()
                    # Auto-fix defanged URLs (hxxps -> https, hxxp -> http, [.] -> .)
                    url_str = url_str.replace('hxxps://', 'https://').replace('hxxp://', 'http://')
                    url_str = url_str.replace('[.]', '.').replace('[dot]', '.')

                    async with semaphore:
                        try:
                            # Small random delay to avoid detection patterns
                            delay = random.randint(MIN_DELAY_MS, MAX_DELAY_MS) / 1000.0
                            await asyncio.sleep(delay)

                            content_bytes, kind = await fetch_url_bytes_rendered(
                                url_str,
                                browser,
                                timeout_ms=30000,
                                scroll_passes=2
                            )
                            html_content = content_bytes.decode('utf-8', errors='ignore')
                            text = _extract_text_from_html(html_content)

                            # Debug logging for text extraction
                            logger.debug(f"URL {url_str[:50]}... extracted {len(text)} chars")

                            # Check for blocked responses
                            if is_blocked_response(text, url_str):
                                logger.warning(f"Blocked response detected for {url_str[:80]}... (extracted {len(text)} chars)")

                                # Try Nitter/Twitter mirrors as fallback
                                is_twitter = any(d in url_str.lower() for d in ['twitter.com', 'x.com', 't.co'])
                                if is_twitter:
                                    nitter_urls = get_nitter_urls(url_str)
                                    for nitter_url, mirror_name in nitter_urls:
                                        logger.info(f"Trying mirror fallback ({mirror_name}): {nitter_url}")
                                        try:
                                            # twiiit.com auto-redirects to working mirror
                                            # Playwright follows redirects automatically
                                            nitter_bytes, _ = await fetch_url_bytes_rendered(
                                                nitter_url,
                                                browser,
                                                timeout_ms=15000,
                                                scroll_passes=1
                                            )
                                            nitter_text = _extract_text_from_html(nitter_bytes.decode('utf-8', errors='ignore'))
                                            if nitter_text and len(nitter_text.strip()) > 100 and not is_blocked_response(nitter_text, nitter_url):
                                                logger.info(f"Mirror fallback successful ({mirror_name}): {len(nitter_text)} chars")
                                                return (idx, nitter_text[:50000])
                                            else:
                                                logger.debug(f"Mirror {mirror_name} returned insufficient content")
                                        except Exception as nitter_e:
                                            logger.debug(f"Mirror {mirror_name} failed: {nitter_e}")
                                            continue  # Try next instance

                                # Mark as blocked
                                return (idx, f'[BLOCKED: {url_str}]')

                            return (idx, text[:50000])
                        except Exception as e:
                            logger.warning(f"Failed to fetch URL {url_str}: {e}")

                            # Try Nitter/Twitter mirrors as fallback for Twitter URLs (on exception too)
                            is_twitter = any(d in url_str.lower() for d in ['twitter.com', 'x.com', 't.co'])
                            if is_twitter:
                                nitter_urls = get_nitter_urls(url_str)
                                for nitter_url, mirror_name in nitter_urls:
                                    logger.info(f"Trying mirror fallback after exception ({mirror_name}): {nitter_url}")
                                    try:
                                        nitter_bytes, _ = await fetch_url_bytes_rendered(
                                            nitter_url,
                                            browser,
                                            timeout_ms=15000,
                                            scroll_passes=1
                                        )
                                        nitter_text = _extract_text_from_html(nitter_bytes.decode('utf-8', errors='ignore'))
                                        if nitter_text and len(nitter_text.strip()) > 100 and not is_blocked_response(nitter_text, nitter_url):
                                            logger.info(f"Mirror fallback successful after exception ({mirror_name}): {len(nitter_text)} chars")
                                            return (idx, nitter_text[:50000])
                                        else:
                                            logger.debug(f"Mirror {mirror_name} returned insufficient content")
                                    except Exception as nitter_e:
                                        logger.debug(f"Mirror {mirror_name} failed: {nitter_e}")
                                        continue

                            # Try Wayback Machine as fallback
                            if not url_str.startswith('https://web.archive.org/'):
                                try:
                                    wayback_url = await get_wayback_snapshot(url_str)
                                    if wayback_url:
                                        logger.info(f"Trying Wayback Machine: {wayback_url}")
                                        content_bytes, kind = await fetch_url_bytes_rendered(
                                            wayback_url,
                                            browser,
                                            timeout_ms=30000,
                                            scroll_passes=2
                                        )
                                        text = _extract_text_from_html(content_bytes.decode('utf-8', errors='ignore'))
                                        # Verify Wayback content isn't blocked either
                                        if text and len(text.strip()) > 100 and not is_blocked_response(text, wayback_url):
                                            logger.info(f"Wayback fallback successful: {len(text)} chars")
                                            return (idx, text[:50000])
                                        else:
                                            logger.warning(f"Wayback returned blocked/empty content for {url_str}")
                                except Exception as wb_e:
                                    logger.warning(f"Wayback fallback also failed for {url_str}: {wb_e}")

                            # Return failure message instead of empty string
                            return (idx, f'[FETCH FAILED: {url_str}]')

                # Create tasks only for rows that need fetching (unfetched rows)
                rows_to_fetch = df.index[needs_fetch_mask].tolist()
                tasks = [fetch_single_url(idx, df.at[idx, url_col]) for idx in rows_to_fetch]
                logger.info(f"Sheet '{sheet}': Fetching {len(tasks)} URLs in parallel (max {MAX_CONCURRENT_FETCHES} concurrent)")

                # Execute tasks and log progress as they complete
                # Save incremental checkpoints every CHECKPOINT_INTERVAL URLs
                CHECKPOINT_INTERVAL = 10
                completed = 0
                fetched_count = 0
                last_checkpoint = 0

                try:
                    for coro in asyncio.as_completed(tasks):
                        idx, text = await coro
                        # Update dataframe immediately as each URL completes
                        df.at[idx, text_column] = text

                        # Debug: log what we're actually saving with preview
                        text_preview = (text[:100] + '...') if text and len(text) > 100 else text
                        logger.info(f"Row {idx}: Saved {len(text) if text else 0} chars to '{text_column}'")
                        logger.debug(f"Row {idx} preview: {text_preview}")

                        # Verify the assignment worked by reading it back
                        saved_value = df.at[idx, text_column]
                        if saved_value != text:
                            logger.error(f"Row {idx}: MISMATCH! Assigned {len(text)} chars but DataFrame has {len(saved_value) if saved_value else 0}")

                        if text and not text.startswith('[FETCH FAILED') and not text.startswith('[BLOCKED'):
                            fetched_count += 1
                        completed += 1

                        if completed % 10 == 0 or completed == len(tasks):
                            logger.info(f"Fetched {completed}/{len(tasks)} URLs...")

                        # Save incremental checkpoint every CHECKPOINT_INTERVAL URLs
                        if completed - last_checkpoint >= CHECKPOINT_INTERVAL and completed < len(tasks):
                            modified_dfs[sheet] = df
                            # Debug: verify DataFrame has data before saving
                            non_empty = df[text_column].notna() & (df[text_column].astype(str).str.strip() != '')
                            logger.info(f"Checkpoint: DataFrame has {non_empty.sum()} non-empty '{text_column}' cells")
                            # Build checkpoint Excel
                            checkpoint_output = io.BytesIO()
                            with pd.ExcelWriter(checkpoint_output, engine='openpyxl') as writer:
                                for sn in excel_file.sheet_names:
                                    if sn in modified_dfs:
                                        modified_dfs[sn].to_excel(writer, sheet_name=sn, index=False)
                                    else:
                                        pd.read_excel(excel_file, sheet_name=sn).to_excel(writer, sheet_name=sn, index=False)
                            checkpoint_output.seek(0)
                            checkpoint_bytes = checkpoint_output.read()
                            save_checkpoint(checkpoint_bytes, "1_urls", original_filename, is_partial=True)

                            # Verify the saved Excel has the data by reading it back
                            verify_df = pd.read_excel(io.BytesIO(checkpoint_bytes), sheet_name=sheet)
                            if text_column in verify_df.columns:
                                verify_non_empty = verify_df[text_column].notna() & (verify_df[text_column].astype(str).str.strip() != '')
                                verify_non_fetch_failed = verify_non_empty & (~verify_df[text_column].astype(str).str.startswith('[FETCH FAILED'))
                                logger.info(f"Checkpoint VERIFIED: Excel has {verify_non_empty.sum()} non-empty '{text_column}' cells ({verify_non_fetch_failed.sum()} actual content)")
                                # Log a sample of actual content
                                sample_rows = verify_df[verify_non_fetch_failed].head(2)
                                for _, row in sample_rows.iterrows():
                                    sample_text = str(row[text_column])[:80]
                                    logger.info(f"Sample content: {sample_text}...")
                            else:
                                logger.error(f"Checkpoint FAILED: '{text_column}' column not found in saved Excel!")

                            last_checkpoint = completed

                except (KeyboardInterrupt, asyncio.CancelledError) as e:
                    # Save emergency checkpoint on Ctrl+C or cancellation
                    logger.warning(f"URL fetch interrupted after {completed}/{len(tasks)} URLs - saving emergency checkpoint...")
                    url_fetch_interrupted = True
                    modified_dfs[sheet] = df
                    checkpoint_output = io.BytesIO()
                    with pd.ExcelWriter(checkpoint_output, engine='openpyxl') as writer:
                        for sn in excel_file.sheet_names:
                            if sn in modified_dfs:
                                modified_dfs[sn].to_excel(writer, sheet_name=sn, index=False)
                            else:
                                pd.read_excel(excel_file, sheet_name=sn).to_excel(writer, sheet_name=sn, index=False)
                    checkpoint_output.seek(0)
                    checkpoint_bytes = checkpoint_output.read()
                    save_checkpoint(checkpoint_bytes, "1_urls_interrupted", original_filename, is_partial=True)
                    logger.info(f"Emergency checkpoint saved with {completed} URLs fetched")
                    raise  # Re-raise to stop processing

                url_fetch_count += fetched_count
                logger.info(f"Sheet '{sheet}': Populated {fetched_count} Body cells from URLs")

            modified_dfs[sheet] = df

        if url_fetch_count > 0:
            logger.info(f"Total URLs fetched: {url_fetch_count}")
            # Rebuild Excel bytes with Body column populated
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                for sheet_name in excel_file.sheet_names:
                    if sheet_name in modified_dfs:
                        modified_dfs[sheet_name].to_excel(writer, sheet_name=sheet_name, index=False)
                    else:
                        pd.read_excel(excel_file, sheet_name=sheet_name).to_excel(writer, sheet_name=sheet_name, index=False)
            output.seek(0)
            b = output.read()
            # Re-open Excel file with new data
            excel_file = pd.ExcelFile(io.BytesIO(b))

            # CHECKPOINT 1: After URL extraction
            save_checkpoint(b, "1_urls_extracted", original_filename)

        texts = _texts_from_excel_bytes(b, sheet_names=process_sheets, text_column=text_column)
        logger.info(f"Extracted {len(texts)} texts from Excel")

        # Translate texts to English and create "Body (Translated)" column
        # Handle partial completion: only translate rows with empty translations
        trans_col = "Body (Translated)"

        # Build a flat list of all texts and existing translations, tracking what needs translation
        all_translations = []  # Final list of translations (existing + new)
        texts_to_translate = []  # Texts that need translation
        indices_to_translate = []  # Indices in all_translations that need filling

        text_idx = 0
        for sheet in sheets_to_process:
            if sheet not in excel_file.sheet_names:
                continue
            df = pd.read_excel(excel_file, sheet_name=sheet)
            if text_column not in df.columns:
                continue

            # Initialize translation column if it doesn't exist
            has_trans_col = trans_col in df.columns

            for idx in df.index:
                if pd.notna(df.at[idx, text_column]):
                    body_text = str(df.at[idx, text_column])
                    existing_trans = df.at[idx, trans_col] if has_trans_col and pd.notna(df.at[idx, trans_col]) else None

                    if existing_trans and str(existing_trans).strip():
                        # Already translated - keep existing
                        all_translations.append(str(existing_trans))
                    else:
                        # Needs translation
                        all_translations.append(None)  # Placeholder
                        texts_to_translate.append(body_text)
                        indices_to_translate.append(len(all_translations) - 1)
                    text_idx += 1

        if texts_to_translate:
            logger.info(f"Translating {len(texts_to_translate)} texts ({len(all_translations) - len(texts_to_translate)} already translated)...")

            # Process translations in chunks with incremental saves
            TRANS_CHUNK_SIZE = 20
            total_translated = 0

            for chunk_start in range(0, len(texts_to_translate), TRANS_CHUNK_SIZE):
                chunk_end = min(chunk_start + TRANS_CHUNK_SIZE, len(texts_to_translate))
                chunk_texts = texts_to_translate[chunk_start:chunk_end]
                chunk_indices = indices_to_translate[chunk_start:chunk_end]

                # Translate this chunk
                chunk_translations = translate_batch(chunk_texts)

                # Fill in the translations at the correct indices
                for i, trans in enumerate(chunk_translations):
                    all_translations[chunk_indices[i]] = trans

                total_translated += len(chunk_translations)
                logger.info(f"Translated {total_translated}/{len(texts_to_translate)} texts...")

                # Save incremental checkpoint after each chunk (except the last one - that's saved below)
                if chunk_end < len(texts_to_translate):
                    # Update Excel sheets with translations so far
                    temp_trans_idx = 0
                    for sheet in sheets_to_process:
                        if sheet not in modified_dfs:
                            if sheet in excel_file.sheet_names:
                                modified_dfs[sheet] = pd.read_excel(excel_file, sheet_name=sheet)
                            else:
                                continue
                        df = modified_dfs[sheet]
                        if text_column in df.columns:
                            if trans_col not in df.columns:
                                df[trans_col] = None
                            for idx in df.index:
                                if pd.notna(df.at[idx, text_column]):
                                    if temp_trans_idx < len(all_translations) and all_translations[temp_trans_idx] is not None:
                                        df.at[idx, trans_col] = all_translations[temp_trans_idx]
                                    temp_trans_idx += 1
                            modified_dfs[sheet] = df
                    # Save checkpoint
                    checkpoint_output = io.BytesIO()
                    with pd.ExcelWriter(checkpoint_output, engine='openpyxl') as writer:
                        for sn in excel_file.sheet_names:
                            if sn in modified_dfs:
                                modified_dfs[sn].to_excel(writer, sheet_name=sn, index=False)
                            else:
                                pd.read_excel(excel_file, sheet_name=sn).to_excel(writer, sheet_name=sn, index=False)
                    checkpoint_output.seek(0)
                    save_checkpoint(checkpoint_output.read(), "2_translated", original_filename, is_partial=True)

            logger.info(f"Translated {len([t for t in all_translations if t])} texts total")

            # Update the Excel sheets with all translations
            trans_idx = 0
            for sheet in sheets_to_process:
                if sheet not in modified_dfs:
                    if sheet in excel_file.sheet_names:
                        modified_dfs[sheet] = pd.read_excel(excel_file, sheet_name=sheet)
                    else:
                        continue

                df = modified_dfs[sheet]
                if text_column in df.columns:
                    if trans_col not in df.columns:
                        df[trans_col] = None
                    for idx in df.index:
                        if pd.notna(df.at[idx, text_column]):
                            if trans_idx < len(all_translations):
                                df.at[idx, trans_col] = all_translations[trans_idx]
                                trans_idx += 1
                    modified_dfs[sheet] = df
        else:
            logger.info("SKIPPING translation (all rows already translated)")

        translated_texts = all_translations

        # Rebuild Excel bytes with translation column
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            for sheet_name in excel_file.sheet_names:
                if sheet_name in modified_dfs:
                    modified_dfs[sheet_name].to_excel(writer, sheet_name=sheet_name, index=False)
                else:
                    pd.read_excel(excel_file, sheet_name=sheet_name).to_excel(writer, sheet_name=sheet_name, index=False)
        output.seek(0)
        b = output.read()
        excel_file = pd.ExcelFile(io.BytesIO(b))

        # CHECKPOINT 2: After translation
        save_checkpoint(b, "2_translated", original_filename)

        # Use translated texts for analysis
        analysis_texts = translated_texts

        # Prepare results list - one dict per text row
        results = [{} for _ in analysis_texts]

        def _extract_confidence(val):
            """Extract numeric confidence from various formats"""
            if isinstance(val, (int, float)):
                return round(float(val), 4)
            if isinstance(val, dict):
                # Try common keys: score, confidence, value
                for key in ('score', 'confidence', 'value'):
                    if key in val and isinstance(val[key], (int, float)):
                        return round(float(val[key]), 4)
                # If dict has numeric values, return max
                nums = [v for v in val.values() if isinstance(v, (int, float))]
                if nums:
                    return round(max(nums), 4)
            return 0.0

        def _get_top_from_dict(pred_dict):
            """Get top label and confidence from a prediction dict"""
            # Filter to only items with numeric or extractable values
            scored_items = []
            for k, v in pred_dict.items():
                if k in ('label', 'score', 'confidence', 'labels', 'scores'):
                    continue  # Skip metadata keys
                conf = _extract_confidence(v)
                scored_items.append((k, conf))
            if scored_items:
                top = max(scored_items, key=lambda x: x[1])
                return top[0], top[1]
            return None, 0.0

        # Run stance detection if requested - handle partial completion
        if extract_stance:
            # Check which rows already have stance populated
            texts_needing_stance = []
            indices_needing_stance = []
            result_idx = 0

            for sheet in sheets_to_process:
                if sheet not in excel_file.sheet_names:
                    continue
                df = pd.read_excel(excel_file, sheet_name=sheet)
                if text_column not in df.columns:
                    continue

                has_stance_col = 'Stance' in df.columns

                for idx in df.index:
                    if pd.notna(df.at[idx, text_column]) and result_idx < len(results):
                        existing_stance = df.at[idx, 'Stance'] if has_stance_col and pd.notna(df.at[idx, 'Stance']) else None

                        if existing_stance and str(existing_stance).strip() and str(existing_stance).strip() != '[UNAVAILABLE]':
                            # Already has stance - load it
                            results[result_idx]['Stance'] = str(existing_stance)
                            if has_stance_col and 'Stance_Confidence' in df.columns:
                                results[result_idx]['Stance_Confidence'] = df.at[idx, 'Stance_Confidence'] if pd.notna(df.at[idx, 'Stance_Confidence']) else 0
                            if has_stance_col and 'Hypothesis' in df.columns:
                                results[result_idx]['Hypothesis'] = df.at[idx, 'Hypothesis'] if pd.notna(df.at[idx, 'Hypothesis']) else generated_hypothesis
                            else:
                                results[result_idx]['Hypothesis'] = generated_hypothesis
                        else:
                            # Needs stance detection
                            texts_needing_stance.append(analysis_texts[result_idx])
                            indices_needing_stance.append(result_idx)
                        result_idx += 1

            if texts_needing_stance:
                logger.info(f"Running stance detection on {len(texts_needing_stance)} rows ({len(results) - len(texts_needing_stance)} already have stance)")
                logger.info(f"Hypothesis: {generated_hypothesis}")

                # Process stance detection in chunks with incremental saves
                STANCE_CHUNK_SIZE = 20
                total_processed = 0

                for chunk_start in range(0, len(texts_needing_stance), STANCE_CHUNK_SIZE):
                    chunk_end = min(chunk_start + STANCE_CHUNK_SIZE, len(texts_needing_stance))
                    chunk_texts = texts_needing_stance[chunk_start:chunk_end]
                    chunk_indices = indices_needing_stance[chunk_start:chunk_end]

                    # Run stance detection on this chunk
                    stance_preds = run_task(chunk_texts, preset=stance_preset, claim=generated_hypothesis)

                    for i, pred in enumerate(stance_preds):
                        result_idx = chunk_indices[i]
                        text = chunk_texts[i]

                        # Check if source text was a failed fetch
                        if text.startswith('[FETCH FAILED'):
                            results[result_idx]['Stance'] = '[UNAVAILABLE]'
                            results[result_idx]['Stance_Confidence'] = 0
                            results[result_idx]['Hypothesis'] = generated_hypothesis
                        elif isinstance(pred, dict):
                            results[result_idx]['Stance'] = pred.get('stance', 'NEUTRAL')
                            scores = pred.get('scores', {})
                            stance_label = pred.get('stance', 'NEUTRAL')
                            confidence = scores.get(stance_label, scores.get(stance_label.lower(), 0))
                            results[result_idx]['Stance_Confidence'] = round(float(confidence), 4) if isinstance(confidence, (int, float)) else 0
                            results[result_idx]['Hypothesis'] = generated_hypothesis

                    total_processed += len(chunk_texts)
                    logger.info(f"Stance detection: {total_processed}/{len(texts_needing_stance)} processed...")

                    # Save incremental checkpoint after each chunk (except last)
                    if chunk_end < len(texts_needing_stance):
                        checkpoint_bytes = _process_excel_with_predictions(b, results, sheet_names=process_sheets, text_column=text_column, preset="stance")
                        save_checkpoint(checkpoint_bytes, "3_stance", original_filename, is_partial=True)

                # CHECKPOINT 3: After stance detection
                checkpoint_bytes = _process_excel_with_predictions(b, results, sheet_names=process_sheets, text_column=text_column, preset="stance")
                save_checkpoint(checkpoint_bytes, "3_stance", original_filename)
            else:
                logger.info("SKIPPING stance detection (all rows already have stance)")

        # Run theme/topic extraction if requested - handle partial completion
        # Set lbls first (needed for narrative generation regardless of theme extraction)
        if guidance_labels:
            lbls = guidance_labels
            logger.info(f"THEME LABELS SOURCE: Guidance sheet ({len(lbls)} labels): {lbls}")
        elif labels:
            lbls = _parse_labels_csv(labels)
            logger.info(f"THEME LABELS SOURCE: API parameter ({len(lbls)} labels): {lbls}")
        else:
            lbls = DEFAULT_ZS_LABELS
            logger.info(f"THEME LABELS SOURCE: DEFAULT_ZS_LABELS ({len(lbls)} labels): {lbls}")

        if extract_themes:
            # Check which rows already have themes populated
            texts_needing_themes = []
            indices_needing_themes = []
            result_idx = 0

            for sheet in sheets_to_process:
                if sheet not in excel_file.sheet_names:
                    continue
                df = pd.read_excel(excel_file, sheet_name=sheet)
                if text_column not in df.columns:
                    continue

                has_themes_col = 'Themes' in df.columns

                for idx in df.index:
                    if pd.notna(df.at[idx, text_column]) and result_idx < len(results):
                        existing_themes = df.at[idx, 'Themes'] if has_themes_col and pd.notna(df.at[idx, 'Themes']) else None

                        if existing_themes and str(existing_themes).strip() and str(existing_themes).strip() != '[UNAVAILABLE]':
                            # Already has themes - load them
                            results[result_idx]['Themes'] = str(existing_themes)
                            if has_themes_col and 'Themes_Confidence' in df.columns:
                                results[result_idx]['Themes_Confidence'] = df.at[idx, 'Themes_Confidence'] if pd.notna(df.at[idx, 'Themes_Confidence']) else ''
                        else:
                            # Needs theme extraction
                            texts_needing_themes.append(analysis_texts[result_idx])
                            indices_needing_themes.append(result_idx)
                        result_idx += 1

            if texts_needing_themes:
                logger.info(f"Running theme extraction on {len(texts_needing_themes)} rows ({len(results) - len(texts_needing_themes)} already have themes)")
                logger.info(f"Using {len(lbls)} theme labels, top_themes={top_themes}")

                # Process theme extraction in chunks with incremental saves
                THEME_CHUNK_SIZE = 20
                total_processed = 0

                for chunk_start in range(0, len(texts_needing_themes), THEME_CHUNK_SIZE):
                    chunk_end = min(chunk_start + THEME_CHUNK_SIZE, len(texts_needing_themes))
                    chunk_texts = texts_needing_themes[chunk_start:chunk_end]
                    chunk_indices = indices_needing_themes[chunk_start:chunk_end]

                    # Run theme extraction on this chunk
                    theme_preds = run_task(chunk_texts, preset=theme_preset, labels=lbls)

                    for i, pred in enumerate(theme_preds):
                        result_idx = chunk_indices[i]
                        text = chunk_texts[i]

                        # Check if source text was a failed fetch
                        if text.startswith('[FETCH FAILED'):
                            results[result_idx]['Themes'] = '[UNAVAILABLE]'
                            results[result_idx]['Themes_Confidence'] = '0'
                        elif isinstance(pred, dict):
                            # Unwrap "topics" key if present (run_task wraps results)
                            if 'topics' in pred:
                                pred = pred['topics']

                            if 'labels' in pred and 'scores' in pred:
                                # Zero-shot format: {'labels': [...], 'scores': [...]}
                                paired = list(zip(pred['labels'], pred['scores']))
                                paired.sort(key=lambda x: x[1], reverse=True)
                                top_n = paired[:top_themes]
                                results[result_idx]['Themes'] = ', '.join([p[0] for p in top_n])
                                results[result_idx]['Themes_Confidence'] = ', '.join([str(round(p[1], 4)) for p in top_n])
                            elif 'label' in pred:
                                # Single label format
                                results[result_idx]['Themes'] = pred['label']
                                results[result_idx]['Themes_Confidence'] = _extract_confidence(pred.get('score', pred.get('confidence', 0)))
                            else:
                                # Format: {'politics': 0.8, 'economy': 0.1, ...}
                                scored_items = [(k, v) for k, v in pred.items() if isinstance(v, (int, float))]
                                if scored_items:
                                    scored_items.sort(key=lambda x: x[1], reverse=True)
                                    top_n = scored_items[:top_themes]
                                    results[result_idx]['Themes'] = ', '.join([p[0] for p in top_n])
                                    results[result_idx]['Themes_Confidence'] = ', '.join([str(round(p[1], 4)) for p in top_n])

                    total_processed += len(chunk_texts)
                    logger.info(f"Theme extraction: {total_processed}/{len(texts_needing_themes)} processed...")

                    # Save incremental checkpoint after each chunk (except last)
                    if chunk_end < len(texts_needing_themes):
                        checkpoint_bytes = _process_excel_with_predictions(b, results, sheet_names=process_sheets, text_column=text_column, preset="themes")
                        save_checkpoint(checkpoint_bytes, "4_themes", original_filename, is_partial=True)

                # CHECKPOINT 4: After theme extraction
                checkpoint_bytes = _process_excel_with_predictions(b, results, sheet_names=process_sheets, text_column=text_column, preset="themes")
                save_checkpoint(checkpoint_bytes, "4_themes", original_filename)
            else:
                logger.info("SKIPPING theme extraction (all rows already have themes)")

        # Generate narratives if requested (requires themes to be extracted) - handle partial completion
        if generate_narrative and extract_themes:
            # Check which rows already have narratives populated
            texts_needing_narrative = []
            themes_needing_narrative = []
            indices_needing_narrative = []
            result_idx = 0

            for sheet in sheets_to_process:
                if sheet not in excel_file.sheet_names:
                    continue
                df = pd.read_excel(excel_file, sheet_name=sheet)
                if text_column not in df.columns:
                    continue

                has_narr_col = 'Narrative' in df.columns

                for idx in df.index:
                    if pd.notna(df.at[idx, text_column]) and result_idx < len(results):
                        existing_narr = df.at[idx, 'Narrative'] if has_narr_col and pd.notna(df.at[idx, 'Narrative']) else None

                        if existing_narr and str(existing_narr).strip() and not str(existing_narr).strip().startswith('[NARRATIVE UNAVAILABLE'):
                            # Already has narrative - load it
                            results[result_idx]['Narrative'] = str(existing_narr)
                        else:
                            # Needs narrative generation
                            texts_needing_narrative.append(analysis_texts[result_idx])
                            top_theme = results[result_idx].get('Themes', '').split(',')[0].strip()
                            themes_needing_narrative.append(top_theme)
                            indices_needing_narrative.append(result_idx)
                        result_idx += 1

            if texts_needing_narrative:
                logger.info(f"Generating narratives for {len(texts_needing_narrative)} rows ({len(results) - len(texts_needing_narrative)} already have narratives)...")
                theme_labels = lbls if 'lbls' in dir() else guidance_labels or DEFAULT_ZS_LABELS

                # Process narrative generation in chunks with incremental saves
                # Narratives are slow (LLM generation), so use smaller chunks
                NARRATIVE_CHUNK_SIZE = 10
                total_processed = 0

                for chunk_start in range(0, len(texts_needing_narrative), NARRATIVE_CHUNK_SIZE):
                    chunk_end = min(chunk_start + NARRATIVE_CHUNK_SIZE, len(texts_needing_narrative))
                    chunk_texts = texts_needing_narrative[chunk_start:chunk_end]
                    chunk_themes = themes_needing_narrative[chunk_start:chunk_end]
                    chunk_indices = indices_needing_narrative[chunk_start:chunk_end]

                    # Generate narratives for this chunk (with few-shot learning if training data available)
                    narratives = generate_narratives_batch(
                        chunk_texts, theme_labels, chunk_themes,
                        training_store=training_store_instance
                    )

                    for i, narrative in enumerate(narratives):
                        result_idx = chunk_indices[i]
                        text = chunk_texts[i]

                        # Check if the source text was a failed fetch
                        if text.startswith('[FETCH FAILED'):
                            results[result_idx]['Narrative'] = '[NARRATIVE UNAVAILABLE: URL fetch failed]'
                        else:
                            results[result_idx]['Narrative'] = narrative

                    total_processed += len(chunk_texts)
                    logger.info(f"Narrative generation: {total_processed}/{len(texts_needing_narrative)} processed...")

                    # Save incremental checkpoint after each chunk (except last)
                    if chunk_end < len(texts_needing_narrative):
                        checkpoint_bytes = _process_excel_with_predictions(b, results, sheet_names=process_sheets, text_column=text_column, preset="narratives")
                        save_checkpoint(checkpoint_bytes, "5_narratives", original_filename, is_partial=True)

                logger.info(f"Generated {total_processed} narratives total")

                # CHECKPOINT 5: After narrative generation
                checkpoint_bytes = _process_excel_with_predictions(b, results, sheet_names=process_sheets, text_column=text_column, preset="narratives")
                save_checkpoint(checkpoint_bytes, "5_narratives", original_filename)
            else:
                logger.info("SKIPPING narrative generation (all rows already have narratives)")

        elapsed_time = time.time() - start_time
        logger.info(f"Processing complete: {len(results)} rows with stance={extract_stance}, themes={extract_themes}, narratives={generate_narrative} in {elapsed_time:.2f}s")

        # Create annotated Excel (use process_sheets to exclude Guidance sheet)
        excel_bytes = _process_excel_with_predictions(
            b,
            results,
            sheet_names=process_sheets,
            text_column=text_column,
            preset=f"sentiment+themes"
        )

        output_filename = original_filename.rsplit('.', 1)[0] + '_analyzed.xlsx'
        response = _make_download(
            output_filename,
            excel_bytes,
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        response.headers["X-Processing-Time"] = f"{elapsed_time:.2f}"
        response.headers["X-Rows-Processed"] = str(len(texts))
        return response

    except Exception as e:
        logger.error(f"Batch Excel processing failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Batch Excel processing failed: {str(e)}")


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


# ============================================================
# Annotation Comparison Endpoint
# ============================================================

@app.post("/compare-annotations")
async def compare_annotations(
    automated_file: UploadFile = File(..., description="Excel file with automated annotations"),
    human_file: UploadFile = File(..., description="Excel file with human annotations"),
    url_column: str = Query("URL", description="Column name containing URLs for row matching"),
    sheets: Optional[str] = Query(None, description="Comma-separated sheet names to compare (all sheets if not specified)"),
    compare_stance: bool = Query(True, description="Compare Stance/Sentiment columns"),
    compare_themes: bool = Query(True, description="Compare Themes columns"),
    compare_narrative: bool = Query(False, description="Compare Narrative columns"),
):
    """
    Compare automated annotations against human annotations.

    Matches rows by URL and produces a diff showing:
    - Agreement/disagreement on Stance
    - Agreement/disagreement on Themes
    - Overall accuracy metrics

    Returns an Excel file with side-by-side comparison and a summary sheet.
    """
    logger.info(f"Comparing annotations: automated={automated_file.filename}, human={human_file.filename}, sheets={sheets}")

    try:
        auto_bytes = await automated_file.read()
        human_bytes = await human_file.read()

        auto_excel = pd.ExcelFile(io.BytesIO(auto_bytes))
        human_excel = pd.ExcelFile(io.BytesIO(human_bytes))

        # Parse specified sheets if provided
        specified_sheets = None
        if sheets:
            specified_sheets = [s.strip() for s in sheets.split(',') if s.strip()]
            logger.info(f"User specified sheets: {specified_sheets}")

        # Find common sheets (excluding Guidance, Pivot, etc.)
        skip_sheets = {'guidance', 'pivot', 'summary', 'stats', 'comparison'}
        auto_sheets = [s for s in auto_excel.sheet_names if s.lower() not in skip_sheets]
        human_sheets = [s for s in human_excel.sheet_names if s.lower() not in skip_sheets]

        # If user specified sheets, filter to only those
        if specified_sheets:
            auto_sheets = [s for s in auto_sheets if s in specified_sheets or s.lower() in [ss.lower() for ss in specified_sheets]]
            human_sheets = [s for s in human_sheets if s in specified_sheets or s.lower() in [ss.lower() for ss in specified_sheets]]
            logger.info(f"Filtered to specified sheets - auto: {auto_sheets}, human: {human_sheets}")

        # Try to match sheets by name
        common_sheets = []
        for auto_sheet in auto_sheets:
            for human_sheet in human_sheets:
                if auto_sheet.lower() == human_sheet.lower():
                    common_sheets.append((auto_sheet, human_sheet))
                    break

        if not common_sheets:
            # If no name matches, use first sheet from each
            if auto_sheets and human_sheets:
                common_sheets = [(auto_sheets[0], human_sheets[0])]
                logger.warning(f"No matching sheet names, using {auto_sheets[0]} and {human_sheets[0]}")

        if not common_sheets:
            raise HTTPException(status_code=400, detail="No comparable sheets found in the files")

        all_comparisons = []
        summary_stats = {
            "total_matched_rows": 0,
            "stance_matches": 0,
            "stance_mismatches": 0,
            "themes_matches": 0,
            "themes_partial_matches": 0,
            "themes_mismatches": 0,
            "unmatched_auto_rows": 0,
            "unmatched_human_rows": 0,
            "confusion_matrix": {},  # Tracks Human→Auto stance transitions
        }

        for auto_sheet, human_sheet in common_sheets:
            auto_df = pd.read_excel(auto_excel, sheet_name=auto_sheet)
            human_df = pd.read_excel(human_excel, sheet_name=human_sheet)

            # Find URL column (case-insensitive)
            auto_url_col = None
            human_url_col = None

            for col in auto_df.columns:
                if col.lower() == url_column.lower() or 'url' in col.lower():
                    auto_url_col = col
                    break

            for col in human_df.columns:
                if col.lower() == url_column.lower() or 'url' in col.lower():
                    human_url_col = col
                    break

            if not auto_url_col or not human_url_col:
                logger.warning(f"URL column not found in sheets {auto_sheet}/{human_sheet}, skipping")
                continue

            # Normalize URLs for matching (strip whitespace, lowercase)
            def normalize_url(url):
                if pd.isna(url):
                    return None
                return str(url).strip().lower()

            auto_df['_norm_url'] = auto_df[auto_url_col].apply(normalize_url)
            human_df['_norm_url'] = human_df[human_url_col].apply(normalize_url)

            # Create URL to row mapping for human annotations
            human_by_url = {}
            for idx, row in human_df.iterrows():
                norm_url = row['_norm_url']
                if norm_url:
                    human_by_url[norm_url] = row

            # Find stance/sentiment columns
            def find_stance_col(df):
                for col in df.columns:
                    col_lower = col.lower()
                    if col_lower in ('stance', 'sentiment') or 'stance' in col_lower:
                        return col
                return None

            def find_themes_col(df):
                for col in df.columns:
                    if col.lower() == 'themes' or 'theme' in col.lower():
                        return col
                return None

            def find_narrative_col(df):
                for col in df.columns:
                    col_lower = col.lower()
                    if 'narrative' in col_lower or 'summary' in col_lower:
                        return col
                return None

            auto_stance_col = find_stance_col(auto_df)
            human_stance_col = find_stance_col(human_df)
            auto_themes_col = find_themes_col(auto_df)
            human_themes_col = find_themes_col(human_df)
            auto_narrative_col = find_narrative_col(auto_df)
            human_narrative_col = find_narrative_col(human_df)

            logger.info(f"Sheet {auto_sheet}: auto_stance={auto_stance_col}, human_stance={human_stance_col}")
            logger.info(f"Sheet {auto_sheet}: auto_themes={auto_themes_col}, human_themes={human_themes_col}")

            # Build comparison rows
            comparison_rows = []
            matched_human_urls = set()

            for idx, auto_row in auto_df.iterrows():
                norm_url = auto_row['_norm_url']
                if not norm_url:
                    continue

                human_row = human_by_url.get(norm_url)

                comp = {
                    'URL': auto_row[auto_url_col],
                    'Sheet': auto_sheet,
                }

                if human_row is not None:
                    matched_human_urls.add(norm_url)
                    summary_stats["total_matched_rows"] += 1

                    # Compare stance
                    if compare_stance and auto_stance_col and human_stance_col:
                        auto_stance = str(auto_row.get(auto_stance_col, '')).strip().upper() if pd.notna(auto_row.get(auto_stance_col)) else ''
                        human_stance = str(human_row.get(human_stance_col, '')).strip().upper() if pd.notna(human_row.get(human_stance_col)) else ''

                        # Normalize stance values - comprehensive mapping
                        stance_map = {
                            # Positive variants → SUPPORT
                            'POSITIVE': 'SUPPORT', 'POS': 'SUPPORT', 'FAVORABLE': 'SUPPORT',
                            'SUPPORTIVE': 'SUPPORT', 'PRO': 'SUPPORT', 'AGREE': 'SUPPORT',
                            'FOR': 'SUPPORT', 'YES': 'SUPPORT', 'ENTAILMENT': 'SUPPORT',
                            # Negative variants → OPPOSE
                            'NEGATIVE': 'OPPOSE', 'NEG': 'OPPOSE', 'UNFAVORABLE': 'OPPOSE',
                            'OPPOSING': 'OPPOSE', 'ANTI': 'OPPOSE', 'CON': 'OPPOSE',
                            'AGAINST': 'OPPOSE', 'DISAGREE': 'OPPOSE', 'NO': 'OPPOSE',
                            'CONTRADICTION': 'OPPOSE',
                            # Neutral variants
                            'NEU': 'NEUTRAL', 'MIXED': 'NEUTRAL', 'NONE': 'NEUTRAL',
                            'N/A': 'NEUTRAL', 'NA': 'NEUTRAL', 'UNKNOWN': 'NEUTRAL',
                        }
                        auto_stance_norm = stance_map.get(auto_stance, auto_stance)
                        human_stance_norm = stance_map.get(human_stance, human_stance)

                        comp['Auto_Stance'] = auto_stance_norm
                        comp['Human_Stance'] = human_stance_norm
                        comp['Auto_Stance_Raw'] = auto_stance
                        comp['Human_Stance_Raw'] = human_stance

                        # Track confusion matrix
                        confusion_key = f"{human_stance_norm}→{auto_stance_norm}"
                        summary_stats["confusion_matrix"][confusion_key] = summary_stats["confusion_matrix"].get(confusion_key, 0) + 1

                        if auto_stance_norm == human_stance_norm:
                            comp['Stance_Match'] = 'MATCH'
                            summary_stats["stance_matches"] += 1
                        else:
                            comp['Stance_Match'] = 'MISMATCH'
                            summary_stats["stance_mismatches"] += 1

                    # Compare themes
                    if compare_themes and auto_themes_col and human_themes_col:
                        auto_themes_raw = str(auto_row.get(auto_themes_col, '')) if pd.notna(auto_row.get(auto_themes_col)) else ''
                        human_themes_raw = str(human_row.get(human_themes_col, '')) if pd.notna(human_row.get(human_themes_col)) else ''

                        # Parse themes (comma or semicolon separated)
                        auto_themes_set = set(t.strip().lower() for t in auto_themes_raw.replace(';', ',').split(',') if t.strip())
                        human_themes_set = set(t.strip().lower() for t in human_themes_raw.replace(';', ',').split(',') if t.strip())

                        comp['Auto_Themes'] = auto_themes_raw
                        comp['Human_Themes'] = human_themes_raw

                        # Don't count empty vs empty as a match
                        if not auto_themes_set and not human_themes_set:
                            comp['Themes_Match'] = 'BOTH_EMPTY'
                            # Don't count in any category - skip
                        elif auto_themes_set == human_themes_set:
                            comp['Themes_Match'] = 'MATCH'
                            summary_stats["themes_matches"] += 1
                        elif auto_themes_set & human_themes_set:  # Any overlap
                            overlap = auto_themes_set & human_themes_set
                            comp['Themes_Match'] = f'PARTIAL ({len(overlap)} common)'
                            summary_stats["themes_partial_matches"] += 1
                        else:
                            comp['Themes_Match'] = 'MISMATCH'
                            summary_stats["themes_mismatches"] += 1

                    # Compare narrative (just show side by side, hard to auto-compare)
                    if compare_narrative and auto_narrative_col and human_narrative_col:
                        comp['Auto_Narrative'] = str(auto_row.get(auto_narrative_col, ''))[:500] if pd.notna(auto_row.get(auto_narrative_col)) else ''
                        comp['Human_Narrative'] = str(human_row.get(human_narrative_col, ''))[:500] if pd.notna(human_row.get(human_narrative_col)) else ''

                else:
                    # No matching human annotation
                    comp['Auto_Stance'] = str(auto_row.get(auto_stance_col, '')) if auto_stance_col and pd.notna(auto_row.get(auto_stance_col)) else ''
                    comp['Human_Stance'] = '[NOT FOUND]'
                    comp['Stance_Match'] = 'NO_MATCH_ROW'
                    summary_stats["unmatched_auto_rows"] += 1

                comparison_rows.append(comp)

            # Check for human rows not in automated
            for norm_url, human_row in human_by_url.items():
                if norm_url not in matched_human_urls:
                    summary_stats["unmatched_human_rows"] += 1

            all_comparisons.extend(comparison_rows)

        # Create output Excel
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Comparison sheet
            if all_comparisons:
                comp_df = pd.DataFrame(all_comparisons)
                comp_df.to_excel(writer, sheet_name='Comparison', index=False)

            # Summary sheet
            total_stance = summary_stats["stance_matches"] + summary_stats["stance_mismatches"]
            total_themes = summary_stats["themes_matches"] + summary_stats["themes_partial_matches"] + summary_stats["themes_mismatches"]

            summary_data = {
                'Metric': [
                    'Total Matched Rows',
                    'Stance Matches',
                    'Stance Mismatches',
                    'Stance Accuracy',
                    'Themes Exact Matches',
                    'Themes Partial Matches',
                    'Themes Mismatches',
                    'Themes Accuracy (exact)',
                    'Themes Accuracy (partial OK)',
                    'Unmatched Auto Rows',
                    'Unmatched Human Rows',
                ],
                'Value': [
                    summary_stats["total_matched_rows"],
                    summary_stats["stance_matches"],
                    summary_stats["stance_mismatches"],
                    f"{(summary_stats['stance_matches'] / total_stance * 100):.1f}%" if total_stance > 0 else "N/A",
                    summary_stats["themes_matches"],
                    summary_stats["themes_partial_matches"],
                    summary_stats["themes_mismatches"],
                    f"{(summary_stats['themes_matches'] / total_themes * 100):.1f}%" if total_themes > 0 else "N/A",
                    f"{((summary_stats['themes_matches'] + summary_stats['themes_partial_matches']) / total_themes * 100):.1f}%" if total_themes > 0 else "N/A",
                    summary_stats["unmatched_auto_rows"],
                    summary_stats["unmatched_human_rows"],
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)

            # Confusion Matrix sheet (shows Human label → Auto label transitions)
            if summary_stats["confusion_matrix"]:
                confusion_data = {
                    'Human → Auto': list(summary_stats["confusion_matrix"].keys()),
                    'Count': list(summary_stats["confusion_matrix"].values())
                }
                confusion_df = pd.DataFrame(confusion_data)
                confusion_df = confusion_df.sort_values('Count', ascending=False)
                confusion_df.to_excel(writer, sheet_name='Confusion Matrix', index=False)

        output.seek(0)

        # Generate filename
        from pathlib import Path
        base_name = Path(automated_file.filename or "comparison").stem
        output_filename = f"{base_name}_vs_human_diff.xlsx"

        logger.info(f"Comparison complete: {summary_stats}")

        return StreamingResponse(
            output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename={output_filename}"}
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Annotation comparison failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    port = int(os.getenv("SERVICE_PORT", 8001))

    logger.info("=" * 70)
    logger.info("CiceroWatch NLP Service")
    logger.info("=" * 70)
    logger.info(f"Listening on: http://0.0.0.0:{port}")
    logger.info("Features: Sentiment, Entities, Topics, URL Analysis, Stance Detection")
    logger.info("Batch processing enabled: CLASSIFY_BATCH_SIZE=16, ZEROSHOT_BATCH_SIZE=8, STANCE_BATCH_SIZE=32")
    logger.info("Health checks: every 10 minutes")
    logger.info("=" * 70)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
