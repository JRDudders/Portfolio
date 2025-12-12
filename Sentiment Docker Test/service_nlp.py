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
):
    """
    Batch process Excel file with stance detection AND theme extraction.

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
    def save_checkpoint(excel_bytes: bytes, stage: str, filename: str):
        """Save checkpoint file to temp directory"""
        import os
        from pathlib import Path
        checkpoint_dir = Path("/app/temp/checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        base_name = Path(filename).stem
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
            logger.info(f"Guidance sheet has {len(guidance_df)} rows, {len(guidance_df.columns)} columns")

            all_values = set()

            # Find the numeric ranking column (could be any column, not just A)
            # Then use the next column for labels
            ranking_col = None
            label_col = None

            for col_idx in range(len(guidance_df.columns)):
                col_data = guidance_df.iloc[:, col_idx].dropna()
                if len(col_data) == 0:
                    continue
                # Check if this column is mostly numeric (ranking numbers 1,2,3...)
                numeric_count = sum(1 for v in col_data if str(v).strip().replace('.', '').isdigit())
                if numeric_count >= len(col_data) * 0.3 and numeric_count >= 3:
                    ranking_col = col_idx
                    # Use the next column for labels if it exists
                    if col_idx + 1 < len(guidance_df.columns):
                        label_col = col_idx + 1
                        logger.info(f"Detected numeric ranking in column {col_idx}, using column {label_col} for labels")
                    break

            # If no ranking column found, find first column with text content
            if label_col is None:
                for col_idx in range(len(guidance_df.columns)):
                    col_data = guidance_df.iloc[:, col_idx].dropna()
                    # Check if column has substantial text (not just numbers)
                    text_count = sum(1 for v in col_data if not str(v).strip().replace('.', '').isdigit() and len(str(v).strip()) > 2)
                    if text_count >= 3:
                        label_col = col_idx
                        logger.info(f"Using column {col_idx} for labels (text content detected)")
                        break

            if label_col is None:
                label_col = 0  # Fallback to first column

            # Extract labels from the identified column
            for val in guidance_df.iloc[:, label_col].dropna():
                val_str = str(val).strip()
                # Skip header-like rows (description text)
                val_lower = val_str.lower()
                if any(skip in val_lower for skip in ['hierarchy', 'informal', 'instance', 'mention', 'focus', 'equally', 'theme:']):
                    continue
                # Skip pure numbers
                if val_str.replace('.', '').isdigit():
                    continue
                # Skip empty or very short values
                if val_str and len(val_str) > 1:
                    all_values.add(val_str)

            guidance_labels = list(all_values)
            logger.info(f"Extracted {len(guidance_labels)} theme labels from Guidance sheet: {guidance_labels}")

        # Generate hypothesis from guidance if not provided
        generated_hypothesis = hypothesis
        if not generated_hypothesis and guidance_labels:
            # Create hypothesis from guidance themes
            themes_summary = ", ".join(guidance_labels[:5])
            if len(guidance_labels) > 5:
                themes_summary += f" (and {len(guidance_labels) - 5} more)"
            generated_hypothesis = f"U.S. priorities as outlined in the guidance ({themes_summary}) are viewed favorably and having a positive influence."
            logger.info(f"Generated hypothesis: {generated_hypothesis}")
        elif not generated_hypothesis:
            generated_hypothesis = "U.S. strategic priorities and interests are viewed favorably in this content."
            logger.info(f"Using default hypothesis: {generated_hypothesis}")

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
                            text = _extract_text_from_html(content_bytes.decode('utf-8', errors='ignore'))
                            return (idx, text[:50000])
                        except Exception as e:
                            logger.warning(f"Failed to fetch URL {url_str}: {e}")

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
                                        return (idx, text[:50000])
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
                        if text:
                            fetched_count += 1
                        completed += 1

                        if completed % 10 == 0 or completed == len(tasks):
                            logger.info(f"Fetched {completed}/{len(tasks)} URLs...")

                        # Save incremental checkpoint every CHECKPOINT_INTERVAL URLs
                        if completed - last_checkpoint >= CHECKPOINT_INTERVAL and completed < len(tasks):
                            modified_dfs[sheet] = df
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
                            save_checkpoint(checkpoint_bytes, f"1_urls_partial_{completed}", original_filename)
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
                    save_checkpoint(checkpoint_bytes, f"1_urls_interrupted_{completed}", original_filename)
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
            new_translations = translate_batch(texts_to_translate)

            # Fill in the new translations at the correct indices
            for i, trans in enumerate(new_translations):
                all_translations[indices_to_translate[i]] = trans

            logger.info(f"Translated {len([t for t in new_translations if t])} texts")

            # Update the Excel sheets with new translations
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
                stance_preds = run_task(texts_needing_stance, preset=stance_preset, claim=generated_hypothesis)

                for i, pred in enumerate(stance_preds):
                    result_idx = indices_needing_stance[i]
                    text = texts_needing_stance[i]

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

                # CHECKPOINT 3: After stance detection
                checkpoint_bytes = _process_excel_with_predictions(b, results, sheet_names=process_sheets, text_column=text_column, preset="stance")
                save_checkpoint(checkpoint_bytes, "3_stance", original_filename)
            else:
                logger.info("SKIPPING stance detection (all rows already have stance)")

        # Run theme/topic extraction if requested - handle partial completion
        # Set lbls first (needed for narrative generation regardless of theme extraction)
        if guidance_labels:
            lbls = guidance_labels
        elif labels:
            lbls = _parse_labels_csv(labels)
        else:
            lbls = DEFAULT_ZS_LABELS

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
                theme_preds = run_task(texts_needing_themes, preset=theme_preset, labels=lbls)

                for i, pred in enumerate(theme_preds):
                    result_idx = indices_needing_themes[i]
                    text = texts_needing_themes[i]

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

                narratives = generate_narratives_batch(texts_needing_narrative, theme_labels, themes_needing_narrative)

                for i, narrative in enumerate(narratives):
                    result_idx = indices_needing_narrative[i]
                    text = texts_needing_narrative[i]

                    # Check if the source text was a failed fetch
                    if text.startswith('[FETCH FAILED'):
                        results[result_idx]['Narrative'] = '[NARRATIVE UNAVAILABLE: URL fetch failed]'
                    else:
                        results[result_idx]['Narrative'] = narrative

                logger.info(f"Generated {len([n for n in narratives if n])} narratives")

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
