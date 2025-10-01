# app.py
from __future__ import annotations
import io
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from adapters import process_url, AdapterOutput
from apputils import process_csv_bytes, process_json_bytes, filename_with_suffix
from render import html_to_paragraphs, extract_title, render_annotated_html, guess_labels_from_html
from nlp import preprocess_text, run_task, MODEL_ID, MODEL_TASK

app = FastAPI(title="Text Processor", version="6.1")

if Path("static").is_dir():
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/healthz")
def healthz():
    return {"status": "ok", "model": MODEL_ID}

@app.get("/", response_class=HTMLResponse)
def index(_: Request):
    p = Path("index.html")
    return HTMLResponse(p.read_text(encoding="utf-8")) if p.exists() else HTMLResponse("<h3>Service up</h3>")

def _boolish(v) -> bool:
    if v is None: return False
    return str(v).strip().lower() in {"1","true","yes","on"}

def _labels_from_qp(raw: Optional[str]) -> Optional[List[str]]:
    if not raw: return None
    out = [s.strip() for s in raw.split(",")]
    out = [s for s in out if s]
    return out or None

@app.post("/predict/file")
async def predict_file(request: Request, file: UploadFile = File(...)):
    qp = request.query_params
    task = qp.get("task"); preset = qp.get("preset")
    labels = _labels_from_qp(qp.get("labels"))
    include_stopwords = _boolish(qp.get("include_stopwords"))

    data = await file.read()
    fname = file.filename or "input"
    ctype = (file.content_type or "").lower()

    if fname.lower().endswith(".csv") or ctype in {"text/csv","application/csv"}:
        out_bytes = process_csv_bytes(data, task=task, preset=preset, labels=labels)
        out_name = filename_with_suffix(fname, "csv")
        return StreamingResponse(io.BytesIO(out_bytes), media_type="text/csv; charset=utf-8",
                                 headers={"Content-Disposition": f'attachment; filename="{out_name}"'})

    if fname.lower().endswith(".json") or ctype in {"application/json","text/json"}:
        out_bytes = process_json_bytes(data, task=task, preset=preset, labels=labels)
        out_name = filename_with_suffix(fname, "json")
        return StreamingResponse(io.BytesIO(out_bytes), media_type="application/json; charset=utf-8",
                                 headers={"Content-Disposition": f'attachment; filename="{out_name}"'})

    if fname.lower().endswith((".htm",".html")) or ctype == "text/html":
        paras = html_to_paragraphs(data)
        effective_task = task or MODEL_TASK or "text-classification"
        eff_labels = labels
        if (effective_task == "zero-shot-classification" or (preset and preset.startswith("zeroshot-"))) and not eff_labels:
            eff_labels = guess_labels_from_html(data, k=12, include_stopwords=include_stopwords)
        texts = [preprocess_text(p) if effective_task != "token-classification" else p for p in paras]
        preds = run_task(texts, task=effective_task, preset=preset, labels=eff_labels)
        title = extract_title(data) or Path(fname).stem
        html_out = render_annotated_html(title, fname, paras, preds, task=effective_task)
        return Response(content=html_out, media_type="text/html; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="{filename_with_suffix(fname, "html")}"'})

    raise HTTPException(400, detail="Upload a .csv, .json, or .html file")

@app.post("/predict/url")
async def predict_url(request: Request):
    body: Dict[str, Any] = await request.json()
    url = body.get("url")
    if not url:
        raise HTTPException(400, detail="Body must include {'url': '<http(s)://...>'}")

    # Rendering controls
    render_js = bool(body.get("render", False))
    renderer = (body.get("renderer") or "auto").lower()  # auto|playwright|selenium
    wait_selector = body.get("wait_selector")  # e.g. "article, main, .entry-content"
    scroll_passes = int(body.get("scroll_passes", 6))  # how many times to scroll
    render_timeout_ms = int(body.get("render_timeout_ms", 25000))
    cookies = body.get("cookies")
    extra_headers = body.get("extra_headers")  # dict of header->value

    # Crawl controls
    crawl = bool(body.get("crawl", False))
    max_pages = int(body.get("max_pages", 50))
    max_depth = int(body.get("max_depth", 2))
    same_host_only = bool(body.get("same_host_only", True))
    delay_ms = int(body.get("delay_ms", 300))

    # NLP controls
    include_stopwords = _boolish(body.get("include_stopwords"))
    task = body.get("task")
    preset = body.get("preset")
    labels = body.get("labels")

    try:
        out: AdapterOutput = await process_url(
            url,
            app=app,
            render=render_js,
            renderer=renderer,
            cookies=cookies,
            crawl=crawl,
            max_pages=max_pages,
            max_depth=max_depth,
            same_host_only=same_host_only,
            delay_ms=delay_ms,
            wait_selector=wait_selector,
            scroll_passes=scroll_passes,
            render_timeout_ms=render_timeout_ms,
            extra_headers=extra_headers,
            task=task,
            preset=preset,
            labels=labels,
            include_stopwords=include_stopwords,
        )
    except Exception as e:
        raise HTTPException(502, detail=str(e))

    headers = {"Content-Disposition": f'attachment; filename="{out.filename}"'}
    return Response(content=out.content, media_type=out.media_type, headers=headers)
