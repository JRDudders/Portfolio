# app.py — thin API
from __future__ import annotations
import io
from pathlib import Path
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from nlp import MODEL_ID
from adapters import process_url, AdapterOutput
from apputils import process_csv_bytes, process_json_bytes, filename_with_suffix
from fetch import shutdown_playwright

app = FastAPI(title="Sentiment Processor", version="4.0")

if Path("static").is_dir():
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.on_event("shutdown")
async def _shutdown():
    await shutdown_playwright(app)

@app.get("/healthz")
def healthz():
    return {"status": "ok", "model": MODEL_ID}

@app.get("/", response_class=HTMLResponse)
def index(_: Request):
    p = Path("index.html")
    return HTMLResponse(p.read_text(encoding="utf-8")) if p.exists() else HTMLResponse("<h3>Service up</h3>")

@app.post("/predict/file")
async def predict_file(file: UploadFile = File(...)):
    data = await file.read()
    fname = file.filename or "input"
    ctype = (file.content_type or "").lower()

    if fname.lower().endswith(".csv") or ctype in {"text/csv","application/csv"}:
        out_bytes = process_csv_bytes(data)
        out_name = filename_with_suffix(fname, "csv")
        return StreamingResponse(io.BytesIO(out_bytes), media_type="text/csv; charset=utf-8",
                                 headers={"Content-Disposition": f'attachment; filename="{out_name}"'})

    if fname.lower().endswith(".json") or ctype in {"application/json","text/json"}:
        out_bytes = process_json_bytes(data)
        out_name = filename_with_suffix(fname, "json")
        return StreamingResponse(io.BytesIO(out_bytes), media_type="application/json; charset=utf-8",
                                 headers={"Content-Disposition": f'attachment; filename="{out_name}"'})

    if fname.lower().endswith((".htm",".html")) or ctype == "text/html":
        # Let the generic URL adapter handle local HTML if you prefer — or keep old path.
        from render import html_to_paragraphs, extract_title, render_annotated_html
        from nlp import preprocess_text, classify_texts
        paras = html_to_paragraphs(data)
        preds = classify_texts([preprocess_text(p) for p in paras])
        title = extract_title(data) or Path(fname).stem
        html_out = render_annotated_html(title, fname, paras, preds)
        return Response(content=html_out, media_type="text/html; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="{filename_with_suffix(fname, "html")}"'})
    raise HTTPException(400, detail="Upload a .csv, .json, or .html file")

@app.post("/predict/url")
async def predict_url(request: Request):
    body = await request.json()
    url = body.get("url")
    if not url:
        raise HTTPException(400, detail="Body must include {'url': '<http(s)://...>'}")
    render = bool(body.get("render", False))
    cookies = body.get("cookies")
    crawl = bool(body.get("crawl", False))
    max_pages = int(body.get("max_pages", 50))
    max_depth = int(body.get("max_depth", 2))
    same_host_only = bool(body.get("same_host_only", True))
    delay_ms = int(body.get("delay_ms", 300))

    try:
        out: AdapterOutput = await process_url(
            url, app=app, render=render, cookies=cookies,
            crawl=crawl, max_pages=max_pages, max_depth=max_depth,
            same_host_only=same_host_only, delay_ms=delay_ms
        )
    except Exception as e:
        raise HTTPException(502, detail=str(e))

    headers = {"Content-Disposition": f'attachment; filename="{out.filename}"'}
    return Response(content=out.content, media_type=out.media_type, headers=headers)
