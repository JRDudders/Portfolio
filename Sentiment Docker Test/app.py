# app.py
from __future__ import annotations

import io
import json
import os
import typing as T
from urllib.parse import urlparse

import pandas as pd
import requests
from fastapi import FastAPI, UploadFile, File, Query, Body, HTTPException
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse, PlainTextResponse
from starlette.middleware.cors import CORSMiddleware

from nlp import run_task, PRESETS, DEFAULT_ZS_LABELS, preprocess_for_task
from graph_tasks import load_graph_from_bytes, run_graph_metrics

# Optional render/extraction
from bs4 import BeautifulSoup
import trafilatura

# Optional Playwright (async API only)
PLAYWRIGHT_AVAILABLE = False
try:
    from playwright.async_api import async_playwright  # type: ignore
    PLAYWRIGHT_AVAILABLE = True
except Exception:
    pass


APP_PORT = int(os.getenv("PORT", "8080"))
INDEX_PATH = os.path.join(os.getcwd(), "index.html")

app = FastAPI(title="NLP Service", version="1.0")

# Basic CORS (handy for local testing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------ Utilities ---------------------------------- #

def _as_json_bytes(obj: T.Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")

def _parse_labels_csv(s: str | None) -> T.List[str] | None:
    if not s:
        return None
    labels = [x.strip() for x in s.split(",") if x.strip()]
    return labels or None

def _texts_from_json_bytes(b: bytes) -> T.List[str]:
    data = json.loads(b.decode("utf-8"))
    if isinstance(data, list):
        if all(isinstance(x, str) for x in data):
            return T.cast(T.List[str], data)
        if all(isinstance(x, dict) for x in data):
            out = []
            for obj in data:
                if "text" in obj and isinstance(obj["text"], str):
                    out.append(obj["text"])
            if out:
                return out
    raise ValueError("JSON must be a list of strings or list of {'text': ...} objects")

def _texts_from_csv_bytes(b: bytes) -> T.List[str]:
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
    resp = StreamingResponse(io.BytesIO(payload), media_type=mime)
    resp.headers["Content-Disposition"] = f'attachment; filename="{name}"'
    return resp

def _extract_text_from_html(html: str, base_url: str | None = None) -> str:
    # Try trafilatura first
    try:
        extracted = trafilatura.extract(html, include_comments=False, favor_recall=False, url=base_url)
        if extracted and extracted.strip():
            return extracted.strip()
    except Exception:
        pass
    # Fallback: BeautifulSoup get_text
    soup = BeautifulSoup(html, "lxml")
    # Drop script/style
    for t in soup(["script", "style", "noscript"]):
        t.extract()
    text = soup.get_text("\n", strip=True)
    return text

async def _render_with_playwright(
    url: str,
    *,
    wait_selector: str | None,
    scroll_passes: int,
    timeout_ms: int,
    cookies_header: str | None,
    extra_headers: dict | None,
) -> str:
    if not PLAYWRIGHT_AVAILABLE:
        raise HTTPException(status_code=400, detail="Playwright is not installed/available")

    # Derive cookie dicts for context.add_cookies
    cookie_list: T.List[dict] = []
    if cookies_header:
        # Cookie header: "name=value; name2=value2"
        parts = [p.strip() for p in cookies_header.split(";") if "=" in p]
        host = urlparse(url).hostname or ""
        for p in parts:
            name, value = p.split("=", 1)
            cookie_list.append({"name": name.strip(), "value": value.strip(), "domain": host, "path": "/"})

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        if extra_headers:
            await context.set_extra_http_headers(extra_headers)
        if cookie_list:
            try:
                await context.add_cookies(cookie_list)
            except Exception:
                # Ignore bad cookies
                pass
        page = await context.new_page()
        await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        # Optional wait for a selector
        if wait_selector:
            try:
                await page.wait_for_selector(wait_selector, timeout=timeout_ms)
            except Exception:
                pass
        # Scroll to trigger lazy loaders
        for _ in range(max(0, int(scroll_passes))):
            try:
                await page.evaluate("window.scrollBy(0, document.body.scrollHeight);")
                await page.wait_for_timeout(300)
            except Exception:
                break
        html = await page.content()
        await context.close()
        await browser.close()
        return html

def _fetch_simple(url: str, cookies_header: str | None, extra_headers: dict | None) -> str:
    headers = {"User-Agent": "Mozilla/5.0", "Accept-Language": "en-US,en;q=0.9"}
    if extra_headers:
        headers.update({str(k): str(v) for k, v in extra_headers.items()})
    if cookies_header:
        headers["Cookie"] = cookies_header
    r = requests.get(url, headers=headers, timeout=25)
    r.raise_for_status()
    return r.text


# ------------------------------- Routes ------------------------------------ #

@app.get("/", response_class=FileResponse)
def home():
    if not os.path.exists(INDEX_PATH):
        return PlainTextResponse("index.html not found in working directory.", status_code=404)
    return FileResponse(INDEX_PATH, media_type="text/html")

@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "presets": sorted(list(PRESETS.keys())),
        "playwright": PLAYWRIGHT_AVAILABLE,
    }


# ---- Predict from FILE ----------------------------------------------------- #
@app.post("/predict/file")
async def predict_file(
    file: UploadFile = File(...),
    preset: str | None = Query(None, description="Preset name from nlp.PRESETS"),
    labels: str | None = Query(None, description="Comma-separated labels for zero-shot"),
    include_stopwords: bool | None = Query(False),
):
    try:
        b = await file.read()
        name = (file.filename or "").lower()
        if name.endswith(".json"):
            texts = _texts_from_json_bytes(b)
        elif name.endswith(".csv"):
            texts = _texts_from_csv_bytes(b)
        elif name.endswith((".html", ".htm")):
            html = b.decode("utf-8", errors="ignore")
            text = _extract_text_from_html(html)
            texts = [text]
        else:
            # try JSON first, then CSV
            try:
                texts = _texts_from_json_bytes(b)
            except Exception:
                texts = _texts_from_csv_bytes(b)

        # Optional preprocessing per task (keeps raw for NER)
        task = PRESETS.get(preset, (None, None, {}))[0] if preset else None
        texts = [preprocess_for_task(t, task or "") for t in texts]

        lbls = _parse_labels_csv(labels)
        if (not lbls) and preset and "zeroshot" in preset:
            lbls = DEFAULT_ZS_LABELS

        result = run_task(texts, preset=preset, labels=lbls)

        payload = _as_json_bytes({"preset": preset, "results": result})
        return _make_download("predictions.json", payload, "application/json")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"File processing failed: {e}")


# ---- Predict from URL ------------------------------------------------------ #
class UrlBody(T.TypedDict, total=False):
    url: str
    preset: str
    labels: T.List[str] | None
    include_stopwords: bool
    render: bool
    renderer: str  # "auto" | "playwright" | "selenium" (selenium not implemented here)
    wait_selector: str | None
    scroll_passes: int
    render_timeout_ms: int
    cookies: str | None
    extra_headers: dict | None

@app.post("/predict/url")
async def predict_url(body: UrlBody = Body(...)):
    url = body.get("url")
    if not url:
        raise HTTPException(status_code=400, detail="Missing 'url'")

    preset = body.get("preset")
    labels = body.get("labels")
    render = bool(body.get("render", False))
    renderer = (body.get("renderer") or "auto").lower()
    wait_selector = body.get("wait_selector")
    scroll_passes = int(body.get("scroll_passes") or 8)
    timeout_ms = int(body.get("render_timeout_ms") or 25000)
    cookies_header = body.get("cookies")
    extra_headers = body.get("extra_headers") or {}

    # Fetch HTML (rendered or simple)
    try:
        if render and (renderer in {"auto", "playwright"}) and PLAYWRIGHT_AVAILABLE:
            html = await _render_with_playwright(
                url,
                wait_selector=wait_selector,
                scroll_passes=scroll_passes,
                timeout_ms=timeout_ms,
                cookies_header=cookies_header,
                extra_headers=extra_headers,
            )
        else:
            html = _fetch_simple(url, cookies_header, extra_headers)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Fetch/render failed: {e}")

    # Extract text & run task
    text = _extract_text_from_html(html, base_url=url)
    task = PRESETS.get(preset, (None, None, {}))[0] if preset else None
    texts = [preprocess_for_task(text, task or "")]
    lbls = labels or (DEFAULT_ZS_LABELS if (preset and "zeroshot" in preset) else None)

    try:
        result = run_task(texts, preset=preset, labels=lbls)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"NLP task failed: {e}")

    payload = _as_json_bytes({
        "preset": preset,
        "url": url,
        "results": result,
        "length_chars": len(text),
    })
    # We return JSON; your UI will save with this filename (overrides default .html)
    return _make_download("url-output.json", payload, "application/json")


# ---- Graph metrics --------------------------------------------------------- #
@app.post("/graph/metrics")
async def graph_metrics(
    file: UploadFile = File(..., description="Edge list CSV/JSON"),
    tasks: str = Query("degrees,pagerank", description="Comma-separated: degrees,pagerank,bfs,triangles"),
    bfs_source: str | None = Query(None),
    pagerank_alpha: float = Query(0.85),
    pagerank_iters: int = Query(40),
    pagerank_tol: float = Query(1e-6),
    triangles_limit: int = Query(20000),
):
    try:
        b = await file.read()
        kind = "csv" if (file.filename or "").lower().endswith(".csv") else "json"
        g = load_graph_from_bytes(b, kind=kind)
        task_list = [t.strip() for t in tasks.split(",") if t.strip()]
        res = run_graph_metrics(
            g,
            tasks=task_list,
            pagerank_alpha=pagerank_alpha,
            pagerank_iters=pagerank_iters,
            pagerank_tol=pagerank_tol,
            bfs_source=bfs_source,
            triangles_limit=triangles_limit,
        )
        return _make_download("graph-metrics.json", _as_json_bytes(res))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Graph metrics failed: {e}")
