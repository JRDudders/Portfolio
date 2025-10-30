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
    """Parse comma-separated labels, returning None for empty/whitespace strings."""
    if not s or not s.strip():  # Explicitly handle empty strings and whitespace
        return None
    labels = [x.strip() for x in s.split(",") if x.strip()]
    return labels if labels else None  # More explicit than 'or'

def _detect_graph_file_kind(filename: str | None) -> str:
    """Detect graph file type from filename extension."""
    if not filename:
        return "csv"  # Default fallback

    filename_lower = filename.lower()
    if filename_lower.endswith(".csv"):
        return "csv"
    elif filename_lower.endswith(".json"):
        return "json"
    elif filename_lower.endswith(".edge"):
        return "edge"
    elif filename_lower.endswith(".node"):
        return "node"
    else:
        return "csv"  # Default fallback

def _convert_to_json_serializable(obj):
    """Convert numpy/pandas types to JSON-serializable Python native types."""
    import numpy as np

    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_json_serializable(item) for item in obj]
    elif pd.isna(obj):
        return None
    else:
        return obj

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
    r = requests.get(url, headers=headers, timeout=300)
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

        # Keep original texts for output, preprocess separately
        original_texts = texts  # Save original
        processed_texts = [preprocess_for_task(t, task or "") for t in texts]

        # Only parse and use labels for zero-shot tasks
        lbls = None
        if preset and "zeroshot" in preset:
            lbls = _parse_labels_csv(labels)
            if not lbls:  # Use defaults if no labels provided for zero-shot
                lbls = DEFAULT_ZS_LABELS

        # Run task on processed texts
        predictions = run_task(processed_texts, preset=preset, labels=lbls)

        # Merge original texts with predictions
        if (task == "token-classification") or (preset and "ner" in preset):
            # NER: keep entities format
            output = [{"text": t, "entities": p.get("entities", [])} for t, p in zip(original_texts, predictions)]
        else:
            # Classification: merge text with scores
            output = []
            for t, p in zip(original_texts, predictions):
                result_dict = {"text": t}
                if isinstance(p, dict):
                    result_dict.update(p)  # Add labels, scores, etc.
                output.append(result_dict)

        payload = _as_json_bytes({"preset": preset, "results": output})
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

    # Keep original text for output, preprocess separately
    original_text = text
    processed_text = preprocess_for_task(text, task or "")

    # Only use labels for zero-shot tasks
    lbls = None
    if preset and "zeroshot" in preset:
        lbls = labels if labels else DEFAULT_ZS_LABELS

    try:
        predictions = run_task([processed_text], preset=preset, labels=lbls)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"NLP task failed: {e}")

    # Merge original text with predictions
    if (task == "token-classification") or (preset and "ner" in preset):
        output = [{"text": original_text, "entities": predictions[0].get("entities", [])}]
    else:
        result_dict = {"text": original_text}
        if predictions and isinstance(predictions[0], dict):
            result_dict.update(predictions[0])
        output = [result_dict]

    payload = _as_json_bytes({
        "preset": preset,
        "url": url,
        "results": output,
        "length_chars": len(text),
    })
    # We return JSON; your UI will save with this filename (overrides default .html)
    return _make_download("url-output.json", payload, "application/json")


# ---- Batch sentiment analysis ---------------------------------------------- #
class BatchTextRequest(T.TypedDict, total=False):
    texts: T.List[str]
    preset: str
    labels: T.List[str] | None
    preprocess: bool

@app.post("/predict/batch")
async def predict_batch(body: BatchTextRequest = Body(...)):
    """
    Efficiently process multiple texts for sentiment analysis or other NLP tasks.
    Optimized for batch processing with automatic preprocessing.
    """
    texts = body.get("texts")
    if not texts or not isinstance(texts, list):
        raise HTTPException(status_code=400, detail="Missing or invalid 'texts' array")

    preset = body.get("preset", "sentiment-twitter")
    labels = body.get("labels")
    do_preprocess = body.get("preprocess", True)

    try:
        # Preprocess if requested
        task = PRESETS.get(preset, (None, None, {}))[0] if preset else None

        # Keep original texts for output
        original_texts = texts
        if do_preprocess:
            processed_texts = [preprocess_for_task(t, task or "") for t in texts]
        else:
            processed_texts = texts

        # Only use labels for zero-shot tasks
        lbls = None
        if preset and "zeroshot" in preset:
            lbls = labels if labels else DEFAULT_ZS_LABELS

        # Run batch inference
        predictions = run_task(processed_texts, preset=preset, labels=lbls)

        # Merge original texts with predictions
        if (task == "token-classification") or (preset and "ner" in preset):
            output = [{"text": t, "entities": p.get("entities", [])} for t, p in zip(original_texts, predictions)]
        else:
            output = []
            for t, p in zip(original_texts, predictions):
                result_dict = {"text": t}
                if isinstance(p, dict):
                    result_dict.update(p)
                output.append(result_dict)

        # Format response
        response = {
            "preset": preset,
            "count": len(texts),
            "results": output,
        }

        return JSONResponse(content=response)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch processing failed: {e}")


# ---- Graph metrics --------------------------------------------------------- #
@app.post("/graph/metrics")
async def graph_metrics(
    edges_file: UploadFile = File(..., description="Edge list (CSV/JSON/.edge)"),
    nodes_file: UploadFile | None = File(None, description="Optional node attributes (CSV/JSON/.node)"),
    tasks: str = Query("degrees,pagerank", description="Comma-separated: degrees,pagerank,bfs,triangles"),
    bfs_source: str | None = Query(None),
    pagerank_alpha: float = Query(0.85),
    pagerank_iters: int = Query(40),
    pagerank_tol: float = Query(1e-6),
    triangles_limit: int = Query(20000),
):
    """Compute multiple graph metrics in one call."""
    try:
        b = await edges_file.read()
        kind = _detect_graph_file_kind(edges_file.filename)

        nodes_bytes = None
        nodes_kind = None
        if nodes_file:
            nodes_bytes = await nodes_file.read()
            nodes_kind = _detect_graph_file_kind(nodes_file.filename)

        g = load_graph_from_bytes(b, kind=kind, nodes_bytes=nodes_bytes, nodes_kind=nodes_kind)
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
        res["has_node_attributes"] = g.nodes is not None
        return _make_download("graph-metrics.json", _as_json_bytes(res))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Graph metrics failed: {e}")


@app.post("/graph/degrees")
async def graph_degrees(
    edges_file: UploadFile = File(..., description="Edge list (CSV/JSON/.edge)"),
    nodes_file: UploadFile | None = File(None, description="Optional node attributes (CSV/JSON/.node)"),
):
    """Compute in-degree, out-degree, and total degree for all nodes."""
    try:
        from graph_tasks import degrees, merge_node_attributes
        b = await edges_file.read()
        kind = _detect_graph_file_kind(edges_file.filename)

        nodes_bytes = None
        nodes_kind = None
        if nodes_file:
            nodes_bytes = await nodes_file.read()
            nodes_kind = _detect_graph_file_kind(nodes_file.filename)

        g = load_graph_from_bytes(b, kind=kind, nodes_bytes=nodes_bytes, nodes_kind=nodes_kind)
        df = degrees(g)
        df = merge_node_attributes(df, g)
        result = {
            "n_nodes": g.n,
            "n_edges": int(g.edges.shape[0]),
            "degrees": df.to_dict(orient="records"),
            "has_node_attributes": g.nodes is not None,
        }
        return _make_download("degrees.json", _as_json_bytes(result))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Degree computation failed: {e}")


@app.post("/graph/pagerank")
async def graph_pagerank(
    edges_file: UploadFile = File(..., description="Edge list (CSV/JSON/.edge)"),
    nodes_file: UploadFile | None = File(None, description="Optional node attributes (CSV/JSON/.node)"),
    alpha: float = Query(0.85, description="Damping factor (0.0-1.0)"),
    iters: int = Query(40, description="Maximum iterations"),
    tol: float = Query(1e-6, description="Convergence tolerance"),
):
    """Compute PageRank scores for all nodes."""
    try:
        from graph_tasks import pagerank, merge_node_attributes
        b = await edges_file.read()
        kind = _detect_graph_file_kind(edges_file.filename)

        nodes_bytes = None
        nodes_kind = None
        if nodes_file:
            nodes_bytes = await nodes_file.read()
            nodes_kind = _detect_graph_file_kind(nodes_file.filename)

        g = load_graph_from_bytes(b, kind=kind, nodes_bytes=nodes_bytes, nodes_kind=nodes_kind)
        df = pagerank(g, alpha=alpha, iters=iters, tol=tol)
        df = merge_node_attributes(df, g)
        result = {
            "n_nodes": g.n,
            "n_edges": int(g.edges.shape[0]),
            "pagerank": df.to_dict(orient="records"),
            "parameters": {"alpha": alpha, "iters": iters, "tol": tol},
            "has_node_attributes": g.nodes is not None,
        }
        return _make_download("pagerank.json", _as_json_bytes(result))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PageRank computation failed: {e}")


@app.post("/graph/bfs")
async def graph_bfs(
    edges_file: UploadFile = File(..., description="Edge list (CSV/JSON/.edge)"),
    nodes_file: UploadFile | None = File(None, description="Optional node attributes (CSV/JSON/.node)"),
    source: str = Query(..., description="Source node ID for BFS"),
):
    """Compute breadth-first search distances from a source node."""
    try:
        from graph_tasks import bfs, merge_node_attributes
        b = await edges_file.read()
        kind = _detect_graph_file_kind(edges_file.filename)

        nodes_bytes = None
        nodes_kind = None
        if nodes_file:
            nodes_bytes = await nodes_file.read()
            nodes_kind = _detect_graph_file_kind(nodes_file.filename)

        g = load_graph_from_bytes(b, kind=kind, nodes_bytes=nodes_bytes, nodes_kind=nodes_kind)
        df = bfs(g, source)
        df = merge_node_attributes(df, g)
        result = {
            "n_nodes": g.n,
            "n_edges": int(g.edges.shape[0]),
            "source_node": source,
            "distances": df.to_dict(orient="records"),
            "has_node_attributes": g.nodes is not None,
        }
        return _make_download("bfs.json", _as_json_bytes(result))
    except KeyError as e:
        raise HTTPException(status_code=404, detail=f"Source node not found: {e}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"BFS computation failed: {e}")


@app.post("/graph/triangles")
async def graph_triangles(
    edges_file: UploadFile = File(..., description="Edge list (CSV/JSON/.edge) for undirected graph"),
    nodes_file: UploadFile | None = File(None, description="Optional node attributes (CSV/JSON/.node)"),
    max_nodes: int = Query(20000, description="Skip if graph has more nodes than this"),
):
    """Count triangles in an undirected graph."""
    try:
        from graph_tasks import triangles_undirected
        b = await edges_file.read()
        kind = _detect_graph_file_kind(edges_file.filename)

        nodes_bytes = None
        nodes_kind = None
        if nodes_file:
            nodes_bytes = await nodes_file.read()
            nodes_kind = _detect_graph_file_kind(nodes_file.filename)

        g = load_graph_from_bytes(b, kind=kind, nodes_bytes=nodes_bytes, nodes_kind=nodes_kind)
        tri_result = triangles_undirected(g, max_nodes=max_nodes)
        result = {
            "n_nodes": g.n,
            "n_edges": int(g.edges.shape[0]),
            "has_node_attributes": g.nodes is not None,
            **tri_result
        }
        return _make_download("triangles.json", _as_json_bytes(result))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Triangle counting failed: {e}")


@app.post("/graph/load")
async def graph_load(
    edges_file: UploadFile = File(..., description="Edge list (CSV/JSON/.edge)"),
    nodes_file: UploadFile | None = File(None, description="Optional node attributes (CSV/JSON/.node)"),
    compute_centrality: bool = Query(True, description="Compute centrality measures (PageRank, Betweenness, Eigenvector)"),
):
    """Load and validate a graph, returning basic statistics, visualization data, and centrality measures."""
    try:
        from graph_tasks import pagerank, betweenness_centrality, eigenvector_centrality

        b = await edges_file.read()
        kind = _detect_graph_file_kind(edges_file.filename)

        nodes_bytes = None
        nodes_kind = None
        if nodes_file:
            nodes_bytes = await nodes_file.read()
            nodes_kind = _detect_graph_file_kind(nodes_file.filename)

        g = load_graph_from_bytes(b, kind=kind, nodes_bytes=nodes_bytes, nodes_kind=nodes_kind)

        # Collect sample nodes
        sample_nodes = g.idx_to_id[:min(10, len(g.idx_to_id))]

        # Compute centrality measures if requested
        centrality_data = {}
        if compute_centrality:
            try:
                # PageRank
                pr_df = pagerank(g)
                centrality_data['pagerank'] = {row['node']: float(row['pr']) for _, row in pr_df.iterrows()}

                # Betweenness
                bc_df = betweenness_centrality(g)
                centrality_data['betweenness'] = {row['node']: float(row['centrality']) for _, row in bc_df.iterrows()}

                # Eigenvector
                ec_df = eigenvector_centrality(g)
                centrality_data['eigenvector'] = {row['node']: float(row['centrality']) for _, row in ec_df.iterrows()}

                # Degree (simple count)
                degree_dict = {}
                for node_id in g.idx_to_id:
                    degree_dict[node_id] = 0
                for _, row in g.edges.iterrows():
                    src = g.idx_to_id[int(row["src_idx"])]
                    tgt = g.idx_to_id[int(row["dst_idx"])]
                    degree_dict[src] = degree_dict.get(src, 0) + 1
                    degree_dict[tgt] = degree_dict.get(tgt, 0) + 1
                centrality_data['degree'] = degree_dict
            except Exception as e:
                print(f"[app] Centrality computation failed: {e}")
                centrality_data = {}

        # Prepare node list with attributes for visualization
        nodes_list = []
        for idx, node_id in enumerate(g.idx_to_id):
            node_data = {"id": node_id, "index": idx}
            if g.nodes is not None:
                # Add node attributes if available
                node_row = g.nodes[g.nodes["id"] == node_id]
                if not node_row.empty:
                    for col in g.nodes.columns:
                        if col not in ["id", "idx"]:
                            node_data[col] = _convert_to_json_serializable(node_row.iloc[0][col])
            nodes_list.append(node_data)

        # Prepare edge list for visualization
        edges_list = []
        for _, row in g.edges.iterrows():
            edge_data = {
                "source": g.idx_to_id[int(row["src_idx"])],
                "target": g.idx_to_id[int(row["dst_idx"])]
            }
            if "weight" in row:
                edge_data["weight"] = float(row["weight"])
            edges_list.append(edge_data)

        result = {
            "n_nodes": g.n,
            "n_edges": int(g.edges.shape[0]),
            "sample_nodes": sample_nodes,
            "has_graphblas": g.gb_matrix is not None,
            "has_node_attributes": g.nodes is not None,
            "edge_columns": list(g.edges.columns),
            "node_attributes": list(g.nodes.columns) if g.nodes is not None else [],
            "nodes": nodes_list,
            "edges": edges_list,
            "centrality": centrality_data if compute_centrality else {},
        }
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Graph loading failed: {e}")


@app.post("/graph/ego-network")
async def graph_ego_network(
    files: T.List[UploadFile] = File(..., description="Multiple graph files (.edges, .circles, .feat, .egofeat, .featnames, .node)"),
    ego_id: str | None = Query(None, description="Ego node ID (optional)"),
):
    """
    Load and visualize an ego network from multiple related files.
    Accepts any combination of: .edges, .circles, .feat, .egofeat, .featnames, .node
    """
    try:
        from graph_tasks import load_ego_network_from_files

        # Read all files and organize by extension
        files_dict = {}
        for file in files:
            filename = file.filename or ""
            content = await file.read()

            # Determine file type from extension
            if filename.endswith('.edges') or filename.endswith('.edge'):
                files_dict['edges'] = content
            elif filename.endswith('.circles'):
                files_dict['circles'] = content
            elif filename.endswith('.feat'):
                files_dict['feat'] = content
            elif filename.endswith('.egofeat'):
                files_dict['egofeat'] = content
            elif filename.endswith('.featnames'):
                files_dict['featnames'] = content
            elif filename.endswith('.node'):
                files_dict['node'] = content

        # Infer ego_id from filenames if not provided
        if not ego_id and files:
            # Try to extract from first filename (e.g., "0.edges" -> ego_id = "0")
            first_file = files[0].filename or ""
            ego_id = first_file.split('.')[0] if '.' in first_file else None

        # Load the ego network
        g = load_ego_network_from_files(files_dict, ego_id=ego_id)

        # Prepare visualization data
        sample_nodes = g.idx_to_id[:min(10, len(g.idx_to_id))]

        # Prepare node list with all attributes
        nodes_list = []
        for idx, node_id in enumerate(g.idx_to_id):
            node_data = {"id": str(node_id), "index": int(idx)}

            # Add node attributes if available
            if g.nodes is not None:
                node_row = g.nodes[g.nodes["id"] == node_id]
                if not node_row.empty:
                    for col in g.nodes.columns:
                        if col not in ["id", "idx"]:
                            val = node_row.iloc[0][col]
                            node_data[col] = _convert_to_json_serializable(val)

            # Add circle membership if available
            if g.circles is not None:
                node_circles = g.circles[g.circles["node"] == node_id]["circle"].tolist()
                if node_circles:
                    node_data["circles"] = [str(c) for c in node_circles]

            nodes_list.append(node_data)

        # Prepare edge list for visualization
        edges_list = []
        for _, row in g.edges.iterrows():
            edge_data = {
                "source": str(g.idx_to_id[int(row["src_idx"])]),
                "target": str(g.idx_to_id[int(row["dst_idx"])])
            }
            if "weight" in row:
                edge_data["weight"] = float(row["weight"])
            edges_list.append(edge_data)

        # Get circle information
        circles_info = None
        if g.circles is not None:
            circles_raw = g.circles.groupby("circle")["node"].apply(list).to_dict()
            # Convert to JSON-serializable format
            circles_info = {str(k): [str(node) for node in v] for k, v in circles_raw.items()}

        # Compute centrality measures
        centrality_data = {}
        try:
            from graph_tasks import pagerank, betweenness_centrality, eigenvector_centrality

            # PageRank
            pr_df = pagerank(g)
            centrality_data['pagerank'] = {row['node']: float(row['pr']) for _, row in pr_df.iterrows()}

            # Betweenness
            bc_df = betweenness_centrality(g)
            centrality_data['betweenness'] = {row['node']: float(row['centrality']) for _, row in bc_df.iterrows()}

            # Eigenvector
            ec_df = eigenvector_centrality(g)
            centrality_data['eigenvector'] = {row['node']: float(row['centrality']) for _, row in ec_df.iterrows()}

            # Degree (simple count)
            degree_dict = {}
            for node_id in g.idx_to_id:
                degree_dict[str(node_id)] = 0
            for _, row in g.edges.iterrows():
                src = str(g.idx_to_id[int(row["src_idx"])])
                tgt = str(g.idx_to_id[int(row["dst_idx"])])
                degree_dict[src] = degree_dict.get(src, 0) + 1
                degree_dict[tgt] = degree_dict.get(tgt, 0) + 1
            centrality_data['degree'] = degree_dict
        except Exception as e:
            print(f"[app] Ego network centrality computation failed: {e}")
            centrality_data = {}

        result = {
            "n_nodes": int(g.n),
            "n_edges": int(g.edges.shape[0]),
            "sample_nodes": [str(n) for n in sample_nodes],
            "ego_id": str(g.ego_id) if g.ego_id else None,
            "has_circles": g.circles is not None,
            "has_features": g.features is not None,
            "has_featnames": g.featnames is not None,
            "has_ego_features": g.ego_features is not None,
            "circles": circles_info,
            "feature_count": int(len(g.features.columns) - 1) if g.features is not None else 0,
            "nodes": nodes_list,
            "edges": edges_list,
            "centrality": centrality_data,
        }
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ego network loading failed: {e}")


# ------------------------------ Audio Anti-Spoofing ------------------------ #

@app.get("/audio/model-status")
async def audio_model_status():
    """Check if audio anti-spoofing models are available"""
    try:
        from audio_antispoofing import check_models_available
        status = check_models_available()
        model_ready = all(status.values())
        return JSONResponse(content={
            "model_ready": model_ready,
            "details": status
        })
    except Exception as e:
        return JSONResponse(content={
            "model_ready": False,
            "error": str(e)
        })


@app.post("/audio/download-model")
async def audio_download_model():
    """Download required models for audio anti-spoofing"""
    try:
        from audio_antispoofing import download_models
        download_models()
        return JSONResponse(content={"status": "success", "message": "Models downloaded"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model download failed: {e}")


@app.post("/audio/analyze")
async def audio_analyze(file: UploadFile = File(...)):
    """Analyze audio file for deepfake detection"""
    import tempfile
    import shutil
    from pathlib import Path

    # Validate file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ['.flac', '.wav']:
        raise HTTPException(status_code=400, detail="Only FLAC and WAV files are supported")

    # Save uploaded file to temp location
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        # Run prediction
        from audio_antispoofing import predict_audio
        prediction, confidence, score = predict_audio(tmp_path)

        # Clean up temp file
        Path(tmp_path).unlink()

        return JSONResponse(content={
            "prediction": prediction,
            "confidence": float(confidence),
            "score": float(score),
            "filename": file.filename
        })

    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail="Model files not found. Please download models first.")
    except Exception as e:
        # Clean up temp file if it exists
        if 'tmp_path' in locals():
            try:
                Path(tmp_path).unlink()
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Audio analysis failed: {e}")
