# app.py
# Headless FastAPI service: CSV/JSON -> sentiment -> same type out (download)
# No tkinter, no templates folder. Serves ./index.html directly if present.

from typing import Annotated, Dict, Any, List, Optional, Tuple
from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

import io, os, json, shutil
from pathlib import Path

import requests
import pandas as pd
from transformers import pipeline, AutoConfig

# ----------------------- Config -----------------------

MODEL_ID = os.getenv("MODEL_ID", "cardiffnlp/twitter-roberta-base-sentiment-latest")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))
MAX_LEN = int(os.getenv("MAX_LEN", "256"))
HTTP_TIMEOUT = (10, 120)
MAX_DOWNLOAD_BYTES = 100 * 1024 * 1024

app = FastAPI(title="Tweet Sentiment Service", version="1.0")

# Optional: serve /static if a directory exists
if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Lazy pipeline
_PIPE = None
_CFG = None
def get_pipe():
    global _PIPE, _CFG
    if _PIPE is None:
        _PIPE = pipeline("text-classification", model=MODEL_ID)
        _CFG = AutoConfig.from_pretrained(MODEL_ID)
    return _PIPE, _CFG

# ----------------------- NLP helpers -----------------------

def preprocess_tweet(s: str) -> str:
    out = []
    for t in s.split(" "):
        t = "@user" if t.startswith("@") and len(t) > 1 else t
        t = "http" if t.startswith("http") else t
        out.append(t)
    return " ".join(out)

def classify_texts(texts: List[str]) -> List[Dict[str, Any]]:
    pipe, _ = get_pipe()
    results: List[Dict[str, Any]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        chunk = texts[i:i+BATCH_SIZE]
        preds = pipe(
            chunk,
            truncation=True, padding=True,
            top_k=None, return_all_scores=True,
            max_length=MAX_LEN, batch_size=BATCH_SIZE,
        )
        for scores in preds:
            m = {d["label"]: float(d["score"]) for d in scores}
            lab, sc = max(m.items(), key=lambda kv: kv[1]) if m else ("", 0.0)
            results.append({"scores": m, "top": {"label": lab, "score": sc}})
    return results

# ----------------------- I/O helpers -----------------------

def _infer_kind_from_headers(url: str, headers: Dict[str, str]) -> Optional[str]:
    ct = (headers.get("Content-Type") or "").split(";")[0].strip().lower()
    if ct in {"application/json", "text/json"}: return "json"
    if ct in {"text/csv", "application/csv"}:   return "csv"
    if url.lower().endswith(".json"):           return "json"
    if url.lower().endswith(".csv"):            return "csv"
    return None

def _fetch_url_bytes(url: str) -> Tuple[bytes, str]:
    with requests.get(url, stream=True, timeout=HTTP_TIMEOUT) as r:
        r.raise_for_status()
        kind = _infer_kind_from_headers(url, r.headers)
        buf = bytearray()
        for chunk in r.iter_content(8192):
            if chunk:
                buf += chunk
                if len(buf) > MAX_DOWNLOAD_BYTES:
                    raise HTTPException(413, detail="Download too large (>100MB)")
        if kind is None:
            head = bytes(buf[:2048]).lstrip()
            kind = "json" if head.startswith((b"{", b"[")) else "csv"
        return bytes(buf), kind

def _pick_text_column(df: pd.DataFrame) -> str:
    for c in df.columns:
        if c.lower() in {"text", "tweet", "content"}:
            return c
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]):
            return c
    raise HTTPException(400, detail="No suitable text column found in CSV")

def _process_csv_bytes(csv_bytes: bytes) -> bytes:
    try:
        df = pd.read_csv(io.BytesIO(csv_bytes))
    except UnicodeDecodeError:
        df = pd.read_csv(io.BytesIO(csv_bytes), encoding="latin-1")
    text_col = _pick_text_column(df)
    texts = df[text_col].astype(str).map(preprocess_tweet).tolist()
    preds = classify_texts(texts)

    all_labels = set()
    for p in preds: all_labels.update(p["scores"].keys())
    for lab in sorted(all_labels):
        df[f"score_{lab}"] = [p["scores"].get(lab, float("nan")) for p in preds]
    df["top_label"] = [p["top"]["label"] for p in preds]
    df["top_score"]  = [p["top"]["score"]  for p in preds]

    return df.to_csv(index=False).encode("utf-8")

def _process_json_bytes(json_bytes: bytes) -> bytes:
    data = json.loads(json_bytes.decode("utf-8"))
    if isinstance(data, list) and (not data or isinstance(data[0], str)):
        texts = [preprocess_tweet(x) for x in data]
        preds = classify_texts(texts)
        out = [{"text": t, "scores": p["scores"], "top": p["top"]} for t, p in zip(data, preds)]
        return json.dumps(out, ensure_ascii=False, indent=2).encode("utf-8")
    if isinstance(data, list) and isinstance(data[0], dict):
        key = next((k for k in ("text","tweet","content") if k in data[0]), None)
        if key is None:
            raise HTTPException(400, detail="JSON objects must include a 'text'/'tweet'/'content' field")
        texts = [preprocess_tweet(str(obj.get(key, ""))) for obj in data]
        preds = classify_texts(texts)
        out = []
        for obj, p in zip(data, preds):
            new_obj = dict(obj)
            new_obj["scores"] = p["scores"]
            new_obj["top"] = p["top"]
            out.append(new_obj)
        return json.dumps(out, ensure_ascii=False, indent=2).encode("utf-8")
    raise HTTPException(400, detail="JSON must be a list of strings or list of objects")

def _filename_with_suffix(name: str, suffix: str) -> str:
    stem = Path(name).stem or "output"
    return f"{stem}_scored.{suffix}"

# ----------------------- Endpoints -----------------------

@app.get("/healthz")
def healthz():
    return {"status": "ok", "model": MODEL_ID}

@app.get("/", response_class=HTMLResponse)
def index(_: Request):
    p = Path("index.html")
    if p.exists():
        return HTMLResponse(p.read_text(encoding="utf-8"))
    return HTMLResponse(
        "<h3>Service is up</h3><p>POST <code>/predict/file</code> (multipart) or "
        "POST <code>/predict/url</code> with <code>{\"url\": \"https://...csv|json\"}</code>.</p>",
        status_code=200
    )

@app.post("/predict/file")
async def predict_file(file: Annotated[UploadFile, File(...)]):
    buf = io.BytesIO()
    await file.seek(0)
    shutil.copyfileobj(file.file, buf)
    raw = buf.getvalue()

    fname = file.filename or "input"
    if fname.lower().endswith(".csv") or file.content_type in {"text/csv", "application/csv"}:
        out_bytes = _process_csv_bytes(raw)
        out_name = _filename_with_suffix(fname, "csv")
        media = "text/csv; charset=utf-8"
    elif fname.lower().endswith(".json") or file.content_type in {"application/json", "text/json"}:
        out_bytes = _process_json_bytes(raw)
        out_name = _filename_with_suffix(fname, "json")
        media = "application/json; charset=utf-8"
    else:
        raise HTTPException(400, detail="Please upload a .csv or .json")

    return StreamingResponse(io.BytesIO(out_bytes), media_type=media,
                             headers={"Content-Disposition": f'attachment; filename="{out_name}"'})

@app.post("/predict/url")
async def predict_url(request: Request):
    body = await request.json()
    url = body.get("url")
    if not url:
        raise HTTPException(400, detail="Body must include {'url': '<http(s)://...>'}")
    raw, kind = _fetch_url_bytes(url)

    if kind == "csv":
        out_bytes = _process_csv_bytes(raw)
        media = "text/csv; charset=utf-8"
        out_name = _filename_with_suffix(Path(url).name or "input.csv", "csv")
    else:
        out_bytes = _process_json_bytes(raw)
        media = "application/json; charset=utf-8"
        out_name = _filename_with_suffix(Path(url).name or "input.json", "json")

    return StreamingResponse(io.BytesIO(out_bytes), media_type=media,
                             headers={"Content-Disposition": f'attachment; filename="{out_name}"'})
