# apputils.py — small shared helpers used by adapters/app
from __future__ import annotations
import io, json
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

def filename_with_suffix(name: str, suffix: str) -> str:
    stem = Path(name).stem or "output"
    return f"{stem}_scored.{suffix}"

def _pick_text_column(df: pd.DataFrame) -> str:
    for c in df.columns:
        if c.lower() in {"text","tweet","content"}: return c
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]): return c
    raise ValueError("No suitable text column found in CSV")

def process_csv_bytes(csv_bytes: bytes) -> bytes:
    try:
        df = pd.read_csv(io.BytesIO(csv_bytes))
    except UnicodeDecodeError:
        df = pd.read_csv(io.BytesIO(csv_bytes), encoding="latin-1")
    col = _pick_text_column(df)
    from nlp import preprocess_text, classify_texts
    texts = df[col].astype(str).map(preprocess_text).tolist()
    preds = classify_texts(texts)

    all_labels = set()
    for p in preds: all_labels.update(p["scores"].keys())
    for lab in sorted(all_labels):
        df[f"score_{lab}"] = [p["scores"].get(lab, float("nan")) for p in preds]
    df["top_label"] = [p["top"]["label"] for p in preds]
    df["top_score"]  = [p["top"]["score"] for p in preds]
    return df.to_csv(index=False).encode("utf-8")

def process_json_bytes(json_bytes: bytes) -> bytes:
    data = json.loads(json_bytes.decode("utf-8"))
    from nlp import preprocess_text, classify_texts
    if isinstance(data, list) and (not data or isinstance(data[0], str)):
        texts = [preprocess_text(x) for x in data]
        preds = classify_texts(texts)
        out = [{"text": t, "scores": p["scores"], "top": p["top"]} for t, p in zip(data, preds)]
        return json.dumps(out, ensure_ascii=False, indent=2).encode("utf-8")
    if isinstance(data, list) and isinstance(data[0], dict):
        key = next((k for k in ("text","tweet","content") if k in data[0]), None)
        if key is None: raise ValueError("JSON objects must include a 'text'/'tweet'/'content' field")
        texts = [preprocess_text(str(obj.get(key, ""))) for obj in data]
        preds = classify_texts(texts)
        out = []
        for obj, p in zip(data, preds):
            new_obj = dict(obj); new_obj["scores"] = p["scores"]; new_obj["top"] = p["top"]; out.append(new_obj)
        return json.dumps(out, ensure_ascii=False, indent=2).encode("utf-8")
    raise ValueError("JSON must be a list of strings or list of objects")
