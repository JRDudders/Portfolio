# apputils.py — CSV/JSON processors aware of task/preset/labels
from __future__ import annotations
import io, json
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

from nlp import preprocess_for_task, run_task

def filename_with_suffix(name: str, suffix: str) -> str:
    stem = Path(name).stem or "output"
    return f"{stem}_scored.{suffix}"

def _truncate_text(text: str, max_length: int = 500) -> str:
    """Truncate text to max_length characters, adding ellipsis if truncated"""
    if len(text) <= max_length:
        return text
    return text[:max_length].rstrip() + "..."

def _pick_text_column(df: pd.DataFrame) -> str:
    for c in df.columns:
        if c.lower() in {"text", "tweet", "content"}:
            return c
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]):
            return c
    raise ValueError("No suitable text column found in CSV")

def process_csv_bytes(csv_bytes: bytes, *, task: Optional[str] = None,
                      preset: Optional[str] = None, labels: Optional[List[str]] = None) -> bytes:
    try:
        df = pd.read_csv(io.BytesIO(csv_bytes))
    except UnicodeDecodeError:
        df = pd.read_csv(io.BytesIO(csv_bytes), encoding="latin-1")
    col = _pick_text_column(df)
    texts = df[col].astype(str).tolist()
    proc = [preprocess_for_task(t, task or "text-classification") for t in texts]
    preds = run_task(proc, task=task, preset=preset, labels=labels)

    if (task == "token-classification") or (preset and "ner" in preset):
        df["entities_json"] = [json.dumps(p.get("entities", []), ensure_ascii=False) for p in preds]
    else:
        all_labels = set()
        for p in preds:
            all_labels.update(p.get("topics", {}).keys())
        for lab in sorted(all_labels):
            df[f"score_{lab}"] = [p.get("topics", {}).get(lab, float("nan")) for p in preds]
        # Calculate top label and score from topics dict
        def get_top_label(topics):
            if not topics:
                return ""
            return max(topics.items(), key=lambda x: x[1])[0]
        def get_top_score(topics):
            if not topics:
                return float("nan")
            return max(topics.values())
        df["top_label"] = [get_top_label(p.get("topics", {})) for p in preds]
        df["top_score"] = [get_top_score(p.get("topics", {})) for p in preds]

    return df.to_csv(index=False).encode("utf-8")

def process_json_bytes(json_bytes: bytes, *, task: Optional[str] = None,
                       preset: Optional[str] = None, labels: Optional[List[str]] = None) -> bytes:
    data = json.loads(json_bytes.decode("utf-8"))

    # List of strings
    if isinstance(data, list) and (not data or isinstance(data[0], str)):
        texts = [str(x) for x in data]
        proc = [preprocess_for_task(t, task or "text-classification") for t in texts]
        preds = run_task(proc, task=task, preset=preset, labels=labels)
        if (task == "token-classification") or (preset and "ner" in preset):
            out = [{"text (analyzed in full, truncated for display)": _truncate_text(t), "entities": p.get("entities", [])} for t, p in zip(texts, preds)]
        else:
            out = []
            for t, p in zip(texts, preds):
                result = {"text (analyzed in full, truncated for display)": _truncate_text(t)}
                result.update(p)  # Add topics dict
                out.append(result)
        return json.dumps(out, ensure_ascii=False, indent=2).encode("utf-8")

    # List of objects with text-ish key
    if isinstance(data, list) and data and isinstance(data[0], dict):
        key = next((k for k in ("text", "tweet", "content") if k in data[0]), None)
        if key is None:
            raise ValueError("JSON objects must include a 'text'/'tweet'/'content' field")
        texts = [str(obj.get(key, "")) for obj in data]
        proc = [preprocess_for_task(t, task or "text-classification") for t in texts]
        preds = run_task(proc, task=task, preset=preset, labels=labels)
        out = []
        for obj, p in zip(data, preds):
            new_obj = dict(obj)
            # Remove original text field and add truncated version
            original_text = new_obj.pop(key, "")
            new_obj["text (analyzed in full, truncated for display)"] = _truncate_text(original_text)
            if (task == "token-classification") or (preset and "ner" in preset):
                new_obj["entities"] = p.get("entities", [])
            else:
                # Add topics dict directly
                new_obj.update(p)
            out.append(new_obj)
        return json.dumps(out, ensure_ascii=False, indent=2).encode("utf-8")

    raise ValueError("JSON must be a list of strings or list of objects")
