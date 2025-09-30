# nlp.py — model + preprocessing + batched inference
from __future__ import annotations
from typing import Any, Dict, List, Tuple
import os

from transformers import AutoConfig, pipeline

MODEL_ID = os.getenv("MODEL_ID", "cardiffnlp/twitter-roberta-base-sentiment-latest")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))
MAX_LEN = int(os.getenv("MAX_LEN", "256"))

_PIPE = None
_CFG = None

def get_pipe():
    global _PIPE, _CFG
    if _PIPE is None:
        _PIPE = pipeline("text-classification", model=MODEL_ID)
        _CFG = AutoConfig.from_pretrained(MODEL_ID)
    return _PIPE, _CFG

def preprocess_text(s: str) -> str:
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
        chunk = texts[i:i + BATCH_SIZE]
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
