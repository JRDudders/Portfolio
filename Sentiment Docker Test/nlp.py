# nlp.py — model presets + unified runner (sentiment, zero-shot, NER) with chunking
from __future__ import annotations
import os, re
from typing import Any, Dict, List, Optional, Tuple
from transformers import pipeline

# Defaults (env-tunable)
MODEL_ID = os.getenv("MODEL_ID", "cardiffnlp/twitter-roberta-base-sentiment-latest")
MODEL_TASK = os.getenv("MODEL_TASK", "text-classification")  # "text-classification" | "token-classification" | "zero-shot-classification"
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))
MAX_LEN = int(os.getenv("MAX_LEN", "256"))

# Optional NLTK sentence splitter; we still run without it
try:
    from nltk.tokenize import sent_tokenize
    _HAS_NLTK = True
except Exception:
    sent_tokenize = None
    _HAS_NLTK = False

# Presets you can select via ?preset=... or JSON {"preset": "..."}
MODEL_PRESETS: Dict[str, Tuple[str, str, Dict]] = {
    # Sentiment / emotion / toxicity
    "sentiment-twitter": ("text-classification", "cardiffnlp/twitter-roberta-base-sentiment-latest", {}),
    "sentiment-sst2":    ("text-classification", "distilbert-base-uncased-finetuned-sst-2-english", {}),
    "emotion":           ("text-classification", "j-hartmann/emotion-english-distilroberta-base", {}),
    "offensive-twitter": ("text-classification", "cardiffnlp/twitter-roberta-base-offensive", {}),
    "toxic-roberta":     ("text-classification", "unitary/unbiased-toxic-roberta", {}),

    # NER
    "ner-conll":         ("token-classification", "dslim/bert-base-NER", {"aggregation_strategy": "simple"}),
    "ner-multi":         ("token-classification", "Davlan/bert-base-multilingual-cased-ner-hrl", {"aggregation_strategy": "simple"}),

    # Zero-shot “topic-ish”
    "zeroshot-bart":     ("zero-shot-classification", "facebook/bart-large-mnli", {}),
    "zeroshot-xnli":     ("zero-shot-classification", "joeddav/xlm-roberta-large-xnli", {}),
}

# Fallback taxonomy for zero-shot when auto-labeling finds little
DEFAULT_ZS_LABELS = [
    "politics","technology","business","finance","science","health","sports",
    "entertainment","education","environment","world","usa","opinion","culture",
    "travel","food","art","music","gaming","real estate","autos"
]

# Cache of pipelines by (task, model_id)
_PIPES: Dict[Tuple[str, str], Any] = {}

def get_pipe(task: Optional[str] = None, model_id: Optional[str] = None, **kwargs):
    task = task or MODEL_TASK
    model_id = model_id or MODEL_ID
    key = (task, model_id)
    if key not in _PIPES:
        # put batch_size in the constructor; not in the call
        _PIPES[key] = pipeline(task, model=model_id, batch_size=BATCH_SIZE, **kwargs)
    return _PIPES[key]

def preprocess_text(s: str) -> str:
    # Tweet-style normalization; safe for generic text classification
    out = []
    for t in s.split(" "):
        t = "@user" if t.startswith("@") and len(t) > 1 else t
        t = "http" if t.startswith("http") else t
        out.append(t)
    return " ".join(out)

def preprocess_for_task(s: str, task: str) -> str:
    # For NER, keep raw text to preserve offsets
    return s if task == "token-classification" else preprocess_text(s)

# Chunking controls
CHAR_LEN_THRESHOLD = int(os.getenv("CHAR_LEN_THRESHOLD", "1500"))     # when to start chunking long paragraphs
TARGET_CHUNK_CHARS = int(os.getenv("TARGET_CHUNK_CHARS", "800"))      # size target per chunk

def _sentences(text: str) -> List[str]:
    if _HAS_NLTK and sent_tokenize:
        try:
            sents = sent_tokenize(text)
            if sents:
                return sents
        except Exception:
            pass
    # fallback: regex split on sentence-ish boundaries
    parts = re.split(r"(?<=[.!?])\s+|\n{2,}", text)
    return [p for p in parts if p.strip()]

def _chunk_by_sentences(text: str, target_chars: int = TARGET_CHUNK_CHARS) -> List[str]:
    sents = _sentences(text)
    chunks: List[str] = []
    cur: List[str] = []
    n = 0
    for s in sents:
        if n + len(s) > target_chars and cur:
            chunks.append(" ".join(cur)); cur = [s]; n = len(s)
        else:
            cur.append(s); n += len(s)
    if cur:
        chunks.append(" ".join(cur))
    return chunks or [text]

def _avg_dicts(dicts: List[Dict[str, float]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    cnt = 0
    for d in dicts:
        for k, v in d.items():
            out[k] = out.get(k, 0.0) + float(v)
        cnt += 1
    if cnt:
        for k in list(out.keys()):
            out[k] /= cnt
    return out

def run_task(
    texts: List[str],
    *,
    task: Optional[str] = None,
    model_id: Optional[str] = None,
    preset: Optional[str] = None,
    labels: Optional[List[str]] = None,  # for zero-shot
) -> List[Dict[str, Any]]:
    """
    Uniform output:
    - text/zero-shot: {"scores": {label: p}, "top": {"label": lab, "score": p}}
    - token-classification (NER): {"entities": [{text,label,start,end,score}, ...]}
    """
    if preset:
        task, model_id, extra = MODEL_PRESETS[preset]
    else:
        task = task or MODEL_TASK
        model_id = model_id or MODEL_ID
        extra = {}

    if task in (None, "text-classification"):
        pipe = get_pipe("text-classification", model_id, **extra)
        out: List[Dict[str, Any]] = []
        for txt in texts:
            parts = _chunk_by_sentences(txt, TARGET_CHUNK_CHARS) if len(txt) > CHAR_LEN_THRESHOLD else [txt]
            preds = pipe(
                parts,
                truncation=True,
                padding=True,
                top_k=None,
                return_all_scores=True,
                max_length=MAX_LEN,
            )
            if isinstance(preds, dict):
                preds = [preds]
            m_list = [{d["label"]: float(d["score"]) for d in scores} for scores in preds]
            agg = _avg_dicts(m_list) if m_list else {}
            lab, sc = max(agg.items(), key=lambda kv: kv[1]) if agg else ("", 0.0)
            out.append({"scores": agg, "top": {"label": lab, "score": sc}})
        return out

    if task == "zero-shot-classification":
        if not labels:
            raise ValueError("zero-shot-classification requires 'labels'.")
        pipe = get_pipe("zero-shot-classification", model_id, **extra)
        model_max = int(getattr(pipe.tokenizer, "model_max_length", 512) or 512)
        max_len = min(MAX_LEN, model_max)

        out: List[Dict[str, Any]] = []
        for txt in texts:
            parts = _chunk_by_sentences(txt, TARGET_CHUNK_CHARS) if len(txt) > CHAR_LEN_THRESHOLD else [txt]
            part_scores: List[Dict[str, float]] = []
            for chunk in parts:
                r = pipe(
                    chunk,
                    candidate_labels=labels,
                    multi_label=True,
                    truncation=True,
                    padding=True,
                    max_length=max_len,
                )
                if isinstance(r, dict):
                    r = [r]
                for rr in r:
                    part_scores.append(dict(zip(rr["labels"], [float(x) for x in rr["scores"]])))
            agg = _avg_dicts(part_scores) if part_scores else {}
            lab, sc = max(agg.items(), key=lambda kv: kv[1]) if agg else ("", 0.0)
            out.append({"scores": agg, "top": {"label": lab, "score": sc}})
        return out

    if task == "token-classification":
        pipe = get_pipe("token-classification", model_id, **extra)
        out: List[Dict[str, Any]] = []
        for txt in texts:
            ents = pipe(txt)
            items = []
            for e in ents:
                items.append({
                    "text": e.get("word") or e.get("entity_group") or "",
                    "label": e.get("entity_group") or e.get("entity") or "",
                    "start": int(e.get("start", 0)),
                    "end": int(e.get("end", 0)),
                    "score": float(e.get("score", 0.0)),
                })
            out.append({"entities": items})
        return out

    raise ValueError(f"Unsupported task: {task}")
