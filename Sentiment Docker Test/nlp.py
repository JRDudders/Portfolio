# nlp.py
from __future__ import annotations
"""
Unified NLP dispatcher for the FastAPI service.

Supports:
- Transformers (HF): sentiment/text-classification, zero-shot, NER (token-classification)
- spaCy tasks: NER, POS/DEP/LEMMA, sentence segmentation
- Stanza tasks: POS/DEP/LEMMA, sentence segmentation
- Sentence-Transformers: embeddings (+ BERTopic via sbert_tasks)
- Topic modeling (NMF / KMeans) via topics.py

Design:
- PRESETS: name -> (task, model_id, kwargs)
- build_hf_pipeline(): LRU-cached HF pipelines
- run_task(): single entry point (texts: list[str]) -> list[dict] (or list[list[dict]] for NER)

Notes:
- Long text is chunked for *classification* tasks to avoid max-length issues, then averaged.
- NER is NOT chunked to preserve spans.
"""

from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import os
import math
import re

# ------------------------ Defaults & Presets -------------------------------- #

# Default model/task used by /healthz and when callers omit both
MODEL_TASK = "text-classification"
MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Reasonable default label set for zero-shot if the caller provides none
DEFAULT_ZS_LABELS: List[str] = [
    "politics", "economy", "health", "science", "technology",
    "sports", "entertainment", "climate", "crime", "education",
    "misinformation", "opinion",
]

# Max tokens for single pass; we chunk roughly by words (not tokenizer-perfect),
# but good enough to avoid 512/1024 overflow issues in common encoders.
_CLASSIFY_MAX_WORDS = int(os.getenv("CLASSIFY_MAX_WORDS", "320"))  # ~ <= 512 tokens ballpark


# ------------------------ External Task Modules ----------------------------- #

# Optional modules are imported lazily inside dispatch to avoid import cost at app start.
# Just keep the names here for type hints / readability.

# spaCy tasks
# - spacy_tasks.spacy_ner(texts)
# - spacy_tasks.spacy_pos_dep_lemma(texts)
# - spacy_tasks.spacy_sentences(texts)

# Stanza tasks
# - stanza_tasks.stanza_pos_dep(texts, lang="en")
# - stanza_tasks.stanza_sentences(texts, lang="en")

# Sentence-Transformers tasks
# - sbert_tasks.sbert_embeddings(texts, model="all-MiniLM-L6-v2")
# - sbert_tasks.sbert_topics(texts, model="all-MiniLM-L6-v2")

# Topics (tf-idf based)
# - topics.topics_nmf(texts, n_topics=..)
# - topics.topics_kmeans(texts, n_clusters=. .)


# ---------------------------- HF Pipelines ---------------------------------- #

@lru_cache(maxsize=16)
def _hf_pipeline_cache(task: str, model_id: str, key: str = ""):
    """
    Cache HF pipeline objects. `key` encodes kwargs that affect pipeline creation.
    """
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification

    if task in {"text-classification", "sentiment-analysis", "zero-shot-classification"}:
        tok = AutoTokenizer.from_pretrained(model_id)
        mdl = AutoModelForSequenceClassification.from_pretrained(model_id)
        return pipeline("text-classification" if task != "zero-shot-classification" else "zero-shot-classification",
                        model=mdl, tokenizer=tok, device=-1)

    if task == "token-classification":
        tok = AutoTokenizer.from_pretrained(model_id)
        mdl = AutoModelForTokenClassification.from_pretrained(model_id)
        # aggregation handled at call-time via kwargs
        return pipeline("token-classification", model=mdl, tokenizer=tok, device=-1)

    raise ValueError(f"Unsupported HF task: {task}")


def _avg_scores(label_sets: List[Dict[str, float]]) -> Dict[str, float]:
    # Mean of per-chunk probabilities for each label
    if not label_sets:
        return {}
    acc: Dict[str, float] = {}
    for d in label_sets:
        for k, v in d.items():
            acc[k] = acc.get(k, 0.0) + float(v)
    n = float(len(label_sets))
    return {k: v / n for k, v in acc.items()}


def _words(text: str) -> List[str]:
    # crude whitespace split; avoids tokenizer import here
    return re.findall(r"\S+", text or "")


def _chunks_for_classification(text: str, max_words: int) -> List[str]:
    ws = _words(text)
    if len(ws) <= max_words:
        return [text]
    # chunk by words; keep boundaries roughly sentence-ish where possible
    chunks: List[str] = []
    start = 0
    while start < len(ws):
        end = min(len(ws), start + max_words)
        chunk = " ".join(ws[start:end])
        chunks.append(chunk)
        start = end
    return chunks


def _hf_text_classification(
    texts: List[str],
    *,
    model_id: str,
    multi_label: bool = False,
) -> List[Dict[str, Any]]:
    """
    Sentiment / general text classification.
    For long texts, chunk and average label probs over chunks.
    """
    task = "text-classification"
    pipe = _hf_pipeline_cache(task, model_id, key=f"multi={multi_label}")

    results: List[Dict[str, Any]] = []
    for t in texts:
        # run by chunks to avoid length errors
        chunks = _chunks_for_classification(t, _CLASSIFY_MAX_WORDS)
        # First run to see label space (for fixed-label heads we get one label per class on HF side)
        out_per_chunk = pipe(chunks, truncation=True, padding=True, top_k=None if multi_label else 2)
        # Normalize output to dict of label->score for averaging
        label_sets: List[Dict[str, float]] = []
        for out in out_per_chunk:
            if isinstance(out, dict) and "label" in out:
                # Some pipelines return single best label; ask for top_k next time if needed
                label_sets.append({out["label"]: float(out["score"])})
            else:
                # Usually a list of {label, score}
                if isinstance(out, list):
                    label_sets.append({e["label"]: float(e["score"]) for e in out})
                else:
                    # unexpected; fallback: wrap
                    label_sets.append({})
        avg = _avg_scores(label_sets)
        # return topic->score mapping, sorted by score descending
        sorted_items = sorted(avg.items(), key=lambda kv: kv[1], reverse=True)
        results.append({
            "topics": {label: float(score) for label, score in sorted_items}
        })
    return results


def _hf_zero_shot(
    texts: List[str],
    *,
    model_id: str,
    candidate_labels: Optional[List[str]],
    multi_label: bool = True,
    hypothesis_template: str = "This text is about {}.",
) -> List[Dict[str, Any]]:
    task = "zero-shot-classification"
    pipe = _hf_pipeline_cache(task, model_id)

    labels = candidate_labels or DEFAULT_ZS_LABELS
    # Chunk long docs; run per chunk and average per-label probabilities
    results: List[Dict[str, Any]] = []
    for t in texts:
        chunks = _chunks_for_classification(t, _CLASSIFY_MAX_WORDS)
        per_label_probs: List[Dict[str, float]] = []

        for ch in chunks:
            out = pipe(
                ch,
                candidate_labels=labels,
                multi_label=multi_label,
                hypothesis_template=hypothesis_template,
            )
            # HF zero-shot returns dict with 'labels' and 'scores'
            per_label_probs.append({lbl: float(scr) for lbl, scr in zip(out["labels"], out["scores"])})

        avg = _avg_scores(per_label_probs)
        ordered = sorted(avg.items(), key=lambda kv: kv[1], reverse=True)
        results.append({"topics": {label: float(score) for label, score in ordered}})
    return results


def _hf_stance_detection(
    texts: List[str],
    *,
    model_id: str,
    claim: str,
    hypothesis_template: str = "{}",
) -> List[Dict[str, Any]]:
    """
    NLI-based stance detection (arxiv:2305.01723).

    Uses Natural Language Inference to classify stance:
    - ENTAILMENT -> SUPPORT
    - CONTRADICTION -> OPPOSE
    - NEUTRAL -> NEUTRAL

    Args:
        texts: List of texts to classify
        model_id: NLI model (e.g., microsoft/deberta-v3-base-mnli)
        claim: The claim/hypothesis to classify stance towards
        hypothesis_template: Template for formatting hypothesis (default: "{}")

    Returns:
        List of dicts with stance labels and confidence scores
    """
    task = "zero-shot-classification"
    pipe = _hf_pipeline_cache(task, model_id)

    # NLI uses entailment/contradiction/neutral labels
    nli_labels = ["entailment", "contradiction", "neutral"]

    # Format the hypothesis using template
    hypothesis = hypothesis_template.format(claim)

    results: List[Dict[str, Any]] = []
    for t in texts:
        # Chunk long texts to avoid length errors
        chunks = _chunks_for_classification(t, _CLASSIFY_MAX_WORDS)
        per_label_probs: List[Dict[str, float]] = []

        for ch in chunks:
            # Run NLI classification
            out = pipe(
                ch,
                candidate_labels=nli_labels,
                multi_label=False,  # Single stance per text
                hypothesis_template=hypothesis,
            )
            # Collect probabilities for each NLI label
            per_label_probs.append({
                lbl: float(scr)
                for lbl, scr in zip(out["labels"], out["scores"])
            })

        # Average probabilities across chunks
        avg = _avg_scores(per_label_probs)

        # Map NLI labels to stance labels
        stance_map = {
            "entailment": "SUPPORT",
            "contradiction": "OPPOSE",
            "neutral": "NEUTRAL"
        }

        stance_scores = {
            stance_map[nli_label]: score
            for nli_label, score in avg.items()
        }

        # Get predicted stance (highest score)
        predicted_stance = max(stance_scores.items(), key=lambda x: x[1])[0]

        results.append({
            "stance": predicted_stance,
            "scores": stance_scores,
            "claim": claim
        })

    return results


def _hf_token_classification(
    texts: List[str],
    *,
    model_id: str,
    aggregation_strategy: str = "simple",
) -> List[List[Dict[str, Any]]]:
    """
    NER etc. Do NOT chunk (to preserve spans).
    Returns per-text a list of entity dicts from HF pipeline.
    """
    task = "token-classification"
    pipe = _hf_pipeline_cache(task, model_id, key=f"agg={aggregation_strategy}")
    # Note: truncation is handled automatically by the tokenizer in the pipeline
    return pipe(texts, aggregation_strategy=aggregation_strategy)


# ----------------------------- PRESETS -------------------------------------- #

PRESETS: Dict[str, Tuple[str, Optional[str], Dict[str, Any]]] = {
    # Sentiment / general classification
    "sentiment-twitter": ("text-classification", "cardiffnlp/twitter-roberta-base-sentiment-latest", {}),
    "sentiment-sst2":    ("text-classification", "distilbert-base-uncased-finetuned-sst-2-english", {}),

    # Zero-shot (English + Multilingual)
    "zeroshot-bart":     ("zero-shot-classification", "facebook/bart-large-mnli", {}),
    "zeroshot-mdeberta": ("zero-shot-classification", "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", {}),

    # Stance Detection (NLI-based) - See arxiv:2305.01723
    "stance-deberta":    ("stance-detection", "microsoft/deberta-v3-base-mnli", {}),
    "stance-deberta-large": ("stance-detection", "microsoft/deberta-v3-large-mnli", {}),

    # NER (HF)
    "ner-conll":         ("token-classification", "dslim/bert-base-NER", {"aggregation_strategy": "simple"}),
    "ner-bertbase":      ("token-classification", "dslim/bert-base-NER", {"aggregation_strategy": "simple"}),

    # Topics (tf-idf based)
    "topics-nmf":        ("topics-nmf", None, {"n_topics": 10}),
    "topics-kmeans":     ("topics-kmeans", None, {"n_clusters": 10}),

    # NOTE: Disabled presets (missing implementation files):
    # - spacy-ner, spacy-posdep, spacy-sents (needs spacy_tasks.py)
    # - stanza-posdep, stanza-sents (needs stanza_tasks.py)
    # - sbert-embed, bertopic (needs sbert_tasks.py)
    # These can be re-enabled by creating the corresponding *_tasks.py modules
}


# -------------------------- Public API Helpers ------------------------------ #

def preprocess_text(text: str) -> str:
    """
    Light normalization consistent with earlier examples.
    - Collapse @handles and http links
    """
    if not text:
        return ""
    new = []
    for tok in text.split():
        t = "@user" if tok.startswith("@") and len(tok) > 1 else tok
        t = "http" if t.startswith("http") else t
        new.append(t)
    return " ".join(new)


def preprocess_for_task(text: str, task: str) -> str:
    if task in {"token-classification", "spacy-ner"}:
        # keep raw (span offsets matter)
        return text
    return preprocess_text(text)


def run_task(
    texts: List[str],
    *,
    task: Optional[str] = None,
    preset: Optional[str] = None,
    labels: Optional[List[str]] = None,
    claim: Optional[str] = None,
) -> List[Any]:
    """
    Unified entry point.
    Returns:
      - text/zero-shot classification: list[{"topics": {label: score, ...}}]
      - token-classification (NER):   list[list[ent-dict]]
      - stance-detection:             list[{"stance": str, "scores": {...}, "claim": str}]
      - spaCy/Stanza POS/DEP/LEMMA:   list[dict]
      - SBERT embeddings:             list[{"text":..,"embedding":[...]}]
      - BERTopic:                      dict wrapped in a one-element list
      - Topics (NMF/KMeans):          dict wrapped in a one-element list
    """
    if not isinstance(texts, list):
        raise TypeError("texts must be a list[str]")

    # Resolve preset to (task, model_id, kwargs)
    kwargs: Dict[str, Any] = {}
    model_id: Optional[str] = None
    if preset:
        if preset not in PRESETS:
            raise KeyError(f"Unknown preset: {preset}")
        task_from_preset, model_id, preset_kwargs = PRESETS[preset]
        task = task or task_from_preset
        kwargs.update(preset_kwargs)

    # Default task/model if still missing
    task = task or MODEL_TASK
    model_id = model_id or (MODEL_ID if task in {"text-classification", "zero-shot-classification", "token-classification", "stance-detection"} else None)

    # ---------------- HF transformers ----------------
    if task in {"text-classification", "sentiment-analysis"}:
        return _hf_text_classification(texts, model_id=model_id)  # type: ignore[arg-type]

    if task == "zero-shot-classification":
        return _hf_zero_shot(texts, model_id=model_id, candidate_labels=labels)  # type: ignore[arg-type]

    if task == "token-classification":
        agg = kwargs.get("aggregation_strategy", "simple")
        return _hf_token_classification(texts, model_id=model_id, aggregation_strategy=agg)  # type: ignore[arg-type]

    if task == "stance-detection":
        if not claim:
            raise ValueError("stance-detection requires a 'claim' parameter")
        hypothesis_template = kwargs.get("hypothesis_template", "{}")
        return _hf_stance_detection(texts, model_id=model_id, claim=claim, hypothesis_template=hypothesis_template)  # type: ignore[arg-type]

    # NOTE: The following tasks are disabled (missing implementation files):
    # - spacy-ner, spacy-posdep, spacy-sents (needs spacy_tasks.py)
    # - stanza-posdep, stanza-sents (needs stanza_tasks.py)
    # - sbert-embed, bertopic (needs sbert_tasks.py)
    # These can be re-enabled by creating the corresponding *_tasks.py modules

    # ---------------- Topics (tf-idf) ----------------
    if task == "topics-nmf":
        from topics import topics_nmf
        return [topics_nmf(texts, **kwargs)]

    if task == "topics-kmeans":
        from topics import topics_kmeans
        return [topics_kmeans(texts, **kwargs)]

    raise ValueError(f"Unknown task: {task}")


__all__ = [
    "MODEL_TASK",
    "MODEL_ID",
    "DEFAULT_ZS_LABELS",
    "PRESETS",
    "preprocess_text",
    "preprocess_for_task",
    "run_task",
]
