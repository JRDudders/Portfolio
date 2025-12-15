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

Authentication:
- Some models may require HuggingFace authentication. Set environment variable:
  export HUGGINGFACE_API_KEY='your_token_here'  OR  export HF_TOKEN='your_token_here'

Performance:
- All classification tasks use batched inference for significant speedup
- GPU automatically detected and used if available (CUDA) with CPU fallback
- Batch processing only activates for datasets with >10 chunks (avoids overhead on small datasets)
- Environment variables for batch sizes:
  * STANCE_BATCH_SIZE=32 (stance detection)
  * CLASSIFY_BATCH_SIZE=16 (text classification/sentiment)
  * ZEROSHOT_BATCH_SIZE=8 (zero-shot classification)
- Text truncation for display: 1000 characters (actual analysis uses full text)
- Typical throughput (texts/minute):
  * Sentiment (CPU): ~2000-4000, (GPU): ~8000-15000
  * Zero-shot (CPU): ~500-1000, (GPU): ~2000-5000
  * Stance (CPU): ~1000-2000, (GPU): ~5000-10000
"""

from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import os
import math
import re
import ssl
import urllib.request

# Disable SSL verification for corporate firewalls (Zscaler, etc.)
# This must be done BEFORE importing transformers/huggingface_hub
if os.getenv("HF_HUB_DISABLE_SSL_VERIFY", "").lower() in ("1", "true", "yes"):
    # Disable SSL verification globally for urllib
    ssl._create_default_https_context = ssl._create_unverified_context

    # Clear CA bundle env vars to prevent SSL verification
    os.environ["CURL_CA_BUNDLE"] = ""
    os.environ["REQUESTS_CA_BUNDLE"] = ""
    os.environ["SSL_CERT_FILE"] = ""

    # Disable requests SSL verification
    import requests
    from requests.adapters import HTTPAdapter
    requests.packages.urllib3.disable_warnings()

    # Create a custom session with SSL verification disabled
    class SSLDisabledAdapter(HTTPAdapter):
        def init_poolmanager(self, *args, **kwargs):
            kwargs['ssl_context'] = ssl._create_unverified_context()
            return super().init_poolmanager(*args, **kwargs)

    # Monkey-patch requests to disable SSL verification
    _original_request = requests.Session.request
    def _patched_request(self, *args, **kwargs):
        kwargs['verify'] = False
        return _original_request(self, *args, **kwargs)
    requests.Session.request = _patched_request

    # Configure huggingface_hub to disable SSL
    try:
        import huggingface_hub
        from huggingface_hub import configure_http_backend

        def _backend_factory() -> requests.Session:
            session = requests.Session()
            session.verify = False
            return session

        configure_http_backend(backend_factory=_backend_factory)
    except ImportError:
        pass  # huggingface_hub not installed yet

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
    Automatically uses GPU if available.
    """
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification
    import torch

    # Detect GPU availability: device=0 for GPU, device=-1 for CPU
    device = 0 if torch.cuda.is_available() else -1
    if device == 0:
        print(f"[nlp] Loading {model_id} on GPU (CUDA)")
    else:
        print(f"[nlp] Loading {model_id} on CPU")

    # Get HuggingFace token from environment (needed for some models like DeBERTa-MNLI)
    hf_token = os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN")

    if task in {"text-classification", "sentiment-analysis", "zero-shot-classification"}:
        tok = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        mdl = AutoModelForSequenceClassification.from_pretrained(model_id, token=hf_token)
        return pipeline("text-classification" if task != "zero-shot-classification" else "zero-shot-classification",
                        model=mdl, tokenizer=tok, device=device)

    if task == "token-classification":
        tok = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        mdl = AutoModelForTokenClassification.from_pretrained(model_id, token=hf_token)
        # aggregation handled at call-time via kwargs
        return pipeline("token-classification", model=mdl, tokenizer=tok, device=device)

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
    Sentiment / general text classification with batched inference.
    For long texts, chunk and average label probs over chunks.
    """
    import torch

    task = "text-classification"
    pipe = _hf_pipeline_cache(task, model_id, key=f"multi={multi_label}")

    # Prepare all chunks with text index mapping
    all_chunks = []
    chunk_to_text_idx = []

    for text_idx, t in enumerate(texts):
        chunks = _chunks_for_classification(t, _CLASSIFY_MAX_WORDS)
        for chunk in chunks:
            all_chunks.append(chunk)
            chunk_to_text_idx.append(text_idx)

    # Only use batching if dataset is large enough (>10 chunks)
    batch_size = int(os.getenv("CLASSIFY_BATCH_SIZE", "16"))
    use_batching = len(all_chunks) >= 10

    if use_batching:
        print(f"[classify] Processing {len(texts)} texts ({len(all_chunks)} chunks) in batches")

        # Process all chunks in batches
        all_outputs = []
        for batch_idx in range(0, len(all_chunks), batch_size):
            batch_chunks = all_chunks[batch_idx:batch_idx + batch_size]
            batch_out = pipe(batch_chunks, truncation=True, padding=True, top_k=None if multi_label else 2)
            all_outputs.extend(batch_out)
    else:
        # Small dataset: process sequentially
        all_outputs = pipe(all_chunks, truncation=True, padding=True, top_k=None if multi_label else 2)

    # Aggregate chunk results per text
    text_chunk_outputs: List[List] = [[] for _ in range(len(texts))]
    for chunk_idx, out in enumerate(all_outputs):
        text_idx = chunk_to_text_idx[chunk_idx]
        text_chunk_outputs[text_idx].append(out)

    # Build final results
    results: List[Dict[str, Any]] = []
    for out_per_chunk in text_chunk_outputs:
        # Normalize output to dict of label->score for averaging
        label_sets: List[Dict[str, float]] = []
        for out in out_per_chunk:
            if isinstance(out, dict) and "label" in out:
                # Some pipelines return single best label
                label_sets.append({out["label"]: float(out["score"])})
            else:
                # Usually a list of {label, score}
                if isinstance(out, list):
                    label_sets.append({e["label"]: float(e["score"]) for e in out})
                else:
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
    """Zero-shot classification with batched inference."""
    task = "zero-shot-classification"
    pipe = _hf_pipeline_cache(task, model_id)

    labels = candidate_labels or DEFAULT_ZS_LABELS

    # Prepare all chunks with text index mapping
    all_chunks = []
    chunk_to_text_idx = []

    for text_idx, t in enumerate(texts):
        chunks = _chunks_for_classification(t, _CLASSIFY_MAX_WORDS)
        for chunk in chunks:
            all_chunks.append(chunk)
            chunk_to_text_idx.append(text_idx)

    # Only use batching if dataset is large enough (>10 chunks)
    batch_size = int(os.getenv("ZEROSHOT_BATCH_SIZE", "8"))  # Smaller batch for zero-shot (more compute intensive)
    use_batching = len(all_chunks) >= 10

    if use_batching:
        print(f"[zeroshot] Processing {len(texts)} texts ({len(all_chunks)} chunks) in batches")

        # Process all chunks in batches
        all_outputs = []
        for batch_idx in range(0, len(all_chunks), batch_size):
            batch_chunks = all_chunks[batch_idx:batch_idx + batch_size]
            # Process batch one chunk at a time (zero-shot pipeline doesn't support true batching)
            for ch in batch_chunks:
                out = pipe(
                    ch,
                    candidate_labels=labels,
                    multi_label=multi_label,
                    hypothesis_template=hypothesis_template,
                )
                all_outputs.append(out)
    else:
        # Small dataset: process sequentially
        all_outputs = []
        for ch in all_chunks:
            out = pipe(
                ch,
                candidate_labels=labels,
                multi_label=multi_label,
                hypothesis_template=hypothesis_template,
            )
            all_outputs.append(out)

    # Aggregate chunk results per text
    text_chunk_probs: List[List[Dict[str, float]]] = [[] for _ in range(len(texts))]
    for chunk_idx, out in enumerate(all_outputs):
        text_idx = chunk_to_text_idx[chunk_idx]
        # HF zero-shot returns dict with 'labels' and 'scores'
        probs = {lbl: float(scr) for lbl, scr in zip(out["labels"], out["scores"])}
        text_chunk_probs[text_idx].append(probs)

    # Build final results
    results: List[Dict[str, Any]] = []
    for per_chunk_probs in text_chunk_probs:
        avg = _avg_scores(per_chunk_probs)
        ordered = sorted(avg.items(), key=lambda kv: kv[1], reverse=True)
        results.append({"topics": {label: float(score) for label, score in ordered}})

    return results


@lru_cache(maxsize=4)
def _load_nli_model(model_id: str):
    """Cache NLI model and tokenizer for stance detection."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import time

    # Get HuggingFace token from environment
    hf_token = os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN")

    # Retry logic for model loading (handles connection issues)
    max_retries = 3
    retry_delay = 2  # seconds

    for attempt in range(max_retries):
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                token=hf_token,
                resume_download=True,  # Resume partial downloads
                local_files_only=False  # Allow downloading
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                model_id,
                token=hf_token,
                resume_download=True,
                local_files_only=False
            )
            return tokenizer, model
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"[nlp] Model loading attempt {attempt + 1} failed: {e}. Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                raise RuntimeError(
                    f"Failed to load model {model_id} after {max_retries} attempts. "
                    f"This may be due to: (1) network connectivity issues, "
                    f"(2) the model being too large to download quickly, or "
                    f"(3) HuggingFace API issues. Try again later or check your connection. "
                    f"Original error: {e}"
                )


def _hf_stance_detection(
    texts: List[str],
    *,
    model_id: str,
    claim: str,
    hypothesis_template: str = "{}",
) -> List[Dict[str, Any]]:
    """
    NLI-based stance detection (arxiv:2305.01723).

    Uses Natural Language Inference to classify stance by treating:
    - Premise: the text to classify
    - Hypothesis: the claim
    - ENTAILMENT -> SUPPORT
    - CONTRADICTION -> OPPOSE
    - NEUTRAL -> NEUTRAL

    Args:
        texts: List of texts to classify
        model_id: NLI model (e.g., MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli)
        claim: The claim/hypothesis to classify stance towards
        hypothesis_template: Template for formatting hypothesis (default: "{}")

    Returns:
        List of dicts with stance labels and confidence scores
    """
    import torch

    # Load cached model and tokenizer
    tokenizer, model = _load_nli_model(model_id)

    # Check for GPU availability (with CPU fallback)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = model.to(device)
    except Exception as e:
        print(f"[stance] Failed to move model to {device}, falling back to CPU: {e}")
        device = torch.device("cpu")
        model = model.to(device)

    # Format the hypothesis (claim) using template
    hypothesis = hypothesis_template.format(claim)

    # NLI models typically use these labels
    label_mapping = {
        0: "contradiction",
        1: "neutral",
        2: "entailment"
    }

    # Stance mapping from NLI labels
    stance_map = {
        "entailment": "SUPPORT",
        "contradiction": "OPPOSE",
        "neutral": "NEUTRAL"
    }

    # Prepare all text-chunk to text-index mappings
    all_chunks = []
    chunk_to_text_idx = []

    for text_idx, text in enumerate(texts):
        chunks = _chunks_for_classification(text, _CLASSIFY_MAX_WORDS)
        for chunk in chunks:
            all_chunks.append(chunk)
            chunk_to_text_idx.append(text_idx)

    # Batch size for processing (adjust based on memory)
    batch_size = int(os.getenv("STANCE_BATCH_SIZE", "32"))

    # Only use batching if dataset is large enough (>10 chunks)
    use_batching = len(all_chunks) >= 10

    # Process all chunks in batches
    all_probs = []
    total_batches = (len(all_chunks) + batch_size - 1) // batch_size

    if use_batching:
        print(f"[stance] Processing {len(texts)} texts ({len(all_chunks)} chunks) in {total_batches} batches on {device}")
    else:
        print(f"[stance] Processing {len(texts)} texts ({len(all_chunks)} chunks) sequentially on {device}")

    for batch_idx in range(0, len(all_chunks), batch_size):
        batch_chunks = all_chunks[batch_idx:batch_idx + batch_size]

        # Encode batch of text pairs
        inputs = tokenizer(
            batch_chunks,
            [hypothesis] * len(batch_chunks),  # Same hypothesis for all
            truncation=True,
            padding=True,
            return_tensors="pt"
        )

        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get model predictions for batch
        with torch.no_grad():
            outputs = model(**inputs)
            batch_probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # Store results
        all_probs.extend(batch_probs.cpu().tolist())

        # Progress logging
        if (batch_idx // batch_size + 1) % 10 == 0 or batch_idx + batch_size >= len(all_chunks):
            print(f"[stance] Processed {min(batch_idx + batch_size, len(all_chunks))}/{len(all_chunks)} chunks")

    # Aggregate chunk results per text
    text_chunk_probs: List[List[Dict[str, float]]] = [[] for _ in range(len(texts))]

    for chunk_idx, probs in enumerate(all_probs):
        text_idx = chunk_to_text_idx[chunk_idx]
        nli_probs = {
            label_mapping.get(i, f"label_{i}"): float(probs[i])
            for i in range(len(probs))
        }
        text_chunk_probs[text_idx].append(nli_probs)

    # Build final results
    results: List[Dict[str, Any]] = []
    for per_chunk_probs in text_chunk_probs:
        # Average probabilities across chunks for this text
        avg_nli = _avg_scores(per_chunk_probs)

        # Map NLI labels to stance labels
        stance_scores = {
            stance_map.get(nli_label, "NEUTRAL"): score
            for nli_label, score in avg_nli.items()
            if nli_label in stance_map
        }

        # Get predicted stance (highest score)
        predicted_stance = max(stance_scores.items(), key=lambda x: x[1])[0]

        results.append({
            "stance": predicted_stance,
            "scores": stance_scores,
            "claim": claim
        })

    print(f"[stance] Completed processing {len(texts)} texts")
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
    # Using publicly available NLI models (no authentication required)
    # Note: First-time use downloads models (~500MB for base, ~1.5GB for large)
    "stance-deberta":    ("stance-detection", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", {}),
    "stance-deberta-large": ("stance-detection", "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli", {}),

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


# -------------------- Translation (Multilingual to English) -------------------- #

_translation_model = None
_translation_tokenizer = None

def _load_translation_model():
    """Load translation model (Helsinki-NLP opus-mt-mul-en)"""
    global _translation_model, _translation_tokenizer

    if _translation_model is not None:
        return _translation_model, _translation_tokenizer

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model_id = "Helsinki-NLP/opus-mt-mul-en"
    hf_token = os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN")

    print(f"[nlp] Loading translation model: {model_id}")

    _translation_tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    _translation_model = AutoModelForSeq2SeqLM.from_pretrained(model_id, token=hf_token)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _translation_model = _translation_model.to(device)

    print(f"[nlp] Translation model loaded on {device.upper()}")
    return _translation_model, _translation_tokenizer


def translate_to_english(text: str, max_length: int = 512) -> str:
    """
    Translate text to English using Helsinki-NLP opus-mt-mul-en.

    Args:
        text: Text in any language
        max_length: Maximum output length

    Returns:
        English translation
    """
    import torch

    if not text or not text.strip():
        return ""

    model, tokenizer = _load_translation_model()
    device = next(model.parameters()).device

    # Truncate input if too long
    inputs = tokenizer(text[:5000], return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length)

    translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated


def translate_batch(texts: List[str], max_length: int = 512) -> List[str]:
    """
    Translate multiple texts to English.

    Args:
        texts: List of texts in any language
        max_length: Maximum output length per text

    Returns:
        List of English translations
    """
    translations = []
    for i, text in enumerate(texts):
        if not text or not text.strip():
            translations.append("")
            continue

        try:
            translated = translate_to_english(text, max_length)
            translations.append(translated)
        except Exception as e:
            print(f"[nlp] Translation failed for item {i}: {e}")
            translations.append(text)  # Fallback to original

        if (i + 1) % 20 == 0:
            print(f"[nlp] Translated {i + 1}/{len(texts)} texts...")

    return translations


# -------------------- Local LLM for Narrative Generation -------------------- #

_narrative_model = None
_narrative_tokenizer = None

def _load_narrative_model():
    """Load local LLM for narrative generation (Phi-3-mini)"""
    global _narrative_model, _narrative_tokenizer

    if _narrative_model is not None:
        return _narrative_model, _narrative_tokenizer

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "microsoft/Phi-3-mini-4k-instruct"
    hf_token = os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN")

    print(f"[nlp] Loading narrative LLM: {model_id}")

    _narrative_tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token, trust_remote_code=True)
    _narrative_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=hf_token,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    if not torch.cuda.is_available():
        _narrative_model = _narrative_model.to("cpu")

    print(f"[nlp] Narrative LLM loaded on {'GPU' if torch.cuda.is_available() else 'CPU'}")
    return _narrative_model, _narrative_tokenizer


def generate_narrative(body_text: str, themes: List[str], top_theme: str) -> str:
    """
    Generate a narrative explaining how the body text relates to identified themes.

    Args:
        body_text: The extracted text content
        themes: List of all possible theme categories
        top_theme: The top identified theme for this text

    Returns:
        A brief narrative explaining the relevance
    """
    import torch

    model, tokenizer = _load_narrative_model()

    # Truncate body text if too long
    max_body_chars = 2000
    if len(body_text) > max_body_chars:
        body_text = body_text[:max_body_chars] + "..."

    themes_str = ", ".join(themes)

    prompt = f"""<|user|>
You are an analyst. Given the following text and its identified theme, write a brief 1-2 sentence narrative explaining how the text relates to the theme "{top_theme}".

Possible themes: {themes_str}
Identified theme: {top_theme}

Text:
{body_text}

Write a concise narrative (1-2 sentences) explaining the relevance to the theme. If the theme is "Other", note any tangential relevance to other themes.<|end|>
<|assistant|>"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()


def generate_narrative_with_examples(
    body_text: str,
    themes: List[str],
    top_theme: str,
    few_shot_examples: Optional[List[Dict[str, Any]]] = None
) -> str:
    """
    Generate a narrative with optional few-shot examples from training data.

    Args:
        body_text: The extracted text content
        themes: List of all possible theme categories
        top_theme: The top identified theme for this text
        few_shot_examples: Optional list of similar examples with their narratives

    Returns:
        A brief narrative explaining the relevance
    """
    import torch

    model, tokenizer = _load_narrative_model()

    # Truncate body text if too long
    max_body_chars = 1500 if few_shot_examples else 2000
    if len(body_text) > max_body_chars:
        body_text = body_text[:max_body_chars] + "..."

    themes_str = ", ".join(themes)

    # Build few-shot examples section
    examples_section = ""
    if few_shot_examples:
        examples_section = "\nHere are examples of how to write narratives:\n"
        for i, ex in enumerate(few_shot_examples[:3], 1):  # Max 3 examples
            ex_text = ex.get('text', '')[:400]
            ex_themes = ', '.join(ex.get('themes', []))
            ex_narrative = ex.get('narrative', '')[:200]
            if ex_narrative:
                examples_section += f"\nExample {i}:\nText: {ex_text}...\nThemes: {ex_themes}\nNarrative: {ex_narrative}\n"

    prompt = f"""<|user|>
You are an analyst. Given the following text and its identified theme, write a brief 1-2 sentence narrative explaining how the text relates to the theme "{top_theme}".

Possible themes: {themes_str}
Identified theme: {top_theme}
{examples_section}
Text to analyze:
{body_text}

Write a concise narrative (1-2 sentences) explaining the relevance to the theme. Match the style of the examples if provided. If the theme is "Other", note any tangential relevance to other themes.<|end|>
<|assistant|>"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()


def generate_narratives_batch(
    bodies: List[str],
    themes: List[str],
    top_themes: List[str],
    training_store: Optional[Any] = None,
) -> List[str]:
    """
    Generate narratives for multiple texts.

    Args:
        bodies: List of body texts
        themes: List of all possible theme categories
        top_themes: List of top theme for each text
        training_store: Optional TrainingStore for few-shot learning

    Returns:
        List of narrative strings
    """
    narratives = []
    for i, (body, top_theme) in enumerate(zip(bodies, top_themes)):
        if not body or not body.strip():
            narratives.append("")
            continue

        try:
            # Get few-shot examples if training store is available
            few_shot_examples = None
            if training_store:
                few_shot_examples = training_store.get_few_shot_examples(
                    body, task='narrative', num_examples=3
                )

            if few_shot_examples:
                narrative = generate_narrative_with_examples(body, themes, top_theme, few_shot_examples)
            else:
                narrative = generate_narrative(body, themes, top_theme)
            narratives.append(narrative)
        except Exception as e:
            print(f"[nlp] Narrative generation failed for item {i}: {e}")
            narratives.append("")

        if (i + 1) % 10 == 0:
            print(f"[nlp] Generated {i + 1}/{len(bodies)} narratives...")

    return narratives


__all__ = [
    "MODEL_TASK",
    "MODEL_ID",
    "DEFAULT_ZS_LABELS",
    "PRESETS",
    "preprocess_text",
    "preprocess_for_task",
    "run_task",
]
