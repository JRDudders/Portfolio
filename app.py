"""
tweeteval_100_sentiment_json.py
Prereqs: pip install transformers datasets torch
This is the app.py for a docker conatiner I'm building to run huggingface pipeline models, specifically sentiment analysis at the moment
"""

import json, os, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, pipeline
from datasets import load_dataset
from typing import Any, Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# -------- Config via env --------
MODEL_ID = os.getenv("MODEL_ID", "cardiffnlp/twitter-roberta-base-sentiment-latest")
TASK = os.getenv("TASK", "text-classification")  # e.g., "text-classification", "token-classification", "question-answering", "image-classification", "automatic-speech-recognition"
TRUST_REMOTE_CODE = os.getenv("TRUST_REMOTE_CODE", "false").lower() in {"1","true","yes"}
PIPELINE_KWARGS = os.getenv("PIPELINE_KWARGS", "{}")  # JSON string, e.g. {"top_k": null, "return_all_scores": true}
DEVICE_MAP = os.getenv("DEVICE_MAP", "auto")          # "auto" or integer device id
DTYPE = os.getenv("DTYPE", "auto")                    # "auto" | "float16" | "bfloat16" | "float32"

def _dtype_from_env(dtype: str):
    if dtype == "auto":
        return "auto"
    m = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return m.get(dtype.lower(), "auto")

def _safe_json_loads(s: str):
    try:
        return json.loads(s) if s.strip() else {}
    except Exception:
        return {}

EXTRA = _safe_json_loads(PIPELINE_KWARGS)

# -------- App --------
app = FastAPI(title="Transformers Pipeline Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

PIPE = None

def build_pipeline(model_id: str, task: str):
    dtype = _dtype_from_env(DTYPE)
    kwargs = dict(
        model=model_id,
        task=task,
        device_map=DEVICE_MAP,
        torch_dtype=dtype if dtype != "auto" else None,
        trust_remote_code=TRUST_REMOTE_CODE,
    )
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    return pipeline(**kwargs)

@app.on_event("startup")
def _startup():
    global PIPE
    PIPE = build_pipeline(MODEL_ID, TASK)
''' i actually am not sure if `/` will work, but let's find out
@app.get("/")
def entrypoint():
    return {"hello": "world"}
'''    
@app.get("/")
def entrypoint():
    return main()

@app.get("/healthz")
def healthz():
    return {"status": "ok", "model": MODEL_ID, "task": TASK}

class PredictRequest(BaseModel):
    inputs: Any
    parameters: Optional[dict] = None

class ReloadRequest(BaseModel):
    model_id: Optional[str] = None
    task: Optional[str] = None

@app.post("/predict")
def predict(req: PredictRequest):
    global PIPE
    if PIPE is None:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    call_kwargs = dict(EXTRA)
    if req.parameters:
        call_kwargs.update(req.parameters)
    try:
        outputs = PIPE(req.inputs, **call_kwargs)
        return {"outputs": outputs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reload")
def reload_model(req: ReloadRequest):
    global PIPE, MODEL_ID, TASK
    MODEL_ID = req.model_id or MODEL_ID
    TASK = req.task or TASK
    try:
        PIPE = build_pipeline(MODEL_ID, TASK)
        return {"status": "reloaded", "model": MODEL_ID, "task": TASK}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reload failed: {e}")
        
        
MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
SPLIT = "test"          # from cardiffnlp/tweet_eval: sentiment
SUBSET = "sentiment"
N = 100
BATCH_SIZE = 32
MAX_LEN = 256

def preprocess(text: str) -> str:
    # Same normalization as your original code
    out = []
    for t in text.split(" "):
        t = "@user" if t.startswith("@") and len(t) > 1 else t
        t = "http" if t.startswith("http") else t
        out.append(t)
    return " ".join(out)

def batched(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i+n], i, min(i+n, len(xs))

def main():
    """
    Optimized batch sentiment analysis using pre-loaded pipeline.
    Uses the global PIPE to avoid duplicate model loading.
    """
    global PIPE
    if PIPE is None:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")

    # 1) Load data (TweetEval sentiment/test)
    ds = load_dataset("cardiffnlp/tweet_eval", SUBSET, split=SPLIT)
    texts = ds["text"][:N]
    proc_texts = [preprocess(t) for t in texts]

    # 2) Use the pre-loaded pipeline for efficient inference
    try:
        # Process in batches for better memory management
        predictions = []
        for batch, lo, hi in batched(proc_texts, BATCH_SIZE):
            batch_results = PIPE(batch, truncation=True, max_length=MAX_LEN)

            # Convert pipeline output to the expected format
            for t_orig, result in zip(texts[lo:hi], batch_results):
                if isinstance(result, list):
                    # Multiple labels returned
                    scores = {r["label"]: float(r["score"]) for r in result}
                else:
                    # Single label returned
                    scores = {result["label"]: float(result["score"])}
                predictions.append({"text": t_orig, "scores": scores})

        result = {
            "dataset": "cardiffnlp/tweet_eval:sentiment:test",
            "model": MODEL_ID,
            "count": len(predictions),
            "predictions": predictions,
        }

        # 3) Save to file
        output_path = "tweeteval_sentiment_100.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {e}")

if __name__ == "__main__":
    main()
