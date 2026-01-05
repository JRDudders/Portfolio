#!/usr/bin/env python3
"""
Pre-download HuggingFace models for batch Excel processing.
Run during Docker build to cache models in the image.
"""
import os
import sys

# Models used by batch Excel processing
MODELS = [
    # Sentiment analysis
    ("cardiffnlp/twitter-roberta-base-sentiment-latest", "AutoModelForSequenceClassification"),

    # Zero-shot classification (themes)
    ("facebook/bart-large-mnli", "AutoModelForSequenceClassification"),
    ("MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", "AutoModelForSequenceClassification"),

    # Stance detection
    ("MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", "AutoModelForSequenceClassification"),

    # Named entity recognition
    ("dslim/bert-base-NER", "AutoModelForTokenClassification"),

    # Translation (multilingual to English)
    ("Helsinki-NLP/opus-mt-mul-en", "AutoModelForSeq2SeqLM"),

    # Narrative generation (Phi-3)
    ("microsoft/Phi-3-mini-4k-instruct", "AutoModelForCausalLM"),
]

def download_models():
    """Download all required models to HF_HOME cache."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification
    from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM

    hf_token = os.environ.get("HF_TOKEN", None)

    model_classes = {
        "AutoModelForSequenceClassification": AutoModelForSequenceClassification,
        "AutoModelForTokenClassification": AutoModelForTokenClassification,
        "AutoModelForSeq2SeqLM": AutoModelForSeq2SeqLM,
        "AutoModelForCausalLM": AutoModelForCausalLM,
    }

    for model_id, model_class_name in MODELS:
        print(f"[download] Downloading {model_id}...")
        try:
            # Download tokenizer
            AutoTokenizer.from_pretrained(model_id, token=hf_token, trust_remote_code=True)

            # Download model
            model_class = model_classes[model_class_name]
            if model_class_name == "AutoModelForCausalLM":
                # Phi-3 needs trust_remote_code
                model_class.from_pretrained(model_id, token=hf_token, trust_remote_code=True)
            else:
                model_class.from_pretrained(model_id, token=hf_token)

            print(f"[download] ✓ {model_id}")
        except Exception as e:
            print(f"[download] ✗ {model_id}: {e}", file=sys.stderr)
            # Continue with other models - some may be optional

if __name__ == "__main__":
    download_models()
    print("[download] Model download complete")
