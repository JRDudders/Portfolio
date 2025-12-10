#!/usr/bin/env python3
"""Download HuggingFace models for batch Excel processing."""
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import os
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'

import requests
requests.packages.urllib3.disable_warnings()
_original_request = requests.Session.request
def _patched_request(self, *args, **kwargs):
    kwargs['verify'] = False
    return _original_request(self, *args, **kwargs)
requests.Session.request = _patched_request

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, AutoModelForSeq2SeqLM

# 1. Stance Detection (NLI-based)
print('Downloading MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli for stance detection...')
AutoTokenizer.from_pretrained('MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli')
AutoModelForSequenceClassification.from_pretrained('MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli')

# 2. Zero-shot Classification (for themes)
print('Downloading facebook/bart-large-mnli...')
AutoTokenizer.from_pretrained('facebook/bart-large-mnli')
AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli')

# 3. Translation (multilingual to English)
print('Downloading Helsinki-NLP/opus-mt-mul-en...')
AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-mul-en')
AutoModelForSeq2SeqLM.from_pretrained('Helsinki-NLP/opus-mt-mul-en')

# 4. Narrative Generation (local LLM)
print('Downloading microsoft/Phi-3-mini-4k-instruct...')
AutoTokenizer.from_pretrained('microsoft/Phi-3-mini-4k-instruct', trust_remote_code=True)
AutoModelForCausalLM.from_pretrained('microsoft/Phi-3-mini-4k-instruct', trust_remote_code=True)

print('All minimal models downloaded successfully!')
