#!/usr/bin/env python3
"""
Test HuggingFace connectivity and diagnose firewall/SSL issues.
"""

import os
import sys
import ssl
import warnings

# Apply the same SSL bypass as nlp.py
if os.getenv("HF_HUB_DISABLE_SSL_VERIFY", "1") == "1":
    ssl._create_default_https_context = ssl._create_unverified_context

    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    os.environ['REQUESTS_CA_BUNDLE'] = ''
    os.environ['CURL_CA_BUNDLE'] = ''

    try:
        import requests
        _original_request = requests.Session.request

        def _unverified_request(self, method, url, **kwargs):
            kwargs['verify'] = False
            return _original_request(self, method, url, **kwargs)

        requests.Session.request = _unverified_request
        print("✓ SSL verification disabled")
    except Exception as e:
        print(f"✗ Could not disable SSL: {e}")

print("\n" + "="*60)
print("HuggingFace Connectivity Test")
print("="*60 + "\n")

# Test 1: Basic HTTP connection
print("Test 1: Basic HTTP connection to huggingface.co")
try:
    import requests
    response = requests.get("https://huggingface.co", timeout=10, verify=False)
    print(f"✓ Status: {response.status_code}")
    print(f"✓ Connection successful\n")
except Exception as e:
    print(f"✗ Connection failed: {e}")
    print("Your firewall may be blocking ALL connections to HuggingFace\n")
    sys.exit(1)

# Test 2: HuggingFace API
print("Test 2: HuggingFace API access")
try:
    model_id = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    url = f"https://huggingface.co/api/models/{model_id}"
    response = requests.get(url, timeout=10, verify=False)
    print(f"✓ Status: {response.status_code}")
    print(f"✓ API accessible\n")
except Exception as e:
    print(f"✗ API access failed: {e}\n")

# Test 3: Download a small config file
print("Test 3: Download model config file")
try:
    model_id = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    url = f"https://huggingface.co/{model_id}/resolve/main/config.json"

    print(f"Downloading: {url}")
    response = requests.get(url, timeout=30, verify=False)

    if response.status_code == 200:
        print(f"✓ Downloaded {len(response.content)} bytes")
        print(f"✓ File download working\n")
    else:
        print(f"✗ Status: {response.status_code}")
        print(f"✗ Download failed\n")
except Exception as e:
    print(f"✗ Download failed: {e}")
    print("Your firewall is likely blocking file downloads from HuggingFace\n")
    sys.exit(1)

# Test 4: HuggingFace Hub library
print("Test 4: HuggingFace Hub library with progress")
try:
    from huggingface_hub import hf_hub_download
    from tqdm.auto import tqdm

    print("Attempting to download tokenizer config (small file)...")

    # Set token if available
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")

    file = hf_hub_download(
        repo_id="cardiffnlp/twitter-roberta-base-sentiment-latest",
        filename="tokenizer_config.json",
        token=token,
        resume_download=True,
    )
    print(f"✓ Downloaded to: {file}")
    print(f"✓ HuggingFace Hub library working\n")

except Exception as e:
    print(f"✗ Hub download failed: {e}")
    print("There may be an issue with the HuggingFace Hub library\n")
    import traceback
    traceback.print_exc()

# Test 5: Full model download with progress
print("Test 5: Full model download (this may take a few minutes)")
print("This will show actual download progress...\n")

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import time

    model_id = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")

    print(f"Downloading tokenizer for {model_id}...")
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    print(f"✓ Tokenizer downloaded in {time.time() - start:.1f}s\n")

    print(f"Downloading model for {model_id}...")
    print("(This is ~500MB, may take 2-5 minutes depending on network speed)")
    start = time.time()
    model = AutoModelForSequenceClassification.from_pretrained(model_id, token=token)
    print(f"✓ Model downloaded in {time.time() - start:.1f}s\n")

    print("="*60)
    print("✓ ALL TESTS PASSED")
    print("="*60)
    print("\nYour connection is working. The model should load successfully.")
    print("Try running 'python run_local.py' again.\n")

except Exception as e:
    print(f"✗ Model download failed: {e}")
    print("\nDiagnosis: Your firewall is likely blocking large file downloads")
    print("from HuggingFace, even though small files work.\n")
    print("Possible solutions:")
    print("1. Contact IT to whitelist huggingface.co and cdn.huggingface.co")
    print("2. Use a VPN to bypass the corporate firewall")
    print("3. Download the model on a different network, then copy the cache folder")
    print("4. Use Docker with pre-downloaded models\n")
    import traceback
    traceback.print_exc()
