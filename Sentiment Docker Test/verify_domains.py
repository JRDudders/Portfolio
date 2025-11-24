#!/usr/bin/env python3
"""
Verify all domains involved in HuggingFace model downloads.

This script attempts to download a model and logs ALL network requests
to identify which domains need to be whitelisted by IT.
"""

import os
import sys
import urllib.request
import urllib.parse
from unittest.mock import patch
import ssl

# Disable SSL verification for testing
ssl._create_default_https_context = ssl._create_unverified_context

# Track all URLs accessed
accessed_urls = []
accessed_domains = set()

# Monkey-patch urllib to log all requests
original_urlopen = urllib.request.urlopen

def logged_urlopen(url, *args, **kwargs):
    """Wrapper that logs all URL requests."""
    url_str = url if isinstance(url, str) else url.get_full_url()
    accessed_urls.append(url_str)

    # Extract domain
    parsed = urllib.parse.urlparse(url_str)
    domain = parsed.netloc
    accessed_domains.add(domain)

    print(f"  → Accessing: {domain}")

    # Disable SSL verification
    kwargs['context'] = ssl._create_unverified_context()

    return original_urlopen(url, *args, **kwargs)

urllib.request.urlopen = logged_urlopen

print("=" * 80)
print("HUGGINGFACE DOMAIN VERIFICATION TEST")
print("=" * 80)
print()
print("This test will attempt to download a small model and log ALL domains accessed.")
print("Share these domains with your IT department for whitelisting.")
print()
print("=" * 80)
print()

# Also patch requests library if available
try:
    import requests

    original_request = requests.Session.request

    def logged_request(self, method, url, **kwargs):
        """Wrapper that logs all requests."""
        accessed_urls.append(url)

        # Extract domain
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc
        accessed_domains.add(domain)

        print(f"  → Accessing: {domain}")

        # Disable SSL verification
        kwargs['verify'] = False

        return original_request(self, method, url, **kwargs)

    requests.Session.request = logged_request
    print("✓ Patched requests library for logging")
except ImportError:
    print("⚠ requests library not available")

print()

# Now attempt model download
print("Starting model download test...")
print()

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    # Use a small model for testing
    model_id = "distilbert-base-uncased-finetuned-sst-2-english"

    print(f"Attempting to download: {model_id}")
    print()
    print("Network requests:")
    print("-" * 80)

    # Try to download tokenizer (smaller, faster)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print("-" * 80)
    print()
    print("✓ Download completed successfully!")

except Exception as e:
    print("-" * 80)
    print()
    print(f"✗ Download failed: {str(e)}")

print()
print("=" * 80)
print("DOMAINS ACCESSED DURING DOWNLOAD:")
print("=" * 80)
print()

if accessed_domains:
    for domain in sorted(accessed_domains):
        print(f"  • {domain}")
else:
    print("  (No domains captured - may need different patching method)")

print()
print("=" * 80)
print("INSTRUCTIONS FOR IT DEPARTMENT:")
print("=" * 80)
print()
print("Please whitelist the following domains:")
print()
print("  1. huggingface.co         (Main site)")
print("  2. cdn.huggingface.co     (CDN for model files)")
print("  3. cdn-lfs.huggingface.co (Large File Storage)")
print()
print("Additional domains that MAY be involved:")
print("  • s3.amazonaws.com        (AWS S3 storage)")
print("  • *.cloudfront.net        (AWS CloudFront CDN)")
print("  • raw.githubusercontent.com (Sometimes used for configs)")
print()
print("If the test above captured domains, those should be prioritized.")
print()

# Test specific domains directly
print("=" * 80)
print("TESTING SPECIFIC DOMAINS:")
print("=" * 80)
print()

test_domains = [
    "https://huggingface.co",
    "https://cdn.huggingface.co",
    "https://cdn-lfs.huggingface.co",
    "https://s3.amazonaws.com",
]

for url in test_domains:
    try:
        import requests
        response = requests.get(url, timeout=5, verify=False)
        print(f"✓ {url:40s} - Status: {response.status_code}")
    except Exception as e:
        print(f"✗ {url:40s} - Error: {str(e)[:40]}")

print()
print("=" * 80)
print("FULL URL LOG:")
print("=" * 80)
print()

if accessed_urls:
    for idx, url in enumerate(accessed_urls, 1):
        print(f"{idx}. {url}")
else:
    print("(No URLs captured)")

print()
