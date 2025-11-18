#!/usr/bin/env python3
"""
Manual model download script with chunked downloading and retries.
Use this when the corporate firewall blocks large HuggingFace downloads.

This downloads model files piece-by-piece with retry logic.
"""

import os
import sys
import time
import ssl
import requests
from pathlib import Path
from tqdm import tqdm

# Apply SSL bypass
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['CURL_CA_BUNDLE'] = ''

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Monkeypatch requests
_original_request = requests.Session.request

def _unverified_request(self, method, url, **kwargs):
    kwargs['verify'] = False
    return _original_request(self, method, url, **kwargs)

requests.Session.request = _unverified_request

print("✓ SSL verification disabled\n")

def get_hf_cache_dir():
    """Get HuggingFace cache directory."""
    cache_dir = os.getenv("HF_HOME") or os.getenv("HUGGINGFACE_HUB_CACHE")
    if cache_dir:
        return Path(cache_dir)

    if os.name == 'nt':  # Windows
        base = Path(os.getenv("USERPROFILE", "~"))
        cache_dir = base / ".cache" / "huggingface"
    else:  # Unix
        cache_dir = Path.home() / ".cache" / "huggingface"

    return cache_dir.expanduser()

def download_file_chunked(url, destination, chunk_size=1024*1024, max_retries=10):
    """
    Download a file in chunks with retry logic.

    Args:
        url: URL to download
        destination: Local file path to save to
        chunk_size: Size of chunks to download (1MB default)
        max_retries: Maximum number of retry attempts
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    # Check if file already exists and get size
    existing_size = 0
    if destination.exists():
        existing_size = destination.stat().st_size
        print(f"  Found existing partial download: {existing_size:,} bytes")

    session = requests.Session()
    retry_count = 0

    while retry_count < max_retries:
        try:
            # Set Range header for resuming download
            headers = {}
            if existing_size > 0:
                headers['Range'] = f'bytes={existing_size}-'

            print(f"  Attempting download (attempt {retry_count + 1}/{max_retries})...")
            response = session.get(url, headers=headers, stream=True, timeout=30, verify=False)

            if response.status_code == 416:  # Range not satisfiable = already complete
                print(f"  ✓ File already fully downloaded")
                return True

            if response.status_code not in [200, 206]:  # 206 = Partial Content
                print(f"  ✗ HTTP {response.status_code}")
                retry_count += 1
                time.sleep(2 ** retry_count)  # Exponential backoff
                continue

            # Get total file size
            total_size = existing_size
            if 'Content-Length' in response.headers:
                total_size += int(response.headers['Content-Length'])

            # Open file in append mode if resuming, write mode if new
            mode = 'ab' if existing_size > 0 else 'wb'

            # Download with progress bar
            with open(destination, mode) as f:
                with tqdm(
                    total=total_size,
                    initial=existing_size,
                    unit='B',
                    unit_scale=True,
                    desc=f"  {destination.name}"
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            print(f"  ✓ Download complete: {destination.name}")
            return True

        except (requests.exceptions.RequestException, IOError) as e:
            print(f"  ✗ Error: {e}")
            retry_count += 1
            if retry_count < max_retries:
                wait_time = 2 ** retry_count
                print(f"  Retrying in {wait_time}s...")
                time.sleep(wait_time)

            # Update existing size in case we got some data
            if destination.exists():
                existing_size = destination.stat().st_size

    print(f"  ✗ Failed after {max_retries} attempts")
    return False

def download_model(model_id="cardiffnlp/twitter-roberta-base-sentiment-latest"):
    """Download all files for a HuggingFace model."""

    print(f"Downloading model: {model_id}\n")

    # Get cache directory
    cache_dir = get_hf_cache_dir()
    model_cache_name = f"models--{model_id.replace('/', '--')}"
    model_dir = cache_dir / "hub" / model_cache_name / "snapshots"

    print(f"Cache directory: {cache_dir}")
    print(f"Model directory: {model_dir}\n")

    # Get model files list
    base_url = f"https://huggingface.co/{model_id}/resolve/main"

    # Essential files for sentiment model
    files_to_download = [
        "config.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer.json",
        "pytorch_model.bin",  # Large file - 501MB
    ]

    # Try to get commit hash
    print("Getting latest commit hash...")
    try:
        response = requests.get(
            f"https://huggingface.co/api/models/{model_id}",
            verify=False,
            timeout=10
        )
        commit_hash = response.json().get('sha', 'main')
        print(f"✓ Commit hash: {commit_hash}\n")
    except:
        commit_hash = 'main'
        print(f"⚠ Could not get commit hash, using 'main'\n")

    # Create snapshot directory
    snapshot_dir = model_dir / commit_hash
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # Download each file
    success = True
    for filename in files_to_download:
        url = f"{base_url}/{filename}"
        destination = snapshot_dir / filename

        print(f"\nDownloading: {filename}")

        if not download_file_chunked(url, destination):
            print(f"✗ Failed to download {filename}")
            success = False

            if filename == "pytorch_model.bin":
                print("\n⚠ The large model file (pytorch_model.bin) failed to download.")
                print("This is the file your firewall is blocking (501MB).")
                print("\nTrying alternative: model.safetensors format...")

                # Try safetensors format instead
                alt_url = f"{base_url}/model.safetensors"
                alt_destination = snapshot_dir / "model.safetensors"

                if download_file_chunked(alt_url, alt_destination):
                    print("✓ Successfully downloaded safetensors format!")
                    success = True
                else:
                    print("✗ Safetensors format also failed")
                    break

    if success:
        print("\n" + "="*60)
        print("✓ MODEL DOWNLOAD COMPLETE")
        print("="*60)
        print(f"\nModel saved to: {snapshot_dir}")
        print("\nYou can now run: python run_local.py")
        return True
    else:
        print("\n" + "="*60)
        print("✗ MODEL DOWNLOAD FAILED")
        print("="*60)
        print("\nYour firewall is blocking large file downloads.")
        print("You will need IT to whitelist: cdn-lfs.huggingface.co")
        return False

if __name__ == "__main__":
    model_id = "cardiffnlp/twitter-roberta-base-sentiment-latest"

    if len(sys.argv) > 1:
        model_id = sys.argv[1]

    download_model(model_id)
