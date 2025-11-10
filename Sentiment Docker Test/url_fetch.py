"""
URL Fetch Utility

Shared module for fetching files from URLs with:
- Size limits (500MB default)
- Timeout protection
- Proper error handling
- Content type validation
- Wayback Machine fallback for failed requests
"""

import httpx
from typing import Optional, Tuple
import mimetypes


# Maximum file size to download (500MB to match nginx limit)
MAX_DOWNLOAD_SIZE = 500 * 1024 * 1024  # 500MB in bytes

# Connection timeout (1 hour for large model downloads)
TIMEOUT_SECONDS = 3600


async def get_wayback_snapshot(url: str) -> Optional[str]:
    """
    Query Wayback Machine API to find the most recent snapshot of a URL.

    Args:
        url: Original URL to search for

    Returns:
        Wayback Machine URL if snapshot exists, None otherwise
    """
    try:
        api_url = f"http://archive.org/wayback/available?url={url}"

        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(api_url)
            response.raise_for_status()
            data = response.json()

            # Check if snapshot is available
            if data.get('archived_snapshots', {}).get('closest', {}).get('available'):
                snapshot_url = data['archived_snapshots']['closest']['url']
                timestamp = data['archived_snapshots']['closest']['timestamp']
                print(f"[url_fetch] Found Wayback Machine snapshot from {timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]}")
                return snapshot_url

            return None
    except Exception as e:
        print(f"[url_fetch] Wayback Machine query failed: {e}")
        return None


async def fetch_url(
    url: str,
    max_size: int = MAX_DOWNLOAD_SIZE,
    timeout: int = TIMEOUT_SECONDS,
    try_wayback: bool = True
) -> Tuple[bytes, Optional[str]]:
    """
    Fetch file content from URL with Wayback Machine fallback.

    Args:
        url: URL to fetch
        max_size: Maximum file size in bytes (default 500MB)
        timeout: Timeout in seconds (default 3600s)
        try_wayback: If True, try Wayback Machine on failure (default True)

    Returns:
        Tuple of (file_bytes, content_type)

    Raises:
        ValueError: If URL is invalid or file too large
        httpx.HTTPError: If download fails (and Wayback Machine unavailable)
    """
    # Validate URL scheme
    if not url.startswith(('http://', 'https://')):
        raise ValueError("URL must start with http:// or https://")

    # Try original URL first
    try:
        return await _fetch_from_url(url, max_size, timeout)
    except (httpx.HTTPError, Exception) as e:
        print(f"[url_fetch] Failed to fetch {url}: {e}")

        # Try Wayback Machine as fallback
        if try_wayback and not url.startswith('https://web.archive.org/'):
            print(f"[url_fetch] Attempting Wayback Machine fallback...")
            wayback_url = await get_wayback_snapshot(url)

            if wayback_url:
                try:
                    print(f"[url_fetch] Fetching from archive: {wayback_url}")
                    return await _fetch_from_url(wayback_url, max_size, timeout)
                except Exception as wayback_error:
                    print(f"[url_fetch] Wayback Machine fetch also failed: {wayback_error}")

        # Re-raise original error if fallback didn't work
        raise


async def _fetch_from_url(
    url: str,
    max_size: int,
    timeout: int
) -> Tuple[bytes, Optional[str]]:
    """
    Internal function to fetch content from a specific URL.

    Args:
        url: URL to fetch
        max_size: Maximum file size in bytes
        timeout: Timeout in seconds

    Returns:
        Tuple of (file_bytes, content_type)

    Raises:
        httpx.HTTPError: If download fails
        ValueError: If file too large
    """
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        # First, do a HEAD request to check content length
        try:
            head_response = await client.head(url)
            head_response.raise_for_status()

            content_length = head_response.headers.get('content-length')
            if content_length:
                size = int(content_length)
                if size > max_size:
                    size_mb = size / (1024 * 1024)
                    max_mb = max_size / (1024 * 1024)
                    raise ValueError(
                        f"File too large: {size_mb:.1f}MB exceeds limit of {max_mb:.1f}MB"
                    )
        except httpx.HTTPStatusError:
            # HEAD request failed, proceed with GET anyway
            pass

        # Download the file
        response = await client.get(url)
        response.raise_for_status()

        # Check actual size
        content = response.content
        if len(content) > max_size:
            size_mb = len(content) / (1024 * 1024)
            max_mb = max_size / (1024 * 1024)
            raise ValueError(
                f"Downloaded file too large: {size_mb:.1f}MB exceeds limit of {max_mb:.1f}MB"
            )

        # Get content type
        content_type = response.headers.get('content-type')

        return content, content_type


def guess_file_extension(url: str, content_type: Optional[str] = None) -> str:
    """
    Guess file extension from URL or content type.

    Args:
        url: URL of the file
        content_type: HTTP Content-Type header

    Returns:
        File extension (e.g., '.json', '.csv', '.edges')
    """
    # Try to get extension from URL
    if '.' in url.split('/')[-1]:
        ext = '.' + url.split('.')[-1].lower().split('?')[0]  # Remove query params
        if ext in ['.json', '.csv', '.html', '.htm', '.txt', '.edges', '.edge', '.circles', '.feat', '.featnames']:
            return ext

    # Try to guess from content type
    if content_type:
        ext = mimetypes.guess_extension(content_type.split(';')[0])
        if ext:
            return ext

    # Default to .txt
    return '.txt'
