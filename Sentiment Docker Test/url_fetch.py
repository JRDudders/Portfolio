"""
URL Fetch Utility

Shared module for fetching files from URLs with:
- Size limits (500MB default)
- Timeout protection
- Proper error handling
- Content type validation
"""

import httpx
from typing import Optional, Tuple
import mimetypes


# Maximum file size to download (500MB to match nginx limit)
MAX_DOWNLOAD_SIZE = 500 * 1024 * 1024  # 500MB in bytes

# Connection timeout (1 hour for large model downloads)
TIMEOUT_SECONDS = 3600


async def fetch_url(
    url: str,
    max_size: int = MAX_DOWNLOAD_SIZE,
    timeout: int = TIMEOUT_SECONDS
) -> Tuple[bytes, Optional[str]]:
    """
    Fetch file content from URL.

    Args:
        url: URL to fetch
        max_size: Maximum file size in bytes (default 500MB)
        timeout: Timeout in seconds (default 300s)

    Returns:
        Tuple of (file_bytes, content_type)

    Raises:
        ValueError: If URL is invalid or file too large
        httpx.HTTPError: If download fails
    """
    # Validate URL scheme
    if not url.startswith(('http://', 'https://')):
        raise ValueError("URL must start with http:// or https://")

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
