# fetch.py — HTTP/JS fetching, Reddit API, polite crawler
from __future__ import annotations
import asyncio, io, os, re
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse, urljoin, urlunparse, urlencode
from collections import deque

import requests
from bs4 import BeautifulSoup
import urllib.robotparser as robotparser

from playwright.async_api import async_playwright, Browser

HTTP_TIMEOUT = (10, 120)                 # (connect, read)
MAX_DOWNLOAD_BYTES = 100 * 1024 * 1024   # 100 MB
REDDIT_LIMIT = int(os.getenv("REDDIT_LIMIT", "200"))

DEFAULT_FETCH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}

# ---------- Generic fetch (requests) ----------
def _infer_kind(url: str, headers: Dict[str, str]) -> Optional[str]:
    ct = (headers.get("Content-Type") or "").split(";")[0].strip().lower()
    if ct in {"application/json","text/json"}: return "json"
    if ct in {"text/csv","application/csv"}:   return "csv"
    if ct in {"text/html"}:                    return "html"
    low = url.lower()
    if low.endswith(".json"): return "json"
    if low.endswith(".csv"):  return "csv"
    if low.endswith((".htm",".html")): return "html"
    return None

def fetch_url_bytes_sync(url: str) -> Tuple[bytes, str]:
    with requests.get(url, stream=True, timeout=HTTP_TIMEOUT, headers=DEFAULT_FETCH_HEADERS) as r:
        r.raise_for_status()
        kind = _infer_kind(url, r.headers)
        buf = bytearray()
        for chunk in r.iter_content(8192):
            if chunk:
                buf += chunk
                if len(buf) > MAX_DOWNLOAD_BYTES:
                    raise RuntimeError("Download too large (>100MB)")
        if kind is None:
            head = bytes(buf[:2048]).lstrip().lower()
            if head.startswith((b"{", b"[")): kind = "json"
            elif b"<html" in head or b"<!doctype html" in head: kind = "html"
            else: kind = "csv"
        return bytes(buf), kind

async def fetch_url_bytes(url: str) -> Tuple[bytes, str]:
    return await asyncio.to_thread(fetch_url_bytes_sync, url)

# ---------- Playwright (async) ----------
async def ensure_browser(app) -> Browser:
    if not hasattr(app.state, "playwright") or app.state.playwright is None:
        app.state.playwright = await async_playwright().start()
    if not hasattr(app.state, "browser") or app.state.browser is None:
        app.state.browser = await app.state.playwright.chromium.launch(
            headless=True, args=["--no-sandbox"]
        )
    return app.state.browser

async def shutdown_playwright(app):
    try:
        if getattr(app.state, "browser", None):
            await app.state.browser.close()
    finally:
        if getattr(app.state, "playwright", None):
            await app.state.playwright.stop()

async def fetch_url_bytes_rendered(
    url: str,
    browser: Browser,
    timeout_ms: int = 20000,
    cookies_header: Optional[str] = None,
) -> Tuple[bytes, str]:
    host = (urlparse(url).hostname or "").lstrip(".")
    ctx = await browser.new_context(
        user_agent=DEFAULT_FETCH_HEADERS["User-Agent"],
        locale="en-US",
    )
    await ctx.set_extra_http_headers({
        k: v for k, v in DEFAULT_FETCH_HEADERS.items() if k.lower() != "user-agent"
    })
    if cookies_header:
        jar = []
        for part in cookies_header.split(";"):
            if "=" in part:
                name, val = part.split("=", 1)
                jar.append({
                    "name": name.strip(),
                    "value": val.strip(),
                    "domain": "." + host if not host.startswith(".") else host,
                    "path": "/",
                    "httpOnly": False,
                    "secure": True,
                    "sameSite": "Lax",
                })
        if jar: await ctx.add_cookies(jar)
    page = await ctx.new_page()
    await page.goto(url, wait_until="networkidle", timeout=timeout_ms)
    html = (await page.content()).encode("utf-8", errors="ignore")
    await ctx.close()
    return html, "html"

# ---------- Reddit helpers ----------
def _reddit_api_url(url: str, limit: int, after: Optional[str] = None) -> str:
    p = urlparse(url)
    path = p.path.rstrip("/")
    if not path.endswith(".json"):
        path += "/.json"
    qs = {"limit": max(1, min(100, limit))}
    if after: qs["after"] = after
    return urlunparse(("https", "www.reddit.com", path, "", urlencode(qs), ""))

def fetch_reddit_texts_sync(url: str, limit: int = REDDIT_LIMIT) -> List[str]:
    texts: List[str] = []
    after = None
    fetched = 0
    sess = requests.Session(); sess.headers.update(DEFAULT_FETCH_HEADERS)

    for _ in range(20):  # paginate
        api = _reddit_api_url(url, limit=min(100, max(0, limit - fetched)), after=after)
        r = sess.get(api, timeout=HTTP_TIMEOUT); r.raise_for_status(); j = r.json()

        if isinstance(j, list) and len(j) >= 2:
            post = j[0]["data"]["children"][0]["data"]
            title = (post.get("title") or "").strip()
            body  = (post.get("selftext") or "").strip()
            main  = (title + ("\n\n"+body if body else "")).strip()
            if len(main.split()) >= 4: texts.append(main)
            for c in j[1]["data"]["children"]:
                if c.get("kind") == "t1":
                    body = (c["data"].get("body") or "").strip()
                    if len(body.split()) >= 4: texts.append(body)
            break

        data = j["data"]; children = data.get("children", [])
        for ch in children:
            d = ch.get("data", {})
            title = (d.get("title") or "").strip()
            body  = (d.get("selftext") or "").strip()
            txt = (title + ("\n\n"+body if body else "")).strip()
            if len(txt.split()) >= 4: texts.append(txt)

        fetched += len(children)
        after = data.get("after")
        if not after or fetched >= limit: break
    return texts

async def fetch_reddit_texts(url: str, limit: int = REDDIT_LIMIT) -> List[str]:
    return await asyncio.to_thread(fetch_reddit_texts_sync, url, limit)

# ---------- Polite crawler ----------
class CrawlResult(Tuple[str, bytes, str]): pass
#   element structure: (url, content_bytes, kind)

async def crawl_site(
    start_url: str,
    *,
    app,
    max_pages: int = 50,
    max_depth: int = 2,
    same_host_only: bool = True,
    delay_ms: int = 300,
    render: bool = False,
    cookies: Optional[str] = None,
) -> List[Tuple[str, bytes, str]]:
    """
    BFS crawl within constraints. Respects robots.txt (best-effort).
    Robust: per-page try/except, skips binary/oversized.
    """
    start = urlparse(start_url)
    start_host = (start.hostname or "").lower()

    # robots.txt
    rp = robotparser.RobotFileParser()
    rp.set_url(urlunparse((start.scheme, start.netloc, "/robots.txt", "", "", "")))
    try:
        rp.read()
    except Exception:
        pass  # be permissive if robots fetch fails

    seen = set()
    q = deque([(start_url, 0)])
    results: List[Tuple[str, bytes, str]] = []

    # Preload browser if render
    browser = await ensure_browser(app) if render else None

    while q and len(results) < max_pages:
        url, depth = q.popleft()
        if url in seen: continue
        seen.add(url)

        # robots check
        try:
            if hasattr(rp, "can_fetch") and not rp.can_fetch(DEFAULT_FETCH_HEADERS["User-Agent"], url):
                continue
        except Exception:
            pass

        # fetch
        try:
            if render:
                assert browser is not None
                raw, kind = await fetch_url_bytes_rendered(url, browser, cookies_header=cookies)
            else:
                raw, kind = await fetch_url_bytes(url)
        except Exception:
            continue  # skip on fetch errors

        # keep only text-y kinds
        if kind not in {"html", "json", "csv"}: continue
        results.append((url, raw, kind))

        # enqueue links if HTML
        if kind == "html" and depth < max_depth:
            try:
                soup = BeautifulSoup(raw, "lxml")
                for a in soup.find_all("a", href=True):
                    href = a.get("href")
                    if href.startswith("mailto:") or href.startswith("javascript:"): continue
                    nxt = urljoin(url, href)
                    p = urlparse(nxt)
                    if same_host_only and (p.hostname or "").lower() != start_host: continue
                    if p.scheme not in {"http", "https"}: continue
                    if nxt not in seen:
                        q.append((nxt, depth + 1))
            except Exception:
                pass

        # small delay to be polite
        if delay_ms > 0:
            await asyncio.sleep(delay_ms / 1000.0)

    return results
