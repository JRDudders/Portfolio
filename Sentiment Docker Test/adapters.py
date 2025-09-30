# adapters.py — pluggable registry + adapters (generic + reddit, with crawl support)
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Type, Protocol
import re
from urllib.parse import urlparse
from pathlib import Path

from fastapi import FastAPI

# imports from our modules
from nlp import preprocess_text, classify_texts
from fetch import (
    fetch_url_bytes, fetch_url_bytes_rendered, ensure_browser,
    fetch_reddit_texts, crawl_site
)
from render import html_to_paragraphs, extract_title, render_annotated_html, render_site_report
from apputils import process_csv_bytes, process_json_bytes, filename_with_suffix  # see below

@dataclass
class AdapterOutput:
    content: bytes
    media_type: str
    filename: str

class Adapter(Protocol):
    domains: Tuple[str, ...]
    paths: Tuple[str, ...]
    async def process(self, url: str, *, app: FastAPI, render: bool,
                      cookies: Optional[str], crawl: bool,
                      max_pages: int, max_depth: int, same_host_only: bool,
                      delay_ms: int) -> AdapterOutput: ...

_REGISTRY: List[Type[Adapter]] = []

def register(*, domains: Tuple[str, ...], paths: Tuple[str, ...] = ()):
    def deco(cls: Type[Adapter]):
        cls.domains = domains; cls.paths = paths
        _REGISTRY.append(cls); return cls
    return deco

def _matches(cls: Type[Adapter], url: str) -> bool:
    p = urlparse(url); host = (p.hostname or "").lower(); path = p.path or "/"
    return any(host.endswith(d) for d in cls.domains) and (not cls.paths or any(re.search(rx, path) for rx in cls.paths))

def pick_adapter(url: str) -> Type[Adapter]:
    for cls in _REGISTRY:
        if _matches(cls, url): return cls
    return GenericAdapter

async def process_url(url: str, *, app: FastAPI, render: bool, cookies: Optional[str],
                      crawl: bool, max_pages: int, max_depth: int, same_host_only: bool, delay_ms: int) -> AdapterOutput:
    adapter = pick_adapter(url)()
    return await adapter.process(url, app=app, render=render, cookies=cookies,
                                 crawl=crawl, max_pages=max_pages, max_depth=max_depth,
                                 same_host_only=same_host_only, delay_ms=delay_ms)

# ---------- Reddit ----------
@register(domains=("reddit.com","old.reddit.com"))
class RedditAdapter:
    async def process(self, url: str, *, app: FastAPI, render: bool, cookies: Optional[str],
                      crawl: bool, max_pages: int, max_depth: int, same_host_only: bool, delay_ms: int) -> AdapterOutput:
        texts = await fetch_reddit_texts(url)
        if not texts:
            raise RuntimeError("No posts/comments found")
        preds = classify_texts([preprocess_text(t) for t in texts])
        title = (urlparse(url).path.rstrip("/") or "/").split("/")[-1] or "Reddit"
        html = render_annotated_html(title, url, texts, preds).encode("utf-8")
        name = filename_with_suffix(Path(url).name or "reddit", "html")
        return AdapterOutput(html, "text/html; charset=utf-8", name)

# ---------- Generic (JSON/CSV/HTML + optional crawl) ----------
@register(domains=("",))
class GenericAdapter:
    async def process(self, url: str, *, app: FastAPI, render: bool, cookies: Optional[str],
                      crawl: bool, max_pages: int, max_depth: int, same_host_only: bool, delay_ms: int) -> AdapterOutput:
        if crawl:
            # Crawl site and aggregate into a single report
            pages = await crawl_site(url, app=app, max_pages=max_pages, max_depth=max_depth,
                                     same_host_only=same_host_only, delay_ms=delay_ms,
                                     render=render, cookies=cookies)
            rendered: List[tuple[str, List[str], List[dict]]] = []
            for page_url, raw, kind in pages:
                if kind == "json":
                    # classify values if list/objects -> stringify minimal view
                    try:
                        # light extract: prefer treating as paragraphs of text
                        import json as _json
                        data = _json.loads(raw.decode("utf-8"))
                        if isinstance(data, list):
                            strs = [str(x) for x in data][:500]
                        else:
                            strs = [str(data)]
                    except Exception:
                        strs = [raw[:1000].decode("utf-8", "ignore")]
                    paras = strs
                elif kind == "csv":
                    # don’t explode CSV here – keep brief
                    paras = [raw[:1200].decode("utf-8", "ignore")]
                else:
                    paras = html_to_paragraphs(raw)
                preds = classify_texts([preprocess_text(p) for p in paras])
                rendered.append((page_url, paras, preds))
            report = render_site_report(rendered).encode("utf-8")
            name = filename_with_suffix(Path(url).name or "site", "html")
            return AdapterOutput(report, "text/html; charset=utf-8", name)

        # Single page/file flow
        raw, kind = await fetch_url_bytes(url)
        if kind == "html":
            head = raw[:4096].lower()
            if render or any(x in head for x in [b"enable javascript", b"turn on javascript", b"noscript",
                                                 b"cloudflare", b"bot detection", b"checking your browser"]):
                browser = await ensure_browser(app)
                raw, kind = await fetch_url_bytes_rendered(url, browser, cookies_header=cookies)

        if kind == "json":
            out = process_json_bytes(raw); name = filename_with_suffix(Path(url).name or "input.json", "json")
            return AdapterOutput(out, "application/json; charset=utf-8", name)

        if kind == "csv":
            out = process_csv_bytes(raw); name = filename_with_suffix(Path(url).name or "input.csv", "csv")
            return AdapterOutput(out, "text/csv; charset=utf-8", name)

        # HTML
        paras = html_to_paragraphs(raw)
        preds = classify_texts([preprocess_text(p) for p in paras])
        title = extract_title(raw) or (Path(url).name or "Scored Page")
        html = render_annotated_html(title, url, paras, preds).encode("utf-8")
        name = filename_with_suffix(Path(url).name or "page.html", "html")
        return AdapterOutput(html, "text/html; charset=utf-8", name)
