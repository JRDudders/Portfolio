# adapters.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Type, Protocol, Dict, Any
import re
from urllib.parse import urlparse
from pathlib import Path

from fastapi import FastAPI

from nlp import preprocess_for_task, run_task, DEFAULT_ZS_LABELS
from fetch import (
    fetch_url_bytes, fetch_url_bytes_rendered, ensure_browser,
    fetch_reddit_texts, crawl_site, fetch_url_bytes_rendered_selenium
)
from render import (
    html_to_paragraphs, extract_title, render_annotated_html, render_site_report,
    guess_labels_from_html
)
from apputils import process_csv_bytes, process_json_bytes, filename_with_suffix

@dataclass
class AdapterOutput:
    content: bytes
    media_type: str
    filename: str

class Adapter(Protocol):
    domains: Tuple[str, ...]
    paths: Tuple[str, ...]
    async def process(self, url: str, *, app: FastAPI,
                      render: bool, renderer: str, cookies: Optional[str],
                      crawl: bool, max_pages: int, max_depth: int, same_host_only: bool, delay_ms: int,
                      wait_selector: Optional[str], scroll_passes: int, render_timeout_ms: int,
                      extra_headers: Optional[Dict[str, str]],
                      task: Optional[str], preset: Optional[str], labels: Optional[List[str]],
                      include_stopwords: bool) -> AdapterOutput: ...

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

def _resolve_zero_shot_labels(labels: Optional[List[str]], html_raw: bytes, include_stopwords: bool) -> List[str]:
    if labels and isinstance(labels, list) and len(labels) > 0:
        return [str(x).strip() for x in labels if str(x).strip()]
    auto = guess_labels_from_html(html_raw, k=12, include_stopwords=include_stopwords)
    if auto:
        merged, seen = [], set()
        for w in auto + DEFAULT_ZS_LABELS:
            if w not in seen:
                seen.add(w); merged.append(w)
            if len(merged) >= 12:
                break
        return merged
    return DEFAULT_ZS_LABELS[:12]

def _looks_nav_only(html: bytes) -> bool:
    low = html[:20000].lower()
    if len(low) < 500: return True
    # consent / noscript / skeleton-only clues
    if b"enable javascript" in low or b"turn on javascript" in low or b"noscript" in low:
        return True
    # missing common content containers
    containers = (b"<article", b"role=\"main\"", b"entry-content", b"post-content", b"story-content")
    return not any(c in low for c in containers)

@register(domains=("reddit.com", "old.reddit.com"))
class RedditAdapter:
    async def process(self, url: str, *, app: FastAPI,
                      render: bool, renderer: str, cookies: Optional[str],
                      crawl: bool, max_pages: int, max_depth: int, same_host_only: bool, delay_ms: int,
                      wait_selector: Optional[str], scroll_passes: int, render_timeout_ms: int,
                      extra_headers: Optional[Dict[str, str]],
                      task: Optional[str], preset: Optional[str], labels: Optional[List[str]],
                      include_stopwords: bool) -> AdapterOutput:
        texts = await fetch_reddit_texts(url)
        if not texts:
            raise RuntimeError("No posts/comments found")
        effective_task = task or "text-classification"
        proc = [preprocess_for_task(t, effective_task) for t in texts]
        preds = run_task(proc, task=effective_task, preset=preset, labels=labels)
        title = (urlparse(url).path.rstrip("/") or "/").split("/")[-1] or "Reddit"
        html = render_annotated_html(title, url, texts, preds, task=effective_task).encode("utf-8")
        name = filename_with_suffix(Path(url).name or "reddit", "html")
        return AdapterOutput(html, "text/html; charset=utf-8", name)

@register(domains=("",))
class GenericAdapter:
    async def process(self, url: str, *, app: FastAPI,
                      render: bool, renderer: str, cookies: Optional[str],
                      crawl: bool, max_pages: int, max_depth: int, same_host_only: bool, delay_ms: int,
                      wait_selector: Optional[str], scroll_passes: int, render_timeout_ms: int,
                      extra_headers: Optional[Dict[str, str]],
                      task: Optional[str], preset: Optional[str], labels: Optional[List[str]],
                      include_stopwords: bool) -> AdapterOutput:

        # Crawl mode
        if crawl:
            pages = await crawl_site(url, app=app, max_pages=max_pages, max_depth=max_depth,
                                     same_host_only=same_host_only, delay_ms=delay_ms,
                                     render=render, cookies=cookies)
            rendered: List[tuple[str, List[str], List[dict], str]] = []
            for page_url, raw, kind in pages:
                if kind == "json":
                    import json as _json
                    try:
                        data = _json.loads(raw.decode("utf-8"))
                        paras = [str(x) for x in (data if isinstance(data, list) else [data])][:500]
                    except Exception:
                        paras = [raw[:1000].decode("utf-8", "ignore")]
                elif kind == "csv":
                    paras = [raw[:1200].decode("utf-8", "ignore")]
                else:
                    paras = html_to_paragraphs(raw)

                effective_task = task or "text-classification"
                eff_labels: Optional[List[str]] = None
                if effective_task == "zero-shot-classification" or (preset and preset.startswith("zeroshot-")):
                    html_raw = raw if kind == "html" else b""
                    eff_labels = _resolve_zero_shot_labels(labels, html_raw, include_stopwords)

                proc = [preprocess_for_task(p, effective_task) for p in paras]
                preds = run_task(proc, task=effective_task, preset=preset, labels=eff_labels)
                rendered.append((page_url, paras, preds, effective_task))
            report = render_site_report(rendered).encode("utf-8")
            name = filename_with_suffix(Path(url).name or "site", "html")
            return AdapterOutput(report, "text/html; charset=utf-8", name)

        # Single page/file
        raw, kind = await fetch_url_bytes(url)

        if kind == "html":
            async def _render_playwright():
                browser = await ensure_browser(app)
                return await fetch_url_bytes_rendered(
                    url, browser,
                    timeout_ms=render_timeout_ms,
                    cookies_header=cookies,
                    wait_selector=wait_selector,
                    scroll_passes=scroll_passes,
                    extra_headers=extra_headers,
                )

            async def _render_selenium():
                return await fetch_url_bytes_rendered_selenium(
                    url,
                    cookies_header=cookies,
                    wait_selector=wait_selector,
                    scroll_passes=scroll_passes,
                    timeout_ms=render_timeout_ms,
                    extra_headers=extra_headers,
                )

            # explicit renderer
            if render and renderer in {"playwright","selenium"}:
                raw, kind = await (_render_playwright() if renderer == "playwright" else _render_selenium())
            # auto upgrade when nav-only OR render=true
            elif render or _looks_nav_only(raw):
                try:
                    raw, kind = await _render_playwright()
                except Exception:
                    raw, kind = await _render_selenium()

        if kind == "json":
            out = process_json_bytes(raw, task=task, preset=preset, labels=labels)
            name = filename_with_suffix(Path(url).name or "input.json", "json")
            return AdapterOutput(out, "application/json; charset=utf-8", name)

        if kind == "csv":
            out = process_csv_bytes(raw, task=task, preset=preset, labels=labels)
            name = filename_with_suffix(Path(url).name or "input.csv", "csv")
            return AdapterOutput(out, "text/csv; charset=utf-8", name)

        # HTML
        paras = html_to_paragraphs(raw)
        effective_task = task or "text-classification"
        eff_labels: Optional[List[str]] = None
        if effective_task == "zero-shot-classification" or (preset and preset.startswith("zeroshot-")):
            eff_labels = _resolve_zero_shot_labels(labels, raw, include_stopwords)
        proc = [preprocess_for_task(p, effective_task) for p in paras]
        preds = run_task(proc, task=effective_task, preset=preset, labels=eff_labels)
        title = extract_title(raw) or (Path(url).name or "Scored Page")
        html = render_annotated_html(title, url, paras, preds, task=effective_task).encode("utf-8")
        name = filename_with_suffix(Path(url).name or "page.html", "html")
        return AdapterOutput(html, "text/html; charset=utf-8", name)

# keep this shim so app.py imports stay stable
async def process_url(
    url: str, *, app: FastAPI,
    render: bool, renderer: str, cookies: Optional[str],
    crawl: bool, max_pages: int, max_depth: int, same_host_only: bool, delay_ms: int,
    wait_selector: Optional[str], scroll_passes: int, render_timeout_ms: int,
    extra_headers: Optional[Dict[str, str]],
    task: Optional[str], preset: Optional[str], labels: Optional[List[str]],
    include_stopwords: bool,
) -> AdapterOutput:
    adapter = pick_adapter(url)()
    return await adapter.process(
        url, app=app, render=render, renderer=renderer, cookies=cookies,
        crawl=crawl, max_pages=max_pages, max_depth=max_depth, same_host_only=same_host_only, delay_ms=delay_ms,
        wait_selector=wait_selector, scroll_passes=scroll_passes, render_timeout_ms=render_timeout_ms,
        extra_headers=extra_headers,
        task=task, preset=preset, labels=labels, include_stopwords=include_stopwords,
    )

__all__ = ["process_url", "AdapterOutput", "pick_adapter", "register"]
