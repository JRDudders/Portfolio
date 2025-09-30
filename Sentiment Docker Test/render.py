# render.py — HTML extraction + annotated HTML report
from __future__ import annotations
from typing import Any, Dict, List, Optional
import os, re, html as htmlmod
from pathlib import Path

try:
    import trafilatura
except Exception:
    trafilatura = None
try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

MAX_HTML_ITEMS = int(os.getenv("MAX_HTML_ITEMS", "300"))

def html_to_paragraphs(html_bytes: bytes) -> List[str]:
    text = ""
    if trafilatura is not None:
        try:
            extracted = trafilatura.extract(html_bytes, include_comments=False, include_tables=False)
            if extracted: text = extracted
        except Exception:
            text = ""
    if not text:
        if BeautifulSoup is None:
            raise RuntimeError("HTML parsing requires trafilatura or beautifulsoup4")
        soup = BeautifulSoup(html_bytes, "lxml" if BeautifulSoup else "html.parser")
        for tag in soup(["script","style","noscript"]): tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    paras = [p for p in paras if len(p.split()) >= 4]
    return paras[:MAX_HTML_ITEMS]

def extract_title(html_bytes: bytes) -> Optional[str]:
    try:
        if BeautifulSoup:
            soup = BeautifulSoup(html_bytes, "lxml" if BeautifulSoup else "html.parser")
            if soup.title and soup.title.string:
                return soup.title.string.strip()
    except Exception:
        pass
    return None

def _sentiment_color(scores: Dict[str, float]) -> str:
    p_pos = scores.get("positive", 0.0); p_neg = scores.get("negative", 0.0)
    polarity = max(-1.0, min(1.0, p_pos - p_neg))
    hue = (polarity + 1.0) * 60.0
    sat = 45
    light = 22
    return f"hsl({hue:.0f} {sat}% {light}%)"

def render_annotated_html(title: str, source: str,
                          paras: List[str], preds: List[Dict[str, Any]]) -> str:
    rows = []
    for t, p in zip(paras, preds):
        col = _sentiment_color(p["scores"])
        label = htmlmod.escape(p["top"]["label"])
        score = f'{p["top"]["score"]:.3f}'
        esc = htmlmod.escape(t)
        rows.append(
            '<article class="item" style="--col:%s">'
            '<div class="badge">%s %s</div>'
            '<div class="txt">%s</div>'
            '</article>' % (col, label, score, esc)
        )
    CSS = (
        "<style>\n"
        "  :root{--bg:#0b0c10;--fg:#e6e6e6;--muted:#9aa0a6;--card:#13151b;--stroke:#23262d;}\n"
        "  *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,-apple-system,Segoe UI,Roboto,Arial}\n"
        "  .wrap{max-width:1100px;margin:32px auto;padding:0 16px}\n"
        "  h1{margin:0 0 6px;font-size:26px} .meta{color:var(--muted);margin-bottom:18px}\n"
        "  .item{position:relative;border:1px solid var(--stroke);border-left:8px solid var(--col);background:var(--card);border-radius:12px;padding:14px 14px 14px 16px;margin:14px 0}\n"
        "  .badge{position:absolute;top:8px;right:8px;background:rgba(0,0,0,.2);color:#e9eefc;border:1px solid var(--stroke);border-radius:999px;padding:3px 8px;font-size:12px}\n"
        "  .txt{white-space:pre-wrap;margin:0}\n"
        "</style>\n"
    )
    title_esc = htmlmod.escape(title or "Scored Page")
    source_esc = htmlmod.escape(source or "")
    parts = [
        "<!doctype html>",
        '<html lang="en"><meta charset="utf-8">',
        f"<title>{title_esc}</title>",
        CSS,
        '<div class="wrap">',
        f"<h1>{title_esc}</h1>",
        f'<div class="meta">{source_esc}</div>',
        *rows,
        "</div></html>",
    ]
    return "\n".join(parts)

def render_site_report(pages: List[tuple[str, List[str], List[Dict[str, Any]]]]) -> str:
    """pages: list of (url, paragraphs, preds) → single HTML report with sections"""
    sections = []
    for url, paras, preds in pages:
        title = htmlmod.escape(url)
        block = render_annotated_html(title, url, paras, preds)
        # strip <html> wrapper to embed as section
        inner = block.split('<div class="wrap">', 1)[-1].rsplit("</div></html>", 1)[0]
        sections.append(inner)
    shell = (
        "<!doctype html><html lang='en'><meta charset='utf-8'>"
        "<title>Site Sentiment Report</title>"
        "<body style='margin:0;background:#0b0c10;color:#e6e6e6;font:16px/1.5 system-ui,-apple-system,Segoe UI,Roboto,Arial'>"
        + "".join(sections) + "</body></html>"
    )
    return shell
