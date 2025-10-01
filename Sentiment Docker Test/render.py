# render.py — HTML extraction + annotated renderers (classification + NER) + auto-labels
from __future__ import annotations
from typing import Any, Dict, List, Optional
import os, re, html as htmlmod
from hashlib import md5
from collections import Counter

# Optional extractors
try:
    import trafilatura
except Exception:
    trafilatura = None
try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

# NLTK stopwords (we keep corpora in Docker; locally, we fall back to empty set)
try:
    from nltk.corpus import stopwords
    try:
        _STOP = set(stopwords.words("english"))
    except Exception:
        _STOP = set()
except Exception:
    _STOP = set()

MAX_HTML_ITEMS = int(os.getenv("MAX_HTML_ITEMS", "300"))

def html_to_paragraphs(html_bytes: bytes) -> List[str]:
    text = ""
    if trafilatura is not None:
        try:
            extracted = trafilatura.extract(html_bytes, include_comments=False, include_tables=False)
            if extracted:
                text = extracted
        except Exception:
            text = ""
    if not text:
        if BeautifulSoup is None:
            raise RuntimeError("HTML parsing requires trafilatura or beautifulsoup4")
        soup = BeautifulSoup(html_bytes, "lxml" if BeautifulSoup else "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
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

def _simple_tokens(s: str) -> List[str]:
    return [t.strip("-") for t in re.findall(r"[A-Za-z][A-Za-z\-]{2,}", s.lower())]

def guess_labels_from_html(html_bytes: bytes, k: int = 12, include_stopwords: bool = False) -> List[str]:
    if BeautifulSoup is None:
        return []
    soup = BeautifulSoup(html_bytes, "lxml")
    parts: List[str] = []
    if soup.title and soup.title.string:
        parts.append(soup.title.string)
    for m in soup.find_all("meta"):
        name = (m.get("name") or "").lower()
        prop = (m.get("property") or "").lower()
        if name in {"keywords", "news_keywords"}:
            parts.append(m.get("content") or "")
        if prop in {"og:title", "og:description", "twitter:title", "twitter:description"}:
            parts.append(m.get("content") or "")
    for h in soup.find_all(["h1", "h2", "h3"]):
        parts.append(h.get_text(" ", strip=True))

    text = " ".join(parts)
    toks = _simple_tokens(text)
    if not include_stopwords:
        toks = [t for t in toks if t not in _STOP]
    if not toks:
        return []
    top = [w for (w, _) in Counter(toks).most_common(60)]
    out, seen = [], set()
    for w in top:
        if w not in seen:
            seen.add(w)
            out.append(w)
        if len(out) >= k:
            break
    return out

def _sentiment_hsl(scores: Dict[str, float]) -> str:
    p_pos = scores.get("positive", 0.0)
    p_neg = scores.get("negative", 0.0)
    polarity = max(-1.0, min(1.0, p_pos - p_neg))   # [-1,1]
    hue = (polarity + 1.0) * 60.0                   # 0..120
    sat = 45
    light = 22
    return f"hsl({hue:.0f} {sat}% {light}%)"

def _label_color(label: str) -> str:
    h = int(md5(label.encode("utf-8")).hexdigest(), 16) % 360
    return f"hsl({h} 50% 30%)"

_BASE_CSS = (
    "<style>\n"
    "  :root{--bg:#0b0c10;--fg:#e6e6e6;--muted:#9aa0a6;--card:#13151b;--stroke:#23262d;}\n"
    "  *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,-apple-system,Segoe UI,Roboto,Arial}\n"
    "  .wrap{max-width:1100px;margin:32px auto;padding:0 16px}\n"
    "  h1{margin:0 0 6px;font-size:26px} .meta{color:var(--muted);margin-bottom:18px}\n"
    "  .item{position:relative;border:1px solid var(--stroke);border-left:8px solid var(--col);background:var(--card);border-radius:12px;padding:14px 14px 14px 16px;margin:14px 0}\n"
    "  .badge{position:absolute;top:8px;right:8px;background:rgba(0,0,0,.2);color:#e9eefc;border:1px solid var(--stroke);border-radius:999px;padding:3px 8px;font-size:12px}\n"
    "  .txt{white-space:pre-wrap;margin:0}\n"
    "  mark.ent{padding:0 .2em .05em;border-radius:.2em;border-bottom:1px dotted rgba(255,255,255,.25)}\n"
    "</style>\n"
)

def render_annotated_html_classification(title: str, source: str,
                                         paras: List[str], preds: List[Dict[str, Any]]) -> str:
    rows: List[str] = []
    for t, p in zip(paras, preds):
        col = _sentiment_hsl(p.get("scores", {}))
        label = htmlmod.escape(p.get("top", {}).get("label", ""))
        score = f'{p.get("top", {}).get("score", 0.0):.3f}'
        esc = htmlmod.escape(t)
        rows.append(
            '<article class="item" style="--col:%s">'
            '<div class="badge">%s %s</div>'
            '<div class="txt">%s</div>'
            '</article>' % (col, label, score, esc)
        )
    title_esc = htmlmod.escape(title or "Scored Page")
    source_esc = htmlmod.escape(source or "")
    return (
        "<!doctype html><html lang='en'><meta charset='utf-8'>"
        f"<title>{title_esc}</title>"
        f"{_BASE_CSS}"
        "<div class='wrap'>"
        f"<h1>{title_esc}</h1><div class='meta'>{source_esc}</div>"
        + "".join(rows) + "</div></html>"
    )

def _annotate_text_with_entities(txt: str, ents: List[Dict[str, Any]]) -> str:
    ents = [e for e in ents if 0 <= int(e.get("start", 0)) < int(e.get("end", 0)) <= len(txt)]
    ents.sort(key=lambda e: (int(e.get("start", 0)), -int(e.get("end", 0))))
    out: List[str] = []
    pos = 0
    for e in ents:
        s = int(e.get("start", 0)); eidx = int(e.get("end", 0))
        if s > pos:
            out.append(htmlmod.escape(txt[pos:s]))
        span = htmlmod.escape(txt[s:eidx])
        lbl = htmlmod.escape(e.get("label", "ENT"))
        score = f'{float(e.get("score", 0.0)):.3f}'
        color = _label_color(lbl)
        out.append(f"<mark class='ent' style='background:color-mix(in oklab,{color} 35%, #1a1d24)' title='{lbl} {score}'>{span}</mark>")
        pos = eidx
    if pos < len(txt):
        out.append(htmlmod.escape(txt[pos:]))
    return "".join(out)

def render_annotated_html_ner(title: str, source: str,
                              paras: List[str], preds: List[Dict[str, Any]]) -> str:
    rows: List[str] = []
    for t, p in zip(paras, preds):
        ents = p.get("entities", [])
        col = "#3b82f6"
        esc = _annotate_text_with_entities(t, ents)
        counts: Dict[str, int] = {}
        for e in ents:
            counts[e.get("label", "ENT")] = counts.get(e.get("label", "ENT"), 0) + 1
        badge = ", ".join(f"{htmlmod.escape(k)}:{v}" for k, v in sorted(counts.items()))
        rows.append(
            '<article class="item" style="--col:%s">'
            '<div class="badge">%s</div>'
            '<div class="txt">%s</div>'
            '</article>' % (col, badge or "entities:0", esc)
        )
    title_esc = htmlmod.escape(title or "NER Page")
    source_esc = htmlmod.escape(source or "")
    return (
        "<!doctype html><html lang='en'><meta charset='utf-8'>"
        f"<title>{title_esc}</title>"
        f"{_BASE_CSS}"
        "<div class='wrap'>"
        f"<h1>{title_esc}</h1><div class='meta'>{source_esc}</div>"
        + "".join(rows) + "</div></html>"
    )

def render_annotated_html(title: str, source: str,
                          paras: List[str], preds: List[Dict[str, Any]],
                          task: Optional[str] = "text-classification") -> str:
    if task == "token-classification":
        return render_annotated_html_ner(title, source, paras, preds)
    return render_annotated_html_classification(title, source, paras, preds)

def render_site_report(pages: List[tuple[str, List[str], List[Dict[str, Any]], str]]) -> str:
    # pages: (url, paragraphs, preds, task)
    sections: List[str] = []
    for url, paras, preds, task in pages:
        title = url
        block = render_annotated_html(title, url, paras, preds, task)
        inner = block.split('<div class="wrap">', 1)[-1].rsplit("</div></html>", 1)[0]
        sections.append(inner)
    return ("<!doctype html><html lang='en'><meta charset='utf-8'>"
            "<title>Site Report</title>"
            "<body style='margin:0;background:#0b0c10;color:#e6e6e6;font:16px/1.5 system-ui,-apple-system,Segoe UI,Roboto,Arial'>"
            + "".join(sections) + "</body></html>")
