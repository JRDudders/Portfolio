# Sentiment Processor (FastAPI + Transformers)

A small, production-friendly FastAPI service that:
- Scores **JSON/CSV** text with a Hugging Face **Transformers** pipeline.
- Annotates **web pages** (and entire sites, optionally) with per-paragraph sentiment.
- Handles **JS-heavy pages** via Playwright/Chromium.
- Is **extensible** via a pluggable **adapter registry** (e.g., Reddit adapter included).

The web UI (`index.html`) supports file upload, URL processing, optional cookies, and crawl controls.

---

## Features

- **Model:** `cardiffnlp/twitter-roberta-base-sentiment-latest` (3-way sentiment)
- **Inputs:** JSON (list of strings or list of objects with `text`/`tweet`/`content`), CSV (auto-detects text column), HTML, or URL
- **Outputs:** Same type for JSON/CSV; **annotated HTML** for webpages / crawled sites
- **JS rendering:** Playwright + Chromium
- **Crawling:** polite BFS crawl (robots-aware best-effort), depth/page/host limits, delay
- **Adapters:** Pluggable domain handlers (Generic, Reddit included)

---

## Repo Layout

```
.
├─ app.py              # Thin FastAPI app (routes only)
├─ index.html          # Minimal front-end (file upload, URL, crawl controls)
├─ nlp.py              # Model init, preprocessing, batched inference
├─ fetch.py            # requests/Playwright fetchers, Reddit API, crawler
├─ render.py           # HTML extraction + annotated HTML renderers
├─ adapters.py         # Pluggable registry (Generic + Reddit)
├─ apputils.py         # CSV/JSON processors + small helpers
├─ req.txt             # Python dependencies
└─ Dockerfile.cpu      # Container build (CPU)
```

---

## Quickstart (Docker)

> Requires Docker Desktop (Windows/macOS) or Docker Engine (Linux).

```bash
# From repo root
docker build -f Dockerfile.cpu -t cicerowatch5/pipeline:cpu .
docker run --rm -p 8080:8080 --name hfpipe cicerowatch5/pipeline:cpu
```

Open http://localhost:8080 — use the UI to upload a file or process a URL.

**Notes (Windows):**
- Make sure Docker Desktop is **running** and you’re using a valid context:
  ```powershell
  docker context ls
  docker version      # should show a Server section
  ```
- If Playwright errors on missing deps, rebuild once Desktop is healthy.

---

## Local Dev (without Docker)

```bash
python -m venv .venv
# Windows PowerShell:   .\.venv\Scripts\Activate.ps1
# macOS/Linux:          source .venv/bin/activate

pip install --upgrade pip
pip install -r req.txt

# First-time Playwright browser install:
python -m playwright install chromium

# Run the service
python -m uvicorn app:app --host 0.0.0.0 --port 8080 --lifespan on
```

---

## Configuration

Environment variables (optional):

| Var              | Default                                       | Description                                   |
|------------------|-----------------------------------------------|-----------------------------------------------|
| `MODEL_ID`       | `cardiffnlp/twitter-roberta-base-sentiment-latest` | HF model repo for pipeline                    |
| `BATCH_SIZE`     | `64`                                          | Inference batch size                          |
| `MAX_LEN`        | `256`                                         | Max tokens per input                          |
| `MAX_HTML_ITEMS` | `300`                                         | Max paragraphs per page                       |
| `REDDIT_LIMIT`   | `200`                                         | Max posts/comments pulled via Reddit adapter  |

---

## Endpoints

### `GET /`
Serves `index.html` if present. Simple UI for upload/URL/crawl.

### `GET /healthz`
```json
{"status": "ok", "model": "<MODEL_ID>"}
```

### `POST /predict/file`
- **Body:** `multipart/form-data` with `file` (JSON/CSV/HTML).
- **Response:**  
  - JSON/CSV → same type with `scores` and `top` label per row/item  
  - HTML → annotated HTML download

### `POST /predict/url`
- **Body (JSON):**
```json
{
  "url": "https://example.com/article.html",
  "render": false,
  "cookies": "opt=1; session=abc...", 
  "crawl": false,
  "max_pages": 50,
  "max_depth": 2,
  "same_host_only": true,
  "delay_ms": 300
}
```
- **Behavior:**
  - If `crawl=false`: processes a single URL
    - JSON/CSV → same type out
    - HTML → annotated HTML
    - If the page is JS-gated, service retries with Playwright (or set `render=true`)
  - If `crawl=true`: BFS crawl within limits → aggregated **site report** (single annotated HTML)
- **Response:** downloadable file (content-type set; `Content-Disposition` filename provided)

---

## Front-End (index.html)

- **File upload:** hidden `<input type="file">` + button → `/predict/file`
- **URL form:** URL + optional **Enable JavaScript** (Playwright), optional **Cookies**
- **Crawl controls:** checkbox + `max_pages`, `max_depth`, `same_host_only`, `delay_ms`
- Auto-downloads results and shows status text.

> **Cookies:** Paste a raw `Cookie` header string when a site requires login (e.g., X/Twitter). Cookies are **not stored** and are only used for that request context.

---

## Adapters (Pluggable Registry)

`adapters.py` defines a simple registry:

- **GenericAdapter**: default handler for JSON/CSV/HTML (+ optional crawl)
- **RedditAdapter**: uses Reddit’s `.json` listing/comments to avoid infinite scroll

Add a new adapter by subclassing and decorating:

```python
@register(domains=("x.com","twitter.com"))
class TwitterAdapter:
    async def process(self, url, *, app, render, cookies, crawl, max_pages, max_depth, same_host_only, delay_ms):
        # likely need cookies + JS render
        browser = await ensure_browser(app)
        raw, kind = await fetch_url_bytes_rendered(url, browser, cookies_header=cookies)
        # parse → paragraphs, classify → HTML
        paras = html_to_paragraphs(raw)
        preds = classify_texts([preprocess_text(p) for p in paras])
        html = render_annotated_html("Twitter", url, paras, preds).encode("utf-8")
        return AdapterOutput(html, "text/html; charset=utf-8", "twitter_scored.html")
```

No changes to `app.py` are required — the registry picks it up automatically.

---

## Crawler

Implemented in `fetch.py:crawl_site`:

- **BFS** with `max_pages`, `max_depth`, `same_host_only`, `delay_ms`
- Skips `mailto:` / `javascript:` / non-HTTP(S)
- Best-effort **robots.txt** (will skip disallowed paths when robots is reachable)
- Per-page **try/except** (errors are skipped, crawl continues)
- HTML pages are parsed → paragraphs → scores; JSON/CSV are summarized concisely in site reports

> Tip: Use moderate `max_pages` and a small `delay_ms` to be polite. Some sites will throttle/deny bots.

---

## Input & Output Formats

### JSON (input)
- **List of strings**:
  ```json
  ["I love this", "This is bad"]
  ```
- **List of objects** (one of the keys must exist): `text` | `tweet` | `content`
  ```json
  [{"text":"Great work."},{"text":"Could be better."}]
  ```

### JSON (output)
- For list-of-strings input:
  ```json
  [
    {"text":"I love this","scores":{"negative":0.02,"neutral":0.10,"positive":0.88},"top":{"label":"positive","score":0.88}},
    ...
  ]
  ```
- For list-of-objects:
  original fields preserved + added `scores` and `top`.

### CSV
- Auto-detects a text column (prefers `text`/`tweet`/`content`, otherwise first object-dtype column).
- Adds columns: `score_negative`, `score_neutral`, `score_positive`, `top_label`, `top_score`.

### Annotated HTML
- Dark neutral cards, **colored left border** and **pill badge** indicating the top label & score.
- One block per paragraph (HTML) or per item (crawl aggregation).

---

## Model Notes

- Default pipeline: `text-classification` (CardiffNLP Twitter RoBERTa).
- Preprocessing: tweets → replace `@user` and URLs with placeholders.
- Batch inference (`BATCH_SIZE`) and truncation (`MAX_LEN`) are configurable.

To swap models:
```bash
export MODEL_ID=distilbert-base-uncased-finetuned-sst-2-english
```

---

## Troubleshooting

- **Docker can’t connect / EOF:** Ensure Docker Desktop is running; `docker version` must show a **Server**. Switch context if needed:
  ```powershell
  docker context use default
  wsl --shutdown   # then reopen Docker Desktop, on Windows
  ```
- **Playwright errors:** Rebuild with network stable. The Dockerfile runs:
  ```
  python -m playwright install --with-deps chromium
  ```
  Locally:
  ```
  python -m playwright install chromium
  ```
- **Twitter/X shows “enable JavaScript / login” page:** That’s an interstitial. Provide a valid `Cookie` header from a logged-in browser, or implement an OAuth adapter. The service supports passing a raw cookie string in `/predict/url`.
- **Reddit returns few items:** Use the Reddit adapter by calling `/predict/url` on a `reddit.com` URL (we hit the `.json` API and page through).
- **Package versions:** `req.txt` pins broad minimums; if you change Python versions, re-resolve if pip reports incompatibilities.

---

## Security

- Cookies passed via `/predict/url` are **not persisted** by default; they’re only injected into the Playwright context for that request.
- If you deploy this, consider:
  - rate limiting,
  - domain allowlists,
  - size/time limits (already included),
  - TLS termination and secrets management.

---

## License

This repo uses open-source model weights and libraries with their own licenses (Hugging Face Transformers, Playwright, etc.). Review and comply with those licenses in your deployment context.

---
