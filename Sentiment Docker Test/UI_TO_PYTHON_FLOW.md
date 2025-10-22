# Web UI → Python Backend Flow

This document shows exactly how the "Run" buttons in the web UI call Python functions.

## 📍 Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     index.html (Frontend)                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ User clicks "Run"
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              JavaScript Event Handler (Lines 169 or 204)    │
│  document.getElementById('run-file').onclick = async ()=>{}  │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ Makes HTTP POST request
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   fetch('/predict/file')                    │
│                   fetch('/predict/url')                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ HTTP Request over network
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   app.py (Python Backend)                   │
│               FastAPI receives the request                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ Routes to endpoint
                              ▼
┌─────────────────────────────────────────────────────────────┐
│          @app.post("/predict/file") decorator               │
│          async def predict_file(...)                        │
│                                                             │
│                        OR                                   │
│                                                             │
│          @app.post("/predict/url") decorator                │
│          async def predict_url(...)                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ Calls NLP functions
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              nlp.py → run_task(texts, ...)                  │
│        transformers pipeline for sentiment analysis         │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ Returns results
                              ▼
┌─────────────────────────────────────────────────────────────┐
│           Response sent back to browser as JSON             │
│            Browser downloads the results file               │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔍 Detailed Code Trace

### 1. **HTML Button** (index.html)

#### File Upload Button
```html
<!-- Line 69 in index.html -->
<button class="btn primary" id="run-file">Run</button>
```

#### URL Processing Button
```html
<!-- Line 116 in index.html -->
<button class="btn primary" id="run-url">Run</button>
```

---

### 2. **JavaScript Event Handler** (index.html)

#### File Upload Handler
```javascript
// Lines 169-200 in index.html
document.getElementById('run-file').onclick = async ()=>{
    const f = fileIn.files?.[0];
    if(!f){ setStatus(statusFile, 'Choose a file first.', 'warn'); return; }

    // Build FormData
    const fd = new FormData();
    fd.append('file', f, f.name);

    // Make HTTP POST request to Python backend
    const res = await fetch('/predict/file?'+qs.toString(), {
        method:'POST',
        body: fd
    });

    // Handle response (download file)
    const blob = await res.blob();
    downloadBlob(blob, name);
};
```

#### URL Processing Handler
```javascript
// Lines 204-254 in index.html
document.getElementById('run-url').onclick = async ()=>{
    const url = document.getElementById('url').value.trim();

    const body = {
        url, render, renderer,
        preset, labels,
        // ... other parameters
    };

    // Make HTTP POST request to Python backend
    const res = await fetch('/predict/url', {
        method:'POST',
        headers:{'content-type':'application/json'},
        body: JSON.stringify(body)
    });

    // Handle response (download file)
    const blob = await res.blob();
    downloadBlob(blob, name);
};
```

---

### 3. **Python Endpoint Decorator** (app.py)

#### File Upload Endpoint
```python
# Lines 187-226 in app.py
@app.post("/predict/file")
async def predict_file(
    file: UploadFile = File(...),
    preset: str | None = Query(None),
    labels: str | None = Query(None),
    include_stopwords: bool | None = Query(False),
):
    # This is the Python function that gets called!
    # Read the uploaded file
    b = await file.read()

    # Extract texts from file
    texts = _texts_from_json_bytes(b)  # or _texts_from_csv_bytes

    # Preprocess
    texts = [preprocess_for_task(t, task or "") for t in texts]

    # Run the NLP task (THIS IS THE MAIN PROCESSING!)
    result = run_task(texts, preset=preset, labels=lbls)

    # Return results as downloadable file
    payload = _as_json_bytes({"preset": preset, "results": result})
    return _make_download("predictions.json", payload, "application/json")
```

#### URL Processing Endpoint
```python
# Lines 242-292 in app.py
@app.post("/predict/url")
async def predict_url(body: UrlBody = Body(...)):
    # This is the Python function that gets called!
    url = body.get("url")

    # Fetch the URL content
    html = _fetch_simple(url, cookies_header, extra_headers)

    # Extract text from HTML
    text = _extract_text_from_html(html, base_url=url)

    # Preprocess
    texts = [preprocess_for_task(text, task or "")]

    # Run the NLP task (THIS IS THE MAIN PROCESSING!)
    result = run_task(texts, preset=preset, labels=lbls)

    # Return results
    payload = _as_json_bytes({
        "preset": preset,
        "url": url,
        "results": result,
    })
    return _make_download("url-output.json", payload, "application/json")
```

---

### 4. **Core NLP Processing** (nlp.py)

```python
# Lines 275-365 in nlp.py
def run_task(
    texts: List[str],
    *,
    task: Optional[str] = None,
    preset: Optional[str] = None,
    labels: Optional[List[str]] = None,
) -> List[Any]:
    """
    This is the main NLP processing function!
    Called by the endpoints above.
    """

    # Determine task type (sentiment, zero-shot, NER, etc.)
    if task in {"text-classification", "sentiment-analysis"}:
        return _hf_text_classification(texts, model_id=model_id)

    if task == "zero-shot-classification":
        return _hf_zero_shot(texts, model_id=model_id, candidate_labels=labels)

    # Uses Hugging Face transformers pipeline
    # Loads models like cardiffnlp/twitter-roberta-base-sentiment-latest
```

---

## 🎯 Key Connection Points

### Request Flow

| Location | Code | What It Does |
|----------|------|--------------|
| **index.html:69** | `<button id="run-file">` | HTML button user clicks |
| **index.html:169** | `document.getElementById('run-file').onclick` | JavaScript event handler |
| **index.html:187** | `fetch('/predict/file', {...})` | HTTP POST request |
| **app.py:187** | `@app.post("/predict/file")` | FastAPI route decorator |
| **app.py:187** | `async def predict_file(...)` | **Python function that runs!** |
| **app.py:220** | `result = run_task(...)` | Calls NLP processing |
| **nlp.py:275** | `def run_task(...)` | Main NLP function |
| **nlp.py:311** | `_hf_text_classification(...)` | Transformer model inference |

---

## 🔧 How to Customize

### Want to add a new button?

#### Step 1: Add HTML button (index.html)
```html
<button class="btn primary" id="run-custom">My Custom Action</button>
```

#### Step 2: Add JavaScript handler (index.html)
```javascript
document.getElementById('run-custom').onclick = async () => {
    const res = await fetch('/my-custom-endpoint', {
        method: 'POST',
        headers: {'content-type': 'application/json'},
        body: JSON.stringify({ param1: 'value1' })
    });

    const data = await res.json();
    console.log(data);
};
```

#### Step 3: Add Python endpoint (app.py)
```python
@app.post("/my-custom-endpoint")
async def my_custom_function(body: dict = Body(...)):
    param1 = body.get("param1")

    # Your custom logic here
    result = do_something(param1)

    return {"result": result}
```

---

## 📡 Communication Protocol

The connection uses **HTTP POST requests**:

```
Browser (JavaScript)           →  [HTTP POST]  →  FastAPI (Python)
                                  /predict/file
                                  /predict/url
                                  /graph/degrees
                                  etc.

Browser (JavaScript)           ←  [HTTP Response]  ←  FastAPI (Python)
receives JSON or file download                        returns data
```

### Request Format

**File Upload:**
```http
POST /predict/file?preset=sentiment-twitter HTTP/1.1
Content-Type: multipart/form-data

[file contents]
```

**URL Processing:**
```http
POST /predict/url HTTP/1.1
Content-Type: application/json

{
  "url": "https://example.com",
  "preset": "sentiment-twitter",
  "render": false
}
```

### Response Format

Python sends back a **file download** with these headers:
```http
HTTP/1.1 200 OK
Content-Type: application/json
Content-Disposition: attachment; filename="predictions.json"

[JSON data]
```

---

## 🧪 Testing the Connection

You can test the Python endpoints directly without the UI:

### Test File Upload Endpoint
```bash
curl -X POST "http://localhost:8080/predict/file?preset=sentiment-twitter" \
  -F "file=@test.csv"
```

### Test URL Endpoint
```bash
curl -X POST "http://localhost:8080/predict/url" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com", "preset": "sentiment-twitter"}'
```

### Test from Browser Console
```javascript
// Open browser console (F12) on http://localhost:8080
fetch('/predict/url', {
    method: 'POST',
    headers: {'content-type': 'application/json'},
    body: JSON.stringify({
        url: 'https://news.ycombinator.com',
        preset: 'sentiment-twitter'
    })
}).then(r => r.json()).then(console.log);
```

---

## 🎨 Visual Summary

```
┌──────────────┐
│  User clicks │  index.html line 69 or 116
│  "Run" button│
└──────┬───────┘
       │
       │ JavaScript onclick handler
       │ (index.html lines 169 or 204)
       │
       ▼
┌──────────────┐
│    fetch()   │  Makes HTTP request
│   /predict/* │  Lines 187 or 237
└──────┬───────┘
       │
       │ Network request
       │
       ▼
┌──────────────┐
│ @app.post()  │  FastAPI decorator in app.py
│   decorator  │  Lines 187 or 242
└──────┬───────┘
       │
       │ Routes request to function
       │
       ▼
┌──────────────┐
│ predict_file │  Python function executes!
│      or      │  Lines 188-226 or 243-292
│ predict_url  │
└──────┬───────┘
       │
       │ Calls NLP processing
       │
       ▼
┌──────────────┐
│  run_task()  │  Main NLP logic in nlp.py
│              │  Line 275
└──────┬───────┘
       │
       │ Returns results
       │
       ▼
┌──────────────┐
│   Browser    │  Downloads JSON file
│   receives   │
│   response   │
└──────────────┘
```

---

## 🔗 File References

| File | Lines | Purpose |
|------|-------|---------|
| `index.html` | 69, 116 | Run buttons |
| `index.html` | 169-200, 204-254 | JavaScript click handlers |
| `index.html` | 187, 237 | `fetch()` API calls |
| `app.py` | 187-226 | `/predict/file` endpoint |
| `app.py` | 242-292 | `/predict/url` endpoint |
| `app.py` | 220, 281 | Calls `run_task()` |
| `nlp.py` | 275-365 | Main NLP processing |
| `nlp.py` | 127-165 | Text classification logic |

---

## 💡 Quick Reference

**Want to find where a button calls Python?**

1. Find the button ID in HTML (e.g., `id="run-file"`)
2. Search for `getElementById('run-file')` in JavaScript
3. Look for `fetch('/some-endpoint')` in that handler
4. Search app.py for `@app.post("/some-endpoint")`
5. That's the Python function that runs!

**Example:**
```
Button → id="run-file"
     → document.getElementById('run-file').onclick
     → fetch('/predict/file')
     → @app.post("/predict/file")
     → async def predict_file() ← **This Python function runs!**
```
