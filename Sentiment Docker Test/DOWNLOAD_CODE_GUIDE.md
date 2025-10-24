# JSON Download Code - Complete Flow

This document shows exactly where and how the scored JSON file gets downloaded.

## 📍 The Complete Download Flow

```
Backend (Python)                          Frontend (JavaScript)
─────────────────────────────────────────────────────────────────

1. Create JSON payload
   app.py:222-223
   ┌──────────────────────────────┐
   │ payload = _as_json_bytes({   │
   │   "preset": preset,          │
   │   "results": result          │
   │ })                           │
   └──────────────────────────────┘
              │
              ▼
2. Prepare download response
   app.py:84-87
   ┌──────────────────────────────┐
   │ def _make_download(...)      │
   │   resp = StreamingResponse() │
   │   resp.headers[              │
   │     "Content-Disposition"    │
   │   ] = 'attachment;           │
   │        filename="file.json"' │
   └──────────────────────────────┘
              │
              ▼
3. Return response
   app.py:223
   ┌──────────────────────────────┐
   │ return _make_download(       │
   │   "predictions.json",        │
   │   payload,                   │
   │   "application/json"         │
   │ )                            │
   └──────────────────────────────┘
              │
              │ HTTP Response
              │ ───────────────────────────────►
              │                                 │
              │                                 ▼
              │                    4. Receive response
              │                       index.html:192
              │                    ┌─────────────────────┐
              │                    │ const blob =        │
              │                    │   await res.blob()  │
              │                    └─────────────────────┘
              │                                 │
              │                                 ▼
              │                    5. Extract filename
              │                       index.html:193-194
              │                    ┌─────────────────────────┐
              │                    │ const disp =            │
              │                    │   res.headers.get(      │
              │                    │     'Content-           │
              │                    │      Disposition'       │
              │                    │   );                    │
              │                    │ const name =            │
              │                    │   filenameFrom          │
              │                    │   Disposition(disp)     │
              │                    └─────────────────────────┘
              │                                 │
              │                                 ▼
              │                    6. Trigger download!
              │                       index.html:195
              │                    ┌─────────────────────────┐
              │                    │ downloadBlob(blob,name) │
              │                    └─────────────────────────┘
              │                                 │
              │                                 ▼
              │                    7. Create download link
              │                       index.html:149-153
              │                    ┌─────────────────────────┐
              │                    │ function downloadBlob() │
              │                    │   url =                 │
              │                    │     URL.createObject    │
              │                    │     URL(blob)           │
              │                    │   a = createElement('a')│
              │                    │   a.href = url          │
              │                    │   a.download = name     │
              │                    │   a.click()  ← DOWNLOAD!│
              │                    └─────────────────────────┘
```

---

## 🔍 Code Breakdown

### 1️⃣ **Backend: Create JSON Payload**

**File:** `app.py`
**Lines:** 48-49, 220-223

```python
# Convert Python dict to JSON bytes
def _as_json_bytes(obj: T.Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")

# In predict_file() function:
result = run_task(texts, preset=preset, labels=lbls)  # NLP processing

payload = _as_json_bytes({"preset": preset, "results": result})
```

**What it does:**
- Takes the sentiment analysis results (Python dict)
- Converts to JSON string with pretty formatting (`indent=2`)
- Encodes to bytes (UTF-8)

---

### 2️⃣ **Backend: Prepare Download Response**

**File:** `app.py`
**Lines:** 84-87

```python
def _make_download(name: str, payload: bytes, mime: str = "application/json") -> StreamingResponse:
    resp = StreamingResponse(io.BytesIO(payload), media_type=mime)
    resp.headers["Content-Disposition"] = f'attachment; filename="{name}"'
    return resp
```

**What it does:**
- Creates a `StreamingResponse` (FastAPI response type)
- Sets `Content-Type: application/json`
- Sets `Content-Disposition: attachment; filename="predictions.json"`
  - This header tells the browser to **download** instead of display
  - The `filename=` part sets the default download filename

---

### 3️⃣ **Backend: Return Response**

**File:** `app.py`
**Lines:** 223 (for file), 290-292 (for URL)

```python
# For file upload endpoint:
return _make_download("predictions.json", payload, "application/json")

# For URL endpoint:
return _make_download("url-output.json", payload, "application/json")
```

**What it does:**
- Calls `_make_download()` with:
  - Filename: `"predictions.json"` or `"url-output.json"`
  - Payload: The JSON bytes
  - MIME type: `"application/json"`
- FastAPI sends this as HTTP response to browser

---

### 4️⃣ **Frontend: Receive Response**

**File:** `index.html`
**Lines:** 186-192

```javascript
try {
    // Make HTTP request to backend
    const res = await fetch('/predict/file?'+qs.toString(), {
        method:'POST',
        body: fd
    });

    // Check if successful
    if(!res.ok){
        const err = await res.text().catch(()=>res.statusText);
        throw new Error(err);
    }

    // Convert response to blob (binary data)
    const blob = await res.blob();
```

**What it does:**
- `fetch()` makes HTTP POST request to `/predict/file`
- Waits for response from Python backend
- `res.blob()` reads the response body as binary data (Blob object)

---

### 5️⃣ **Frontend: Extract Filename**

**File:** `index.html`
**Lines:** 144-148, 193-194

```javascript
// Helper function to parse Content-Disposition header
function filenameFromDisposition(h){
    if(!h) return null;
    const m = /filename\*=UTF-8''([^;]+)|filename="?([^"]+)"?/i.exec(h);
    return decodeURIComponent(m?.[1] || m?.[2] || '');
}

// Extract filename from response headers
const disp = res.headers.get('Content-Disposition');
const name = filenameFromDisposition(disp) || ('output-'+Date.now());
```

**What it does:**
- Gets `Content-Disposition` header from response
  - Example: `attachment; filename="predictions.json"`
- Uses regex to extract the filename (`predictions.json`)
- Falls back to `output-<timestamp>` if no filename found

---

### 6️⃣ **Frontend: Trigger Download**

**File:** `index.html`
**Line:** 195

```javascript
downloadBlob(blob, name);
```

**What it does:**
- Calls the `downloadBlob()` function with:
  - `blob`: The binary JSON data
  - `name`: The filename to use (`predictions.json`)

---

### 7️⃣ **Frontend: Create Download Link (THE ACTUAL DOWNLOAD!)**

**File:** `index.html`
**Lines:** 149-153

```javascript
function downloadBlob(blob, name){
    const url = URL.createObjectURL(blob);           // Create temporary URL
    const a = document.createElement('a');           // Create <a> element
    a.href = url;                                    // Set href to blob URL
    a.download = name||'output';                     // Set download filename
    document.body.appendChild(a);                    // Add to page
    a.click();                                       // ← TRIGGER DOWNLOAD!
    a.remove();                                      // Remove element
    URL.revokeObjectURL(url);                        // Clean up URL
}
```

**What it does:**
1. **Create blob URL:** `blob:http://localhost:8080/abc-123-def`
2. **Create invisible link:** `<a href="blob:..." download="predictions.json">`
3. **Add to page:** Temporarily insert the `<a>` element
4. **Click it programmatically:** `a.click()` - **THIS TRIGGERS THE DOWNLOAD!**
5. **Clean up:** Remove the element and revoke the blob URL

**This is the magic moment** - when `a.click()` is called, the browser's download manager kicks in and saves the file!

---

## 🎯 Key Files & Line Numbers

| What | File | Lines | Code |
|------|------|-------|------|
| **Create JSON** | `app.py` | 48-49 | `_as_json_bytes()` |
| **Set download headers** | `app.py` | 84-87 | `_make_download()` |
| **Return response** | `app.py` | 223 | `return _make_download(...)` |
| **Receive blob** | `index.html` | 192 | `const blob = await res.blob()` |
| **Extract filename** | `index.html` | 193-194 | `filenameFromDisposition()` |
| **Download blob** | `index.html` | 195 | `downloadBlob(blob, name)` |
| **🎯 ACTUAL DOWNLOAD** | `index.html` | 151 | `a.click()` ← **HERE!** |

---

## 🧪 Test the Download Code

### Test from Browser Console

Open your browser console (F12) on `http://localhost:8080` and run:

```javascript
// Create a test JSON blob
const data = {test: "Hello, world!", scores: [0.9, 0.05, 0.05]};
const json = JSON.stringify(data, null, 2);
const blob = new Blob([json], {type: 'application/json'});

// Trigger download using the same function as the UI
const url = URL.createObjectURL(blob);
const a = document.createElement('a');
a.href = url;
a.download = 'test-download.json';
document.body.appendChild(a);
a.click();  // ← Should download test-download.json!
a.remove();
URL.revokeObjectURL(url);
```

### Test the Full Flow

```javascript
// Simulate the full file upload flow
async function testDownload() {
    // Create test file
    const testData = ["I love this!", "This is terrible."];
    const jsonStr = JSON.stringify(testData);
    const file = new Blob([jsonStr], {type: 'application/json'});

    // Upload to backend
    const fd = new FormData();
    fd.append('file', file, 'test.json');

    const res = await fetch('/predict/file?preset=sentiment-twitter', {
        method: 'POST',
        body: fd
    });

    // Download response
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'test-results.json';
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);

    console.log('Download triggered!');
}

testDownload();
```

---

## 🔧 How to Customize

### Change the Downloaded Filename

**Backend (Python):**
```python
# In app.py, change the filename parameter:
return _make_download("my-custom-name.json", payload, "application/json")
```

**Frontend (JavaScript):**
```javascript
// In index.html, change the fallback name:
const name = filenameFromDisposition(disp) || 'my-custom-output.json';
```

### Download as Different Format

**CSV Example:**

**Backend:**
```python
import pandas as pd

# Convert results to DataFrame
df = pd.DataFrame(results)
csv_bytes = df.to_csv(index=False).encode('utf-8')

# Return as CSV download
return _make_download("results.csv", csv_bytes, "text/csv")
```

**Frontend stays the same** - it will download whatever the backend sends!

### Add Download Progress

```javascript
// Replace res.blob() with manual reading for progress
const reader = res.body.getReader();
const contentLength = +res.headers.get('Content-Length');
let receivedLength = 0;
const chunks = [];

while(true) {
    const {done, value} = await reader.read();
    if (done) break;

    chunks.push(value);
    receivedLength += value.length;

    const progress = (receivedLength / contentLength) * 100;
    console.log(`Download progress: ${progress.toFixed(0)}%`);
}

const blob = new Blob(chunks);
downloadBlob(blob, 'download.json');
```

---

## 🐛 Common Issues

### Issue: File downloads as `.txt` instead of `.json`

**Cause:** MIME type not set correctly
**Fix:** Ensure `app.py:84-87` sets `media_type="application/json"`

### Issue: Browser displays JSON instead of downloading

**Cause:** Missing `Content-Disposition: attachment` header
**Fix:** Check `app.py:86` sets the header correctly

### Issue: Downloaded filename is wrong

**Cause:** Regex in `filenameFromDisposition()` not matching
**Fix:** Check the `Content-Disposition` header format in browser DevTools (Network tab)

### Issue: Download doesn't start

**Cause:** `a.click()` might be blocked
**Fix:** Ensure `a` is attached to DOM before clicking:
```javascript
document.body.appendChild(a);  // Must be before click()
a.click();
```

---

## 💡 Summary

**The download happens in one line:**
```javascript
a.click();  // index.html line 151 ← THIS LINE DOWNLOADS THE FILE!
```

**Full path:**
1. Python creates JSON → `_as_json_bytes()`
2. Python sets headers → `_make_download()`
3. Python returns response → `return _make_download(...)`
4. JavaScript receives blob → `await res.blob()`
5. JavaScript gets filename → `filenameFromDisposition()`
6. JavaScript creates link → `createElement('a')`
7. **JavaScript clicks link** → `a.click()` ← **DOWNLOAD!**

The browser's download manager takes over and saves the file to the user's Downloads folder.
