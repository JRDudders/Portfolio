# fetch.py
from __future__ import annotations
import asyncio, os, time, json, ssl
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse, urljoin, urlunparse, urlencode
from collections import deque

# Disable SSL certificate verification globally (for corporate firewalls like Zscaler)
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

import requests
import random
# Disable SSL warnings and patch requests to skip verification
requests.packages.urllib3.disable_warnings()
_original_session_request = requests.Session.request
def _patched_session_request(self, *args, **kwargs):
    kwargs['verify'] = False
    return _original_session_request(self, *args, **kwargs)
requests.Session.request = _patched_session_request

# Also patch module-level request functions
_original_get = requests.get
_original_post = requests.post
def _patched_get(*args, **kwargs):
    kwargs['verify'] = False
    return _original_get(*args, **kwargs)
def _patched_post(*args, **kwargs):
    kwargs['verify'] = False
    return _original_post(*args, **kwargs)
requests.get = _patched_get
requests.post = _patched_post

from bs4 import BeautifulSoup

SEL_ENGINE = os.getenv("SELENIUM_ENGINE", "chrome")

HTTP_TIMEOUT = (3600, 3600)  # (connect_timeout, read_timeout) in seconds - 1 hour for large model downloads
MAX_DOWNLOAD_BYTES = 100 * 1024 * 1024
REDDIT_LIMIT = int(os.getenv("REDDIT_LIMIT", "200"))

# Pool of realistic, recent user agents (updated Dec 2024)
USER_AGENT_POOL = [
    # Chrome on Windows (most common)
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
    # Chrome on Mac
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    # Firefox on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) Gecko/20100101 Firefox/133.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:132.0) Gecko/20100101 Firefox/132.0",
    # Edge on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0",
]

def get_random_user_agent() -> str:
    """Return a random user agent from the pool"""
    return random.choice(USER_AGENT_POOL)

def get_headers_for_user_agent(ua: str) -> dict:
    """Generate matching headers for a given user agent"""
    headers = {
        "User-Agent": ua,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Upgrade-Insecure-Requests": "1",
    }

    # Add sec-ch-ua headers for Chrome/Edge browsers (important for anti-bot detection)
    if "Chrome/" in ua:
        # Extract Chrome version
        import re
        chrome_match = re.search(r'Chrome/(\d+)', ua)
        if chrome_match:
            version = chrome_match.group(1)
            if "Edg/" in ua:
                headers["sec-ch-ua"] = f'"Microsoft Edge";v="{version}", "Chromium";v="{version}", "Not?A_Brand";v="99"'
            else:
                headers["sec-ch-ua"] = f'"Google Chrome";v="{version}", "Chromium";v="{version}", "Not?A_Brand";v="99"'
            headers["sec-ch-ua-mobile"] = "?0"
            headers["sec-ch-ua-platform"] = '"Windows"' if "Windows" in ua else '"macOS"'
            headers["sec-fetch-dest"] = "document"
            headers["sec-fetch-mode"] = "navigate"
            headers["sec-fetch-site"] = "none"
            headers["sec-fetch-user"] = "?1"

    return headers

# Default headers (will be overridden per-request with random UA)
DEFAULT_FETCH_HEADERS = get_headers_for_user_agent(USER_AGENT_POOL[0])

def _infer_kind(url: str, headers: Dict[str, str]) -> Optional[str]:
    ct = (headers.get("Content-Type") or "").split(";")[0].strip().lower()
    if ct in {"application/json", "text/json"}: return "json"
    if ct in {"text/csv", "application/csv"}:   return "csv"
    if ct in {"text/html"}:                    return "html"
    low = url.lower()
    if low.endswith(".json"): return "json"
    if low.endswith(".csv"):  return "csv"
    if low.endswith((".htm", ".html")): return "html"
    return None

def fetch_url_bytes_sync(url: str, extra_headers: Optional[Dict[str,str]] = None) -> Tuple[bytes, str]:
    headers = dict(DEFAULT_FETCH_HEADERS)
    if extra_headers:
        headers.update({str(k): str(v) for k, v in extra_headers.items()})
    with requests.get(url, stream=True, timeout=HTTP_TIMEOUT, headers=headers) as r:
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

async def fetch_url_bytes(url: str, extra_headers: Optional[Dict[str,str]] = None) -> Tuple[bytes, str]:
    return await asyncio.to_thread(fetch_url_bytes_sync, url, extra_headers)

# -------- Playwright (async, with stealth, wait_selector, auto-scroll, cookie banners) --------
from playwright.async_api import async_playwright, Browser
try:
    from playwright_stealth import stealth_async
except Exception:
    stealth_async = None

async def ensure_browser(app) -> Browser:
    if not hasattr(app.state, "playwright") or app.state.playwright is None:
        app.state.playwright = await async_playwright().start()
    if not hasattr(app.state, "browser") or app.state.browser is None:
        # Use system Chromium if available, otherwise let Playwright find its own
        chromium_path = os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH") or os.environ.get("CHROME_BIN")
        # Comprehensive anti-detection browser arguments
        launch_args = {
            "headless": True,
            "args": [
                "--no-sandbox",
                "--disable-blink-features=AutomationControlled",
                "--disable-infobars",
                "--disable-dev-shm-usage",
                "--disable-browser-side-navigation",
                "--disable-gpu",
                "--disable-features=VizDisplayCompositor",
                "--disable-extensions",
                # Disable automation flags
                "--disable-automation",
                "--disable-blink-features=AutomationControlled",
                # Make window size realistic
                "--window-size=1920,1080",
                "--start-maximized",
                # Disable various detection vectors
                "--disable-component-extensions-with-background-pages",
                "--disable-default-apps",
                "--disable-backgrounding-occluded-windows",
                "--disable-renderer-backgrounding",
                "--disable-background-timer-throttling",
                # WebRTC leak prevention
                "--disable-webrtc-hw-encoding",
                "--disable-webrtc-hw-decoding",
                # Additional stealth
                "--no-first-run",
                "--no-service-autorun",
                "--password-store=basic",
                "--use-mock-keychain",
            ],
        }
        if chromium_path and os.path.exists(chromium_path):
            launch_args["executable_path"] = chromium_path
        app.state.browser = await app.state.playwright.chromium.launch(**launch_args)
    return app.state.browser

async def shutdown_playwright(app):
    try:
        if getattr(app.state, "browser", None):
            await app.state.browser.close()
    finally:
        if getattr(app.state, "playwright", None):
            await app.state.playwright.stop()

async def _dismiss_cookie_banners(page):
    selectors = [
        "button#onetrust-accept-btn-handler",
        "button[aria-label*='Accept']",
        "button:has-text('Accept')",
        "button:has-text('I Agree')",
        "[data-testid='accept-all']",
        "button:has-text('Got it')",
    ]
    for sel in selectors:
        try:
            el = await page.query_selector(sel)
            if el: await el.click(timeout=500)
        except Exception:
            pass

async def _auto_scroll(page, passes: int = 6):
    """Scroll page with human-like behavior"""
    for i in range(max(0, passes)):
        # Randomize scroll amount and timing to appear more human
        scroll_percent = random.uniform(0.7, 0.95)
        await page.evaluate(f"window.scrollBy(0, document.body.scrollHeight * {scroll_percent});")
        # Variable wait time between scrolls
        await page.wait_for_timeout(random.randint(300, 800))

async def _human_like_mouse_move(page):
    """Simulate human-like mouse movement"""
    try:
        # Move mouse to random positions to simulate human behavior
        for _ in range(random.randint(2, 4)):
            x = random.randint(100, 1200)
            y = random.randint(100, 700)
            await page.mouse.move(x, y)
            await page.wait_for_timeout(random.randint(50, 150))
    except Exception:
        pass

# Common screen resolutions to randomize (appear more natural)
SCREEN_RESOLUTIONS = [
    {"width": 1920, "height": 1080},
    {"width": 1366, "height": 768},
    {"width": 1536, "height": 864},
    {"width": 1440, "height": 900},
    {"width": 1280, "height": 720},
]

async def fetch_url_bytes_rendered(
    url: str,
    browser: Browser,
    timeout_ms: int = 3600000,  # 1 hour
    cookies_header: Optional[str] = None,
    wait_selector: Optional[str] = None,
    scroll_passes: int = 6,
    extra_headers: Optional[Dict[str,str]] = None,
) -> Tuple[bytes, str]:
    host = (urlparse(url).hostname or "").lstrip(".")

    # Randomize user agent and get matching headers
    user_agent = get_random_user_agent()
    ua_headers = get_headers_for_user_agent(user_agent)

    # Randomize viewport to avoid fingerprinting
    viewport = random.choice(SCREEN_RESOLUTIONS)

    # Determine platform from user agent for consistent fingerprint
    is_mac = "Macintosh" in user_agent
    is_windows = "Windows" in user_agent

    ctx = await browser.new_context(
        user_agent=user_agent,
        locale="en-US",
        viewport=viewport,
        screen=viewport,  # Match screen to viewport
        ignore_https_errors=True,
        # Additional anti-detection context options
        color_scheme="light",
        device_scale_factor=random.choice([1, 1.25, 1.5, 2]),  # Common DPI scales
        has_touch=False,
        is_mobile=False,
        java_script_enabled=True,
        timezone_id="America/New_York",  # Common timezone
        geolocation=None,
        permissions=[],
        extra_http_headers=ua_headers,
    )

    # Set headers (excluding user-agent which is set in context)
    headers = {k: v for k, v in ua_headers.items() if k.lower() != "user-agent"}
    if extra_headers:
        headers.update({str(k): str(v) for k, v in extra_headers.items()})
    await ctx.set_extra_http_headers(headers)

    # Optional cookies
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
        if jar:
            await ctx.add_cookies(jar)

    page = await ctx.new_page()

    # Apply stealth plugin if available
    if stealth_async:
        try:
            await stealth_async(page)
        except Exception:
            pass

    # Additional JavaScript stealth overrides (run before page loads)
    await page.add_init_script("""
        // Override navigator.webdriver
        Object.defineProperty(navigator, 'webdriver', {
            get: () => undefined,
        });

        // Override navigator.plugins to look real
        Object.defineProperty(navigator, 'plugins', {
            get: () => [
                { name: 'Chrome PDF Plugin', filename: 'internal-pdf-viewer' },
                { name: 'Chrome PDF Viewer', filename: 'mhjfbmdgcfjbbpaeojofohoefgiehjai' },
                { name: 'Native Client', filename: 'internal-nacl-plugin' },
            ],
        });

        // Override navigator.languages
        Object.defineProperty(navigator, 'languages', {
            get: () => ['en-US', 'en'],
        });

        // Override permissions API
        const originalQuery = window.navigator.permissions.query;
        window.navigator.permissions.query = (parameters) => (
            parameters.name === 'notifications' ?
                Promise.resolve({ state: Notification.permission }) :
                originalQuery(parameters)
        );

        // Remove automation-related properties
        delete window.cdc_adoQpoasnfa76pfcZLmcfl_Array;
        delete window.cdc_adoQpoasnfa76pfcZLmcfl_Promise;
        delete window.cdc_adoQpoasnfa76pfcZLmcfl_Symbol;

        // Override chrome.runtime to appear as regular Chrome
        window.chrome = {
            runtime: {},
            loadTimes: function() {},
            csi: function() {},
            app: {},
        };

        // Make WebGL vendor/renderer look real
        const getParameterProxyHandler = {
            apply: function(target, thisArg, args) {
                const param = args[0];
                const gl = thisArg;
                // UNMASKED_VENDOR_WEBGL
                if (param === 37445) {
                    return 'Google Inc. (Intel)';
                }
                // UNMASKED_RENDERER_WEBGL
                if (param === 37446) {
                    return 'ANGLE (Intel, Intel(R) UHD Graphics 620, OpenGL 4.1)';
                }
                return target.apply(thisArg, args);
            }
        };

        // Override WebGL getParameter
        const originalGetParameter = WebGLRenderingContext.prototype.getParameter;
        WebGLRenderingContext.prototype.getParameter = new Proxy(originalGetParameter, getParameterProxyHandler);
        if (typeof WebGL2RenderingContext !== 'undefined') {
            const originalGetParameter2 = WebGL2RenderingContext.prototype.getParameter;
            WebGL2RenderingContext.prototype.getParameter = new Proxy(originalGetParameter2, getParameterProxyHandler);
        }
    """)

    # Navigate to page
    await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)

    # Simulate human behavior - random initial delay
    await page.wait_for_timeout(random.randint(500, 1500))

    # Mouse movement to simulate human presence
    await _human_like_mouse_move(page)

    await _dismiss_cookie_banners(page)
    await _auto_scroll(page, scroll_passes)

    try:
        if wait_selector:
            await page.wait_for_selector(wait_selector, timeout=min(timeout_ms, 8000))
        else:
            await page.wait_for_load_state("networkidle", timeout=timeout_ms)
    except Exception:
        pass

    # Final mouse movement before capturing content
    await _human_like_mouse_move(page)

    html = (await page.content()).encode("utf-8", errors="ignore")
    await ctx.close()
    return html, "html"

# -------- Selenium (sync, with undetected-chromedriver path, wait_selector, auto-scroll) --------
def _selenium_render_sync(
    url: str,
    cookies_header: Optional[str] = None,
    engine: str = "chrome",
    wait_selector: Optional[str] = None,
    scroll_passes: int = 6,
    timeout_ms: int = 3600000,  # 1 hour
    extra_headers: Optional[Dict[str,str]] = None,
) -> Tuple[bytes, str]:
    if engine == "firefox":
        from selenium import webdriver
        from selenium.webdriver.firefox.options import Options
        opts = Options(); opts.add_argument("-headless")
        driver = webdriver.Firefox(options=opts)
    else:
        try:
            import undetected_chromedriver as uc
            opts = uc.ChromeOptions()
            opts.add_argument("--headless=new")
            opts.add_argument("--no-sandbox")
            opts.add_argument("--disable-dev-shm-usage")
            if extra_headers and "User-Agent" in extra_headers:
                opts.add_argument(f'--user-agent={extra_headers["User-Agent"]}')
            driver = uc.Chrome(options=opts)
        except Exception:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            opts = Options()
            opts.add_argument("--headless=new")
            opts.add_argument("--no-sandbox")
            opts.add_argument("--disable-dev-shm-usage")
            if extra_headers and "User-Agent" in extra_headers:
                opts.add_argument(f'--user-agent={extra_headers["User-Agent"]}')
            driver = webdriver.Chrome(options=opts)

    # Optional cookies
    if cookies_header:
        from http.cookies import SimpleCookie
        host = urlparse(url).hostname or ""
        base = f"https://{host}"
        driver.get(base)
        c = SimpleCookie(); c.load(cookies_header)
        for m in c.values():
            try:
                driver.add_cookie({"name": m.key, "value": m.value, "path": "/", "domain": host})
            except Exception:
                pass

    driver.set_page_load_timeout(timeout_ms/1000.0)
    driver.get(url)
    # dismiss banners (best-effort)
    try:
        for sel in ["//button[contains(.,'Accept')]", "//button[contains(.,'I Agree')]"]:
            els = driver.find_elements("xpath", sel)
            if els: 
                try: els[0].click()
                except Exception: pass
    except Exception:
        pass

    # auto-scroll
    for _ in range(max(0, scroll_passes)):
        try:
            driver.execute_script("window.scrollBy(0, document.body.scrollHeight * 0.9);")
        except Exception:
            break
        time.sleep(0.5)

    # wait for selector (very crude CSS->JS)
    if wait_selector:
        try:
            # querySelectorAll returns length
            for _ in range(16):
                present = driver.execute_script(
                    "try { return document.querySelectorAll(arguments[0]).length; } catch(e){ return 0; }",
                    wait_selector
                )
                if present and int(present) > 0:
                    break
                time.sleep(0.5)
        except Exception:
            pass

    html = driver.page_source.encode("utf-8", "ignore")
    try: driver.quit()
    except Exception: pass
    return html, "html"

async def fetch_url_bytes_rendered_selenium(
    url: str,
    cookies_header: Optional[str] = None,
    wait_selector: Optional[str] = None,
    scroll_passes: int = 6,
    timeout_ms: int = 3600000,  # 1 hour
    extra_headers: Optional[Dict[str,str]] = None,
) -> Tuple[bytes, str]:
    return await asyncio.to_thread(
        _selenium_render_sync, url, cookies_header, os.getenv("SELENIUM_ENGINE","chrome"),
        wait_selector, scroll_passes, timeout_ms, extra_headers
    )

# -------- Reddit + crawler (unchanged except extra_headers pass-through to plain fetch) --------
def _reddit_api_url(url: str, limit: int, after: Optional[str] = None) -> str:
    p = urlparse(url); path = p.path.rstrip("/")
    if not path.endswith(".json"): path += "/.json"
    qs = {"limit": max(1, min(100, limit))}
    if after: qs["after"] = after
    return urlunparse(("https", "www.reddit.com", path, "", urlencode(qs), ""))

def fetch_reddit_texts_sync(url: str, limit: int = REDDIT_LIMIT) -> List[str]:
    texts: List[str] = []; after = None; fetched = 0
    sess = requests.Session(); sess.headers.update(DEFAULT_FETCH_HEADERS)
    for _ in range(20):
        api = _reddit_api_url(url, limit=min(100, max(0, limit - fetched)), after=after)
        r = sess.get(api, timeout=HTTP_TIMEOUT); r.raise_for_status(); j = r.json()
        if isinstance(j, list) and len(j) >= 2:
            post = j[0]["data"]["children"][0]["data"]
            title = (post.get("title") or "").strip()
            body  = (post.get("selftext") or "").strip()
            main  = (title + ("\n\n" + body if body else "")).strip()
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
            txt = (title + ("\n\n" + body if body else "")).strip()
            if len(txt.split()) >= 4: texts.append(txt)
        fetched += len(children); after = data.get("after")
        if not after or fetched >= limit: break
    return texts

async def fetch_reddit_texts(url: str, limit: int = REDDIT_LIMIT) -> List[str]:
    return await asyncio.to_thread(fetch_reddit_texts_sync, url, limit)

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
    start = urlparse(start_url); start_host = (start.hostname or "").lower()
    import urllib.robotparser as robotparser
    rp = robotparser.RobotFileParser()
    try:
        rp.set_url(f"{start.scheme}://{start.netloc}/robots.txt"); rp.read()
    except Exception:
        rp = None

    seen = set(); q = deque([(start_url, 0)]); results: List[Tuple[str, bytes, str]] = []
    browser = await ensure_browser(app) if render else None

    while q and len(results) < max_pages:
        url = q.popleft()[0]; depth = q.popleft()[1] if False else 0  # keep BFS depth if you prefer; simplified here
        if url in seen: continue
        seen.add(url)

        try:
            if rp and hasattr(rp, "can_fetch") and not rp.can_fetch(DEFAULT_FETCH_HEADERS["User-Agent"], url):
                continue
        except Exception:
            pass

        try:
            if render and browser:
                raw, kind = await fetch_url_bytes_rendered(url, browser, cookies_header=cookies)
            else:
                raw, kind = await fetch_url_bytes(url)
        except Exception:
            if render:
                try:
                    raw, kind = await fetch_url_bytes_rendered_selenium(url, cookies_header=cookies)
                except Exception:
                    continue
            else:
                continue

        if kind not in {"html","json","csv"}: continue
        results.append((url, raw, kind))

        if delay_ms > 0:
            await asyncio.sleep(delay_ms/1000.0)

    return results
