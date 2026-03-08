"""LinkedIn Automation Agent.

Handles:
  - Browser-based LinkedIn login (Playwright) with optional TOTP 2FA
  - Trending topic research via Tavily
  - Human-like post generation via LLM
  - Post publishing via Playwright

Uses Playwright's SYNC API inside a thread executor because Windows'
default asyncio SelectorEventLoop cannot spawn subprocesses.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
from typing import Optional

from langchain_core.messages import SystemMessage, HumanMessage

from config import get_settings
from services.llm_utils import clean_llm_json, create_llm
from services.retry import retry

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

LINKEDIN_LOGIN_URL = "https://www.linkedin.com/login"
LINKEDIN_FEED_URL = "https://www.linkedin.com/feed/"

POST_GENERATION_PROMPT = """You are a professional LinkedIn content creator. Your posts feel
genuinely human — they read like a real person sharing a real thought, not an AI output.

STRICT STYLE RULES:
- Vary sentence length: mix short punchy lines with longer thoughtful ones.
- Use a conversational, first-person voice. Share a brief personal angle or opinion.
- NEVER use these overused AI words/phrases: "delve", "leverage", "game-changer",
  "at the end of the day", "in today's fast-paced world", "it's worth noting",
  "without further ado", "I'm excited to share".
- Use line breaks between paragraphs for readability (LinkedIn formatting).
- Optionally end with 2-4 relevant hashtags on a separate line.
- Keep it between 150-300 words. Not too short, not a wall of text.
- Write like a thoughtful professional, not a motivational poster.

TOPIC / CONTEXT:
{topic}

RECENT RESEARCH (use this as inspiration, do NOT copy verbatim):
{research}

Write the LinkedIn post now. Return ONLY the post text, nothing else."""

RESEARCH_PROMPT = """Find the most interesting recent developments, news, and trends about:
{topic}

Focus on the last 7 days. Return concise bullet points summarizing the key findings."""


# ── Sync Playwright Helpers (run inside thread executor) ─────────────────────

MAX_2FA_WAIT_SECONDS = 120

SESSION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
SESSION_FILE = os.path.join(SESSION_DIR, "linkedin_session.json")

_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

_LOGGED_IN_INDICATORS = ("feed", "mynetwork", "messaging", "notifications", "jobs", "in/")
_CHALLENGE_INDICATORS = ("checkpoint", "challenge", "login", "uas/login")


def _human_delay_sync(min_ms: int = 500, max_ms: int = 2000):
    time.sleep(random.randint(min_ms, max_ms) / 1000)


def _save_session(context):
    """Persist browser cookies/localStorage to disk."""
    try:
        os.makedirs(SESSION_DIR, exist_ok=True)
        context.storage_state(path=SESSION_FILE)
        logger.info(f"Session saved to {SESSION_FILE}")
    except Exception as e:
        logger.warning(f"Could not save session: {e}")


def _has_saved_session() -> bool:
    return os.path.isfile(SESSION_FILE) and os.path.getsize(SESSION_FILE) > 50


def _create_context(browser, with_session: bool = True):
    """Create a browser context, optionally loading a saved session."""
    kwargs = {
        "user_agent": _BROWSER_UA,
        "viewport": {"width": 1280, "height": 800},
    }
    if with_session and _has_saved_session():
        try:
            kwargs["storage_state"] = SESSION_FILE
            logger.info("Restoring saved LinkedIn session")
        except Exception:
            pass
    return browser.new_context(**kwargs)


def _is_logged_in(page) -> bool:
    """Check current URL to see if we're on an authenticated page."""
    try:
        url = page.url.lower()
        return any(ind in url for ind in _LOGGED_IN_INDICATORS)
    except Exception:
        return False


def _check_logged_in(page) -> bool:
    """Check URL indicators to determine if we've reached an authenticated page."""
    try:
        current = page.url.lower()
        if any(ind in current for ind in _LOGGED_IN_INDICATORS):
            return True
        path = current.replace("https://www.linkedin.com", "").strip("/")
        if path == "" or path == "feed":
            return True
    except Exception:
        return True
    return False


def _try_click_continue(page) -> bool:
    """After 2FA approval, LinkedIn may show a button instead of auto-redirecting.
    Try to find and click it."""
    continue_selectors = [
        'button[type="submit"]',
        'button:has-text("Continue")',
        'button:has-text("Done")',
        'button:has-text("Next")',
        'button:has-text("Verify")',
        'button:has-text("Yes, this was me")',
        'button:has-text("Submit")',
        'a:has-text("Continue")',
        '#reset-password-submit-button',
        'button.primary-action-new',
        'input[type="submit"]',
    ]
    for sel in continue_selectors:
        try:
            loc = page.locator(sel)
            if loc.count() > 0 and loc.first.is_visible():
                logger.info(f"  Found post-2FA button: {sel} — clicking")
                loc.first.click()
                _human_delay_sync(2000, 4000)
                return True
        except Exception:
            continue
    return False


def _wait_for_login_complete(page, totp_secret: Optional[str]) -> tuple[bool, str]:
    """Wait for login to complete, handling TOTP or push-notification 2FA.

    After approval, LinkedIn often shows a button (Continue/Done/Submit)
    instead of auto-redirecting. This function detects and clicks through.
    """
    # Handle TOTP code entry if a secret was provided
    if totp_secret:
        try:
            import pyotp
            totp = pyotp.TOTP(totp_secret)
            code = totp.now()
            verification_input = page.locator('input[name="pin"]')
            if verification_input.count() > 0:
                verification_input.fill(code)
                _human_delay_sync(500, 1000)
                submit_btn = page.locator('button[type="submit"]')
                if submit_btn.count() > 0:
                    submit_btn.click()
                    page.wait_for_load_state("domcontentloaded", timeout=15000)
                    _human_delay_sync(2000, 3000)
        except ImportError:
            logger.warning("pyotp not installed — skipping TOTP entry")

    elapsed = 0
    poll_interval = 3
    last_url = ""

    while elapsed < MAX_2FA_WAIT_SECONDS:
        # Check if we're already logged in
        if _check_logged_in(page):
            _human_delay_sync(2000, 3000)
            logger.info("Login confirmed — landed on authenticated page")
            return True, "Login successful"

        try:
            current = page.url.lower()
        except Exception:
            return True, "Login assumed successful (browser navigated away)"

        # Detect if the page changed (2FA was approved and page updated)
        url_changed = current != last_url
        last_url = current

        is_on_challenge = any(c in current for c in _CHALLENGE_INDICATORS)

        if is_on_challenge:
            # Try clicking any post-approval buttons (Continue, Done, Submit, etc.)
            clicked = _try_click_continue(page)
            if clicked:
                _human_delay_sync(2000, 3000)
                try:
                    page.wait_for_load_state("domcontentloaded", timeout=10000)
                except Exception:
                    pass
                if _check_logged_in(page):
                    logger.info("Login confirmed after clicking continue")
                    return True, "Login successful"

            logger.info(f"  Waiting for 2FA approval... ({elapsed}s / {MAX_2FA_WAIT_SECONDS}s)")
        else:
            logger.info(f"  Post-login page: {current} ({elapsed}s / {MAX_2FA_WAIT_SECONDS}s)")
            # Non-challenge, non-logged-in page — try clicking through
            _try_click_continue(page)

        time.sleep(poll_interval)
        elapsed += poll_interval

        try:
            page.wait_for_load_state("domcontentloaded", timeout=5000)
        except Exception:
            pass

    # Final check
    if _check_logged_in(page):
        return True, "Login successful"

    return False, f"Login timed out waiting for 2FA approval ({MAX_2FA_WAIT_SECONDS}s)"


def _click_start_post(page) -> bool:
    """Find and click the 'Start a post' trigger at the TOP of the feed.

    Uses JS to locate the element, then Playwright's mouse to click it
    with real mousedown/mouseup events that LinkedIn's handlers respond to.
    """
    page.evaluate("window.scrollTo(0, 0)")
    _human_delay_sync(500, 1000)

    # Use JS to find the element and return its bounding box
    bbox = page.evaluate("""() => {
        // Strategy 1: Find the element with exact "Start a post" text
        const candidates = document.querySelectorAll('button, [role="button"], [tabindex], span, div, p');
        for (const el of candidates) {
            const text = (el.textContent || '').trim();
            if (text === 'Start a post') {
                const rect = el.getBoundingClientRect();
                if (rect.top > 0 && rect.top < 600 && rect.width > 50 && rect.height > 10) {
                    return {x: rect.x + rect.width / 2, y: rect.y + rect.height / 2, method: 'exact-text'};
                }
            }
        }

        // Strategy 2: Find by partial text match on broader elements
        for (const el of candidates) {
            const text = (el.textContent || el.getAttribute('placeholder') || '').trim();
            if (text.includes('Start a post')) {
                const rect = el.getBoundingClientRect();
                if (rect.top > 0 && rect.top < 600 && rect.width > 100 && rect.height > 20) {
                    return {x: rect.x + rect.width / 2, y: rect.y + rect.height / 2, method: 'partial-text'};
                }
            }
        }

        // Strategy 3: CSS class selectors
        const triggers = document.querySelectorAll(
            '.share-box-feed-entry__trigger, .share-box-feed-entry__top-bar'
        );
        for (const el of triggers) {
            const rect = el.getBoundingClientRect();
            if (rect.top > 0 && rect.top < 600 && rect.width > 50) {
                return {x: rect.x + rect.width / 2, y: rect.y + rect.height / 2, method: 'css-class'};
            }
        }

        return null;
    }""")

    if bbox:
        x, y = bbox["x"], bbox["y"]
        method = bbox.get("method", "unknown")
        logger.info(f"Found 'Start a post' at ({x:.0f}, {y:.0f}) via {method}")

        # Use Playwright's real mouse events (mousedown + mouseup + click)
        page.mouse.move(x, y)
        _human_delay_sync(100, 300)
        page.mouse.down()
        _human_delay_sync(50, 150)
        page.mouse.up()
        _human_delay_sync(500, 1000)

        logger.info("Clicked 'Start a post' with real mouse events")
        return True

    logger.warning("Could not find 'Start a post' element")
    return False


def _wait_for_post_modal(page) -> bool:
    """Wait for the post creation modal/dialog to open."""
    # Use a single combined selector with a reasonable timeout
    combined = '[role="dialog"], .share-box--is-open, .artdeco-modal, .share-creation-state'
    try:
        page.wait_for_selector(combined, timeout=8000)
        logger.info("Post modal opened")
        return True
    except Exception:
        pass

    # Quick check: is the editor already visible? (modal opened but selector didn't match)
    try:
        editor = page.locator('.ql-editor[contenteditable="true"]')
        if editor.count() > 0 and editor.first.is_visible():
            logger.info("Post editor detected — modal is open")
            return True
    except Exception:
        pass

    return False


def _find_post_editor(page):
    """Find the text editor INSIDE the post creation modal, not a random comment box."""
    # First try to scope to the modal/dialog
    modal = None
    for modal_sel in ['[role="dialog"]', '.share-box--is-open', '.artdeco-modal', '.share-creation-state', '.share-box']:
        try:
            loc = page.locator(modal_sel)
            if loc.count() > 0 and loc.first.is_visible():
                modal = loc.first
                break
        except Exception:
            continue

    # Search within modal if found, otherwise fall back to page
    scope = modal if modal else page

    editor_selectors = [
        '.ql-editor[contenteditable="true"]',
        '[role="textbox"][contenteditable="true"]',
        'div[contenteditable="true"][data-placeholder]',
        'div.ql-editor',
        '[role="textbox"]',
    ]
    for sel in editor_selectors:
        try:
            loc = scope.locator(sel)
            if loc.count() > 0 and loc.first.is_visible():
                logger.info(f"Found editor via: {sel} (scoped to {'modal' if modal else 'page'})")
                return loc.first
        except Exception:
            continue
    return None


def _find_submit_post_button(page):
    """Find the 'Post' submit button INSIDE the post creation modal."""
    # Scope to modal
    modal = None
    for modal_sel in ['[role="dialog"]', '.share-box--is-open', '.artdeco-modal', '.share-creation-state', '.share-box']:
        try:
            loc = page.locator(modal_sel)
            if loc.count() > 0 and loc.first.is_visible():
                modal = loc.first
                break
        except Exception:
            continue

    scope = modal if modal else page

    button_selectors = [
        'button.share-actions__primary-action',
        'button.share-actions__primary-action:has-text("Post")',
        'button[data-control-name="share.post"]',
    ]
    for sel in button_selectors:
        try:
            loc = scope.locator(sel)
            if loc.count() > 0 and loc.first.is_visible():
                logger.info(f"Found Post button via: {sel} (scoped to {'modal' if modal else 'page'})")
                return loc.first
        except Exception:
            continue

    # Last resort within modal: find button whose text is exactly "Post"
    if modal:
        try:
            buttons = modal.locator('button')
            for i in range(buttons.count()):
                btn = buttons.nth(i)
                text = btn.inner_text().strip()
                if text == "Post":
                    logger.info("Found Post button via exact text match in modal")
                    return btn
        except Exception:
            pass

    return None


def _safe_close(browser):
    try:
        browser.close()
    except Exception:
        pass


def _do_fresh_login(page, email: str, password: str, totp_secret: Optional[str]) -> tuple[bool, str]:
    """Full login with credentials + 2FA wait."""
    page.goto(LINKEDIN_LOGIN_URL, wait_until="domcontentloaded")
    _human_delay_sync(1000, 2000)

    page.fill("#username", email)
    _human_delay_sync(300, 800)
    page.fill("#password", password)
    _human_delay_sync(500, 1000)

    page.click('button[type="submit"]')
    page.wait_for_load_state("domcontentloaded", timeout=15000)
    _human_delay_sync(2000, 4000)

    return _wait_for_login_complete(page, totp_secret)


def _ensure_logged_in(
    browser, email: str, password: str, totp_secret: Optional[str], headless: bool
) -> tuple:
    """Try saved session first; if expired, do a fresh login. Returns (context, page, success, msg)."""

    # Attempt 1: Reuse saved session
    if _has_saved_session():
        logger.info("Trying saved session...")
        context = _create_context(browser, with_session=True)
        page = context.new_page()
        page.goto(LINKEDIN_FEED_URL, wait_until="domcontentloaded")
        _human_delay_sync(2000, 4000)

        if _is_logged_in(page):
            logger.info("Saved session is still valid — skipping login")
            return context, page, True, "Session restored (no 2FA needed)"

        logger.info("Saved session expired — doing fresh login")
        try:
            page.close()
            context.close()
        except Exception:
            pass

    # Attempt 2: Fresh login
    context = _create_context(browser, with_session=False)
    page = context.new_page()
    success, msg = _do_fresh_login(page, email, password, totp_secret)

    if success:
        _save_session(context)

    return context, page, success, msg


def _login_sync(
    email: str,
    password: str,
    totp_secret: Optional[str],
    headless: bool,
) -> tuple[bool, str]:
    """Sync Playwright login — called via run_in_executor."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return False, "Playwright is not installed. Run: uv pip install playwright && playwright install chromium"

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=headless,
                args=["--disable-blink-features=AutomationControlled"],
            )
            context, page, success, msg = _ensure_logged_in(
                browser, email, password, totp_secret, headless
            )
            _safe_close(browser)
            return success, msg

    except Exception as e:
        return False, f"Login error: {str(e)}"


def _publish_sync(
    email: str,
    password: str,
    post_content: str,
    totp_secret: Optional[str],
    headless: bool,
) -> tuple[bool, str]:
    """Sync Playwright publish — called via run_in_executor."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return False, "Playwright is not installed"

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=headless,
                args=["--disable-blink-features=AutomationControlled"],
            )

            context, page, success, msg = _ensure_logged_in(
                browser, email, password, totp_secret, headless
            )
            if not success:
                _safe_close(browser)
                return False, msg

            # Navigate to feed
            if "feed" not in page.url.lower():
                logger.info("Navigating to LinkedIn feed...")
                page.goto(LINKEDIN_FEED_URL, wait_until="domcontentloaded")
                _human_delay_sync(2000, 3000)

            # Wait for the feed content to appear (don't use networkidle — LinkedIn never idles)
            logger.info("Waiting for feed to render...")
            try:
                page.wait_for_selector('.share-box-feed-entry__trigger, button:has-text("Start a post"), [data-placeholder]', timeout=10000)
            except Exception:
                pass
            _human_delay_sync(1000, 2000)

            # Click "Start a post"
            logger.info("Clicking 'Start a post'...")
            if not _click_start_post(page):
                try:
                    page.screenshot(path="debug_feed.png")
                    logger.error("Saved debug screenshot to debug_feed.png")
                except Exception:
                    pass
                _safe_close(browser)
                return False, f"Could not find 'Start a post' button. Current URL: {page.url}"
            _human_delay_sync(1000, 2000)

            # Wait for the post creation modal/dialog to open
            logger.info("Waiting for post creation dialog to open...")
            if not _wait_for_post_modal(page):
                logger.warning("Post modal not detected — trying to proceed anyway")
            _human_delay_sync(1000, 2000)

            # Find the post editor inside the modal
            logger.info("Looking for post editor in dialog...")
            editor = _find_post_editor(page)
            if not editor:
                _safe_close(browser)
                return False, "Could not find post editor"

            editor.click()
            _human_delay_sync(500, 1000)

            logger.info("Typing post content...")
            paragraphs = post_content.split("\n")
            for i, para in enumerate(paragraphs):
                if para.strip():
                    editor.type(para, delay=random.randint(20, 60))
                if i < len(paragraphs) - 1:
                    editor.press("Enter")
                    _human_delay_sync(200, 500)

            _human_delay_sync(1000, 2000)

            # Find and click the Post submit button
            logger.info("Looking for Post submit button...")
            post_btn = _find_submit_post_button(page)
            if not post_btn:
                _safe_close(browser)
                return False, "Could not find Post submit button"

            logger.info("Clicking Post button...")
            post_btn.click()
            _human_delay_sync(3000, 5000)

            # Save session again after successful post (refreshes cookies)
            _save_session(context)

            _safe_close(browser)
            return True, "Post published successfully"

    except Exception as e:
        return False, f"Publish error: {str(e)}"


# ── Async Wrappers (offload sync Playwright to thread) ──────────────────────

async def login_linkedin(
    email: str,
    password: str,
    totp_secret: Optional[str] = None,
    headless: bool = True,
) -> tuple[bool, str]:
    """Log in to LinkedIn via Playwright (thread-safe on Windows)."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, _login_sync, email, password, totp_secret, headless
    )


async def publish_post_to_linkedin(
    email: str,
    password: str,
    post_content: str,
    totp_secret: Optional[str] = None,
    headless: bool = True,
) -> tuple[bool, str]:
    """Publish a post to LinkedIn via Playwright (thread-safe on Windows)."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, _publish_sync, email, password, post_content, totp_secret, headless
    )


# ── Topic Research (Tavily + SerpAPI) ─────────────────────────────────────────

async def _tavily_research(topic: str) -> list[str]:
    """Search via Tavily and return a list of summary strings."""
    settings = get_settings()
    if not settings.tavily_api_key:
        return []
    try:
        from langchain_tavily import TavilySearch

        os.environ["TAVILY_API_KEY"] = settings.tavily_api_key
        tool = TavilySearch(max_results=5, search_depth="advanced")
        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(
            None, tool.invoke, f"latest {topic} news trends 2026"
        )

        summaries: list[str] = []
        if isinstance(raw, list):
            for item in raw[:5]:
                if isinstance(item, dict):
                    title = item.get("title", "")
                    content = item.get("content", item.get("snippet", ""))[:250]
                    if title or content:
                        summaries.append(f"- {title}: {content}")
        elif raw:
            summaries.append(str(raw)[:500])
        return summaries
    except Exception as e:
        logger.warning(f"Tavily research failed: {e}")
        return []


async def _serpapi_research(topic: str) -> list[str]:
    """Search via SerpAPI (Google) and return a list of summary strings."""
    settings = get_settings()
    if not settings.serp_api_key:
        return []
    try:
        import httpx

        queries = [
            f"latest {topic} news 2026",
            f"{topic} recent developments trends",
        ]
        summaries: list[str] = []
        async with httpx.AsyncClient(timeout=30) as client:
            for q in queries:
                resp = await client.get(
                    "https://serpapi.com/search.json",
                    params={
                        "q": q,
                        "api_key": settings.serp_api_key,
                        "num": 5,
                        "engine": "google",
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                for r in data.get("organic_results", [])[:5]:
                    title = r.get("title", "")
                    snippet = r.get("snippet", "")[:250]
                    if title or snippet:
                        summaries.append(f"- {title}: {snippet}")
        return summaries
    except Exception as e:
        logger.warning(f"SerpAPI research failed: {e}")
        return []


@retry(max_attempts=2, backoff_factor=1.5)
async def research_trending_topics(topic: str) -> str:
    """Search the web via Tavily AND SerpAPI in parallel, merge results."""
    tavily_task = _tavily_research(topic)
    serpapi_task = _serpapi_research(topic)

    tavily_results, serpapi_results = await asyncio.gather(
        tavily_task, serpapi_task, return_exceptions=True
    )

    # Normalise in case gather returned an exception
    if isinstance(tavily_results, BaseException):
        logger.warning(f"Tavily gather error: {tavily_results}")
        tavily_results = []
    if isinstance(serpapi_results, BaseException):
        logger.warning(f"SerpAPI gather error: {serpapi_results}")
        serpapi_results = []

    # Merge and deduplicate (by first 60 chars of each line)
    seen: set[str] = set()
    merged: list[str] = []
    for item in [*tavily_results, *serpapi_results]:
        key = item[:60].lower().strip()
        if key not in seen:
            seen.add(key)
            merged.append(item)

    if not merged:
        return "No research results found from Tavily or Google."

    logger.info(
        f"Research complete: {len(tavily_results)} Tavily + "
        f"{len(serpapi_results)} SerpAPI = {len(merged)} unique results"
    )
    return "\n".join(merged[:12])


# ── Post Generation ──────────────────────────────────────────────────────────

async def generate_linkedin_post(
    topic: str,
    research_context: Optional[str] = None,
) -> str:
    """Generate a human-like LinkedIn post about a topic.

    If research_context is not provided, it will be fetched via Tavily first.
    """
    if not research_context:
        research_context = await research_trending_topics(topic)

    llm = create_llm(temperature=0.8)

    response = await llm.ainvoke([
        SystemMessage(content=POST_GENERATION_PROMPT.format(
            topic=topic,
            research=research_context,
        )),
    ])

    return response.content.strip()
