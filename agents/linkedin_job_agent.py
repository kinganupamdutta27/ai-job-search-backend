"""LinkedIn Job Auto-Apply Agent.

Handles:
  - Job search on LinkedIn (browser automation) + external (Tavily/SerpAPI)
  - Easy Apply form filling with multi-step modal handling
  - External ATS site form filling via LLM-guided automation
  - LLM-powered answers for custom application questions
  - Resume/CV file upload during application

Uses Playwright's SYNC API inside a thread executor (Windows compatibility).
Reuses session management from the existing linkedin_agent.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import random
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from langchain_core.messages import SystemMessage, HumanMessage

from config import get_settings
from services.llm_utils import clean_llm_json, create_llm

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

LINKEDIN_JOBS_URL = "https://www.linkedin.com/jobs/"
LINKEDIN_FEED_URL = "https://www.linkedin.com/feed/"
LINKEDIN_JOB_SEARCH_URL = "https://www.linkedin.com/jobs/search/"

MAX_APPLY_PER_SESSION = 50
DELAY_BETWEEN_APPS_MIN = 30_000
DELAY_BETWEEN_APPS_MAX = 90_000
BREAK_EVERY_N = 10
BREAK_DURATION_MIN = 120_000
BREAK_DURATION_MAX = 300_000
MAX_VISION_CALLS_PER_JOB = 3
MAX_VISION_APPLY_RETRIES = 2

FORM_FIELD_PROMPT = """You are an expert job application assistant. Given a candidate's profile and the
form fields from a job application, provide the best answers.

CANDIDATE PROFILE:
{profile}

APPLICATION PREFERENCES:
{preferences}

FORM FIELDS TO FILL (JSON array):
{fields}

For each field, return a JSON array of objects with this schema:
[
  {{
    "field_id": "<the field identifier>",
    "value": "<the answer to fill>"
  }}
]

Rules:
- For years of experience questions, use the number from the profile.
- For salary questions, use the expected salary from preferences.
- For yes/no or boolean questions about willingness (relocate, travel, etc.), answer based on preferences.
- For authorization/visa questions, use work_authorization from preferences.
- For notice period questions, use notice_period from preferences.
- For select/dropdown fields, pick the CLOSEST matching option from the available choices.
- For text fields, provide a concise professional answer.
- If the question is about a specific skill or technology, check the profile skills list.
- Keep answers honest and aligned with the candidate's actual experience.
- Return ONLY the JSON array, no markdown or extra text."""

EXTERNAL_FORM_PROMPT = """You are an expert at filling job application forms on company career pages.
Given the page content (form labels and input fields) and the candidate's profile,
identify which fields to fill and provide the values.

CANDIDATE PROFILE:
{profile}

PREFERENCES:
{preferences}

PAGE FORM CONTENT:
{form_html}

Return a JSON array of actions to take:
[
  {{
    "selector": "CSS selector or label text to identify the field",
    "action": "fill" | "select" | "click" | "upload",
    "value": "the value to enter or option to select"
  }}
]

Rules:
- Map name, email, phone, location from the profile
- For resume/CV upload fields, use action "upload"
- For dropdowns, use "select" with the closest matching option text
- For checkboxes/radio buttons, use "click"
- Skip CAPTCHA fields
- Return ONLY the JSON array"""

VISION_FORM_PROMPT = """You are an expert job application assistant with VISUAL understanding.
You are looking at a screenshot of a LinkedIn Easy Apply form step.

CANDIDATE PROFILE:
{profile}

APPLICATION PREFERENCES:
{preferences}

INSTRUCTIONS:
1. Identify every visible form field in the screenshot (text inputs, dropdowns, radio buttons, checkboxes, textareas, file uploads).
2. For each field, determine the best answer based on the candidate profile and preferences above.
3. If a field appears already filled with the correct value, mark it as "skip".
4. If you see a "Next", "Review", or "Submit application" button, include a click_button action for it as the LAST action.
5. For file upload fields (resume/CV), use action "upload_file".

Return ONLY a JSON array of actions. Each action object must have:
- "action": one of "fill_text", "select_option", "click_radio", "click_checkbox", "upload_file", "fill_textarea", "click_button", "skip"
- "label": the visible label text of the field or button (exactly as shown)
- "value": the value to fill/select/click (empty string for upload_file and click_button)

Example response:
[
  {{"action": "fill_text", "label": "First name", "value": "Anupam"}},
  {{"action": "fill_text", "label": "Last name", "value": "Dutta"}},
  {{"action": "select_option", "label": "Years of experience", "value": "3"}},
  {{"action": "click_radio", "label": "Are you legally authorized to work?", "value": "Yes"}},
  {{"action": "upload_file", "label": "Resume", "value": ""}},
  {{"action": "click_button", "label": "Next", "value": ""}}
]

Rules:
- For years of experience, use the number from the profile.
- For salary/CTC questions, use expected_salary or expected_ctc from preferences.
- For yes/no authorization questions, answer based on work_authorization.
- For notice period, use notice_period from preferences.
- For select/dropdown fields, pick the CLOSEST matching option visible in the screenshot.
- Keep answers honest and aligned with the candidate's actual experience.
- If a field is disabled or greyed out, skip it.
- Return ONLY the JSON array, no markdown fences or extra text."""

VISION_APPLY_BUTTON_PROMPT = """You are an expert at navigating LinkedIn job pages.
You are looking at a screenshot of a LinkedIn page where we need to find and click
an APPLY or EASY APPLY button.

Analyze the screenshot and respond with a JSON object describing the button to click.

Look for:
1. A prominent "Easy Apply" button (usually blue/green with LinkedIn logo)
2. An "Apply" button or link
3. A "Apply on company website" button
4. Any other application-related button

If you see a job listing sidebar on the left and a job detail panel on the right,
the Apply button is usually in the right panel near the top.

If the page shows "Similar Jobs" or a job collection, look for the Easy Apply button
in the job detail section on the right side.

Return a JSON object:
{{
  "found": true/false,
  "button_text": "the exact text on the button",
  "description": "brief description of where the button is",
  "is_easy_apply": true if it's an Easy Apply button (not external),
  "approximate_position": "top-right" | "center" | "below-title" | "other",
  "nearby_text": "any text near the button to help identify it (job title, company name)"
}}

If NO apply button is visible at all, return {{"found": false}}.
Return ONLY the JSON object, no markdown fences or extra text."""


# ── Reuse helpers from linkedin_agent ────────────────────────────────────────

def _human_delay_sync(min_ms: int = 500, max_ms: int = 2000):
    time.sleep(random.randint(min_ms, max_ms) / 1000)


def _safe_close(browser):
    try:
        browser.close()
    except Exception:
        pass


def _ensure_canonical_job_view(page, original_url: str) -> bool:
    """If we're on a non-standard LinkedIn page (collection, similar-jobs,
    search result), extract the job ID and navigate to /jobs/view/<id>/.

    Tries up to 2 times with verification. Returns True if redirect performed.
    """
    for attempt in range(2):
        current_url = page.url
        # Match both /jobs/view/12345 and /jobs/view/slug-text-12345
        is_direct_view = bool(re.search(r'/jobs/view/(\d+)/?(\?|$)', current_url))
        if is_direct_view:
            return attempt > 0

        if "linkedin.com" not in current_url:
            return False

        # Extract job ID from various URL patterns
        job_id = None
        for url_to_check in (current_url, original_url):
            for pattern in (
                r'currentJobId=(\d+)',
                r'referenceJobId=(\d+)',
                r'/jobs/view/[a-zA-Z0-9_-]*?(\d{7,})',
                r'/jobs/view/(\d+)',
                r'originToLandingJobPostings=(\d+)',
            ):
                m = re.search(pattern, url_to_check)
                if m:
                    job_id = m.group(1)
                    break
            if job_id:
                break

        if not job_id:
            logger.warning(f"Could not extract job ID from: {current_url}")
            return False

        direct_url = f"https://www.linkedin.com/jobs/view/{job_id}/"
        logger.info(f"Non-standard LinkedIn page — redirecting to {direct_url} (attempt {attempt + 1})")
        try:
            page.goto(direct_url, wait_until="networkidle", timeout=20000)
        except Exception:
            try:
                page.goto(direct_url, wait_until="domcontentloaded", timeout=15000)
            except Exception as e:
                logger.warning(f"Navigation to canonical URL failed: {e}")
                return False
        _human_delay_sync(2000, 4000)

        # Verify we actually arrived
        final_url = page.url
        if re.search(r'/jobs/view/\d+', final_url):
            logger.info(f"Successfully redirected to: {final_url}")
            return True
        else:
            logger.warning(f"Redirect attempt {attempt + 1} failed — still on {final_url}")

    return False


# Reuse session infrastructure from linkedin_agent
SESSION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
SESSION_FILE = os.path.join(SESSION_DIR, "linkedin_session.json")

_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

_LOGGED_IN_INDICATORS = ("feed", "mynetwork", "messaging", "notifications", "jobs", "in/")


def _has_saved_session() -> bool:
    return os.path.isfile(SESSION_FILE) and os.path.getsize(SESSION_FILE) > 50


def _create_context(browser, with_session: bool = True):
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


def _save_session(context):
    try:
        os.makedirs(SESSION_DIR, exist_ok=True)
        context.storage_state(path=SESSION_FILE)
        logger.info(f"Session saved to {SESSION_FILE}")
    except Exception as e:
        logger.warning(f"Could not save session: {e}")


def _is_logged_in(page) -> bool:
    try:
        url = page.url.lower()
        return any(ind in url for ind in _LOGGED_IN_INDICATORS)
    except Exception:
        return False


def _do_fresh_login(page, email: str, password: str, totp_secret: Optional[str]) -> tuple[bool, str]:
    from agents.linkedin_agent import _wait_for_login_complete
    page.goto("https://www.linkedin.com/login", wait_until="domcontentloaded")
    _human_delay_sync(1000, 2000)
    page.fill("#username", email)
    _human_delay_sync(300, 800)
    page.fill("#password", password)
    _human_delay_sync(500, 1000)
    page.click('button[type="submit"]')
    page.wait_for_load_state("domcontentloaded", timeout=15000)
    _human_delay_sync(2000, 4000)
    return _wait_for_login_complete(page, totp_secret)


def _ensure_logged_in(browser, email: str, password: str, totp_secret: Optional[str]) -> tuple:
    if _has_saved_session():
        logger.info("Trying saved session for job apply...")
        context = _create_context(browser, with_session=True)
        page = context.new_page()
        page.goto(LINKEDIN_FEED_URL, wait_until="domcontentloaded")
        _human_delay_sync(2000, 4000)
        if _is_logged_in(page):
            logger.info("Saved session valid — skipping login")
            return context, page, True, "Session restored"
        logger.info("Session expired — fresh login needed")
        try:
            page.close()
            context.close()
        except Exception:
            pass

    context = _create_context(browser, with_session=False)
    page = context.new_page()
    success, msg = _do_fresh_login(page, email, password, totp_secret)
    if success:
        _save_session(context)
    return context, page, success, msg


# ── Job Search on LinkedIn (Browser) ─────────────────────────────────────────

def _search_jobs_linkedin(
    page, keywords: str, location: str, experience_level: str,
    date_posted: str, easy_apply_only: bool,
) -> list[dict]:
    """Search LinkedIn Jobs via browser and return job card data."""
    logger.info(f"Searching LinkedIn Jobs: '{keywords}' in '{location}'")

    # Build search URL with query params
    params = [f"keywords={keywords.replace(' ', '%20')}"]
    if location:
        params.append(f"location={location.replace(' ', '%20')}")

    # Experience level mapping
    exp_map = {
        "internship": "1", "entry": "2", "associate": "3",
        "mid-senior": "4", "director": "5", "executive": "6",
    }
    if experience_level and experience_level.lower() in exp_map:
        params.append(f"f_E={exp_map[experience_level.lower()]}")

    # Date posted mapping
    date_map = {"past 24h": "r86400", "past week": "r604800", "past month": "r2592000"}
    if date_posted and date_posted.lower() in date_map:
        params.append(f"f_TPR={date_map[date_posted.lower()]}")

    if easy_apply_only:
        params.append("f_AL=true")

    search_url = f"{LINKEDIN_JOB_SEARCH_URL}?{'&'.join(params)}"
    logger.info(f"Navigating to: {search_url}")

    page.goto(search_url, wait_until="domcontentloaded")
    _human_delay_sync(3000, 5000)

    # Scroll to load more results
    for _ in range(3):
        page.evaluate("window.scrollBy(0, 800)")
        _human_delay_sync(1000, 2000)

    # Extract job cards
    jobs = page.evaluate("""() => {
        function getTitle(el) {
            // Prefer aria-label (clean, single title) over textContent
            const aria = el.getAttribute('aria-label');
            if (aria) return aria.trim();

            // Try innerText (respects visibility, skips hidden SR-only spans)
            if (el.innerText) {
                const t = el.innerText.replace(/\\s+/g, ' ').trim();
                if (t) return t;
            }
            return (el.textContent || '').replace(/\\s+/g, ' ').trim();
        }

        function cleanText(el) {
            if (el.innerText) {
                const t = el.innerText.replace(/\\s+/g, ' ').trim();
                if (t) return t;
            }
            return (el.textContent || '').replace(/\\s+/g, ' ').trim();
        }

        const cards = document.querySelectorAll('.job-card-container, .jobs-search-results__list-item, [data-job-id]');
        const results = [];
        const seen = new Set();

        for (const card of cards) {
            try {
                const titleEl = card.querySelector('.job-card-list__title, .job-card-container__link, a[href*="/jobs/view/"]');
                const companyEl = card.querySelector('.job-card-container__primary-description, .artdeco-entity-lockup__subtitle, .job-card-container__company-name');
                const locationEl = card.querySelector('.job-card-container__metadata-item, .artdeco-entity-lockup__caption, .job-card-container__metadata-wrapper');

                const title = titleEl ? getTitle(titleEl) : '';
                const company = companyEl ? cleanText(companyEl) : '';
                const location = locationEl ? cleanText(locationEl) : '';

                let url = '';
                const linkEl = card.querySelector('a[href*="/jobs/view/"]');
                if (linkEl) {
                    url = linkEl.href.split('?')[0];
                }

                if (!title || !url) continue;
                if (seen.has(url)) continue;
                seen.add(url);

                const isEasyApply = card.textContent.includes('Easy Apply');

                results.push({
                    title: title.substring(0, 300),
                    company: company.substring(0, 200),
                    location: location.substring(0, 200),
                    url: url,
                    is_easy_apply: isEasyApply,
                });
            } catch (e) {
                continue;
            }
        }
        return results;
    }""")

    logger.info(f"Found {len(jobs)} jobs on LinkedIn")
    return jobs


# ── Job Search External (Tavily + SerpAPI) ───────────────────────────────────

async def _search_jobs_tavily(keywords: str, location: str) -> list[dict]:
    settings = get_settings()
    if not settings.tavily_api_key:
        return []
    try:
        from langchain_tavily import TavilySearch
        os.environ["TAVILY_API_KEY"] = settings.tavily_api_key
        tool = TavilySearch(max_results=10, search_depth="advanced")
        query = f"LinkedIn jobs {keywords} {location} site:linkedin.com/jobs"
        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(None, tool.invoke, query)

        jobs = []
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict):
                    url = item.get("url", "")
                    if "linkedin.com/jobs" in url:
                        clean_url = url.split("?")[0]
                        jobs.append({
                            "title": item.get("title", "").split(" - ")[0].strip()[:300],
                            "company": "",
                            "location": location,
                            "url": clean_url,
                            "is_easy_apply": "/jobs/view/" in clean_url,
                        })
        return jobs
    except Exception as e:
        logger.warning(f"Tavily job search failed: {e}")
        return []


async def _search_jobs_serpapi(keywords: str, location: str) -> list[dict]:
    settings = get_settings()
    if not settings.serp_api_key:
        return []
    try:
        import httpx
        jobs = []
        query = f"site:linkedin.com/jobs/view {keywords} {location}"
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                "https://serpapi.com/search.json",
                params={
                    "q": query,
                    "api_key": settings.serp_api_key,
                    "num": 15,
                    "engine": "google",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            for r in data.get("organic_results", []):
                url = r.get("link", "")
                if "linkedin.com/jobs" in url:
                    title = r.get("title", "").split(" - ")[0].strip()
                    clean_url = url.split("?")[0]
                    jobs.append({
                        "title": title[:300],
                        "company": "",
                        "location": location,
                        "url": clean_url,
                        "is_easy_apply": "/jobs/view/" in clean_url,
                    })
        return jobs
    except Exception as e:
        logger.warning(f"SerpAPI job search failed: {e}")
        return []


async def search_jobs_external(keywords: str, location: str) -> list[dict]:
    """Search for LinkedIn job URLs via Tavily + SerpAPI in parallel."""
    tavily_task = _search_jobs_tavily(keywords, location)
    serp_task = _search_jobs_serpapi(keywords, location)
    tavily_results, serp_results = await asyncio.gather(
        tavily_task, serp_task, return_exceptions=True
    )
    if isinstance(tavily_results, BaseException):
        tavily_results = []
    if isinstance(serp_results, BaseException):
        serp_results = []

    seen: set[str] = set()
    merged: list[dict] = []
    for job in [*tavily_results, *serp_results]:
        url = job.get("url", "")
        if url and url not in seen:
            seen.add(url)
            merged.append(job)

    logger.info(f"External search: {len(tavily_results)} Tavily + {len(serp_results)} SerpAPI = {len(merged)} unique")
    return merged


# ── Easy Apply Form Handling ─────────────────────────────────────────────────

def _read_form_fields(page) -> list[dict]:
    """Extract all visible form fields from the current Easy Apply step.

    Scopes strictly to the Easy Apply modal/form. Uses two passes:
    1. Group-based: finds labelled groups (.fb-dash-form-element, etc.)
    2. Loose scan: visible inputs/selects/textareas inside the modal only.

    Filters out navbar/page-level elements (search bar, language picker).
    CSS-escapes React-generated IDs that contain special characters.
    """
    fields = page.evaluate("""() => {
        // ── Strictly scope to Easy Apply modal/form ──
        // Try increasingly specific selectors; NEVER fall back to body
        const modalSelectors = [
            '.jobs-easy-apply-modal',
            '.jobs-easy-apply-content',
            '.artdeco-modal[role="dialog"]',
            '[role="dialog"]',
        ];
        let modal = null;
        for (const sel of modalSelectors) {
            const el = document.querySelector(sel);
            if (el && el.offsetParent !== null) { modal = el; break; }
        }
        // For SDUI full-page flow (no dialog modal), look for the apply form section
        if (!modal) {
            const sduiForms = document.querySelectorAll(
                '.jobs-easy-apply-form-section, .jobs-apply-form, form[data-test-form]'
            );
            for (const f of sduiForms) {
                if (f.offsetParent !== null) { modal = f; break; }
            }
        }
        if (!modal) return [];

        const fields = [];
        const seenIds = new Set();

        // Labels to ignore — these are page-level elements, not form fields
        const NOISE_LABELS = new Set([
            'search', 'select language', 'messaging',
            'notifications', 'home', 'my network', 'jobs',
        ]);

        function isNoiseLabel(label) {
            return NOISE_LABELS.has(label.toLowerCase().trim());
        }

        function addField(f) {
            if (isNoiseLabel(f.label)) return;
            const key = f.id || f.selector;
            if (seenIds.has(key)) return;
            seenIds.add(key);
            fields.push(f);
        }

        function cssEscape(id) {
            // CSS.escape handles special chars like colons in React IDs
            if (typeof CSS !== 'undefined' && CSS.escape) {
                return '#' + CSS.escape(id);
            }
            // Manual fallback: escape chars that break CSS selectors
            return '#' + id.replace(/([^\\w-])/g, '\\\\$1');
        }

        function getLabelFor(el) {
            const aria = el.getAttribute('aria-label');
            if (aria) return aria.trim();

            if (el.id) {
                try {
                    const lbl = modal.querySelector('label[for="' + CSS.escape(el.id) + '"]')
                              || document.querySelector('label[for="' + el.id + '"]');
                    if (lbl) return lbl.textContent.replace(/\\s+/g,' ').trim();
                } catch(e) {}
            }

            const groupSelectors = [
                '.fb-dash-form-element',
                '.jobs-easy-apply-form-section__grouping',
                'div[data-test-form-element]',
                '.artdeco-text-input',
            ];
            for (const gs of groupSelectors) {
                const grp = el.closest(gs);
                if (grp) {
                    const lbl = grp.querySelector('label, legend, .fb-dash-form-element__label, .artdeco-text-input--label, span.t-14');
                    if (lbl) return lbl.textContent.replace(/\\s+/g,' ').trim();
                }
            }

            let parent = el.parentElement;
            for (let i = 0; i < 4 && parent && parent !== modal; i++) {
                const lbl = parent.querySelector('label, legend, span.t-14, span.t-bold');
                if (lbl && lbl.textContent.trim().length > 2) {
                    return lbl.textContent.replace(/\\s+/g,' ').trim();
                }
                parent = parent.parentElement;
            }

            return el.getAttribute('placeholder') || '';
        }

        function makeSelector(el, fallback) {
            if (el.id) return cssEscape(el.id);
            if (el.name) return el.tagName.toLowerCase() + '[name="' + el.name + '"]';
            // Generate a unique data attribute as last resort
            const uid = Math.random().toString(36).substr(2, 8);
            el.setAttribute('data-ff', uid);
            return '[data-ff="' + uid + '"]';
        }

        // ── Pass 1: group-based scan ──
        const groups = modal.querySelectorAll(
            '.fb-dash-form-element, .jobs-easy-apply-form-section__grouping, div[data-test-form-element]'
        );
        for (const group of groups) {
            const labelEl = group.querySelector('label, .fb-dash-form-element__label, .artdeco-text-input--label, legend, span.t-14');
            const label = labelEl ? labelEl.textContent.replace(/\\s+/g,' ').trim() : '';

            const textInput = group.querySelector('input[type="text"], input[type="email"], input[type="tel"], input[type="number"], input:not([type]):not([type="hidden"]):not([type="file"]):not([type="radio"]):not([type="checkbox"])');
            if (textInput && textInput.offsetParent !== null) {
                addField({ id: textInput.id || textInput.name || 'text_'+fields.length, label, type: textInput.type||'text', value: textInput.value||'', required: textInput.required||label.includes('*'), selector: makeSelector(textInput, 'input') });
                continue;
            }
            const selectEl = group.querySelector('select');
            if (selectEl && selectEl.offsetParent !== null) {
                const opts = [];
                for (const o of selectEl.options) { if(o.value) opts.push({value:o.value,text:o.textContent.trim()}); }
                addField({ id: selectEl.id||selectEl.name||'select_'+fields.length, label, type:'select', value:selectEl.value||'', options:opts, required:selectEl.required||label.includes('*'), selector:makeSelector(selectEl,'select') });
                continue;
            }
            const radios = group.querySelectorAll('input[type="radio"]');
            if (radios.length > 0) {
                const opts = [];
                for (const r of radios) {
                    const rl = group.querySelector('label[for="'+r.id+'"]');
                    opts.push({value:r.value, text:rl?rl.textContent.trim():r.value});
                }
                addField({ id:radios[0].name||'radio_'+fields.length, label, type:'radio', value:'', options:opts, required:true, selector:'input[name="'+radios[0].name+'"]' });
                continue;
            }
            const textarea = group.querySelector('textarea');
            if (textarea && textarea.offsetParent !== null) {
                addField({ id:textarea.id||textarea.name||'textarea_'+fields.length, label, type:'textarea', value:textarea.value||'', required:textarea.required||label.includes('*'), selector:makeSelector(textarea,'textarea') });
                continue;
            }
            const fileInput = group.querySelector('input[type="file"]');
            if (fileInput) {
                addField({ id:fileInput.id||fileInput.name||'file_'+fields.length, label, type:'file', value:'', required:false, selector:makeSelector(fileInput,'input[type="file"]') });
                continue;
            }
        }

        // ── Pass 2: loose scan — ONLY inside modal ──
        const allInputs = modal.querySelectorAll('input, select, textarea');
        for (const el of allInputs) {
            if (el.offsetParent === null) continue;
            const t = el.type || el.tagName.toLowerCase();
            if (t === 'hidden' || t === 'submit' || t === 'button') continue;

            // Skip elements inside navbars, search bars, or headers
            if (el.closest('nav, header, .global-nav, .search-global-typeahead')) continue;

            const label = getLabelFor(el);
            const sel = makeSelector(el, el.tagName.toLowerCase());

            if (el.tagName === 'SELECT') {
                const opts = [];
                for (const o of el.options) { if(o.value) opts.push({value:o.value,text:o.textContent.trim()}); }
                addField({ id:el.id||el.name||'select_'+fields.length, label, type:'select', value:el.value||'', options:opts, required:el.required||label.includes('*'), selector:sel });
            } else if (el.tagName === 'TEXTAREA') {
                addField({ id:el.id||el.name||'textarea_'+fields.length, label, type:'textarea', value:el.value||'', required:el.required||label.includes('*'), selector:sel });
            } else if (el.type === 'file') {
                addField({ id:el.id||el.name||'file_'+fields.length, label, type:'file', value:'', required:false, selector:sel });
            } else if (el.type === 'radio' || el.type === 'checkbox') {
                // Skip in loose scan
            } else {
                addField({ id:el.id||el.name||'text_'+fields.length, label, type:el.type||'text', value:el.value||'', required:el.required||label.includes('*'), selector:sel });
            }
        }

        return fields;
    }""")

    logger.info(f"Found {len(fields)} form fields on current step")
    return fields


def _normalise_question(text: str) -> str:
    """Normalise a question/label for fuzzy matching in the Q&A memory."""
    return re.sub(r'[^a-z0-9 ]', '', text.lower()).strip()


def _load_saved_qa_sync() -> dict[str, str]:
    """Load all saved Q&A pairs from DB (sync, for use inside thread executor)."""
    import sqlite3
    from database import DB_PATH

    qa_map: dict[str, str] = {}
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cur = conn.cursor()
        cur.execute("SELECT question_normalised, answer FROM saved_qa WHERE answer != ''")
        for row in cur.fetchall():
            qa_map[row[0]] = row[1]
        conn.close()
        logger.info(f"Loaded {len(qa_map)} saved Q&A pairs from memory")
    except Exception as e:
        logger.warning(f"Could not load saved Q&A: {e}")
    return qa_map


def _save_qa_pairs_sync(questions_answered: dict[str, str]):
    """Save Q&A pairs to DB. Updates existing entries, inserts new ones.

    Saves ALL questions — even those with empty answers — so users can
    see unanswered questions in the frontend and provide values for next run.
    """
    import sqlite3
    from database import DB_PATH

    if not questions_answered:
        return

    saved = 0
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cur = conn.cursor()

        cur.execute(
            "CREATE TABLE IF NOT EXISTS saved_qa ("
            "id TEXT PRIMARY KEY, question TEXT NOT NULL, "
            "question_normalised TEXT NOT NULL, answer TEXT NOT NULL DEFAULT '', "
            "field_type TEXT NOT NULL DEFAULT 'text', source TEXT NOT NULL DEFAULT 'auto', "
            "times_used INTEGER NOT NULL DEFAULT 1, "
            "created_at TEXT NOT NULL DEFAULT (datetime('now')), "
            "updated_at TEXT NOT NULL DEFAULT (datetime('now')))"
        )

        for question, answer in questions_answered.items():
            if not question:
                continue
            normalised = _normalise_question(question)
            if not normalised:
                continue

            answer_str = str(answer) if answer else ""

            existing = cur.execute(
                "SELECT id, times_used, answer FROM saved_qa WHERE question_normalised = ?",
                (normalised,),
            ).fetchone()

            now = datetime.now(timezone.utc).isoformat()
            if existing:
                new_answer = answer_str if answer_str else existing[2]
                cur.execute(
                    "UPDATE saved_qa SET answer = ?, times_used = ?, updated_at = ? WHERE id = ?",
                    (new_answer, existing[1] + 1, now, existing[0]),
                )
            else:
                cur.execute(
                    "INSERT INTO saved_qa (id, question, question_normalised, answer, field_type, source, times_used, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, 'text', 'auto', 1, ?, ?)",
                    (str(uuid.uuid4()), question, normalised, answer_str, now, now),
                )
            saved += 1

        conn.commit()
        conn.close()
        logger.info(f"Saved {saved}/{len(questions_answered)} Q&A pairs to memory")
    except Exception as e:
        logger.warning(f"Could not save Q&A pairs: {e}", exc_info=True)


def _fill_field_deterministic(field: dict, profile: dict, preferences: dict) -> Optional[str]:
    """Try to fill a field deterministically from profile/preferences data.

    Uses keyword-based matching on the field label. Each rule is a tuple of
    (keywords_to_match, value_getter) where ALL keywords must appear in the label.
    Rules are checked in order; first match wins.
    """
    label = field.get("label", "").lower().strip()
    field_type = field.get("type", "")

    if field_type == "file":
        return "__FILE_UPLOAD__"

    if not label:
        return None

    total_yoe_float = float(preferences.get("years_of_experience", profile.get("years_of_experience", 0)))
    yoe = str(int(total_yoe_float))
    yoe_months = str(int((total_yoe_float % 1) * 12))
    name = profile.get("name", "")
    first_name = name.split()[0] if name else ""
    last_name = " ".join(name.split()[1:]) if name else ""
    expected_salary = preferences.get("expected_salary", "") or preferences.get("expected_ctc", "")
    current_ctc = preferences.get("current_ctc", "")
    expected_ctc = preferences.get("expected_ctc", "")
    notice_period = preferences.get("notice_period", "")
    work_auth = preferences.get("work_authorization", "")
    phone = profile.get("phone", "")
    email_val = profile.get("email", "")
    # Location: try profile, then preferred_locations, then additional_info
    location = (
        profile.get("location", "")
        or (preferences.get("preferred_locations", []) or [""])[0]
        or preferences.get("additional_info", {}).get("location", "")
        or preferences.get("additional_info", {}).get("city", "")
    )
    pref_locations = preferences.get("preferred_locations", [])
    willing_relocate = preferences.get("willing_to_relocate", False)

    linkedin_url = profile.get("linkedin", "") or preferences.get("additional_info", {}).get("linkedin", "")
    portfolio_url = profile.get("portfolio", "") or preferences.get("additional_info", {}).get("portfolio", "")
    summary = profile.get("summary", "")

    has_auth = work_auth in ("Citizen", "PR", "Work Visa", "Yes", "yes", "true", "True")

    logger.debug(
        f"Deterministic fill context — label='{label}', location='{location}', "
        f"ctc='{current_ctc}', expected='{expected_ctc}', yoe='{yoe}', "
        f"notice='{notice_period}'"
    )

    # ── Rule-based matching: (required_keywords, value) ──
    # Keywords are checked with ALL-match: every keyword must appear in label.
    rules: list[tuple[list[str], str]] = [
        # Name fields
        (["first", "name"], first_name),
        (["last", "name"], last_name),
        (["full", "name"], name),

        # Contact
        (["email"], email_val),
        (["phone", "number"], phone),
        (["phone"], phone),
        (["mobile"], phone),

        # CTC / Salary — specific before generic
        (["current", "ctc", "inr"], current_ctc),
        (["expected", "ctc", "inr"], expected_ctc),
        (["current", "ctc"], current_ctc),
        (["expected", "ctc"], expected_ctc),
        (["current", "salary"], current_ctc),
        (["expected", "salary"], expected_salary or expected_ctc),
        (["ctc", "inr"], current_ctc),
        (["ctc"], current_ctc),
        (["salary"], expected_salary or expected_ctc),
        (["lpa"], current_ctc),
        (["compensation"], expected_salary or expected_ctc),
        (["annual", "package"], current_ctc),

        # Notice period
        (["notice", "period"], notice_period),
        (["notice", "day"], notice_period),
        (["notice"], notice_period),
        (["joining", "time"], notice_period),
        (["joining"], notice_period),

        # Experience — years
        (["total", "years", "professional"], yoe),
        (["years", "professional", "experience"], yoe),
        (["total", "years", "experience"], yoe),
        (["years", "experience"], yoe),
        (["total", "experience"], yoe),
        (["overall", "experience"], yoe),
        (["years", "of", "experience"], yoe),

        # Experience — months (for "additional months" dropdown)
        (["additional", "months"], yoe_months),
        (["months", "experience"], yoe_months),

        # Work authorization / visa
        (["work", "authorization"], work_auth or ("Yes" if has_auth else "")),
        (["authorized", "work"], "Yes" if has_auth else "No"),
        (["legally", "authorized"], "Yes" if has_auth else "No"),
        (["visa", "sponsor"], "No" if has_auth else "Yes"),
        (["sponsorship"], "No" if has_auth else "Yes"),
        (["require", "visa"], "No" if has_auth else "Yes"),
        (["visa"], work_auth),

        # Location / City
        (["location", "city"], location),
        (["city"], location),
        (["current", "location"], location),
        (["location"], location),
        (["current", "city"], location),

        # Relocation
        (["relocat"], "Yes" if willing_relocate else "No"),

        # LinkedIn / Portfolio
        (["linkedin"], linkedin_url),
        (["portfolio"], portfolio_url),
        (["website"], portfolio_url),
        (["github"], preferences.get("additional_info", {}).get("github", "")),

        # Summary / Cover letter
        (["cover", "letter"], summary),

        # Common flexibility questions — default to Yes
        (["comfortable"], "Yes"),
        (["willing"], "Yes"),
        (["open", "to"], "Yes"),
        (["agree"], "Yes"),

        # Night shift / rotational
        (["night", "shift"], "Yes"),
        (["rotational"], "Yes"),
        (["work", "from", "office"], "Yes"),
        (["hybrid"], "Yes"),
        (["remote"], "Yes"),

        # Gender / Diversity (skip these — leave for LLM)
        # (["gender"], ""),
        # (["race"], ""),

        # "Are you" questions — default Yes
        (["are you", "18"], "Yes"),
        (["above 18"], "Yes"),
    ]

    for keywords, value in rules:
        if all(kw in label for kw in keywords) and value:
            return str(value)

    # ── Technology-specific experience questions ──
    # e.g. "How many years of experience do you have with Django REST Framework?"
    # Check if the question asks about a skill the candidate has
    if "experience" in label and ("year" in label or "how many" in label or "how long" in label):
        skills = [s.lower() for s in profile.get("skills", [])]
        for skill in skills:
            if skill in label:
                return yoe
        # Skill not in profile — return "0" rather than leaving blank
        return "0"

    return None


def _answer_with_llm_sync(fields: list[dict], profile: dict, preferences: dict) -> list[dict]:
    """Use GPT-4o (sync) to answer custom application questions.

    Uses the OpenAI SDK directly to avoid asyncio event-loop conflicts
    when called from within a Playwright sync thread.
    """
    from openai import OpenAI
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)
    prompt = FORM_FIELD_PROMPT.format(
        profile=json.dumps(profile, indent=2),
        preferences=json.dumps(preferences, indent=2),
        fields=json.dumps(fields, indent=2),
    )
    try:
        resp = client.chat.completions.create(
            model=settings.openai_model,
            temperature=0.2,
            messages=[{"role": "system", "content": prompt}],
        )
        cleaned = clean_llm_json(resp.choices[0].message.content or "")
        return json.loads(cleaned)
    except Exception as e:
        logger.warning(f"LLM form field answer error: {e}")
        return []


def _vision_analyze_form_sync(
    page, profile: dict, preferences: dict, cv_path: str,
    saved_qa: Optional[dict[str, str]] = None,
) -> list[dict]:
    """Take a screenshot of the current form step, send it to GPT-4o Vision,
    and return a list of structured fill/click actions.

    This is the fallback when DOM-based field parsing fails (e.g. SDUI forms,
    custom question layouts, or obfuscated class names).
    """
    import base64
    from openai import OpenAI

    logger.info("Vision AI: taking screenshot of current form step...")
    screenshot_bytes = page.screenshot(full_page=False)
    b64_img = base64.b64encode(screenshot_bytes).decode()

    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)

    # Inject saved Q&A into preferences so Vision AI uses them
    enriched_prefs = dict(preferences)
    if saved_qa:
        enriched_prefs["previously_answered_questions"] = {
            k: v for k, v in saved_qa.items() if v
        }

    prompt_text = VISION_FORM_PROMPT.format(
        profile=json.dumps(profile, indent=2),
        preferences=json.dumps(enriched_prefs, indent=2),
    )

    try:
        resp = client.chat.completions.create(
            model=settings.openai_model,
            temperature=0.1,
            max_tokens=2000,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{b64_img}",
                        "detail": "high",
                    }},
                ],
            }],
        )
        raw = resp.choices[0].message.content or ""
        cleaned = clean_llm_json(raw)
        actions = json.loads(cleaned)
        logger.info(f"Vision AI returned {len(actions)} actions")
        return actions
    except Exception as e:
        logger.warning(f"Vision AI analysis failed: {e}")
        return []


def _execute_vision_actions_sync(
    page, actions: list[dict], cv_path: str,
) -> dict:
    """Execute the action list returned by the Vision AI on the current page.

    Uses Playwright's accessible-name locators (get_by_label, get_by_role,
    get_by_text) which are resilient to class-name changes.
    Returns a dict of {question_label: answer_value} for tracking.
    """
    questions_answered: dict[str, str] = {}

    for act in actions:
        action = act.get("action", "")
        label = act.get("label", "").strip()
        value = act.get("value", "").strip()

        if action == "skip" or not label:
            continue

        try:
            if action == "fill_text":
                el = _find_input_by_label(page, label)
                if el and el.is_visible():
                    el.clear()
                    _human_delay_sync(100, 300)
                    el.type(value, delay=random.randint(20, 50))
                    questions_answered[label] = value
                    logger.info(f"Vision fill_text: '{label}' = '{value}'")
                else:
                    logger.warning(f"Vision fill_text: field '{label}' not found/visible")

            elif action == "fill_textarea":
                el = _find_input_by_label(page, label, tag="textarea")
                if el and el.is_visible():
                    el.clear()
                    _human_delay_sync(100, 300)
                    el.type(value, delay=random.randint(15, 40))
                    questions_answered[label] = value
                    logger.info(f"Vision fill_textarea: '{label}'")
                else:
                    logger.warning(f"Vision fill_textarea: '{label}' not found/visible")

            elif action == "select_option":
                el = _find_input_by_label(page, label, tag="select")
                if el and el.is_visible():
                    try:
                        el.select_option(label=value)
                    except Exception:
                        try:
                            el.select_option(value=value)
                        except Exception:
                            # Attempt partial text match on option labels
                            options = el.locator("option").all_text_contents()
                            best = _best_option_text(value, options)
                            if best:
                                el.select_option(label=best)
                    questions_answered[label] = value
                    logger.info(f"Vision select_option: '{label}' = '{value}'")
                else:
                    logger.warning(f"Vision select_option: '{label}' not found/visible")

            elif action == "click_radio":
                clicked = _click_radio_by_label(page, label, value)
                if clicked:
                    questions_answered[label] = value
                    logger.info(f"Vision click_radio: '{label}' = '{value}'")
                else:
                    logger.warning(f"Vision click_radio: '{label}' / '{value}' not found")

            elif action == "click_checkbox":
                try:
                    cb = page.get_by_label(label)
                    if cb.count() > 0 and cb.first.is_visible():
                        if not cb.first.is_checked():
                            cb.first.check()
                        questions_answered[label] = "checked"
                        logger.info(f"Vision click_checkbox: '{label}'")
                except Exception:
                    logger.warning(f"Vision click_checkbox: '{label}' not found")

            elif action == "upload_file":
                if cv_path and os.path.isfile(cv_path):
                    file_inputs = page.locator('input[type="file"]')
                    if file_inputs.count() > 0:
                        file_inputs.first.set_input_files(cv_path)
                        _human_delay_sync(1000, 2000)
                        questions_answered[label or "Resume"] = os.path.basename(cv_path)
                        logger.info(f"Vision upload_file: {cv_path}")

            elif action == "click_button":
                btn = page.get_by_role("button", name=label)
                if btn.count() > 0 and btn.first.is_visible():
                    btn.first.click()
                    _human_delay_sync(1500, 3000)
                    logger.info(f"Vision click_button: '{label}'")
                else:
                    # Fallback to link role (SDUI uses <a> for some buttons)
                    link = page.get_by_role("link", name=label)
                    if link.count() > 0 and link.first.is_visible():
                        link.first.click()
                        _human_delay_sync(1500, 3000)
                        logger.info(f"Vision click_button (link): '{label}'")

            _human_delay_sync(300, 700)

        except Exception as e:
            logger.warning(f"Vision action '{action}' for '{label}' failed: {e}")

    return questions_answered


def _find_input_by_label(page, label: str, tag: str = "input"):
    """Find a form element by its associated label text.

    Uses multiple strategies: Playwright get_by_label, JS label proximity,
    aria-label matching, and placeholder matching.
    """
    # Strategy 1: Playwright's built-in get_by_label
    try:
        loc = page.get_by_label(label, exact=False)
        if loc.count() > 0 and loc.first.is_visible():
            return loc.first
    except Exception:
        pass

    # Strategy 2: Playwright get_by_placeholder for short labels
    if len(label) > 3:
        try:
            loc = page.get_by_placeholder(label, exact=False)
            if loc.count() > 0 and loc.first.is_visible():
                return loc.first
        except Exception:
            pass

    # Strategy 3: JS-based proximity search (handles SDUI and custom layouts)
    label_lower = label.lower().replace("*", "").strip()
    try:
        result = page.evaluate("""(params) => {
            const labelText = params.labelText;
            const tag = params.tag;

            function esc(id) {
                return (typeof CSS !== 'undefined' && CSS.escape) ? '#' + CSS.escape(id) : '#' + id.replace(/([^\\w-])/g, '\\\\$1');
            }

            function selectorFor(inp) {
                if (inp.id) return esc(inp.id);
                if (inp.name) return inp.tagName.toLowerCase() + '[name="' + inp.name + '"]';
                const idx = Math.random().toString(36).substr(2, 8);
                inp.setAttribute('data-vt', idx);
                return '[data-vt="' + idx + '"]';
            }

            function findNearbyInput(lbl) {
                const forId = lbl.getAttribute('for');
                if (forId) {
                    const el = document.getElementById(forId);
                    if (el && el.offsetParent !== null) return selectorFor(el);
                }

                const containers = [
                    '.fb-dash-form-element',
                    '.jobs-easy-apply-form-section__grouping',
                    'div[data-test-form-element]',
                    '.artdeco-text-input',
                ];
                for (const cs of containers) {
                    const grp = lbl.closest(cs);
                    if (grp) {
                        const inp = grp.querySelector(tag + ', input, select, textarea');
                        if (inp && inp.offsetParent !== null) return selectorFor(inp);
                    }
                }

                let parent = lbl.parentElement;
                for (let i = 0; i < 5 && parent; i++) {
                    const inp = parent.querySelector(tag + ', input:not([type="hidden"]):not([type="submit"]), select, textarea');
                    if (inp && inp.offsetParent !== null && !inp.closest('button')) return selectorFor(inp);
                    parent = parent.parentElement;
                }
                return null;
            }

            // Scan all text-bearing elements
            const candidates = document.querySelectorAll('label, legend, span, p, div.t-14, div.t-bold');
            for (const lbl of candidates) {
                const txt = (lbl.textContent || '').replace(/\\s+/g, ' ').trim().toLowerCase().replace('*','').trim();
                if (txt.includes(labelText) || labelText.includes(txt.substring(0, 20))) {
                    const sel = findNearbyInput(lbl);
                    if (sel) return sel;
                }
            }

            // Also check aria-label on inputs directly
            const allInputs = document.querySelectorAll('input, select, textarea');
            for (const inp of allInputs) {
                if (inp.offsetParent === null) continue;
                const ariaLabel = (inp.getAttribute('aria-label') || '').toLowerCase();
                const placeholder = (inp.getAttribute('placeholder') || '').toLowerCase();
                if (ariaLabel.includes(labelText) || placeholder.includes(labelText)) {
                    return selectorFor(inp);
                }
            }

            return null;
        }""", {"labelText": label_lower, "tag": tag})
        if result:
            loc = page.locator(result)
            if loc.count() > 0 and loc.first.is_visible():
                return loc.first
    except Exception:
        pass

    return None


def _click_radio_by_label(page, group_label: str, value_label: str) -> bool:
    """Click a radio button identified by its group label and option value text."""
    # Strategy 1: get_by_role with name matching the value
    try:
        radio = page.get_by_label(value_label, exact=False)
        if radio.count() > 0 and radio.first.is_visible():
            radio.first.check()
            return True
    except Exception:
        pass

    # Strategy 2: find the group by label text, then the radio by value
    try:
        found = page.evaluate(f"""(groupLabel, optionLabel) => {{
            const containers = document.querySelectorAll('fieldset, .fb-dash-form-element, .jobs-easy-apply-form-section__grouping, div');
            for (const c of containers) {{
                const legendOrLabel = c.querySelector('legend, label, span');
                if (!legendOrLabel) continue;
                if (!legendOrLabel.textContent.trim().toLowerCase().includes(groupLabel.toLowerCase())) continue;
                const radios = c.querySelectorAll('input[type="radio"]');
                for (const r of radios) {{
                    const rLabel = c.querySelector('label[for="' + r.id + '"]');
                    const rText = rLabel ? rLabel.textContent.trim() : r.value;
                    if (rText.toLowerCase().includes(optionLabel.toLowerCase())) {{
                        r.click();
                        return true;
                    }}
                }}
            }}
            return false;
        }}""", group_label, value_label)
        return bool(found)
    except Exception:
        return False


def _best_option_text(target: str, options: list[str]) -> Optional[str]:
    """Find the best matching option text from a list of option labels."""
    target_lower = target.lower().strip()
    for opt in options:
        if opt.strip().lower() == target_lower:
            return opt.strip()
    for opt in options:
        if target_lower in opt.strip().lower() or opt.strip().lower() in target_lower:
            return opt.strip()
    return None


def _fill_form_fields_sync(
    page, fields: list[dict], answers: list[dict], cv_path: str
) -> dict:
    """Fill form fields on the current Easy Apply step. Returns questions->answers dict."""
    questions_answered = {}
    answer_map = {a.get("field_id", ""): a.get("value", "") for a in answers}

    def _safe_locate(page_, selector_, label_, tag_hint="input"):
        """Locate an element by selector, with fallback to get_by_label."""
        if selector_:
            try:
                loc = page_.locator(selector_)
                if loc.count() > 0 and loc.first.is_visible():
                    return loc.first
            except Exception:
                pass
        if label_:
            try:
                loc = page_.get_by_label(label_, exact=False)
                if loc.count() > 0 and loc.first.is_visible():
                    return loc.first
            except Exception:
                pass
        return None

    for field in fields:
        fid = field.get("id", "")
        label = field.get("label", "")
        ftype = field.get("type", "")
        selector = field.get("selector", "")
        value = answer_map.get(fid, "")

        if not value and ftype != "file":
            if label:
                questions_answered[label] = ""
            continue

        try:
            if ftype == "file":
                file_inputs = page.locator('input[type="file"]')
                if file_inputs.count() > 0 and cv_path and os.path.isfile(cv_path):
                    file_inputs.first.set_input_files(cv_path)
                    _human_delay_sync(1000, 2000)
                    questions_answered[label or "Resume Upload"] = os.path.basename(cv_path)
                    logger.info(f"Uploaded resume: {cv_path}")
                continue

            if ftype in ("text", "email", "tel", "number", ""):
                el = _safe_locate(page, selector, label, "input")
                if el and el.is_visible():
                    el.clear()
                    _human_delay_sync(100, 300)
                    el.type(str(value), delay=random.randint(20, 50))
                    questions_answered[label] = value

            elif ftype == "select":
                el = _safe_locate(page, selector, label, "select")
                if el and el.is_visible():
                    # Try to select by matching option text
                    options = field.get("options", [])
                    best_option = _find_best_option(value, options)
                    if best_option:
                        el.select_option(value=best_option.get("value", ""))
                    else:
                        el.select_option(label=value)
                    questions_answered[label] = value

            elif ftype == "radio":
                options = field.get("options", [])
                best = _find_best_option(value, options)
                if best:
                    radio_sel = f'input[type="radio"][value="{best["value"]}"]'
                    radio = page.locator(radio_sel)
                    if radio.count() > 0:
                        radio.first.click()
                        questions_answered[label] = best.get("text", value)

            elif ftype == "textarea":
                el = _safe_locate(page, selector, label, "textarea")
                if el and el.is_visible():
                    el.clear()
                    _human_delay_sync(100, 300)
                    el.type(str(value), delay=random.randint(15, 40))
                    questions_answered[label] = value

            _human_delay_sync(300, 700)

        except Exception as e:
            logger.warning(f"Error filling field '{label}': {e}")

    return questions_answered


def _find_best_option(value: str, options: list[dict]) -> Optional[dict]:
    """Find the closest matching option for a given value."""
    if not options or not value:
        return None
    value_lower = value.lower().strip()

    # Exact text match
    for opt in options:
        if opt.get("text", "").lower().strip() == value_lower:
            return opt
        if opt.get("value", "").lower().strip() == value_lower:
            return opt

    # Numeric match for year/month dropdowns: "2" should match "2 years"
    try:
        numeric_val = int(float(value_lower))
        for opt in options:
            opt_text = opt.get("text", "").lower().strip()
            # "2 years", "2 months", "2", etc.
            if opt_text.startswith(str(numeric_val) + " ") or opt_text == str(numeric_val):
                return opt
            if opt.get("value", "").strip() == str(numeric_val):
                return opt
    except (ValueError, TypeError):
        pass

    # Partial text match
    for opt in options:
        if value_lower in opt.get("text", "").lower():
            return opt
        if value_lower in opt.get("value", "").lower():
            return opt

    # Reverse partial: option text in our value
    for opt in options:
        opt_text = opt.get("text", "").lower().strip()
        if opt_text and opt_text in value_lower:
            return opt

    return None


# ── Easy Apply Flow ──────────────────────────────────────────────────────────

def _dismiss_overlays(page):
    """Close premium promotions, cookie banners, or any overlay that blocks interaction."""
    overlay_selectors = [
        'button[aria-label="Dismiss"]',
        'button[aria-label="Got it"]',
        '.artdeco-modal__dismiss',
        'button.artdeco-toast-item__dismiss',
        '[data-test-modal-close-btn]',
    ]
    for sel in overlay_selectors:
        try:
            loc = page.locator(sel)
            if loc.count() > 0 and loc.first.is_visible():
                loc.first.click(timeout=2000)
                _human_delay_sync(300, 600)
                logger.info(f"Dismissed overlay via: {sel}")
        except Exception:
            continue

    # Close the premium promotion card's "X" if visible
    try:
        page.evaluate("""() => {
            const btns = document.querySelectorAll('button');
            for (const b of btns) {
                const label = (b.getAttribute('aria-label') || '').toLowerCase();
                if (label.includes('dismiss') || label.includes('close')) {
                    const rect = b.getBoundingClientRect();
                    if (rect.width > 0 && rect.width < 50 && rect.top > 0 && rect.top < 600) {
                        b.click();
                    }
                }
            }
        }""")
    except Exception:
        pass


def _click_easy_apply_button(page) -> bool:
    """Find and click the Easy Apply / Apply button on a job listing page.

    LinkedIn renders apply buttons in three different ways:
      Type A: <button id="jobs-apply-button-id" class="jobs-apply-button">
      Type B: <button class="apply-button ...">Apply</button>  (public page)
      Type C: <a aria-label="Easy Apply to this job"
                 data-view-name="job-apply-button" href="...">  (SDUI flow)
    """
    _human_delay_sync(500, 1000)
    page.evaluate("window.scrollTo(0, 0)")
    _human_delay_sync(500, 1000)

    # Wait for ANY apply-related element to appear (covers all three types)
    try:
        page.wait_for_selector(
            '#jobs-apply-button-id, '
            'button.jobs-apply-button, '
            'button.apply-button, '
            '[data-view-name="job-apply-button"], '
            '[data-live-test-job-apply-button]',
            timeout=12000, state="visible",
        )
        logger.info("Apply element detected on page")
    except Exception:
        logger.info("No apply element found within 12s, trying JS scan...")

    # ── Ordered selector list covering all three LinkedIn rendering types ──
    playwright_selectors = [
        ('#jobs-apply-button-id', 'stable button ID'),
        ('button[data-live-test-job-apply-button]', 'data-live-test attr'),
        ('button.jobs-apply-button', 'jobs-apply-button class'),
        ('button[aria-label*="Easy Apply"]', 'button aria-label'),
        ('button.apply-button', 'public page apply-button'),
        ('a[data-view-name="job-apply-button"]', 'SDUI anchor apply'),
        ('a[aria-label*="Easy Apply"]', 'anchor aria-label'),
    ]

    url_before_click = page.url
    for sel, label in playwright_selectors:
        try:
            loc = page.locator(sel)
            if loc.count() > 0 and loc.first.is_visible():
                loc.first.scroll_into_view_if_needed()
                _human_delay_sync(300, 600)
                loc.first.click()
                # SDUI anchors navigate the page — give extra time
                is_anchor = sel.startswith("a[") or sel.startswith("a#")
                wait_ms = (3000, 5000) if is_anchor else (1500, 2500)
                _human_delay_sync(*wait_ms)
                logger.info(f"Clicked apply via [{label}]: {sel}")
                url_after = page.url
                if url_after != url_before_click:
                    logger.info(f"Page navigated after apply click: {url_after}")
                return True
        except Exception as e:
            logger.debug(f"[{label}] {sel} failed: {e}")

    # ── Playwright role-based locators (button + link) ──
    for role in ("button", "link"):
        for name in ("Easy Apply", "Apply"):
            try:
                loc = page.get_by_role(role, name=name)
                if loc.count() > 0 and loc.first.is_visible():
                    loc.first.scroll_into_view_if_needed()
                    _human_delay_sync(300, 600)
                    loc.first.click()
                    _human_delay_sync(1500, 2500)
                    logger.info(f"Clicked apply via get_by_role({role}, name={name!r})")
                    return True
            except Exception:
                pass

    # ── Dismiss overlays and retry with JS ──
    _dismiss_overlays(page)
    _human_delay_sync(1000, 2000)

    # Diagnostic dump
    diag = page.evaluate("""() => {
        const all = document.querySelectorAll('button, [role="button"], a');
        const info = [];
        for (const el of all) {
            const text = (el.textContent || '').replace(/\\s+/g, ' ').trim().substring(0, 80);
            const aria = el.getAttribute('aria-label') || '';
            const dvn = el.getAttribute('data-view-name') || '';
            const id = el.id || '';
            const rect = el.getBoundingClientRect();
            if (text.toLowerCase().includes('apply') || aria.toLowerCase().includes('apply') || dvn.includes('apply')) {
                info.push({
                    tag: el.tagName, id, text, aria, dvn,
                    w: Math.round(rect.width), h: Math.round(rect.height),
                    top: Math.round(rect.top), visible: rect.width > 0 && rect.height > 0
                });
            }
        }
        return info;
    }""")
    logger.info(f"Apply elements on page ({len(diag)}): {json.dumps(diag, indent=2)}")

    # JS direct click — handles all three types
    clicked = page.evaluate("""() => {
        // Priority 1: button by ID
        const byId = document.getElementById('jobs-apply-button-id');
        if (byId) {
            byId.scrollIntoView({block: 'center'});
            byId.click();
            return {ok: true, method: 'id'};
        }
        // Priority 2: SDUI anchor with data-view-name
        const sdui = document.querySelector('a[data-view-name="job-apply-button"]');
        if (sdui) {
            const r = sdui.getBoundingClientRect();
            if (r.width > 0 && r.height > 0) {
                sdui.scrollIntoView({block: 'center'});
                sdui.click();
                return {ok: true, method: 'sdui-anchor'};
            }
        }
        // Priority 3: public page apply button
        const pub = document.querySelector('button.apply-button');
        if (pub) {
            const r = pub.getBoundingClientRect();
            if (r.width > 0 && r.height > 0) {
                pub.scrollIntoView({block: 'center'});
                pub.click();
                return {ok: true, method: 'public-apply-btn'};
            }
        }
        // Priority 4: any element with Easy Apply / Apply text
        const els = document.querySelectorAll('button, [role="button"], a');
        for (const el of els) {
            const text = (el.textContent || '').replace(/\\s+/g, ' ').trim();
            if (text.includes('Easy Apply') || (el.tagName === 'BUTTON' && text === 'Apply')) {
                const r = el.getBoundingClientRect();
                if (r.width > 20 && r.height > 10 && r.top >= 0 && r.top < 1200) {
                    el.scrollIntoView({block: 'center'});
                    el.click();
                    return {ok: true, method: 'text-match', tag: el.tagName, text: text.substring(0, 60)};
                }
            }
        }
        return {ok: false};
    }""")
    if clicked and clicked.get("ok"):
        _human_delay_sync(1500, 2500)
        logger.info(f"Clicked apply via JS: {clicked}")
        return True

    # Mouse simulation as absolute last resort
    bbox = page.evaluate("""() => {
        const candidates = [
            document.getElementById('jobs-apply-button-id'),
            document.querySelector('a[data-view-name="job-apply-button"]'),
            document.querySelector('button.apply-button'),
            document.querySelector('button.jobs-apply-button'),
        ];
        for (const el of candidates) {
            if (el) {
                el.scrollIntoView({block: 'center'});
                const r = el.getBoundingClientRect();
                if (r.width > 0 && r.height > 0)
                    return {x: r.x + r.width / 2, y: r.y + r.height / 2};
            }
        }
        return null;
    }""")
    if bbox:
        page.mouse.move(bbox["x"], bbox["y"])
        _human_delay_sync(200, 400)
        page.mouse.down()
        _human_delay_sync(50, 150)
        page.mouse.up()
        _human_delay_sync(1500, 2500)
        logger.info("Clicked apply via mouse simulation")
        return True

    logger.warning(f"DOM-based apply button detection exhausted. URL: {page.url}")
    return False


def _vision_find_and_click_apply_sync(page) -> bool:
    """Use GPT-4o Vision to visually locate and click the Apply button.

    Fallback when all DOM-based selectors fail — takes a screenshot, asks
    the Vision model to identify the button, then uses multiple strategies:
    1. JS-based search prioritising actual <button> elements over decorative text
    2. Playwright role-based locators
    3. Mouse-coordinate click on the found element's bounding box
    """
    from openai import OpenAI

    logger.info("Vision AI: scanning for Apply button via screenshot...")
    screenshot_bytes = page.screenshot(full_page=False)
    b64_img = base64.b64encode(screenshot_bytes).decode()

    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)

    try:
        resp = client.chat.completions.create(
            model=settings.openai_model,
            temperature=0.1,
            max_tokens=500,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": VISION_APPLY_BUTTON_PROMPT},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{b64_img}",
                        "detail": "high",
                    }},
                ],
            }],
        )
        raw = resp.choices[0].message.content or ""
        cleaned = clean_llm_json(raw)
        info = json.loads(cleaned)
        logger.info(f"Vision AI apply button analysis: {json.dumps(info)}")

        if not info.get("found"):
            logger.warning("Vision AI: no apply button found in screenshot")
            return False

        btn_text = info.get("button_text", "").strip()
        if not btn_text:
            logger.warning("Vision AI found button but no text returned")
            return False

        # Strategy 1: JS search — prioritise <button> and <a> with apply-like
        # classes over decorative <span>/<p> labels.  On collection pages
        # "Easy Apply" appears both as a badge label AND the real button.
        bbox = page.evaluate("""(btnText) => {
            const normalised = btnText.toLowerCase().trim();

            // Pass 1: real interactive elements (button, a, [role=button])
            const interactive = document.querySelectorAll(
                'button, a, [role="button"]'
            );
            for (const el of interactive) {
                const text = (el.textContent || '').replace(/\\s+/g, ' ').trim().toLowerCase();
                if (text.includes(normalised) || normalised.includes(text)) {
                    const r = el.getBoundingClientRect();
                    // Must look like a real button (decent size, in viewport)
                    if (r.width > 40 && r.height > 20 && r.top >= 0 && r.top < 1200) {
                        el.scrollIntoView({block: 'center'});
                        const r2 = el.getBoundingClientRect();
                        return {
                            ok: true, method: 'interactive',
                            text: text.substring(0, 60),
                            x: r2.x + r2.width / 2,
                            y: r2.y + r2.height / 2,
                        };
                    }
                }
            }

            // Pass 2: any element (including spans) but require larger size
            // to skip small badge labels
            const all = document.querySelectorAll('*');
            for (const el of all) {
                const text = (el.textContent || '').replace(/\\s+/g, ' ').trim().toLowerCase();
                if (text === normalised || text === 'easy apply') {
                    const r = el.getBoundingClientRect();
                    if (r.width > 60 && r.height > 25 && r.top >= 0 && r.top < 1200) {
                        el.scrollIntoView({block: 'center'});
                        const r2 = el.getBoundingClientRect();
                        return {
                            ok: true, method: 'any-element',
                            text: text.substring(0, 60),
                            x: r2.x + r2.width / 2,
                            y: r2.y + r2.height / 2,
                        };
                    }
                }
            }

            return {ok: false};
        }""", btn_text)

        if bbox and bbox.get("ok"):
            logger.info(f"Vision AI: found element via JS ({bbox.get('method')}): '{bbox.get('text')}' at ({bbox['x']:.0f}, {bbox['y']:.0f})")
            # Use native mouse simulation for maximum reliability
            page.mouse.move(bbox["x"], bbox["y"])
            _human_delay_sync(200, 400)
            page.mouse.down()
            _human_delay_sync(50, 150)
            page.mouse.up()
            _human_delay_sync(1500, 2500)
            logger.info("Vision AI: clicked apply via mouse simulation on found element")
            return True

        # Strategy 2: Playwright role-based locator
        for role in ("button", "link"):
            try:
                loc = page.get_by_role(role, name=btn_text)
                if loc.count() > 0 and loc.first.is_visible():
                    loc.first.scroll_into_view_if_needed()
                    _human_delay_sync(300, 600)
                    bbox2 = loc.first.bounding_box()
                    if bbox2:
                        cx = bbox2["x"] + bbox2["width"] / 2
                        cy = bbox2["y"] + bbox2["height"] / 2
                        page.mouse.move(cx, cy)
                        _human_delay_sync(200, 400)
                        page.mouse.down()
                        _human_delay_sync(50, 150)
                        page.mouse.up()
                        _human_delay_sync(1500, 2500)
                        logger.info(f"Vision AI: clicked apply via role({role}) + mouse at ({cx:.0f}, {cy:.0f})")
                        return True
            except Exception:
                pass

        # Strategy 3: Search for the LinkedIn-specific Easy Apply button classes
        # and click via coordinates — handles non-standard rendering
        apply_bbox = page.evaluate("""() => {
            // LinkedIn Easy Apply button has recognisable CSS classes
            const selectors = [
                'button.jobs-apply-button',
                'button[class*="apply"]',
                'a[data-view-name="job-apply-button"]',
                'button[aria-label*="Easy Apply"]',
                'a[aria-label*="Easy Apply"]',
            ];
            for (const sel of selectors) {
                const el = document.querySelector(sel);
                if (el) {
                    el.scrollIntoView({block: 'center'});
                    const r = el.getBoundingClientRect();
                    if (r.width > 0 && r.height > 0)
                        return {x: r.x + r.width / 2, y: r.y + r.height / 2};
                }
            }
            return null;
        }""")
        if apply_bbox:
            page.mouse.move(apply_bbox["x"], apply_bbox["y"])
            _human_delay_sync(200, 400)
            page.mouse.down()
            _human_delay_sync(50, 150)
            page.mouse.up()
            _human_delay_sync(1500, 2500)
            logger.info(f"Vision AI: clicked apply via CSS class mouse sim at ({apply_bbox['x']:.0f}, {apply_bbox['y']:.0f})")
            return True

        logger.warning(f"Vision AI identified button '{btn_text}' but could not locate it in DOM")
        return False

    except Exception as e:
        logger.warning(f"Vision AI apply button detection failed: {e}")
        return False


def _wait_for_easy_apply_modal(page) -> bool:
    """Wait for the Easy Apply form — either a modal overlay or an SDUI apply page."""

    def _is_sdui_flow(url: str) -> bool:
        return "/apply/" in url or "openSDUIApplyFlow" in url

    def _check_sdui(label: str) -> bool:
        url = page.url
        if _is_sdui_flow(url):
            logger.info(f"SDUI apply flow detected ({label}): {url}")
            try:
                page.wait_for_selector(
                    'form, [role="main"] input, [role="main"] select, '
                    '.jobs-easy-apply-content, .jobs-easy-apply-modal',
                    timeout=10000,
                )
                logger.info("SDUI apply form loaded")
            except Exception:
                _human_delay_sync(2000, 3000)
            return True
        return False

    if _check_sdui("initial"):
        return True

    # Standard modal + broader SDUI selectors
    modal_selectors = [
        '.jobs-easy-apply-modal',
        '.jobs-easy-apply-content',
        '[role="dialog"][aria-label*="apply"]',
        '[role="dialog"][aria-label*="Apply"]',
        '.artdeco-modal--layer-default',
        '.artdeco-modal:has(form)',
        '[data-test-modal-id="easy-apply-modal"]',
        '.jpac-modal-content',
    ]
    combined = ", ".join(modal_selectors)

    for attempt in range(4):
        if _check_sdui(f"attempt {attempt + 1}"):
            return True

        try:
            page.wait_for_selector(combined, timeout=5000)
            logger.info("Easy Apply modal opened")
            return True
        except Exception:
            pass

        # Check via JS if any dialog/form has appeared (catches new LinkedIn UI variants)
        has_form = page.evaluate("""() => {
            const modals = document.querySelectorAll(
                '[role="dialog"], .artdeco-modal, .jobs-easy-apply-modal, .jpac-modal-content'
            );
            for (const m of modals) {
                if (m.offsetParent !== null && m.querySelector('form, input, select, textarea')) {
                    return true;
                }
            }
            // Check for full-page apply form (no modal wrapper)
            const applyForms = document.querySelectorAll(
                '.jobs-easy-apply-form-section, form[data-test-form], .jobs-apply-form'
            );
            for (const f of applyForms) {
                if (f.offsetParent !== null) return true;
            }
            return false;
        }""")
        if has_form:
            logger.info(f"Easy Apply form detected via JS scan on attempt {attempt + 1}")
            return True

        logger.info(f"Modal wait attempt {attempt + 1}/4 — not yet visible")
        _human_delay_sync(1500, 2500)

    logger.warning("Easy Apply modal/form did not open after 4 attempts")
    return False


def _click_next_or_submit(page) -> str:
    """Click Next, Review, or Submit on the Easy Apply modal or SDUI page.

    Returns: 'next', 'review', 'submit', 'done', or 'none'.
    """
    # Try to scope to modal first, fall back to full page for SDUI flow
    modal_selectors = [
        '.jobs-easy-apply-modal', '.jobs-easy-apply-content',
        '[role="dialog"]', '.artdeco-modal',
    ]
    modal = None
    for sel in modal_selectors:
        try:
            loc = page.locator(sel)
            if loc.count() > 0 and loc.first.is_visible():
                modal = loc.first
                break
        except Exception:
            continue

    scope = modal if modal else page

    # ── Ordered button detection: submit > review > next ──
    # Each entry: (selectors, role_names, return_value, log_label)
    button_groups = [
        (
            [
                'button[aria-label="Submit application"]',
                'button:has-text("Submit application")',
                'button:has-text("Submit")',
            ],
            ["Submit application", "Submit"],
            "submit",
            "Submit application",
        ),
        (
            [
                'button[aria-label="Review your application"]',
                'button:has-text("Review")',
            ],
            ["Review your application", "Review"],
            "review",
            "Review",
        ),
        (
            [
                'button[aria-label="Continue to next step"]',
                'button:has-text("Next")',
                'footer button.artdeco-button--primary',
            ],
            ["Continue to next step", "Next"],
            "next",
            "Next",
        ),
    ]

    for selectors, role_names, ret_val, label in button_groups:
        # Strategy 1: CSS selectors
        for sel in selectors:
            try:
                loc = scope.locator(sel)
                if loc.count() > 0 and loc.first.is_visible():
                    loc.first.scroll_into_view_if_needed()
                    _human_delay_sync(200, 400)
                    loc.first.click()
                    _human_delay_sync(1500, 3000)
                    logger.info(f"Clicked {label} via selector: {sel}")
                    return ret_val
            except Exception:
                continue

        # Strategy 2: Playwright role-based locator
        for rname in role_names:
            try:
                loc = scope.get_by_role("button", name=rname)
                if loc.count() > 0 and loc.first.is_visible():
                    loc.first.scroll_into_view_if_needed()
                    _human_delay_sync(200, 400)
                    loc.first.click()
                    _human_delay_sync(1500, 3000)
                    logger.info(f"Clicked {label} via role(button, name={rname!r})")
                    return ret_val
            except Exception:
                continue

    # Strategy 3: JS fallback — scan all buttons by text content
    js_result = page.evaluate("""() => {
        const priorities = [
            {text: 'submit application', ret: 'submit'},
            {text: 'submit', ret: 'submit'},
            {text: 'review', ret: 'review'},
            {text: 'next', ret: 'next'},
        ];
        const btns = document.querySelectorAll('button, [role="button"]');
        for (const p of priorities) {
            for (const btn of btns) {
                const t = (btn.textContent || '').replace(/\\s+/g, ' ').trim().toLowerCase();
                if (t.includes(p.text)) {
                    const r = btn.getBoundingClientRect();
                    if (r.width > 20 && r.height > 10 && r.top >= 0) {
                        btn.scrollIntoView({block: 'center'});
                        return {x: r.x + r.width / 2, y: r.y + r.height / 2, ret: p.ret, text: t.substring(0, 40)};
                    }
                }
            }
        }
        return null;
    }""")
    if js_result:
        page.mouse.move(js_result["x"], js_result["y"])
        _human_delay_sync(200, 400)
        page.mouse.down()
        _human_delay_sync(50, 150)
        page.mouse.up()
        _human_delay_sync(1500, 3000)
        logger.info(f"Clicked '{js_result['text']}' via JS mouse sim at ({js_result['x']:.0f}, {js_result['y']:.0f})")
        return js_result["ret"]

    return "none"


def _check_application_submitted(page) -> bool:
    """Check if the application was successfully submitted.

    Looks for LinkedIn's post-application confirmation patterns:
    modal with 'application was sent', success banners, or the
    'Done' button that only appears after successful submission.
    """
    # Strategy 1: look for known success modal / text
    success_phrases = [
        'your application was sent',
        'application submitted',
        'application has been submitted',
        'your application was submitted',
        'successfully applied',
    ]
    try:
        body_text = (page.text_content("body") or "").lower()
        for phrase in success_phrases:
            if phrase in body_text:
                logger.info(f"Submission confirmed via text: '{phrase}'")
                return True
    except Exception:
        pass

    # Strategy 2: modal with success content
    modal_selectors = [
        '.artdeco-modal:has-text("application was sent")',
        '.artdeco-modal:has-text("application submitted")',
        '[role="dialog"]:has-text("application was sent")',
        '.jobs-post-apply-modal',
    ]
    for sel in modal_selectors:
        try:
            loc = page.locator(sel)
            if loc.count() > 0 and loc.first.is_visible():
                logger.info(f"Submission confirmed via modal: {sel}")
                return True
        except Exception:
            continue

    # Strategy 3: the Easy Apply modal/form is GONE (closed after submit)
    try:
        easy_modal = page.locator('.jobs-easy-apply-modal, .jobs-easy-apply-content, [role="dialog"]:has-text("Easy Apply")')
        if easy_modal.count() == 0:
            # No Easy Apply form visible + we just clicked Submit => likely success
            # BUT make sure we're not just on a plain job page
            done_btn = page.locator('button:has-text("Done"), button:has-text("Not now")')
            if done_btn.count() > 0 and done_btn.first.is_visible():
                logger.info("Submission confirmed: modal gone + Done/Not now button visible")
                return True
    except Exception:
        pass

    # Strategy 4: JS check for post-apply elements
    try:
        js_confirmed = page.evaluate("""() => {
            // Look for any element with "application was sent"
            const allText = document.body.innerText.toLowerCase();
            if (allText.includes('your application was sent')) return true;
            if (allText.includes('application submitted')) return true;
            // Post-apply buttons
            const btns = document.querySelectorAll('button');
            for (const b of btns) {
                const t = b.textContent.trim().toLowerCase();
                if (t === 'done' && b.offsetParent !== null) return true;
            }
            return false;
        }""")
        if js_confirmed:
            logger.info("Submission confirmed via JS scan")
            return True
    except Exception:
        pass

    return False


def _dismiss_post_apply_modal(page):
    """Close any post-application modal (e.g., 'Your application was sent')."""
    dismiss_selectors = [
        'button[aria-label="Dismiss"]',
        'button:has-text("Done")',
        'button:has-text("Not now")',
        '.artdeco-modal__dismiss',
    ]
    for sel in dismiss_selectors:
        try:
            loc = page.locator(sel)
            if loc.count() > 0 and loc.first.is_visible():
                loc.first.click()
                _human_delay_sync(500, 1000)
                return
        except Exception:
            continue


def _apply_easy_sync(
    page, job: dict, profile: dict, preferences: dict, cv_path: str,
    llm_answer_fn: Callable,
    saved_qa: Optional[dict[str, str]] = None,
    modal_already_open: bool = False,
) -> dict:
    """Execute the Easy Apply flow for a single job. Returns application result.

    If modal_already_open=True, skip navigation and button clicking (used when
    redirected from the external apply flow).
    """
    job_url = job.get("url", "")
    result = {
        "job_title": job.get("title", ""),
        "company": job.get("company", ""),
        "job_url": job_url,
        "job_location": job.get("location", ""),
        "apply_type": "easy_apply",
        "status": "failed",
        "questions_answered": {},
        "error_message": "",
    }

    try:
        if not modal_already_open:
            # Navigate to job page
            logger.info(f"Navigating to job: {job.get('title', '')} at {job.get('company', '')}")
            try:
                page.goto(job_url, wait_until="networkidle", timeout=20000)
            except Exception:
                page.goto(job_url, wait_until="domcontentloaded", timeout=15000)
            _human_delay_sync(2000, 4000)

            # Ensure we're on the canonical /jobs/view/<id>/ page
            _ensure_canonical_job_view(page, job_url)

            # Click Easy Apply — DOM selectors first, then Vision AI fallback
            apply_clicked = _click_easy_apply_button(page)

            if not apply_clicked:
                logger.info("DOM selectors failed — trying Vision AI to find Apply button")
                apply_clicked = _vision_find_and_click_apply_sync(page)

            if not apply_clicked:
                result["status"] = "skipped"
                result["error_message"] = "Easy Apply button not found (DOM + Vision AI)"
                logger.warning(
                    f"SKIPPED {job.get('title','')}: Apply button not found "
                    f"(DOM + Vision AI). URL: {page.url}"
                )
                return result

            # Wait for modal
            if not _wait_for_easy_apply_modal(page):
                result["status"] = "skipped"
                result["error_message"] = "Easy Apply modal did not open"
                logger.warning(f"SKIPPED {job.get('title','')}: Easy Apply modal did not open. URL: {job_url}")
                return result
        else:
            logger.info("Modal already open — skipping navigation and button click")

        # Multi-step form loop (max 10 steps as safety valve)
        all_questions = {}
        vision_calls = 0
        prev_field_sig = ""  # Signature of previous step's fields for stuck detection
        stuck_count = 0

        for step in range(10):
            logger.info(f"Processing Easy Apply step {step + 1}...")
            _human_delay_sync(1000, 2000)

            # Read form fields via DOM parsing
            fields = _read_form_fields(page)

            # ── Stuck detection: if the same fields appear 2+ times, we're not advancing ──
            field_sig = "|".join(sorted(f.get("label", "") for f in fields)) if fields else ""
            if field_sig and field_sig == prev_field_sig:
                stuck_count += 1
                if stuck_count >= 2:
                    logger.warning(
                        f"Stuck on same fields for {stuck_count + 1} steps: [{field_sig}]. "
                        f"Falling back to Vision AI."
                    )
                    fields = []  # Force Vision AI path
            else:
                stuck_count = 0
            prev_field_sig = field_sig

            if fields:
                # ── Primary path: DOM-based fill ──
                # Priority: 1) deterministic rules  2) saved Q&A memory  3) LLM
                unfilled = []
                qa_memory = saved_qa or {}

                for field in fields:
                    det_value = _fill_field_deterministic(field, profile, preferences)
                    if det_value == "__FILE_UPLOAD__":
                        file_inputs = page.locator('input[type="file"]')
                        if file_inputs.count() > 0 and cv_path and os.path.isfile(cv_path):
                            file_inputs.first.set_input_files(cv_path)
                            _human_delay_sync(1000, 2000)
                            all_questions["Resume Upload"] = os.path.basename(cv_path)
                    elif det_value:
                        field["_answer"] = det_value
                        logger.info(f"Deterministic: '{field.get('label','')}' = '{det_value}'")
                    else:
                        # Check saved Q&A memory
                        label = field.get("label", "")
                        normalised = _normalise_question(label)
                        mem_answer = qa_memory.get(normalised)
                        if mem_answer:
                            field["_answer"] = mem_answer
                            logger.info(f"Q&A Memory: '{label}' = '{mem_answer}'")
                        else:
                            unfilled.append(field)
                            logger.info(f"No match (rules/memory) for: '{label}' -> LLM")

                det_answers = [
                    {"field_id": f["id"], "value": f["_answer"]}
                    for f in fields if "_answer" in f
                ]
                if det_answers:
                    q = _fill_form_fields_sync(page, [f for f in fields if "_answer" in f], det_answers, cv_path)
                    all_questions.update(q)

                if unfilled:
                    logger.info(f"Using LLM for {len(unfilled)} unfilled fields...")
                    llm_answers = llm_answer_fn(unfilled, profile, preferences)
                    q = _fill_form_fields_sync(page, unfilled, llm_answers, cv_path)
                    all_questions.update(q)
            else:
                # ── Fallback: Vision AI when DOM parsing finds nothing ──
                if vision_calls < MAX_VISION_CALLS_PER_JOB:
                    logger.info(
                        f"No DOM fields found on step {step + 1}, "
                        f"activating Vision AI (call {vision_calls + 1}/{MAX_VISION_CALLS_PER_JOB})..."
                    )
                    vision_actions = _vision_analyze_form_sync(
                        page, profile, preferences, cv_path,
                        saved_qa=saved_qa,
                    )
                    if vision_actions:
                        q = _execute_vision_actions_sync(page, vision_actions, cv_path)
                        all_questions.update(q)
                    vision_calls += 1
                else:
                    logger.warning(
                        f"No DOM fields and vision call limit reached ({MAX_VISION_CALLS_PER_JOB}), "
                        f"skipping step {step + 1}"
                    )

            _human_delay_sync(500, 1000)

            # Click Next/Review/Submit (skip if vision already clicked a button)
            action = _click_next_or_submit(page)

            if action == "submit":
                _human_delay_sync(3000, 5000)
                if _check_application_submitted(page):
                    result["status"] = "applied"
                    result["questions_answered"] = all_questions
                    logger.info(f"Successfully applied to: {job.get('title', '')}")
                    _dismiss_post_apply_modal(page)
                    return result
                else:
                    logger.warning(
                        f"Clicked Submit but confirmation not detected for: {job.get('title', '')}"
                    )
                    result["status"] = "failed"
                    result["error_message"] = "Clicked Submit but no confirmation detected"
                    result["questions_answered"] = all_questions
                    _dismiss_post_apply_modal(page)
                    return result

            elif action == "review":
                _human_delay_sync(1000, 2000)
                submit_action = _click_next_or_submit(page)
                if submit_action == "submit":
                    _human_delay_sync(3000, 5000)
                    if _check_application_submitted(page):
                        result["status"] = "applied"
                        result["questions_answered"] = all_questions
                        logger.info(f"Successfully applied to: {job.get('title', '')}")
                        _dismiss_post_apply_modal(page)
                        return result
                    else:
                        logger.warning(
                            f"Clicked Submit after Review but confirmation not detected: {job.get('title', '')}"
                        )
                        result["status"] = "failed"
                        result["error_message"] = "Submit after Review — no confirmation detected"
                        result["questions_answered"] = all_questions
                        _dismiss_post_apply_modal(page)
                        return result
                elif submit_action == "review":
                    # Stuck in review loop — try once more
                    _human_delay_sync(1000, 2000)
                    last_try = _click_next_or_submit(page)
                    if last_try == "submit":
                        _human_delay_sync(3000, 5000)
                        if _check_application_submitted(page):
                            result["status"] = "applied"
                            result["questions_answered"] = all_questions
                            _dismiss_post_apply_modal(page)
                            return result
                    result["status"] = "failed"
                    result["error_message"] = "Stuck in review loop — could not submit"
                    result["questions_answered"] = all_questions
                    return result
                else:
                    result["status"] = "failed"
                    result["error_message"] = "Review page opened but no Submit button found"
                    result["questions_answered"] = all_questions
                    return result

            elif action == "none":
                logger.warning(f"No Next/Submit button found on step {step + 1}")
                result["error_message"] = f"Stuck on step {step + 1} — no navigation button"
                result["questions_answered"] = all_questions
                break

        # If we exhausted steps — always save whatever Q&A we collected
        if result["status"] != "applied":
            result["status"] = "failed"
            result["error_message"] = result.get("error_message", "") or "Exceeded maximum form steps without confirmation"
        result["questions_answered"] = all_questions

    except Exception as e:
        result["error_message"] = str(e)
        logger.error(f"Easy Apply error for {job_url}: {e}")

    return result


# ── External Apply Flow ──────────────────────────────────────────────────────

def _apply_external_sync(
    page, job: dict, profile: dict, preferences: dict, cv_path: str,
    llm_form_fn: Callable,
    llm_answer_fn: Optional[Callable] = None,
    saved_qa: Optional[dict[str, str]] = None,
) -> dict:
    """Attempt to apply on an external ATS site. Returns application result."""
    result = {
        "job_title": job.get("title", ""),
        "company": job.get("company", ""),
        "job_url": job.get("url", ""),
        "job_location": job.get("location", ""),
        "apply_type": "external",
        "status": "failed",
        "questions_answered": {},
        "error_message": "",
    }

    try:
        page.goto(job["url"], wait_until="domcontentloaded")
        _human_delay_sync(2000, 4000)

        # If LinkedIn redirected us to a collection/similar-jobs page,
        # navigate to the canonical /jobs/view/<id>/ URL first.
        if _ensure_canonical_job_view(page, job.get("url", "")):
            logger.info("Redirected to canonical job view from external flow")

        # Find the Apply button (not Easy Apply)
        apply_clicked = False
        apply_selectors = [
            'button.jobs-apply-button',
            'button:has-text("Apply")',
            'a:has-text("Apply")',
            'a.jobs-apply-button',
        ]
        for sel in apply_selectors:
            try:
                loc = page.locator(sel)
                if loc.count() > 0 and loc.first.is_visible():
                    # Check if it opens a new tab
                    with page.context.expect_page(timeout=5000) as new_page_info:
                        loc.first.click()
                    new_page = new_page_info.value
                    new_page.wait_for_load_state("domcontentloaded", timeout=15000)
                    _human_delay_sync(2000, 4000)
                    apply_clicked = True

                    # Now on external site — try to fill the form
                    form_html = new_page.evaluate("""() => {
                        const forms = document.querySelectorAll('form');
                        if (forms.length === 0) return document.body.innerText.substring(0, 3000);
                        let html = '';
                        for (const form of forms) {
                            html += form.outerHTML.substring(0, 2000) + '\\n';
                        }
                        return html.substring(0, 5000);
                    }""")

                    if not form_html or len(form_html) < 50:
                        result["status"] = "skipped"
                        result["error_message"] = "External page has no recognizable form"
                        new_page.close()
                        return result

                    # Use LLM to analyze and fill the form
                    actions = llm_form_fn(form_html, profile, preferences)

                    for action in actions:
                        act_type = action.get("action", "")
                        selector = action.get("selector", "")
                        value = action.get("value", "")

                        try:
                            if act_type == "fill":
                                el = new_page.locator(selector).first
                                if el.is_visible():
                                    el.clear()
                                    el.type(value, delay=random.randint(20, 50))
                                    result["questions_answered"][selector] = value
                            elif act_type == "select":
                                el = new_page.locator(selector).first
                                if el.is_visible():
                                    el.select_option(label=value)
                                    result["questions_answered"][selector] = value
                            elif act_type == "click":
                                el = new_page.locator(selector).first
                                if el.is_visible():
                                    el.click()
                            elif act_type == "upload" and cv_path:
                                el = new_page.locator('input[type="file"]').first
                                if el:
                                    el.set_input_files(cv_path)
                                    result["questions_answered"]["Resume"] = os.path.basename(cv_path)
                            _human_delay_sync(300, 700)
                        except Exception as e:
                            logger.warning(f"External form action failed: {e}")

                    result["status"] = "applied"
                    result["error_message"] = "External form filled (manual verification recommended)"
                    new_page.close()
                    return result
            except Exception:
                continue

        if not apply_clicked:
            # Check if this "external" job actually has an Easy Apply button
            logger.info("External selectors failed — checking for Easy Apply on this page")
            found_easy = False
            if _click_easy_apply_button(page):
                logger.info("Found Easy Apply on 'external' job — switching to Easy Apply flow")
                if _wait_for_easy_apply_modal(page):
                    found_easy = True

            if not found_easy:
                logger.info("Trying Vision AI to find apply button on external job page")
                if _vision_find_and_click_apply_sync(page):
                    _human_delay_sync(2000, 4000)
                    if _wait_for_easy_apply_modal(page):
                        found_easy = True

            if found_easy and llm_answer_fn:
                logger.info("Running full Easy Apply form-fill on auto-detected Easy Apply")
                easy_result = _apply_easy_sync(
                    page, job, profile, preferences, cv_path,
                    llm_answer_fn, saved_qa=saved_qa,
                    modal_already_open=True,
                )
                easy_result["apply_type"] = "easy_apply"
                easy_result["error_message"] = (
                    easy_result.get("error_message", "") or
                    "Redirected from external to Easy Apply"
                )
                return easy_result
            elif found_easy:
                result["apply_type"] = "easy_apply"
                result["status"] = "failed"
                result["error_message"] = "Easy Apply modal opened but no LLM answer function available"
                return result

            result["status"] = "skipped"
            result["error_message"] = "Could not find or click external Apply button (DOM + Vision AI)"

    except Exception as e:
        result["status"] = "skipped"
        result["error_message"] = f"External apply error: {str(e)}"
        logger.warning(f"External apply error: {e}")

    return result


# ── Main Orchestrator (sync, runs in thread) ─────────────────────────────────

def _apply_to_jobs_sync(
    email: str,
    password: str,
    totp_secret: Optional[str],
    profile: dict,
    preferences: dict,
    linkedin_jobs: list[dict],
    cv_path: str,
    max_apps: int,
    session_id: str,
    update_callback: Callable,
) -> dict:
    """Main sync function that applies to jobs. Runs inside a thread executor."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return {"success": False, "error": "Playwright is not installed"}

    summary = {
        "success": True,
        "applied": 0,
        "skipped": 0,
        "failed": 0,
        "results": [],
    }

    def _llm_answer_sync(fields, prof, prefs):
        return _answer_with_llm_sync(fields, prof, prefs)

    def _llm_form_sync(form_html, prof, prefs):
        return _analyze_external_form_sync(form_html, prof, prefs)

    # Load saved Q&A memory for reuse across all jobs in this session
    saved_qa = _load_saved_qa_sync()

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=False,
                args=["--disable-blink-features=AutomationControlled"],
            )
            context, page, success, msg = _ensure_logged_in(
                browser, email, password, totp_secret
            )
            if not success:
                _safe_close(browser)
                return {"success": False, "error": f"Login failed: {msg}"}

            applied_total = 0
            for idx, job in enumerate(linkedin_jobs):
                if applied_total >= max_apps:
                    logger.info(f"Reached max applications ({max_apps})")
                    break

                # Break after every N applications
                if applied_total > 0 and applied_total % BREAK_EVERY_N == 0:
                    logger.info(f"Taking a break after {applied_total} applications...")
                    _human_delay_sync(BREAK_DURATION_MIN, BREAK_DURATION_MAX)
                    _save_session(context)

                job_url = job.get("url", "")
                is_easy = job.get("is_easy_apply", False)

                # Any LinkedIn job URL should use the Easy Apply flow.
                # SerpAPI may use subdomains like in.linkedin.com, uk.linkedin.com
                if not is_easy and "linkedin.com" in job_url and "/jobs/" in job_url:
                    is_easy = True
                    logger.info(
                        f"Auto-promoting external job to Easy Apply "
                        f"(LinkedIn URL detected: {job_url[:100]})"
                    )

                logger.info(
                    f"[{idx + 1}/{len(linkedin_jobs)}] Applying to: "
                    f"{job.get('title', '')} @ {job.get('company', '')} "
                    f"({'Easy Apply' if is_easy else 'External'})"
                )

                if is_easy:
                    result = _apply_easy_sync(
                        page, job, profile, preferences, cv_path,
                        _llm_answer_sync, saved_qa=saved_qa,
                    )
                else:
                    result = _apply_external_sync(
                        page, job, profile, preferences, cv_path, _llm_form_sync,
                        llm_answer_fn=_llm_answer_sync, saved_qa=saved_qa,
                    )

                summary["results"].append(result)

                # Save Q&A pairs to memory for future runs
                qa = result.get("questions_answered", {})
                if qa:
                    _save_qa_pairs_sync(qa)
                    # Also update in-memory cache for remaining jobs
                    for q_text, a_text in qa.items():
                        norm = _normalise_question(q_text)
                        if norm and a_text:
                            saved_qa[norm] = str(a_text)

                if result["status"] == "applied":
                    summary["applied"] += 1
                    applied_total += 1
                elif result["status"] == "skipped":
                    summary["skipped"] += 1
                else:
                    summary["failed"] += 1

                # Update DB via callback
                try:
                    update_callback(session_id, result, summary)
                except Exception as e:
                    logger.warning(f"Status update callback failed: {e}")

                # Human-like delay between applications
                if idx < len(linkedin_jobs) - 1 and applied_total < max_apps:
                    _human_delay_sync(DELAY_BETWEEN_APPS_MIN, DELAY_BETWEEN_APPS_MAX)

            _save_session(context)
            _safe_close(browser)

    except Exception as e:
        summary["success"] = False
        summary["error"] = str(e)
        logger.error(f"Job apply session error: {e}")

    return summary


def _analyze_external_form_sync(
    form_html: str, profile: dict, preferences: dict
) -> list[dict]:
    """Use LLM (sync) to analyze an external job application form."""
    from openai import OpenAI
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)
    prompt = EXTERNAL_FORM_PROMPT.format(
        profile=json.dumps(profile, indent=2),
        preferences=json.dumps(preferences, indent=2),
        form_html=form_html[:5000],
    )
    try:
        resp = client.chat.completions.create(
            model=settings.openai_model,
            temperature=0.2,
            messages=[{"role": "system", "content": prompt}],
        )
        cleaned = clean_llm_json(resp.choices[0].message.content or "")
        return json.loads(cleaned)
    except Exception as e:
        logger.warning(f"LLM external form parse error: {e}")
        return []


# ── Public Async API ─────────────────────────────────────────────────────────

async def search_and_apply_to_jobs(
    email: str,
    password: str,
    totp_secret: Optional[str],
    profile: dict,
    preferences: dict,
    criteria: dict,
    cv_path: str,
    max_apps: int,
    session_id: str,
    update_callback: Callable,
) -> dict:
    """Search for jobs and apply — runs Playwright in a thread executor.

    This is the main entry point called from the route handler.
    """
    # First: external search (async)
    logger.info("Starting external job search (Tavily + SerpAPI)...")
    external_jobs = await search_jobs_external(
        criteria.get("keywords", ""),
        criteria.get("location", ""),
    )

    # Combine: the sync function will also search LinkedIn via browser
    def _run_sync():
        from playwright.sync_api import sync_playwright

        all_jobs = list(external_jobs)

        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=False,
                args=["--disable-blink-features=AutomationControlled"],
            )
            context, page, success, msg = _ensure_logged_in(
                browser, email, password, totp_secret
            )
            if not success:
                _safe_close(browser)
                return {"success": False, "error": f"Login failed: {msg}"}

            # Search on LinkedIn browser
            logger.info("Searching jobs on LinkedIn browser...")
            linkedin_jobs = _search_jobs_linkedin(
                page,
                criteria.get("keywords", ""),
                criteria.get("location", ""),
                criteria.get("experience_level", ""),
                criteria.get("date_posted", ""),
                criteria.get("easy_apply_only", True),
            )

            # Merge and deduplicate
            seen_urls: set[str] = set()
            merged: list[dict] = []
            for job in linkedin_jobs:
                url = job.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    merged.append(job)
            for job in all_jobs:
                url = job.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    merged.append(job)

            logger.info(f"Total unique jobs found: {len(merged)} (LinkedIn: {len(linkedin_jobs)}, External: {len(external_jobs)})")

            # Update session with job count
            try:
                update_callback(session_id, None, {"total_jobs": len(merged), "status": "applying"})
            except Exception:
                pass

            def _llm_answer_sync(fields, prof, prefs):
                return _answer_with_llm_sync(fields, prof, prefs)

            def _llm_form_sync(form_html, prof, prefs):
                return _analyze_external_form_sync(form_html, prof, prefs)

            saved_qa = _load_saved_qa_sync()

            # Enrich preferences with search location as fallback
            search_loc = criteria.get("location", "")
            if search_loc:
                if not preferences.get("preferred_locations"):
                    preferences["preferred_locations"] = [search_loc]
                elif search_loc not in preferences["preferred_locations"]:
                    preferences["preferred_locations"].append(search_loc)
            # Also push into profile if profile has no location
            if search_loc and not profile.get("location"):
                profile["location"] = search_loc

            # Apply loop
            summary = {"success": True, "applied": 0, "skipped": 0, "failed": 0, "results": []}
            applied_total = 0

            for idx, job in enumerate(merged):
                if applied_total >= max_apps:
                    logger.info(f"Reached max applications ({max_apps})")
                    break

                if applied_total > 0 and applied_total % BREAK_EVERY_N == 0:
                    logger.info(f"Taking a break after {applied_total} applications...")
                    _human_delay_sync(BREAK_DURATION_MIN, BREAK_DURATION_MAX)
                    _save_session(context)

                is_easy = job.get("is_easy_apply", False)
                logger.info(
                    f"[{idx + 1}/{min(len(merged), max_apps)}] "
                    f"{job.get('title', '')} @ {job.get('company', '')} "
                    f"({'Easy Apply' if is_easy else 'External'})"
                )

                if is_easy:
                    app_result = _apply_easy_sync(
                        page, job, profile, preferences, cv_path, _llm_answer_sync,
                        saved_qa=saved_qa,
                    )
                else:
                    app_result = _apply_external_sync(
                        page, job, profile, preferences, cv_path, _llm_form_sync,
                        llm_answer_fn=_llm_answer_sync, saved_qa=saved_qa,
                    )

                summary["results"].append(app_result)

                status = app_result["status"]
                err = app_result.get("error_message", "")
                logger.info(
                    f"  -> Result: {status.upper()}"
                    + (f" ({err})" if err else "")
                    + f" | Running totals: applied={summary['applied']}, "
                    f"skipped={summary['skipped']}, failed={summary['failed']}"
                )

                if status == "applied":
                    summary["applied"] += 1
                    applied_total += 1
                elif status == "skipped":
                    summary["skipped"] += 1
                else:
                    summary["failed"] += 1

                try:
                    update_callback(session_id, app_result, summary)
                except Exception as e:
                    logger.warning(f"Callback error: {e}")

                if idx < len(merged) - 1 and applied_total < max_apps:
                    _human_delay_sync(DELAY_BETWEEN_APPS_MIN, DELAY_BETWEEN_APPS_MAX)

            _save_session(context)
            _safe_close(browser)
            return summary

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _run_sync)
