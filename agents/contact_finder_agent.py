"""Standalone Contact Finder Agent.

Accepts a natural-language prompt (e.g. "Find HR emails of IT companies in Kolkata"),
searches the web via Tavily + SerpAPI, scrapes company pages, extracts and classifies
contacts by role, verifies them, and persists to the database.
"""

from __future__ import annotations

import asyncio
import json
import logging
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from langchain_core.messages import SystemMessage, HumanMessage
from sqlalchemy import select

from config import get_settings
from database import async_session_maker
from models.db_models import CompanyEntity, HRContactEntity, ContactFinderRunEntity
from services.email_utils import extract_emails
from services.email_verifier import verify_email, VerificationStatus
from services.llm_utils import clean_llm_json, create_llm
from services.retry import retry

logger = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# ── Prompt Templates ─────────────────────────────────────────────────────────

PARSE_PROMPT = """You are an expert at understanding user search queries for finding business contacts.

Parse the following user request into a structured JSON object:
{{
  "target_roles": ["HR Manager", "VP", "CTO", ...],
  "industries": ["IT", "Tech", "Software", ...],
  "locations": ["Kolkata", ...],
  "company_names": []
}}

Extract roles, industries, locations, and specific company names if mentioned.
If the user mentions general categories like "IT companies", set industries accordingly.
Return ONLY the JSON object, no extra text.

User request: {prompt}"""

CLASSIFY_CONTACTS_PROMPT = """You are an expert at classifying business contacts by their role.

Given the following page content from a company website and a list of email addresses found on the page,
classify each email by the likely role of the person/department.

Page content (truncated):
{page_text}

Emails found: {emails}

Return a JSON array where each item is:
{{
  "name": "Person name or department name",
  "email": "the@email.com",
  "role": "HR Manager / VP / CTO / Recruiter / Talent Acquisition / General"
}}

Only include emails that appear to belong to real people or departments at this company.
Do NOT include generic portal emails, placeholder emails, or tracking pixels.
Return ONLY the JSON array."""

SEARCH_QUERIES_PROMPT = """Generate {count} specific Google search queries to find HR/recruiter
contact emails for companies matching this criteria:

Industries: {industries}
Locations: {locations}
Target roles: {roles}
Specific companies: {companies}

Generate diverse queries that will surface company career pages, team pages, and contact directories.
Examples of good queries:
- "IT companies Kolkata HR email contact"
- "software companies Kolkata careers page"
- "tech startups Kolkata VP engineering email"
- "<company name> HR recruiter email"

Return ONLY a JSON array of query strings, nothing else."""


# ── Web Fetching ─────────────────────────────────────────────────────────────

@retry(max_attempts=2, backoff_factor=1.5)
async def _fetch_page(url: str) -> tuple[str, list[str]]:
    """Fetch a page and return (text_content, extracted_emails)."""
    headers = {"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"}
    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            html = resp.text

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        if len(text) > 4000:
            text = text[:4000] + "\n...[truncated]"

        emails = extract_emails(html)
        return text, emails
    except Exception as e:
        logger.warning(f"Failed to fetch {url}: {e}")
        return "", []


def _extract_domain(url: str) -> str:
    """Get a clean domain from a URL, skipping known portal domains."""
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    skip = {
        "google.com", "google.co.in", "bing.com",
        "naukri.com", "indeed.com", "indeed.co.in",
        "linkedin.com", "glassdoor.com", "glassdoor.co.in",
        "foundit.in", "instahyre.com", "monster.com",
    }
    for s in skip:
        if s in host:
            return ""
    if host.startswith("www."):
        host = host[4:]
    return host


# ── Search via Tavily ────────────────────────────────────────────────────────

@retry(max_attempts=2, backoff_factor=1.5)
async def _tavily_search(query: str, max_results: int = 5) -> list[dict]:
    settings = get_settings()
    if not settings.tavily_api_key:
        return []
    try:
        import os
        from langchain_tavily import TavilySearch

        os.environ["TAVILY_API_KEY"] = settings.tavily_api_key
        tool = TavilySearch(max_results=max_results, search_depth="advanced")
        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(None, tool.invoke, query)
        results = []
        if isinstance(raw, list):
            for item in raw[:max_results]:
                if isinstance(item, dict):
                    results.append({
                        "title": item.get("title", ""),
                        "url": item.get("url", item.get("link", "")),
                        "snippet": item.get("content", item.get("snippet", "")),
                    })
        return results
    except Exception as e:
        logger.warning(f"Tavily search failed: {e}")
        return []


# ── Search via SerpAPI ───────────────────────────────────────────────────────

@retry(max_attempts=2, backoff_factor=1.5)
async def _serpapi_search(query: str, num: int = 5) -> list[dict]:
    settings = get_settings()
    if not settings.serp_api_key:
        return []
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                "https://serpapi.com/search.json",
                params={"q": query, "api_key": settings.serp_api_key, "num": num, "engine": "google"},
            )
            resp.raise_for_status()
            data = resp.json()
        return [
            {"title": r.get("title", ""), "url": r.get("link", ""), "snippet": r.get("snippet", "")}
            for r in data.get("organic_results", [])[:num]
        ]
    except Exception as e:
        logger.warning(f"SerpAPI search failed: {e}")
        return []


# ── Database Persistence ─────────────────────────────────────────────────────

async def _persist_contact(
    email: str, name: str, role: str, source: str,
    company_name: str, domain: str,
) -> bool:
    """Save a single contact to the DB. Returns True if new."""
    async with async_session_maker() as db:
        try:
            result = await db.execute(
                select(CompanyEntity).where(CompanyEntity.domain == domain)
            )
            company = result.scalar_one_or_none()
            if not company:
                company = CompanyEntity(name=company_name or domain, domain=domain)
                db.add(company)
                await db.flush()

            existing = await db.execute(
                select(HRContactEntity).where(HRContactEntity.email == email)
            )
            if existing.scalar_one_or_none() is not None:
                return False

            db.add(HRContactEntity(
                company_id=company.id, name=name, email=email,
                role=role, source=source,
            ))
            await db.commit()
            return True
        except Exception as e:
            await db.rollback()
            logger.warning(f"Failed to persist contact {email}: {e}")
            return False


# ── Main Entry Point ─────────────────────────────────────────────────────────

async def find_contacts(run_id: str, prompt: str, max_companies: int = 20) -> dict:
    """Run a standalone contact discovery search.

    Returns a dict with keys: contacts, companies_found, contacts_found, status
    """
    llm = create_llm(temperature=0.1)

    # Step 1: Parse user prompt into structured query
    logger.info(f"[ContactFinder {run_id}] Parsing prompt...")
    try:
        resp = await llm.ainvoke([
            SystemMessage(content=PARSE_PROMPT.format(prompt=prompt)),
        ])
        parsed = json.loads(clean_llm_json(resp.content))
    except Exception as e:
        logger.error(f"Failed to parse prompt: {e}")
        parsed = {"target_roles": ["HR"], "industries": [], "locations": [], "company_names": []}

    roles = parsed.get("target_roles", ["HR"])
    industries = parsed.get("industries", [])
    locations = parsed.get("locations", [])
    companies = parsed.get("company_names", [])

    logger.info(
        f"[ContactFinder {run_id}] Parsed — roles={roles}, "
        f"industries={industries}, locations={locations}, companies={companies}"
    )

    # Step 2: Generate search queries via LLM
    try:
        resp = await llm.ainvoke([
            SystemMessage(content=SEARCH_QUERIES_PROMPT.format(
                count=min(max_companies, 10),
                industries=", ".join(industries) or "general",
                locations=", ".join(locations) or "any",
                roles=", ".join(roles),
                companies=", ".join(companies) or "none specified",
            )),
        ])
        queries = json.loads(clean_llm_json(resp.content))
        if not isinstance(queries, list):
            queries = []
    except Exception as e:
        logger.warning(f"Failed to generate queries: {e}")
        queries = [f"{' '.join(industries)} companies {' '.join(locations)} HR email contact"]

    logger.info(f"[ContactFinder {run_id}] Generated {len(queries)} search queries")

    # Step 3: Run all queries in parallel via Tavily + SerpAPI
    tasks = []
    for q in queries:
        tasks.append(_tavily_search(q, max_results=5))
        tasks.append(_serpapi_search(q, num=5))

    all_results = await asyncio.gather(*tasks, return_exceptions=True)
    search_hits: list[dict] = []
    for result in all_results:
        if isinstance(result, list):
            search_hits.extend(result)

    # Deduplicate by URL
    seen_urls: set[str] = set()
    unique_hits: list[dict] = []
    for hit in search_hits:
        url = hit.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_hits.append(hit)

    logger.info(f"[ContactFinder {run_id}] {len(unique_hits)} unique pages to process")

    # Step 4: Scrape pages and extract contacts
    all_contacts: list[dict] = []
    companies_seen: set[str] = set()
    processed = 0

    for hit in unique_hits[:max_companies * 2]:
        url = hit.get("url", "")
        domain = _extract_domain(url)
        if not domain or domain in companies_seen:
            continue

        page_text, page_emails = await _fetch_page(url)
        if not page_emails and not page_text:
            continue

        companies_seen.add(domain)
        company_name = hit.get("title", domain).split(" - ")[0].split(" | ")[0].strip()[:200]

        # Classify contacts via LLM if we have emails
        classified: list[dict] = []
        if page_emails:
            try:
                resp = await llm.ainvoke([
                    SystemMessage(content=CLASSIFY_CONTACTS_PROMPT.format(
                        page_text=page_text[:2000],
                        emails=", ".join(page_emails[:10]),
                    )),
                ])
                classified = json.loads(clean_llm_json(resp.content))
                if not isinstance(classified, list):
                    classified = []
            except Exception:
                classified = [
                    {"name": "Contact", "email": e, "role": "General"}
                    for e in page_emails[:5]
                ]

        # Verify and persist each contact
        for contact in classified:
            email = contact.get("email", "").lower()
            if not email:
                continue
            vr = await verify_email(email, source="extracted")
            if vr.overall_status == VerificationStatus.INVALID:
                continue

            saved = await _persist_contact(
                email=email,
                name=contact.get("name", "Contact"),
                role=contact.get("role", "General"),
                source="contact_finder",
                company_name=company_name,
                domain=domain,
            )
            all_contacts.append({
                "company": company_name,
                "domain": domain,
                "name": contact.get("name", "Contact"),
                "email": email,
                "role": contact.get("role", "General"),
                "source": "contact_finder",
                "is_new": saved,
            })

        processed += 1
        if processed >= max_companies:
            break
        await asyncio.sleep(0.3)

    # Step 5: Update the run record
    summary = {
        "contacts": all_contacts,
        "companies_found": len(companies_seen),
        "contacts_found": len(all_contacts),
        "queries_used": queries,
        "parsed_query": parsed,
    }

    async with async_session_maker() as db:
        try:
            result = await db.execute(
                select(ContactFinderRunEntity).where(ContactFinderRunEntity.id == run_id)
            )
            run = result.scalar_one_or_none()
            if run:
                run.status = "completed"
                run.results_json = summary
                run.contacts_found = len(all_contacts)
                run.companies_found = len(companies_seen)
                await db.commit()
        except Exception as e:
            logger.error(f"Failed to update run {run_id}: {e}")

    logger.info(
        f"[ContactFinder {run_id}] Done — {len(all_contacts)} contacts "
        f"from {len(companies_seen)} companies"
    )
    return summary
