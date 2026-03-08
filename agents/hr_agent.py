"""LangGraph Node — HR / Contact Extraction Agent.

For each job listing, fetches the job page and attempts to extract
HR/recruiter contact information. All discovered contacts are
verified via the email verification pipeline and persisted to
the SQLite database grouped by company domain.
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

from database import async_session_maker
from graph.state import GraphState
from graph.models import JobListing, HRContact
from models.db_models import CompanyEntity, HRContactEntity
from services.email_utils import extract_emails, _is_placeholder_email, BLACKLIST_DOMAINS
from services.email_verifier import verify_email, VerificationStatus
from services.llm_utils import clean_llm_json, create_llm
from services.retry import retry

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

CONTACT_EXTRACTION_PROMPT = """You are an expert at extracting HR/recruiter contact information from job posting content.

STRICT RULES:
- ONLY return email addresses that are EXPLICITLY VISIBLE in the page content below.
- Do NOT invent, guess, or hallucinate email addresses.
- Do NOT return placeholder/demo emails like user@example.com, name@domain.com, test@company.com.
- Do NOT return emails belonging to job portals (naukri, indeed, linkedin, glassdoor, etc.).
- If a real company domain is clearly identified in the content AND no explicit emails are found, you may suggest ONLY these pattern emails: hr@<domain>, careers@<domain>, recruitment@<domain>.
- If the company domain is unclear or the URL is a job portal, return an empty array.

Return a JSON array of contacts. Each contact:
{{
  "name": "Contact name (or 'Hiring Manager' if unknown)",
  "email": "actual-email@real-domain.com",
  "role": "HR Manager / Talent Acquisition / Recruiter",
  "source": "extracted" if found on page, "ai_inferred" if pattern-based
}}

Company domain from URL: {domain}

If you cannot find any real email, return an empty array [].
Return ONLY the JSON array, no extra text."""


# ── Page Fetching ────────────────────────────────────────────────────────────

@retry(max_attempts=2, backoff_factor=1.5)
async def _fetch_page_content(url: str) -> tuple[str, list[str]]:
    """Fetch a web page and extract text content + emails found."""
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml",
    }

    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            html = response.text

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        if len(text) > 4000:
            text = text[:4000] + "\n...[truncated]"

        # Use shared email extractor (handles dedup + blacklist)
        emails = extract_emails(html)

        return text, emails

    except Exception as e:
        logger.warning(f"⚠️ Failed to fetch {url}: {e}")
        return "", []


def _infer_company_domain(url: str) -> str:
    """Extract the likely company domain from a job listing URL."""
    parsed = urlparse(url)
    host = parsed.netloc.lower()

    portal_domains = [
        "naukri.com", "indeed.com", "indeed.co.in", "linkedin.com",
        "glassdoor.com", "glassdoor.co.in", "foundit.in", "instahyre.com",
    ]
    for portal in portal_domains:
        if portal in host:
            return ""

    if host.startswith("www."):
        host = host[4:]

    return host


def _domain_from_email(email: str) -> str:
    """Extract domain from an email address."""
    if "@" in email:
        return email.split("@")[1].lower()
    return ""


_PATTERN_PREFIXES = {"hr", "careers", "recruitment", "jobs", "hiring", "talent"}


def _validate_ai_email(email: str, page_text: str, company_domain: str) -> bool:
    """Validate an AI-returned email is plausible, not hallucinated."""
    lower = email.lower()
    domain = lower.split("@")[1] if "@" in lower else ""

    if domain in BLACKLIST_DOMAINS:
        return False
    if _is_placeholder_email(lower):
        return False

    local = lower.split("@")[0]
    if local in _PATTERN_PREFIXES and company_domain and domain == company_domain:
        return True

    if lower in page_text.lower():
        return True

    if domain and company_domain and domain != company_domain:
        return False

    return False


def _generate_pattern_emails(company_domain: str) -> list[HRContact]:
    """Generate common HR email patterns for a company domain."""
    if not company_domain:
        return []

    patterns = [
        ("hr", "HR Department"),
        ("careers", "Careers Team"),
        ("recruitment", "Recruitment Team"),
        ("jobs", "Jobs Inbox"),
        ("hiring", "Hiring Team"),
        ("talent", "Talent Acquisition"),
    ]
    contacts = []
    for prefix, role_name in patterns[:3]:  # Top 3 patterns
        contacts.append(
            HRContact(
                name=role_name,
                email=f"{prefix}@{company_domain}",
                source="pattern",
            )
        )
    return contacts


# ── Database Persistence ─────────────────────────────────────────────────────

async def _persist_contacts_to_db(
    contacts: list[HRContact],
    company_name: str,
) -> int:
    """
    Persist discovered HR contacts to the database.

    Contacts are deduplicated by email. Companies are
    deduplicated by domain. Returns count of new contacts saved.
    """
    saved = 0
    async with async_session_maker() as db:
        try:
            for contact in contacts:
                domain = _domain_from_email(contact.email)
                if not domain:
                    continue

                # Upsert company by domain
                result = await db.execute(
                    select(CompanyEntity).where(CompanyEntity.domain == domain)
                )
                company = result.scalar_one_or_none()
                if not company:
                    company = CompanyEntity(
                        name=company_name or domain,
                        domain=domain,
                    )
                    db.add(company)
                    await db.flush()

                # Upsert contact by email (skip if exists)
                existing = await db.execute(
                    select(HRContactEntity).where(
                        HRContactEntity.email == contact.email
                    )
                )
                if existing.scalar_one_or_none() is None:
                    db_contact = HRContactEntity(
                        company_id=company.id,
                        name=contact.name,
                        email=contact.email,
                        role=getattr(contact, "role", "HR"),
                        source=contact.source,
                    )
                    db.add(db_contact)
                    saved += 1

            await db.commit()
        except Exception as e:
            await db.rollback()
            logger.warning(f"⚠️ Failed to persist contacts: {e}")

    return saved


# ── Main Agent Node ──────────────────────────────────────────────────────────

async def extract_contacts(state: GraphState) -> dict:
    """
    LangGraph node: Extract HR contacts from job listing pages.

    For each job listing:
    1. Fetches the page and scrapes emails
    2. Uses AI (OpenAI) to infer contacts from page content
    3. Falls back to pattern-based email generation
    4. Persists ALL discovered contacts to the SQLite database

    Reads: state["job_listings"]
    Writes: state["enriched_listings"], state["current_step"]
    """
    logger.info("📧 Starting HR contact extraction...")

    listings_data = state.get("job_listings", [])
    if not listings_data:
        return {
            "errors": ["No job listings to extract contacts from"],
            "current_step": "failed",
        }

    llm = create_llm(temperature=0.1)

    enriched: list[dict] = []
    total_contacts_saved = 0

    for i, listing_data in enumerate(listings_data):
        listing = JobListing(**listing_data)
        logger.info(
            f"  📧 [{i + 1}/{len(listings_data)}] Extracting contacts for: "
            f"{listing.title} @ {listing.company}"
        )

        contacts: list[HRContact] = []
        page_text = ""
        page_emails: list[str] = []

        # Step 1: Fetch the page and look for emails
        if listing.url:
            page_text, page_emails = await _fetch_page_content(listing.url)

            if page_emails:
                for email in page_emails[:5]:
                    contacts.append(
                        HRContact(name="Hiring Manager", email=email, source="extracted")
                    )

        # Step 2: Use AI to extract contacts from page content
        if page_text and not contacts:
            domain = _infer_company_domain(listing.url)
            try:
                messages = [
                    SystemMessage(
                        content=CONTACT_EXTRACTION_PROMPT.format(domain=domain or "unknown")
                    ),
                    HumanMessage(content=f"Job posting content:\n\n{page_text}"),
                ]
                response = await llm.ainvoke(messages)
                content = clean_llm_json(response.content)

                ai_contacts = json.loads(content)
                for c in ai_contacts:
                    candidate = HRContact(**c)
                    if _validate_ai_email(candidate.email, page_text, domain):
                        contacts.append(candidate)
                    else:
                        logger.info(
                            f"  🚫 Discarded AI-hallucinated email: {candidate.email}"
                        )

            except Exception as e:
                logger.warning(f"  ⚠️ AI extraction failed: {e}")

        # Step 3: Fall back to pattern-based emails
        if not contacts:
            domain = _infer_company_domain(listing.url)
            contacts = _generate_pattern_emails(domain)

        # Step 4: Verify emails before persisting
        verified_contacts: list[HRContact] = []
        for contact in contacts:
            vr = await verify_email(contact.email, source=contact.source)
            if vr.overall_status == VerificationStatus.INVALID:
                logger.info(f"  🚫 Skipping invalid email: {contact.email} ({vr.detail})")
                continue
            verified_contacts.append(contact)

        # Step 5: Persist verified contacts to database
        if verified_contacts:
            saved = await _persist_contacts_to_db(verified_contacts, listing.company)
            total_contacts_saved += saved
            if saved:
                logger.info(f"  💾 Saved {saved} new contacts to DB for {listing.company}")

        # Update listing with verified contacts
        listing.hr_contacts = verified_contacts
        enriched.append(listing.model_dump())

        # Rate limit
        await asyncio.sleep(0.3)

    contacts_found = sum(len(l.get("hr_contacts", [])) for l in enriched)
    logger.info(
        f"✅ Contact extraction complete: {contacts_found} contacts "
        f"across {len(enriched)} listings | {total_contacts_saved} new contacts saved to DB"
    )

    return {
        "enriched_listings": enriched,
        "current_step": "contacts_extracted",
    }
