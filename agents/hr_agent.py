"""LangGraph Node — HR / Contact Extraction Agent.

For each job listing, fetches the job page and attempts to extract
HR/recruiter contact information (name and email).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from config import get_settings
from graph.state import GraphState
from graph.models import JobListing, HRContact

logger = logging.getLogger(__name__)

# ── User-Agent for page fetching ─────────────────────────────────────────────

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

EMAIL_REGEX = re.compile(
    r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    re.IGNORECASE,
)

CONTACT_EXTRACTION_PROMPT = """You are an expert at extracting HR/recruiter contact information from job posting content.

Analyze the following job posting page content and extract any HR, recruiter, or hiring manager contact information.

Return a JSON array of contacts found. Each contact should have:
{
  "name": "Contact name (or 'Hiring Manager' if unknown)",
  "email": "contact@example.com",
  "source": "extracted"
}

If no direct contacts are found, try to infer likely HR email patterns based on the company domain.
Common patterns: hr@domain.com, careers@domain.com, recruitment@domain.com, jobs@domain.com

Company domain from URL: {domain}

If you cannot find or infer any email, return an empty array [].
Return ONLY the JSON array, no extra text."""


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

        # Extract emails directly from HTML
        emails = list(set(EMAIL_REGEX.findall(html)))
        # Filter noise
        blacklist = {"example.com", "test.com", "email.com", "domain.com", "sentry.io"}
        emails = [
            e.lower()
            for e in emails
            if e.split("@")[1].lower() not in blacklist
            and not e.endswith((".png", ".jpg", ".gif"))
        ]

        return text, emails

    except Exception as e:
        logger.warning(f"⚠️ Failed to fetch {url}: {e}")
        return "", []


def _infer_company_domain(url: str) -> str:
    """Extract the likely company domain from a job listing URL."""
    parsed = urlparse(url)
    host = parsed.netloc.lower()

    # Strip portal domains
    portal_domains = ["naukri.com", "indeed.com", "linkedin.com", "glassdoor.com"]
    for portal in portal_domains:
        if portal in host:
            return ""  # Can't infer company domain from portal URL

    # Remove www prefix
    if host.startswith("www."):
        host = host[4:]

    return host


def _generate_pattern_emails(company_domain: str) -> list[HRContact]:
    """Generate common HR email patterns for a company domain."""
    if not company_domain:
        return []

    patterns = ["hr", "careers", "recruitment", "jobs", "hiring", "talent"]
    contacts = []
    for prefix in patterns[:3]:  # Top 3 patterns
        contacts.append(
            HRContact(
                name="HR Department",
                email=f"{prefix}@{company_domain}",
                source="pattern",
            )
        )
    return contacts


async def extract_contacts(state: GraphState) -> dict:
    """
    LangGraph node: Extract HR contacts from job listing pages.

    Reads: state["job_listings"]
    Writes: state["enriched_listings"], state["current_step"]
    """
    logger.info("📧 Starting HR contact extraction...")
    settings = get_settings()

    listings_data = state.get("job_listings", [])
    if not listings_data:
        return {
            "errors": ["No job listings to extract contacts from"],
            "current_step": "failed",
        }

    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0.1,
    )

    enriched: list[dict] = []

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
                for email in page_emails[:3]:
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
                content = response.content.strip()

                if content.startswith("```"):
                    content = content.split("\n", 1)[1]
                    if content.endswith("```"):
                        content = content[:-3]
                    content = content.strip()

                ai_contacts = json.loads(content)
                for c in ai_contacts:
                    contacts.append(HRContact(**c))

            except Exception as e:
                logger.warning(f"  ⚠️ AI extraction failed: {e}")

        # Step 3: Fall back to pattern-based emails
        if not contacts:
            domain = _infer_company_domain(listing.url)
            contacts = _generate_pattern_emails(domain)

        # Update listing with contacts
        listing.hr_contacts = contacts
        enriched.append(listing.model_dump())

        # Rate limit
        await asyncio.sleep(0.3)

    contacts_found = sum(len(l.get("hr_contacts", [])) for l in enriched)
    logger.info(
        f"✅ Contact extraction complete: {contacts_found} contacts "
        f"across {len(enriched)} listings"
    )

    return {
        "enriched_listings": enriched,
        "current_step": "contacts_extracted",
    }
