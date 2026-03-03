"""Shared email extraction and validation utilities.

Consolidates duplicated email regex/blacklist code from
hr_agent.py and scrape_server.py into a single source of truth.
"""

from __future__ import annotations

import re

# ── Email Regex ──────────────────────────────────────────────────────────────

EMAIL_REGEX = re.compile(
    r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    re.IGNORECASE,
)

# ── Blacklisted Domains ─────────────────────────────────────────────────────

BLACKLIST_DOMAINS = frozenset({
    "example.com",
    "test.com",
    "email.com",
    "domain.com",
    "sentry.io",
    "wixpress.com",
    "w3.org",
    "schema.org",
    "googleapis.com",
    "googletagmanager.com",
    "facebook.com",
    "twitter.com",
})

# ── File-extension false positives ───────────────────────────────────────────

_FALSE_EXTENSIONS = (".png", ".jpg", ".gif", ".svg", ".webp", ".css", ".js")


def extract_emails(text: str) -> list[str]:
    """Extract and deduplicate email addresses from text.

    Filters out:
      - Known blacklisted domains (test/example/tracking)
      - File-extension false positives (image@foo.png)
      - Duplicates (case-insensitive)

    Args:
        text: Raw text or HTML content.

    Returns:
        Deduplicated list of lowercase email addresses.
    """
    raw = EMAIL_REGEX.findall(text)
    seen: set[str] = set()
    cleaned: list[str] = []

    for email in raw:
        lower = email.lower()
        if lower in seen:
            continue

        domain = lower.split("@")[1]
        if domain in BLACKLIST_DOMAINS:
            continue
        if lower.endswith(_FALSE_EXTENSIONS):
            continue

        seen.add(lower)
        cleaned.append(lower)

    return cleaned


def is_valid_email_format(email: str) -> bool:
    """Check if an email has a valid structural format.

    Validates:
      - Matches the email regex
      - No consecutive dots in local part
      - Domain has at least one dot
      - TLD is 2-10 characters
    """
    if not EMAIL_REGEX.fullmatch(email):
        return False

    local, domain = email.rsplit("@", 1)

    # No consecutive dots
    if ".." in local or ".." in domain:
        return False

    # Domain must have at least one dot
    if "." not in domain:
        return False

    # TLD must be 2-10 chars
    tld = domain.rsplit(".", 1)[1]
    if len(tld) < 2 or len(tld) > 10:
        return False

    return True
