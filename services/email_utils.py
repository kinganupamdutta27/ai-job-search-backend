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
    # Generic placeholder / test domains
    "example.com", "example.org", "example.net",
    "test.com", "test.org", "test.net",
    "email.com", "domain.com", "company.com",
    "yourcompany.com", "yourdomain.com", "yoursite.com",
    "placeholder.com", "sample.com", "demo.com",
    "abc.com", "xyz.com", "foo.com", "bar.com",
    "mailinator.com", "tempmail.com", "throwaway.email",
    "noreply.com", "no-reply.com",
    # Tracking / infrastructure
    "sentry.io", "wixpress.com",
    "w3.org", "schema.org",
    "googleapis.com", "googletagmanager.com",
    "google.com", "gstatic.com",
    "cloudflare.com", "cloudfront.net",
    "amazonaws.com", "azurewebsites.net",
    # Social / not recruiters
    "facebook.com", "twitter.com", "instagram.com",
    "youtube.com", "tiktok.com", "pinterest.com",
    # Job portals (their own support/system addresses are not HR contacts)
    "naukri.com", "indeed.com", "indeed.co.in",
    "linkedin.com", "glassdoor.com", "glassdoor.co.in",
    "foundit.in", "instahyre.com", "monster.com",
    "monster.co.in", "shine.com", "timesjobs.com",
    "hirist.com", "iimjobs.com", "updazz.com",
    # Common false-positive domains from page templates
    "wordpress.com", "wordpress.org", "wpengine.com",
    "squarespace.com", "wix.com", "shopify.com",
    "hubspot.com", "mailchimp.com", "sendgrid.net",
    "intercom.io", "zendesk.com", "freshdesk.com",
})

# ── Placeholder local-part patterns ─────────────────────────────────────────

_PLACEHOLDER_LOCAL_PARTS = frozenset({
    "user", "username", "name", "yourname", "your.name",
    "email", "youremail", "your.email", "your-email",
    "someone", "somebody", "anyone",
    "example", "test", "testing", "sample", "demo",
    "placeholder", "dummy", "fake", "temp",
    "abc", "xyz", "xxx", "aaa", "bbb",
    "john.doe", "jane.doe", "john", "jane",
    "firstname.lastname", "first.last",
    "noreply", "no-reply", "no.reply",
    "donotreply", "do-not-reply", "do.not.reply",
    "mailer-daemon", "postmaster", "webmaster",
})

_PLACEHOLDER_PATTERNS = re.compile(
    r"^(user\d+|test\d+|example\d+|sample\d+|demo\d+|temp\d+|dummy\d+)$",
    re.IGNORECASE,
)

# ── File-extension false positives ───────────────────────────────────────────

_FALSE_EXTENSIONS = (".png", ".jpg", ".gif", ".svg", ".webp", ".css", ".js")


def _is_placeholder_email(email: str) -> bool:
    """Detect common placeholder/template email addresses."""
    local = email.split("@")[0].lower()
    if local in _PLACEHOLDER_LOCAL_PARTS:
        return True
    if _PLACEHOLDER_PATTERNS.match(local):
        return True
    return False


def extract_emails(text: str) -> list[str]:
    """Extract and deduplicate email addresses from text.

    Filters out:
      - Known blacklisted domains (test/example/tracking/portals)
      - Placeholder local parts (user@, test@, name@, etc.)
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
        if _is_placeholder_email(lower):
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
