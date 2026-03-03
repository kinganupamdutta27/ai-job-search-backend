"""Email verification pipeline — format, disposable, MX, and optional SMTP checks.

Provides a 4-stage verification pipeline:
  1. Format validation (regex + structural checks)
  2. Disposable domain detection (~100 common providers)
  3. MX record lookup (DNS-based domain validation)
  4. Optional SMTP probe (RCPT TO verification)

Usage:
    result = await verify_email("hr@example.com")
    if result.overall_status == "valid": ...
"""

from __future__ import annotations

import asyncio
import logging
import re
import socket
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from services.email_utils import is_valid_email_format

logger = logging.getLogger(__name__)


# ── Verification Result ──────────────────────────────────────────────────────


class VerificationStatus(str, Enum):
    VALID = "valid"
    RISKY = "risky"
    INVALID = "invalid"


class VerificationResult(BaseModel):
    """Result of the email verification pipeline."""

    email: str = Field(description="The email address that was verified")
    format_valid: bool = Field(default=False, description="Passed format validation")
    is_disposable: bool = Field(default=False, description="Uses a disposable email domain")
    mx_exists: bool = Field(default=False, description="Domain has valid MX records")
    smtp_verified: Optional[bool] = Field(
        default=None, description="SMTP RCPT TO check result (None = skipped)"
    )
    overall_status: VerificationStatus = Field(default=VerificationStatus.INVALID)
    detail: str = Field(default="", description="Human-readable verification summary")


# ── Disposable Email Domains ─────────────────────────────────────────────────

DISPOSABLE_DOMAINS = frozenset({
    "10minutemail.com", "guerrillamail.com", "guerrillamail.info",
    "mailinator.com", "tempmail.com", "throwaway.email",
    "yopmail.com", "sharklasers.com", "guerrillamailblock.com",
    "grr.la", "dispostable.com", "trashmail.com",
    "trashmail.me", "mailnesia.com", "maildrop.cc",
    "mailcatch.com", "tempail.com", "tempr.email",
    "temp-mail.org", "fakeinbox.com", "safetymail.info",
    "getairmail.com", "mailexpire.com", "discard.email",
    "mohmal.com", "getnada.com", "anonbox.net",
    "jetable.org", "mytemp.email", "trash-mail.com",
    "tmpmail.net", "tmpmail.org", "binkmail.com",
    "mailscrap.com", "tempinbox.com", "spamgourmet.com",
    "mintemail.com", "emailondeck.com", "tmail.ws",
    "33mail.com", "spamfree24.org", "incognitomail.org",
    "temp-mail.io", "tempmailo.com", "inboxbear.com",
    "crazymailing.com", "harakirimail.com", "mailsac.com",
    "burnermail.io", "guerrillamail.de", "guerrillamail.net",
    "guerrillamail.org", "guerrillamail.biz",
    # Common Indian disposable providers
    "mailinator.in", "yopmail.fr", "yopmail.net",
})


# ── Stage 1: Format Check ───────────────────────────────────────────────────


def check_format(email: str) -> bool:
    """Validate email format using structural checks."""
    return is_valid_email_format(email)


# ── Stage 2: Disposable Check ────────────────────────────────────────────────


def check_disposable(email: str) -> bool:
    """Check if the email uses a known disposable email domain.

    Returns True if the domain is disposable (bad).
    """
    domain = email.rsplit("@", 1)[1].lower()
    return domain in DISPOSABLE_DOMAINS


# ── Stage 3: MX Record Lookup ────────────────────────────────────────────────


async def check_mx_record(email: str) -> bool:
    """Check if the email domain has valid MX records.

    Uses dnspython if available, falls back to socket-based lookup.
    """
    domain = email.rsplit("@", 1)[1].lower()

    # Try dnspython first (more accurate)
    try:
        import dns.resolver

        loop = asyncio.get_event_loop()
        answers = await loop.run_in_executor(
            None, lambda: dns.resolver.resolve(domain, "MX")
        )
        return len(answers) > 0
    except ImportError:
        # Fallback to socket-based DNS lookup
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: socket.getaddrinfo(domain, 25)
            )
            return len(result) > 0
        except (socket.gaierror, socket.herror, OSError):
            return False
    except Exception:
        return False


# ── Stage 4: SMTP Probe (Optional) ──────────────────────────────────────────


async def check_smtp(email: str, timeout: int = 10) -> Optional[bool]:
    """Probe the SMTP server to check if the email address is deliverable.

    Performs a HELO → MAIL FROM → RCPT TO handshake without actually
    sending an email.

    Returns:
        True if SMTP accepts the recipient
        False if SMTP rejects the recipient
        None if the check could not be performed
    """
    domain = email.rsplit("@", 1)[1].lower()

    try:
        import dns.resolver

        loop = asyncio.get_event_loop()
        mx_records = await loop.run_in_executor(
            None, lambda: dns.resolver.resolve(domain, "MX")
        )
        mx_host = str(sorted(mx_records, key=lambda r: r.preference)[0].exchange)

    except Exception:
        return None

    try:
        import smtplib

        def _smtp_check():
            try:
                server = smtplib.SMTP(mx_host, 25, timeout=timeout)
                server.ehlo("verify.local")
                server.mail("verify@verify.local")
                code, _ = server.rcpt(email)
                server.quit()
                return code == 250
            except Exception:
                return None

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _smtp_check)

    except Exception:
        return None


# ── Full Pipeline ────────────────────────────────────────────────────────────


async def verify_email(
    email: str,
    check_smtp_enabled: bool = False,
) -> VerificationResult:
    """Run the full email verification pipeline.

    Args:
        email:               Email address to verify.
        check_smtp_enabled:  If True, also perform SMTP RCPT TO check.

    Returns:
        VerificationResult with all stage results and overall status.
    """
    result = VerificationResult(email=email)

    # Stage 1: Format
    result.format_valid = check_format(email)
    if not result.format_valid:
        result.overall_status = VerificationStatus.INVALID
        result.detail = "Invalid email format"
        return result

    # Stage 2: Disposable
    result.is_disposable = check_disposable(email)
    if result.is_disposable:
        result.overall_status = VerificationStatus.INVALID
        result.detail = "Disposable email domain detected"
        return result

    # Stage 3: MX Record
    result.mx_exists = await check_mx_record(email)
    if not result.mx_exists:
        result.overall_status = VerificationStatus.INVALID
        result.detail = "Domain has no MX records (cannot receive email)"
        return result

    # Stage 4: SMTP Probe (optional)
    if check_smtp_enabled:
        result.smtp_verified = await check_smtp(email)
        if result.smtp_verified is False:
            result.overall_status = VerificationStatus.RISKY
            result.detail = "SMTP server rejected the recipient address"
            return result

    # All checks passed
    result.overall_status = VerificationStatus.VALID
    result.detail = "All verification checks passed"
    return result
