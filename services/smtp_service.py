"""Centralized SMTP service for sending emails.

Eliminates duplication between workflow.py and email_server.py.
Handles connection, auth, message construction, attachments, and retry.
"""

from __future__ import annotations

import logging
import os
import uuid
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

import aiosmtplib

from config import get_settings
from services.exceptions import SMTPError, SMTPNotConfigured
from services.retry import retry

logger = logging.getLogger(__name__)


@retry(max_attempts=3, backoff_factor=2)
async def send_email(
    to_email: str,
    subject: str,
    body_html: str,
    body_text: str,
    attachment_path: Optional[str] = None,
) -> tuple[bool, str]:
    """Send an email via configured SMTP.

    Args:
        to_email:        Recipient email address.
        subject:         Email subject.
        body_html:       HTML email body.
        body_text:       Plain text email body.
        attachment_path: Optional file path to attach (e.g., CV pdf).

    Returns:
        Tuple of (success: bool, message: str)
        where message is the message_id on success or error string on failure.

    Raises:
        SMTPNotConfigured: If SMTP credentials are missing from config.
        SMTPError:         On persistent send failure.
    """
    settings = get_settings()

    if not settings.smtp_email or not settings.smtp_password:
        raise SMTPNotConfigured()

    msg = MIMEMultipart("alternative")
    msg["From"] = settings.smtp_email
    msg["To"] = to_email
    msg["Subject"] = subject
    
    # Needs a stable message ID for tracking
    msg_id = f"<{uuid.uuid4()}@jobsearch-automation>"
    msg["Message-ID"] = msg_id

    # Attach bodies
    msg.attach(MIMEText(body_text, "plain", "utf-8"))
    msg.attach(MIMEText(body_html, "html", "utf-8"))

    # Attach file if available
    if attachment_path and os.path.exists(attachment_path):
        with open(attachment_path, "rb") as f:
            attachment = MIMEApplication(f.read())
            filename = os.path.basename(attachment_path)
            attachment.add_header(
                "Content-Disposition", "attachment", filename=filename
            )
            msg.attach(attachment)

    try:
        await aiosmtplib.send(
            msg,
            hostname=settings.smtp_host,
            port=settings.smtp_port,
            username=settings.smtp_email,
            password=settings.smtp_password,
            start_tls=True,
        )
        return True, msg_id

    except Exception as e:
        logger.error(f"Failed to send email to {to_email}: {e}")
        raise SMTPError(detail=str(e)) from e
