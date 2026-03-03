"""MCP Server — Email Sending Tool (SMTP)."""

from __future__ import annotations

import json
import os
import uuid
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

import aiosmtplib
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent


# ── Email Send Implementation ────────────────────────────────────────────────


async def _send_email(
    to: str,
    subject: str,
    body_html: str,
    body_text: str,
    attachment_path: str | None = None,
) -> dict[str, Any]:
    """Send an email via SMTP with optional attachment."""
    smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    smtp_email = os.environ.get("SMTP_EMAIL", "")
    smtp_password = os.environ.get("SMTP_PASSWORD", "")

    if not smtp_email or not smtp_password:
        raise ValueError("SMTP_EMAIL and SMTP_PASSWORD must be configured")

    # Build the message
    msg = MIMEMultipart("alternative")
    msg["From"] = smtp_email
    msg["To"] = to
    msg["Subject"] = subject
    msg["Message-ID"] = f"<{uuid.uuid4()}@jobsearch-automation>"

    # Attach plain text and HTML versions
    msg.attach(MIMEText(body_text, "plain", "utf-8"))
    msg.attach(MIMEText(body_html, "html", "utf-8"))

    # Attach file if provided
    if attachment_path and os.path.exists(attachment_path):
        with open(attachment_path, "rb") as f:
            attachment = MIMEApplication(f.read())
            filename = os.path.basename(attachment_path)
            attachment.add_header(
                "Content-Disposition", "attachment", filename=filename
            )
            msg.attach(attachment)

    # Send
    try:
        await aiosmtplib.send(
            msg,
            hostname=smtp_host,
            port=smtp_port,
            username=smtp_email,
            password=smtp_password,
            start_tls=True,
        )
        return {
            "success": True,
            "message_id": msg["Message-ID"],
            "to": to,
            "error": "",
        }
    except Exception as e:
        return {
            "success": False,
            "message_id": "",
            "to": to,
            "error": str(e),
        }


# ── MCP Server Definition ───────────────────────────────────────────────────

app = Server("email-server")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """Expose the send_email tool."""
    return [
        Tool(
            name="send_email",
            description=(
                "Send an email via SMTP. Supports HTML and plain text bodies "
                "with an optional file attachment (e.g., a CV). "
                "Requires SMTP_HOST, SMTP_PORT, SMTP_EMAIL, and SMTP_PASSWORD "
                "environment variables to be configured."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "Recipient email address",
                    },
                    "subject": {
                        "type": "string",
                        "description": "Email subject line",
                    },
                    "body_html": {
                        "type": "string",
                        "description": "HTML email body",
                    },
                    "body_text": {
                        "type": "string",
                        "description": "Plain text email body (fallback)",
                    },
                    "attachment_path": {
                        "type": "string",
                        "description": "Optional file path for attachment (e.g., CV)",
                    },
                },
                "required": ["to", "subject", "body_html", "body_text"],
            },
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""
    if name == "send_email":
        try:
            result = await _send_email(
                to=arguments["to"],
                subject=arguments["subject"],
                body_html=arguments["body_html"],
                body_text=arguments["body_text"],
                attachment_path=arguments.get("attachment_path"),
            )
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [
                TextContent(type="text", text=json.dumps({"error": str(e), "success": False}))
            ]
    else:
        return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]


# ── Entry Point ──────────────────────────────────────────────────────────────

async def main():
    """Run the MCP email server via stdio transport."""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
