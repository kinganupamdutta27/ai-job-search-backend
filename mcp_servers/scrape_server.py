"""MCP Server — Web Scrape / Content Extraction Tool."""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent


# ── Web Scraping Implementation ──────────────────────────────────────────────

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]

EMAIL_REGEX = re.compile(
    r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    re.IGNORECASE,
)


def _extract_emails(text: str) -> list[str]:
    """Extract email addresses from text content."""
    found = EMAIL_REGEX.findall(text)
    # Filter out common false positives
    blacklist_domains = {"example.com", "test.com", "email.com", "domain.com"}
    cleaned = []
    for email in found:
        domain = email.split("@")[1].lower()
        if domain not in blacklist_domains and not email.endswith(".png") and not email.endswith(".jpg"):
            cleaned.append(email.lower())
    return list(set(cleaned))


async def _fetch_page(url: str) -> dict[str, Any]:
    """Fetch a web page and extract its text content and metadata."""
    import random

    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        html = response.text

    soup = BeautifulSoup(html, "html.parser")

    # Remove script and style elements
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    # Extract title
    title = soup.title.string.strip() if soup.title and soup.title.string else ""

    # Extract main text content
    text_content = soup.get_text(separator="\n", strip=True)

    # Limit text length to avoid token overflow
    if len(text_content) > 5000:
        text_content = text_content[:5000] + "\n... [truncated]"

    # Extract emails from the full text
    emails_found = _extract_emails(html)

    # Extract useful links
    links = []
    for a_tag in soup.find_all("a", href=True)[:20]:
        href = a_tag["href"]
        link_text = a_tag.get_text(strip=True)
        if href.startswith("mailto:"):
            email = href.replace("mailto:", "").split("?")[0]
            if email not in emails_found:
                emails_found.append(email)
        elif href.startswith("http"):
            links.append({"text": link_text[:100], "url": href})

    return {
        "title": title,
        "text_content": text_content,
        "emails_found": emails_found,
        "links": links,
        "source_url": url,
        "domain": urlparse(url).netloc,
    }


# ── MCP Server Definition ───────────────────────────────────────────────────

app = Server("scrape-server")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """Expose the fetch_page_content tool."""
    return [
        Tool(
            name="fetch_page_content",
            description=(
                "Fetch a web page and extract its text content, title, "
                "email addresses found on the page, and links. "
                "Useful for extracting HR contact information from job posting pages."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL of the page to fetch",
                    },
                },
                "required": ["url"],
            },
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""
    if name == "fetch_page_content":
        url = arguments["url"]
        try:
            result = await _fetch_page(url)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [
                TextContent(
                    type="text",
                    text=json.dumps({"error": str(e), "url": url}),
                )
            ]
    else:
        return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]


# ── Entry Point ──────────────────────────────────────────────────────────────

async def main():
    """Run the MCP scrape server via stdio transport."""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
