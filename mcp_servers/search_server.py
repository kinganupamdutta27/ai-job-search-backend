"""MCP Server — Web Search Tool (SerpAPI)."""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent


# ── SerpAPI Search Implementation ────────────────────────────────────────────

SERP_API_URL = "https://serpapi.com/search.json"


async def _search_serpapi(query: str, num_results: int = 10) -> list[dict[str, str]]:
    """Execute a web search via SerpAPI and return structured results."""
    api_key = os.environ.get("SERP_API_KEY", "")
    if not api_key:
        raise ValueError("SERP_API_KEY environment variable not set")

    params = {
        "q": query,
        "api_key": api_key,
        "num": num_results,
        "engine": "google",
    }

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(SERP_API_URL, params=params)
        response.raise_for_status()
        data = response.json()

    results: list[dict[str, str]] = []
    for item in data.get("organic_results", [])[:num_results]:
        results.append(
            {
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", ""),
            }
        )
    return results


# ── MCP Server Definition ───────────────────────────────────────────────────

app = Server("search-server")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """Expose the web_search tool."""
    return [
        Tool(
            name="web_search",
            description=(
                "Search the web using Google via SerpAPI. "
                "Returns a list of search results with title, URL, and snippet. "
                "Use site: operators to search specific job portals "
                "(e.g., 'site:naukri.com python developer jobs')."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query string",
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return (default 10, max 20)",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""
    if name == "web_search":
        query = arguments["query"]
        num_results = min(arguments.get("num_results", 10), 20)
        try:
            results = await _search_serpapi(query, num_results)
            return [TextContent(type="text", text=json.dumps(results, indent=2))]
        except Exception as e:
            return [
                TextContent(type="text", text=json.dumps({"error": str(e)}))
            ]
    else:
        return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]


# ── Entry Point ──────────────────────────────────────────────────────────────

async def main():
    """Run the MCP search server via stdio transport."""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
