"""LangGraph Node — Job Search Agent.

Searches for relevant job openings across multiple portals using BOTH:
  1. Tavily Search API (via langchain-tavily) — AI-optimized, deep search
  2. SerpAPI — Google search with site: operators

Results from both are merged and deduplicated for maximum coverage.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

import httpx

from config import get_settings
from graph.state import GraphState
from graph.models import CVProfile, JobListing

logger = logging.getLogger(__name__)

# ── Search Query Templates ───────────────────────────────────────────────────

PORTAL_QUERIES = [
    'site:naukri.com {role} jobs {location}',
    'site:indeed.com {role} jobs {location}',
    'site:linkedin.com/jobs {role} openings {location}',
    '{role} job openings {location} hiring 2026',
]

TAVILY_QUERIES = [
    '{role} job openings {location} hiring',
    '{role} careers {location} naukri indeed linkedin',
    '{role} developer jobs {location} apply',
]

SERP_API_URL = "https://serpapi.com/search.json"


# ── Tavily Search (Native LangChain Integration) ────────────────────────────

async def _search_tavily(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """
    Execute a web search via Tavily using the official langchain-tavily package.

    Tavily provides AI-optimized search results that are more relevant
    and structured compared to traditional search engines.
    """
    settings = get_settings()
    api_key = settings.tavily_api_key

    if not api_key:
        logger.debug("Tavily API key not set, skipping Tavily search")
        return []

    try:
        from langchain_tavily import TavilySearch

        # Set API key in environment (required by langchain-tavily)
        os.environ["TAVILY_API_KEY"] = api_key

        tool = TavilySearch(
            max_results=max_results,
            search_depth="advanced",  # Deep search for better job results
            include_answer=False,
            include_raw_content=False,
        )

        # TavilySearch is sync, run in executor
        loop = asyncio.get_event_loop()
        raw_results = await loop.run_in_executor(None, tool.invoke, query)

        results: list[dict[str, str]] = []

        # Parse results — TavilySearch returns a list of dicts or a string
        if isinstance(raw_results, list):
            for item in raw_results[:max_results]:
                if isinstance(item, dict):
                    results.append({
                        "title": item.get("title", item.get("name", "")),
                        "url": item.get("url", item.get("link", "")),
                        "snippet": item.get("content", item.get("snippet", "")),
                    })
        elif isinstance(raw_results, str):
            # If it returns a formatted string, try to parse
            logger.debug(f"Tavily returned string: {raw_results[:200]}")

        logger.info(f"  🔍 Tavily returned {len(results)} results for: {query}")
        return results

    except ImportError:
        logger.warning("langchain-tavily not installed. Run: pip install langchain-tavily")
        return []
    except Exception as e:
        logger.warning(f"Tavily search failed: {e}")
        return []


# ── SerpAPI Search ───────────────────────────────────────────────────────────

async def _search_serpapi(query: str, num_results: int = 5) -> list[dict[str, str]]:
    """Execute a web search via SerpAPI (Google)."""
    settings = get_settings()
    api_key = settings.serp_api_key

    if not api_key:
        logger.debug("SerpAPI key not set, skipping SerpAPI search")
        return []

    params = {
        "q": query,
        "api_key": api_key,
        "num": num_results,
        "engine": "google",
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(SERP_API_URL, params=params)
            response.raise_for_status()
            data = response.json()

        results: list[dict[str, str]] = []
        for item in data.get("organic_results", [])[:num_results]:
            results.append({
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", ""),
            })

        logger.info(f"  🔍 SerpAPI returned {len(results)} results for: {query}")
        return results
    except Exception as e:
        logger.warning(f"SerpAPI search failed: {e}")
        return []


# ── Utility Functions ────────────────────────────────────────────────────────

def _detect_source(url: str) -> str:
    """Detect which job portal a URL belongs to."""
    url_lower = url.lower()
    if "naukri.com" in url_lower:
        return "naukri"
    elif "indeed.com" in url_lower:
        return "indeed"
    elif "linkedin.com" in url_lower:
        return "linkedin"
    elif "glassdoor.com" in url_lower:
        return "glassdoor"
    else:
        return "google"


def _deduplicate_listings(listings: list[dict]) -> list[dict]:
    """Remove duplicate job listings based on URL."""
    seen_urls: set[str] = set()
    unique: list[dict] = []
    for listing in listings:
        url = listing.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique.append(listing)
    return unique


def _extract_company(title: str, snippet: str) -> str:
    """Try to extract company name from the title or snippet."""
    for sep in [" - ", " at ", " | ", " — "]:
        if sep in title:
            parts = title.split(sep)
            if len(parts) >= 2:
                return parts[-1].strip()
    for sep in [" - ", " at ", " | "]:
        if sep in snippet:
            parts = snippet.split(sep)
            if len(parts) >= 2:
                return parts[0].strip()[:50]
    return "Unknown Company"


# ── Main Agent Node ──────────────────────────────────────────────────────────

async def search_jobs(state: GraphState) -> dict:
    """
    LangGraph node: Search for relevant jobs using BOTH Tavily and SerpAPI.

    Strategy:
      - Tavily: AI-optimized deep search with broader queries
      - SerpAPI: Google search with site: operators for specific portals
      - Results are merged and deduplicated

    Reads: state["cv_profile"], state["search_location"], state["max_jobs"]
    Writes: state["job_listings"], state["current_step"]
    """
    logger.info("🔍 Starting dual job search (Tavily + SerpAPI)...")

    cv_profile_data = state.get("cv_profile")
    if not cv_profile_data:
        return {
            "errors": ["No CV profile available for job search"],
            "current_step": "failed",
        }

    profile = CVProfile(**cv_profile_data)
    location = state.get("search_location", "India")
    max_jobs = state.get("max_jobs", 20)
    settings = get_settings()

    all_listings: list[dict] = []
    tavily_available = bool(settings.tavily_api_key)
    serpapi_available = bool(settings.serp_api_key)

    logger.info(
        f"  📊 Search engines: Tavily={'✅' if tavily_available else '❌'}, "
        f"SerpAPI={'✅' if serpapi_available else '❌'}"
    )

    for role in profile.preferred_roles[:3]:  # Top 3 roles

        # ── Tavily Search (AI-optimized, deep) ──
        if tavily_available:
            for query_template in TAVILY_QUERIES:
                query = query_template.format(role=role, location=location)
                logger.info(f"  🟣 [Tavily] Searching: {query}")
                try:
                    results = await _search_tavily(query, max_results=5)
                    for r in results:
                        listing = JobListing(
                            title=r["title"],
                            company=_extract_company(r["title"], r["snippet"]),
                            location=location,
                            url=r["url"],
                            source=_detect_source(r["url"]),
                            description_snippet=r["snippet"][:300],
                        )
                        all_listings.append(listing.model_dump())
                except Exception as e:
                    logger.warning(f"  ⚠️ Tavily failed for '{query}': {e}")
                await asyncio.sleep(0.3)

        # ── SerpAPI Search (Google with site: operators) ──
        if serpapi_available:
            for query_template in PORTAL_QUERIES:
                query = query_template.format(role=role, location=location)
                logger.info(f"  🔵 [SerpAPI] Searching: {query}")
                try:
                    results = await _search_serpapi(query, num_results=5)
                    for r in results:
                        listing = JobListing(
                            title=r["title"],
                            company=_extract_company(r["title"], r["snippet"]),
                            location=location,
                            url=r["url"],
                            source=_detect_source(r["url"]),
                            description_snippet=r["snippet"][:300],
                        )
                        all_listings.append(listing.model_dump())
                except Exception as e:
                    logger.warning(f"  ⚠️ SerpAPI failed for '{query}': {e}")
                await asyncio.sleep(0.5)

    if not tavily_available and not serpapi_available:
        return {
            "errors": ["No search API configured. Please set TAVILY_API_KEY and/or SERP_API_KEY."],
            "current_step": "failed",
        }

    # Deduplicate and limit
    unique_listings = _deduplicate_listings(all_listings)[:max_jobs]

    logger.info(
        f"✅ Found {len(unique_listings)} unique job listings "
        f"(from {len(all_listings)} total results across both engines)"
    )

    return {
        "job_listings": unique_listings,
        "current_step": "jobs_searched",
    }
