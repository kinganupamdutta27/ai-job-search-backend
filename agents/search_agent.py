"""LangGraph Node — Job Search Agent.

Searches for relevant job openings across multiple portals using BOTH:
  1. Tavily Search API (via langchain-tavily) — AI-optimized, deep search
  2. SerpAPI — Google search with site: operators

Covers 7+ platforms: Naukri, Indeed, LinkedIn, Glassdoor, Foundit,
Instahyre, and general web — all searched in parallel for maximum coverage.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import httpx

from config import get_settings
from graph.state import GraphState
from graph.models import CVProfile, JobListing
from services.retry import retry

logger = logging.getLogger(__name__)

# ── Platform-Specific Search Templates (SerpAPI / Google) ────────────────────

PLATFORM_QUERIES = {
    "naukri":     'site:naukri.com "{role}" jobs {location}',
    "indeed":     'site:indeed.co.in OR site:indeed.com "{role}" jobs {location}',
    "linkedin":   'site:linkedin.com/jobs "{role}" {location}',
    "glassdoor":  'site:glassdoor.co.in OR site:glassdoor.com "{role}" jobs {location}',
    "foundit":    'site:foundit.in "{role}" jobs {location}',
    "instahyre":  'site:instahyre.com "{role}" jobs {location}',
    "general":    '"{role}" hiring {location} apply email HR contact 2026',
}

# ── Tavily Broad Search Queries ──────────────────────────────────────────────

TAVILY_QUERIES = [
    '{role} jobs {location} hiring naukri indeed linkedin glassdoor',
    '{role} careers {location} apply foundit instahyre',
    '{role} openings {location} HR email contact recruitment 2026',
]

SERP_API_URL = "https://serpapi.com/search.json"


# ── Tavily Search ────────────────────────────────────────────────────────────

@retry(max_attempts=2, backoff_factor=1.5)
async def _search_tavily(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """Execute a web search via Tavily (AI-optimized deep search)."""
    settings = get_settings()
    api_key = settings.tavily_api_key

    if not api_key:
        return []

    try:
        from langchain_tavily import TavilySearch

        os.environ["TAVILY_API_KEY"] = api_key

        tool = TavilySearch(
            max_results=max_results,
            search_depth="advanced",
            include_answer=False,
            include_raw_content=False,
        )

        loop = asyncio.get_event_loop()
        raw_results = await loop.run_in_executor(None, tool.invoke, query)

        results: list[dict[str, str]] = []
        if isinstance(raw_results, list):
            for item in raw_results[:max_results]:
                if isinstance(item, dict):
                    results.append({
                        "title": item.get("title", item.get("name", "")),
                        "url": item.get("url", item.get("link", "")),
                        "snippet": item.get("content", item.get("snippet", "")),
                    })

        logger.info(f"  🔍 Tavily returned {len(results)} results for: {query}")
        return results

    except ImportError:
        logger.warning("langchain-tavily not installed")
        return []
    except Exception as e:
        logger.warning(f"Tavily search failed: {e}")
        return []


# ── SerpAPI Search ───────────────────────────────────────────────────────────

@retry(max_attempts=2, backoff_factor=1.5)
async def _search_serpapi(query: str, num_results: int = 5) -> list[dict[str, str]]:
    """Execute a web search via SerpAPI (Google)."""
    settings = get_settings()
    api_key = settings.serp_api_key

    if not api_key:
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
    portals = {
        "naukri.com": "naukri",
        "indeed.co": "indeed",
        "indeed.com": "indeed",
        "linkedin.com": "linkedin",
        "glassdoor.co": "glassdoor",
        "glassdoor.com": "glassdoor",
        "foundit.in": "foundit",
        "instahyre.com": "instahyre",
    }
    for domain, source in portals.items():
        if domain in url_lower:
            return source
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


# ── Parallel Search Coroutines ───────────────────────────────────────────────

async def _run_serp_for_platform(
    platform: str, query_template: str, role: str, location: str
) -> list[dict]:
    """Run a single SerpAPI search for one platform query."""
    query = query_template.format(role=role, location=location)
    logger.info(f"  🔵 [SerpAPI/{platform}] {query}")
    try:
        results = await _search_serpapi(query, num_results=5)
        listings = []
        for r in results:
            listing = JobListing(
                title=r["title"],
                company=_extract_company(r["title"], r["snippet"]),
                location=location,
                url=r["url"],
                source=_detect_source(r["url"]),
                description_snippet=r["snippet"][:300],
            )
            listings.append(listing.model_dump())
        return listings
    except Exception as e:
        logger.warning(f"  ⚠️ SerpAPI/{platform} failed: {e}")
        return []


async def _run_tavily_query(
    query_template: str, role: str, location: str
) -> list[dict]:
    """Run a single Tavily search query."""
    query = query_template.format(role=role, location=location)
    logger.info(f"  🟣 [Tavily] {query}")
    try:
        results = await _search_tavily(query, max_results=5)
        listings = []
        for r in results:
            listing = JobListing(
                title=r["title"],
                company=_extract_company(r["title"], r["snippet"]),
                location=location,
                url=r["url"],
                source=_detect_source(r["url"]),
                description_snippet=r["snippet"][:300],
            )
            listings.append(listing.model_dump())
        return listings
    except Exception as e:
        logger.warning(f"  ⚠️ Tavily failed: {e}")
        return []


# ── Main Agent Node ──────────────────────────────────────────────────────────

async def search_jobs(state: GraphState) -> dict:
    """
    LangGraph node: Search for relevant jobs across 7+ platforms in parallel.

    Uses BOTH Tavily (AI-optimized) and SerpAPI (Google site: operators)
    for maximum coverage across Naukri, Indeed, LinkedIn, Glassdoor,
    Foundit, Instahyre, and the general web.
    """
    logger.info("🔍 Starting multi-platform job search (7+ platforms)...")

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

    tavily_available = bool(settings.tavily_api_key)
    serpapi_available = bool(settings.serp_api_key)

    logger.info(
        f"  📊 Engines: Tavily={'✅' if tavily_available else '❌'}, "
        f"SerpAPI={'✅' if serpapi_available else '❌'}"
    )
    logger.info(
        f"  🎯 Platforms: {', '.join(PLATFORM_QUERIES.keys())} "
        f"({len(PLATFORM_QUERIES)} total)"
    )

    all_tasks = []

    for role in profile.preferred_roles[:3]:  # Top 3 roles

        # SerpAPI: one task per platform
        if serpapi_available:
            for platform, query_template in PLATFORM_QUERIES.items():
                all_tasks.append(
                    _run_serp_for_platform(platform, query_template, role, location)
                )

        # Tavily: broader AI-powered queries
        if tavily_available:
            for query_template in TAVILY_QUERIES:
                all_tasks.append(
                    _run_tavily_query(query_template, role, location)
                )

    if not all_tasks:
        return {
            "errors": ["No search API configured. Set TAVILY_API_KEY and/or SERP_API_KEY."],
            "current_step": "failed",
        }

    # Execute ALL searches in parallel
    logger.info(f"  ⚡ Launching {len(all_tasks)} parallel search tasks...")
    results = await asyncio.gather(*all_tasks, return_exceptions=True)

    all_listings: list[dict] = []
    for result in results:
        if isinstance(result, list):
            all_listings.extend(result)
        elif isinstance(result, Exception):
            logger.warning(f"  ⚠️ A search task failed: {result}")

    # Deduplicate and limit
    unique_listings = _deduplicate_listings(all_listings)[:max_jobs]

    # Report platform distribution
    sources = {}
    for l in unique_listings:
        src = l.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1
    logger.info(
        f"✅ Found {len(unique_listings)} unique jobs "
        f"(from {len(all_listings)} total) — by platform: {sources}"
    )

    return {
        "job_listings": unique_listings,
        "current_step": "jobs_searched",
    }
