"""Rate limiter for web scraping to prevent IP bans and respect targets.

Provides domain-based rate limiting (delay between requests to same domain)
and global concurrency limits. Handles Retry-After and 429/503 backoff.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


class DomainRateLimiter:
    """Limits the rate of requests per domain to avoid rate-limiting."""

    def __init__(self, requests_per_second: float = 2.0, max_concurrent: int = 5):
        self.delay_between_requests = 1.0 / requests_per_second
        self.global_semaphore = asyncio.Semaphore(max_concurrent)
        self.domain_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self.last_request_time: dict[str, float] = defaultdict(float)

    async def acquire(self, domain: str) -> None:
        """Wait if necessary before making a request to the domain.

        Applies both the global concurrency limit and the per-domain delay.
        """
        await self.global_semaphore.acquire()

        lock = self.domain_locks[domain]
        await lock.acquire()

        try:
            now = asyncio.get_event_loop().time()
            last = self.last_request_time[domain]
            elapsed = now - last

            if elapsed < self.delay_between_requests:
                sleep_time = self.delay_between_requests - elapsed
                await asyncio.sleep(sleep_time)

            self.last_request_time[domain] = asyncio.get_event_loop().time()
        finally:
            lock.release()

    def release(self) -> None:
        """Release the global concurrency slot."""
        self.global_semaphore.release()

    async def handle_429(self, domain: str, response: httpx.Response) -> float:
        """Handle a 429 Too Many Requests response.

        Returns the recommended backoff time in seconds (from Retry-After header
        or fallback default). Locks the domain for that duration.
        """
        retry_after = response.headers.get("Retry-After")
        backoff = 30.0  # default backoff

        if retry_after:
            try:
                backoff = float(retry_after)
            except ValueError:
                # Need to parse HTTP date? Simplification: stick to default
                pass

        logger.warning(
            f"🚫 Rate limited (429) by {domain}. Backing off for {backoff}s."
        )

        # Force the last request time into the future to effectively lock the domain
        lock = self.domain_locks[domain]
        await lock.acquire()
        try:
            self.last_request_time[domain] = asyncio.get_event_loop().time() + backoff
        finally:
            lock.release()

        return backoff


# Singleton instance
rate_limiter = DomainRateLimiter(requests_per_second=1.0, max_concurrent=5)
