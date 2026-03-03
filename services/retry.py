"""Retry decorator with exponential backoff for async functions.

Use for external API calls (SerpAPI, Tavily, page fetching, SMTP)
to improve resilience against transient failures.
"""

from __future__ import annotations

import asyncio
import functools
import logging
from typing import Any, Callable, TypeVar

import httpx

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Default exceptions to retry on
RETRYABLE_EXCEPTIONS = (
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.ReadTimeout,
    ConnectionError,
    TimeoutError,
    OSError,
)


def retry(
    max_attempts: int = 3,
    backoff_factor: float = 1.5,
    exceptions: tuple[type[Exception], ...] = RETRYABLE_EXCEPTIONS,
) -> Callable:
    """Async retry decorator with exponential backoff.

    Args:
        max_attempts:   Maximum number of attempts (including the first).
        backoff_factor: Multiplier for delay between retries.
                        Delay = backoff_factor ** attempt (1.5s, 2.25s, 3.375s, ...)
        exceptions:     Tuple of exception types to retry on.

    Usage:
        @retry(max_attempts=3, backoff_factor=2)
        async def fetch_data(url: str) -> dict: ...
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Exception | None = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts:
                        delay = backoff_factor ** attempt
                        logger.warning(
                            f"⚠️ {func.__name__} attempt {attempt}/{max_attempts} "
                            f"failed: {e}. Retrying in {delay:.1f}s..."
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            f"❌ {func.__name__} failed after {max_attempts} "
                            f"attempts: {e}"
                        )

            raise last_exception  # type: ignore[misc]

        return wrapper

    return decorator
