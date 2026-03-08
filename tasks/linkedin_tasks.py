"""Celery tasks for scheduled LinkedIn post publishing.

These tasks run in a Celery worker process. Each scheduled post:
  1. Optionally re-researches the topic for freshness
  2. Optionally re-generates the post content
  3. Publishes via Playwright browser automation
  4. Updates the DB record with the result
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from celery_app import app

logger = logging.getLogger(__name__)


def _run_async(coro):
    """Run an async coroutine from a sync Celery task."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, coro).result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


async def _execute_publish(post_id: str) -> dict:
    """Core async logic for publishing a scheduled post."""
    from dotenv import load_dotenv
    load_dotenv()

    from sqlalchemy import select
    from database import async_session_maker
    from models.db_models import LinkedInPostEntity, LinkedInCredentialEntity
    from services.crypto_service import decrypt
    from agents.linkedin_agent import (
        publish_post_to_linkedin,
        generate_linkedin_post,
        research_trending_topics,
    )

    async with async_session_maker() as db:
        result = await db.execute(
            select(LinkedInPostEntity).where(LinkedInPostEntity.id == post_id)
        )
        post = result.scalar_one_or_none()
        if not post:
            return {"success": False, "error": f"Post {post_id} not found"}

        if post.status != "scheduled":
            return {"success": False, "error": f"Post status is {post.status}, not scheduled"}

        # Get credentials
        cred_result = await db.execute(select(LinkedInCredentialEntity).limit(1))
        cred = cred_result.scalar_one_or_none()
        if not cred:
            post.status = "failed"
            post.error = "No LinkedIn credentials configured"
            await db.commit()
            return {"success": False, "error": "No credentials"}

        email = decrypt(cred.encrypted_email)
        password = decrypt(cred.encrypted_password)
        totp = decrypt(cred.encrypted_totp_secret) if cred.encrypted_totp_secret else None

        content = post.content

        # If topic is set, optionally refresh the content for timeliness
        if post.topic and not content.strip():
            try:
                research = await research_trending_topics(post.topic)
                content = await generate_linkedin_post(post.topic, research)
                post.content = content
            except Exception as e:
                logger.warning(f"Content generation failed for {post_id}: {e}")

        # Publish
        success, message = await publish_post_to_linkedin(
            email, password, content, totp, headless=True
        )

        if success:
            post.status = "published"
            post.published_at = datetime.now(timezone.utc)
        else:
            post.status = "failed"
            post.error = message

        await db.commit()

    return {"success": success, "message": message, "post_id": post_id}


@app.task(bind=True, name="tasks.linkedin_tasks.publish_scheduled_post", max_retries=2)
def publish_scheduled_post(self, post_id: str) -> dict:
    """Celery task: Publish a scheduled LinkedIn post.

    Retries up to 2 times on failure with exponential backoff.
    """
    logger.info(f"[Celery] Publishing scheduled post {post_id}...")

    try:
        result = _run_async(_execute_publish(post_id))

        if not result.get("success") and self.request.retries < self.max_retries:
            raise Exception(result.get("error", "Unknown error"))

        return result

    except Exception as exc:
        logger.error(f"[Celery] Failed to publish {post_id}: {exc}")
        self.retry(exc=exc, countdown=60 * (2 ** self.request.retries))
