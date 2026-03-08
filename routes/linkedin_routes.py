"""API routes for LinkedIn automation — credentials, post generation, publishing, scheduling."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db, async_session_maker
from models.db_models import LinkedInCredentialEntity, LinkedInPostEntity
from services.crypto_service import encrypt, decrypt
from agents.linkedin_agent import (
    login_linkedin,
    generate_linkedin_post,
    publish_post_to_linkedin,
    research_trending_topics,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/linkedin", tags=["LinkedIn"])


# ── Request / Response Models ────────────────────────────────────────────────

class CredentialRequest(BaseModel):
    email: str = Field(..., min_length=3)
    password: str = Field(..., min_length=1)
    totp_secret: Optional[str] = Field(default=None)


class GeneratePostRequest(BaseModel):
    topic: str = Field(..., min_length=3)


class PublishPostRequest(BaseModel):
    post_id: str = Field(..., description="ID of the post to publish")


class SchedulePostRequest(BaseModel):
    topic: str = Field(..., min_length=3)
    content: Optional[str] = Field(default=None, description="Pre-generated content, or leave empty to auto-generate")
    scheduled_at: str = Field(..., description="ISO 8601 datetime for scheduled publish")


# ── Credential Endpoints ─────────────────────────────────────────────────────

@router.post("/auth")
async def save_credentials(req: CredentialRequest, db: AsyncSession = Depends(get_db)):
    """Save (or update) LinkedIn credentials. Existing credentials are replaced."""
    result = await db.execute(select(LinkedInCredentialEntity).limit(1))
    cred = result.scalar_one_or_none()

    if cred:
        cred.encrypted_email = encrypt(req.email)
        cred.encrypted_password = encrypt(req.password)
        cred.encrypted_totp_secret = encrypt(req.totp_secret) if req.totp_secret else None
    else:
        cred = LinkedInCredentialEntity(
            encrypted_email=encrypt(req.email),
            encrypted_password=encrypt(req.password),
            encrypted_totp_secret=encrypt(req.totp_secret) if req.totp_secret else None,
        )
        db.add(cred)

    await db.commit()
    return {"saved": True, "message": "Credentials saved securely"}


@router.post("/auth/verify")
async def verify_credentials(db: AsyncSession = Depends(get_db)):
    """Test LinkedIn login with saved credentials."""
    cred = await _get_credentials(db)

    email = decrypt(cred.encrypted_email)
    password = decrypt(cred.encrypted_password)
    totp = decrypt(cred.encrypted_totp_secret) if cred.encrypted_totp_secret else None

    # headless=False so the user can interact with any 2FA challenge page
    success, message = await login_linkedin(email, password, totp, headless=False)

    if success:
        cred.last_verified_at = datetime.now(timezone.utc)
        await db.commit()

    return {"success": success, "message": message}


@router.get("/auth/status")
async def auth_status(db: AsyncSession = Depends(get_db)):
    """Check if LinkedIn credentials are saved."""
    result = await db.execute(select(LinkedInCredentialEntity).limit(1))
    cred = result.scalar_one_or_none()

    if not cred:
        return {"configured": False, "last_verified": None}

    return {
        "configured": True,
        "last_verified": cred.last_verified_at.isoformat() if cred.last_verified_at else None,
    }


# ── Post Generation ──────────────────────────────────────────────────────────

@router.post("/post/generate")
async def generate_post(req: GeneratePostRequest, db: AsyncSession = Depends(get_db)):
    """Generate a human-like LinkedIn post about a topic."""
    research = await research_trending_topics(req.topic)
    content = await generate_linkedin_post(req.topic, research)

    post = LinkedInPostEntity(
        content=content,
        topic=req.topic,
        status="draft",
    )
    db.add(post)
    await db.commit()
    await db.refresh(post)

    return {
        "post_id": post.id,
        "content": content,
        "topic": req.topic,
        "research_summary": research[:500],
        "status": "draft",
    }


@router.put("/post/{post_id}")
async def update_post(post_id: str, content: str = "", db: AsyncSession = Depends(get_db)):
    """Update a draft post's content before publishing."""
    result = await db.execute(
        select(LinkedInPostEntity).where(LinkedInPostEntity.id == post_id)
    )
    post = result.scalar_one_or_none()
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    if post.status not in ("draft", "scheduled"):
        raise HTTPException(status_code=400, detail=f"Cannot edit a {post.status} post")

    post.content = content
    await db.commit()
    return {"updated": True, "post_id": post_id}


# ── Publish ──────────────────────────────────────────────────────────────────

@router.post("/post/publish")
async def publish_post(req: PublishPostRequest, db: AsyncSession = Depends(get_db)):
    """Publish a draft post to LinkedIn immediately."""
    cred = await _get_credentials(db)

    result = await db.execute(
        select(LinkedInPostEntity).where(LinkedInPostEntity.id == req.post_id)
    )
    post = result.scalar_one_or_none()
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    if post.status == "published":
        raise HTTPException(status_code=400, detail="Post is already published")

    email = decrypt(cred.encrypted_email)
    password = decrypt(cred.encrypted_password)
    totp = decrypt(cred.encrypted_totp_secret) if cred.encrypted_totp_secret else None

    # headless=False so the user can interact with any 2FA challenge page
    success, message = await publish_post_to_linkedin(
        email, password, post.content, totp, headless=False
    )

    if success:
        post.status = "published"
        post.published_at = datetime.now(timezone.utc)
    else:
        post.status = "failed"
        post.error = message

    await db.commit()
    return {"success": success, "message": message, "post_id": post.id, "status": post.status}


# ── Scheduling ───────────────────────────────────────────────────────────────

@router.post("/post/schedule")
async def schedule_post(req: SchedulePostRequest, db: AsyncSession = Depends(get_db)):
    """Schedule a post for future publishing via Celery."""
    cred = await _get_credentials(db)

    try:
        scheduled_dt = datetime.fromisoformat(req.scheduled_at)
        if scheduled_dt.tzinfo is None:
            scheduled_dt = scheduled_dt.replace(tzinfo=timezone.utc)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid datetime format. Use ISO 8601.")

    content = req.content
    if not content:
        research = await research_trending_topics(req.topic)
        content = await generate_linkedin_post(req.topic, research)

    post = LinkedInPostEntity(
        content=content,
        topic=req.topic,
        status="scheduled",
        scheduled_at=scheduled_dt,
    )
    db.add(post)
    await db.commit()
    await db.refresh(post)

    # Queue the Celery task
    try:
        from tasks.linkedin_tasks import publish_scheduled_post

        eta = scheduled_dt
        task = publish_scheduled_post.apply_async(
            args=[post.id],
            eta=eta,
        )
        post.celery_task_id = task.id
        await db.commit()
    except Exception as e:
        logger.warning(f"Failed to queue Celery task (Redis may be down): {e}")

    return {
        "post_id": post.id,
        "content": content,
        "topic": req.topic,
        "status": "scheduled",
        "scheduled_at": scheduled_dt.isoformat(),
        "celery_task_id": post.celery_task_id,
    }


# ── List Posts ───────────────────────────────────────────────────────────────

@router.get("/posts")
async def list_posts(
    status: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """List all LinkedIn posts, optionally filtered by status."""
    stmt = select(LinkedInPostEntity).order_by(LinkedInPostEntity.created_at.desc()).limit(100)
    if status:
        stmt = stmt.where(LinkedInPostEntity.status == status)

    result = await db.execute(stmt)
    posts = result.scalars().all()

    return {
        "total": len(posts),
        "posts": [
            {
                "id": p.id,
                "content": p.content[:200] + ("..." if len(p.content) > 200 else ""),
                "full_content": p.content,
                "topic": p.topic,
                "status": p.status,
                "scheduled_at": p.scheduled_at.isoformat() if p.scheduled_at else None,
                "published_at": p.published_at.isoformat() if p.published_at else None,
                "error": p.error,
                "created_at": p.created_at.isoformat() if p.created_at else None,
            }
            for p in posts
        ],
    }


@router.get("/schedules")
async def list_schedules(db: AsyncSession = Depends(get_db)):
    """List all scheduled (pending) posts."""
    result = await db.execute(
        select(LinkedInPostEntity)
        .where(LinkedInPostEntity.status == "scheduled")
        .order_by(LinkedInPostEntity.scheduled_at.asc())
    )
    posts = result.scalars().all()

    return {
        "total": len(posts),
        "schedules": [
            {
                "id": p.id,
                "topic": p.topic,
                "content": p.content[:200] + ("..." if len(p.content) > 200 else ""),
                "scheduled_at": p.scheduled_at.isoformat() if p.scheduled_at else None,
                "celery_task_id": p.celery_task_id,
            }
            for p in posts
        ],
    }


@router.delete("/schedules/{post_id}")
async def cancel_schedule(post_id: str, db: AsyncSession = Depends(get_db)):
    """Cancel a scheduled post."""
    result = await db.execute(
        select(LinkedInPostEntity).where(LinkedInPostEntity.id == post_id)
    )
    post = result.scalar_one_or_none()
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    if post.status != "scheduled":
        raise HTTPException(status_code=400, detail=f"Post is not scheduled (status: {post.status})")

    if post.celery_task_id:
        try:
            from celery_app import app as celery_app
            celery_app.control.revoke(post.celery_task_id, terminate=True)
        except Exception as e:
            logger.warning(f"Failed to revoke Celery task: {e}")

    post.status = "cancelled"
    await db.commit()

    return {"cancelled": True, "post_id": post_id}


# ── Helpers ──────────────────────────────────────────────────────────────────

async def _get_credentials(db: AsyncSession) -> LinkedInCredentialEntity:
    """Fetch the stored LinkedIn credentials or raise 400."""
    result = await db.execute(select(LinkedInCredentialEntity).limit(1))
    cred = result.scalar_one_or_none()
    if not cred:
        raise HTTPException(
            status_code=400,
            detail="LinkedIn credentials not configured. Save them first via POST /api/linkedin/auth",
        )
    return cred
