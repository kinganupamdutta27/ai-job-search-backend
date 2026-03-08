"""API routes for the standalone Contact Finder feature."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db, async_session_maker
from models.db_models import ContactFinderRunEntity
from agents.contact_finder_agent import find_contacts

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/contact-finder", tags=["Contact Finder"])


class StartRequest(BaseModel):
    prompt: str = Field(..., min_length=5, description="Natural language search query")
    max_companies: int = Field(default=20, ge=1, le=100)


@router.post("/start")
async def start_contact_finder(req: StartRequest):
    """Start a new contact discovery run in the background."""
    async with async_session_maker() as db:
        run = ContactFinderRunEntity(prompt=req.prompt, status="running")
        db.add(run)
        await db.commit()
        await db.refresh(run)
        run_id = run.id

    asyncio.create_task(
        _run_finder(run_id, req.prompt, req.max_companies)
    )

    return {"run_id": run_id, "status": "running", "prompt": req.prompt}


async def _run_finder(run_id: str, prompt: str, max_companies: int):
    """Background wrapper that catches exceptions and marks the run as failed."""
    try:
        await find_contacts(run_id, prompt, max_companies)
    except Exception as e:
        logger.error(f"Contact finder run {run_id} failed: {e}")
        async with async_session_maker() as db:
            result = await db.execute(
                select(ContactFinderRunEntity).where(
                    ContactFinderRunEntity.id == run_id
                )
            )
            run = result.scalar_one_or_none()
            if run:
                run.status = "failed"
                run.results_json = {"error": str(e)}
                await db.commit()


@router.get("/runs")
async def list_finder_runs(db: AsyncSession = Depends(get_db)):
    """List all contact finder runs."""
    result = await db.execute(
        select(ContactFinderRunEntity)
        .order_by(ContactFinderRunEntity.created_at.desc())
        .limit(50)
    )
    runs = result.scalars().all()

    return {
        "total": len(runs),
        "runs": [
            {
                "id": r.id,
                "prompt": r.prompt,
                "status": r.status,
                "contacts_found": r.contacts_found,
                "companies_found": r.companies_found,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in runs
        ],
    }


@router.get("/{run_id}/status")
async def get_finder_status(run_id: str, db: AsyncSession = Depends(get_db)):
    """Poll the status and results of a contact finder run."""
    result = await db.execute(
        select(ContactFinderRunEntity).where(ContactFinderRunEntity.id == run_id)
    )
    run = result.scalar_one_or_none()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    return {
        "run_id": run.id,
        "prompt": run.prompt,
        "status": run.status,
        "contacts_found": run.contacts_found,
        "companies_found": run.companies_found,
        "results": run.results_json,
        "created_at": run.created_at.isoformat() if run.created_at else None,
        "updated_at": run.updated_at.isoformat() if run.updated_at else None,
    }
