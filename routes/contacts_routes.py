"""API routes for HR contact database management."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from database import get_db
from models.db_models import CompanyEntity, HRContactEntity

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/contacts", tags=["Contacts"])


@router.get("")
async def list_contacts(
    domain: str | None = Query(None, description="Filter by company domain"),
    company: str | None = Query(None, description="Filter by company name (partial match)"),
    limit: int = Query(100, le=500),
    db: AsyncSession = Depends(get_db),
):
    """List all HR contacts, optionally filtered by domain or company name."""
    stmt = (
        select(HRContactEntity)
        .join(CompanyEntity)
        .options(selectinload(HRContactEntity.company))
    )

    if domain:
        stmt = stmt.where(CompanyEntity.domain == domain)
    if company:
        stmt = stmt.where(CompanyEntity.name.ilike(f"%{company}%"))

    stmt = stmt.order_by(HRContactEntity.created_at.desc()).limit(limit)
    result = await db.execute(stmt)
    contacts = result.scalars().all()

    return {
        "total": len(contacts),
        "contacts": [
            {
                "id": c.id,
                "name": c.name,
                "email": c.email,
                "role": c.role,
                "source": c.source,
                "verified": c.verified,
                "company_name": c.company.name if c.company else None,
                "company_domain": c.company.domain if c.company else None,
                "created_at": c.created_at.isoformat() if c.created_at else None,
            }
            for c in contacts
        ],
    }


@router.get("/companies")
async def list_companies(
    db: AsyncSession = Depends(get_db),
):
    """List all companies grouped by domain with contact counts."""
    stmt = (
        select(
            CompanyEntity.id,
            CompanyEntity.name,
            CompanyEntity.domain,
            CompanyEntity.industry,
            CompanyEntity.created_at,
            func.count(HRContactEntity.id).label("contact_count"),
        )
        .outerjoin(HRContactEntity)
        .group_by(CompanyEntity.id)
        .order_by(func.count(HRContactEntity.id).desc())
    )

    result = await db.execute(stmt)
    rows = result.all()

    return {
        "total": len(rows),
        "companies": [
            {
                "id": r.id,
                "name": r.name,
                "domain": r.domain,
                "industry": r.industry,
                "contact_count": r.contact_count,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ],
    }


@router.delete("/{contact_id}")
async def delete_contact(
    contact_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Delete a bad or unwanted HR contact."""
    result = await db.execute(
        select(HRContactEntity).where(HRContactEntity.id == contact_id)
    )
    contact = result.scalar_one_or_none()
    if not contact:
        raise HTTPException(status_code=404, detail="Contact not found")

    await db.delete(contact)
    return {"deleted": True, "id": contact_id}
