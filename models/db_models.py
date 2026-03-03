"""SQLAlchemy ORM models for persistent workflow storage."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    return str(uuid.uuid4())


class WorkflowRunEntity(Base):
    """Persisted record of a single LangGraph workflow execution."""

    __tablename__ = "workflow_runs"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_new_uuid
    )
    status: Mapped[str] = mapped_column(
        String(32), default="running", index=True
    )
    state_json: Mapped[dict] = mapped_column(
        JSON, default=dict, nullable=False
    )
    config_json: Mapped[dict] = mapped_column(
        JSON, default=dict, nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        onupdate=_utcnow,
        server_default=func.now(),
    )

    def __repr__(self) -> str:
        return f"<WorkflowRun id={self.id!r} status={self.status!r}>"


# ── Company & HR Contact Tables ──────────────────────────────────────────────


class CompanyEntity(Base):
    """A company discovered during job searches, keyed by domain."""

    __tablename__ = "companies"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_new_uuid
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    domain: Mapped[str] = mapped_column(
        String(200), unique=True, index=True, nullable=False
    )
    industry: Mapped[str | None] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow,
        server_default=func.now(),
    )

    # Relationships
    contacts: Mapped[list["HRContactEntity"]] = relationship(
        back_populates="company", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Company name={self.name!r} domain={self.domain!r}>"


class HRContactEntity(Base):
    """An HR / recruiter contact discovered for a company."""

    __tablename__ = "hr_contacts"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_new_uuid
    )
    company_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("companies.id"), index=True, nullable=False
    )
    name: Mapped[str] = mapped_column(
        String(200), default="Hiring Manager"
    )
    email: Mapped[str] = mapped_column(
        String(254), unique=True, index=True, nullable=False
    )
    role: Mapped[str] = mapped_column(
        String(100), default="HR"
    )
    source: Mapped[str] = mapped_column(
        String(30), default="extracted"
    )
    verified: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow,
        server_default=func.now(),
    )

    # Relationships
    company: Mapped["CompanyEntity"] = relationship(back_populates="contacts")

    def __repr__(self) -> str:
        return f"<HRContact email={self.email!r} company_id={self.company_id!r}>"
