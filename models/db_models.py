"""SQLAlchemy ORM models for persistent workflow storage."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, func
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


# ── Contact Finder Runs ──────────────────────────────────────────────────────


class ContactFinderRunEntity(Base):
    """Tracks an independent contact discovery run."""

    __tablename__ = "contact_finder_runs"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_new_uuid
    )
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(
        String(32), default="running", index=True
    )
    results_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    contacts_found: Mapped[int] = mapped_column(Integer, default=0)
    companies_found: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow,
        server_default=func.now(),
    )

    def __repr__(self) -> str:
        return f"<ContactFinderRun id={self.id!r} status={self.status!r}>"


# ── LinkedIn Tables ──────────────────────────────────────────────────────────


class LinkedInCredentialEntity(Base):
    """Encrypted LinkedIn login credentials."""

    __tablename__ = "linkedin_credentials"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_new_uuid
    )
    encrypted_email: Mapped[str] = mapped_column(Text, nullable=False)
    encrypted_password: Mapped[str] = mapped_column(Text, nullable=False)
    encrypted_totp_secret: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_verified_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow,
        server_default=func.now(),
    )

    def __repr__(self) -> str:
        return f"<LinkedInCredential id={self.id!r}>"


class LinkedInPostEntity(Base):
    """A LinkedIn post (draft, scheduled, or published)."""

    __tablename__ = "linkedin_posts"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_new_uuid
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    topic: Mapped[str] = mapped_column(String(500), default="")
    status: Mapped[str] = mapped_column(
        String(32), default="draft", index=True
    )
    scheduled_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    published_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    celery_task_id: Mapped[str | None] = mapped_column(
        String(255), nullable=True
    )
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow,
        server_default=func.now(),
    )

    def __repr__(self) -> str:
        return f"<LinkedInPost id={self.id!r} status={self.status!r}>"


# ── Job Auto-Apply Tables ─────────────────────────────────────────────────


class JobApplyProfileEntity(Base):
    """Stores resume data + application preferences for auto-apply."""

    __tablename__ = "job_apply_profiles"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_new_uuid
    )
    cv_file_path: Mapped[str] = mapped_column(Text, nullable=False)
    cv_profile_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    expected_salary: Mapped[str | None] = mapped_column(String(100), nullable=True)
    current_ctc: Mapped[str | None] = mapped_column(String(100), nullable=True)
    expected_ctc: Mapped[str | None] = mapped_column(String(100), nullable=True)
    notice_period: Mapped[str | None] = mapped_column(String(50), nullable=True)
    work_authorization: Mapped[str | None] = mapped_column(String(100), nullable=True)
    willing_to_relocate: Mapped[bool] = mapped_column(Boolean, default=False)
    preferred_job_titles: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    preferred_locations: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    years_of_experience: Mapped[float] = mapped_column(Float, default=0.0)
    additional_info: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow,
        server_default=func.now(),
    )

    sessions: Mapped[list["JobApplySessionEntity"]] = relationship(
        back_populates="profile", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<JobApplyProfile id={self.id!r}>"


class JobApplySessionEntity(Base):
    """Tracks one auto-apply run (searching + applying to jobs)."""

    __tablename__ = "job_apply_sessions"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_new_uuid
    )
    profile_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("job_apply_profiles.id"), index=True, nullable=False
    )
    status: Mapped[str] = mapped_column(
        String(32), default="searching", index=True
    )
    search_criteria_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    max_applications: Mapped[int] = mapped_column(Integer, default=10)
    applied_count: Mapped[int] = mapped_column(Integer, default=0)
    skipped_count: Mapped[int] = mapped_column(Integer, default=0)
    failed_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow,
        server_default=func.now(),
    )

    profile: Mapped["JobApplyProfileEntity"] = relationship(back_populates="sessions")
    applications: Mapped[list["JobApplicationEntity"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<JobApplySession id={self.id!r} status={self.status!r}>"


class JobApplicationEntity(Base):
    """Tracks an individual job application attempt."""

    __tablename__ = "job_applications"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_new_uuid
    )
    session_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("job_apply_sessions.id"), index=True, nullable=False
    )
    job_title: Mapped[str] = mapped_column(String(300), default="")
    company: Mapped[str] = mapped_column(String(300), default="")
    job_url: Mapped[str] = mapped_column(Text, default="")
    job_location: Mapped[str] = mapped_column(String(200), default="")
    apply_type: Mapped[str] = mapped_column(String(30), default="unknown")
    status: Mapped[str] = mapped_column(
        String(32), default="pending", index=True
    )
    questions_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    applied_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )

    session: Mapped["JobApplySessionEntity"] = relationship(back_populates="applications")

    def __repr__(self) -> str:
        return f"<JobApplication id={self.id!r} status={self.status!r}>"


class SavedQAEntity(Base):
    """Stores question-answer pairs learned from past applications.

    When the system encounters a form field and fills it (deterministically
    or via LLM), the Q&A pair is saved here.  On subsequent runs the stored
    answer is used directly, eliminating LLM calls for repeat questions.
    """

    __tablename__ = "saved_qa"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_new_uuid
    )
    question: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    question_normalised: Mapped[str] = mapped_column(
        Text, nullable=False, index=True,
    )
    answer: Mapped[str] = mapped_column(Text, nullable=False, default="")
    field_type: Mapped[str] = mapped_column(String(30), default="text")
    source: Mapped[str] = mapped_column(
        String(30), default="llm",
    )
    times_used: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow,
        server_default=func.now(),
    )

    def __repr__(self) -> str:
        return f"<SavedQA q={self.question_normalised!r} a={self.answer!r}>"
