"""API routes for LinkedIn Job Auto-Apply — profile CRUD, sessions, live status."""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from config import get_settings
from database import get_db, async_session_maker
from models.db_models import (
    JobApplyProfileEntity,
    JobApplySessionEntity,
    JobApplicationEntity,
    LinkedInCredentialEntity,
    SavedQAEntity,
)
from services.crypto_service import decrypt
from services.cv_parser import parse_cv
from graph.models import (
    CVProfile,
    JobApplyPreferences,
    JobApplyRequest,
    JobApplySessionStatus,
    JobSearchCriteria,
    ApplicationResult,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/job-apply", tags=["Job Auto-Apply"])


# ── Request Models ───────────────────────────────────────────────────────────

class UpdatePreferencesRequest(BaseModel):
    expected_salary: str = ""
    current_ctc: str = ""
    expected_ctc: str = ""
    notice_period: str = "Immediate"
    work_authorization: str = ""
    willing_to_relocate: bool = False
    preferred_job_titles: list[str] = []
    preferred_locations: list[str] = []
    years_of_experience: float = 0
    additional_info: dict = {}


class StartSessionRequest(BaseModel):
    profile_id: str
    keywords: str = Field(..., min_length=1)
    location: str = ""
    experience_level: str = ""
    date_posted: str = ""
    easy_apply_only: bool = True
    max_applications: int = Field(default=10, ge=1, le=50)


# ── Profile Endpoints ────────────────────────────────────────────────────────

@router.post("/profile")
async def create_profile(
    file: UploadFile = File(...),
    expected_salary: str = Form(""),
    current_ctc: str = Form(""),
    expected_ctc: str = Form(""),
    notice_period: str = Form("Immediate"),
    work_authorization: str = Form(""),
    willing_to_relocate: bool = Form(False),
    preferred_job_titles: str = Form("[]"),
    preferred_locations: str = Form("[]"),
    years_of_experience: float = Form(0),
    additional_info: str = Form("{}"),
    db: AsyncSession = Depends(get_db),
):
    """Upload CV and save profile with preferences."""
    settings = get_settings()
    os.makedirs(settings.upload_dir, exist_ok=True)

    ext = os.path.splitext(file.filename or "resume.pdf")[1].lower()
    if ext not in (".pdf", ".docx", ".txt"):
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    file_id = str(uuid.uuid4())
    file_path = os.path.join(settings.upload_dir, f"{file_id}{ext}")

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    # Parse CV
    try:
        cv_text = parse_cv(file_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CV parsing failed: {str(e)}")

    # Extract structured profile via LLM
    from services.llm_utils import clean_llm_json, create_llm
    from langchain_core.messages import SystemMessage, HumanMessage
    from agents.cv_agent import CV_ANALYSIS_PROMPT

    llm = create_llm(temperature=0.1)
    messages = [
        SystemMessage(content=CV_ANALYSIS_PROMPT),
        HumanMessage(content=f"Here is the CV text to analyze:\n\n{cv_text}"),
    ]
    response = await llm.ainvoke(messages)
    cleaned = clean_llm_json(response.content)
    cv_profile_data = json.loads(cleaned)
    cv_profile = CVProfile(**cv_profile_data)

    # Parse JSON form fields
    try:
        job_titles = json.loads(preferred_job_titles)
    except Exception:
        job_titles = [t.strip() for t in preferred_job_titles.split(",") if t.strip()]

    try:
        locations = json.loads(preferred_locations)
    except Exception:
        locations = [l.strip() for l in preferred_locations.split(",") if l.strip()]

    try:
        extra_info = json.loads(additional_info)
    except Exception:
        extra_info = {}

    # Auto-fill from CV if not provided
    if not job_titles and cv_profile.preferred_roles:
        job_titles = cv_profile.preferred_roles
    if years_of_experience == 0 and cv_profile.years_of_experience > 0:
        years_of_experience = cv_profile.years_of_experience

    profile = JobApplyProfileEntity(
        cv_file_path=file_path,
        cv_profile_json=cv_profile.model_dump(),
        expected_salary=expected_salary,
        current_ctc=current_ctc,
        expected_ctc=expected_ctc,
        notice_period=notice_period,
        work_authorization=work_authorization,
        willing_to_relocate=willing_to_relocate,
        preferred_job_titles=job_titles,
        preferred_locations=locations,
        years_of_experience=years_of_experience,
        additional_info=extra_info,
    )
    db.add(profile)
    await db.commit()
    await db.refresh(profile)

    logger.info(f"Created job apply profile: {profile.id} (CV: {cv_profile.name})")

    return {
        "profile_id": profile.id,
        "cv_profile": cv_profile.model_dump(),
        "preferences": {
            "expected_salary": expected_salary,
            "current_ctc": current_ctc,
            "expected_ctc": expected_ctc,
            "notice_period": notice_period,
            "work_authorization": work_authorization,
            "willing_to_relocate": willing_to_relocate,
            "preferred_job_titles": job_titles,
            "preferred_locations": locations,
            "years_of_experience": years_of_experience,
            "additional_info": extra_info,
        },
    }


@router.get("/profile")
async def get_profile(db: AsyncSession = Depends(get_db)):
    """Get the most recent saved profile."""
    result = await db.execute(
        select(JobApplyProfileEntity)
        .order_by(JobApplyProfileEntity.created_at.desc())
        .limit(1)
    )
    profile = result.scalar_one_or_none()
    if not profile:
        return {"profile": None}

    return {
        "profile": {
            "id": profile.id,
            "cv_file_path": profile.cv_file_path,
            "cv_profile": profile.cv_profile_json,
            "expected_salary": profile.expected_salary,
            "current_ctc": profile.current_ctc,
            "expected_ctc": profile.expected_ctc,
            "notice_period": profile.notice_period,
            "work_authorization": profile.work_authorization,
            "willing_to_relocate": profile.willing_to_relocate,
            "preferred_job_titles": profile.preferred_job_titles,
            "preferred_locations": profile.preferred_locations,
            "years_of_experience": profile.years_of_experience,
            "additional_info": profile.additional_info,
            "created_at": profile.created_at.isoformat() if profile.created_at else None,
        }
    }


@router.put("/profile/{profile_id}")
async def update_profile(
    profile_id: str,
    req: UpdatePreferencesRequest,
    db: AsyncSession = Depends(get_db),
):
    """Update preferences on an existing profile without re-uploading CV."""
    result = await db.execute(
        select(JobApplyProfileEntity).where(JobApplyProfileEntity.id == profile_id)
    )
    profile = result.scalar_one_or_none()
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    profile.expected_salary = req.expected_salary
    profile.current_ctc = req.current_ctc
    profile.expected_ctc = req.expected_ctc
    profile.notice_period = req.notice_period
    profile.work_authorization = req.work_authorization
    profile.willing_to_relocate = req.willing_to_relocate
    profile.preferred_job_titles = req.preferred_job_titles
    profile.preferred_locations = req.preferred_locations
    profile.years_of_experience = req.years_of_experience
    profile.additional_info = req.additional_info

    await db.commit()
    return {"updated": True, "profile_id": profile_id}


# ── Session Endpoints ────────────────────────────────────────────────────────

@router.post("/start")
async def start_apply_session(
    req: StartSessionRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """Start a new auto-apply session."""
    # Verify profile exists
    result = await db.execute(
        select(JobApplyProfileEntity).where(JobApplyProfileEntity.id == req.profile_id)
    )
    profile_entity = result.scalar_one_or_none()
    if not profile_entity:
        raise HTTPException(status_code=404, detail="Profile not found")

    # Verify LinkedIn credentials exist
    cred_result = await db.execute(select(LinkedInCredentialEntity).limit(1))
    cred = cred_result.scalar_one_or_none()
    if not cred:
        raise HTTPException(
            status_code=400,
            detail="LinkedIn credentials not configured. Save them first via LinkedIn page.",
        )

    # Create session
    search_criteria = {
        "keywords": req.keywords,
        "location": req.location,
        "experience_level": req.experience_level,
        "date_posted": req.date_posted,
        "easy_apply_only": req.easy_apply_only,
    }

    session = JobApplySessionEntity(
        profile_id=req.profile_id,
        status="searching",
        search_criteria_json=search_criteria,
        max_applications=req.max_applications,
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)

    session_id = session.id

    # Prepare data for background task
    email = decrypt(cred.encrypted_email)
    password = decrypt(cred.encrypted_password)
    totp = decrypt(cred.encrypted_totp_secret) if cred.encrypted_totp_secret else None

    profile_data = profile_entity.cv_profile_json
    preferences_data = {
        "expected_salary": profile_entity.expected_salary or "",
        "current_ctc": profile_entity.current_ctc or "",
        "expected_ctc": profile_entity.expected_ctc or "",
        "notice_period": profile_entity.notice_period or "Immediate",
        "work_authorization": profile_entity.work_authorization or "",
        "willing_to_relocate": profile_entity.willing_to_relocate,
        "preferred_job_titles": profile_entity.preferred_job_titles or [],
        "preferred_locations": profile_entity.preferred_locations or [],
        "years_of_experience": profile_entity.years_of_experience,
        "additional_info": profile_entity.additional_info or {},
    }
    cv_path = profile_entity.cv_file_path

    # Launch background task
    background_tasks.add_task(
        _run_apply_session,
        session_id, email, password, totp,
        profile_data, preferences_data, search_criteria, cv_path,
        req.max_applications,
    )

    logger.info(f"Started auto-apply session: {session_id}")

    return {
        "session_id": session_id,
        "status": "searching",
        "max_applications": req.max_applications,
        "search_criteria": search_criteria,
    }


async def _run_apply_session(
    session_id: str,
    email: str, password: str, totp: Optional[str],
    profile: dict, preferences: dict, criteria: dict,
    cv_path: str, max_apps: int,
):
    """Background task that runs the apply session."""
    from agents.linkedin_job_agent import search_and_apply_to_jobs

    def _update_callback(sid: str, app_result: Optional[dict], summary: dict):
        """Synchronous callback to update DB from the Playwright thread.

        Uses raw sqlite3 to avoid asyncio event loop conflicts when called
        from within run_in_executor.
        """
        import sqlite3
        from database import DB_PATH

        try:
            conn = sqlite3.connect(str(DB_PATH))
            cur = conn.cursor()

            # Update session counts
            status_val = summary.get("status")
            applied = summary.get("applied")
            skipped = summary.get("skipped")
            failed = summary.get("failed")

            updates = []
            params: list = []
            if status_val is not None:
                updates.append("status = ?")
                params.append(status_val)
            if applied is not None:
                updates.append("applied_count = ?")
                params.append(applied)
            if skipped is not None:
                updates.append("skipped_count = ?")
                params.append(skipped)
            if failed is not None:
                updates.append("failed_count = ?")
                params.append(failed)

            if updates:
                params.append(sid)
                cur.execute(
                    f"UPDATE job_apply_sessions SET {', '.join(updates)} WHERE id = ?",
                    params,
                )

            # Insert application record if provided
            if app_result and app_result.get("job_url"):
                app_id = str(uuid.uuid4())
                applied_at = (
                    datetime.now(timezone.utc).isoformat()
                    if app_result.get("status") == "applied"
                    else None
                )
                cur.execute(
                    """INSERT INTO job_applications
                       (id, session_id, job_title, company, job_url, job_location,
                        apply_type, status, questions_json, error_message, applied_at, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        app_id,
                        sid,
                        app_result.get("job_title", ""),
                        app_result.get("company", ""),
                        app_result.get("job_url", ""),
                        app_result.get("job_location", ""),
                        app_result.get("apply_type", "unknown"),
                        app_result.get("status", "failed"),
                        json.dumps(app_result.get("questions_answered", {})),
                        app_result.get("error_message"),
                        applied_at,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )

            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"DB update callback failed: {e}")

    try:
        result = await search_and_apply_to_jobs(
            email=email,
            password=password,
            totp_secret=totp,
            profile=profile,
            preferences=preferences,
            criteria=criteria,
            cv_path=cv_path,
            max_apps=max_apps,
            session_id=session_id,
            update_callback=_update_callback,
        )

        # Finalize session status
        async with async_session_maker() as db:
            sess_result = await db.execute(
                select(JobApplySessionEntity).where(JobApplySessionEntity.id == session_id)
            )
            sess = sess_result.scalar_one_or_none()
            if sess:
                if result.get("success"):
                    sess.status = "completed"
                else:
                    sess.status = "failed"
                sess.applied_count = result.get("applied", 0)
                sess.skipped_count = result.get("skipped", 0)
                sess.failed_count = result.get("failed", 0)
                await db.commit()

        logger.info(
            f"Auto-apply session {session_id} finished: "
            f"{result.get('applied', 0)} applied, "
            f"{result.get('skipped', 0)} skipped, "
            f"{result.get('failed', 0)} failed"
        )

    except Exception as e:
        logger.error(f"Auto-apply session {session_id} error: {e}")
        async with async_session_maker() as db:
            sess_result = await db.execute(
                select(JobApplySessionEntity).where(JobApplySessionEntity.id == session_id)
            )
            sess = sess_result.scalar_one_or_none()
            if sess:
                sess.status = "failed"
                await db.commit()


@router.get("/sessions")
async def list_sessions(db: AsyncSession = Depends(get_db)):
    """List all auto-apply sessions."""
    result = await db.execute(
        select(JobApplySessionEntity)
        .order_by(JobApplySessionEntity.created_at.desc())
        .limit(50)
    )
    sessions = result.scalars().all()

    return {
        "total": len(sessions),
        "sessions": [
            {
                "id": s.id,
                "profile_id": s.profile_id,
                "status": s.status,
                "search_criteria": s.search_criteria_json,
                "max_applications": s.max_applications,
                "applied_count": s.applied_count,
                "skipped_count": s.skipped_count,
                "failed_count": s.failed_count,
                "created_at": s.created_at.isoformat() if s.created_at else None,
            }
            for s in sessions
        ],
    }


@router.get("/sessions/{session_id}/status")
async def get_session_status(session_id: str, db: AsyncSession = Depends(get_db)):
    """Get detailed status of an auto-apply session including all application results."""
    result = await db.execute(
        select(JobApplySessionEntity).where(JobApplySessionEntity.id == session_id)
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Fetch applications
    apps_result = await db.execute(
        select(JobApplicationEntity)
        .where(JobApplicationEntity.session_id == session_id)
        .order_by(JobApplicationEntity.created_at.asc())
    )
    applications = apps_result.scalars().all()

    return {
        "session_id": session.id,
        "status": session.status,
        "max_applications": session.max_applications,
        "applied_count": session.applied_count,
        "skipped_count": session.skipped_count,
        "failed_count": session.failed_count,
        "search_criteria": session.search_criteria_json,
        "created_at": session.created_at.isoformat() if session.created_at else None,
        "applications": [
            {
                "id": a.id,
                "job_title": a.job_title,
                "company": a.company,
                "job_url": a.job_url,
                "job_location": a.job_location,
                "apply_type": a.apply_type,
                "status": a.status,
                "questions_answered": a.questions_json,
                "error_message": a.error_message,
                "applied_at": a.applied_at.isoformat() if a.applied_at else None,
            }
            for a in applications
        ],
    }


# ── Saved Q&A Endpoints ──────────────────────────────────────────────────────

async def _backfill_qa_from_applications(db: AsyncSession):
    """Backfill saved_qa from existing job_applications.questions_json."""
    import re

    def _normalise(text: str) -> str:
        return re.sub(r'[^a-z0-9 ]', '', text.lower()).strip()

    apps = await db.execute(
        select(JobApplicationEntity).where(
            JobApplicationEntity.questions_json.isnot(None),
            JobApplicationEntity.questions_json != "{}",
        )
    )
    rows = apps.scalars().all()
    added = 0
    for app in rows:
        try:
            qa_data = json.loads(app.questions_json) if isinstance(app.questions_json, str) else app.questions_json
        except Exception:
            continue
        if not isinstance(qa_data, dict):
            continue
        for question, answer in qa_data.items():
            if not question:
                continue
            norm = _normalise(question)
            if not norm:
                continue
            existing = await db.execute(
                select(SavedQAEntity).where(SavedQAEntity.question_normalised == norm)
            )
            if existing.scalar_one_or_none():
                continue
            entity = SavedQAEntity(
                question=question,
                question_normalised=norm,
                answer=str(answer) if answer else "",
                field_type="text",
                source="backfill",
                times_used=1,
            )
            db.add(entity)
            added += 1
    if added > 0:
        await db.commit()
        logger.info(f"Backfilled {added} Q&A pairs from past applications")


@router.get("/qa")
async def list_saved_qa(db: AsyncSession = Depends(get_db)):
    """List all saved question-answer pairs, ordered by most recently used."""
    result = await db.execute(
        select(SavedQAEntity).order_by(SavedQAEntity.updated_at.desc())
    )
    items = result.scalars().all()

    # Auto-backfill from past applications if saved_qa is empty
    if not items:
        await _backfill_qa_from_applications(db)
        result = await db.execute(
            select(SavedQAEntity).order_by(SavedQAEntity.updated_at.desc())
        )
        items = result.scalars().all()

    return {
        "total": len(items),
        "qa_pairs": [
            {
                "id": qa.id,
                "question": qa.question,
                "answer": qa.answer,
                "field_type": qa.field_type,
                "source": qa.source,
                "times_used": qa.times_used,
                "updated_at": qa.updated_at.isoformat() if qa.updated_at else None,
            }
            for qa in items
        ],
    }


@router.post("/qa/backfill")
async def backfill_qa(db: AsyncSession = Depends(get_db)):
    """Manually backfill saved_qa from existing job application history."""
    await _backfill_qa_from_applications(db)
    result = await db.execute(select(SavedQAEntity))
    count = len(result.scalars().all())
    return {"backfilled": True, "total_qa_pairs": count}


class UpdateQARequest(BaseModel):
    answer: str


@router.put("/qa/{qa_id}")
async def update_saved_qa(
    qa_id: str, req: UpdateQARequest, db: AsyncSession = Depends(get_db)
):
    """Update the answer for a saved Q&A pair."""
    result = await db.execute(
        select(SavedQAEntity).where(SavedQAEntity.id == qa_id)
    )
    qa = result.scalar_one_or_none()
    if not qa:
        raise HTTPException(status_code=404, detail="Q&A pair not found")
    qa.answer = req.answer
    qa.source = "user"
    await db.commit()
    return {"updated": True, "id": qa_id}


@router.delete("/qa/{qa_id}")
async def delete_saved_qa(qa_id: str, db: AsyncSession = Depends(get_db)):
    """Delete a saved Q&A pair."""
    result = await db.execute(
        select(SavedQAEntity).where(SavedQAEntity.id == qa_id)
    )
    qa = result.scalar_one_or_none()
    if not qa:
        raise HTTPException(status_code=404, detail="Q&A pair not found")
    await db.delete(qa)
    await db.commit()
    return {"deleted": True, "id": qa_id}
