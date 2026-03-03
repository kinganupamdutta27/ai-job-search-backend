"""API routes for workflow execution and management.

Persists workflow runs in async SQLite via SQLAlchemy 2.0.
"""

from __future__ import annotations

import asyncio
import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from config import get_settings
from database import get_db
from graph.workflow import get_workflow
from graph.models import EmailStatus
from models.db_models import WorkflowRunEntity
from services.cv_parser import parse_cv
from services.exceptions import CVParseError, WorkflowNotFound, WorkflowStateError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/workflow", tags=["Workflow"])


# ── Request / Response Models ────────────────────────────────────────────────


class WorkflowStartRequest(BaseModel):
    """Request body for starting a workflow."""

    cv_file_path: str = Field(description="Path to the uploaded CV file")
    search_location: str = Field(default="India", description="Job search location")
    max_jobs: int = Field(default=20, description="Maximum number of jobs to search")
    base_template: Optional[str] = Field(
        default=None, description="Custom email template (HTML)"
    )


class ReviewDecision(BaseModel):
    """Review decision for a single email draft."""

    email_id: str
    approved: bool


class ReviewRequest(BaseModel):
    """Request body for submitting review decisions."""

    decisions: list[ReviewDecision]


# ── Helper ───────────────────────────────────────────────────────────────────


async def _get_run(
    db: AsyncSession, run_id: str
) -> WorkflowRunEntity:
    """Fetch a run or raise structured 404."""
    result = await db.execute(
        select(WorkflowRunEntity).where(WorkflowRunEntity.id == run_id)
    )
    run = result.scalar_one_or_none()
    if run is None:
        raise WorkflowNotFound(run_id)
    return run


# ── Routes ───────────────────────────────────────────────────────────────────


@router.post("/start")
async def start_workflow(
    request: WorkflowStartRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Start the full AI workflow pipeline.

    The workflow will:
    1. Parse and analyze the CV
    2. Search for relevant jobs
    3. Extract HR contacts
    4. Generate personalized emails
    5. Pause for human review (HITL interrupt)

    Returns run_id for tracking.
    """
    settings = get_settings()
    run_id = str(uuid.uuid4())

    # Parse CV text
    try:
        cv_text = parse_cv(request.cv_file_path)
    except Exception as e:
        raise CVParseError(detail=f"Failed to parse CV: {str(e)}")

    # Prepare initial state
    initial_state = {
        "cv_file_path": request.cv_file_path,
        "cv_text": cv_text,
        "search_location": request.search_location,
        "max_jobs": request.max_jobs,
        "base_template": request.base_template or "",
        "run_id": run_id,
        "current_step": "starting",
        "errors": [],
        "job_listings": [],
    }

    config = {"configurable": {"thread_id": run_id}}

    # Persist initial run to DB
    entity = WorkflowRunEntity(
        id=run_id,
        status="running",
        state_json=initial_state,
        config_json=config,
    )
    db.add(entity)
    await db.flush()

    logger.info(f"🚀 Starting workflow run: {run_id}")

    # Execute the workflow (will pause at human_review interrupt)
    workflow = get_workflow()
    try:
        async for event in workflow.astream(initial_state, config=config):
            if isinstance(event, dict):
                for key, value in event.items():
                    if isinstance(value, dict):
                        entity.state_json = {**entity.state_json, **value}

        # Get the final state after interrupt
        state_snapshot = workflow.get_state(config)
        if state_snapshot and state_snapshot.values:
            entity.state_json = {**entity.state_json, **state_snapshot.values}

        current_step = entity.state_json.get("current_step", "unknown")

        if current_step == "awaiting_review" or state_snapshot.next:
            entity.status = "awaiting_review"
        elif "failed" in current_step:
            entity.status = "failed"
        else:
            entity.status = "completed"

    except Exception as e:
        logger.error(f"❌ Workflow failed: {e}")
        entity.status = "failed"
        entity.state_json = {**entity.state_json, "errors": [str(e)]}

    await db.flush()

    return {
        "run_id": run_id,
        "status": entity.status,
        "current_step": entity.state_json.get("current_step"),
        "jobs_found": len(entity.state_json.get("job_listings", [])),
        "emails_generated": len(entity.state_json.get("draft_emails", [])),
        "errors": entity.state_json.get("errors", []),
    }


@router.get("/{run_id}/status")
async def get_workflow_status(
    run_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get the current status of a workflow run."""
    run = await _get_run(db, run_id)
    state = run.state_json

    return {
        "run_id": run_id,
        "status": run.status,
        "current_step": state.get("current_step"),
        "cv_profile": state.get("cv_profile"),
        "jobs_found": len(state.get("job_listings", [])),
        "job_listings": state.get("enriched_listings") or state.get("job_listings", []),
        "emails_generated": len(state.get("draft_emails", [])),
        "draft_emails": state.get("draft_emails", []),
        "sent_results": state.get("sent_results", []),
        "errors": state.get("errors", []),
    }


@router.post("/{run_id}/review")
async def submit_review(
    run_id: str,
    request: ReviewRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Submit human review decisions for draft emails.

    Each email can be approved or rejected. After review,
    the workflow resumes and sends approved emails.
    """
    run = await _get_run(db, run_id)
    if run.status != "awaiting_review":
        raise WorkflowStateError(
            detail=f"Workflow is not awaiting review (current status: {run.status})"
        )

    state = dict(run.state_json)

    # Process decisions
    draft_emails = state.get("draft_emails", [])
    approved_emails = []
    rejected_ids = []

    decision_map = {d.email_id: d.approved for d in request.decisions}

    for email in draft_emails:
        email_id = email.get("id", "")
        if decision_map.get(email_id, False):
            email["status"] = EmailStatus.APPROVED.value
            approved_emails.append(email)
        else:
            email["status"] = EmailStatus.REJECTED.value
            rejected_ids.append(email_id)

    # Update state
    state["approved_emails"] = approved_emails
    state["rejected_email_ids"] = rejected_ids
    run.state_json = state
    run.status = "sending"

    logger.info(
        f"📝 Review submitted: {len(approved_emails)} approved, "
        f"{len(rejected_ids)} rejected"
    )

    # Try to resume the LangGraph workflow, fall back to direct send
    workflow = get_workflow()
    config = run.config_json
    resumed_via_langgraph = False

    try:
        # Check if LangGraph still has a valid checkpoint for this run
        checkpoint_state = workflow.get_state(config)
        if checkpoint_state and checkpoint_state.next:
            logger.info("🔄 Resuming workflow via LangGraph checkpoint...")

            # CRITICAL: as_node tells LangGraph which node produced this
            # state update, so it can correctly route to the next node.
            await workflow.aupdate_state(
                config,
                {
                    "approved_emails": approved_emails,
                    "rejected_email_ids": rejected_ids,
                    "current_step": "review_complete",
                },
                as_node="human_review",
            )

            # Resume with a timeout to prevent indefinite hanging
            async def _resume():
                async for event in workflow.astream(None, config=config):
                    if isinstance(event, dict):
                        for key, value in event.items():
                            if isinstance(value, dict):
                                state.update(value)

            await asyncio.wait_for(_resume(), timeout=120)
            resumed_via_langgraph = True
            run.state_json = state
            run.status = "completed"
            logger.info("✅ Workflow resumed and completed via LangGraph")

        else:
            logger.warning(
                "⚠️ No valid LangGraph checkpoint found (server may have "
                "restarted). Falling back to direct email sending."
            )

    except asyncio.TimeoutError:
        logger.warning("⚠️ LangGraph resume timed out, falling back to direct send")
    except Exception as e:
        logger.warning(
            f"⚠️ LangGraph resume failed: {e}. Falling back to direct send."
        )

    # Fallback: send emails directly without LangGraph
    if not resumed_via_langgraph:
        try:
            from graph.workflow import send_emails

            send_state = {**state, "approved_emails": approved_emails}
            result = await send_emails(send_state)
            state.update(result)
            run.state_json = state
            run.status = "completed"
            logger.info("✅ Emails sent via direct fallback")

        except Exception as e:
            logger.error(f"❌ Direct email send also failed: {e}")
            run.status = "failed"
            state.setdefault("errors", []).append(str(e))
            run.state_json = state

    await db.flush()

    return {
        "run_id": run_id,
        "status": run.status,
        "approved_count": len(approved_emails),
        "rejected_count": len(rejected_ids),
        "sent_results": state.get("sent_results", []),
        "errors": state.get("errors", []),
    }


@router.get("/runs")
async def list_runs(db: AsyncSession = Depends(get_db)):
    """List all workflow runs."""
    result = await db.execute(
        select(WorkflowRunEntity).order_by(WorkflowRunEntity.created_at.desc())
    )
    runs = result.scalars().all()
    return {
        "runs": [
            {
                "run_id": r.id,
                "status": r.status,
                "current_step": r.state_json.get("current_step"),
                "jobs_found": len(r.state_json.get("job_listings", [])),
                "emails_generated": len(r.state_json.get("draft_emails", [])),
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in runs
        ]
    }
