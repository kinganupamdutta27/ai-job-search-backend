"""API routes for workflow execution and management."""

from __future__ import annotations

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from config import get_settings
from graph.workflow import get_workflow
from graph.models import WorkflowRun, WorkflowStep, EmailStatus
from services.cv_parser import parse_cv

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/workflow", tags=["Workflow"])

# ── In-memory storage for workflow runs ──────────────────────────────────────
_workflow_runs: dict[str, dict] = {}


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


@router.post("/start")
async def start_workflow(request: WorkflowStartRequest):
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
        raise HTTPException(status_code=400, detail=f"Failed to parse CV: {str(e)}")

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

    # Store the run
    _workflow_runs[run_id] = {
        "run_id": run_id,
        "status": "running",
        "state": initial_state,
        "config": {"configurable": {"thread_id": run_id}},
    }

    logger.info(f"🚀 Starting workflow run: {run_id}")

    # Execute the workflow (will pause at human_review interrupt)
    workflow = get_workflow()
    try:
        config = {"configurable": {"thread_id": run_id}}
        result = None
        async for event in workflow.astream(initial_state, config=config):
            # Update stored state with latest event
            if isinstance(event, dict):
                for key, value in event.items():
                    if isinstance(value, dict):
                        _workflow_runs[run_id]["state"].update(value)

        # Get the final state after interrupt
        state_snapshot = workflow.get_state(config)
        if state_snapshot and state_snapshot.values:
            _workflow_runs[run_id]["state"].update(state_snapshot.values)

        current_step = _workflow_runs[run_id]["state"].get("current_step", "unknown")

        if current_step == "awaiting_review" or state_snapshot.next:
            _workflow_runs[run_id]["status"] = "awaiting_review"
        elif "failed" in current_step:
            _workflow_runs[run_id]["status"] = "failed"
        else:
            _workflow_runs[run_id]["status"] = "completed"

    except Exception as e:
        logger.error(f"❌ Workflow failed: {e}")
        _workflow_runs[run_id]["status"] = "failed"
        _workflow_runs[run_id]["state"]["errors"] = [str(e)]

    return {
        "run_id": run_id,
        "status": _workflow_runs[run_id]["status"],
        "current_step": _workflow_runs[run_id]["state"].get("current_step"),
        "jobs_found": len(_workflow_runs[run_id]["state"].get("job_listings", [])),
        "emails_generated": len(_workflow_runs[run_id]["state"].get("draft_emails", [])),
        "errors": _workflow_runs[run_id]["state"].get("errors", []),
    }


@router.get("/{run_id}/status")
async def get_workflow_status(run_id: str):
    """Get the current status of a workflow run."""
    if run_id not in _workflow_runs:
        raise HTTPException(status_code=404, detail=f"Workflow run not found: {run_id}")

    run = _workflow_runs[run_id]
    state = run["state"]

    return {
        "run_id": run_id,
        "status": run["status"],
        "current_step": state.get("current_step"),
        "cv_profile": state.get("cv_profile"),
        "jobs_found": len(state.get("job_listings", [])),
        "job_listings": state.get("enriched_listings") or state.get("job_listings", []),
        "draft_emails": state.get("draft_emails", []),
        "sent_results": state.get("sent_results", []),
        "errors": state.get("errors", []),
    }


@router.post("/{run_id}/review")
async def submit_review(run_id: str, request: ReviewRequest):
    """
    Submit human review decisions for draft emails.

    Each email can be approved or rejected. After review,
    the workflow resumes and sends approved emails.
    """
    if run_id not in _workflow_runs:
        raise HTTPException(status_code=404, detail=f"Workflow run not found: {run_id}")

    run = _workflow_runs[run_id]
    if run["status"] != "awaiting_review":
        raise HTTPException(
            status_code=400,
            detail=f"Workflow is not awaiting review (status: {run['status']})",
        )

    # Process decisions
    draft_emails = run["state"].get("draft_emails", [])
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

    # Update state and resume workflow
    run["state"]["approved_emails"] = approved_emails
    run["state"]["rejected_email_ids"] = rejected_ids
    run["status"] = "sending"

    logger.info(
        f"📝 Review submitted: {len(approved_emails)} approved, "
        f"{len(rejected_ids)} rejected"
    )

    # Resume the workflow from the interrupt point
    workflow = get_workflow()
    config = run["config"]

    try:
        # Update the state with review results and resume
        await workflow.aupdate_state(
            config,
            {
                "approved_emails": approved_emails,
                "rejected_email_ids": rejected_ids,
                "current_step": "review_complete",
            },
        )

        # Continue execution
        async for event in workflow.astream(None, config=config):
            if isinstance(event, dict):
                for key, value in event.items():
                    if isinstance(value, dict):
                        run["state"].update(value)

        run["status"] = "completed"

    except Exception as e:
        logger.error(f"❌ Workflow resume failed: {e}")
        run["status"] = "failed"
        run["state"]["errors"].append(str(e))

    return {
        "run_id": run_id,
        "status": run["status"],
        "approved_count": len(approved_emails),
        "rejected_count": len(rejected_ids),
        "sent_results": run["state"].get("sent_results", []),
        "errors": run["state"].get("errors", []),
    }


@router.get("/runs")
async def list_runs():
    """List all workflow runs."""
    runs = []
    for run_id, run in _workflow_runs.items():
        runs.append({
            "run_id": run_id,
            "status": run["status"],
            "current_step": run["state"].get("current_step"),
            "jobs_found": len(run["state"].get("job_listings", [])),
            "emails_generated": len(run["state"].get("draft_emails", [])),
        })
    return {"runs": runs}
