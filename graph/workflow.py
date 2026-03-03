"""LangGraph Workflow — Main orchestration graph.

Defines the multi-agent pipeline:
  START → analyze_cv → search_jobs → extract_contacts → generate_emails
        → human_review (interrupt) → send_emails → END
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Literal

import aiosmtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from config import get_settings
from graph.state import GraphState
from graph.models import DraftEmail, EmailStatus, SendResult
from agents.cv_agent import analyze_cv
from agents.search_agent import search_jobs
from agents.hr_agent import extract_contacts
from agents.email_agent import generate_emails

logger = logging.getLogger(__name__)


# ── Human Review Node ────────────────────────────────────────────────────────

async def human_review(state: GraphState) -> dict:
    """
    LangGraph node: Human-in-the-loop review point.

    This node is an INTERRUPT point. The workflow will pause here and
    wait for the user to approve/reject emails via the API.

    When the user submits their review decisions, the workflow resumes
    with the approved_emails populated.
    """
    logger.info("⏸️ Workflow paused — awaiting human review of draft emails")

    draft_emails = state.get("draft_emails", [])
    approved = state.get("approved_emails", [])
    rejected_ids = state.get("rejected_email_ids", [])

    # If already reviewed (resumed after interrupt), filter accordingly
    if approved or rejected_ids:
        logger.info(
            f"  ✅ Review complete: {len(approved)} approved, "
            f"{len(rejected_ids)} rejected"
        )
        return {
            "current_step": "review_complete",
        }

    # Default: mark all as pending review (workflow will interrupt here)
    return {
        "approved_emails": [],
        "current_step": "awaiting_review",
    }


# ── Email Send Node ──────────────────────────────────────────────────────────

async def send_emails(state: GraphState) -> dict:
    """
    LangGraph node: Send approved emails via SMTP.

    Reads: state["approved_emails"], state["cv_file_path"]
    Writes: state["sent_results"], state["current_step"]
    """
    logger.info("📤 Sending approved emails...")
    settings = get_settings()

    approved = state.get("approved_emails", [])
    cv_path = state.get("cv_file_path", "")

    if not approved:
        logger.info("  ℹ️ No approved emails to send")
        return {
            "sent_results": [],
            "current_step": "completed",
        }

    if not settings.smtp_email or not settings.smtp_password:
        return {
            "errors": ["SMTP credentials not configured"],
            "current_step": "failed",
        }

    results: list[dict] = []

    for email_data in approved:
        draft = DraftEmail(**email_data)
        logger.info(f"  📤 Sending to {draft.to_email}...")

        try:
            msg = MIMEMultipart("alternative")
            msg["From"] = settings.smtp_email
            msg["To"] = draft.to_email
            msg["Subject"] = draft.subject

            msg.attach(MIMEText(draft.body_text, "plain", "utf-8"))
            msg.attach(MIMEText(draft.body_html, "html", "utf-8"))

            # Attach CV if available
            if cv_path and os.path.exists(cv_path):
                with open(cv_path, "rb") as f:
                    attachment = MIMEApplication(f.read())
                    filename = os.path.basename(cv_path)
                    attachment.add_header(
                        "Content-Disposition", "attachment", filename=filename
                    )
                    msg.attach(attachment)

            await aiosmtplib.send(
                msg,
                hostname=settings.smtp_host,
                port=settings.smtp_port,
                username=settings.smtp_email,
                password=settings.smtp_password,
                start_tls=True,
            )

            result = SendResult(
                email_id=draft.id,
                success=True,
                message_id=msg.get("Message-ID", ""),
            )
            results.append(result.model_dump())
            logger.info(f"  ✅ Sent to {draft.to_email}")

        except Exception as e:
            result = SendResult(
                email_id=draft.id,
                success=False,
                error=str(e),
            )
            results.append(result.model_dump())
            logger.error(f"  ❌ Failed to send to {draft.to_email}: {e}")

    sent_count = sum(1 for r in results if r["success"])
    logger.info(f"✅ Email sending complete: {sent_count}/{len(results)} sent")

    return {
        "sent_results": results,
        "current_step": "completed",
    }


# ── Conditional Routing ──────────────────────────────────────────────────────

def route_after_review(state: GraphState) -> Literal["send_emails", "__end__"]:
    """Route after human review: send approved or end if none approved."""
    approved = state.get("approved_emails", [])
    if approved:
        return "send_emails"
    return "__end__"


# ── Build the Graph ──────────────────────────────────────────────────────────

def build_workflow() -> StateGraph:
    """
    Build and compile the LangGraph workflow.

    Returns a compiled graph with an in-memory checkpointer
    for state persistence across the human-in-the-loop interrupt.
    """
    graph = StateGraph(GraphState)

    # Add nodes
    graph.add_node("analyze_cv", analyze_cv)
    graph.add_node("search_jobs", search_jobs)
    graph.add_node("extract_contacts", extract_contacts)
    graph.add_node("generate_emails", generate_emails)
    graph.add_node("human_review", human_review)
    graph.add_node("send_emails", send_emails)

    # Define edges (linear pipeline with conditional routing after review)
    graph.add_edge(START, "analyze_cv")
    graph.add_edge("analyze_cv", "search_jobs")
    graph.add_edge("search_jobs", "extract_contacts")
    graph.add_edge("extract_contacts", "generate_emails")
    graph.add_edge("generate_emails", "human_review")
    graph.add_conditional_edges(
        "human_review",
        route_after_review,
        {
            "send_emails": "send_emails",
            "__end__": END,
        },
    )
    graph.add_edge("send_emails", END)

    # Compile with memory checkpointer for HITL interrupt support
    memory = MemorySaver()
    return graph.compile(
        checkpointer=memory,
        interrupt_before=["human_review"],  # Pause before human review
    )


# ── Singleton Workflow Instance ──────────────────────────────────────────────

_workflow = None


def get_workflow():
    """Get or create the singleton workflow instance."""
    global _workflow
    if _workflow is None:
        _workflow = build_workflow()
    return _workflow
