"""LangGraph workflow state definition."""

from __future__ import annotations

from typing import Annotated, TypedDict

from graph.models import (
    CVProfile,
    DraftEmail,
    JobListing,
    SendResult,
)


def _merge_lists(left: list, right: list) -> list:
    """Reducer that merges two lists (append, no duplicates by identity)."""
    return left + right


class GraphState(TypedDict, total=False):
    """The shared state that flows through the LangGraph pipeline."""

    # ── Inputs ──────────────────────────────────────────────────────────
    cv_file_path: str
    cv_text: str
    base_template: str
    search_location: str
    max_jobs: int

    # ── CV Analysis Output ──────────────────────────────────────────────
    cv_profile: dict  # Serialized CVProfile

    # ── Job Search Output ───────────────────────────────────────────────
    job_listings: Annotated[list[dict], _merge_lists]  # Serialized JobListing[]

    # ── HR Extraction Output ────────────────────────────────────────────
    enriched_listings: list[dict]  # JobListing[] with hr_contacts populated

    # ── Email Generation Output ─────────────────────────────────────────
    draft_emails: list[dict]  # Serialized DraftEmail[]

    # ── Human Review Output ─────────────────────────────────────────────
    approved_emails: list[dict]
    rejected_email_ids: list[str]

    # ── Send Output ─────────────────────────────────────────────────────
    sent_results: list[dict]  # Serialized SendResult[]

    # ── Metadata ────────────────────────────────────────────────────────
    current_step: str
    errors: Annotated[list[str], _merge_lists]
    run_id: str
