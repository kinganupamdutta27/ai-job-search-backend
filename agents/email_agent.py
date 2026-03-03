"""LangGraph Node — Email Generation Agent.

Generates personalized job application emails based on the CV profile,
job listing, and HR contact information. Dynamically adjusts content
based on the job role.
"""

from __future__ import annotations

import json
import logging
import uuid

from langchain_core.messages import SystemMessage, HumanMessage

from graph.state import GraphState
from graph.models import CVProfile, JobListing, DraftEmail, EmailStatus, HRContact
from services.email_templates import (
    get_role_template,
    render_email_html,
    render_email_text,
    get_template,
)
from services.llm_utils import clean_llm_json, create_llm
from services.prompt_builder import build_job_email_prompt, build_proactive_email_prompt

logger = logging.getLogger(__name__)


def _get_job_email_prompt(
    profile: CVProfile,
    listing: JobListing,
    contact_name: str,
    role_emp: str,
    tone: str,
) -> SystemMessage:
    """Generate prompt for a job-specific email using the prompt builder."""
    prompt_str = build_job_email_prompt(
        profile=profile,
        listing=listing,
        contact=HRContact(name=contact_name, email=""),
        tone=tone,
        role_emphasis=role_emp,
    )
    return SystemMessage(content=prompt_str)


def _get_proactive_email_prompt(
    profile: CVProfile,
    company: str,
    contact_name: str,
    tone: str,
) -> SystemMessage:
    """Generate prompt for a proactive cold email using the prompt builder."""
    prompt_str = build_proactive_email_prompt(
        profile=profile,
        company=company,
        contact=HRContact(name=contact_name, email=""),
        tone=tone,
    )
    return SystemMessage(content=prompt_str)


# ── Internal Generation Handlers ─────────────────────────────────────────────

from services.retry import retry

@retry(max_attempts=2, backoff_factor=1.5)
async def _generate_email_with_ai(
    llm,
    prompt: SystemMessage,
    profile: CVProfile,
    listing: JobListing,
    contact: HRContact,
    custom_template: dict,
) -> dict:
    """Generates a single email via AI with retry logic, returning DraftEmail dump."""
    messages = [
        prompt,
        HumanMessage(
            content="Generate the personalized email now. Remember to make it specific."
        ),
    ]
    response = await llm.ainvoke(messages)
    content = clean_llm_json(response.content)
    email_data = json.loads(content)

    body_html = render_email_html(
        greeting=email_data["greeting"],
        body=email_data["body"],
        closing=email_data["closing"],
        sender_name=profile.name,
        sender_email=profile.email,
        sender_phone=profile.phone,
        skills_highlight=email_data.get("skills_highlight", ""),
        custom_template=custom_template,
    )
    body_text = render_email_text(
        greeting=email_data["greeting"],
        body=email_data["body"],
        closing=email_data["closing"],
        sender_name=profile.name,
        sender_email=profile.email,
        sender_phone=profile.phone,
        skills_highlight=email_data.get("skills_highlight", ""),
    )

    return DraftEmail(
        id=str(uuid.uuid4()),
        to_email=contact.email,
        to_name=contact.name,
        subject=email_data["subject"],
        body_html=body_html,
        body_text=body_text,
        job_title=listing.title,
        company=listing.company,
        job_url=listing.url,
        status=EmailStatus.PENDING_REVIEW,
    ).model_dump()


def _generate_fallback_email(
    profile: CVProfile,
    listing: JobListing,
    contact: HRContact,
    custom_template: dict,
) -> dict:
    """Generates a template-based fallback email if AI fails."""
    logger.warning(f"Using template fallback for {contact.email}")
    
    greeting = f"Dear {contact.name}," if contact.name else "Dear Hiring Team,"
    skills = ", ".join(profile.skills[:5])
    
    body = (
        f"I am writing to express my interest in the {listing.title} position at {listing.company}. "
        f"With my background matching your requirements, particularly in {skills}, "
        "I believe I would be a valuable addition to your team."
    )
    closing = "I have attached my resume for your review. Looking forward to hearing from you."
    
    body_html = render_email_html(
        greeting=greeting,
        body=body,
        closing=closing,
        sender_name=profile.name,
        sender_email=profile.email,
        sender_phone=profile.phone,
        skills_highlight=skills,
        custom_template=custom_template,
    )
    body_text = render_email_text(
        greeting=greeting,
        body=body,
        closing=closing,
        sender_name=profile.name,
        sender_email=profile.email,
        sender_phone=profile.phone,
        skills_highlight=skills,
    )

    return DraftEmail(
        id=str(uuid.uuid4()),
        to_email=contact.email,
        to_name=contact.name,
        subject=f"Application for {listing.title} - {profile.name}",
        body_html=body_html,
        body_text=body_text,
        job_title=listing.title,
        company=listing.company,
        job_url=listing.url,
        status=EmailStatus.PENDING_REVIEW,
    ).model_dump()


async def generate_emails(state: GraphState) -> dict:
    """
    LangGraph node: Generate personalized emails for each job+contact pair.

    Reads: state["cv_profile"], state["enriched_listings"], state["base_template"]
    Writes: state["draft_emails"], state["current_step"]
    """
    logger.info("✉️ Starting email generation...")

    cv_data = state.get("cv_profile")
    listings_data = state.get("enriched_listings", [])

    if not cv_data:
        return {"errors": ["No CV profile for email generation"], "current_step": "failed"}
    if not listings_data:
        return {"errors": ["No job listings for email generation"], "current_step": "failed"}

    profile = CVProfile(**cv_data)
    custom_template = state.get("base_template") or get_template("default")

    llm = create_llm(temperature=0.7)  # Slightly creative for emails

    draft_emails: list[dict] = []

    for listing_data in listings_data:
        listing = JobListing(**listing_data)

        # Fallback: if no HR contacts found, generate a generic one
        contacts = listing.hr_contacts
        if not contacts:
            from graph.models import HRContact
            company_slug = listing.company.lower().replace(" ", "").replace(",", "")[:20]
            fallback_email = f"careers@{company_slug}.com"
            contacts = [HRContact(name="Hiring Manager", email=fallback_email, source="fallback")]
            logger.info(f"  📌 No contacts for {listing.title} — using fallback: {fallback_email}")

        # Get role-specific template guidance
        role_guidance = get_role_template(listing.title)

        # Get most recent experience
        recent_role = "Not specified"
        if profile.experience:
            exp = profile.experience[0]
            recent_role = f"{exp.title} at {exp.company} ({exp.duration})"

        # Generate email for each contact
        for contact in contacts:
            # Decide mode: proactive vs job-specific
            is_proactive = contact.source in ("fallback", "pattern")

            if is_proactive:
                prompt = _get_proactive_email_prompt(
                    profile=profile,
                    company=listing.company,
                    contact_name=contact.name,
                    tone=role_guidance["tone"],
                )
            else:
                prompt = _get_job_email_prompt(
                    profile=profile,
                    listing=listing,
                    contact_name=contact.name,
                    role_emp=role_guidance["emphasis"],
                    tone=role_guidance["tone"],
                )

            try:
                draft_dump = await _generate_email_with_ai(
                    llm,
                    prompt,
                    profile,
                    listing,
                    contact,
                    custom_template,
                )
                draft_emails.append(draft_dump)
                logger.info(
                    f"  ✅ Generated email for {contact.email} "
                    f"({listing.title} @ {listing.company})"
                )
            except Exception as e:
                logger.error(f"  ❌ AI Email generation failed for {listing.title}: {e}")
                # Fallback
                draft_dump = _generate_fallback_email(
                    profile, listing, contact, custom_template
                )
                draft_emails.append(draft_dump)

    logger.info(f"✅ Generated {len(draft_emails)} draft emails")

    return {
        "draft_emails": draft_emails,
        "current_step": "emails_generated",
    }
