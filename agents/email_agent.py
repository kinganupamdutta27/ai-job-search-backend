"""LangGraph Node — Email Generation Agent.

Generates personalized job application emails based on the CV profile,
job listing, and HR contact information. Dynamically adjusts content
based on the job role.
"""

from __future__ import annotations

import json
import logging
import uuid

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from config import get_settings
from graph.state import GraphState
from graph.models import CVProfile, JobListing, DraftEmail, EmailStatus
from services.email_templates import (
    get_role_template,
    render_email_html,
    render_email_text,
    get_template,
)

logger = logging.getLogger(__name__)

EMAIL_GENERATION_PROMPT = """You are an expert at writing professional job application emails.

Write a personalized email for the following job application. The email should be:
- Professional but warm and personable
- Tailored to the specific job role and company
- Highlighting relevant skills from the candidate's profile
- {tone}
- Concise (max 200 words for body)

Candidate Profile:
- Name: {name}
- Skills: {skills}
- Recent Role: {recent_role}
- Years of Experience: {years_exp}
- Summary: {summary}

Job Details:
- Title: {job_title}
- Company: {company}
- Description: {job_snippet}

HR Contact Name: {hr_name}

Role emphasis areas: {emphasis}

Return a JSON object with:
{{
  "subject": "Compelling email subject line",
  "greeting": "Dear [Name]," or "Dear Hiring Manager,",
  "body": "Main email body (highlight 2-3 relevant skills/experiences)",
  "skills_highlight": "Comma-separated list of 4-5 most relevant skills for this role",
  "closing": "Professional closing line (e.g., 'Looking forward to discussing...')"
}}

Return ONLY the JSON object."""


async def generate_emails(state: GraphState) -> dict:
    """
    LangGraph node: Generate personalized emails for each job+contact pair.

    Reads: state["cv_profile"], state["enriched_listings"], state["base_template"]
    Writes: state["draft_emails"], state["current_step"]
    """
    logger.info("✉️ Starting email generation...")
    settings = get_settings()

    cv_data = state.get("cv_profile")
    listings_data = state.get("enriched_listings", [])

    if not cv_data:
        return {"errors": ["No CV profile for email generation"], "current_step": "failed"}
    if not listings_data:
        return {"errors": ["No job listings for email generation"], "current_step": "failed"}

    profile = CVProfile(**cv_data)
    custom_template = state.get("base_template") or get_template("default")

    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0.7,  # Slightly creative for emails
    )

    draft_emails: list[dict] = []

    for listing_data in listings_data:
        listing = JobListing(**listing_data)

        if not listing.hr_contacts:
            logger.info(f"  ⏭️ Skipping {listing.title} — no contacts")
            continue

        # Get role-specific template guidance
        role_guidance = get_role_template(listing.title)

        # Get most recent experience
        recent_role = "Not specified"
        if profile.experience:
            exp = profile.experience[0]
            recent_role = f"{exp.title} at {exp.company} ({exp.duration})"

        # Generate email for each contact
        for contact in listing.hr_contacts:
            prompt = EMAIL_GENERATION_PROMPT.format(
                tone=role_guidance["tone"],
                name=profile.name,
                skills=", ".join(profile.skills[:10]),
                recent_role=recent_role,
                years_exp=profile.years_of_experience,
                summary=profile.summary,
                job_title=listing.title,
                company=listing.company,
                job_snippet=listing.description_snippet[:300],
                hr_name=contact.name,
                emphasis=role_guidance["emphasis"],
            )

            try:
                messages = [
                    SystemMessage(content=prompt),
                    HumanMessage(
                        content=(
                            "Generate the personalized email now. "
                            "Remember to make it specific to the role and company."
                        )
                    ),
                ]
                response = await llm.ainvoke(messages)
                content = response.content.strip()

                # Clean JSON
                if content.startswith("```"):
                    content = content.split("\n", 1)[1]
                    if content.endswith("```"):
                        content = content[:-3]
                    content = content.strip()

                email_data = json.loads(content)

                # Render HTML and text versions
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

                draft = DraftEmail(
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
                )
                draft_emails.append(draft.model_dump())

                logger.info(
                    f"  ✅ Generated email for {contact.email} "
                    f"({listing.title} @ {listing.company})"
                )

            except Exception as e:
                logger.warning(
                    f"  ⚠️ Email generation failed for {listing.title}: {e}"
                )
                continue

    logger.info(f"✅ Generated {len(draft_emails)} draft emails")

    return {
        "draft_emails": draft_emails,
        "current_step": "emails_generated",
    }
