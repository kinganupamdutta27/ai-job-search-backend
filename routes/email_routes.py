"""API routes for email template management."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.email_templates import (
    list_templates,
    get_template,
    save_template,
    BASE_EMAIL_TEMPLATE,
    ROLE_TEMPLATES,
)

router = APIRouter(prefix="/api/email", tags=["Email"])


class TemplateUpdate(BaseModel):
    """Request body for updating a template."""

    html_content: str


@router.get("/templates")
async def get_templates():
    """List all available email templates."""
    return {
        "templates": list_templates(),
        "role_templates": {
            k: v for k, v in ROLE_TEMPLATES.items()
        },
    }


@router.get("/templates/{template_id}")
async def get_template_content(template_id: str):
    """Get a specific email template by ID."""
    template = get_template(template_id)
    return {
        "template_id": template_id,
        "html_content": template,
    }


@router.put("/templates/{template_id}")
async def update_template(template_id: str, request: TemplateUpdate):
    """Update or create an email template."""
    save_template(template_id, request.html_content)
    return {
        "template_id": template_id,
        "status": "saved",
    }


@router.get("/templates/default/preview")
async def preview_default_template():
    """Get the default email template with sample data filled in."""
    from services.email_templates import render_email_html

    sample_html = render_email_html(
        greeting="Dear Hiring Manager,",
        body=(
            "I am writing to express my interest in the Python Developer "
            "position at your company. With over 5 years of experience in "
            "Python development, including expertise in FastAPI, Django, and "
            "cloud technologies, I am confident in my ability to contribute "
            "to your team."
        ),
        closing="I look forward to the opportunity to discuss how my skills can benefit your team.",
        sender_name="John Doe",
        sender_email="john.doe@email.com",
        sender_phone="+91 98765 43210",
        skills_highlight="Python, FastAPI, Django, PostgreSQL, Docker",
    )
    return {
        "preview_html": sample_html,
    }
