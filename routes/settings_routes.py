"""API routes for application settings."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/settings", tags=["Settings"])


class SettingsUpdate(BaseModel):
    """Request body for updating settings."""

    openai_api_key: str = Field(default="", description="OpenAI API key")
    openai_model: str = Field(default="gpt-4o", description="OpenAI model")
    langsmith_tracing: bool = Field(default=True)
    langsmith_endpoint: str = Field(default="https://api.smith.langchain.com")
    langsmith_api_key: str = Field(default="")
    langsmith_project: str = Field(default="JOBSEARCH")
    serp_api_key: str = Field(default="", description="SerpAPI key")
    tavily_api_key: str = Field(default="", description="Tavily API key")
    smtp_host: str = Field(default="smtp.gmail.com")
    smtp_port: int = Field(default=587)
    smtp_email: str = Field(default="")
    smtp_password: str = Field(default="", description="Gmail App Password")


def _mask_key(key: str) -> str:
    """Mask an API key for display (show first 4 and last 4 chars)."""
    if len(key) <= 8:
        return "*" * len(key) if key else ""
    return key[:4] + "*" * (len(key) - 8) + key[-4:]


@router.get("")
async def get_settings():
    """Get current settings (API keys are masked)."""
    env_path = Path(".env")
    settings = {}

    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()

                # Mask sensitive values
                if "KEY" in key or "PASSWORD" in key:
                    settings[key.lower()] = _mask_key(value)
                else:
                    settings[key.lower()] = value

    return {"settings": settings}


@router.post("")
async def update_settings(request: SettingsUpdate):
    """
    Save settings to .env file.

    This updates the environment configuration. The server may need
    to be restarted for some changes to take effect.
    """
    env_lines = [
        "# ============================================",
        "# AI Job Search Automation - Environment Config",
        "# ============================================",
        "",
        "# --- OpenAI ---",
        f"OPENAI_API_KEY={request.openai_api_key}",
        f"OPENAI_MODEL={request.openai_model}",
        "",
        "# --- LangSmith (Tracing & Observability) ---",
        f"LANGSMITH_TRACING={'true' if request.langsmith_tracing else 'false'}",
        f"LANGSMITH_ENDPOINT={request.langsmith_endpoint}",
        f"LANGSMITH_API_KEY={request.langsmith_api_key}",
        f'LANGSMITH_PROJECT={request.langsmith_project}',
        "",
        "# --- Search APIs (both used in parallel) ---",
        f"SERP_API_KEY={request.serp_api_key}",
        f"TAVILY_API_KEY={request.tavily_api_key}",
        "",
        "# --- SMTP (Gmail with App Password for 2FA) ---",
        f"SMTP_HOST={request.smtp_host}",
        f"SMTP_PORT={request.smtp_port}",
        f"SMTP_EMAIL={request.smtp_email}",
        f"SMTP_PASSWORD={request.smtp_password}",
        "",
        "# --- Storage ---",
        "UPLOAD_DIR=./uploads",
    ]

    env_path = Path(".env")
    env_path.write_text("\n".join(env_lines), encoding="utf-8")

    # Also update current process env
    os.environ["OPENAI_API_KEY"] = request.openai_api_key
    os.environ["OPENAI_MODEL"] = request.openai_model
    os.environ["SERP_API_KEY"] = request.serp_api_key
    os.environ["TAVILY_API_KEY"] = request.tavily_api_key
    os.environ["SMTP_HOST"] = request.smtp_host
    os.environ["SMTP_PORT"] = str(request.smtp_port)
    os.environ["SMTP_EMAIL"] = request.smtp_email
    os.environ["SMTP_PASSWORD"] = request.smtp_password
    os.environ["LANGSMITH_TRACING"] = str(request.langsmith_tracing).lower()
    os.environ["LANGSMITH_ENDPOINT"] = request.langsmith_endpoint
    os.environ["LANGSMITH_API_KEY"] = request.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = request.langsmith_project

    return {"status": "saved", "message": "Settings updated successfully"}


@router.get("/smtp-guide")
async def get_smtp_guide():
    """Get instructions for setting up Gmail App Password with 2FA."""
    return {
        "title": "Gmail App Password Setup (for 2-Step Verification)",
        "steps": [
            "1. Go to https://myaccount.google.com/apppasswords",
            "2. Sign in with your Google account (2FA must be enabled)",
            "3. At the bottom, under 'App passwords', click 'Select app'",
            "4. Choose 'Mail' (or type 'Job Search Automation')",
            "5. Click 'Generate'",
            "6. Copy the 16-character password shown",
            "7. Paste this password in the SMTP Password field on the Settings page",
            "8. Your regular Gmail password will NOT work with 2FA — use the App Password instead",
        ],
        "smtp_settings": {
            "host": "smtp.gmail.com",
            "port": 587,
            "email": "your-email@gmail.com",
            "password": "(your 16-char app password)",
        },
    }
