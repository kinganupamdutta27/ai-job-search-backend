"""FastAPI Application — AI Job Search & Email Outreach Automation."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from middleware.request_id import RequestIDMiddleware
from services.exceptions import AppError

from config import get_settings
from database import init_db, close_db
from routes.cv_routes import router as cv_router
from routes.workflow_routes import router as workflow_router
from routes.email_routes import router as email_router
from routes.settings_routes import router as settings_router
from routes.contacts_routes import router as contacts_router

# ── Logging Setup ────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── App Lifecycle ────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup
    load_dotenv()
    settings = get_settings()
    settings.setup_langsmith()
    settings.ensure_directories()

    # Initialize async SQLite database
    await init_db()

    logger.info("=" * 60)
    logger.info("🚀 AI Job Search Automation — Starting Up")
    logger.info(f"   OpenAI Model: {settings.openai_model}")
    logger.info(f"   LangSmith: {'enabled' if settings.langsmith_api_key else 'not configured'}")
    logger.info(f"   SerpAPI: {'configured' if settings.serp_api_key else 'not configured'}")
    logger.info(f"   Tavily: {'configured' if settings.tavily_api_key else 'not configured'}")
    logger.info(f"   SMTP: {'configured' if settings.smtp_email else 'not configured'}")
    logger.info(f"   Upload Dir: {settings.upload_dir}")
    logger.info("=" * 60)

    yield

    # Shutdown
    await close_db()
    logger.info("👋 AI Job Search Automation — Shutting Down")


# ── Create FastAPI App ───────────────────────────────────────────────────────

app = FastAPI(
    title="AI Job Search & Email Outreach Automation",
    description=(
        "An AI-powered platform that automates job searching, "
        "HR contact extraction, and personalized email outreach. "
        "Powered by LangGraph multi-agent workflows, MCP tool servers, "
        "and OpenAI."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ─────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Middleware ────────────────────────────────────────────────────────────────

app.add_middleware(RequestIDMiddleware)

# ── Global Exception Handlers ────────────────────────────────────────────────


@app.exception_handler(AppError)
async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
    """Return structured JSON for all application-level errors."""
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error(
        f"AppError [{exc.code}] (req={request_id}): {exc.detail}"
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.code,
            "detail": exc.detail,
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


# ── Mount Routers ────────────────────────────────────────────────────────────

app.include_router(cv_router)
app.include_router(workflow_router)
app.include_router(email_router)
app.include_router(settings_router)
app.include_router(contacts_router)


# ── Health Check ─────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def root():
    """Health check / root endpoint."""
    return {
        "name": "AI Job Search & Email Outreach Automation",
        "status": "running",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
async def health():
    """Detailed health check."""
    settings = get_settings()
    return {
        "status": "healthy",
        "openai_configured": bool(settings.openai_api_key),
        "serpapi_configured": bool(settings.serp_api_key),
        "tavily_configured": bool(settings.tavily_api_key),
        "smtp_configured": bool(settings.smtp_email and settings.smtp_password),
        "langsmith_configured": bool(settings.langsmith_api_key),
    }
