"""Structured application exceptions with error codes and HTTP status mapping."""

from __future__ import annotations


class AppError(Exception):
    """Base application error with error code and HTTP status."""

    def __init__(
        self,
        code: str = "internal_error",
        detail: str = "An unexpected error occurred",
        status_code: int = 500,
    ) -> None:
        self.code = code
        self.detail = detail
        self.status_code = status_code
        super().__init__(detail)


# ── CV / Resume Errors ───────────────────────────────────────────────────────


class CVParseError(AppError):
    """Raised when CV file parsing fails."""

    def __init__(self, detail: str = "Failed to parse CV file") -> None:
        super().__init__(code="cv_parse_failed", detail=detail, status_code=400)


class CVAnalysisError(AppError):
    """Raised when AI CV analysis fails."""

    def __init__(self, detail: str = "CV analysis failed") -> None:
        super().__init__(code="cv_analysis_failed", detail=detail, status_code=500)


# ── Search Errors ────────────────────────────────────────────────────────────


class SearchError(AppError):
    """Raised when job search fails across all engines."""

    def __init__(self, detail: str = "Job search failed") -> None:
        super().__init__(code="search_failed", detail=detail, status_code=502)


class SearchAPINotConfigured(AppError):
    """Raised when no search API keys are configured."""

    def __init__(self) -> None:
        super().__init__(
            code="search_not_configured",
            detail="No search API configured. Set TAVILY_API_KEY and/or SERP_API_KEY.",
            status_code=503,
        )


# ── Contact Extraction Errors ────────────────────────────────────────────────


class ContactExtractionError(AppError):
    """Raised when HR contact extraction fails."""

    def __init__(self, detail: str = "Contact extraction failed") -> None:
        super().__init__(code="contact_extraction_failed", detail=detail, status_code=500)


# ── Email Errors ─────────────────────────────────────────────────────────────


class EmailGenerationError(AppError):
    """Raised when AI email generation fails."""

    def __init__(self, detail: str = "Email generation failed") -> None:
        super().__init__(code="email_generation_failed", detail=detail, status_code=500)


class SMTPError(AppError):
    """Raised when email sending via SMTP fails."""

    def __init__(self, detail: str = "SMTP email send failed") -> None:
        super().__init__(code="smtp_failed", detail=detail, status_code=502)


class SMTPNotConfigured(AppError):
    """Raised when SMTP credentials are not configured."""

    def __init__(self) -> None:
        super().__init__(
            code="smtp_not_configured",
            detail="SMTP credentials not configured. Set SMTP_EMAIL and SMTP_PASSWORD.",
            status_code=503,
        )


# ── Credential Errors ────────────────────────────────────────────────────────


class CredentialError(AppError):
    """Raised when credential storage/retrieval fails."""

    def __init__(self, detail: str = "Credential operation failed") -> None:
        super().__init__(code="credential_error", detail=detail, status_code=500)


# ── Workflow Errors ──────────────────────────────────────────────────────────


class WorkflowNotFound(AppError):
    """Raised when a workflow run is not found."""

    def __init__(self, run_id: str) -> None:
        super().__init__(
            code="workflow_not_found",
            detail=f"Workflow run not found: {run_id}",
            status_code=404,
        )


class WorkflowStateError(AppError):
    """Raised when a workflow is in an invalid state for the requested operation."""

    def __init__(self, detail: str) -> None:
        super().__init__(code="workflow_state_error", detail=detail, status_code=400)
