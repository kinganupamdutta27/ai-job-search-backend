"""API routes for CV upload and analysis."""

from __future__ import annotations

import os
import uuid
import logging

from fastapi import APIRouter, UploadFile, File, HTTPException

from config import get_settings
from services.cv_parser import parse_cv
from agents.cv_agent import analyze_cv

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/cv", tags=["CV"])


@router.post("/upload")
async def upload_cv(file: UploadFile = File(...)):
    """
    Upload a CV file (PDF, DOCX, or TXT).

    Returns:
        file_id and file path for use in workflow.
    """
    settings = get_settings()
    settings.ensure_directories()

    # Validate file type
    allowed_extensions = {".pdf", ".docx", ".txt"}
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {', '.join(allowed_extensions)}",
        )

    # Save file
    file_id = str(uuid.uuid4())
    filename = f"{file_id}{ext}"
    file_path = os.path.join(settings.upload_dir, filename)

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    logger.info(f"📁 CV uploaded: {file.filename} → {file_path}")

    # Parse CV text
    try:
        cv_text = parse_cv(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse CV: {str(e)}")

    return {
        "file_id": file_id,
        "filename": file.filename,
        "file_path": file_path,
        "text_length": len(cv_text),
        "text_preview": cv_text[:500] + ("..." if len(cv_text) > 500 else ""),
    }


@router.get("/{file_id}/text")
async def get_cv_text(file_id: str):
    """Get the raw text of an uploaded CV."""
    settings = get_settings()
    upload_dir = settings.upload_dir

    # Find the file
    for filename in os.listdir(upload_dir):
        if filename.startswith(file_id):
            file_path = os.path.join(upload_dir, filename)
            cv_text = parse_cv(file_path)
            return {"file_id": file_id, "text": cv_text}

    raise HTTPException(status_code=404, detail=f"CV not found: {file_id}")
