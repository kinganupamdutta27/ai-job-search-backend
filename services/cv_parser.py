"""CV file parser — extracts text from PDF and DOCX files."""

from __future__ import annotations

import os
from pathlib import Path


def parse_pdf(file_path: str) -> str:
    """Extract text content from a PDF file using pdfplumber."""
    import pdfplumber

    text_parts: list[str] = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n\n".join(text_parts)


def parse_docx(file_path: str) -> str:
    """Extract text content from a DOCX file using python-docx."""
    from docx import Document

    doc = Document(file_path)
    text_parts: list[str] = []
    for paragraph in doc.paragraphs:
        if paragraph.text.strip():
            text_parts.append(paragraph.text)
    return "\n\n".join(text_parts)


def parse_cv(file_path: str) -> str:
    """
    Parse a CV file and return its text content.

    Supports PDF (.pdf) and Word (.docx) formats.

    Args:
        file_path: Path to the CV file.

    Returns:
        Extracted text content.

    Raises:
        ValueError: If file format is not supported.
        FileNotFoundError: If the file does not exist.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"CV file not found: {file_path}")

    ext = path.suffix.lower()

    if ext == ".pdf":
        return parse_pdf(file_path)
    elif ext == ".docx":
        return parse_docx(file_path)
    elif ext == ".txt":
        return path.read_text(encoding="utf-8")
    else:
        raise ValueError(
            f"Unsupported CV format: {ext}. Supported: .pdf, .docx, .txt"
        )
