"""CV file parser — extracts text from PDF and DOCX files.

Supports PDF, DOCX, and TXT formats with fallback parsing:
  - Primary: pdfplumber (PDF) / python-docx (DOCX)
  - Fallback: PyPDF2 if pdfplumber returns empty text
  - Returns structured error with fallback_mode flag if all parsers fail
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


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


def _parse_pdf_fallback(file_path: str) -> str:
    """Fallback PDF parser using PyPDF2 when pdfplumber returns empty text."""
    try:
        from PyPDF2 import PdfReader

        reader = PdfReader(file_path)
        text_parts: list[str] = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text and page_text.strip():
                text_parts.append(page_text)
        return "\n\n".join(text_parts)
    except ImportError:
        logger.warning("PyPDF2 not installed — PDF fallback unavailable")
        return ""
    except Exception as e:
        logger.warning(f"PyPDF2 fallback also failed: {e}")
        return ""


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

    Supports PDF (.pdf) and Word (.docx) formats with fallback:
      1. PDF: Try pdfplumber → if empty, try PyPDF2
      2. DOCX: python-docx
      3. TXT: direct read

    Args:
        file_path: Path to the CV file.

    Returns:
        Extracted text content.

    Raises:
        ValueError: If file format is not supported or all parsers return empty.
        FileNotFoundError: If the file does not exist.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"CV file not found: {file_path}")

    ext = path.suffix.lower()

    if ext == ".pdf":
        text = parse_pdf(file_path)
        if not text.strip():
            logger.warning(
                "⚠️ pdfplumber returned empty text, trying PyPDF2 fallback..."
            )
            text = _parse_pdf_fallback(file_path)
        if not text.strip():
            raise ValueError(
                "PDF parsing returned empty text from all parsers. "
                "The PDF may be image-based (scanned). "
                "Please upload a text-based PDF or DOCX file."
            )
        return text

    elif ext == ".docx":
        text = parse_docx(file_path)
        if not text.strip():
            raise ValueError(
                "DOCX parsing returned empty text. "
                "The file may be corrupted or contain only images."
            )
        return text

    elif ext == ".txt":
        return path.read_text(encoding="utf-8")

    else:
        raise ValueError(
            f"Unsupported CV format: {ext}. Supported: .pdf, .docx, .txt"
        )
