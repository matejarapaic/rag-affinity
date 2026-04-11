"""
Extracts text from file bytes and splits into chunks for upload.
Supports: PDF, DOCX, plain text, markdown, and Google Docs (exported as text).
"""

import io
import logging
from typing import List

import pypdf
import docx

import config

logger = logging.getLogger(__name__)


def extract_text(file_bytes: bytes, mime_type: str, filename: str) -> str:
    """Extract plain text from file bytes based on MIME type."""
    if mime_type == "application/pdf":
        return _extract_pdf(file_bytes)
    elif mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return _extract_docx(file_bytes)
    elif mime_type in ("text/plain", "text/markdown", "application/vnd.google-apps.document"):
        return file_bytes.decode("utf-8", errors="replace")
    else:
        logger.warning(f"Unsupported MIME type '{mime_type}' for '{filename}', skipping.")
        return ""


def _extract_pdf(file_bytes: bytes) -> str:
    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    return "\n".join(
        page.extract_text() for page in reader.pages if page.extract_text()
    )


def _extract_docx(file_bytes: bytes) -> str:
    doc = docx.Document(io.BytesIO(file_bytes))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def chunk_text(text: str) -> List[str]:
    """
    Split text using the same settings as the existing RAG ingest.py
    (RecursiveCharacterTextSplitter, 500 chars, 50 overlap) so chunks
    are consistent across both ingestion paths.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    return splitter.split_text(text)
