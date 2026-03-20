"""
Google Drive → RAG Ingestion Pipeline (Pinecone edition)
=========================================================
Matches your existing ingest.py stack exactly:
  - fastembed for embeddings
  - Pinecone for vector storage
  - LangChain RecursiveCharacterTextSplitter for chunking
  - PyMuPDF for PDF parsing

Usage:
    python ingest_gdrive.py                  # ingest new/changed files only
    python ingest_gdrive.py --force          # re-ingest everything
    python ingest_gdrive.py --file-id <id>   # ingest one specific file
"""

import uuid
import io
import json
import argparse
import logging
import os
from pathlib import Path

import fitz  # PyMuPDF
from fastembed import TextEmbedding
from pinecone import Pinecone
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from config import PINECONE_API_KEY, PINECONE_INDEX, EMBEDDING_MODEL

# ---------------------------------------------------------------------------
# Config — add GDRIVE_FOLDER_ID to your .env or config.py
# ---------------------------------------------------------------------------

GDRIVE_FOLDER_ID     = os.getenv("GDRIVE_FOLDER_ID", "")
SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "service_account.json")
STATE_FILE           = ".gdrive_ingest_state.json"  # skips unchanged files on re-runs

# ---------------------------------------------------------------------------
# Initialise — same pattern as your existing ingest.py
# ---------------------------------------------------------------------------

embedder = TextEmbedding(EMBEDDING_MODEL)
pc       = Pinecone(api_key=PINECONE_API_KEY)
index    = pc.Index(PINECONE_INDEX)
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Google Drive helpers
# ---------------------------------------------------------------------------

def get_drive_service():
    scopes = ["https://www.googleapis.com/auth/drive.readonly"]
    creds  = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=scopes
    )
    return build("drive", "v3", credentials=creds)


def list_files(service, folder_id: str) -> list[dict]:
    """List all PDFs and Google Docs in a Drive folder."""
    files, page_token = [], None
    # Match both uploaded PDFs and native Google Docs
    query = (
        f"'{folder_id}' in parents and trashed=false and ("
        f"mimeType='application/pdf' or "
        f"mimeType='application/vnd.google-apps.document'"
        f")"
    )
    while True:
        resp = service.files().list(
            q=query,
            fields="nextPageToken, files(id, name, modifiedTime, md5Checksum, mimeType)",
            pageToken=page_token,
            pageSize=100,
        ).execute()
        files.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    log.info(f"Found {len(files)} files in Drive folder")
    return files


def download_file(service, file_meta: dict) -> bytes:
    """Download a file — exports Google Docs as PDF automatically."""
    file_id  = file_meta["id"]
    mimetype = file_meta.get("mimeType", "")

    if mimetype == "application/vnd.google-apps.document":
        # Export Google Doc as PDF
        request = service.files().export_media(
            fileId=file_id, mimeType="application/pdf"
        )
    else:
        # Download regular PDF
        request = service.files().get_media(fileId=file_id)

    buffer = io.BytesIO()
    dl     = MediaIoBaseDownload(buffer, request)
    done   = False
    while not done:
        _, done = dl.next_chunk()
    return buffer.getvalue()

# ---------------------------------------------------------------------------
# Reuse your existing helpers exactly as written in ingest.py
# ---------------------------------------------------------------------------

def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        doc  = fitz.open(stream=file_bytes, filetype="pdf")
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return text
    except Exception as e:
        raise ValueError(f"Failed to parse PDF: {e}")


def embed_batch(texts: list[str]) -> list[list[float]]:
    return [v.tolist() for v in embedder.embed(texts)]

# ---------------------------------------------------------------------------
# Change tracking — skip files that haven't changed in Drive
# ---------------------------------------------------------------------------

def load_state() -> dict:
    if Path(STATE_FILE).exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}


def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

# ---------------------------------------------------------------------------
# Core ingestion — mirrors your ingest_document() but pulls from Drive
# ---------------------------------------------------------------------------

def ingest_drive_file(service, file_meta: dict, force: bool = False):
    file_id   = file_meta["id"]
    filename  = file_meta["name"]
    checksum  = file_meta.get("md5Checksum", "")

    state = load_state()

    # Skip unchanged files
    if not force and state.get(file_id) == checksum:
        log.info(f"  Skipping (unchanged): {filename}")
        return

    log.info(f"  Ingesting: {filename}")

    file_bytes = download_file(service, file_meta)
    text       = extract_text_from_pdf(file_bytes)

    if not text.strip():
        log.warning(f"  Empty or unreadable: {filename} — skipping")
        return

    # Same splitter settings as your ingest.py
    chunks = splitter.split_text(text)
    if not chunks:
        log.warning(f"  No chunks produced from {filename} — skipping")
        return

    log.info(f"  {len(chunks)} chunks")

    # Same pattern as your ingest_document()
    author_name = filename.rsplit(".", 1)[0]
    doc_id      = str(uuid.uuid4())

    all_embeddings = []
    for i in range(0, len(chunks), 100):
        all_embeddings.extend(embed_batch(chunks[i : i + 100]))

    # Same vector shape as your existing Pinecone vectors
    # Added source + file_id fields so you can filter by origin later
    vectors = [
        {
            "id": f"{doc_id}_{i}",
            "values": embedding,
            "metadata": {
                "doc_id":      doc_id,
                "author_name": author_name,
                "text":        chunk,
                "source":      "google_drive",
                "file_id":     file_id,
                "filename":    filename,
            },
        }
        for i, (chunk, embedding) in enumerate(zip(chunks, all_embeddings))
    ]

    # Same upsert logic as your ingest.py
    try:
        for i in range(0, len(vectors), 100):
            index.upsert(vectors=vectors[i : i + 100])
    except Exception as e:
        raise RuntimeError(f"Pinecone upsert error: {e}")

    log.info(f"  ✓ {len(chunks)} chunks uploaded for '{filename}'")

    state[file_id] = checksum
    save_state(state)

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run(force: bool = False, single_file_id: str = None):
    if not GDRIVE_FOLDER_ID and not single_file_id:
        raise ValueError("Set the GDRIVE_FOLDER_ID environment variable")

    service = get_drive_service()

    if single_file_id:
        meta = service.files().get(
            fileId=single_file_id,
            fields="id, name, modifiedTime, md5Checksum, mimeType"
        ).execute()
        ingest_drive_file(service, meta, force=True)
    else:
        for file_meta in list_files(service, GDRIVE_FOLDER_ID):
            ingest_drive_file(service, file_meta, force=force)

    log.info("✓ Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest Google Drive PDFs into Pinecone RAG")
    parser.add_argument("--file-id", help="Ingest a single Drive file by ID")
    parser.add_argument("--force",   action="store_true", help="Re-ingest all files")
    args = parser.parse_args()
    run(force=args.force, single_file_id=args.file_id)