"""
Polls a specific Google Drive folder for new or updated files.
Uses a service account for authentication.
"""

import io
import logging
from typing import Generator

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

import config

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

# MIME type to export format for Google Workspace files
GOOGLE_DOC_EXPORT_MIME = "text/plain"


def _get_service():
    creds = service_account.Credentials.from_service_account_file(
        config.GOOGLE_SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )
    return build("drive", "v3", credentials=creds)


def list_files(folder_id: str) -> list[dict]:
    """List all supported files in the given Drive folder."""
    service = _get_service()
    query = f"'{folder_id}' in parents and trashed = false"
    results = []
    page_token = None

    while True:
        response = (
            service.files()
            .list(
                q=query,
                fields="nextPageToken, files(id, name, mimeType, modifiedTime)",
                pageToken=page_token,
            )
            .execute()
        )
        files = response.get("files", [])
        results.extend(
            f for f in files if f["mimeType"] in config.SUPPORTED_MIME_TYPES
        )
        page_token = response.get("nextPageToken")
        if not page_token:
            break

    logger.info(f"Found {len(results)} supported files in folder.")
    return results


def download_file(file: dict) -> bytes:
    """Download file content as bytes. Exports Google Docs as plain text."""
    service = _get_service()
    file_id = file["id"]
    mime_type = file["mimeType"]

    if mime_type == "application/vnd.google-apps.document":
        request = service.files().export_media(
            fileId=file_id, mimeType=GOOGLE_DOC_EXPORT_MIME
        )
    else:
        request = service.files().get_media(fileId=file_id)

    buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()

    return buffer.getvalue()
