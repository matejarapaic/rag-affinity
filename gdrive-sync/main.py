"""
Entry point. Polls a Google Drive folder on a schedule, extracts text
from new/updated files, chunks it, and upserts to a Pinecone index
with integrated embedding (llama-text-embed-v2).

Usage:
    python main.py
"""

import logging
import time

import schedule

import config
import gdrive_poller
import document_processor
import pinecone_uploader
import tracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def sync():
    """Check for new/updated files and upload chunks to Pinecone."""
    logger.info("Starting sync...")

    try:
        files = gdrive_poller.list_files(config.GOOGLE_DRIVE_FOLDER_ID)
    except Exception as e:
        logger.error(f"Failed to list Drive files: {e}")
        return

    new_count = 0
    for file in files:
        file_id = file["id"]
        filename = file["name"]
        mime_type = file["mimeType"]
        modified_time = file["modifiedTime"]

        if tracker.is_processed(file_id, modified_time):
            logger.debug(f"Skipping already-processed file: {filename}")
            continue

        logger.info(f"New/updated file: {filename}")

        try:
            file_bytes = gdrive_poller.download_file(file)
            text = document_processor.extract_text(file_bytes, mime_type, filename)

            if not text.strip():
                logger.warning(f"No text extracted from '{filename}', skipping.")
                tracker.mark_processed(file_id, modified_time)
                continue

            chunks = document_processor.chunk_text(text)
            logger.info(f"  -> {len(chunks)} chunks from '{filename}'")

            pinecone_uploader.upload_chunks(chunks, file_id, filename)
            tracker.mark_processed(file_id, modified_time)
            new_count += 1

        except Exception as e:
            logger.error(f"Error processing '{filename}': {e}")

    logger.info(f"Sync complete. {new_count} file(s) uploaded.")


def main():
    logger.info(
        f"Pipeline started. Polling every {config.POLL_INTERVAL_SECONDS}s. "
        f"Watching folder: {config.GOOGLE_DRIVE_FOLDER_ID}"
    )

    sync()

    schedule.every(config.POLL_INTERVAL_SECONDS).seconds.do(sync)

    while True:
        schedule.run_pending()
        time.sleep(10)


if __name__ == "__main__":
    main()
