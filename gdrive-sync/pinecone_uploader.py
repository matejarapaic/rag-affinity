"""
Uploads text chunks to rag-affinity-v2 using upsert_records.
Pinecone's integrated embedding (llama-text-embed-v2, 768 dims) handles vectorization.
"""

import logging
from typing import List

from pinecone import Pinecone

import config

logger = logging.getLogger(__name__)

_index = None


def _get_index():
    global _index
    if _index is None:
        pc = Pinecone(api_key=config.PINECONE_API_KEY)
        _index = pc.Index(config.PINECONE_INDEX_NAME)
    return _index


def upload_chunks(chunks: List[str], file_id: str, filename: str, batch_size: int = 96) -> None:
    """
    Send text chunks to Pinecone via upsert_records. Pinecone embeds them
    automatically using the index's integrated llama-text-embed-v2 model.
    """
    index = _get_index()
    author_name = filename.rsplit(".", 1)[0]

    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start: batch_start + batch_size]

        records = [
            {
                "_id": f"{file_id}__chunk_{batch_start + i}",
                "text": chunk,
                "author_name": author_name,
                "doc_id": file_id,
                "file_id": file_id,
                "chunk_index": batch_start + i,
            }
            for i, chunk in enumerate(batch)
        ]

        index.upsert_records(
            namespace=config.PINECONE_NAMESPACE,
            records=records,
        )

        logger.info(
            f"Upserted batch {batch_start // batch_size + 1} "
            f"({len(records)} chunks) for '{filename}'."
        )
