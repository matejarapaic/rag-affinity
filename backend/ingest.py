import logging
import uuid
import hashlib
import fitz  # PyMuPDF
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_text_splitters import RecursiveCharacterTextSplitter
from graph import add_document_to_graph

logger = logging.getLogger(__name__)

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "affinity_rag"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

_embedder = TextEmbedding(model_name=EMBED_MODEL)
_qdrant = QdrantClient(url=QDRANT_URL)

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)


def _ensure_collection(vector_size: int):
    existing = [c.name for c in _qdrant.get_collections().collections]
    if COLLECTION_NAME not in existing:
        _qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return text
    except Exception as e:
        raise ValueError(f"Failed to parse PDF: {e}")


def extract_text_from_txt(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8", errors="replace")
    except Exception as e:
        raise ValueError(f"Failed to read text file: {e}")


def ingest_document(file_bytes: bytes, filename: str, content_type: str) -> dict:
    if content_type == "application/pdf" or filename.lower().endswith(".pdf"):
        text = extract_text_from_pdf(file_bytes)
    else:
        text = extract_text_from_txt(file_bytes)

    if not text.strip():
        raise ValueError("Document appears to be empty or unreadable.")

    author_name = filename.rsplit(".", 1)[0]
    chunks = splitter.split_text(text)
    if not chunks:
        raise ValueError("No chunks produced from document.")

    doc_id = str(uuid.uuid4())
    chunk_id_base = int(hashlib.md5(doc_id.encode()).hexdigest(), 16) % (10**12)

    embeddings = list(_embedder.embed(chunks))
    _ensure_collection(len(embeddings[0]))

    points = [
        PointStruct(
            id=chunk_id_base + i,
            vector=emb.tolist(),
            payload={
                "chunk_id": chunk_id_base + i,
                "doc_id": doc_id,
                "author_name": author_name,
                "matter_id": "",
                "doc_type": "upload",
                "text": chunk,
                "source": "upload",
            },
        )
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
    ]

    try:
        _qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    except Exception as e:
        raise RuntimeError(f"Qdrant upsert error: {e}")

    try:
        add_document_to_graph(doc_id, chunks, author_name)
    except Exception as e:
        logger.warning("Graph build failed for doc %s: %s", doc_id, e)

    return {"doc_id": doc_id, "chunks_uploaded": len(chunks)}
