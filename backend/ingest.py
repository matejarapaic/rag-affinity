import logging
import uuid
import fitz  # PyMuPDF
from pinecone import Pinecone
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import PINECONE_API_KEY, PINECONE_INDEX, PINECONE_NAMESPACE
from graph import add_document_to_graph

logger = logging.getLogger(__name__)

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)


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

    records = [
        {
            "_id": f"{doc_id}_{i}",
            "text": chunk,
            "author_name": author_name,
            "doc_id": doc_id,
        }
        for i, chunk in enumerate(chunks)
    ]

    try:
        batch_size = 96
        for i in range(0, len(records), batch_size):
            index.upsert_records(
                namespace=PINECONE_NAMESPACE,
                records=records[i: i + batch_size],
            )
    except Exception as e:
        raise RuntimeError(f"Pinecone upsert error: {e}")

    try:
        add_document_to_graph(doc_id, chunks, author_name)
    except Exception as e:
        logger.warning("Graph build failed for doc %s: %s", doc_id, e)

    return {"doc_id": doc_id, "chunks_uploaded": len(chunks)}
