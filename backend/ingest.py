import uuid
import fitz  # PyMuPDF
from fastembed import TextEmbedding
from pinecone import Pinecone
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import PINECONE_API_KEY, PINECONE_INDEX, EMBEDDING_MODEL

embedder = TextEmbedding(EMBEDDING_MODEL)
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


def embed_batch(texts: list[str]) -> list[list[float]]:
    return [v.tolist() for v in embedder.embed(texts)]


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

    all_embeddings = []
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        all_embeddings.extend(embed_batch(chunks[i : i + batch_size]))

    vectors = [
        {
            "id": f"{doc_id}_{i}",
            "values": embedding,
            "metadata": {"doc_id": doc_id, "author_name": author_name, "text": chunk},
        }
        for i, (chunk, embedding) in enumerate(zip(chunks, all_embeddings))
    ]

    try:
        for i in range(0, len(vectors), 100):
            index.upsert(vectors=vectors[i : i + 100])
    except Exception as e:
        raise RuntimeError(f"Pinecone upsert error: {e}")

    return {"doc_id": doc_id, "chunks_uploaded": len(chunks)}
