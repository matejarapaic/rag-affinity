import json
from concurrent.futures import ThreadPoolExecutor

import anthropic
from fastembed import TextEmbedding
from qdrant_client import QdrantClient

from config import ANTHROPIC_API_KEY, CHAT_MODEL
from graph import graph_search

anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
_embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
_qdrant = QdrantClient(url="http://localhost:6333")
QDRANT_COLLECTION = "affinity_rag"
VECTOR_SIZE = 384  # BAAI/bge-small-en-v1.5

# Ensure the collection exists so queries never fail with "collection not found"
def _ensure_collection():
    try:
        from qdrant_client.models import Distance, VectorParams
        existing = [c.name for c in _qdrant.get_collections().collections]
        if QDRANT_COLLECTION not in existing:
            _qdrant.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("Could not ensure Qdrant collection: %s", e)

_ensure_collection()

SYSTEM_PROMPT = (
    "You are a knowledgeable assistant for Affinity. "
    "Answer using ONLY the retrieved context below. "
    "Context is retrieved via both semantic vector search and knowledge graph traversal. "
    "When citing sources, always use the file_name provided in the source label (e.g. 'Source 1 — Meeting for the Blind'). "
    "Never cite raw IDs or doc_id values. "
    "\n\n"
    "Each source includes metadata: file_name, modified_time (when the document was last edited in Google Drive), "
    "and ingested_at (when it was added to the knowledge base). "
    "You are allowed — and encouraged — to reason from this metadata to answer temporal questions. "
    "For example: if a document is titled 'Maryland Trip Notes' and its modified_time is 2026-03-15, "
    "you can reasonably infer the Maryland trip occurred around that date, and say so clearly. "
    "Always state when you are inferring from metadata rather than explicit document content, "
    "e.g. 'Based on when this document was last edited (March 15 2026), it appears the trip took place around that time.' "
    "If the context is truly insufficient even with metadata, say so — do not hallucinate."
)

MAX_HISTORY_TURNS = 6


def _get_all_filenames() -> list[str]:
    """Return distinct file_name values from the collection."""
    try:
        points, _ = _qdrant.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=1000,
            with_payload=["file_name"],
            with_vectors=False,
        )
        seen = set()
        names = []
        for p in points:
            n = (p.payload or {}).get("file_name", "")
            if n and n not in seen:
                seen.add(n)
                names.append(n)
        return names
    except Exception:
        return []


def _filename_matches(query_text: str, top_k: int = 5) -> list[dict]:
    """If the query mentions a known filename, return all chunks for that file."""
    query_lower = query_text.lower()
    filenames = _get_all_filenames()

    matched_file = None
    best_score = 0
    for name in filenames:
        # Count how many words from the filename appear in the query
        words = [w for w in name.lower().replace("-", " ").replace("_", " ").split() if len(w) > 2]
        hits = sum(1 for w in words if w in query_lower)
        if hits > best_score:
            best_score = hits
            matched_file = name

    if not matched_file or best_score < 2:
        return []

    # Pull all chunks for the matched file
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        points, _ = _qdrant.scroll(
            collection_name=QDRANT_COLLECTION,
            scroll_filter=Filter(must=[FieldCondition(key="file_name", match=MatchValue(value=matched_file))]),
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        return [
            {
                "id": str(p.id),
                "score": 1.5,  # boost filename matches above vector results
                "metadata": p.payload,
                "source": "filename_match",
            }
            for p in points
        ]
    except Exception:
        return []


def query_qdrant(query_text: str, top_k: int = 5) -> list[dict]:
    try:
        # First check if query is referencing a specific filename
        name_matches = _filename_matches(query_text, top_k)

        vector = list(_embedder.embed([query_text]))[0].tolist()
        result = _qdrant.query_points(
            collection_name=QDRANT_COLLECTION,
            query=vector,
            limit=top_k,
            with_payload=True,
        )
        vector_matches = []
        for hit in result.points:
            vector_matches.append({
                "id": str(hit.id),
                "score": hit.score,
                "metadata": hit.payload,
                "source": "vector",
            })

        # Merge: filename matches first, then vector results (deduped)
        seen = {m["id"] for m in name_matches}
        merged = list(name_matches)
        for m in vector_matches:
            if m["id"] not in seen:
                merged.append(m)

        return merged[:top_k * 2] if name_matches else vector_matches
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("Qdrant query error (returning empty): %s", e)
        return []


def hybrid_retrieve(query_text: str, top_k: int = 5) -> list[dict]:
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_vector = executor.submit(query_qdrant, query_text, top_k)
        future_graph = executor.submit(graph_search, query_text, top_k)
        vector_matches = future_vector.result()
        graph_matches = future_graph.result()

    seen_ids: set[str] = set()
    merged: list[dict] = []

    for match in vector_matches:
        seen_ids.add(match["id"])
        merged.append(match)

    for match in graph_matches:
        if match["id"] not in seen_ids:
            seen_ids.add(match["id"])
            merged.append(match)

    return merged


def build_anthropic_messages(user_message: str, matches: list[dict], history: list[dict]):
    blocks = []
    for i, match in enumerate(matches, start=1):
        meta = match.get("metadata", {})
        source_type = match.get("source", "vector")
        file_name = meta.get("file_name") or meta.get("author_name") or meta.get("doc_id") or "unknown"

        # Build metadata line so Claude can reason temporally
        meta_parts = []
        if meta.get("modified_time"):
            meta_parts.append(f"modified_time={meta['modified_time']}")
        if meta.get("ingested_at"):
            meta_parts.append(f"ingested_at={meta['ingested_at']}")
        if meta.get("doc_type"):
            meta_parts.append(f"doc_type={meta['doc_type']}")
        meta_str = f" | {', '.join(meta_parts)}" if meta_parts else ""

        label = f"[Source {i} — {file_name}{meta_str} via {source_type}]"
        blocks.append(f"{label}:\n{meta.get('text', '')}")

    context_str = "\n\n".join(blocks) if blocks else "No relevant context found."
    system_content = f"{SYSTEM_PROMPT}\n\n--- RETRIEVED CONTEXT ---\n{context_str}\n--- END CONTEXT ---"

    messages = list(history[-(MAX_HISTORY_TURNS * 2):])
    messages.append({"role": "user", "content": user_message})
    return system_content, messages


def _build_sources(matches: list[dict]) -> list[dict]:
    return [
        {
            "doc_id":       m.get("metadata", {}).get("doc_id", "unknown"),
            "file_name":    m.get("metadata", {}).get("file_name", ""),
            "author_name":  m.get("metadata", {}).get("author_name", "unknown"),
            "chunk_preview": m.get("metadata", {}).get("text", "")[:120],
            "source":       m.get("source", "vector"),
        }
        for m in matches
    ]


THINKING_BUDGET = 8000   # tokens Claude can spend reasoning
MAX_TOKENS      = 12000  # must be > THINKING_BUDGET

def chat(user_message: str, history: list[dict]) -> dict:
    matches = hybrid_retrieve(user_message)
    system_content, messages = build_anthropic_messages(user_message, matches, history)

    try:
        response = anthropic_client.messages.create(
            model=CHAT_MODEL,
            max_tokens=MAX_TOKENS,
            thinking={"type": "enabled", "budget_tokens": THINKING_BUDGET},
            system=system_content,
            messages=messages,
        )
        # Extract the text block (thinking blocks are separate content blocks)
        response_text = next(
            (b.text for b in response.content if b.type == "text"), ""
        )
    except Exception as e:
        raise RuntimeError(f"Claude chat error: {e}")

    return {"response": response_text, "sources": _build_sources(matches)}


def chat_stream(user_message: str, history: list[dict]):
    """Generator that yields SSE-formatted chunks, then a final sources event."""
    try:
        matches = hybrid_retrieve(user_message)
    except Exception as e:
        yield f"data: [ERROR] Retrieval error: {e}\n\n"
        return

    system_content, messages = build_anthropic_messages(user_message, matches, history)

    try:
        with anthropic_client.messages.stream(
            model=CHAT_MODEL,
            max_tokens=MAX_TOKENS,
            thinking={"type": "enabled", "budget_tokens": THINKING_BUDGET},
            system=system_content,
            messages=messages,
        ) as stream:
            # text_stream automatically skips thinking blocks — only final answer text comes through
            for text in stream.text_stream:
                yield f"data: {text.replace(chr(10), chr(92) + 'n')}\n\n"
    except Exception as e:
        yield f"data: [ERROR] {e}\n\n"
        return

    yield f"event: sources\ndata: {json.dumps(_build_sources(matches))}\n\n"
    yield "data: [DONE]\n\n"
