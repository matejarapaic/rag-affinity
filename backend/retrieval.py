import json
from concurrent.futures import ThreadPoolExecutor

import anthropic
from pinecone import Pinecone

from config import ANTHROPIC_API_KEY, PINECONE_API_KEY, PINECONE_INDEX, PINECONE_NAMESPACE, CHAT_MODEL
from graph import graph_search

anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

SYSTEM_PROMPT = (
    "You are a knowledgeable assistant for Affinity. "
    "Answer using ONLY the retrieved context below. "
    "Context is retrieved via both semantic vector search and knowledge graph traversal. "
    "Cite the source (author_name or doc_id) when possible. "
    "If the context is insufficient to answer the question, say so — do not hallucinate."
)

MAX_HISTORY_TURNS = 6


def query_pinecone(query_text: str, top_k: int = 5) -> list[dict]:
    try:
        result = index.search(
            namespace=PINECONE_NAMESPACE,
            query={"inputs": {"text": query_text}, "top_k": top_k},
            fields=["text", "author_name", "doc_id", "file_id", "chunk_index"],
        )
        matches = []
        for hit in result.result.hits:
            matches.append({
                "id": hit._id,
                "score": hit._score,
                "metadata": hit.fields,
                "source": "vector",
            })
        return matches
    except Exception as e:
        raise RuntimeError(f"Pinecone query error: {e}")


def hybrid_retrieve(query_text: str, top_k: int = 5) -> list[dict]:
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_vector = executor.submit(query_pinecone, query_text, top_k)
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
        label = (
            f"[Source {i} — {meta.get('author_name', 'unknown')} "
            f"({meta.get('doc_id', 'unknown')}) via {source_type}]"
        )
        blocks.append(f"{label}:\n{meta.get('text', '')}")

    context_str = "\n\n".join(blocks) if blocks else "No relevant context found."
    system_content = f"{SYSTEM_PROMPT}\n\n--- RETRIEVED CONTEXT ---\n{context_str}\n--- END CONTEXT ---"

    messages = list(history[-(MAX_HISTORY_TURNS * 2):])
    messages.append({"role": "user", "content": user_message})
    return system_content, messages


def _build_sources(matches: list[dict]) -> list[dict]:
    return [
        {
            "doc_id": m.get("metadata", {}).get("doc_id", "unknown"),
            "author_name": m.get("metadata", {}).get("author_name", "unknown"),
            "chunk_preview": m.get("metadata", {}).get("text", "")[:120],
            "source": m.get("source", "vector"),
        }
        for m in matches
    ]


def chat(user_message: str, history: list[dict]) -> dict:
    matches = hybrid_retrieve(user_message)
    system_content, messages = build_anthropic_messages(user_message, matches, history)

    try:
        response = anthropic_client.messages.create(
            model=CHAT_MODEL,
            max_tokens=1024,
            system=system_content,
            messages=messages,
        )
        response_text = response.content[0].text
    except Exception as e:
        raise RuntimeError(f"Claude chat error: {e}")

    return {"response": response_text, "sources": _build_sources(matches)}


def chat_stream(user_message: str, history: list[dict]):
    """Generator that yields SSE-formatted chunks, then a final sources event."""
    matches = hybrid_retrieve(user_message)
    system_content, messages = build_anthropic_messages(user_message, matches, history)

    try:
        with anthropic_client.messages.stream(
            model=CHAT_MODEL,
            max_tokens=1024,
            system=system_content,
            messages=messages,
        ) as stream:
            for text in stream.text_stream:
                yield f"data: {text.replace(chr(10), chr(92) + 'n')}\n\n"
    except Exception as e:
        yield f"data: [ERROR] {e}\n\n"
        return

    yield f"event: sources\ndata: {json.dumps(_build_sources(matches))}\n\n"
    yield "data: [DONE]\n\n"
