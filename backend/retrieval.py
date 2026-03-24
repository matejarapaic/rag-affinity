import json
import anthropic
from fastembed import TextEmbedding
from pinecone import Pinecone
from config import ANTHROPIC_API_KEY, PINECONE_API_KEY, PINECONE_INDEX, EMBEDDING_MODEL, CHAT_MODEL

embedder = TextEmbedding(EMBEDDING_MODEL)
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

SYSTEM_PROMPT = (
    "You are Affinity's AI assistant — friendly, confident, and easy to talk to. "
    "You represent a company that helps businesses grow through AI automation and data-driven marketing. "

    "When answering questions, follow these guidelines: "
    "1. Speak in a warm, conversational tone — like a knowledgeable team member, not a robot reading a document. "
    "2. Keep answers concise and to the point. Avoid long walls of text. "
    "3. Use simple, everyday language. If a concept is technical, explain it plainly. "
    "4. Never mention 'retrieved context', 'sources', 'chunks', or 'doc_id' in your response — the user should never know these exist. "
    "5. Answer using ONLY the information provided in the context below. "
    "6. If the context doesn't cover the question, say something like: "
    "'That's a great question — I don't have that detail on hand right now, but feel free to reach out to the Affinity team directly.' "
    "Never make up information. "

    "Your goal is to make every visitor feel informed, welcomed, and excited about what Affinity can do for them."
)

MAX_HISTORY_TURNS = 6


def embed_query(text: str) -> list[float]:
    return next(iter(embedder.embed([text]))).tolist()


def query_pinecone(embedding: list[float], top_k: int = 5) -> list[dict]:
    try:
        result = index.query(vector=embedding, top_k=top_k, include_metadata=True)
        return result.get("matches", [])
    except Exception as e:
        raise RuntimeError(f"Pinecone query error: {e}")


def build_anthropic_messages(user_message: str, matches: list[dict], history: list[dict]):
    blocks = []
    for i, match in enumerate(matches, start=1):
        meta = match.get("metadata", {})
        blocks.append(f"[Source {i} — {meta.get('author_name','unknown')} ({meta.get('doc_id','unknown')})]:\n{meta.get('text','')}")
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
        }
        for m in matches
    ]


def chat(user_message: str, history: list[dict]) -> dict:
    embedding = embed_query(user_message)
    matches = query_pinecone(embedding)
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
    embedding = embed_query(user_message)
    matches = query_pinecone(embedding)
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
