from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
import threading
import logging

from config import validate_config
from retrieval import chat, chat_stream
from graph import get_stats, extract_entities, graph_search

validate_config()

# ── Background Drive poller ──────────────────────────────────────────────────
def _start_poller():
    """Start the Google Drive ingestion poller in a background daemon thread."""
    try:
        from poller import poll
        t = threading.Thread(target=poll, args=(300,), daemon=True, name="drive-poller")
        t.start()
        logging.getLogger(__name__).info("Drive poller started (every 300s)")
    except Exception as e:
        logging.getLogger(__name__).warning("Could not start Drive poller: %s", e)

_start_poller()

app = FastAPI(title="Wealthion RAG Chatbot", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Models ──────────────────────────────────────────────────────────────────

class Message(BaseModel):
    role: str   # "user" | "assistant"
    content: str

class ChatRequest(BaseModel):
    message: str
    history: list[Message] = []
    stream: bool = False

class GraphDebugRequest(BaseModel):
    query: str


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}



@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    history = [{"role": m.role, "content": m.content} for m in req.history]

    if req.stream:
        return StreamingResponse(
            chat_stream(req.message, history),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        result = chat(req.message, history)
        return result
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


# ── Graph inspection endpoints ────────────────────────────────────────────────

@app.get("/documents")
def list_documents():
    """Return all unique documents currently in the knowledge base."""
    try:
        from qdrant_client import QdrantClient
        qdrant = QdrantClient(url="http://localhost:6333")
        collection = "affinity_rag"

        seen = {}
        offset = None
        while True:
            points, next_offset = qdrant.scroll(
                collection_name=collection,
                limit=250,
                offset=offset,
                with_payload=["file_name", "doc_id", "modified_time", "ingested_at", "doc_type"],
                with_vectors=False,
            )
            for p in points:
                pl = p.payload or {}
                name = pl.get("file_name") or pl.get("doc_id") or "unknown"
                if name not in seen:
                    seen[name] = {
                        "file_name": name,
                        "doc_type": pl.get("doc_type", ""),
                        "modified_time": pl.get("modified_time", ""),
                        "ingested_at": pl.get("ingested_at", ""),
                    }
            if next_offset is None:
                break
            offset = next_offset

        docs = sorted(seen.values(), key=lambda d: d["file_name"].lower())
        return {"count": len(docs), "documents": docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/graph/stats")
def graph_stats():
    """
    Returns the current state of the knowledge graph.
    Use this to verify that documents were indexed into the graph after upload.
    Example: GET /graph/stats
    """
    return get_stats()


@app.post("/graph/debug")
def graph_debug(req: GraphDebugRequest):
    """
    Shows exactly what the graph layer does for a given query:
    - which entities were extracted from the query
    - which chunks the graph traversal found (before merging with vector search)
    Example: POST /graph/debug  {"query": "What did Apple do in Q3?"}
    """
    entities = extract_entities(req.query)
    matches = graph_search(req.query, top_k=5)
    return {
        "query": req.query,
        "entities_extracted": [{"text": t, "type": et} for t, et in entities],
        "graph_matches": [
            {
                "chunk_id": m["id"],
                "score": m["score"],
                "author_name": m["metadata"]["author_name"],
                "preview": m["metadata"]["text"][:200],
            }
            for m in matches
        ],
    }


if __name__ == "__main__":
    import sys
    # When frozen by PyInstaller, pass the app object directly (can't import by string)
    # and disable the reloader (it re-spawns processes which breaks in frozen mode)
    if getattr(sys, 'frozen', False):
        uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
    else:
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
