from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

from config import validate_config
from retrieval import chat, chat_stream
from graph import get_stats, extract_entities, graph_search

validate_config()

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
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
