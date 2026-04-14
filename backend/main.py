import sys
import logging
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
import uvicorn

from config import (
    validate_config,
    CLERK_PUBLISHABLE_KEY,
    ANTHROPIC_API_KEY,
)
from auth import get_user_id
from user_keys import get_user_key, save_user_key
from retrieval import chat, chat_stream
from graph import get_stats, extract_entities, graph_search

validate_config()

# ── Rate limiting ─────────────────────────────────────────────────────────────
# Use only the direct TCP peer address — never X-Forwarded-For — to prevent
# clients from spoofing a different IP to bypass rate limits.
def _real_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"

limiter = Limiter(key_func=_real_ip, default_limits=["120/minute"])

# ── Background Drive poller ──────────────────────────────────────────────────
def _start_poller():
    try:
        from poller import poll
        t = threading.Thread(target=poll, args=(300,), daemon=True, name="drive-poller")
        t.start()
        logging.getLogger(__name__).info("Drive poller started (every 300s)")
    except Exception as e:
        logging.getLogger(__name__).warning("Could not start Drive poller: %s", e)

_start_poller()

app = FastAPI(title="Affinity RAG", version="2.0.0")

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)


# ── Models ────────────────────────────────────────────────────────────────────

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: list[Message] = []
    stream: bool = False

class GraphDebugRequest(BaseModel):
    query: str

class ApiKeyRequest(BaseModel):
    api_key: str


# ── Public endpoints (no auth) ────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/clerk-config")
@limiter.limit("5/15minutes")          # login-adjacent: strict limit
def clerk_config(request: Request):
    """
    Returns the Clerk publishable key so the frontend can initialise the SDK.
    Returns null when Clerk is not configured (single-user dev mode).
    """
    return {"publishable_key": CLERK_PUBLISHABLE_KEY or None}


# ── Settings endpoints ────────────────────────────────────────────────────────

@app.get("/settings")
@limiter.limit("30/minute")
def get_settings(request: Request, user_id: str = Depends(get_user_id)):
    """Return the current API key status for the authenticated user."""
    user_key   = get_user_key(user_id)
    active_key = user_key or ANTHROPIC_API_KEY or ""

    return {
        "api_key_set":     bool(active_key),
        "api_key_source":  "user" if user_key else ("system" if ANTHROPIC_API_KEY else "none"),
        "api_key_preview": f"sk-ant-...{active_key[-6:]}" if active_key else None,
    }


@app.post("/settings/api-key")
@limiter.limit("5/15minutes")          # login-adjacent: strict limit
def save_api_key(request: Request, req: ApiKeyRequest, user_id: str = Depends(get_user_id)):
    """Validate and persist the user's Anthropic API key."""
    key = req.api_key.strip()
    if not key.startswith("sk-ant-"):
        raise HTTPException(
            status_code=400,
            detail="Invalid Anthropic API key — must start with sk-ant-"
        )
    save_user_key(user_id, key)
    return {"status": "saved", "preview": f"sk-ant-...{key[-6:]}"}


# ── Chat endpoint ─────────────────────────────────────────────────────────────

@app.post("/chat")
@limiter.limit("30/minute")
async def chat_endpoint(request: Request, req: ChatRequest, user_id: str = Depends(get_user_id)):
    api_key = get_user_key(user_id) or ANTHROPIC_API_KEY or None
    history = [{"role": m.role, "content": m.content} for m in req.history]

    if req.stream:
        return StreamingResponse(
            chat_stream(req.message, history, api_key=api_key),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        result = chat(req.message, history, api_key=api_key)
        return result
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logging.getLogger(__name__).exception("Chat error")
        raise HTTPException(status_code=500, detail="An internal error occurred")


# ── Documents endpoint ────────────────────────────────────────────────────────

@app.get("/documents")
@limiter.limit("30/minute")
def list_documents(request: Request, user_id: str = Depends(get_user_id)):
    """Return all unique documents currently in the knowledge base."""
    try:
        from qdrant_client import QdrantClient
        qdrant     = QdrantClient(url="http://localhost:6333")
        collection = "affinity_rag"

        seen, offset = {}, None
        while True:
            points, next_offset = qdrant.scroll(
                collection_name=collection,
                limit=250,
                offset=offset,
                with_payload=["file_name", "doc_id", "modified_time", "ingested_at", "doc_type", "source"],
                with_vectors=False,
            )
            for p in points:
                pl      = p.payload or {}
                name    = pl.get("file_name") or pl.get("doc_id") or "unknown"
                file_id = pl.get("file_id") or pl.get("doc_id") or ""
                if name not in seen:
                    seen[name] = {
                        "file_name":     name,
                        "file_id":       file_id,
                        "doc_type":      pl.get("doc_type", ""),
                        "source":        pl.get("source", "google_drive"),
                        "modified_time": pl.get("modified_time", ""),
                        "ingested_at":   pl.get("ingested_at", ""),
                    }
            if next_offset is None:
                break
            offset = next_offset

        docs = sorted(seen.values(), key=lambda d: d["file_name"].lower())
        return {"count": len(docs), "documents": docs}
    except Exception as e:
        logging.getLogger(__name__).exception("Failed to list documents")
        raise HTTPException(status_code=500, detail="Failed to retrieve documents")


@app.delete("/documents/{file_id:path}")
@limiter.limit("20/minute")
def delete_document(request: Request, file_id: str, user_id: str = Depends(get_user_id)):
    """Remove all Qdrant chunks for a given file_id."""
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        qdrant     = QdrantClient(url="http://localhost:6333")
        collection = "affinity_rag"
        qdrant.delete(
            collection_name=collection,
            points_selector=Filter(
                must=[FieldCondition(key="file_id", match=MatchValue(value=file_id))]
            ),
        )
        return {"status": "deleted", "file_id": file_id}
    except Exception as e:
        logging.getLogger(__name__).exception("Failed to delete document")
        raise HTTPException(status_code=500, detail="Failed to delete document")


# ── Document upload endpoint ──────────────────────────────────────────────────

_MIME_BY_EXT = {
    "pdf":  "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "txt":  "text/plain",
}

def _sniff_mime(filename: str, reported: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return _MIME_BY_EXT.get(ext) or reported


@app.post("/upload")
@limiter.limit("10/minute")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    user_id: str = Depends(get_user_id),
):
    """Ingest an uploaded document directly into Qdrant (no Drive storage required)."""
    import hashlib
    from ingest_gdrive import ingest_bytes

    filename   = file.filename or "upload"
    mime_type  = _sniff_mime(filename, file.content_type or "application/octet-stream")
    file_bytes = await file.read()

    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large — maximum size is 50 MB")

    # Synthesise a file-metadata dict — same shape ingest_bytes() expects
    file_id = "upload_" + hashlib.md5(filename.encode() + file_bytes[:64]).hexdigest()[:12]
    file_meta = {
        "id":           file_id,
        "name":         filename,
        "mimeType":     mime_type,
        "modifiedTime": "",
        "md5Checksum":  hashlib.md5(file_bytes).hexdigest(),
    }

    try:
        ingest_bytes(file_bytes, file_meta)
    except Exception as e:
        logging.getLogger(__name__).exception("Ingestion failed for %s", filename)
        raise HTTPException(status_code=500, detail="Ingestion failed — check server logs")

    return {
        "status": "success",
        "file":   {"name": filename, "id": file_id},
    }


# ── Graph inspection endpoints ────────────────────────────────────────────────

@app.get("/graph/stats")
@limiter.limit("30/minute")
def graph_stats(request: Request, user_id: str = Depends(get_user_id)):
    return get_stats()


@app.post("/graph/debug")
@limiter.limit("30/minute")
def graph_debug(request: Request, req: GraphDebugRequest, user_id: str = Depends(get_user_id)):
    entities = extract_entities(req.query)
    matches  = graph_search(req.query, top_k=5)
    return {
        "query":              req.query,
        "entities_extracted": [{"text": t, "type": et} for t, et in entities],
        "graph_matches": [
            {
                "chunk_id":    m["id"],
                "score":       m["score"],
                "author_name": m["metadata"]["author_name"],
                "preview":     m["metadata"]["text"][:200],
            }
            for m in matches
        ],
    }


# ── Serve frontend (must be mounted last so API routes take precedence) ───────

def _frontend_dir() -> Path:
    if getattr(sys, 'frozen', False):
        return Path(sys._MEIPASS) / "frontend"
    return Path(__file__).parent.parent / "frontend"

_fe = _frontend_dir()
if _fe.exists():
    app.mount("/", StaticFiles(directory=str(_fe), html=True), name="frontend")
else:
    logging.getLogger(__name__).warning("Frontend directory not found at %s — UI will not be served", _fe)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if getattr(sys, 'frozen', False):
        uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
    else:
        uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
