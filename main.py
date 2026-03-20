from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

from config import validate_config
from ingest import ingest_document
from retrieval import chat, chat_stream

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


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    allowed_types = {"application/pdf", "text/plain"}
    content_type = file.content_type or ""

    if content_type not in allowed_types and not file.filename.lower().endswith((".pdf", ".txt")):
        raise HTTPException(status_code=415, detail="Only PDF and .txt files are supported.")

    try:
        file_bytes = await file.read()
        result = ingest_document(file_bytes, file.filename, content_type)
        return result
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


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


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
