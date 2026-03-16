# Affinity RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that ingests documents into Pinecone and answers questions using GPT-4o.

## Project Structure

```
rag-chatbot/
├── backend/
│   ├── main.py           # FastAPI app (routes)
│   ├── ingest.py         # PDF/TXT parsing, chunking, embedding, Pinecone upsert
│   ├── retrieval.py      # Query embedding, Pinecone search, GPT-4o call + streaming
│   ├── config.py         # Env var loading & validation
│   └── requirements.txt
├── frontend/
│   └── index.html        # Self-contained chat UI (no build step)
├── .env.example
└── README.md
```

## Prerequisites

- Python 3.10+
- A Pinecone account with an index named `-automation` (1536 dimensions, cosine metric)
- OpenAI API key

## Setup

### 1. Clone / navigate to the project

```bash
cd rag-chatbot
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r backend/requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your keys:

```
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...
PINECONE_INDEX=wealthion-automation
PINECONE_REGION=us-east-1
```

### 5. Start the backend

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.
Swagger docs: `http://localhost:8000/docs`

### 6. Open the frontend

Open `frontend/index.html` directly in a browser, or serve it with any static server:

```bash
# Quick static server (Python)
cd frontend
python -m http.server 3000
# Then open http://localhost:3000
```

## API Endpoints

| Method | Path      | Description                          |
|--------|-----------|--------------------------------------|
| GET    | /health   | Health check                         |
| POST   | /upload   | Ingest a PDF or .txt file            |
| POST   | /chat     | Chat with optional streaming support |

### POST /upload

Form data: `file` (PDF or .txt, max 20 MB recommended)

Response:
```json
{ "doc_id": "uuid", "chunks_uploaded": 42 }
```

### POST /chat

```json
{
  "message": "What does the document say about inflation?",
  "history": [
    { "role": "user",      "content": "Hello" },
    { "role": "assistant", "content": "Hi! How can I help?" }
  ],
  "stream": true
}
```

- `stream: false` → JSON response `{ "response": "...", "sources": [...] }`
- `stream: true`  → Server-Sent Events stream (text/event-stream)

## Embedding into an iframe

```html
<iframe
  src="https://your-host/frontend/index.html"
  width="100%"
  height="700"
  frameborder="0"
  allow="clipboard-write">
</iframe>
```

If the backend is on a different origin, update `API_BASE` at the top of `index.html`.

## Pinecone Index Setup

If you need to create the index from scratch:

```python
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key="YOUR_KEY")
pc.create_index(
    name="wealthion-automation",
    dimension=1536,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)
```
