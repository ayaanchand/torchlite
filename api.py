from __future__ import annotations
import os
from typing import List, Literal, Optional, Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, JSONResponse
from pydantic import BaseModel

# your existing imports
from query import answer_stream, vectordb, TOP_K

load_dotenv()

ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]

class ChatMsg(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class AskRequest(BaseModel):
    # accept both "question" and "query"
    question: Optional[str] = None
    query: Optional[str] = None
    history: List[ChatMsg] = []

    def prompt(self) -> str:
        return (self.question or self.query or "").strip()

# app
app = FastAPI(title="TorchLite RAG", version="0.1-test")

# CORS 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if ALLOWED_ORIGINS == ["*"] else ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# endpoints
@app.get("/health")
def health():
    return {"status": "ok", "top_k": TOP_K}

@app.post("/api/v1/ask")
def ask(req: AskRequest):
    prompt = req.prompt()
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing 'question' (or 'query')")

    # call your RAG pipeline
    try:
        result = answer_stream(
            prompt,
            vectordb,
            chat_history=[m.model_dump() for m in req.history]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG failure: {e}")

    # support both: (answer, docs) or just docs
    answer: str = ""
    docs: List[Any] = []
    if isinstance(result, tuple) and len(result) == 2:
        answer, docs = result
    else:
        docs = result

    # plain-text payload frontend parses
    lines = [f"Answer: {(answer or '').strip()}"]

    # Sources (deduplicate by URL)
    seen = set()
    src_lines = []
    for d in (docs or [])[:TOP_K]:
        url = (d.metadata.get("source_url") or d.metadata.get("source") or "").strip()
        first_line = (d.page_content or "").splitlines()[0] if getattr(d, "page_content", None) else "Source"
        title = (d.metadata.get("title") or first_line).strip()
        if url and url not in seen:
            seen.add(url)
            src_lines.append(f"- {title[:80]} → {url}")

    if src_lines:
        lines.append("Sources:")
        lines.extend(src_lines)

    return PlainTextResponse("\n".join(lines), media_type="text/plain; charset=utf-8")

@app.get("/")
def root():
    return JSONResponse({"ok": True, "message": "TorchLite RAG API (no API key)"})
