from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from query import answer_stream, vectordb, TOP_K  # import TOP_K too
from dotenv import load_dotenv
import os
from typing import List, Literal, Optional

load_dotenv()
API_KEY = os.environ.get("RAG_API_KEY")

class ChatMsg(BaseModel):
    role: Literal["user","assistant"]
    content: str

class AskRequest(BaseModel):
    question: str
    history: List[ChatMsg] = []   # <-- add this

app = FastAPI()

@app.post("/api/v1/ask")
def ask(req: AskRequest):
    # Get answer text + top docs
    answer, docs = answer_stream(req.question, vectordb, chat_history=[m.model_dump() for m in req.history])

    lines = [f"Answer: { (answer or '').strip() }"]

    # Optional: include sources your frontend can parse
    seen = set()
    src_lines = []
    for d in (docs or [])[:TOP_K]:
        url = (d.metadata.get("source_url") or d.metadata.get("source") or "").strip()
        title = (d.metadata.get("title") or d.page_content.splitlines()[0]).strip()
        if url and url not in seen:
            seen.add(url)
            src_lines.append(f"- {title[:80]} → {url}")

    if src_lines:
        lines.append("Sources:")
        lines.extend(src_lines)

    return PlainTextResponse("\n".join(lines), media_type="text/plain; charset=utf-8")
