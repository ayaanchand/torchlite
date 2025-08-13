from __future__ import annotations
import os, sys, time, logging, random
from dotenv import load_dotenv


from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langfuse.langchain import CallbackHandler
from db_utils import get_vectordb
from typing import List, Literal, Optional
from langchain.schema import AIMessage, HumanMessage, SystemMessage

from kb_version import get_kb_version  # reads kb_version from Supabase

logging.getLogger("langchain_community.document_loaders.notiondb").setLevel(logging.ERROR)

load_dotenv()

# Toggles & config 
STREAM_DELAY   = float(os.getenv("STREAM_DELAY", "0.03"))
MAX_CTX_CHARS  = int(os.getenv("MAX_CTX_CHARS", "6000"))
TOP_K          = int(os.getenv("TOP_K", "2"))
MAX_HISTORY_TURNS= int(os.getenv("MAX_HISTORY_TURNS", "8"))  # keep last N turns

SUPABASE_URL   = os.environ["SUPABASE_URL"]
SUPABASE_KEY   = os.environ["SUPABASE_KEY"]
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "torchlite_embeddings")

# Tracing switches
TRACE_ENABLED  = os.getenv("TRACE_ENABLED", "1") == "1"
SAMPLE_RATE    = float(os.getenv("LANGFUSE_SAMPLING_RATE", "1"))

# Models (also reported in metadata)
LLM_MODEL      = os.getenv("LLM_MODEL", "gpt-4o")
EMBED_MODEL    = os.getenv("EMBED_MODEL", "text-embedding-3-small")

ASTROLABS_BLURB = (
    "You are chatting with AstroLabs employees. "
    "AstroLabs is a Dubai-based learning and innovation company that runs "
    "startup programs, coding & data academies, and coworking spaces across the MENA region. "
    "Its mission is to build a thriving tech ecosystem by upskilling talent and supporting founders.\n"
)

SYSTEM_TXT = (
    ASTROLABS_BLURB +
    "You are AstroLabs' knowledge assistant. Use the provided context as your primary source. "
    "Synthesize and summarize across snippets when needed. "
    "If the context is related but incomplete, ask exactly one short clarifying question and stop. "
    "If there is no relevant context at all, reply exactly: I don't know. "
    "Never invent facts."
)

def _to_langchain_history(history: Optional[List[dict]]) -> List:
    """Convert [{'role':'user'|'assistant'|'system','content':str}, ...] -> LC messages."""
    if not history:
        return []
    out: List = []
    # keep only last N turns to control tokens
    trimmed = history[-MAX_HISTORY_TURNS:] if len(history) > MAX_HISTORY_TURNS else history
    for m in trimmed:
        r = (m.get("role") or "").strip()
        c = (m.get("content") or "").strip()
        if not c:
            continue
        if r == "system":
            out.append(SystemMessage(content=c))
        elif r == "user":
            out.append(HumanMessage(content=c))
        elif r == "assistant":
            out.append(AIMessage(content=c))
    return out

# Langfuse callback
langfuse_handler = CallbackHandler()

def get_callbacks():
    """Return callbacks list according to toggle + sampling."""
    if not TRACE_ENABLED:
        return []
    return [langfuse_handler] if random.random() < SAMPLE_RATE else []

# Vector DB 
vectordb = get_vectordb()

# Answer (streaming)
def answer_stream(question: str, vectordb, chat_history: Optional[List[dict]] = None):
    t0 = time.perf_counter()

    # Retrieval
    docs = vectordb.similarity_search(question, k=TOP_K)
    t1   = time.perf_counter()

    # Build bounded context
    ctx_parts, running = [], 0
    for d in docs:
        txt = d.page_content
        if running + len(txt) > MAX_CTX_CHARS:
            break
        ctx_parts.append(txt)
        running += len(txt)
    context = "\n\n".join(ctx_parts)

    # Prompt + LLM
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_TXT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human",
         "Context:\n{context}\n\n"
         "Question: {user_question}\n\n"
         "Instructions:\n"
         "- If the context fully answers the question, answer concisely using the context.\n"
         "- If the context seems related but you need clarification, reply with one sentence starting with 'Follow-up:' and STOP.\n"
         "- If the context is irrelevant, reply exactly: I don't know."
        ),
    ])

    # print("\n--- CONTEXT FED TO LLM ---\n", context[:1500], "\n--------------------------\n")
    chain = prompt | ChatOpenAI(model_name=LLM_MODEL, temperature=0, streaming=True)

    callbacks   = get_callbacks()
    kb_version  = get_kb_version()

    # Stream answer
    # print("🟢 Answer:\n", end="", flush=True)
    llm_start = time.perf_counter()
    answer_parts = []

    for chunk in chain.stream(
        {
            "context":       context,
            "user_question": question,
            "chat_history":  _to_langchain_history(chat_history),
        },
        config={
            "callbacks": get_callbacks(),
            "metadata": {
                "app": "torchlite",
                "env": os.getenv("APP_ENV", "dev"),
                "table": SUPABASE_TABLE,
                "kb_version": get_kb_version(),
                "model": LLM_MODEL,
                "embed_model": EMBED_MODEL,
            },
            "tags": ["torchlite", "retrieval-qa"],
        },
    ):
        token = getattr(chunk, "content", "") or getattr(chunk, "delta", "") or ""
        if not token:
            continue
        answer_parts.append(token)
        print(token, end="", flush=True)
        if STREAM_DELAY > 0:
            time.sleep(STREAM_DELAY)

    print()  # final newline
    llm_end = time.perf_counter()
    answer = "".join(answer_parts)
    # print("\n")

    # Latency
    # total = llm_end - t0
    # print(f"⏱️  retrieval: {t1 - t0:.2f}s | LLM: {llm_end - llm_start:.2f}s | total: {total:.2f}s\n")

    return answer, docs

# Main 
def main():
    question = " ".join(sys.argv[1:]) or "What is the focus of Week 1?"
    ans, docs = answer_stream(question, vectordb, chat_history=[])

    # print(f"📄 Top-{TOP_K} source(s):")
    '''
    seen = set()
    for d in docs[:TOP_K]:
        url   = d.metadata.get("source_url", "URL-missing")
        title = d.page_content.splitlines()[0][:80]
        if url not in seen:                 # avoid printing duplicates
            # print(f"- {title}  →  {url}")
            seen.add(url)

    '''

if __name__ == "__main__":
    main()
