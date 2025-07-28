from __future__ import annotations
import os, sys, time, logging, random
from dotenv import load_dotenv
from supabase import create_client, Client

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import SupabaseVectorStore
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langfuse.langchain import CallbackHandler

from kb_version import get_kb_version  # reads kb_version from Supabase

# Silence loader noise
logging.getLogger("langchain_community.document_loaders.notiondb").setLevel(logging.ERROR)

load_dotenv()

# ── Toggles & config ─────────────────────────────────────────────────────────
STREAM_DELAY   = float(os.getenv("STREAM_DELAY", "0.03"))
MAX_CTX_CHARS  = int(os.getenv("MAX_CTX_CHARS", "6000"))
TOP_K          = int(os.getenv("TOP_K", "3"))

SUPABASE_URL   = os.environ["SUPABASE_URL"]
SUPABASE_KEY   = os.environ["SUPABASE_KEY"]
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "torchlite_embeddings")

# Tracing switches
TRACE_ENABLED  = os.getenv("TRACE_ENABLED", "1") == "1"
SAMPLE_RATE    = float(os.getenv("LANGFUSE_SAMPLING_RATE", "1"))

# Models (also reported in metadata)
LLM_MODEL      = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
EMBED_MODEL    = os.getenv("EMBED_MODEL", "text-embedding-3-small")

SYSTEM_TXT = (
    "You are AstroLabs' knowledge assistant. "
    "Answer the user's question using ONLY the provided sources. "
    "If the answer is not present, reply exactly: I don't know."
)

# Langfuse callback
langfuse_handler = CallbackHandler()

def get_callbacks():
    """Return callbacks list according to toggle + sampling."""
    if not TRACE_ENABLED:
        return []
    return [langfuse_handler] if random.random() < SAMPLE_RATE else []

# ── Vector DB ────────────────────────────────────────────────────────────────
def get_vectordb() -> SupabaseVectorStore:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return SupabaseVectorStore(
        client=supabase,
        embedding=OpenAIEmbeddings(model=EMBED_MODEL),
        table_name=SUPABASE_TABLE,
        query_name=f"match_{SUPABASE_TABLE}",
    )

# ── Answer (streaming) ───────────────────────────────────────────────────────
def answer_stream(question: str, vectordb: SupabaseVectorStore):
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
        MessagesPlaceholder(variable_name="messages"),
        ("human", "Context:\n{context}\n\n{user_question}"),
    ])
    chain = prompt | ChatOpenAI(model_name=LLM_MODEL, temperature=0, streaming=True)

    callbacks   = get_callbacks()
    kb_version  = get_kb_version()

    # Stream answer
    print("🟢 Answer:\n", end="", flush=True)
    llm_start = time.perf_counter()
    for chunk in chain.stream(
        {
            "messages":      [("human", question)],
            "context":       context,
            "user_question": question,
        },
        config={
            "callbacks": callbacks,
            "metadata": {
                "app": "torchlite",
                "env": os.getenv("APP_ENV", "dev"),
                "table": SUPABASE_TABLE,
                "kb_version": kb_version,
                "model": LLM_MODEL,
                "embed_model": EMBED_MODEL,
            },
            "tags": ["torchlite", "retrieval-qa"],
        },
    ):
        txt = getattr(chunk, "content", "") or getattr(chunk, "delta", "")
        if txt:
            print(txt, end="", flush=True)
            if STREAM_DELAY > 0:
                time.sleep(STREAM_DELAY)
    llm_end = time.perf_counter()
    print("\n")

    # Latency
    total = llm_end - t0
    print(f"⏱️  retrieval: {t1 - t0:.2f}s | LLM: {llm_end - llm_start:.2f}s | total: {total:.2f}s\n")

    return docs

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    question = " ".join(sys.argv[1:]) or "What is the focus of Week 1?"
    vectordb = get_vectordb()
    docs     = answer_stream(question, vectordb)

    print(f"📄 Top-{TOP_K} Source(s):")
    seen = set()
    for d in docs[:TOP_K]:
        title = d.page_content.splitlines()[0][:80]
        if title not in seen:
            print("-", title)
            seen.add(title)

if __name__ == "__main__":
    main()
